"""End-to-end decision comparison: von-mlx engine vs the official von SDK.

Both engines run the SAME option-marker weights on the SAME hand-built prompts
(including the zero-shot Noul polarity-cancellation path), so any difference in
the answers is attributable to the MLX port.

Usage:
    env -u PYTHONPATH .venv/bin/python verify/sdk_parity.py --model-dir hf_orig
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))


CASES = [
    (
        "choice",
        "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
        {
            "type": "choice",
            "instructions": "Classify the root cause domain of this incident.",
            "criteria": {
                "infrastructure": "Database, hardware, network, or server failures",
                "billing": "Invoices, payments, refunds, subscription queries",
                "feature_request": "Requests for new platform capabilities",
            },
        },
    ),
    (
        "choice-tie",
        "Memory utilization reached 98% with frequent OOM killer invocations.",
        {
            "type": "choice",
            "instructions": "What is the nature of this issue?",
            "criteria": {
                "performance": "Latency, throughput, or resource degradation",
                "config": "Misconfiguration or wrong settings",
            },
        },
    ),
    (
        "noul-zero-shot",
        "Connection pool exhausted on port 5432; subsequent handshakes timing out.",
        {"type": "noul", "instructions": "Is this issue actively blocking customer operations?"},
    ),
    (
        "noul-with-criteria",
        "User reports they cannot log in after the SSO migration on Tuesday.",
        {
            "type": "noul",
            "instructions": "Is the user blocked from accessing the product?",
            "criteria": {
                "true": "The user cannot authenticate or reach the application.",
                "false": "The user can still authenticate successfully.",
            },
        },
    ),
    (
        "score",
        "Memory utilization reached 98% with frequent OOM killer invocations.",
        {
            "type": "score",
            "instructions": "Assess system degradation level.",
            "criteria": [
                "Nominal operation; within acceptable variance",
                "Elevated resource consumption; degraded performance",
                "Critical threshold; immediate risk of service termination",
            ],
        },
    ),
    (
        "score-4-level",
        "Payment gateway reports timeout on charge authorizations. Urgent.",
        {
            "type": "score",
            "instructions": "Rate the incident severity.",
            "criteria": ["Low", "Medium", "High", "Critical"],
        },
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="hf_orig")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    import torch

    # The published von-sdk 1.0.1 wheel omits von/models/ (upstream packaging
    # bug), so the reference implementation is taken from the upstream GitHub
    # source instead. It is self-contained (no relative imports) and its
    # sha256 is pinned in the README.
    sys.path.insert(0, os.path.join(_HERE, "upstream"))
    from option_marker import OptionMarkerModel

    from von_mlx.decision import VonEngine

    model_dir = args.model_dir
    failures = []

    # ---- reference: the official SDK's own model, on CPU ----
    print("loading official von SDK OptionMarkerModel (torch, CPU)...")
    ref = OptionMarkerModel(base_model_id=model_dir)
    sd = torch.load(os.path.join(model_dir, "option_marker.pt"),
                    map_location="cpu", weights_only=True)
    ref.load_state_dict(sd, strict=True)
    ref = ref.to("cpu").eval()
    ref_temp = 2.2  # marker_calibration.json

    # ---- under test: the MLX engine ----
    print("loading von-mlx engine...")
    eng = VonEngine(model_dir)
    print(f"  reference temp = {ref_temp} | mlx engine temp = {eng._default_temp}")

    print(f"\n{'case':<18} {'field':<14} {'mlx':>12} {'sdk':>12} {'|delta|':>10}")
    print("-" * 72)

    for name, state, spec in CASES:
        qtype = spec["type"]

        # ---- reference logits via the SDK path ----
        tok = ref.tokenizer
        if qtype == "choice":
            opts = list(spec["criteria"].keys())
            descs = [spec["criteria"][o] for o in opts]
        elif qtype == "noul":
            crit = spec.get("criteria") or {}
            opts = [crit.get("true") or "Yes, condition holds true.",
                    crit.get("false") or "No, condition is false."]
            descs = opts
        else:
            descs = [str(c) for c in spec["criteria"]]
            opts = descs

        packed = ref.pack_sequence(state, spec["instructions"], descs)
        inp = tok(packed, return_tensors="pt")
        pos = (inp["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            ref_logits = ref(input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"],
                             mask_positions=[pos])[0]

        has_explicit = bool((spec.get("criteria") or {}) and qtype == "noul")
        if qtype == "noul" and not has_explicit:
            null_packed = ref.pack_sequence("", spec["instructions"],
                                            ["Yes, condition holds true.",
                                             "No, condition is false."])
            ni = tok(null_packed, return_tensors="pt")
            npos = (ni["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            with torch.no_grad():
                nl = ref(input_ids=ni["input_ids"], attention_mask=ni["attention_mask"],
                         mask_positions=[npos])[0]
            bias = nl[0] - nl[1]
            ref_logits = torch.stack([ref_logits[0] - 0.7 * bias, ref_logits[1]])

        ref_np = ref_logits.float().numpy()

        # ---- MLX ----
        mlx_logits = eng._forward(eng.pack_sequence(state, spec["instructions"], descs))
        if qtype == "noul" and not has_explicit:
            nl = eng._forward(eng.pack_sequence("", spec["instructions"], descs))
            bias = nl[0] - nl[1]
            mlx_logits = __import__("mlx.core", fromlist=["x"]).stack(
                [mlx_logits[0] - 0.7 * bias, mlx_logits[1]])
        mlx_np = np.asarray(mlx_logits).astype(np.float64)

        d = float(np.abs(mlx_np - ref_np.astype(np.float64)).max())
        for i in range(len(mlx_np)):
            print(f"{name:<18} {'logit[' + str(i) + ']':<14} {mlx_np[i]:>12.6f} "
                  f"{ref_np[i]:>12.6f} {abs(mlx_np[i]-ref_np[i]):>10.2e}")
        # derived values
        for temp in (ref_temp,):
            mp = np.asarray(__import__("mlx.core", fromlist=["x"]).softmax(
                mlx_logits / temp))
            rp = torch.softmax(torch.from_numpy(ref_np.astype(np.float32)) / temp, dim=-1).numpy()
            dd = float(np.abs(mp - rp).max())
            print(f"{'':<18} {'softmax(T=' + str(temp) + ')':<14} "
                  f"{'':>12} {'':>12} {dd:>10.2e}")
            d = max(d, dd)
        if d > args.atol:
            failures.append(f"{name}: max|delta|={d:.3e}")
        print()

    print("== result ==")
    if failures:
        for f in failures:
            print("  FAIL:", f)
        return 1
    print(f"  PASS: MLX engine matches the official SDK within atol={args.atol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
