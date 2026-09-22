"""Quantization quality report: does 4-bit change any decision?

The raw-logit tolerance used for the fp16 check is not the right gate for a
quantized model. What matters is whether the calibrated probabilities and the
argmax decisions survive quantization. This script reports both.

Usage:
    env -u PYTHONPATH .venv/bin/python verify/quant_report.py --src hf_orig \
        --fp16 out/von-1.0-mlx/fp16 --4bit out/von-1.0-mlx/4bit
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from sdk_parity import CASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hf_orig")
    ap.add_argument("--fp16", default="out/von-1.0-mlx/fp16")
    ap.add_argument("--4bit", dest="q4", default="out/von-1.0-mlx/4bit")
    args = ap.parse_args()

    import torch

    sys.path.insert(0, os.path.join(_HERE, "upstream"))
    from option_marker import OptionMarkerModel

    from von_mlx.decision import VonEngine

    ref = OptionMarkerModel(base_model_id=args.src)
    sd = torch.load(os.path.join(args.src, "option_marker.pt"),
                    map_location="cpu", weights_only=True)
    ref.load_state_dict(sd, strict=True)
    ref = ref.to("cpu").eval()
    tok = ref.tokenizer
    T = 2.2

    engines = {
        "fp16": VonEngine(args.fp16),
        "4bit": VonEngine(args.q4),
    }

    print(f"{'case':<18} {'variant':<6} {'logit max|d|':>13} {'prob max|d|':>12} "
          f"{'argmax':>7} {'score d':>9}")
    print("-" * 72)

    worst_prob = 0.0
    flip_total = 0
    rows = []

    for name, state, spec in CASES:
        qtype = spec["type"]
        if qtype == "choice":
            opts = list(spec["criteria"].keys())
            descs = [spec["criteria"][o] for o in opts]
        elif qtype == "noul":
            crit = spec.get("criteria") or {}
            descs = [crit.get("true") or "Yes, condition holds true.",
                     crit.get("false") or "No, condition is false."]
        else:
            descs = [str(c) for c in spec["criteria"]]

        packed = ref.pack_sequence(state, spec["instructions"], descs)
        inp = tok(packed, return_tensors="pt")
        pos = (inp["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            rl = ref(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                     mask_positions=[pos])[0].float().numpy()

        has_explicit = bool((spec.get("criteria") or {}) and qtype == "noul")
        if qtype == "noul" and not has_explicit:
            np_ = ref.pack_sequence("", spec["instructions"], descs)
            ni = tok(np_, return_tensors="pt")
            npos = (ni["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            with torch.no_grad():
                nl = ref(input_ids=ni["input_ids"], attention_mask=ni["attention_mask"],
                         mask_positions=[npos])[0].float().numpy()
            rl = np.stack([rl[0] - 0.7 * (nl[0] - nl[1]), rl[1]])

        ref_p = torch.softmax(torch.from_numpy(rl / T), dim=-1).numpy()

        for variant, eng in engines.items():
            ml = eng._forward(eng.pack_sequence(state, spec["instructions"], descs))
            ml = np.asarray(ml).astype(np.float64)
            if qtype == "noul" and not has_explicit:
                nl2 = eng._forward(eng.pack_sequence("", spec["instructions"], descs))
                b = float(nl2[0] - nl2[1])
                ml = np.array([ml[0] - 0.7 * b, ml[1]])
            ml_p = np.asarray(
                __import__("mlx.core", fromlist=["x"]).softmax(
                    __import__("mlx.core", fromlist=["x"]).array(ml / T))
            )
            ld = float(np.abs(ml - rl.astype(np.float64)).max())
            pd = float(np.abs(ml_p - ref_p).max())
            same = int(ml.argmax()) == int(rl.argmax())
            if not same:
                flip_total += 1
            worst_prob = max(worst_prob, pd)
            score_d = ""
            if qtype == "score":
                s_ref = sum(i * p for i, p in enumerate(ref_p))
                s_ml = sum(i * p for i, p in enumerate(ml_p))
                score_d = f"{abs(s_ml - s_ref):.4f}"
            print(f"{name:<18} {variant:<6} {ld:>13.3e} {pd:>12.3e} "
                  f"{'OK' if same else 'FLIP':>7} {score_d:>9}")
            rows.append((name, variant, ld, pd, same))

    print("-" * 72)
    print(f"worst probability shift   : {worst_prob:.3e}")
    print(f"argmax flips              : {flip_total}/{len(rows)}")
    print()
    print("interpretation")
    print("  fp16 is the equivalence check: it should track torch to ~1e-2 on raw")
    print("  logits (fp16 storage, 28 layers) with identical argmax every time.")
    print("  For 4bit the raw-logit delta is expected to be large; the gate is the")
    print("  probability/decision column. A 4-bit model is usable when argmax is")
    print("  stable and the probability shift stays small relative to the")
    print("  confidence gaps the caller actually thresholds on.")


if __name__ == "__main__":
    main()
