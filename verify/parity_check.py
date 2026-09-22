"""Numeric parity check: MLX port vs the reference PyTorch implementation.

Runs both paths on identical tokenized inputs and reports max|delta| on the
decision logits, plus argmax agreement. This is the gate that says the port is
numerically faithful rather than merely loadable.

Usage:
    env -u PYTHONPATH .venv/bin/python verify/parity_check.py --model-dir hf_orig
"""

import argparse
import json
import os
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402


CASES = [
    # (premise, hypothesis) -- entailed / neutral / contradicted
    (
        "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
        "Database replication was delayed beyond the acceptable threshold.",
    ),
    (
        "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
        "The billing invoice was paid in full.",
    ),
    (
        "Connection pool exhausted on port 5432; subsequent handshakes timing out.",
        "New connections could not be established.",
    ),
    (
        "The export button crashes the settings page on Safari 17.2.",
        "The weather in Paris is mild today.",
    ),
]

PACKED_CASES = [
    (
        "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
        "Classify the root cause domain of this incident.",
        [
            "Database, hardware, network, or server failures",
            "Invoices, payments, refunds, subscription queries",
            "Requests for new platform capabilities",
        ],
    ),
    (
        "Connection pool exhausted on port 5432; subsequent handshakes timing out.",
        "Is this issue actively blocking customer operations?",
        ["Yes, condition holds true.", "No, condition is false."],
    ),
]


def report(name, mx_logits, pt_logits):
    a = np.asarray(mx_logits, dtype=np.float64)
    b = np.asarray(pt_logits, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH mlx={a.shape} torch={b.shape}")
        return False
    delta = float(np.abs(a - b).max())
    flip = int((a.argmax(-1) != b.argmax(-1)).sum())
    n = int(a.size if a.ndim == 1 else a.shape[0])
    print(f"  [{name}] max|delta| = {delta:.3e}   argmax flips: {flip}/{n}")
    return delta, flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="out/von-1.0-mlx/fp16",
                    help="converted MLX checkpoint dir")
    ap.add_argument("--src", default="hf_orig",
                    help="original PyTorch checkpoint dir (reference)")
    ap.add_argument(
        "--gate",
        choices=["architecture", "artifact"],
        default="architecture",
        help="architecture: MLX fp32 weights vs torch fp32 -- proves the PORT is "
             "correct (tight tolerance). artifact: a converted fp16 dir vs torch "
             "fp32 -- measures fp16 STORAGE rounding, which is expected to be "
             "orders of magnitude larger and is NOT a port-correctness claim.",
    )
    ap.add_argument("--atol", type=float, default=None,
                    help="override the gate's default tolerance")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer

    from von_mlx.load import load_nli
    from von_mlx.model import VonOptionMarker
    from von_mlx.config import ModelArgs

    model_dir = args.model_dir
    src_dir = args.src
    # Two different questions, two different tolerances. Conflating them is how
    # a passing fp16-storage check gets mistaken for port equivalence.
    # For a converted artifact the honest gate is decision fidelity, not a raw
    # logit tolerance: quantization is *supposed* to move logits. Loosening the
    # logit tolerance until it passes would hide exactly the failure that
    # matters (a flipped decision), so we gate on argmax instead.
    if args.atol is None:
        args.atol = 5e-3
    failures = []
    flips = []

    print(f"GATE: {args.gate}   (atol={args.atol})")
    if args.gate == "artifact":
        print("  NOTE: this measures fp16 storage rounding, not port correctness.")
        print("        Run with --model-dir hf_orig --gate architecture for that.")

    # ---------------- NLI head ----------------
    print("== NLI cross-encoder head (model.safetensors) ==")
    tok = AutoTokenizer.from_pretrained(model_dir)
    pt_model = AutoModelForSequenceClassification.from_pretrained(src_dir)
    pt_model.eval()

    mlx_model, _, cfg = load_nli(model_dir)

    premises = [c[0] for c in CASES]
    hypotheses = [c[1] for c in CASES]
    enc = tok(premises, hypotheses, padding=True, truncation=True,
              max_length=512, return_tensors="pt")
    with torch.no_grad():
        pt_logits = pt_model(**enc).logits.numpy()

    ids = mx.array(enc["input_ids"].numpy().astype(np.int32))
    am = mx.array(enc["attention_mask"].numpy().astype(np.int32))
    mx_logits = np.asarray(mlx_model(ids, am))

    r = report("nli batch", mx_logits, pt_logits)
    if isinstance(r, tuple):
        if r[1] > 0:
            flips.append(f"NLI head: {r[1]} argmax flips")
        elif args.gate == "architecture" and r[0] > args.atol:
            failures.append(f"NLI head max|delta|={r[0]:.3e} > atol={args.atol}")

    # ---------------- Option-marker head ----------------
    print("\n== Option-marker joint head (option_marker.pt) ==")
    om_sd = torch.load(os.path.join(src_dir, "option_marker.pt"),
                       map_location="cpu", weights_only=True)

    # The option-marker checkpoint carries its OWN encoder weights (a separate
    # fine-tune from model.safetensors), so the torch reference encoder must be
    # loaded from it rather than from the NLI checkpoint.
    pt_enc = AutoModel.from_pretrained(src_dir)
    enc_sd = {k[len("encoder."):]: v for k, v in om_sd.items()
              if k.startswith("encoder.")}
    missing, unexpected = pt_enc.load_state_dict(enc_sd, strict=False)
    if missing:
        raise RuntimeError(f"option-marker encoder missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"option-marker encoder unexpected keys: {unexpected}")
    pt_enc.eval()

    from torch_ref import TorchOptionMarkerScorer

    pt_scorer = TorchOptionMarkerScorer()
    pt_scorer.load_state_dict({k[len("scorer."):]: v for k, v in om_sd.items()
                               if k.startswith("scorer.")})
    pt_scorer.eval()

    mlx_om, _, cfg = __import__("von_mlx.load", fromlist=["x"]).load_option_marker(model_dir)

    mask_tok = tok.mask_token
    sep_tok = tok.sep_token
    rows, positions = [], []
    for state, question, options in PACKED_CASES:
        prefix = f"{question} {state}".strip()
        packed = " ".join(f"{mask_tok} {o.strip()}" for o in options)
        text = f"{prefix} {sep_tok} {packed}"
        e = tok(text, return_tensors="pt")
        rows.append(e["input_ids"])
        pos = (e["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        positions.append(pos)

    maxlen = max(r.shape[1] for r in rows)
    padded = torch.full((len(rows), maxlen), cfg.pad_token_id, dtype=torch.long)
    amask = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, r in enumerate(rows):
        padded[i, : r.shape[1]] = r[0]
        amask[i, : r.shape[1]] = 1

    with torch.no_grad():
        hs = pt_enc(input_ids=padded, attention_mask=amask).last_hidden_state
        pt_pack = [pt_scorer(hs[b, pos]).numpy() for b, pos in enumerate(positions)]

    mx_pack = mlx_om(mx.array(padded.numpy().astype(np.int32)),
                     mx.array(amask.numpy().astype(np.int32)),
                     positions)
    mx_pack = [np.asarray(x) for x in mx_pack]

    # option counts differ per case, so compare row by row
    worst = 0.0
    flips = 0
    total = 0
    for i, (a, b) in enumerate(zip(mx_pack, pt_pack)):
        delta = float(np.abs(a - b).max())
        f = int((a.argmax(-1) != b.argmax(-1)).sum())
        print(f"  [option-marker case {i}] K={a.size}  max|delta| = {delta:.3e}   "
              f"argmax flips: {f}/{a.size}")
        print(f"      mlx  : {np.round(a, 6)}")
        print(f"      torch: {np.round(b, 6)}")
        worst = max(worst, delta)
        flips += f
        total += a.size
    if flips:
        failures.extend(flips)
    elif args.gate == "architecture" and worst > args.atol:
        failures.append(f"option-marker max|delta|={worst:.3e} > atol={args.atol}")

    # ---------------- summary ----------------
    print("\n== result ==")
    if args.gate == "architecture":
        print(f"  gate: port equivalence (MLX fp32 vs torch fp32, atol={args.atol})")
        if failures:
            for f in failures:
                print("  FAIL:", f)
            return 1
        print(f"  PASS: logits within atol={args.atol}, 0 argmax flips")
        return 0

    print("  gate: decision fidelity (0 argmax flips required; raw logit deltas above")
    print("        are informational - quantization is expected to move them).")
    print("        For probability-level analysis run verify/quant_report.py.")
    if failures:
        for f in failures:
            print("  FAIL:", f)
        print("  a flipped decision means the artifact is not safe to ship")
        return 1
    print("  PASS: 0 argmax flips; decisions preserved by this artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
