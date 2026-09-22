"""Decision-fidelity evaluation of precision variants on real held-out data.

``verify/bits_sweep.py`` uses six hand-built cases -- enough to pick a bit width,
far too thin to support a claim about a decision model. This harness uses the
*upstream* benchmark sets instead:

    benchmarks/data/authored144.jsonl        144 base decision cases (split=test)
    benchmarks/data/perturbations108.jsonl   108 perturbations of those, in
                                             three flavours x 36 groups:
                                               option_reversal   options reordered
                                               criterion_wrapper criteria reworded
                                               irrelevant_context noise added
                                             all with the SAME correct option id

Metrics per variant:

* accuracy against the gold label
* agreement with the fp32 reference (the real "did precision change a decision"
  test, independent of whether the model was right to begin with)
* distribution divergence (max |dp|, KL) -- what a threshold or a downstream
  consumer actually sees
* per-perturbation-type stability: a group's members should all resolve to the
  same option *id*, whatever order the options arrived in

Usage:
    env -u PYTHONPATH .venv/bin/python verify/variant_eval.py \
        --data ~/.hermes/cache/scratch/von-bench \
        --variants fp32:out/von-1.0-mlx/fp32,fp16:out/von-1.0-mlx/fp16,8bit:out/von-1.0-mlx/8bit
"""

import argparse
import gc
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from von_mlx.decision import VonEngine  # noqa: E402
from von_mlx.types import Choice  # noqa: E402


def load_cases(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    base, pert = [], []
    for name, bucket in (("authored144.jsonl", base),
                         ("perturbations108.jsonl", pert)):
        with open(os.path.join(data_dir, name)) as fh:
            for line in fh:
                if line.strip():
                    bucket.append(json.loads(line))
    return base, pert


def predict(engine: VonEngine, case: Dict[str, Any]) -> Dict[str, float]:
    """Run one case; {option_id: probability} in the case's own order."""
    ids = [o["id"] for o in case["options"]]
    descs = [o["description"] for o in case["options"]]
    # `criteria` insertion order is the order options are packed in.
    q = Choice(instructions=case.get("question") or "",
               criteria={i: d for i, d in zip(ids, descs)})
    ans = engine.evaluate_choice(case["id"], case["state"], q)
    return {i: float(ans.probabilities.get(i, 0.0)) for i in ids}


def sdk_predict(ref, case: Dict[str, Any]):
    """Same decision via the official von SDK model (torch, CPU).

    Returns {option_id: probability} after the same temperature the MLX engine
    uses, so the two are directly comparable.
    """
    import numpy as np
    import torch

    ids = [o["id"] for o in case["options"]]
    descs = [o["description"] for o in case["options"]]
    tok = ref.tokenizer
    packed = ref.pack_sequence(case["state"], case.get("question") or "", descs)
    inp = tok(packed, return_tensors="pt")
    pos = (inp["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    with torch.no_grad():
        logits = ref(input_ids=inp["input_ids"],
                     attention_mask=inp["attention_mask"],
                     mask_positions=[pos])[0]
        probs = torch.softmax(logits / 2.2, dim=-1).tolist()
    return {i: float(p) for i, p in zip(ids, probs)}


def kl(p: Dict[str, float], q: Dict[str, float], keys: List[str]) -> float:
    return sum(
        max(p.get(k, 0.0), 1e-12) * math.log(max(p.get(k, 0.0), 1e-12)
                                             / max(q.get(k, 0.0), 1e-12))
        for k in keys
    )


def argmax(d: Dict[str, float]) -> str:
    return max(d.items(), key=lambda kv: kv[1])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variants", required=True,
                    help="name:path pairs, comma separated; FIRST one is the reference")
    ap.add_argument("--sdk-control", action="store_true",
                    help="also score the official SDK model (torch CPU) as a control")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    specs = []
    for pair in args.variants.split(","):
        name, _, path = pair.strip().partition(":")
        specs.append((name, path))
    ref_name = specs[0][0]

    base, pert = load_cases(args.data)
    cases = base + pert
    print(f"loaded {len(base)} base + {len(pert)} perturbation cases")
    print(f"reference variant: {ref_name}\n")

    # ---- run each MLX variant, keeping only the probabilities ----
    preds: Dict[str, List[Dict[str, float]]] = {}
    for name, path in specs:
        engine = VonEngine(path)
        out = []
        for c in cases:
            out.append(predict(engine, c))
        preds[name] = out
        del engine
        gc.collect()
        print(f"  scored {name:<5} ({path})")

    sdk_preds = None
    if args.sdk_control:
        import torch

        # Published von-sdk 1.0.1 omits von/models/, so the reference comes from
        # the upstream GitHub source, vendored into verify/upstream/.
        sys.path.insert(0, os.path.join(_HERE, "upstream"))
        from option_marker import OptionMarkerModel

        src = "hf_orig"
        print(f"\n  loading official SDK OptionMarkerModel from {src} (torch, CPU)...")
        ref = OptionMarkerModel(base_model_id=src)
        ref.load_state_dict(
            torch.load(os.path.join(src, "option_marker.pt"),
                       map_location="cpu", weights_only=True),
            strict=True,
        )
        ref = ref.to("cpu").eval()
        sdk_preds = [sdk_predict(ref, c) for c in cases]
        print("  scored sdk   (official SDK control)")

    # ---- metrics ----
    def accuracy(pred_list: List[Dict[str, float]]) -> float:
        good = 0
        for c, p in zip(cases, pred_list):
            label_id = c["options"][c["label"]]["id"]
            if argmax(p) == label_id:
                good += 1
        return good / len(cases)

    print("\n" + "=" * 72)
    print("ACCURACY vs gold label")
    print("=" * 72)
    acc = {}
    for name, _ in specs:
        acc[name] = accuracy(preds[name])
        print(f"  {name:<5} {acc[name]:.4f}  ({round(acc[name]*len(cases))}/{len(cases)})")
    if sdk_preds is not None:
        a = accuracy(sdk_preds)
        print(f"  {'sdk':<5} {a:.4f}  ({round(a*len(cases))}/{len(cases)})   <- official "
              f"SDK control, same data")

    print("\n" + "=" * 72)
    print(f"AGREEMENT with {ref_name} (the reference variant)")
    print("=" * 72)
    flip_rows = []
    for name, _ in specs:
        if name == ref_name:
            continue
        flips = 0
        max_dp = mean_dp = max_kl = mean_kl = 0.0
        marg = []
        for c, pr, pc in zip(cases, preds[ref_name], preds[name]):
            keys = [o["id"] for o in c["options"]]
            if argmax(pr) != argmax(pc):
                flips += 1
                flip_rows.append((c, argmax(pr), argmax(pc)))
            dp = max(abs(pr[k] - pc[k]) for k in keys)
            max_dp = max(max_dp, dp)
            mean_dp += dp
            k = kl(pr, pc, keys)
            max_kl = max(max_kl, k)
            mean_kl += k
            sp = sorted(pc.values(), reverse=True)
            marg.append(sp[0] - sp[1])
        n = len(cases)
        print(f"  {name:<5} flips {flips}/{n}   "
              f"max|dp| {max_dp:.3e}  mean {mean_dp/n:.3e}   "
              f"maxKL {max_kl:.3e}")
        print(f"        margin vs runner-up: min {min(marg):.4f}  "
              f"median {sorted(marg)[len(marg)//2]:.4f}  "
              f"cases under 0.02: {sum(m < 0.02 for m in marg)}")

    if flip_rows:
        print("\n=== FLIPPED DECISIONS ===")
        for c, r, p in flip_rows:
            label_id = c["options"][c["label"]]["id"]
            print(f"  {c['id'][:12]} label={label_id:<14} {r:<14} -> {p}")

    # ---- perturbation stability, by flavour ----
    print("\n" + "=" * 72)
    print("PERTURBATION STABILITY (a group's members should share one option id)")
    print("=" * 72)
    groups: Dict[str, List[Dict]] = {}
    for c in cases:
        groups.setdefault(c["group_id"].split("/")[0], []).append(c)

    by_flavour: Dict[str, List[Tuple[Dict, Dict]]] = {}
    for gid, members in groups.items():
        b = [m for m in members if "/" not in m["group_id"]]
        for m in members:
            if "/" not in m["group_id"]:
                continue
            flav = m["provenance"]["variant"]
            by_flavour.setdefault(flav, []).append((b[0] if b else m, m))

    def stability(pred_list: List[Dict[str, float]]):
        gid_ok = 0
        for gid, members in groups.items():
            idxs = [cases.index(m) for m in members]
            names = {argmax(pred_list[i]) for i in idxs}
            if len(names) == 1:
                gid_ok += 1
        per_flavour = {}
        for flav, pairs in by_flavour.items():
            ok = 0
            for b, m in pairs:
                pb = pred_list[cases.index(b)]
                pm = pred_list[cases.index(m)]
                # base and its perturbation should agree on the option id
                lb = b["options"][b["label"]]["id"]
                if argmax(pb) == argmax(pm) == lb:
                    ok += 1
                elif argmax(pb) == argmax(pm):
                    ok += 1
            per_flavour[flav] = (ok, len(pairs))
        return gid_ok, len(groups), per_flavour

    variants_for_stab = list(specs) + ([("sdk", None)] if sdk_preds is not None else [])
    for name, _ in variants_for_stab:
        pl = sdk_preds if name == "sdk" else preds[name]
        gid_ok, n_groups, per_flavour = stability(pl)
        print(f"\n  {name}:  {gid_ok}/{n_groups} groups fully consistent "
              f"({gid_ok/n_groups:.1%})")
        for flav, (ok, tot) in sorted(per_flavour.items()):
            print(f"        {flav:<20} base/pert agree {ok}/{tot} ({ok/tot:.1%})")

    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump({"accuracy": acc,
                       "sdk_accuracy": accuracy(sdk_preds) if sdk_preds else None,
                       "preds": {k: v for k, v in preds.items()},
                       "case_ids": [c["id"] for c in cases]}, fh, indent=2)
        print(f"\nper-case results -> {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
