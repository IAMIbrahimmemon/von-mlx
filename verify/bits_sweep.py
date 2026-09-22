"""Bit-width sweep for the option-marker head.

Reports decision fidelity (argmax stability and probability deviation) at
several quantization widths so the published variants are chosen from measured
behaviour rather than assumed.

Usage:
    env -u PYTHONPATH .venv/bin/python verify/bits_sweep.py --src hf_orig
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from sdk_parity import CASES  # noqa: E402
from von_mlx.config import ModelArgs  # noqa: E402
from von_mlx.load import QUANT_GROUP_SIZE, quant_predicate  # noqa: E402
from von_mlx.model import VonOptionMarker  # noqa: E402

T = 2.2


def build(src: str, config: ModelArgs, bits: int | None):
    import torch

    raw = torch.load(os.path.join(src, "option_marker.pt"),
                     map_location="cpu", weights_only=True)
    weights = {k: mx.array(v.detach().cpu().float().numpy()).astype(mx.float16)
               for k, v in raw.items()}
    del raw
    model = VonOptionMarker(config)
    model.load_weights(list(weights.items()))
    model.eval()
    if bits:
        nn.quantize(model, group_size=QUANT_GROUP_SIZE, bits=bits,
                    class_predicate=quant_predicate)
        model.eval()
    return model


def tokenize_and_pack(tok, mask_id, state, question, descs):
    mask = tok.mask_token
    sep = tok.sep_token
    prefix = f"{question} {state}".strip() if question else state.strip()
    packed = " ".join(f"{mask} {d.strip()}" for d in descs)
    text = f"{prefix} {sep} {packed}"
    enc = tok(text, return_tensors="np")
    pos = [int(p) for p in (enc["input_ids"][0] == mask_id).nonzero()[0]]
    return enc, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hf_orig")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.join(_HERE, "upstream"))
    from option_marker import OptionMarkerModel

    config = ModelArgs.from_json(os.path.join(args.src, "config.json"))
    tok = AutoTokenizer.from_pretrained(args.src)
    mask_id = tok.mask_token_id

    import torch

    ref = OptionMarkerModel(base_model_id=args.src)
    sd = torch.load(os.path.join(args.src, "option_marker.pt"),
                    map_location="cpu", weights_only=True)
    ref.load_state_dict(sd, strict=True)
    ref = ref.to("cpu").eval()

    variants = {}
    for bits in (None, 8, 6, 5, 4):
        label = "fp16" if bits is None else f"{bits}bit"
        print(f"building {label}...")
        variants[label] = build(args.src, config, bits)

    print(f"\n{'case':<18} {'variant':<6} {'prob max|d|':>12} {'argmax':>7} "
          f"{'score/conf d':>13}")
    print("-" * 62)

    totals = {k: {"prob": 0.0, "flips": 0, "n": 0} for k in variants}

    for name, state, spec in CASES:
        qtype = spec["type"]
        if qtype == "choice":
            descs = [spec["criteria"][o] for o in spec["criteria"]]
        elif qtype == "noul":
            crit = spec.get("criteria") or {}
            descs = [crit.get("true") or "Yes, condition holds true.",
                     crit.get("false") or "No, condition is false."]
        else:
            descs = [str(c) for c in spec["criteria"]]

        # reference
        packed = ref.pack_sequence(state, spec["instructions"], descs)
        inp = tok(packed, return_tensors="pt")
        pos = (inp["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            rl = ref(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                     mask_positions=[pos])[0].float().numpy()
        has_explicit = bool((spec.get("criteria") or {}) and qtype == "noul")
        if qtype == "noul" and not has_explicit:
            npk = ref.pack_sequence("", spec["instructions"], descs)
            ni = tok(npk, return_tensors="pt")
            npos = (ni["input_ids"][0] == ref.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            with torch.no_grad():
                nl = ref(input_ids=ni["input_ids"], attention_mask=ni["attention_mask"],
                         mask_positions=[npos])[0].float().numpy()
            rl = np.stack([rl[0] - 0.7 * (nl[0] - nl[1]), rl[1]])
        ref_p = torch.softmax(torch.from_numpy(rl / T), dim=-1).numpy()

        for label, model in variants.items():
            enc, mpos = tokenize_and_pack(tok, mask_id, state, spec["instructions"], descs)
            out = model(mx.array(enc["input_ids"].astype("int32")),
                        mx.array(enc["attention_mask"].astype("int32")), [mpos])[0]
            ml = np.asarray(out).astype(np.float64)
            if qtype == "noul" and not has_explicit:
                enc2, mp2 = tokenize_and_pack(tok, mask_id, "", spec["instructions"],
                                              ["Yes, condition holds true.",
                                               "No, condition is false."])
                o2 = model(mx.array(enc2["input_ids"].astype("int32")),
                           mx.array(enc2["attention_mask"].astype("int32")), [mp2])[0]
                b = float(np.asarray(o2)[0] - np.asarray(o2)[1])
                ml = np.array([ml[0] - 0.7 * b, ml[1]])
            ml_p = np.asarray(mx.softmax(mx.array(ml / T)))
            pd = float(np.abs(ml_p - ref_p).max())
            same = int(ml.argmax()) == int(rl.argmax())

            extra = ""
            if qtype == "score":
                extra = f"{abs(sum(i*p for i,p in enumerate(ml_p)) - sum(i*p for i,p in enumerate(ref_p))):.4f}"
            elif qtype == "choice":
                def conf(p):
                    s = sorted(p, reverse=True)
                    return s[0] - (s[1] if len(s) > 1 else 0.0)
                extra = f"{abs(conf(ml_p) - conf(ref_p)):.4f}"

            print(f"{name:<18} {label:<6} {pd:>12.3e} {'OK' if same else 'FLIP':>7} {extra:>13}")
            totals[label]["prob"] = max(totals[label]["prob"], pd)
            totals[label]["flips"] += 0 if same else 1
            totals[label]["n"] += 1
        print()

    print("=" * 62)
    print(f"{'variant':<8} {'worst prob shift':>18} {'argmax flips':>14}")
    for label, t in totals.items():
        print(f"{label:<8} {t['prob']:>18.3e} {t['flips']}/{t['n']:>13}")


if __name__ == "__main__":
    main()
