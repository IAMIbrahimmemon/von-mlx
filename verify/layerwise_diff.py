"""Layer-wise divergence localization: where does MLX drift from PyTorch?

Compares final-norm hidden states after each encoder layer plus the pooled
representation, so a parity gap can be attributed to a specific component
rather than reported as one opaque number.

Usage:
    env -u PYTHONPATH .venv/bin/python verify/layerwise_diff.py --model-dir hf_orig
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

TEXT_A = "Database replication lag on cluster us-west-2 exceeded 45 seconds."
TEXT_B = "Database replication was delayed beyond the acceptable threshold."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="hf_orig")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from von_mlx.config import ModelArgs
    from von_mlx.model import ModernBertModel
    from von_mlx.load import sanitize_nli
    from von_mlx.model import VonNLI

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    enc = tok([TEXT_A], [TEXT_B], padding=True, truncation=True,
              max_length=128, return_tensors="pt")
    ids_np = enc["input_ids"].numpy().astype(np.int32)
    am_np = enc["attention_mask"].numpy().astype(np.int32)
    print("seq_len:", ids_np.shape[1])

    # ---- torch reference: hidden states after every layer ----
    pt = AutoModel.from_pretrained(args.model_dir, output_hidden_states=True)
    pt.eval()
    with torch.no_grad():
        out = pt(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 output_hidden_states=True)
    # out.hidden_states[0] = embeddings output; [i+1] = after layer i
    pt_states = [h.numpy()[0] for h in out.hidden_states]
    print("torch hidden_states:", len(pt_states))

    # ---- MLX: manual layer-by-layer ----
    cfg = ModelArgs.from_json(os.path.join(args.model_dir, "config.json"))
    nli = VonNLI(cfg)
    weights = mx.load(os.path.join(args.model_dir, "model.safetensors"))
    nli.load_weights(list(sanitize_nli({k: v for k, v in weights.items()}, cfg).items()))
    nli.eval()

    from von_mlx.model import build_masks

    ids = mx.array(ids_np)
    am = mx.array(am_np)
    b, s = ids.shape
    gmask, smask = build_masks(am, s, cfg.local_attention // 2)

    mx_states = []
    h = nli.model.embeddings(ids)
    mx_states.append(np.asarray(h[0]))
    for layer in nli.model.layers:
        h = layer(h, gmask, smask)
        mx_states.append(np.asarray(h[0]))
    h_final = nli.model.final_norm(h)
    mx_states.append(np.asarray(h_final[0]))
    print("mlx states:", len(mx_states))

    print("\n layer |        max|delta| |  mean|delta| | max|torch|")
    print("-------+------------------+--------------+-----------")
    n = min(len(pt_states), len(mx_states))
    worst = 0.0
    for i in range(n):
        d = np.abs(pt_states[i] - mx_states[i])
        mx_abs = float(np.abs(pt_states[i]).max())
        print(f"  {i:>4} | {d.max():>16.4e} | {d.mean():>12.4e} | {mx_abs:>9.3f}")
        worst = max(worst, float(d.max()))

    print("\n(0 = embeddings, i = after encoder layer i-1, last = post final_norm)")

    # ---- pooled + head + classifier ----
    with torch.no_grad():
        pt_logits = pt.__class__  # unused
    from transformers import AutoModelForSequenceClassification
    pt_cls = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    pt_cls.eval()
    with torch.no_grad():
        pt_l = pt_cls(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).logits.numpy()
    mx_l = np.asarray(nli(ids, am))
    print(f"\nlogits  torch: {pt_l[0]}")
    print(f"logits  mlx  : {mx_l[0]}")
    print(f"logits  max|delta|: {np.abs(pt_l - mx_l).max():.4e}")
    print(f"hidden  worst max|delta| across layers: {worst:.4e}")
    print(f"logit magnitudes: {np.abs(pt_l).max():.3f}")
    print(f"relative hidden divergence: {worst / max(1e-9, np.abs(pt_states[-1]).max()):.4e}")


if __name__ == "__main__":
    main()
