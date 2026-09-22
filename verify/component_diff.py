"""Component-level diagnosis: isolate which op in a single encoder layer diverges.

Compares, for layer 0 of the Von ModernBERT encoder:
  embeddings -> Wqkv -> RoPE -> SDPA+Wo -> MLP, against PyTorch.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import torch

from transformers import AutoModel, AutoTokenizer

from von_mlx.config import ModelArgs
from von_mlx.model import build_masks
from von_mlx.load import sanitize_nli
from von_mlx.model import VonNLI

DIR = "hf_orig"
TEXT_A = "Database replication lag on cluster us-west-2 exceeded 45 seconds."
TEXT_B = "Database replication was delayed beyond the acceptable threshold."


def d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max())


def main():
    tok = AutoTokenizer.from_pretrained(DIR)
    enc = tok([TEXT_A], [TEXT_B], return_tensors="pt")
    print("input_ids shape:", enc["input_ids"].shape, "| padding used:",
          enc["input_ids"].eq(tok.pad_token_id).any().item())

    cfg = ModelArgs.from_json(os.path.join(DIR, "config.json"))
    nli = VonNLI(cfg)
    w = mx.load(os.path.join(DIR, "model.safetensors"))
    nli.load_weights(list(sanitize_nli({k: v for k, v in w.items()}, cfg).items()))
    nli.eval()

    pt = AutoModel.from_pretrained(DIR)
    pt.eval()

    ids = mx.array(enc["input_ids"].numpy().astype(np.int32))
    am = mx.array(enc["attention_mask"].numpy().astype(np.int32))
    b, s = ids.shape

    # ---------- embeddings ----------
    with torch.no_grad():
        pt_emb = pt.embeddings(input_ids=enc["input_ids"])
    mx_emb = nli.model.embeddings(ids)
    print(f"\nembeddings        max|d| = {d(mx_emb, pt_emb.numpy()):.4e}")

    layer_id = 0
    mx_layer = nli.model.layers[layer_id]
    pt_layer = pt.layers[layer_id]

    # ---------- Wqkv ----------
    with torch.no_grad():
        pt_wqkv = pt_layer.attn.Wqkv(pt_emb)
    mx_wqkv = mx_layer.attn.Wqkv(mx_emb)
    print(f"Wqkv              max|d| = {d(mx_wqkv, pt_wqkv.numpy()):.4e}")

    # ---------- RoPE ----------
    # torch: uses position_embeddings from rotary_emb(sin,cos) computed once at model level
    with torch.no_grad():
        pos_ids = torch.arange(s).unsqueeze(0)
        cos, sin = pt.rotary_emb(pt_emb, position_ids=pos_ids)
    # recompute torch rope manually from the same cos/sin to mirror MLX
    import transformers.models.modernbert.modeling_modernbert as M

    def torch_rope(x_bsd, cos, sin):
        # x: (B,S,H) -> (B,H,S,D); apply HF apply_rotary_pos_emb with unsqueeze_dim=1
        B, S, H = x_bsd.shape
        D = cfg.head_dim
        q = x_bsd.view(B, S, cfg.num_attention_heads, D).transpose(1, 2)
        q = M.apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)[0]
        return q

    mx_q = mx_wqkv.reshape(b, s, 3, cfg.num_attention_heads, cfg.head_dim).transpose(0, 3, 2, 1, 4)[:, :, 0]
    from von_mlx.model import ModernBertAttention

    mx_q_rope = mx_layer.attn.rotary_emb(mx_q)
    pt_q_rope = torch_rope(pt_wqkv, cos, sin)
    print(f"q after RoPE      max|d| = {d(mx_q_rope, pt_q_rope.numpy()):.4e}")
    print(f"  |q| scale  mlx={float(mx.abs(mx_q).max()):.3f} torch={pt_q_rope.abs().max().item():.3f}")

    # ---------- mask comparison ----------
    gmask, smask = build_masks(am, s, cfg.local_attention // 2)
    print(f"\nmask: global all-true = {bool(mx.all(gmask))} | sliding window half = {cfg.local_attention//2}")
    print("  mx sliding row q=6 allowed kv:", mx.array(mx.nonzero(smask[0, 0, 6])[0]).tolist()[:20])

    # torch's own mask for the same shape
    from transformers import masking_utils as MU

    pt_cfg = pt.config
    emb = torch.zeros(b, s, cfg.hidden_size)
    tg = MU.create_bidirectional_mask(config=pt_cfg, inputs_embeds=emb,
                                      attention_mask=enc["attention_mask"])
    ts = MU.create_bidirectional_sliding_window_mask(config=pt_cfg, inputs_embeds=emb,
                                                     attention_mask=enc["attention_mask"])
    print(f"  torch global mask: {None if tg is None else tg.shape}")
    if ts is not None:
        print("  torch sliding q=6 allowed kv:", ts[0, 0, 6].nonzero().flatten().tolist()[:20])

    # ---------- attention output ----------
    with torch.no_grad():
        pt_attn_out = pt_layer.attn(hidden_states=pt_emb, position_embeddings=(cos, sin),
                                    attention_mask=None if (tg is None and ts is None) else {
                                        "full_attention": tg, "sliding_attention": ts},
                                    )
    # torch returns (attn_output, attn_weights)
    pt_attn_out = pt_attn_out[0] if isinstance(pt_attn_out, tuple) else pt_attn_out
    mx_attn_out = mx_layer.attn(mx_emb, gmask, smask)
    print(f"\nattn(Wo) out      max|d| = {d(mx_attn_out, pt_attn_out.numpy()):.4e}")

    # ---------- MLP ----------
    with torch.no_grad():
        pt_mlp = pt_layer.mlp(pt_layer.mlp_norm(pt_emb))
    mx_mlp = mx_layer.mlp(mx_layer.mlp_norm(mx_emb))
    print(f"MLP out           max|d| = {d(mx_mlp, pt_mlp.numpy()):.4e}")

    # ---------- manual MLX vs manual torch MLP from the SAME input ----------
    with torch.no_grad():
        pt_w = pt_layer.mlp.Wi(pt_emb)
    mx_w = mx_layer.mlp.Wi(mx_emb)
    print(f"  MLP Wi         max|d| = {d(mx_w, pt_w.numpy()):.4e}")
    print("  MLP act gate   max|d| = {:.4e}".format(
        d(mx_layer.mlp.act(mx_w[..., : mx_w.shape[-1] // 2]),
          torch.nn.functional.gelu(pt_w[..., : pt_w.shape[-1] // 2]).numpy())))



if __name__ == "__main__":
    main()
