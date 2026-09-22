"""MLX implementation of the ModernBERT encoder used by Von, plus its two heads.

Ported from the reference bidirectional encoder implementation in
Blaizzy/mlx-embeddings (``mlx_embeddings/models/modernbert.py``, MIT), matched
against HuggingFace ``ModernBertForSequenceClassification`` in transformers.

Layer/parameter naming is deliberately identical to the HF checkpoint so the
converter is a straight key passthrough (no renaming, no transposes):

    model.embeddings.tok_embeddings.weight      -> model.embeddings.tok_embeddings.weight
    model.layers.N.attn.Wqkv.weight             -> model.layers.N.attn.Wqkv.weight
    head.dense.weight                           -> head.dense.weight
    classifier.weight                           -> classifier.weight

For the option-marker checkpoint the HF tree is ``encoder.*`` + ``scorer.*``,
which this module mirrors exactly.
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs

NEG_INF = -1e9

#: In MLX >= 0.31 the ``approx`` semantics changed: ``"precise"`` now selects the
#: *tanh* approximation and ``"none"`` is the exact erf formulation. HF's
#: ``ACT2FN["gelu"]`` is the exact erf GELU, so we must ask for ``"none"``.
#: Using ``"precise"`` here silently swaps the activation (max|delta| ~4.7e-4 per
#: GELU evaluation, compounding across 28 layers).
EXACT_GELU = "none"


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def build_masks(
    attention_mask: mx.array, seq_len: int, half_window: int
) -> Tuple[mx.array, mx.array]:
    """Return (global_mask, sliding_window_mask) as boolean arrays.

    ``attention_mask`` is (B, S) with 1 = real token, 0 = padding.
    The global mask broadcasts over heads and query positions: (B, 1, 1, S).
    The sliding mask additionally restricts each query to a +/- half_window
    band: (B, 1, S, S).
    """
    pad = attention_mask.astype(mx.bool_)[:, None, None, :]  # (B,1,1,S)

    idx = mx.arange(seq_len)
    distance = mx.abs(idx[None, :] - idx[:, None])  # (S,S)
    window = distance <= half_window  # (S,S) -> row=query, col=key
    window = window[None, None, :, :]  # (1,1,S,S)

    sliding = mx.logical_and(window, pad)
    return pad, sliding


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU-style MLP: Wi projects to 2*intermediate, split into value/gate."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias
        )
        self.act = nn.GELU(approx=EXACT_GELU)
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        x = self.Wi(hidden_states)
        split = x.shape[-1] // 2
        value, gate = x[..., :split], x[..., split:]
        return self.Wo(self.act(value) * gate)


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.all_head_size = self.head_dim * self.num_heads

        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias
        )

        is_local = (layer_id % config.global_attn_every_n_layers) != 0
        self.is_local = is_local
        self.window = config.local_attention // 2
        rope_theta = config.local_rope_theta if is_local else config.global_rope_theta
        self.rotary_emb = nn.RoPE(dims=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(
        self, hidden_states: mx.array, global_mask: mx.array, sliding_mask: mx.array
    ) -> mx.array:
        batch, seq_len, _ = hidden_states.shape

        qkv = self.Wqkv(hidden_states)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(0, 3, 2, 1, 4)  # (B, H, 3, S, D)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = self.rotary_emb(query)
        key = self.rotary_emb(key)

        mask = sliding_mask if self.is_local else global_mask
        attn = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=mask
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.all_head_size)
        return self.Wo(attn)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: int):
        super().__init__()
        # Layer 0 has no pre-attention norm in ModernBERT.
        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        )
        self.attn = ModernBertAttention(config, layer_id)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def __call__(
        self, hidden_states: mx.array, global_mask: mx.array, sliding_mask: mx.array
    ) -> mx.array:
        normed = self.attn_norm(hidden_states)
        hidden_states = hidden_states + self.attn(normed, global_mask, sliding_mask)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertModel(nn.Module):
    """Bidirectional encoder. Returns the final hidden state (B, S, H)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        batch, seq_len = input_ids.shape
        global_mask, sliding_mask = build_masks(
            attention_mask, seq_len, self.config.local_attention // 2
        )

        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, global_mask, sliding_mask)
        return self.final_norm(hidden_states)


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------


class ModernBertPredictionHead(nn.Module):
    """dense -> gelu -> LayerNorm, as used by both Von heads."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.classifier_bias
        )
        self.act = nn.GELU(approx=EXACT_GELU)
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.norm(self.act(self.dense(hidden_states)))


def _mean_pool(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    mask = attention_mask[..., None].astype(hidden_states.dtype)
    summed = mx.sum(hidden_states * mask, axis=1)
    counts = mx.maximum(mx.sum(mask, axis=1), 1.0)
    return summed / counts


class VonNLI(nn.Module):
    """3-way NLI cross-encoder with mean pooling -> head -> classifier.

    Mirrors ``ModernBertForSequenceClassification``; weight names match
    ``wfzyx/von-1.0`` -> ``model.safetensors`` exactly.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        hidden = self.model(input_ids, attention_mask)
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        if self.config.classifier_pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = _mean_pool(hidden, attention_mask)

        # fp32 head for calibrated logits
        return self.classifier(self.head(pooled).astype(mx.float32))


class OptionMarkerScorer(nn.Module):
    """MLP scoring head over [MASK] marker representations."""

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-05, bias=True)
        self.dense = nn.Linear(hidden_size, hidden_size // 2, bias=True)
        self.act = nn.GELU(approx=EXACT_GELU)
        self.norm = nn.LayerNorm(hidden_size // 2, eps=1e-05, bias=True)
        self.out_proj = nn.Linear(hidden_size // 2, 1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.input_norm(x)
        h = self.act(self.dense(h))
        h = self.norm(h)
        return self.out_proj(h).squeeze(-1)


class VonOptionMarker(nn.Module):
    """Single-pass joint decision model: one sequence, K [MASK] markers.

    Mirrors ``OptionMarkerModel`` from the von SDK; weight names match
    ``wfzyx/von-1.0`` -> ``option_marker.pt`` exactly (``encoder.*`` +
    ``scorer.*``).
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = ModernBertModel(config)
        self.scorer = OptionMarkerScorer(config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        mask_positions: Optional[List[List[int]]] = None,
    ) -> List[mx.array]:
        hidden = self.encoder(input_ids, attention_mask)
        if mask_positions is None:
            return []

        batch_logits: List[mx.array] = []
        for row, positions in enumerate(mask_positions):
            reps = hidden[row, mx.array(positions, dtype=mx.int32)]
            batch_logits.append(self.scorer(reps.astype(mx.float32)))
        return batch_logits


MODEL_CLASSES = {
    "nli": VonNLI,
    "option_marker": VonOptionMarker,
}
