"""Model configuration for the MLX Von port."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelArgs:
    """Mirrors the HF ModernBERT config fields we actually need."""

    model_type: str = "modernbert"
    vocab_size: int = 50368
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 2624
    num_attention_heads: int = 16
    max_position_embeddings: int = 2048

    norm_eps: float = 1e-05
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    classifier_bias: bool = False

    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128

    classifier_pooling: str = "mean"
    num_labels: int = 3
    id2label: Optional[Dict[str, str]] = None
    label2id: Optional[Dict[str, int]] = None

    pad_token_id: int = 50283
    cls_token_id: int = 50281
    sep_token_id: int = 50282

    tie_word_embeddings: bool = True

    # set by the converter when the published weights are quantized
    quantization: Optional[Dict[str, Any]] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ModelArgs":
        rope = cfg.get("rope_parameters") or {}
        global_theta = (rope.get("full_attention") or {}).get(
            "rope_theta", cfg.get("global_rope_theta", 160000.0)
        )
        local_theta = (rope.get("sliding_attention") or {}).get(
            "rope_theta", cfg.get("local_rope_theta", 10000.0)
        )

        id2label = cfg.get("id2label")
        if id2label is not None:
            id2label = {str(k): v for k, v in id2label.items()}

        known = {f for f in cls.__dataclass_fields__}
        kwargs: Dict[str, Any] = {}
        for key in known:
            if key in cfg and key not in ("global_rope_theta", "local_rope_theta"):
                kwargs[key] = cfg[key]
        kwargs["global_rope_theta"] = float(global_theta)
        kwargs["local_rope_theta"] = float(local_theta)
        kwargs["id2label"] = id2label
        kwargs["num_labels"] = len(id2label) if id2label else cfg.get("num_labels", 3)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> "ModelArgs":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            if key == "quantization":
                continue
            out[key] = getattr(self, key)
        return out
