"""Loading + weight sanitization for the MLX Von port.

Two checkpoint flavours are supported for each head:

``fp16``   full-precision MLX arrays (1:1 with the HF/torch tensors)
``4bit``   MLX affine quantization, group_size 64

The quantize predicate lives here because the converter and the loader MUST
agree on it exactly -- if they ever diverge, ``load_weights`` silently fails to
match keys and the model runs with random weights.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .model import MODEL_CLASSES, VonNLI, VonOptionMarker

DEFAULT_NLI_REPO = "wfzyx/von-1.0"
NLI_WEIGHTS = "model.safetensors"
MARKER_WEIGHTS = "option_marker.safetensors"
MARKER_WEIGHTS_LEGACY = "option_marker.pt"

#: Path fragments whose Linear layers stay in fp16. These carry the calibrated
#: decision heads; quantizing them measurably degrades the probability
#: calibration the model is trained for, and they are <1% of the parameters.
FP16_PATH_FRAGMENTS = ("classifier", "scorer", "head")

QUANT_GROUP_SIZE = 64
QUANT_BITS = 4


def quant_predicate(path: str, module: nn.Module) -> bool:
    """Quantize only large 2-D weight matrices; keep heads in fp16.

    Must be identical in ``convert.py`` and in the loader.
    """
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    if any(frag in path for frag in FP16_PATH_FRAGMENTS):
        return False
    if module.weight.shape[-1] % QUANT_GROUP_SIZE != 0:
        return False
    return True


def _linearize(tree: Any, prefix: str = "") -> Dict[str, Tuple[int, ...]]:
    flat: Dict[str, Tuple[int, ...]] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            flat.update(_linearize(value, f"{prefix}{key}."))
    elif isinstance(tree, list):
        for i, item in enumerate(tree):
            flat.update(_linearize(item, f"{prefix}{i}."))
    elif isinstance(tree, mx.array):
        flat[prefix.rstrip(".")] = tuple(tree.shape)
    return flat


def _as_mx(value: Any) -> mx.array:
    return mx.array(value)


# --------------------------------------------------------------------------
# sanitize -- names are kept identical to the source checkpoints, so these are
# passthroughs plus a strict completeness check.
# --------------------------------------------------------------------------


def sanitize(
    weights: Dict[str, mx.array], model: nn.Module, head: str
) -> Dict[str, mx.array]:
    """Verify ``weights`` covers exactly ``model``'s parameters, then pass through.

    Raises rather than silently loading a partially-initialised model.
    """
    expected = set(_linearize(model.parameters()))
    got = set(weights)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise ValueError(
            f"{head} checkpoint does not match the port.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(extra)}): {extra[:8]}\n"
            f"  hint: for a quantized checkpoint, load with quantized=True"
        )
    return weights


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def _read_config(model_dir: str) -> ModelArgs:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as fh:
        return ModelArgs.from_dict(json.load(fh))


def _build(cls, config: ModelArgs, bits: Optional[int]) -> nn.Module:
    model = cls(config)
    if bits:
        nn.quantize(
            model,
            group_size=QUANT_GROUP_SIZE,
            bits=bits,
            class_predicate=quant_predicate,
        )
    return model


def _detect_bits(path: str, cls, config: ModelArgs) -> Optional[int]:
    """Infer the quantized bit width from the checkpoint's packed tensor shapes.

    A quantized ``Linear`` stores its weight packed into uint32 words, so the
    last dimension becomes ``orig_dim * bits / 32``. Comparing that against the
    fp16 module's expected shape recovers ``bits`` exactly, which matters
    because building the module at the wrong width fails to load at all.
    """
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as fh:
        keys = list(fh.keys())
        if not any(k.endswith(".scales") for k in keys):
            return None
        probe = "encoder.embeddings.tok_embeddings.weight"
        if probe not in keys:
            probe = next(k for k in keys if k.endswith(".weight")
                         and not k.endswith((".scales", ".biases")))
        packed = fh.get_slice(probe).get_shape()[-1]

    expected = dict(_linearize(cls(config).parameters()))[probe][-1]
    bits = round(packed * 32 / expected)
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(
            f"cannot infer quantization width for {path}: packed={packed}, "
            f"expected={expected} -> bits={bits}"
        )
    return int(bits)


def _read_quant_config(model_dir: str) -> Optional[int]:
    """Read ``quantization.bits`` from config.json if the converter wrote it."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    quant = cfg.get("quantization")
    if isinstance(quant, dict) and quant.get("bits"):
        return int(quant["bits"])
    return None


def _resolve_weights(name: str, model_dir: str) -> str:
    """Find ``name`` in ``model_dir``, else in its 4bit/ or fp16/ sibling."""
    direct = os.path.join(model_dir, name)
    if os.path.exists(direct):
        return direct
    for variant in ("4bit", "fp16"):
        cand = os.path.join(model_dir, variant, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{name} not found in {model_dir} or its variants")


def load_nli(model_dir: str, quantized: Optional[bool] = None):
    """Load the NLI cross-encoder head. Returns ``(model, tokenizer, config)``.

    The quantization width is taken from ``config.json`` when present, else
    inferred from the checkpoint tensor shapes. ``quantized=False`` forces fp16.
    """
    from transformers import AutoTokenizer

    config = _read_config(model_dir)
    path = _resolve_weights(NLI_WEIGHTS, model_dir)
    bits = None if quantized is False else _read_quant_config(model_dir)
    if bits is None and quantized is not False:
        bits = _detect_bits(path, VonNLI, config)

    weights = {k: _as_mx(v) for k, v in mx.load(path).items()}
    model = _build(VonNLI, config, bits)
    weights = sanitize(weights, model, "nli")
    model.load_weights(list(weights.items()))
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_dir), config


def load_option_marker(model_dir: str, quantized: Optional[bool] = None):
    """Load the option-marker joint scorer. Returns ``(model, tokenizer, config)``.

    The quantization width is taken from ``config.json`` when present, else
    inferred from the checkpoint tensor shapes. ``quantized=False`` forces fp16.
    """
    from transformers import AutoTokenizer

    config = _read_config(model_dir)

    st_path = None
    try:
        st_path = _resolve_weights(MARKER_WEIGHTS, model_dir)
    except FileNotFoundError:
        pass

    if st_path is not None:
        bits = None if quantized is False else _read_quant_config(model_dir)
        if bits is None and quantized is not False:
            bits = _detect_bits(st_path, VonOptionMarker, config)
        weights = {k: _as_mx(v) for k, v in mx.load(st_path).items()}
        model = _build(VonOptionMarker, config, bits)
        weights = sanitize(weights, model, "option_marker")
        model.load_weights(list(weights.items()))
        model.eval()
        return model, AutoTokenizer.from_pretrained(model_dir), config

    # No MLX checkpoint: fall back to the original torch archive.
    legacy = os.path.join(model_dir, MARKER_WEIGHTS_LEGACY)
    if not os.path.exists(legacy):
        raise FileNotFoundError(
            f"neither {MARKER_WEIGHTS} nor {legacy} found in {model_dir}"
        )
    import torch

    raw = torch.load(legacy, map_location="cpu", weights_only=True)
    weights = {k: _as_mx(v.detach().cpu().float().numpy()) for k, v in raw.items()}
    del raw
    model = VonOptionMarker(config)
    weights = sanitize(weights, model, "option_marker")
    model.load_weights(list(weights.items()))
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_dir), config


def load(model_dir: str, head: str = "option_marker", quantized: Optional[bool] = None):
    """Load ``head`` ('nli' or 'option_marker') from ``model_dir``."""
    if head not in MODEL_CLASSES:
        raise ValueError(f"unknown head {head!r}; expected one of {list(MODEL_CLASSES)}")
    if head == "nli":
        return load_nli(model_dir, quantized=quantized)
    return load_option_marker(model_dir, quantized=quantized)
