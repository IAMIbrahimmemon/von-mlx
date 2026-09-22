"""Convert wfzyx/von-1.0 (PyTorch) to MLX checkpoints.

Output layout -- each variant directory is self-contained and loadable:

    <out>/fp16/   config.json tokenizer* calibration* model.safetensors
                  option_marker.safetensors
    <out>/4bit/   same files, with affine-quantized weight tensors

Weight names are preserved 1:1 from the source checkpoints, so nothing is
translated at load time and any mismatch is a hard error rather than silently
random weights.

Quantization note: MLX quantizes in place. The correct order is build fp16
module -> load fp16 weights -> ``nn.quantize`` -> save. Building a quantized
module first and then loading fp16 weights fails, because a quantized Linear
needs ``.scales``/``.biases`` that the fp16 dict does not contain.

Usage:
    env -u PYTHONPATH .venv/bin/python convert.py --src hf_orig --out out/von-1.0-mlx
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from von_mlx.config import ModelArgs  # noqa: E402
from von_mlx.load import (  # noqa: E402
    MARKER_WEIGHTS_LEGACY,
    NLI_WEIGHTS,
    QUANT_BITS,
    QUANT_GROUP_SIZE,
    _linearize,
    quant_predicate,
)
from von_mlx.model import VonNLI, VonOptionMarker  # noqa: E402

SHARED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "calibration.json",
    "marker_calibration.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def flat_arrays(tree: Any, prefix: str = "") -> Dict[str, mx.array]:
    """Flatten a nested MLX parameter tree to {dotted.name: array}."""
    out: Dict[str, mx.array] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(flat_arrays(value, f"{prefix}{key}."))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            out.update(flat_arrays(value, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = tree
    return out


def load_source_weights(src: str, head: str) -> Dict[str, mx.array]:
    """Read the original PyTorch checkpoints into fp16 MLX arrays."""
    if head == "nli":
        from safetensors import safe_open

        weights: Dict[str, mx.array] = {}
        with safe_open(os.path.join(src, NLI_WEIGHTS), "pt") as fh:
            for key in fh.keys():
                weights[key] = mx.array(fh.get_tensor(key).float().numpy()).astype(
                    mx.float16
                )
        return weights

    legacy = os.path.join(src, MARKER_WEIGHTS_LEGACY)
    if not os.path.exists(legacy):
        raise FileNotFoundError(f"missing {legacy}")
    import torch

    raw = torch.load(legacy, map_location="cpu", weights_only=True)
    weights = {
        k: mx.array(v.detach().cpu().float().numpy()).astype(mx.float16)
        for k, v in raw.items()
    }
    del raw
    return weights


def convert_head(cls, src: str, out_dir: str, config: ModelArgs, head: str,
                 bits: int | None) -> Dict[str, Any]:
    weights = load_source_weights(src, head)

    model = cls(config)
    expected = set(_linearize(model.parameters()))
    got = set(weights)
    if got != expected:
        raise ValueError(
            f"{head}: checkpoint/port key mismatch\n"
            f"  missing ({len(expected - got)}): {sorted(expected - got)[:6]}\n"
            f"  extra   ({len(got - expected)}): {sorted(got - expected)[:6]}"
        )

    model.load_weights(list(weights.items()))
    model.eval()

    if bits:
        nn.quantize(
            model,
            group_size=QUANT_GROUP_SIZE,
            bits=bits,
            class_predicate=quant_predicate,
        )
        model.eval()

    name = "model.safetensors" if head == "nli" else "option_marker.safetensors"
    path = os.path.join(out_dir, name)
    flat = flat_arrays(model.parameters())
    mx.save_safetensors(path, flat)

    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "tensors": len(flat),
        "quantized_linears": sum(1 for k in flat if k.endswith(".scales")),
        "dtype": str(next(iter(flat.values())).dtype),
    }


def copy_shared(src: str, dst: str, bits: int | None) -> None:
    for name in SHARED_FILES:
        srcp = os.path.join(src, name)
        if os.path.exists(srcp):
            shutil.copy2(srcp, os.path.join(dst, name))

    # Stamp the quantization into config.json so the checkpoint is
    # self-describing and the loader never has to guess a bit width.
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if bits:
        cfg["quantization"] = {"group_size": QUANT_GROUP_SIZE, "bits": bits}
        cfg["quantization_config"] = {
            "quant_method": "mlx",
            "group_size": QUANT_GROUP_SIZE,
            "bits": bits,
            "mode": "affine",
        }
    else:
        cfg.pop("quantization", None)
        cfg.pop("quantization_config", None)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hf_orig")
    ap.add_argument("--out", default="out/von-1.0-mlx")
    args = ap.parse_args()

    config = ModelArgs.from_json(os.path.join(args.src, "config.json"))
    summary: Dict[str, Any] = {}

    # 8 bits is the smallest width that kept every decision intact in
    # verify/bits_sweep.py; 5- and 4-bit both flipped an argmax and widened the
    # calibrated probability spread by up to 0.38. See README "Verification".
    for variant, bits in (("fp16", None), ("8bit", 8)):
        out_dir = os.path.join(args.out, variant)
        os.makedirs(out_dir, exist_ok=True)
        copy_shared(args.src, out_dir, bits)
        print(f"[{variant}] converting...")
        nli = convert_head(VonNLI, args.src, out_dir, config, "nli", bits)
        print(f"  nli           {nli['bytes']/1e6:>9.2f} MB  "
              f"({nli['quantized_linears']} quantized linears)")
        marker = convert_head(
            VonOptionMarker, args.src, out_dir, config, "option_marker", bits
        )
        print(f"  option_marker {marker['bytes']/1e6:>9.2f} MB  "
              f"({marker['quantized_linears']} quantized linears)")
        summary[variant] = {"nli": nli, "option_marker": marker}

    manifest = {
        "port": "von-mlx",
        "source_model": "wfzyx/von-1.0",
        "architecture": "ModernBertForSequenceClassification (395M, 28 layers)",
        "heads": {
            "nli": {
                "file": "model.safetensors",
                "purpose": "3-way NLI cross-encoder (one forward pass per option)",
                "temperature": 1.1692,
                "val_accuracy": 0.9643,
            },
            "option_marker": {
                "file": "option_marker.safetensors",
                "purpose": "single-pass joint scorer over [MASK] markers",
                "temperature": 2.2,
                "val_accuracy": 0.9866,
                "recommended": True,
            },
        },
        "quantization": {
            "mode": "affine",
            "group_size": QUANT_GROUP_SIZE,
            "bits": 8,
            "fp16_paths": ["classifier", "scorer", "head"],
            "measured": {
                "8bit": {"argmax_flips": "0/6", "worst_prob_shift": 0.041},
                "4bit": {"argmax_flips": "1/6", "worst_prob_shift": 0.382,
                         "verdict": "not recommended - loses a decision"},
            },
        },
        "variants": summary,
    }
    with open(os.path.join(args.out, "von_mlx_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nsummary")
    for variant, heads in summary.items():
        for head, s in heads.items():
            print(f"  {variant:<5} {head:<14} {s['bytes']/1e6:>9.2f} MB  {s['dtype']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
