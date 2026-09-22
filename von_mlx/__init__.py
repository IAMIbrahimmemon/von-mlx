"""von-mlx: MLX (Apple Silicon) port of wfzyx/von-1.0.

Von is a non-autoregressive "System One" decision model: a bidirectional
ModernBERT-Large encoder with two interchangeable decision heads.

  * NLI cross-encoder head  (``model.safetensors``)  -- 3-way entailment logits
  * Option-Marker head      (``option_marker.pt``)   -- single-pass joint scorer

This package provides both, running entirely on Apple Silicon via MLX, plus a
drop-in ``/v1/systemone`` server.
"""

from .config import ModelArgs
from .decision import VonEngine
from .load import load, load_nli, load_option_marker
from .model import VonNLI, VonOptionMarker
from .worker import InferenceWorker

__all__ = [
    "ModelArgs",
    "VonEngine",
    "InferenceWorker",
    "VonNLI",
    "VonOptionMarker",
    "load",
    "load_nli",
    "load_option_marker",
]

__version__ = "1.0.0"
