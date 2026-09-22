"""PyTorch reference pieces used only by the verification scripts.

These mirror the von SDK's PyTorch definitions exactly so the MLX port can be
compared against the real thing rather than against itself.
"""

import torch
import torch.nn as nn


class TorchOptionMarkerScorer(nn.Module):
    """Exact copy of ``von.models.option_marker.OptionMarkerScorer``."""

    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.input_norm(x)
        h = self.act(self.dense(x))
        h = self.norm(h)
        h = self.dropout(h)
        return self.out_proj(h).squeeze(-1)
