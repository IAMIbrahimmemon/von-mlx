"""Option-Marker Joint Decision Model for ModernBERT.

Enables single-pass non-autoregressive decision evaluation:
Pack premise and K options into a single sequence marked by [MASK] tokens.
All options attend to the state and each other simultaneously via full
bidirectional self-attention, eliminating K separate cross-encoder passes.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class OptionMarkerScorer(nn.Module):
    """Calibrated MLP scoring head for option-marker representations."""

    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects (N_options, hidden_size) representations to scalar logits."""
        x = self.input_norm(x)
        h = self.dense(x)
        h = self.act(h)
        h = self.norm(h)
        h = self.dropout(h)
        return self.out_proj(h).squeeze(-1)


class OptionMarkerModel(nn.Module):
    """ModernBERT decision model with single-pass option-marker scoring."""

    def __init__(
        self,
        base_model_id: str = "checkpoints/von-modernbert-rlcd",
        max_position_embeddings: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_id)
        config.max_position_embeddings = max_position_embeddings
        self.encoder = AutoModel.from_pretrained(base_model_id, config=config)
        self.hidden_size = self.encoder.config.hidden_size
        self.scorer = OptionMarkerScorer(hidden_size=self.hidden_size, dropout=dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, model_max_length=max_position_embeddings)
        self.mask_token_id = self.tokenizer.mask_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: List[List[int]],
    ) -> List[torch.Tensor]:
        """Runs single forward pass and returns list of option logits per sample."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, seq_len, H)

        batch_logits = []
        for b, pos_list in enumerate(mask_positions):
            opt_reps = last_hidden[b, pos_list]  # (K, H)
            logits = self.scorer(opt_reps)  # (K,)
            batch_logits.append(logits)

        return batch_logits

    def pack_sequence(
        self,
        state: str,
        question: str,
        options: List[str],
    ) -> Tuple[str, str]:
        """Packs state, question, and candidate options into an option-marker string."""
        mask = self.tokenizer.mask_token
        sep = self.tokenizer.sep_token
        prefix = f"{question} {state}".strip() if question else state.strip()
        opts_packed = " ".join(f"{mask} {opt.strip()}" for opt in options)
        return f"{prefix} {sep} {opts_packed}"
