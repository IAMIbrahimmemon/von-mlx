"""MLX decision engine -- a faithful port of the von SDK's OptionMarkerBackend.

Kept deliberately line-for-line with ``von/backends/option_marker_backend.py``
so that outputs match the reference implementation, including the zero-shot
"Noul" polarity-prior cancellation and the two calibrated temperatures
(``option_marker.pt`` uses T=2.2, ``model.safetensors`` uses T=1.1692).
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx

from .load import load_option_marker
from .types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)


def format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return str(state)


class VonEngine:
    """Single-pass non-autoregressive decision engine on Apple Silicon."""

    def __init__(
        self,
        model_dir: str,
        temperature: Optional[float] = None,
        default_temp: float = 1.0,
    ):
        self.model_dir = model_dir
        self.model, self.tokenizer, self.config = load_option_marker(model_dir)

        calib = os.path.join(model_dir, "marker_calibration.json")
        if temperature is not None:
            self._default_temp = float(temperature)
        elif os.path.exists(calib):
            with open(calib, "r", encoding="utf-8") as fh:
                self._default_temp = float(json.load(fh).get("temperature", default_temp))
        else:
            self._default_temp = default_temp

        self.mask_token_id = self.tokenizer.mask_token_id

    # ------------------------------------------------------------------
    # packing
    # ------------------------------------------------------------------

    def pack_sequence(self, state: str, question: str, options: List[str]) -> str:
        mask = self.tokenizer.mask_token
        sep = self.tokenizer.sep_token
        prefix = f"{question} {state}".strip() if question else state.strip()
        packed = " ".join(f"{mask} {opt.strip()}" for opt in options)
        return f"{prefix} {sep} {packed}"

    def _forward(self, text: str) -> mx.array:
        """Tokenize, locate [MASK] markers, run one forward pass, return logits."""
        enc = self.tokenizer(text, return_tensors="np")
        ids = mx.array(enc["input_ids"].astype("int32"))
        am = mx.array(enc["attention_mask"].astype("int32"))
        positions = [
            int(p) for p in (enc["input_ids"][0] == self.mask_token_id).nonzero()[0]
        ]
        return self.model(ids, am, [positions])[0]

    # ------------------------------------------------------------------
    # primitives
    # ------------------------------------------------------------------

    def evaluate_choice(
        self,
        q_id: str,
        state_text: str,
        q: Choice,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChoiceAnswer:
        options = list(q.criteria.keys())
        if not options:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        eff_temp = self._default_temp if temperature is None else temperature
        descriptions = [
            (q.criteria.get(opt) or opt).strip() for opt in options
        ]
        logits = self._forward(self.pack_sequence(state_text, q.instructions, descriptions))

        scaled = logits / max(eff_temp, 1e-4)
        probs = mx.softmax(scaled)
        mx.eval(probs)

        probs_list = [float(p) for p in probs]
        best_idx = int(mx.argmax(logits).item())
        prob_dict = {opt: round(p, 4) for opt, p in zip(options, probs_list)}
        sorted_p = sorted(probs_list, reverse=True)
        conf = round(
            max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))),
            3,
        )
        return ChoiceAnswer(
            choice=options[best_idx], probabilities=prob_dict, confidence=conf
        )

    def evaluate_noul(
        self,
        q_id: str,
        state_text: str,
        q: Noul,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> NoulAnswer:
        eff_temp = self._default_temp if temperature is None else temperature

        crit = q.criteria or {}
        pos_desc = crit.get("true")
        neg_desc = crit.get("false")
        has_explicit = bool(pos_desc or neg_desc)
        if not pos_desc:
            pos_desc = "Yes, condition holds true."
        if not neg_desc:
            neg_desc = "No, condition is false."

        descriptions = [pos_desc, neg_desc]
        logits = self._forward(self.pack_sequence(state_text, q.instructions, descriptions))

        if not has_explicit:
            # Cancel out the intrinsic negative-polarity prior in zero-shot mode.
            null_logits = self._forward(
                self.pack_sequence("", q.instructions, descriptions)
            )
            bias = null_logits[0] - null_logits[1]
            logits = mx.stack([logits[0] - 0.7 * bias, logits[1]])

        scaled = logits / max(eff_temp, 1e-4)
        probs = mx.softmax(scaled)
        mx.eval(probs)
        prob_true = round(max(0.0, min(1.0, float(probs[0]))), 4)
        return NoulAnswer(noul=prob_true)

    def evaluate_score(
        self,
        q_id: str,
        state_text: str,
        q: Score,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ScoreAnswer:
        levels = q.criteria
        if not levels:
            return ScoreAnswer(score=0.0, confidence=0.0, legend={}, probabilities={})

        eff_temp = self._default_temp if temperature is None else temperature

        legend: Dict[str, str] = {}
        descriptions: List[str] = []
        for i, item in enumerate(levels):
            if isinstance(item, dict):
                what = item.get("what", "")
                examples = item.get("examples", [])
                ex_str = f" Examples: {', '.join(examples)}" if examples else ""
                desc = f"{what}{ex_str}".strip()
            else:
                desc = str(item).strip()
            legend[str(i)] = desc
            descriptions.append(desc)

        logits = self._forward(
            self.pack_sequence(state_text, q.instructions, descriptions)
        )
        scaled = logits / max(eff_temp, 1e-4)
        probs = mx.softmax(scaled)
        mx.eval(probs)
        probs_list = [float(p) for p in probs]

        prob_dict = {str(i): round(p, 4) for i, p in enumerate(probs_list)}
        weighted = round(sum(i * p for i, p in enumerate(probs_list)), 2)
        sorted_p = sorted(probs_list, reverse=True)
        conf = round(
            max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))),
            3,
        )
        return ScoreAnswer(
            score=weighted, confidence=conf, legend=legend, probabilities=prob_dict
        )

    # ------------------------------------------------------------------

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = "von-option-marker",
    ) -> SystemOneResponse:
        state_str = format_state(state)
        answers: Dict[str, Any] = {}
        total_q_chars = 0

        for q_id, q_data in questions.items():
            if isinstance(q_data, dict):
                q_type = q_data.get("type", "choice")
                if q_type == "choice":
                    q_obj = Choice(**q_data)
                elif q_type == "noul":
                    q_obj = Noul(**q_data)
                elif q_type == "score":
                    q_obj = Score(**q_data)
                else:
                    raise ValueError(f"Unknown question type '{q_type}'")
            else:
                q_obj = q_data

            if isinstance(q_obj, Choice):
                answers[q_id] = self.evaluate_choice(q_id, state_str, q_obj)
            elif isinstance(q_obj, Noul):
                answers[q_id] = self.evaluate_noul(q_id, state_str, q_obj)
            elif isinstance(q_obj, Score):
                answers[q_id] = self.evaluate_score(q_id, state_str, q_obj)
            else:
                raise ValueError(f"Unknown question object {q_obj!r}")
            total_q_chars += len(getattr(q_obj, "instructions", "") or "")

        return SystemOneResponse(
            model="von-option-marker",
            answers=answers,
            usage=Usage(
                input_tokens=max(1, len(state_str) // 4) + max(1, total_q_chars // 4),
                output_tokens=len(answers),
            ),
        )
