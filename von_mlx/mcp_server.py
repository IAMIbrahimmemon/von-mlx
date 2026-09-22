"""MCP server exposing Von's decision primitives as tools.

An MCP host (Hermes, Claude Desktop, etc.) calling a decision model through an
OpenAI chat shim is a bad fit: the shim returns a JSON answer object and the
host would treat it as generated prose. This server exposes the primitives
directly instead, so the model becomes a *tool* the host can reason about.

Tools
-----
``decide``      pick one of K options, with a calibrated probability distribution
``judge``       probability that a condition holds
``rate``        expectation over an ordered scale
``system_one``  several heterogeneous questions in one pass
``model_info``  which model/head/temperature is loaded

Run:
    von-mlx-mcp --model-dir ~/models/von-1.0-mlx/8bit

Notes for maintainers
---------------------
* Built on ``mcp.server.mcpserver.MCPServer`` (the current API). The older
  lowlevel ``Server.list_tools()``/``call_tool()`` decorators were **removed in
  mcp 2.x** -- registering them raises ``AttributeError`` at startup, after
  which the process dies and every client sees "Connection closed" during
  ``initialize`` with no useful message. If you hit that, check the server's
  stderr before suspecting the client.
* All MLX work goes through :class:`~von_mlx.worker.InferenceWorker`, because MLX
  streams are thread-local: running a forward pass on an arbitrary thread fails
  with ``RuntimeError: There is no Stream(cpu, 0) in current thread``.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from mcp.server.mcpserver import MCPServer

from .types import Choice, Noul, Score
from .worker import InferenceWorker

DEFAULT_MODEL_DIR = os.environ.get(
    "VON_MLX_MODEL", os.path.expanduser("~/models/von-1.0-mlx/8bit")
)

_MODEL_DIR = DEFAULT_MODEL_DIR
_WORKER: Optional[InferenceWorker] = None

server = MCPServer(
    name="von-mlx",
    title="Von (MLX) decision model",
    instructions=(
        "Calibrated non-autoregressive decision primitives. These tools return "
        "probabilities and choices, never prose. Give descriptive criteria for "
        "every option -- the model matches your state text against the criteria "
        "you supply, so richer criteria measurably improve accuracy."
    ),
)


def _worker() -> InferenceWorker:
    """Load lazily so the MCP handshake is never blocked by a 420 MB read."""
    global _WORKER
    if _WORKER is None:
        _WORKER = InferenceWorker(_MODEL_DIR)
    return _WORKER


def _state(value: Union[str, Dict[str, Any]]) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(f"{k}: {v}" for k, v in value.items())


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@server.tool(
    description=(
        "Make a discrete calibrated decision: choose one option from a set, "
        "returning a probability distribution and a confidence margin. Use for "
        "routing, classification and triage."
    )
)
def decide(
    state: str,
    instructions: str,
    choices: Dict[str, str],
) -> str:
    """Pick one of ``choices`` for ``state``.

    Args:
        state: The situation to judge -- free text, or a JSON object as a string.
        instructions: The question, e.g. 'Classify the root cause domain.'
        choices: Map of option key -> description of what that option means.
            Descriptions materially improve accuracy; prefer them over bare labels.
    """
    engine = _worker().engine
    q = Choice(instructions=instructions, criteria=choices)
    ans = _worker().run(engine.evaluate_choice, "d", state, q)
    return _dump(ans.model_dump())


@server.tool(
    description=(
        "Return the calibrated probability (0..1) that a condition holds. Use for "
        "yes/no gating, guardrails and policy checks. Prefer supplying explicit "
        "true_criteria/false_criteria so the model does not have to infer polarity."
    )
)
def judge(
    state: str,
    instructions: str,
    true_criteria: Optional[str] = None,
    false_criteria: Optional[str] = None,
) -> str:
    """Probability that ``instructions`` holds for ``state``.

    Args:
        state: The situation to judge.
        instructions: The condition, e.g. 'Is this blocking customers?'
        true_criteria: What the state looks like when the condition IS true.
        false_criteria: What the state looks like when the condition is FALSE.
    """
    engine = _worker().engine
    criteria = None
    if true_criteria or false_criteria:
        criteria = {"true": true_criteria or "", "false": false_criteria or ""}
    q = Noul(instructions=instructions, criteria=criteria)
    ans = _worker().run(engine.evaluate_noul, "j", state, q)
    return _dump(
        {
            "probability": ans.noul,
            "verdict": ans.noul >= 0.5,
            "note": (
                "the probability is calibrated; threshold it against your own "
                "decision cost rather than trusting the 0.5 default"
            ),
        }
    )


@server.tool(
    description=(
        "Rate the state on an ordered scale, returning a continuous expected value "
        "in [0, K-1] plus the per-level distribution. Use for severity and quality "
        "scoring."
    )
)
def rate(state: str, instructions: str, levels: List[str]) -> str:
    """Score ``state`` over ordered ``levels`` (lowest first).

    Args:
        state: The situation to rate.
        instructions: What is being rated, e.g. 'Rate the incident severity.'
        levels: Ordered levels from lowest to highest, at least 2.
    """
    if len(levels) < 2:
        return _dump({"error": "levels must contain at least 2 entries"})
    engine = _worker().engine
    q = Score(instructions=instructions, criteria=list(levels))
    ans = _worker().run(engine.evaluate_score, "r", state, q)
    return _dump(ans.model_dump())


@server.tool(
    description=(
        "Evaluate several heterogeneous questions over one state in a single "
        "forward pass. Cheaper than calling decide/judge/rate separately, and the "
        "right tool when you have 3+ questions about the same text."
    )
)
def system_one(state: str, questions: Dict[str, Dict[str, Any]]) -> str:
    """Answer several questions about ``state`` at once.

    Args:
        state: The shared situation, as free text or a JSON object string.
        questions: Map of answer-name -> question. Each question is one of
            ``{"type":"choice","instructions":str,"criteria":{key:desc}}``,
            ``{"type":"noul","instructions":str,"criteria":{"true":str,"false":str}}``,
            or ``{"type":"score","instructions":str,"criteria":[levels]}``.
    """
    engine = _worker().engine
    resp = _worker().run(engine.evaluate, state, questions)
    return json.dumps(json.loads(resp.model_dump_json()), ensure_ascii=False, indent=2)


@server.tool(description="Report which Von model, head and calibration temperature is loaded.")
def model_info() -> str:
    """Return the loaded model's identity and calibration temperature."""
    engine = _worker().engine
    return _dump(
        {
            "model_dir": _MODEL_DIR,
            "head": "option_marker",
            "temperature": engine._default_temp,
            "backend": "mlx",
            "params": 395834371,
            "architecture": "ModernBERT-Large, 28 layers",
        }
    )


def main() -> None:
    global _MODEL_DIR

    import argparse

    ap = argparse.ArgumentParser(prog="von-mlx-mcp")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--temperature", type=float, default=None)
    opts = ap.parse_args()
    _MODEL_DIR = opts.model_dir
    if opts.temperature is not None:
        os.environ["VON_MLX_TEMPERATURE"] = str(opts.temperature)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
