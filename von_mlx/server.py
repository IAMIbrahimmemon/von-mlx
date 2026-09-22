"""OpenAI-compatible + TypeSafe-compatible HTTP server for the MLX Von port.

Endpoints
---------
``GET  /v1/models``
``POST /v1/chat/completions``   -- OpenAI schema; the decision problem travels in
                                   the message content as JSON, or use
                                   ``response_format.type = "json_object"``.
``POST /v1/systemone``          -- TypeSafe Von wire protocol (authoritative).

The OpenAI shim exists purely so Hermes/Claude/any OpenAI client can call the
model without a bespoke integration; ``/v1/systemone`` is the lossless path.
"""

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .types import SystemOneResponse
from .worker import InferenceWorker

PROBLEM_SCHEMA = """{
  "state": "free text or JSON object describing the situation",
  "questions": {
    "<name>": {"type": "choice", "instructions": "...", "criteria": {"key": "description", ...}},
    "<name>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "<name>": {"type": "score",  "instructions": "...", "criteria": ["low", "medium", "high"]}
  }
}"""

SYSTEM_PROMPT = (
    "You are Von, a non-autoregressive decision model. You do not write prose. "
    "Given a JSON decision problem you return only a JSON object of calibrated "
    "answers. The request schema is:\n" + PROBLEM_SCHEMA
)


def create_app(model_dir: str, temperature: Optional[float] = None) -> FastAPI:
    app = FastAPI(title="von-mlx", version="1.0.0")
    worker = InferenceWorker(model_dir, temperature=temperature)
    engine = worker.engine
    model_id = "von-1.0-mlx"

    @app.on_event("shutdown")
    def _shutdown():
        worker.shutdown()

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "meta": {
                        "backend": "mlx",
                        "head": "option_marker",
                        "params": 395834371,
                        "temperature": engine._default_temp,
                    },
                }
            ],
        }

    @app.post("/v1/systemone")
    def systemone(payload: Dict[str, Any]):
        try:
            state = payload["state"]
            questions = payload["questions"]
        except (KeyError, TypeError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            resp: SystemOneResponse = worker.run(
                engine.evaluate, state, questions,
                payload.get("model", "von-option-marker"),
            )
        except (ValueError, TypeError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return json.loads(resp.model_dump_json())

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = model_id
        messages: List[ChatMessage]
        temperature: Optional[float] = None
        response_format: Optional[Dict[str, Any]] = None
        max_tokens: Optional[int] = None

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        user_text = "\n".join(
            m.content for m in req.messages if m.role == "user"
        )
        try:
            problem = json.loads(user_text)
            state = problem["state"]
            questions = problem["questions"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Expected a JSON decision problem in the user "
                                   f"message. Schema: {PROBLEM_SCHEMA}. Got: {exc}",
                        "type": "invalid_request_error",
                    }
                },
            )

        try:
            resp = worker.run(engine.evaluate, state, questions)
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc),
                                   "type": "invalid_request_error"}},
            )
        content = resp.model_dump_json(indent=2)
        now = int(time.time())
        return {
            "id": f"chatcmpl-vonmlx-{now}",
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            },
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_dir": model_dir,
            "temperature": engine._default_temp,
        }

    return app
