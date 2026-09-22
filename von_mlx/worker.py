"""Thread-pinned MLX inference worker, shared by the HTTP and MCP servers.

MLX streams are **thread-local** and created lazily. As soon as inference runs
on a thread that did not create the model, a quantized ``nn.Embedding`` gather
fails with::

    RuntimeError: There is no Stream(cpu, 0) in current thread.

Both FastAPI's threadpool and ``asyncio.to_thread`` schedule onto arbitrary
threads, so neither can be used directly. Loading the model *and* running every
forward pass on one owned thread makes the stream stable and also serializes
GPU access, which is what a single local model wants anyway.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import mlx.core as mx

from .decision import VonEngine


class InferenceWorker:
    def __init__(self, model_dir: str, temperature: Optional[float] = None,
                 warmup: bool = True):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="von-mlx")
        self._model_dir = model_dir
        self._temperature = temperature
        self._warmup = warmup
        self.engine: VonEngine = self._pool.submit(self._init).result()

    def _init(self) -> VonEngine:
        # Establish a default stream for THIS thread before any MLX op runs.
        mx.set_default_stream(mx.new_stream(mx.default_device()))
        engine = VonEngine(self._model_dir, temperature=self._temperature)
        if self._warmup:
            # Compile the graph shapes on this thread so the first real request
            # does not pay the JIT cost.
            engine.evaluate(
                state="warmup",
                questions={"q": {"type": "noul", "instructions": "Is this a warmup?"}},
            )
        return engine

    def run(self, fn: Callable, *args: Any, **kwargs: Any):
        """Run ``fn`` on the owned thread and block until it returns."""
        return self._pool.submit(fn, *args, **kwargs).result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
