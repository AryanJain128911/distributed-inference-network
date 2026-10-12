"""Real local inference; MLX objects stay on one dedicated worker thread."""

import asyncio
import importlib
import platform
import threading
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import aclosing
from typing import Any, TypeVar

from inference_network.backend import BackendChunk, BackendResult
from inference_network.config import MLX_MODEL_ID, MLX_MODEL_REVISION, ProviderSettings
from inference_network.errors import NetworkError
from inference_network.schemas import ChatRequest, TokenUsage

Result = TypeVar("Result")
STREAM_BUFFER_CHUNKS = 8


class _GenerationCancelled(Exception):
    """Internal cooperative stop signal on the model worker."""


class MLXBackend:
    def __init__(self, settings: ProviderSettings) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-inference")
        # The optional, untyped MLX/tokenizer APIs are contained at this boundary.
        self._mx: Any = None
        self._lm: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._sampler: Any = None
        self._loaded = False
        self._closed = False
        self._busy = False
        self.device_name = ""
        self.memory_gb = 0.0

    async def _run(
        self, operation: Callable[[], Result], stop: threading.Event | None = None
    ) -> Result:
        future = asyncio.get_running_loop().run_in_executor(self._executor, operation)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if stop is not None:
                stop.set()
            # A cancelled asyncio waiter does not stop native GPU/thread work.
            # Collect it before the provider releases its execution reservation.
            collected = asyncio.gather(future, return_exceptions=True)
            while not collected.done():
                try:
                    await asyncio.shield(collected)
                except asyncio.CancelledError:
                    # Disconnect and deadline can both cancel the same request.
                    # Neither may release capacity while its worker still runs.
                    continue
            raise

    async def load(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot load a closed MLX backend")
        if self._loaded:
            return
        try:
            await self._run(self._load)
            self._loaded = True
        except Exception as exc:
            raise RuntimeError(
                "MLX model loading failed. Install the mlx extra and download the pinned "
                "model snapshot; see docs/setup.md."
            ) from exc

    def _load(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This backend requires native Apple Silicon macOS")
        self._mx = importlib.import_module("mlx.core")
        if not self._mx.metal.is_available():
            raise RuntimeError("Apple Metal GPU is unavailable")
        self._mx.set_default_device(self._mx.gpu)
        self._lm = importlib.import_module("mlx_lm")
        hub = importlib.import_module("huggingface_hub")
        # Serving never downloads a model or executes code from a model repository.
        snapshot = hub.snapshot_download(
            MLX_MODEL_ID, revision=MLX_MODEL_REVISION, local_files_only=True, token=False
        )
        self._model, self._tokenizer = self._lm.load(
            snapshot,
            lazy=False,
            trust_remote_code=False,
            tokenizer_config={"trust_remote_code": False},
        )
        self._sampler = importlib.import_module("mlx_lm.sample_utils").make_sampler(temp=0)
        device = self._mx.device_info()
        self.device_name = str(device["device_name"])
        self.memory_gb = int(device["memory_size"]) / 1024**3

    async def generate(self, request: ChatRequest) -> BackendResult:
        content: list[str] = []
        terminal: BackendChunk | None = None
        async with aclosing(self.stream(request)) as stream:
            async for chunk in stream:
                content.append(chunk.content)
                if chunk.finish_reason is not None:
                    terminal = chunk
        assert terminal is not None and terminal.finish_reason is not None
        return BackendResult(
            content="".join(content),
            is_simulated=False,
            usage=terminal.usage,
            finish_reason=terminal.finish_reason,
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[BackendChunk, None]:
        if not self._loaded or self._closed:
            raise NetworkError(503, "model_not_loaded", "The local model is not loaded")
        if request.model != MLX_MODEL_ID:
            raise NetworkError(
                404, "unsupported_model", "The local model does not match the request"
            )
        if self._busy:
            raise NetworkError(429, "backend_capacity", "The local model is already generating")
        self._busy = True
        stop = threading.Event()
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[BackendChunk] = asyncio.Queue(maxsize=STREAM_BUFFER_CHUNKS)

        def emit(chunk: BackendChunk) -> None:
            delivery = asyncio.run_coroutine_threadsafe(chunks.put(chunk), loop)
            try:
                while True:
                    if stop.is_set():
                        raise _GenerationCancelled()
                    try:
                        delivery.result(timeout=0.05)
                        return
                    except FutureTimeout:
                        # A slow consumer bounds generation without trapping shutdown
                        # behind a full queue after the consumer has disconnected.
                        continue
            finally:
                if not delivery.done():
                    delivery.cancel()

        worker = asyncio.create_task(self._run(lambda: self._generate(request, stop, emit), stop))
        pending: asyncio.Task[BackendChunk] | None = None
        try:
            while True:
                pending = asyncio.create_task(chunks.get())
                await asyncio.wait((pending, worker), return_when=asyncio.FIRST_COMPLETED)
                # Inspect current state, not asyncio.wait's snapshot: queue.get
                # can finish just after the worker wakes that wait.
                if pending.done():
                    chunk = pending.result()
                    pending = None
                    yield chunk
                if worker.done():
                    if pending is not None:
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
                        pending = None
                    terminal = await worker
                    while not chunks.empty():
                        yield chunks.get_nowait()
                    yield terminal
                    return
        finally:
            stop.set()
            if pending is not None:
                pending.cancel()
            if not worker.done():
                worker.cancel()
            # Iterator closure can arrive while suspended at yield, independently
            # of cancellation inside _run. Own and collect both tasks here as well.
            collected = asyncio.gather(
                *([pending] if pending is not None else []), worker, return_exceptions=True
            )
            cancelled = False
            while not collected.done():
                try:
                    await asyncio.shield(collected)
                except asyncio.CancelledError:
                    cancelled = True
                    continue
            self._busy = False
            if cancelled:
                raise asyncio.CancelledError

    def _generate(
        self,
        request: ChatRequest,
        stop: threading.Event,
        emit: Callable[[BackendChunk], None],
    ) -> BackendChunk:
        def check_cancelled(*_: object) -> None:
            if stop.is_set():
                raise _GenerationCancelled()

        check_cancelled()
        prompt = self._tokenizer.apply_chat_template(
            [message.model_dump() for message in request.messages],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(prompt) > self._settings.max_prompt_tokens:
            raise NetworkError(
                422, "prompt_too_long", "Tokenized prompt exceeds the provider limit"
            )
        character_count = 0
        has_text = False
        last: Any = None
        generator = self._lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._settings.max_output_tokens,
            sampler=self._sampler,
            prefill_step_size=256,
            prompt_progress_callback=check_cancelled,
        )
        try:
            for response in generator:
                check_cancelled()
                character_count += len(response.text)
                if character_count > 8192:
                    raise NetworkError(
                        502, "model_output_limit", "Model output exceeded the text limit"
                    )
                has_text = has_text or bool(response.text.strip())
                if response.text:
                    emit(BackendChunk(content=response.text))
                last = response
        finally:
            generator.close()
            self._mx.synchronize()
            self._mx.clear_cache()
        if last is None or not has_text:
            raise NetworkError(502, "empty_model_output", "Local model generated no text")
        if last.finish_reason not in {"stop", "length"}:
            raise NetworkError(
                502, "invalid_model_result", "Local model returned an invalid finish reason"
            )
        return BackendChunk(
            is_simulated=False,
            usage=TokenUsage(
                prompt_tokens=last.prompt_tokens,
                completion_tokens=last.generation_tokens,
                total_tokens=last.prompt_tokens + last.generation_tokens,
            ),
            finish_reason=last.finish_reason,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._run(self._unload)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._loaded = False

    def _unload(self) -> None:
        if self._mx is not None:
            self._mx.synchronize()
        self._model = None
        self._tokenizer = None
        if self._mx is not None:
            self._mx.clear_cache()
