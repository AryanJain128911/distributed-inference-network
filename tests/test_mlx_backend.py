"""Adapter contracts with fake vendor modules; these tests do not run a model."""

import asyncio
import sys
import threading
import weakref
from contextlib import aclosing
from types import SimpleNamespace

import pytest

from inference_network.backend import SimulatedEchoBackend, StreamingBackend
from inference_network.config import MLX_MODEL_ID, MLX_MODEL_REVISION, ProviderSettings
from inference_network.errors import NetworkError
from inference_network.mlx_backend import MLXBackend
from inference_network.schemas import ChatRequest


class FakeModel:
    pass


class FakeVendor:
    """Controllable vendor boundary, including a blocking native generation step."""

    def __init__(self):
        self.calls = []
        self.threads = []
        self.prompt = [1, 2, 3]
        self.responses = [self.response("Generated ", None), self.response("text.", "stop")]
        self.load_error = None
        self.snapshot_error = None
        self.sampler_error = None
        self.generation_error = None
        self.block_at = None
        self.block_after = None
        self.generated = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.model_reference = None
        self.tokenizer = SimpleNamespace(apply_chat_template=self.chat_template)
        self.mx = SimpleNamespace(
            metal=SimpleNamespace(is_available=lambda: True),
            gpu="fake-metal-device",
            set_default_device=lambda device: self.record("device", device),
            device_info=lambda: {"device_name": "Apple M4 (fake)", "memory_size": 16 * 1024**3},
            synchronize=lambda: self.record("synchronize"),
            clear_cache=lambda: self.record("clear_cache"),
        )

    @staticmethod
    def response(text, finish_reason):
        return SimpleNamespace(
            text=text, finish_reason=finish_reason, prompt_tokens=3, generation_tokens=2
        )

    def record(self, operation, *args, **kwargs):
        self.calls.append((operation, args, kwargs))
        self.threads.append(threading.current_thread())

    def snapshot_download(self, *args, **kwargs):
        self.record("snapshot", *args, **kwargs)
        if self.snapshot_error:
            raise self.snapshot_error
        return "/fake/cached/snapshot"

    def load(self, *args, **kwargs):
        self.record("load", *args, **kwargs)
        if self.load_error:
            raise self.load_error
        if self.block_at == "load":
            self.started.set()
            if not self.release.wait(timeout=3):
                raise TimeoutError("Test did not release model loading")
        model = FakeModel()
        self.model_reference = weakref.ref(model)
        return model, self.tokenizer

    def make_sampler(self, **kwargs):
        self.record("sampler", **kwargs)
        if self.sampler_error:
            raise self.sampler_error
        return "fake-greedy-sampler"

    def chat_template(self, *args, **kwargs):
        self.record("chat_template", *args, **kwargs)
        return self.prompt

    def stream_generate(self, model, tokenizer, **kwargs):
        assert model is self.model_reference()
        assert tokenizer is self.tokenizer
        self.record("generate", **kwargs)
        try:
            if self.block_at:
                self.started.set()
                if not self.release.wait(timeout=3):
                    raise TimeoutError("Test did not release the native generation step")
                if self.block_at == "prefill":
                    kwargs["prompt_progress_callback"](1, len(self.prompt))
            if self.generation_error:
                raise self.generation_error
            for index, response in enumerate(self.responses):
                if self.block_after == index:
                    self.started.set()
                    if not self.release.wait(timeout=3):
                        raise TimeoutError("Test did not release the next native token")
                self.generated += 1
                yield response
        finally:
            self.record("generator_closed")


@pytest.fixture
def vendor(monkeypatch):
    vendor = FakeVendor()
    monkeypatch.setattr("inference_network.mlx_backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("inference_network.mlx_backend.platform.machine", lambda: "arm64")
    for name, module in {
        "mlx.core": vendor.mx,
        "mlx_lm": SimpleNamespace(load=vendor.load, stream_generate=vendor.stream_generate),
        "mlx_lm.sample_utils": SimpleNamespace(make_sampler=vendor.make_sampler),
        "huggingface_hub": SimpleNamespace(snapshot_download=vendor.snapshot_download),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return vendor


@pytest.fixture
async def backend(vendor):
    backend = MLXBackend(ProviderSettings(backend="mlx", max_prompt_tokens=3, max_output_tokens=8))
    try:
        yield backend
    finally:
        vendor.release.set()
        await backend.close()


@pytest.fixture
def request_payload():
    return ChatRequest(model=MLX_MODEL_ID, messages=[{"role": "user", "content": "Hello"}])


async def test_loading_is_offline_pinned_and_all_vendor_work_uses_one_worker(
    backend, vendor, request_payload
):
    await backend.load()
    await backend.load()
    await backend.generate(request_payload)
    assert backend.device_name == "Apple M4 (fake)"
    assert backend.memory_gb == 16
    assert [call for call in vendor.calls if call[0] == "snapshot"] == [
        (
            "snapshot",
            (MLX_MODEL_ID,),
            {"revision": MLX_MODEL_REVISION, "local_files_only": True, "token": False},
        )
    ]
    assert [call for call in vendor.calls if call[0] == "load"] == [
        (
            "load",
            ("/fake/cached/snapshot",),
            {
                "lazy": False,
                "trust_remote_code": False,
                "tokenizer_config": {"trust_remote_code": False},
            },
        )
    ]
    assert ("sampler", (), {"temp": 0}) in vendor.calls
    await backend.close()
    await backend.close()
    assert vendor.model_reference() is None
    assert len(set(vendor.threads)) == 1
    assert vendor.threads[0] is not threading.current_thread()
    assert not vendor.threads[0].is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        await backend.load()


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_generation_returns_text_and_vendor_usage_with_configured_limits(
    backend, vendor, request_payload, finish_reason
):
    vendor.responses[-1].finish_reason = finish_reason
    await backend.load()
    result = await backend.generate(request_payload)
    assert result.content == "Generated text."
    assert result.is_simulated is False
    assert result.finish_reason == finish_reason
    assert result.usage.model_dump() == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    template_call = next(call for call in vendor.calls if call[0] == "chat_template")
    assert template_call[1] == ([{"role": "user", "content": "Hello"}],)
    assert template_call[2] == {"tokenize": True, "add_generation_prompt": True}
    options = next(call[2] for call in vendor.calls if call[0] == "generate")
    assert options["prompt"] == [1, 2, 3]
    assert options["max_tokens"] == 8
    assert options["sampler"] == "fake-greedy-sampler"
    assert options["prefill_step_size"] == 256
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]


@pytest.mark.parametrize("failure_stage", ["snapshot", "load", "sampler"])
async def test_load_failure_does_not_allow_generation_and_close_releases_worker(
    backend, vendor, request_payload, failure_stage
):
    setattr(vendor, f"{failure_stage}_error", OSError("missing model files"))
    with pytest.raises(RuntimeError, match="MLX model loading failed") as failure:
        await backend.load()
    assert isinstance(failure.value.__cause__, OSError)
    with pytest.raises(NetworkError) as unavailable:
        await backend.generate(request_payload)
    assert unavailable.value.code == "model_not_loaded"
    await backend.close()
    if vendor.model_reference is not None:
        assert vendor.model_reference() is None
    assert [call[0] for call in vendor.calls[-2:]] == ["synchronize", "clear_cache"]
    assert not vendor.threads[0].is_alive()


async def test_cancelled_load_waits_for_native_work_and_close_releases_partial_model(
    backend, vendor, request_payload
):
    vendor.block_at = "load"
    loading = asyncio.create_task(backend.load())
    try:
        assert await asyncio.to_thread(vendor.started.wait, 3)
        loading.cancel()
        await asyncio.sleep(0)
        assert not loading.done()
        with pytest.raises(NetworkError) as unavailable:
            await backend.generate(request_payload)
        assert unavailable.value.code == "model_not_loaded"
    finally:
        vendor.release.set()
        with pytest.raises(asyncio.CancelledError):
            await loading
    await backend.close()
    assert vendor.model_reference() is None
    assert not vendor.threads[0].is_alive()


async def test_unloaded_closed_and_wrong_model_requests_are_rejected(
    backend, vendor, request_payload
):
    with pytest.raises(NetworkError) as unloaded:
        await backend.generate(request_payload)
    assert unloaded.value.status == 503
    await backend.load()
    with pytest.raises(NetworkError) as wrong_model:
        await backend.generate(request_payload.model_copy(update={"model": "other-model"}))
    assert wrong_model.value.code == "unsupported_model"
    assert wrong_model.value.status == 404
    await backend.close()
    with pytest.raises(NetworkError) as closed:
        await backend.generate(request_payload)
    assert closed.value.code == "model_not_loaded"
    assert not any(call[0] == "generate" for call in vendor.calls)


async def test_token_limit_rejects_before_generation_and_next_request_recovers(
    backend, vendor, request_payload
):
    await backend.load()
    vendor.prompt = [1, 2, 3, 4]
    with pytest.raises(NetworkError) as too_long:
        await backend.generate(request_payload)
    assert too_long.value.status == 422
    assert too_long.value.code == "prompt_too_long"
    assert not any(call[0] == "generate" for call in vendor.calls)
    vendor.prompt = [1, 2, 3]
    assert (await backend.generate(request_payload)).content == "Generated text."


@pytest.mark.parametrize(
    ("output", "finish_reason", "error_code"),
    [
        (" ", "stop", "empty_model_output"),
        ("x" * 8193, "stop", "model_output_limit"),
        ("text", "invalid", "invalid_model_result"),
    ],
)
async def test_invalid_model_output_is_rejected_and_generator_is_cleaned_up(
    backend, vendor, request_payload, output, finish_reason, error_code
):
    await backend.load()
    vendor.responses = [vendor.response(output, finish_reason)]
    with pytest.raises(NetworkError) as failure:
        await backend.generate(request_payload)
    assert failure.value.code == error_code
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]


async def test_vendor_generation_exception_releases_resources_and_allows_retry(
    backend, vendor, request_payload
):
    await backend.load()
    vendor.generation_error = RuntimeError("native generation failed")
    with pytest.raises(RuntimeError, match="native generation failed"):
        await backend.generate(request_payload)
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]
    vendor.generation_error = None
    assert (await backend.generate(request_payload)).content == "Generated text."


@pytest.mark.parametrize("block_at", ["prefill", "decode"])
@pytest.mark.parametrize("cancellation_count", [1, 2])
async def test_cancellation_keeps_capacity_reserved_until_native_work_finishes(
    backend, vendor, request_payload, block_at, cancellation_count
):
    await backend.load()
    vendor.block_at = block_at
    generation = asyncio.create_task(backend.generate(request_payload))
    try:
        assert await asyncio.to_thread(vendor.started.wait, 3)
        for _ in range(cancellation_count):
            generation.cancel()
            await asyncio.sleep(0)
            assert not generation.done()
        with pytest.raises(NetworkError) as busy:
            await backend.generate(request_payload)
        assert busy.value.status == 429
        assert busy.value.code == "backend_capacity"
    finally:
        vendor.release.set()
        with pytest.raises(asyncio.CancelledError):
            await generation
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]
    vendor.block_at = None
    assert (await backend.generate(request_payload)).content == "Generated text."


async def test_stream_yields_native_segments_before_generation_completes(
    backend, vendor, request_payload
):
    await backend.load()
    vendor.block_after = 1
    async with aclosing(backend.stream(request_payload)) as stream:
        first = await anext(stream)
        assert first.content == "Generated "
        assert first.finish_reason is None
        assert first.usage is None
        assert first.is_simulated is False
        assert await asyncio.to_thread(vendor.started.wait, 3)
        assert not any(call[0] == "generator_closed" for call in vendor.calls)
        with pytest.raises(NetworkError) as busy:
            await backend.generate(request_payload)
        assert busy.value.code == "backend_capacity"
        vendor.release.set()
        remaining = [chunk async for chunk in stream]
    assert [chunk.content for chunk in remaining] == ["text.", ""]
    assert remaining[-1].finish_reason == "stop"
    assert remaining[-1].usage.total_tokens == 5
    assert all(chunk.finish_reason is None for chunk in remaining[:-1])
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]


async def test_stream_terminal_waits_for_native_cleanup(backend, vendor, request_payload):
    await backend.load()

    def synchronize():
        vendor.record("synchronize")
        vendor.started.set()
        if not vendor.release.wait(timeout=3):
            raise TimeoutError("Test did not release native cleanup")

    vendor.mx.synchronize = synchronize
    async with aclosing(backend.stream(request_payload)) as stream:
        assert (await anext(stream)).content == "Generated "
        assert (await anext(stream)).content == "text."
        assert await asyncio.to_thread(vendor.started.wait, 3)
        terminal = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert not terminal.done()
        with pytest.raises(NetworkError) as busy:
            await backend.generate(request_payload)
        assert busy.value.code == "backend_capacity"
        vendor.release.set()
        final = await terminal
        assert final.content == ""
        assert final.finish_reason == "stop"
        assert vendor.calls[-1][0] == "clear_cache"


@pytest.mark.parametrize("cancel_closing", [False, True])
async def test_closing_suspended_stream_retains_capacity_until_worker_stops(
    backend, vendor, request_payload, cancel_closing
):
    await backend.load()
    vendor.block_after = 1
    stream = backend.stream(request_payload)
    assert (await anext(stream)).content == "Generated "
    assert await asyncio.to_thread(vendor.started.wait, 3)
    closing = asyncio.create_task(stream.aclose())
    try:
        await asyncio.sleep(0)
        if cancel_closing:
            for _ in range(2):
                closing.cancel()
                await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(NetworkError) as busy:
            await backend.generate(request_payload)
        assert busy.value.code == "backend_capacity"
    finally:
        vendor.release.set()
        if cancel_closing:
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]
    vendor.block_after = None
    assert (await backend.generate(request_payload)).content == "Generated text."


async def test_slow_stream_consumer_bounds_buffer_and_closes_without_deadlock(
    backend, vendor, request_payload
):
    from inference_network.mlx_backend import STREAM_BUFFER_CHUNKS

    await backend.load()
    vendor.responses = [vendor.response(str(index), None) for index in range(100)]
    vendor.responses[-1].finish_reason = "length"
    buffer_full = threading.Event()
    original_generate = vendor.stream_generate

    def tracked_generate(*args, **kwargs):
        for response in original_generate(*args, **kwargs):
            if vendor.generated == STREAM_BUFFER_CHUNKS + 2:
                buffer_full.set()
            yield response

    backend._lm.stream_generate = tracked_generate
    stream = backend.stream(request_payload)
    assert (await anext(stream)).content == "0"
    assert await asyncio.to_thread(buffer_full.wait, 3)
    # One consumed chunk, eight buffered chunks, and one producer-held chunk.
    assert vendor.generated == STREAM_BUFFER_CHUNKS + 2
    async with asyncio.timeout(3):
        await stream.aclose()
    assert [call[0] for call in vendor.calls[-3:]] == [
        "generator_closed",
        "synchronize",
        "clear_cache",
    ]
    vendor.responses = [vendor.response("Recovered", "stop")]
    assert (await backend.generate(request_payload)).content == "Recovered"


async def test_stream_drains_all_ordered_chunks_when_worker_completes(
    backend, vendor, request_payload
):
    await backend.load()
    vendor.responses = [vendor.response(str(index), None) for index in range(100)]
    vendor.responses[-1].finish_reason = "length"
    async with aclosing(backend.stream(request_payload)) as stream:
        chunks = [chunk async for chunk in stream]
    assert [chunk.content for chunk in chunks[:-1]] == [str(index) for index in range(100)]
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].content == ""


async def test_simulator_stream_is_explicitly_simulated_and_reconstructs_text(request_payload):
    backend = SimulatedEchoBackend()
    assert isinstance(backend, StreamingBackend)
    result = await backend.generate(request_payload)
    chunks = [chunk async for chunk in backend.stream(request_payload)]
    assert len(chunks) > 2
    assert "".join(chunk.content for chunk in chunks) == result.content
    assert all(chunk.is_simulated and chunk.usage is None for chunk in chunks)
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].content == ""
