"""SSE wire validation and response ownership at cancellation boundaries."""

import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from inference_network.errors import NetworkError
from inference_network.registry import Reservation
from inference_network.schemas import (
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatRequest,
    ChunkChoice,
    ResponseMetadata,
    TokenUsage,
)
from inference_network.streaming import DONE_EVENT, EventStreamResponse, encode_chunk, iter_sse
from inference_network.transport import ProviderTransport, SafeConnectFailure

REQUEST = ChatRequest(
    model="test-model", messages=({"role": "user", "content": "Hello"},), stream=True
)
RESERVATION = Reservation(uuid4(), "provider-a", uuid4(), "http://127.0.0.1:8101", False)


def chunk(*, role=None, content=None, finish=None, usage=False, simulated=False):
    return ChatCompletionChunk(
        id="chatcmpl-test",
        created=123,
        model="test-model",
        choices=()
        if usage
        else (
            ChunkChoice(
                delta=ChatCompletionDelta(role=role, content=content), finish_reason=finish
            ),
        ),
        usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        if usage and not simulated
        else None,
        metadata=ResponseMetadata(provider_id="provider-a", simulated=simulated),
    )


def valid_chunks(*, simulated=False):
    return [
        chunk(role="assistant", simulated=simulated),
        chunk(content="Hello ", simulated=simulated),
        chunk(content="世界\n!", simulated=simulated),
        chunk(finish="stop", simulated=simulated),
        chunk(usage=True, simulated=simulated),
    ]


def wire(chunks):
    return b"".join(encode_chunk(item) for item in chunks) + DONE_EVENT


async def pieces(data, size=1):
    for start in range(0, len(data), size):
        yield data[start : start + size]


@pytest.mark.parametrize("size", [1, 2, 7, 256, 65536])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
async def test_sse_preserves_unicode_and_multiple_events_across_read_boundaries(size, newline):
    encoded = wire(valid_chunks()).replace(b"\n", newline)
    events = [event async for event in iter_sse(pieces(encoded, size))]
    assert events[-1].data == "[DONE]"
    payloads = [json.loads(event.data) for event in events[:-1]]
    assert (
        "".join(item["choices"][0]["delta"].get("content", "") for item in payloads[:-1])
        == "Hello 世界\n!"
    )
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert payloads[-2]["choices"][0]["delta"] == {}


async def test_sse_supports_multiline_data_comments_and_event_names():
    data = b': comment\r\nevent: error\r\ndata: {"error":\r\ndata: "example"}\r\n\r\n'
    events = [event async for event in iter_sse(pieces(data))]
    assert len(events) == 1
    assert events[0].event == "error"
    assert json.loads(events[0].data) == {"error": "example"}


@pytest.mark.parametrize(
    "data,limits",
    [
        (b"data: incomplete\n", {}),
        (b"data: \xff\n\n", {}),
        (b"data: 123456\n\n", {"max_event_bytes": 10}),
        (b": x\n\n" * 5, {"max_total_bytes": 20}),
        (b": x\n\n" * 3, {"max_events": 2}),
    ],
)
async def test_sse_rejects_truncated_invalid_or_excessive_wire_data(data, limits):
    with pytest.raises(ValueError):
        _ = [event async for event in iter_sse(pieces(data), **limits)]


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, data, failure=None):
        self.data = data
        self.failure = failure
        self.closed = False

    async def __aiter__(self):
        async for piece in pieces(self.data, 7):
            yield piece
        if self.failure:
            raise self.failure

    async def aclose(self):
        self.closed = True


async def read_transport(data, *, failure=None, reservation=RESERVATION):
    source = ByteStream(data, failure)
    requested = []

    def handle(request):
        requested.append(json.loads(request.content))
        return httpx.Response(200, stream=source, headers={"Content-Type": "text/event-stream"})

    result = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        try:
            async for item in ProviderTransport(client).stream(
                reservation, REQUEST, "chatcmpl-test"
            ):
                result.append(item)
        except NetworkError as exc:
            return result, exc, source, requested
    return result, None, source, requested


@pytest.mark.parametrize("simulated", [False, True])
async def test_transport_reconstructs_stream_and_requests_internal_usage(simulated):
    reservation = Reservation(
        RESERVATION.token,
        RESERVATION.provider_id,
        RESERVATION.instance_id,
        RESERVATION.endpoint,
        simulated,
    )
    expected = valid_chunks(simulated=simulated)
    chunks, error, source, requested = await read_transport(wire(expected), reservation=reservation)
    assert error is None
    assert chunks == expected
    assert source.closed
    assert requested[0]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    "change",
    [
        "id",
        "model",
        "provider",
        "simulated",
        "created",
        "missing_role",
        "repeat_role",
        "early_finish",
        "content_after_finish",
        "missing_finish",
        "missing_usage",
        "repeat_usage",
        "missing_done",
        "duplicate_done",
        "after_done",
        "trailing_partial",
        "error_event",
        "invalid_usage",
        "text_limit",
        "unknown_event",
        "empty_content",
    ],
)
async def test_transport_rejects_corrupt_stream_without_terminal_success(change):
    chunks = valid_chunks()
    if change in {"id", "model", "created"}:
        value = 999 if change == "created" else "wrong"
        chunks[2] = chunks[2].model_copy(update={change: value})
    elif change in {"provider", "simulated"}:
        update = {"provider_id": "wrong"} if change == "provider" else {"simulated": True}
        chunks[2] = chunks[2].model_copy(
            update={"metadata": chunks[2].metadata.model_copy(update=update)}
        )
    elif change == "missing_role":
        chunks.pop(0)
    elif change == "repeat_role":
        chunks.insert(2, chunks[0])
    elif change == "early_finish":
        chunks = [chunks[0], chunks[3], chunks[4]]
    elif change == "content_after_finish":
        chunks.insert(4, chunks[1])
    elif change == "missing_finish":
        chunks.pop(3)
    elif change == "missing_usage":
        chunks.pop()
    elif change == "repeat_usage":
        chunks.append(chunks[-1])
    elif change == "invalid_usage":
        chunks[-1] = chunks[-1].model_copy(update={"usage": None})
    elif change == "text_limit":
        chunks[1] = chunk(content="x" * 8192)
    elif change == "empty_content":
        chunks[1] = chunk(content="")
    data = wire(chunks)
    if change == "missing_done":
        data = data[: -len(DONE_EVENT)]
    elif change == "duplicate_done":
        data += DONE_EVENT
    elif change == "after_done":
        data += encode_chunk(chunks[1])
    elif change == "trailing_partial":
        data += b"data: partial"
    elif change == "error_event":
        data = (
            b"".join(encode_chunk(item) for item in chunks[:3])
            + b'event: error\ndata: {"private":"secret"}\n\n'
        )
    elif change == "unknown_event":
        data = b"event: unknown\n" + data
    received, error, source, _ = await read_transport(data)
    assert error is not None
    assert error.code in {"invalid_provider_response", "provider_error"}
    assert "secret" not in str(error)
    assert not any(item.choices and item.choices[0].finish_reason for item in received)
    assert source.closed


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError("late failure"), httpx.ConnectTimeout("late")]
)
async def test_post_header_connection_exceptions_are_not_safe_to_retry(failure):
    chunks, error, source, _ = await read_transport(
        encode_chunk(chunk(role="assistant")), failure=failure
    )
    assert chunks
    assert error is not None and not isinstance(error, SafeConnectFailure)
    assert source.closed


async def test_pre_header_connect_failure_remains_safe():
    def handle(request):
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(SafeConnectFailure):
            await anext(ProviderTransport(client).stream(RESERVATION, REQUEST, "chatcmpl-test"))


async def response_messages(iterator, deadline_seconds=1):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        await asyncio.Event().wait()

    await EventStreamResponse(iterator, deadline_seconds)({}, receive, send)
    return messages


async def test_response_formats_success_and_closes_owned_iterator():
    closed = asyncio.Event()

    async def produce():
        try:
            for item in valid_chunks():
                yield item
        finally:
            closed.set()

    messages = await response_messages(produce())
    assert messages[0]["type"] == "http.response.start"
    headers = dict(messages[0]["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    assert b"content-length" not in headers
    assert messages[-1]["body"] == DONE_EVENT
    assert not messages[-1]["more_body"]
    assert closed.is_set()


@pytest.mark.parametrize("after_commit", [False, True])
async def test_response_errors_before_commit_propagate_and_active_errors_are_sse(after_commit):
    closed = asyncio.Event()

    async def produce():
        try:
            if after_commit:
                yield chunk(role="assistant")
            raise NetworkError(502, "provider_error", "Generation failed")
        finally:
            closed.set()

    if after_commit:
        messages = await response_messages(produce())
        assert messages[-1]["body"].startswith(b"event: error\n")
        assert DONE_EVENT not in [message.get("body") for message in messages]
    else:
        with pytest.raises(NetworkError, match="Generation failed"):
            await response_messages(produce())
    assert closed.is_set()


@pytest.mark.parametrize("after_commit", [False, True])
async def test_response_deadline_includes_waiting_for_generated_chunks(after_commit):
    closed = asyncio.Event()

    async def produce():
        try:
            if after_commit:
                yield chunk(role="assistant")
            await asyncio.Event().wait()
        finally:
            closed.set()

    if after_commit:
        messages = await response_messages(produce(), deadline_seconds=0.01)
        assert b"request_timeout" in messages[-1]["body"]
        assert DONE_EVENT not in [message.get("body") for message in messages]
    else:
        with pytest.raises(NetworkError) as error:
            await response_messages(produce(), deadline_seconds=0.01)
        assert error.value.code == "request_timeout"
    assert closed.is_set()


async def test_backpressured_send_is_bounded_and_closes_iterator():
    closed = asyncio.Event()
    sent = []

    async def produce():
        try:
            yield chunk(role="assistant")
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message["more_body"]:
            await asyncio.Event().wait()

    async def receive():
        await asyncio.Event().wait()

    await asyncio.wait_for(EventStreamResponse(produce(), 0.01)({}, receive, send), 1)
    assert closed.is_set()
    assert b"request_timeout" in sent[-1]["body"]


@pytest.mark.parametrize("disconnect", [False, True])
async def test_cancellation_waits_for_generation_cleanup_even_when_repeated(disconnect):
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    allow_cleanup = asyncio.Event()
    closed = asyncio.Event()
    disconnected = asyncio.Event()

    async def produce():
        try:
            yield chunk(role="assistant")
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await allow_cleanup.wait()
            closed.set()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    task = asyncio.create_task(EventStreamResponse(produce(), 1)({}, receive, send))
    await asyncio.wait_for(entered.wait(), 1)
    if disconnect:
        disconnected.set()
    else:
        task.cancel()
    await asyncio.wait_for(cleanup_entered.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not closed.is_set()
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


def test_stream_options_require_streaming_and_default_remains_nonstreaming():
    payload = {"model": "test", "messages": [{"role": "user", "content": "Hello"}]}
    assert ChatRequest.model_validate(payload).stream is False
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({**payload, "stream_options": {"include_usage": True}})
    request = ChatRequest.model_validate({**payload, "stream": True, "stream_options": {}})
    assert request.stream_options.include_usage is False


async def test_cr_only_comment_and_empty_events_are_complete_frames():
    assert [event async for event in iter_sse(pieces(b": keepalive\r\r\r"))] == []


async def test_header_send_timeout_does_not_attempt_precommit_error_response():
    sent = []
    closed = asyncio.Event()

    async def produce():
        try:
            yield chunk(role="assistant")
        finally:
            closed.set()

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.start":
            await asyncio.Event().wait()

    async def receive():
        await asyncio.Event().wait()

    await asyncio.wait_for(EventStreamResponse(produce(), 0.01)({}, receive, send), 1)
    assert len([message for message in sent if message["type"] == "http.response.start"]) == 1
    assert b"request_timeout" in sent[-1]["body"]
    assert closed.is_set()
