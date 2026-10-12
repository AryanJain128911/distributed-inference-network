"""Benchmark accounting and wire observation with fake execution only; no GPU work."""

import asyncio
import hashlib
import json
import signal
from dataclasses import asdict, replace

import httpx
import pytest

from inference_network.backend import BackendChunk, BackendResult
from inference_network.config import MLX_MODEL_ID
from inference_network.errors import NetworkError
from inference_network.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatMessage,
    Choice,
    ChunkChoice,
    ResponseMetadata,
    TokenUsage,
)
from inference_network.streaming import DONE_EVENT, encode_chunk
from scripts import benchmark

USAGE = TokenUsage(prompt_tokens=7, completion_tokens=11, total_tokens=18)
METADATA = ResponseMetadata(provider_id="benchmark-provider", simulated=False)


def chunk(*, content=None, role=None, finish=None, usage=False):
    return ChatCompletionChunk(
        id="chatcmpl-benchmark-test",
        created=123,
        model=MLX_MODEL_ID,
        choices=()
        if usage
        else (
            ChunkChoice(
                delta=ChatCompletionDelta(content=content, role=role), finish_reason=finish
            ),
        ),
        usage=USAGE if usage else None,
        metadata=METADATA,
    )


def stream_frames():
    return [
        encode_chunk(chunk(role="assistant")),
        encode_chunk(chunk(content="Hello ")),
        encode_chunk(chunk(content="world!")),
        encode_chunk(chunk(finish="stop")),
        encode_chunk(chunk(usage=True)),
        DONE_EVENT,
    ]


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


class WireStream(httpx.AsyncByteStream):
    def __init__(self, frames, clock=None, failure=None):
        self.frames = frames
        self.clock = clock
        self.failure = failure
        self.closed = False

    async def __aiter__(self):
        for index, frame in enumerate(self.frames, start=1):
            if self.clock is not None:
                self.clock.now = 10.0 + index
            yield frame
        if self.failure is not None:
            raise self.failure

    async def aclose(self):
        self.closed = True


async def observe_http(
    frames,
    *,
    streaming=True,
    status=200,
    clock=None,
    failure=None,
    content_type="text/event-stream",
):
    source = WireStream(frames, clock, failure)
    requested = []

    def respond(request):
        requested.append(json.loads(request.content))
        return httpx.Response(status, stream=source, headers={"Content-Type": content_type})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def operation(request, observation, started):
            await benchmark.http_request(
                client,
                "http://127.0.0.1:8101/internal/chat/completions",
                request,
                observation,
                started,
            )

        observation = await benchmark.measure(operation, 0, streaming, 3)
    assert source.closed
    return observation, requested


async def test_http_stream_measures_first_text_not_role_and_uses_terminal_tokens(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(benchmark.time, "perf_counter", clock)
    observed, requests = await observe_http(stream_frames(), clock=clock)
    assert observed.status == "success"
    assert observed.first_text_seconds == 2
    assert observed.latency_seconds == 6
    assert observed.text_chunks == 2
    assert observed.completion_tokens == 11
    assert observed.prompt_tokens == 7
    assert observed.text_sha256 == hashlib.sha256(b"Hello world!").hexdigest()
    assert observed.finish_reason == "stop"
    assert requests[0]["stream_options"] == {"include_usage": True}


async def test_http_nonstreaming_keeps_generation_first_text_unavailable():
    completion = ChatCompletion(
        id="chatcmpl-benchmark-test",
        created=123,
        model=MLX_MODEL_ID,
        choices=(Choice(message=ChatMessage(role="assistant", content="Hello world!")),),
        usage=USAGE,
        metadata=METADATA,
    )
    observed, requests = await observe_http(
        [completion.model_dump_json().encode()],
        streaming=False,
        content_type="application/json",
    )
    assert observed.status == "success"
    assert observed.first_text_seconds is None
    assert observed.text_chunks == 0
    assert observed.completion_tokens == 11
    assert "stream_options" not in requests[0]


@pytest.mark.parametrize("missing", ["done", "usage", "finish", "content"])
async def test_http_stream_requires_successful_termination_and_real_usage(missing):
    frames = stream_frames()
    if missing == "done":
        frames.pop()
    elif missing == "usage":
        frames.pop(-2)
    elif missing == "finish":
        frames.pop(-3)
    else:
        del frames[1:3]
    observed, _ = await observe_http(frames)
    assert observed.status == "failure"
    assert observed.error_code == "ValueError"
    assert observed.text_sha256 is None


@pytest.mark.parametrize(
    "frames",
    [
        [b'event: error\ndata: {"error": {"code": "provider_error"}}\n\n'],
        stream_frames()[:2] + [b'event: error\ndata: {"error": {"code": "provider_error"}}\n\n'],
    ],
)
async def test_http_stream_error_is_failure_even_after_generated_text(frames):
    observed, _ = await observe_http(frames)
    assert observed.status == "failure"
    assert observed.error_code == "provider_error"
    assert observed.http_status == 200
    assert (observed.first_text_seconds is not None) == (len(frames) > 1)
    assert observed.completion_tokens is None


@pytest.mark.parametrize("error_body", [b"{}", b'{"error": null}', b"[]", b"invalid JSON"])
async def test_malformed_sse_error_records_failure_instead_of_aborting_run(error_body):
    observed, _ = await observe_http([b"event: error\ndata: " + error_body + b"\n\n"])
    assert observed.status == "failure"
    assert observed.error_code


@pytest.mark.parametrize("violation", ["unknown_event", "text_after_finish", "early_usage"])
async def test_http_stream_rejects_protocol_order_and_unknown_event_names(violation):
    frames = stream_frames()
    if violation == "unknown_event":
        frames[1] = b"event: surprise\n" + frames[1]
    elif violation == "text_after_finish":
        frames[2], frames[3] = frames[3], frames[2]
    else:
        frames[0], frames[4] = frames[4], frames[0]
    observed, _ = await observe_http(frames)
    assert observed.status == "failure"


@pytest.mark.parametrize(
    ("status", "body", "expected_status", "expected_code"),
    [
        (429, b'{"error":{"code":"provider_capacity"}}', "failure", "provider_capacity"),
        (504, b'{"error":{"code":"provider_timeout"}}', "timeout", "provider_timeout"),
        (502, b"not json", "failure", "http_error"),
    ],
)
async def test_http_error_accounting_retains_status_and_timeout_classification(
    status, body, expected_status, expected_code
):
    observed, _ = await observe_http([body], status=status)
    assert observed.status == expected_status
    assert observed.error_code == expected_code
    assert observed.http_status == status


async def test_http_read_timeout_preserves_partial_text_and_closes_stream():
    observed, _ = await observe_http(
        stream_frames()[:2], failure=httpx.ReadTimeout("test read timeout")
    )
    assert observed.status == "timeout"
    assert observed.error_code == "ReadTimeout"
    assert observed.first_text_seconds is not None
    assert observed.completion_tokens is None


async def test_deadline_waits_for_operation_cleanup_and_records_timeout():
    cleaned = asyncio.Event()

    async def operation(request, observation, started):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    observed = await benchmark.measure(operation, 0, True, 0.01)
    assert observed.status == "timeout"
    assert cleaned.is_set()
    assert observed.latency_seconds > 0


async def test_external_cancellation_is_preserved_instead_of_counted_as_failure():
    async def operation(request, observation, started):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await benchmark.measure(operation, 0, False, 1)


@pytest.mark.parametrize("streaming", [False, True])
async def test_direct_path_uses_backend_contract_and_terminal_usage(streaming):
    class FakeBackend:
        closed = False

        async def generate(self, request):
            return BackendResult("Hello world!", False, USAGE)

        async def stream(self, request):
            try:
                yield BackendChunk(content="Hello ")
                yield BackendChunk(content="world!")
                yield BackendChunk(usage=USAGE, finish_reason="stop")
            finally:
                self.closed = True

    backend = FakeBackend()

    async def operation(request, observation, started):
        await benchmark.direct_request(backend, request, observation, started)

    observed = await benchmark.measure(operation, 0, streaming, 1)
    assert observed.status == "success"
    assert observed.completion_tokens == 11
    assert observed.text_chunks == (2 if streaming else 0)
    assert (observed.first_text_seconds is not None) == streaming
    assert backend.closed == streaming


def successful_observation(**updates):
    return replace(
        benchmark.Observation(
            prompt_index=0,
            latency_seconds=0.2,
            first_text_seconds=0.05,
            prompt_tokens=7,
            completion_tokens=20,
            text_chunks=2,
            text_sha256="test-hash",
            finish_reason="stop",
        ),
        **updates,
    )


def test_summary_separates_success_latency_from_failures_and_uses_batch_wall_time():
    observations = [
        successful_observation(),
        successful_observation(latency_seconds=0.8, first_text_seconds=0.15, completion_tokens=40),
        benchmark.Observation(0, latency_seconds=0.1, status="failure", error_code="capacity"),
        benchmark.Observation(0, latency_seconds=0.9, status="timeout", error_code="ReadTimeout"),
    ]
    summary = benchmark.summarize(observations, wall_seconds=3)
    assert (summary["attempted"], summary["completed"], summary["failed"]) == (4, 2, 2)
    assert summary["timeouts"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["timeout_rate"] == 0.25
    assert summary["latency_p50_seconds"] == pytest.approx(0.5)
    assert summary["latency_p95_seconds"] == pytest.approx(0.77)
    assert summary["first_text_p50_seconds"] == pytest.approx(0.1)
    assert summary["successful_requests_per_second"] == pytest.approx(2 / 3)
    assert summary["completion_tokens_per_wall_second"] == 20
    assert summary["completion_tokens_per_request_second"] == 60
    assert summary["prompt_tokens"] == 14
    assert summary["completion_tokens"] == 60
    assert summary["errors"] == {"capacity": 1, "ReadTimeout": 1}
    assert "fewer than 30" in summary["percentile_note"]


def test_summary_with_no_success_has_no_success_latency_or_token_throughput():
    summary = benchmark.summarize(
        [benchmark.Observation(0, latency_seconds=1, status="failure")], wall_seconds=2
    )
    assert summary["latency_p50_seconds"] is None
    assert summary["latency_p95_seconds"] is None
    assert summary["first_text_p50_seconds"] is None
    assert summary["completion_tokens_per_request_second"] is None
    assert summary["completion_tokens_per_wall_second"] == 0
    assert summary["successful_requests_per_second"] == 0
    assert summary["failure_rate"] == 1


def test_percentiles_sort_without_mutation_and_handle_single_or_missing_samples():
    values = [0.7, 0.1, 0.4]
    assert benchmark.percentile(values, 0.5) == 0.4
    assert benchmark.percentile(values, 0.95) == pytest.approx(0.67)
    assert values == [0.7, 0.1, 0.4]
    assert benchmark.percentile([0.2], 0.95) == 0.2
    assert benchmark.percentile([], 0.95) is None


async def test_scenario_excludes_warmups_and_records_configured_batch_attempts(monkeypatch):
    monkeypatch.setattr(benchmark, "rss_bytes", lambda pids: {"test": 123})
    requests = []

    async def operation(request, observation, started):
        requests.append(request)
        benchmark.finish_observation(observation, "Hello world!", USAGE, "stop")

    scenario = await benchmark.run_scenario(
        operation,
        path="direct",
        streaming=False,
        concurrency=2,
        repetitions=4,
        warmups=3,
        deadline_seconds=1,
        pids={"test": 123},
    )
    assert len(requests) == 11
    assert len(scenario["warmup"]) == 3
    assert scenario["summary"]["attempted"] == 8
    assert scenario["summary"]["completion_tokens"] == 8 * USAGE.completion_tokens
    assert [item["prompt_index"] for item in scenario["observations"]] == [0, 0, 1, 1, 2, 2, 0, 0]
    assert scenario["rss_bytes_before"] == {"test": 123}
    assert scenario["rss_bytes_after"] == {"test": 123}


async def test_failed_warmup_prevents_reporting_a_warm_scenario(monkeypatch):
    async def operation(request, observation, started):
        raise NetworkError(503, "model_not_loaded", "Test backend unavailable")

    with pytest.raises(RuntimeError, match="warmup failed"):
        await benchmark.run_scenario(
            operation,
            path="direct",
            streaming=True,
            concurrency=1,
            repetitions=1,
            warmups=1,
            deadline_seconds=1,
            pids={},
        )


@pytest.mark.parametrize(
    "change",
    [
        {"text_sha256": "different-hash"},
        {"prompt_tokens": 8},
        {"completion_tokens": 21},
        {"finish_reason": "length"},
    ],
)
def test_equivalence_checks_text_tokens_and_finish_reason_per_prompt(change):
    first = successful_observation()
    scenarios = [
        {"observations": [asdict(first)]},
        {"observations": [asdict(replace(first, **change))]},
    ]
    assert benchmark.check_equivalence(scenarios)["matches"] is False
    assert benchmark.check_equivalence(scenarios)["distinct_outputs_per_prompt"] == {"0": 2}


def test_equivalence_ignores_failures_and_keeps_different_prompts_separate():
    first = successful_observation()
    second = successful_observation(prompt_index=1, text_sha256="another-prompt-output")
    scenarios = [
        {"observations": [asdict(first), asdict(second)]},
        {"observations": [asdict(first), asdict(benchmark.Observation(0, status="failure"))]},
    ]
    assert benchmark.check_equivalence(scenarios)["matches"] is True
    assert benchmark.check_equivalence([])["matches"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--repetitions", "0"],
        ["--warmups", "0"],
        ["--concurrency", "3"],
        ["--concurrency", "1", "1"],
        ["--paths", "direct", "direct"],
        ["--modes", "streaming", "streaming"],
        ["--max-output-tokens", "0"],
        ["--max-output-tokens", "513"],
        ["--timeout", "0"],
        ["--timeout", "121"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
    ],
)
def test_cli_rejects_invalid_workload_before_execution(tmp_path, arguments):
    with pytest.raises(SystemExit) as rejected:
        benchmark.parse_arguments(["--output", str(tmp_path / "run"), *arguments])
    assert rejected.value.code == 2


def test_cli_defaults_preserve_single_slot_and_at_least_thirty_samples(tmp_path):
    arguments = benchmark.parse_arguments(["--output", str(tmp_path / "run")])
    assert arguments.repetitions == 30
    assert arguments.warmups == 3
    assert arguments.concurrency == [1]
    assert arguments.max_output_tokens == 128
    assert arguments.paths == ["direct", "provider", "coordinator"]
    assert arguments.modes == ["streaming", "nonstreaming"]


def test_existing_results_directory_is_never_overwritten(tmp_path, monkeypatch):
    previous = tmp_path / "results.json"
    previous.write_text("previous verification evidence\n")
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        benchmark.main()
    assert previous.read_text() == "previous verification evidence\n"


def valid_result():
    return {
        "output_equivalence": {"matches": True},
        "scenarios": [
            {
                "path": "provider",
                "mode": "streaming",
                "concurrency": 1,
                "summary": {"completed": 1},
                "observations": [asdict(successful_observation())],
            }
        ],
        "startups": {
            "provider": {
                "shutdown": [
                    {
                        "role": "provider",
                        "returncode": 0,
                        "forced_kill": False,
                        "listener_closed": True,
                    }
                ]
            }
        },
    }


@pytest.mark.parametrize(
    ("path", "capacity_code"),
    [
        ("direct", "backend_capacity"),
        ("provider", "provider_capacity"),
        ("coordinator", "no_eligible_provider"),
    ],
)
def test_validation_accepts_only_expected_concurrent_admission_rejections(path, capacity_code):
    result = valid_result()
    scenario = result["scenarios"][0]
    scenario.update(path=path, concurrency=2)
    scenario["observations"].append(
        asdict(benchmark.Observation(0, status="failure", error_code=capacity_code))
    )
    assert benchmark.validate_result(result) == []
    scenario["concurrency"] = 1
    assert benchmark.validate_result(result)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("failure", "provider_error"),
        ("failure", "backend_error"),
        ("failure", "ValueError"),
        ("failure", "no_eligible_provider"),
        ("timeout", "provider_timeout"),
    ],
)
def test_validation_rejects_unexpected_errors_even_with_two_clients(status, code):
    result = valid_result()
    scenario = result["scenarios"][0]
    scenario["concurrency"] = 2
    scenario["observations"].append(
        asdict(benchmark.Observation(0, status=status, error_code=code))
    )
    assert benchmark.validate_result(result)


def test_validation_requires_completed_measurements_and_matching_outputs():
    result = valid_result()
    assert benchmark.validate_result(result) == []
    result["output_equivalence"]["matches"] = False
    assert benchmark.validate_result(result)
    result["output_equivalence"]["matches"] = True
    result["scenarios"][0]["summary"]["completed"] = 0
    result["scenarios"][0]["observations"] = []
    assert benchmark.validate_result(result)


@pytest.mark.parametrize(
    "cleanup_change",
    [{"forced_kill": True}, {"returncode": 1}, {"listener_closed": False}],
)
def test_validation_requires_clean_subprocess_and_listener_shutdown(cleanup_change):
    result = valid_result()
    result["startups"]["provider"]["shutdown"][0].update(cleanup_change)
    assert benchmark.validate_result(result)


def test_validation_accepts_expected_sigterm_exit_with_closed_listener():
    result = valid_result()
    result["startups"]["provider"]["shutdown"][0]["returncode"] = -signal.SIGTERM
    assert benchmark.validate_result(result) == []


@pytest.mark.parametrize("problems", [[], ["Unexpected provider failure: provider_error"]])
def test_main_preserves_results_and_returns_failure_when_validation_fails(
    tmp_path, monkeypatch, problems
):
    output = tmp_path / "new-run"
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--output", str(output)])

    async def measured_result(arguments):
        (arguments.output / "partial.json").write_text("{}\n")
        return {"validation_problems": problems}

    monkeypatch.setattr(benchmark, "benchmark", measured_result)
    monkeypatch.setattr(benchmark, "comparison_markdown", lambda result: "test comparison\n")
    assert benchmark.main() == (1 if problems else 0)
    assert json.loads((output / "results.json").read_text())["validation_problems"] == problems
    assert (output / "comparison.md").read_text() == "test comparison\n"
    assert not (output / "partial.json").exists()
