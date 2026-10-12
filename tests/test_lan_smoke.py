"""LAN smoke contracts with synthetic HTTP peers, never physical LAN evidence."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

import httpx
import pytest
from pydantic import SecretStr

from inference_network.config import MLX_MODEL_ID
from inference_network.pairing import SenderProfile, save_profile
from inference_network.schemas import (
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChunkChoice,
    ResponseMetadata,
    TokenUsage,
)
from inference_network.streaming import DONE_EVENT, encode_chunk
from scripts import lan_smoke

USAGE = TokenUsage(prompt_tokens=7, completion_tokens=11, total_tokens=18)


def arguments(tmp_path, *extra):
    return lan_smoke.parse_arguments(
        [
            "--coordinator-url",
            "http://127.0.0.1:18000",
            "--credentials-file",
            str(tmp_path / "sender.json"),
            "--output",
            str(tmp_path / "evidence.json"),
            *extra,
        ]
    )


def frames(*, provider="studio", simulated=False, completion_id="chatcmpl-lan-test"):
    def chunk(*, role=None, text=None, finish=None, usage=False):
        return encode_chunk(
            ChatCompletionChunk(
                id=completion_id,
                created=123,
                model=MLX_MODEL_ID,
                choices=()
                if usage
                else (
                    ChunkChoice(
                        delta=ChatCompletionDelta(role=role, content=text),
                        finish_reason=finish,
                    ),
                ),
                usage=USAGE if usage and not simulated else None,
                metadata=ResponseMetadata(provider_id=provider, simulated=simulated),
            )
        )

    return [
        chunk(role="assistant"),
        chunk(text="Generated "),
        chunk(text="output."),
        chunk(finish="stop"),
        chunk(usage=True),
        DONE_EVENT,
    ]


class WireStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


async def observe(tmp_path, chunks, *extra):
    stream = WireStream(chunks)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://127.0.0.1:18000"
    ) as client:
        observation = await lan_smoke.observe(client, arguments(tmp_path, *extra), 0)
    assert stream.closed
    return observation, requests


async def test_smoke_records_real_identity_usage_done_and_timing_without_output_text(tmp_path):
    observed, requests = await observe(tmp_path, frames(), "--expected-provider", "studio")
    assert observed.status == "success"
    assert observed.provider_id == "studio"
    assert observed.model == MLX_MODEL_ID
    assert observed.request_id == "chatcmpl-lan-test"
    assert observed.simulated is False
    assert observed.done_received
    assert observed.prompt_tokens == 7
    assert observed.completion_tokens == 11
    assert observed.total_tokens == 18
    assert observed.finish_reason == "stop"
    assert observed.first_text_seconds is not None
    assert observed.latency_seconds >= observed.first_text_seconds
    assert observed.text_sha256 == hashlib.sha256(b"Generated output.").hexdigest()
    assert requests[0].url.path == "/v1/chat/completions"
    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("missing", ["done", "usage", "finish", "text"])
async def test_smoke_rejects_incomplete_or_unaccounted_stream(tmp_path, missing):
    chunks = frames()
    if missing == "text":
        del chunks[1:3]
    else:
        del chunks[{"done": -1, "usage": -2, "finish": -3}[missing]]
    observed, _ = await observe(tmp_path, chunks)
    assert observed.status == "failure"
    assert not observed.done_received
    assert observed.completion_tokens is None


async def test_simulator_is_explicit_and_never_acquires_real_token_throughput(tmp_path):
    rejected, _ = await observe(tmp_path, frames(simulated=True))
    allowed, _ = await observe(tmp_path, frames(simulated=True), "--allow-simulation")
    assert rejected.status == "failure"
    assert allowed.status == "success"
    assert allowed.simulated is True
    assert allowed.completion_tokens is None
    summary = lan_smoke.summarize([allowed], 1)
    assert summary["completion_tokens"] is None
    assert summary["completion_tokens_per_wall_second"] is None
    assert summary["completion_tokens_per_request_second"] is None


async def test_smoke_rejects_an_unexpected_provider(tmp_path):
    observed, _ = await observe(tmp_path, frames(), "--expected-provider", "air")
    assert observed.status == "failure"
    assert observed.provider_id == "studio"


async def test_provider_http_baseline_uses_existing_internal_route(tmp_path):
    observed, requests = await observe(tmp_path, frames(), "--provider-direct")
    assert observed.status == "success"
    assert requests[0].url.path == "/internal/chat/completions"


async def test_cancellation_propagates_and_closes_the_request(tmp_path, monkeypatch):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(lan_smoke, "stream_request", cancelled)
    async with httpx.AsyncClient() as client:
        with pytest.raises(asyncio.CancelledError):
            await lan_smoke.observe(client, arguments(tmp_path), 0)


async def test_run_matches_benchmark_prompts_excludes_warmups_and_marks_verification_boundary(
    tmp_path,
):
    requested = []

    def respond(request):
        if not request.headers.get("Authorization"):
            return httpx.Response(401)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "providers": [
                        {"provider_id": "studio", "simulated": False, "unrelated_secret": "omit me"}
                    ]
                },
            )
        requested.append(json.loads(request.content))
        return httpx.Response(
            200, content=b"".join(frames()), headers={"Content-Type": "text/event-stream"}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        base_url="http://127.0.0.1:18000",
        headers={"Authorization": "Bearer synthetic-test-token"},
    ) as client:
        result = await lan_smoke.run(
            arguments(
                tmp_path, "--warmups", "3", "--repetitions", "4", "--expected-provider", "studio"
            ),
            client,
        )
    assert not result["validation_problems"]
    assert result["unauthenticated_access_rejected"] is True
    assert len(result["warmup"]) == 3
    assert result["summary"]["attempted"] == 4
    assert result["summary"]["completion_tokens"] == 44
    assert result["physical_two_host_verification"] is False
    assert result["client_environment"]["hostname"]
    assert "SSH" in result["verification_limit"]
    assert "omit me" not in json.dumps(result)
    assert [request["messages"][0]["content"] for request in requested] == [
        lan_smoke.PROMPTS[index] for index in (0, 1, 2, 0, 1, 2, 0)
    ]


@pytest.mark.parametrize(
    ("concurrency", "rejection", "successful"),
    [
        (2, "no_eligible_provider", True),
        (1, "no_eligible_provider", False),
        (2, "provider_error", False),
    ],
)
async def test_concurrent_admission_is_counted_without_masking_other_failures(
    tmp_path, monkeypatch, concurrency, rejection, successful
):
    observed = [
        lan_smoke.Observation(0, status="success", provider_id="studio", completion_tokens=11),
        lan_smoke.Observation(0, error_code=rejection, http_status=503),
    ]

    async def synthetic(*args):
        return observed.pop(0)

    async def inventory(*args):
        return []

    async def requires_auth(*args):
        return True

    monkeypatch.setattr(lan_smoke, "observe", synthetic)
    monkeypatch.setattr(lan_smoke, "snapshot", inventory)
    monkeypatch.setattr(lan_smoke, "rejects_unauthenticated", requires_auth)
    extra = ("--repetitions", "2") if concurrency == 1 else ()
    async with httpx.AsyncClient() as client:
        result = await lan_smoke.run(
            arguments(tmp_path, "--concurrency", str(concurrency), *extra), client
        )
    assert bool(result["validation_problems"]) is not successful
    assert result["summary"]["failure_rate"] == 0.5
    assert result["summary"]["admission_rejections"] == (rejection == "no_eligible_provider")


def test_summary_separates_failure_latency_and_uses_actual_tokens():
    first = lan_smoke.Observation(
        0,
        status="success",
        latency_seconds=0.2,
        first_text_seconds=0.05,
        completion_tokens=20,
        simulated=False,
    )
    summary = lan_smoke.summarize(
        [
            first,
            replace(first, latency_seconds=0.8, first_text_seconds=0.15, completion_tokens=40),
            lan_smoke.Observation(0, latency_seconds=20),
        ],
        3,
    )
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["latency_p50_seconds"] == 0.5
    assert summary["latency_p95_seconds"] == pytest.approx(0.77)
    assert summary["completion_tokens_per_wall_second"] == 20
    assert summary["completion_tokens_per_request_second"] == 60
    assert summary["first_text_p50_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "extra",
    [
        ["--coordinator-url", "http://192.168.1.3:8000"],
        ["--coordinator-url", "http://127.0.0.1:8000/secret"],
        ["--timeout", "nan"],
        ["--timeout", "0"],
        ["--timeout", "121"],
        ["--repetitions", "0"],
        ["--warmups", "-1"],
        ["--concurrency", "9"],
        ["--expected-provider", "studio", "--expected-provider", "studio"],
    ],
)
def test_invalid_targets_and_workloads_are_rejected_before_request(tmp_path, extra):
    with pytest.raises(SystemExit) as rejected:
        arguments(tmp_path, *extra)
    assert rejected.value.code == 2


def test_main_uses_private_credentials_and_never_overwrites_evidence(tmp_path, monkeypatch, capsys):
    profile = SenderProfile(
        token=SecretStr("synthetic_pairing_token_" + "x" * 43),
        ssh_host="192.168.1.2",
        ssh_user="tester",
    )
    save_profile(tmp_path / "sender.json", profile)
    seen = []

    async def synthetic_run(args, client):
        seen.append(client.headers["Authorization"])
        return {"validation_problems": [], "physical_two_host_verification": False}

    monkeypatch.setattr(lan_smoke, "run", synthetic_run)
    argv = [
        "--coordinator-url",
        "http://127.0.0.1:18000",
        "--credentials-file",
        str(tmp_path / "sender.json"),
        "--output",
        str(tmp_path / "evidence.json"),
    ]
    assert lan_smoke.main(argv) == 0
    artifact = (tmp_path / "evidence.json").read_text()
    assert seen == [f"Bearer {profile.token.get_secret_value()}"]
    assert os.stat(tmp_path / "evidence.json").st_mode & 0o777 == 0o600
    assert profile.token.get_secret_value() not in artifact + str(capsys.readouterr())
    assert lan_smoke.main(argv) == 2
    assert len(seen) == 1
    assert (tmp_path / "evidence.json").read_text() == artifact


def test_sender_tool_import_does_not_load_any_mlx_module():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.lan_smoke, sys; "
            "assert not any(name.startswith('mlx') or name == 'inference_network.mlx_backend' "
            "for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (503, "no_eligible_provider", "no_eligible_provider"),
        (429, "provider_capacity", "provider_capacity"),
        (401, "secret_token_should_not_be_echoed", "request_rejected"),
        (503, "provider_not_ready", "request_rejected"),
    ],
)
async def test_http_failures_preserve_status_and_safe_admission_codes(
    tmp_path, status, code, expected
):
    def rejected(request):
        return httpx.Response(status, json={"error": {"code": code}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(rejected), base_url="http://127.0.0.1:18000"
    ) as client:
        observed = await lan_smoke.observe(client, arguments(tmp_path), 0)
    assert observed.status == "failure"
    assert observed.http_status == status
    assert observed.error_code == expected
    assert observed.completion_tokens is None


async def test_inventory_authorization_failure_is_recorded_without_response_body(tmp_path):
    def rejected(request):
        return httpx.Response(401, json={"error": {"message": "secret token"}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(rejected), base_url="http://127.0.0.1:18000"
    ) as client:
        result = await lan_smoke.run(arguments(tmp_path), client)
    assert result["validation_problems"] == ["Provider inventory rejected (HTTP 401)."]
    assert "secret token" not in json.dumps(result)
    assert result["observations"] == []


@pytest.mark.parametrize("encoding", ["gzip", "br"])
async def test_inventory_rejects_compression_before_reading_a_body(encoding):
    class UnreadableStream(WireStream):
        async def __aiter__(self):
            raise AssertionError("Compressed inventory must not be decompressed")
            yield b""  # Makes the sentinel an async generator.

    stream = UnreadableStream([])

    def compressed(request):
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, stream=stream, headers={"Content-Encoding": encoding})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(compressed), base_url="http://127.0.0.1:18000"
    ) as client:
        with pytest.raises(ValueError, match="uncompressed"):
            await lan_smoke.snapshot(client, provider_direct=False)
    assert stream.closed


def test_workload_remains_identical_to_original_benchmark():
    from scripts.benchmark import PROMPTS

    assert lan_smoke.PROMPTS == PROMPTS


async def test_every_expected_provider_must_complete_a_request(tmp_path, monkeypatch):
    async def synthetic(*args):
        return lan_smoke.Observation(0, status="success", provider_id="studio")

    async def inventory(*args):
        return []

    async def requires_auth(*args):
        return True

    monkeypatch.setattr(lan_smoke, "observe", synthetic)
    monkeypatch.setattr(lan_smoke, "snapshot", inventory)
    monkeypatch.setattr(lan_smoke, "rejects_unauthenticated", requires_auth)
    async with httpx.AsyncClient() as client:
        result = await lan_smoke.run(
            arguments(
                tmp_path,
                "--expected-provider",
                "studio",
                "--expected-provider",
                "air",
                "--repetitions",
                "2",
            ),
            client,
        )
    assert result["validation_problems"] == [
        "Not every expected provider completed a measured request."
    ]


@pytest.mark.parametrize("status", [200, 302, 403])
async def test_smoke_requires_unauthenticated_access_to_be_rejected(tmp_path, status):
    requests = []

    def unpaired(request):
        requests.append(request)
        return httpx.Response(status, content=b"not recorded")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unpaired),
        base_url="http://127.0.0.1:18000",
        headers={"Authorization": "Bearer synthetic-test-token"},
    ) as client:
        result = await lan_smoke.run(arguments(tmp_path), client)
    assert len(requests) == 1
    assert requests[0].headers["Authorization"] == ""
    assert result["unauthenticated_access_rejected"] is False
    assert result["validation_problems"] == [
        "Endpoint did not reject unauthenticated access with HTTP 401."
    ]
    assert result["observations"] == []


async def test_configured_deadline_reaches_canonical_stream_client(tmp_path, monkeypatch):
    timeouts = []

    async def timeout(*args, timeout_seconds, **kwargs):
        timeouts.append(timeout_seconds)
        raise TimeoutError

    monkeypatch.setattr(lan_smoke, "stream_request", timeout)
    async with httpx.AsyncClient() as client:
        observed = await lan_smoke.observe(client, arguments(tmp_path, "--timeout", "90"), 0)
    assert timeouts == [90]
    assert observed.status == "timeout"
    assert observed.error_code == "request_timeout"
