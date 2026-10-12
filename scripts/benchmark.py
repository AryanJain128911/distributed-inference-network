"""Measure the existing MLX/backend/HTTP paths without changing request routing.

Run from the repository root: python -m scripts.benchmark --output benchmarks/my-run
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from inference_network.config import MLX_MODEL_ID, ProviderSettings
from inference_network.errors import NetworkError
from inference_network.mlx_backend import MLXBackend
from inference_network.schemas import ChatCompletion, ChatCompletionChunk, ChatRequest, TokenUsage
from inference_network.streaming import iter_sse
from scripts.benchmark_runtime import BenchmarkNetwork, environment_metadata, rss_bytes
from scripts.workloads import PROMPTS


@dataclass
class Observation:
    """One attempted request. Failure timing is retained but not mixed into success latency."""

    prompt_index: int
    latency_seconds: float = 0
    first_text_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    text_chunks: int = 0
    text_sha256: str | None = None
    finish_reason: str | None = None
    status: str = "success"
    error_code: str | None = None
    http_status: int | None = None


def request_for(prompt_index: int, streaming: bool) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": MLX_MODEL_ID,
            "messages": [{"role": "user", "content": PROMPTS[prompt_index]}],
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
        }
    )


def finish_observation(
    observation: Observation, text: str, usage: TokenUsage | None, finish: str | None
) -> None:
    if not text.strip() or usage is None or usage.completion_tokens <= 0:
        raise ValueError("Expected generated text and actual positive token usage")
    if finish not in {"stop", "length"}:
        raise ValueError("Missing successful finish reason")
    observation.text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    observation.prompt_tokens = usage.prompt_tokens
    observation.completion_tokens = usage.completion_tokens
    observation.finish_reason = finish


async def direct_request(
    backend: MLXBackend, request: ChatRequest, observation: Observation, started: float
) -> None:
    if not request.stream:
        result = await backend.generate(request)
        if result.is_simulated:
            raise ValueError("Benchmark requires real model execution")
        finish_observation(observation, result.content, result.usage, result.finish_reason)
        return
    content = []
    usage = None
    finish = None
    async with aclosing(backend.stream(request)) as chunks:
        async for chunk in chunks:
            if chunk.is_simulated:
                raise ValueError("Benchmark requires real model execution")
            if chunk.content:
                if observation.first_text_seconds is None:
                    observation.first_text_seconds = time.perf_counter() - started
                observation.text_chunks += 1
                content.append(chunk.content)
            if chunk.finish_reason is not None:
                usage, finish = chunk.usage, chunk.finish_reason
    finish_observation(observation, "".join(content), usage, finish)


async def http_request(
    client: httpx.AsyncClient,
    url: str,
    request: ChatRequest,
    observation: Observation,
    started: float,
) -> None:
    async with client.stream(
        "POST", url, json=request.model_dump(mode="json", exclude_none=True)
    ) as response:
        observation.http_status = response.status_code
        if response.status_code != 200:
            body = await response.aread()
            try:
                code = json.loads(body)["error"]["code"]
            except (ValueError, KeyError, TypeError):
                code = "http_error"
            raise NetworkError(response.status_code, str(code), "Benchmark request failed")
        if not request.stream:
            result = ChatCompletion.model_validate_json(await response.aread())
            if result.model != request.model or result.metadata.simulated:
                raise ValueError("Unexpected model or simulated response")
            finish_observation(
                observation,
                result.choices[0].message.content,
                result.usage,
                result.choices[0].finish_reason,
            )
            return
        if not response.headers.get("content-type", "").startswith("text/event-stream"):
            raise ValueError("Expected SSE response")
        content = []
        usage = None
        finish = None
        identity = None
        done = False
        phase = "role"
        async for event in iter_sse(response.aiter_bytes()):
            if done:
                raise ValueError("Event after DONE")
            if event.event == "error":
                envelope = json.loads(event.data)
                if (
                    not isinstance(envelope, dict)
                    or not isinstance(envelope.get("error"), dict)
                    or not isinstance(envelope["error"].get("code"), str)
                ):
                    raise ValueError("Malformed stream error")
                code = envelope["error"]["code"]
                raise NetworkError(502, str(code), "Benchmark stream failed")
            if event.event != "message":
                raise ValueError("Unsupported SSE event")
            if event.data == "[DONE]":
                if phase != "done":
                    raise ValueError("Premature DONE")
                done = True
                continue
            chunk = ChatCompletionChunk.model_validate_json(event.data)
            current_identity = (chunk.id, chunk.created, chunk.model, chunk.metadata)
            if identity is None:
                identity = current_identity
            if (
                current_identity != identity
                or chunk.model != request.model
                or chunk.metadata.simulated
            ):
                raise ValueError("Unexpected stream identity or simulated response")
            if not chunk.choices:
                if phase != "usage":
                    raise ValueError("Out-of-order usage")
                usage = chunk.usage
                phase = "done"
                continue
            choice = chunk.choices[0]
            if phase == "role":
                if (
                    choice.delta.role != "assistant"
                    or choice.delta.content not in (None, "")
                    or choice.finish_reason is not None
                ):
                    raise ValueError("Expected initial assistant role")
                phase = "content"
                continue
            if phase != "content" or choice.delta.role is not None:
                raise ValueError("Out-of-order content or finish")
            if choice.delta.content:
                if choice.finish_reason is not None:
                    raise ValueError("Text in terminal event")
                if observation.first_text_seconds is None:
                    observation.first_text_seconds = time.perf_counter() - started
                content.append(choice.delta.content)
                observation.text_chunks += 1
            if choice.finish_reason is not None:
                finish = choice.finish_reason
                phase = "usage"
        if not done:
            raise ValueError("Stream ended without DONE")
        finish_observation(observation, "".join(content), usage, finish)


RequestOperation = Callable[[ChatRequest, Observation, float], Awaitable[None]]


async def measure(
    operation: RequestOperation, prompt_index: int, streaming: bool, deadline_seconds: float
) -> Observation:
    request = request_for(prompt_index, streaming)
    observation = Observation(prompt_index)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(deadline_seconds):
            await operation(request, observation, started)
    except (TimeoutError, httpx.TimeoutException) as error:
        observation.status = "timeout"
        observation.error_code = type(error).__name__
    except NetworkError as error:
        observation.status = "timeout" if "timeout" in error.code else "failure"
        observation.error_code = error.code
    except (httpx.HTTPError, ValueError, RuntimeError) as error:
        observation.status = "failure"
        observation.error_code = type(error).__name__
    observation.latency_seconds = time.perf_counter() - started
    return observation


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear interpolation between adjacent sorted observations (inclusive endpoints)."""
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * fraction
    lower = math.floor(offset)
    upper = math.ceil(offset)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (offset - lower)


def summarize(observations: list[Observation], wall_seconds: float) -> dict[str, Any]:
    successful = [item for item in observations if item.status == "success"]
    latencies = [item.latency_seconds for item in successful]
    first_text = [
        item.first_text_seconds for item in successful if item.first_text_seconds is not None
    ]
    tokens = sum(item.completion_tokens or 0 for item in successful)
    timeouts = sum(item.status == "timeout" for item in observations)
    return {
        "attempted": len(observations),
        "completed": len(successful),
        "failed": len(observations) - len(successful),
        "timeouts": timeouts,
        "failure_rate": (len(observations) - len(successful)) / len(observations),
        "timeout_rate": timeouts / len(observations),
        "errors": dict(Counter(item.error_code for item in observations if item.error_code)),
        "wall_seconds": wall_seconds,
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "first_text_p50_seconds": percentile(first_text, 0.5),
        "first_text_p95_seconds": percentile(first_text, 0.95),
        "successful_requests_per_second": len(successful) / wall_seconds,
        "completion_tokens_per_wall_second": tokens / wall_seconds,
        "completion_tokens_per_request_second": tokens / sum(latencies) if latencies else None,
        "prompt_tokens": sum(item.prompt_tokens or 0 for item in successful),
        "completion_tokens": tokens,
        "percentile_note": (
            "Small sample: fewer than 30 successes; percentile estimates are unstable."
            if len(successful) < 30
            else "Descriptive estimates for this run; p95 has few tail observations."
        ),
    }


async def run_scenario(
    operation: RequestOperation,
    *,
    path: str,
    streaming: bool,
    concurrency: int,
    repetitions: int,
    warmups: int,
    deadline_seconds: float,
    pids: dict[str, int],
) -> dict[str, Any]:
    warmup = [
        await measure(operation, index % len(PROMPTS), streaming, deadline_seconds)
        for index in range(warmups)
    ]
    if any(item.status != "success" for item in warmup):
        raise RuntimeError(f"{path} warmup failed: {[asdict(item) for item in warmup]}")
    memory_before = rss_bytes(pids)
    observations = []
    started = time.perf_counter()
    for batch in range(repetitions):
        # Closed-loop batches: one request per client, then wait for the whole batch.
        # The existing one-slot backend/provider decides admission; no retry or queue.
        observations.extend(
            await asyncio.gather(
                *(
                    measure(operation, batch % len(PROMPTS), streaming, deadline_seconds)
                    for _ in range(concurrency)
                )
            )
        )
    wall_seconds = time.perf_counter() - started
    return {
        "path": path,
        "mode": "streaming" if streaming else "nonstreaming",
        "concurrency": concurrency,
        "warmup": [asdict(item) for item in warmup],
        "rss_bytes_before": memory_before,
        "rss_bytes_after": rss_bytes(pids),
        "summary": summarize(observations, wall_seconds),
        "observations": [asdict(item) for item in observations],
    }


def check_equivalence(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    outputs: dict[int, set[tuple[Any, ...]]] = {}
    for scenario in scenarios:
        for item in scenario["observations"]:
            if item["status"] == "success":
                outputs.setdefault(item["prompt_index"], set()).add(
                    tuple(
                        item[key]
                        for key in (
                            "text_sha256",
                            "prompt_tokens",
                            "completion_tokens",
                            "finish_reason",
                        )
                    )
                )
    return {
        "matches": bool(outputs) and all(len(values) == 1 for values in outputs.values()),
        "distinct_outputs_per_prompt": {str(key): len(value) for key, value in outputs.items()},
        "note": "Exact text hash, prompt/completion counts, and finish reason across successes.",
    }


def validate_result(result: dict[str, Any]) -> list[str]:
    """Only intentional single-slot admission failures are expected under offered load."""
    problems = []
    if not result["output_equivalence"]["matches"]:
        problems.append("Successful outputs differ in text, token counts or finish reason")
    capacity_codes = {
        "direct": "backend_capacity",
        "provider": "provider_capacity",
        "coordinator": "no_eligible_provider",
    }
    for scenario in result["scenarios"]:
        summary = scenario["summary"]
        if not summary["completed"]:
            problems.append(f"No completed measurements in {scenario['path']} {scenario['mode']}")
        for item in scenario["observations"]:
            if item["status"] == "success":
                continue
            if (
                scenario["concurrency"] == 1
                or item["status"] == "timeout"
                or item["error_code"] != capacity_codes[scenario["path"]]
            ):
                problems.append(f"Unexpected {scenario['path']} failure: {item['error_code']}")
    for path, startup in result["startups"].items():
        for record in startup.get("shutdown", []):
            if record["forced_kill"] or record["returncode"] not in (0, -signal.SIGTERM):
                problems.append(f"Unclean {path} shutdown: {record['role']}")
            if not record["listener_closed"]:
                problems.append(f"Listener remains open after {path} shutdown: {record['role']}")
    return problems


def comparison_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# MLX inference benchmark",
        "",
        f"Started {result['started_at']}; model revision: {result['model_revision']}.",
        "",
        "| Path | Mode | Clients | Completed/attempted | First text p50 ms | "
        "Latency p50 / p95 ms | Output tokens/s | Success req/s | Fail / timeout |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def milliseconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 1000:.2f}"

    for scenario in result["scenarios"]:
        summary = scenario["summary"]
        lines.append(
            f"| {scenario['path']} | {scenario['mode']} | {scenario['concurrency']} | "
            f"{summary['completed']}/{summary['attempted']} | "
            f"{milliseconds(summary['first_text_p50_seconds'])} | "
            f"{milliseconds(summary['latency_p50_seconds'])} / "
            f"{milliseconds(summary['latency_p95_seconds'])} | "
            f"{summary['completion_tokens_per_wall_second']:.2f} | "
            f"{summary['successful_requests_per_second']:.2f} | "
            f"{summary['failure_rate']:.1%} / {summary['timeout_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Output equivalence: {result['output_equivalence']['matches']}.",
            f"Validation problems: {result.get('validation_problems', [])}.",
            "",
            "Tokens/s uses actual completion tokens divided by measured workload wall time. "
            "Success latency excludes failures; throughput includes their elapsed time. "
            "First text is observable only for streaming. Warmups and startup are excluded.",
            "",
            "Percentiles use linear interpolation. These are descriptive local estimates; "
            "even 30 successes provide few p95 tail observations. Concurrency >1 measures "
            "admission/rejection with the existing single MLX slot, not GPU scaling.",
            "",
            "See results.json for every request, startup and first-request timings, "
            "boundary RSS samples, environment, workload, and cleanup evidence. "
            "RSS is not GPU allocation or peak memory. See docs/benchmarking.md for definitions.",
        ]
    )
    return "\n".join(lines) + "\n"


async def benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    settings = ProviderSettings(
        backend="mlx",
        max_output_tokens=arguments.max_output_tokens,
        request_timeout_seconds=arguments.timeout,
    )
    result: dict[str, Any] = dict(environment_metadata(settings))
    result.update(
        schema_version=1,
        started_at=datetime.now(UTC).isoformat(),
        command=sys.argv,
        workload={
            "prompts": PROMPTS,
            "repetitions_per_scenario": arguments.repetitions,
            "warmup_requests_per_scenario": arguments.warmups,
            "concurrency": arguments.concurrency,
            "paths_in_order": arguments.paths,
            "modes_in_order": arguments.modes,
            "sampling": {"temperature": 0, "algorithm": "greedy", "random_seed": None},
            "max_prompt_tokens": settings.max_prompt_tokens,
            "max_output_tokens": settings.max_output_tokens,
            "request_timeout_seconds": settings.request_timeout_seconds,
            "provider_max_concurrency": 1,
            "load_pattern": "closed-loop simultaneous batches, same prompt within each batch",
        },
        startups={},
        scenarios=[],
    )

    async def scenarios(operation: RequestOperation, path: str, pids: dict[str, int]) -> None:
        cold = await measure(operation, 0, True, arguments.timeout)
        result["startups"][path]["first_request_after_load"] = asdict(cold)
        if cold.status != "success":
            raise RuntimeError(f"First {path} request failed: {asdict(cold)}")
        for concurrency in arguments.concurrency:
            for mode in arguments.modes:
                scenario = await run_scenario(
                    operation,
                    path=path,
                    streaming=mode == "streaming",
                    concurrency=concurrency,
                    repetitions=arguments.repetitions,
                    warmups=arguments.warmups,
                    deadline_seconds=arguments.timeout,
                    pids=pids,
                )
                result["scenarios"].append(scenario)
                (arguments.output / "partial.json").write_text(json.dumps(result, indent=2) + "\n")
                print(
                    f"{path} {mode} clients={concurrency}: "
                    f"{scenario['summary']['completed']}/"
                    f"{scenario['summary']['attempted']} completed",
                    flush=True,
                )

    for path in arguments.paths:
        if path == "direct":
            backend = MLXBackend(settings)
            try:
                started = time.perf_counter()
                await backend.load()
                result["startups"][path] = {
                    "backend_load_seconds": time.perf_counter() - started,
                    "device_name": backend.device_name,
                    "unified_memory_gib": backend.memory_gb,
                    "note": "load() includes vendor imports and cached model initialization; "
                    "filesystem caches are not flushed.",
                }

                await scenarios(
                    partial(direct_request, backend), path, {"direct_process": os.getpid()}
                )
            finally:
                await backend.close()
            result["startups"][path]["backend_closed"] = True
        else:
            network = BenchmarkNetwork(settings, arguments.output / path)
            # Synchronous process startup/cleanup takes place outside timed requests.
            with network:
                result["startups"][path] = {
                    "process_to_ready_seconds": network.startup_seconds,
                    "note": "Includes interpreter, imports, model loading, HTTP and registration; "
                    "not a pure model-load measurement. Filesystem caches are not flushed.",
                }
                base_url = network.provider_url if path == "provider" else network.coordinator_url
                route = (
                    "/internal/chat/completions" if path == "provider" else "/v1/chat/completions"
                )
                async with httpx.AsyncClient(
                    timeout=arguments.timeout, trust_env=False, follow_redirects=False
                ) as client:
                    await scenarios(
                        partial(http_request, client, base_url + route),
                        path,
                        {key: process.pid for key, process in network.processes.items()},
                    )
            result["startups"][path]["shutdown"] = network.shutdown_records
        # Preserve completed paths if a later process or model fails.
        (arguments.output / "partial.json").write_text(json.dumps(result, indent=2) + "\n")
    result["output_equivalence"] = check_equivalence(result["scenarios"])
    result["validation_problems"] = validate_result(result)
    return result


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwritten"
    )
    parser.add_argument("--repetitions", type=int, default=30, help="Measured batches per scenario")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 2), default=[1])
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=("direct", "provider", "coordinator"),
        default=["direct", "provider", "coordinator"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("streaming", "nonstreaming"),
        default=["streaming", "nonstreaming"],
    )
    arguments = parser.parse_args(argv)
    if arguments.repetitions < 1 or arguments.warmups < 1:
        parser.error("repetitions and warmups must be positive")
    if not math.isfinite(arguments.timeout) or not 0 < arguments.timeout <= 120:
        parser.error("timeout must be finite, greater than zero, and at most 120 seconds")
    if not 1 <= arguments.max_output_tokens <= 512:
        parser.error("max-output-tokens must be between 1 and 512")
    for name in ("concurrency", "paths", "modes"):
        values = getattr(arguments, name)
        if len(values) != len(set(values)):
            parser.error(f"{name} must not contain duplicates")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    # Refuse to overwrite historical benchmark evidence.
    arguments.output.mkdir(parents=True, exist_ok=False)

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = asyncio.run(benchmark(arguments))
        (arguments.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        (arguments.output / "comparison.md").write_text(comparison_markdown(result))
        (arguments.output / "partial.json").unlink()
        print(comparison_markdown(result))
        return 1 if result["validation_problems"] else 0
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
