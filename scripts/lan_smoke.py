"""Validate an existing SSH-forwarded network and record repeatable HTTP measurements.

Run from the repository root: python -m scripts.lan_smoke --help
This command never starts services, loads a model, or establishes SSH trust.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from inference_network.client import stream_request
from inference_network.config import MLX_MODEL_ID, MLX_MODEL_REVISION
from inference_network.errors import NetworkError
from inference_network.pairing import ProviderProfile, SenderProfile, load_profile
from inference_network.schemas import ChatRequest, validate_local_url
from scripts.workloads import PROMPTS

CAPACITY_ERRORS = {"no_eligible_provider", "provider_capacity", "backend_capacity"}
PUBLIC_ERROR_CODES = CAPACITY_ERRORS | {
    "unauthorized",
    "forbidden",
    "model_not_found",
    "unsupported_model",
    "provider_not_ready",
    "model_not_loaded",
    "provider_timeout",
    "request_timeout",
    "provider_error",
    "backend_error",
    "invalid_provider_response",
    "prompt_too_long",
    "empty_model_output",
    "model_output_limit",
    "invalid_model_result",
    "streaming_unsupported",
    "simulated_provider_error",
    "http_error",
    "request_rejected",
    "stream_failed",
}
PROVIDER_FIELDS = (
    "provider_id",
    "models",
    "gpu_type",
    "vram_gb",
    "memory_kind",
    "max_concurrency",
    "ready",
    "healthy",
    "online",
    "simulated",
    "streaming",
    "in_flight",
    "heartbeat_age_seconds",
    "completed_requests",
    "failed_requests",
    "registered",
    "connected",
    "active_requests",
    "backend",
)


@dataclass
class Observation:
    """An attempted request, retaining failures separately from success timing."""

    prompt_index: int
    status: str = "failure"
    latency_seconds: float = 0
    first_text_seconds: float | None = None
    request_id: str | None = None
    provider_id: str | None = None
    model: str | None = None
    simulated: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    text_sha256: str | None = None
    finish_reason: str | None = None
    done_received: bool = False
    http_status: int | None = None
    error_code: str | None = None


def request_for(prompt_index: int, model: str) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPTS[prompt_index]}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )


async def observe(
    client: httpx.AsyncClient, arguments: argparse.Namespace, prompt_index: int
) -> Observation:
    observation = Observation(prompt_index)
    started = time.perf_counter()
    path = "/internal/chat/completions" if arguments.provider_direct else "/v1/chat/completions"
    try:
        async with asyncio.timeout(arguments.timeout):
            result = await stream_request(
                client,
                request_for(prompt_index, arguments.model),
                path=path,
                timeout_seconds=arguments.timeout,
            )
        observation.http_status = 200
        observation.request_id = result.request_id
        observation.provider_id = result.provider_id
        observation.model = result.model
        observation.simulated = result.simulated
        observation.first_text_seconds = result.first_text_seconds
        observation.done_received = True  # The canonical client requires the terminal DONE event.
        observation.finish_reason = result.finish_reason
        if result.model != arguments.model:
            raise ValueError("Unexpected model")
        if result.simulated and not arguments.allow_simulation:
            raise ValueError("Real inference is required")
        if not result.content.strip() or result.first_text_seconds is None:
            raise ValueError("Expected streamed generated text")
        if result.finish_reason not in {"stop", "length"}:
            raise ValueError("Missing successful finish reason")
        if not result.simulated and (result.usage is None or result.usage.completion_tokens <= 0):
            raise ValueError("Expected actual positive completion token usage")
        if arguments.expected_provider and result.provider_id not in arguments.expected_provider:
            raise ValueError("Unexpected provider")
        if result.usage is not None:
            observation.prompt_tokens = result.usage.prompt_tokens
            observation.completion_tokens = result.usage.completion_tokens
            observation.total_tokens = result.usage.total_tokens
        observation.text_sha256 = hashlib.sha256(result.content.encode()).hexdigest()
        observation.status = "success"
    except (TimeoutError, httpx.TimeoutException):
        observation.status = "timeout"
        observation.error_code = "request_timeout"
    except NetworkError as error:
        observation.http_status = error.status
        observation.status = "timeout" if error.status == 504 else "failure"
        # Response messages and arbitrary error codes can contain sensitive input.
        observation.error_code = error.code if error.code in PUBLIC_ERROR_CODES else "network_error"
    except (httpx.HTTPError, ValueError) as error:
        observation.error_code = type(error).__name__
    observation.latency_seconds = time.perf_counter() - started
    return observation


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * fraction
    low, high = math.floor(offset), math.ceil(offset)
    return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)


def summarize(observations: list[Observation], wall_seconds: float) -> dict[str, Any]:
    successes = [item for item in observations if item.status == "success"]
    latency = [item.latency_seconds for item in successes]
    first_text = [
        item.first_text_seconds for item in successes if item.first_text_seconds is not None
    ]
    actual_tokens = sum(item.completion_tokens or 0 for item in successes if not item.simulated)
    simulated = any(item.simulated for item in successes)
    attempted = len(observations)
    return {
        "attempted": attempted,
        "completed": len(successes),
        "failed": attempted - len(successes),
        "failure_rate": (attempted - len(successes)) / attempted if attempted else None,
        "timeouts": sum(item.status == "timeout" for item in observations),
        "admission_rejections": sum(item.error_code in CAPACITY_ERRORS for item in observations),
        "errors": dict(Counter(item.error_code for item in observations if item.error_code)),
        "providers_served": dict(Counter(item.provider_id for item in successes)),
        "wall_seconds": wall_seconds,
        "latency_p50_seconds": percentile(latency, 0.5),
        "latency_p95_seconds": percentile(latency, 0.95),
        "first_text_p50_seconds": percentile(first_text, 0.5),
        "first_text_p95_seconds": percentile(first_text, 0.95),
        "successful_requests_per_second": len(successes) / wall_seconds if wall_seconds else None,
        "completion_tokens": None if simulated else actual_tokens,
        "completion_tokens_per_wall_second": (
            actual_tokens / wall_seconds if wall_seconds and not simulated else None
        ),
        "completion_tokens_per_request_second": (
            actual_tokens / sum(latency) if sum(latency) and not simulated else None
        ),
        "percentile_note": (
            "Fewer than 30 successes; percentile estimates are unstable."
            if len(successes) < 30
            else "Descriptive estimates for this run only."
        ),
    }


async def rejects_unauthenticated(client: httpx.AsyncClient, provider_direct: bool) -> bool:
    """A supplied token being accepted alone does not establish enforced authentication."""
    async with client.stream(
        "GET",
        "/health" if provider_direct else "/providers",
        headers={"Authorization": "", "Accept-Encoding": "identity"},
    ) as response:
        return response.status_code == 401


async def snapshot(client: httpx.AsyncClient, provider_direct: bool) -> list[dict[str, Any]]:
    async with client.stream(
        "GET",
        "/health" if provider_direct else "/providers",
        headers={"Accept-Encoding": "identity"},
    ) as response:
        response.raise_for_status()
        if response.headers.get("content-encoding", "identity") != "identity":
            raise ValueError("Expected an uncompressed provider inventory")
        body = bytearray()
        async for part in response.aiter_bytes():
            body.extend(part)
            if len(body) > 512 * 1024:
                raise ValueError("Provider inventory exceeded the response size limit")
        payload = json.loads(body)
    providers = [payload] if provider_direct else payload["providers"]
    if not isinstance(providers, list) or any(not isinstance(item, dict) for item in providers):
        raise ValueError("Invalid provider inventory")
    return [
        {field: item[field] for field in PROVIDER_FIELDS if field in item} for item in providers
    ]


async def run(arguments: argparse.Namespace, client: httpx.AsyncClient) -> dict[str, Any]:
    """Measure the existing services; validate physical host topology separately."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "test": "provider_http" if arguments.provider_direct else "coordinator_routed_http",
        "simulation_allowed": arguments.allow_simulation,
        "unauthenticated_access_rejected": False,
        "physical_two_host_verification": False,
        "verification_limit": (
            "API observations cannot prove two physical hosts or SSH encryption. "
            "Verify host hardware, SSH fingerprints/tunnels, and provider execution separately."
        ),
        "client_environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "owner_declaration": {
            "host_label": arguments.host_label,
            "network_note": arguments.network_note,
        },
        "endpoint": arguments.coordinator_url,
        "expected_providers": arguments.expected_provider,
        "workload": {
            "source": "scripts.workloads.PROMPTS (unchanged from V0.1.3 benchmarks)",
            "prompts": PROMPTS,
            "model": arguments.model,
            "expected_model_revision": MLX_MODEL_REVISION
            if arguments.model == MLX_MODEL_ID
            else None,
            "repetitions": arguments.repetitions,
            "warmups": arguments.warmups,
            "concurrency": arguments.concurrency,
            "timeout_seconds": arguments.timeout,
            "generation_settings": (
                "Provider-configured; match output limit and model revision separately."
            ),
            "pattern": "Closed-loop batches, matching prompt per batch; no retries or queue.",
        },
        "providers_before": [],
        "providers_after": [],
        "warmup": [],
        "observations": [],
        "validation_problems": [],
    }
    problems = result["validation_problems"]
    try:
        async with asyncio.timeout(arguments.timeout):
            result["unauthenticated_access_rejected"] = await rejects_unauthenticated(
                client, arguments.provider_direct
            )
            if not result["unauthenticated_access_rejected"]:
                problems.append("Endpoint did not reject unauthenticated access with HTTP 401.")
                return result
            result["providers_before"] = await snapshot(client, arguments.provider_direct)
        warmup = [
            await observe(client, arguments, index % len(PROMPTS))
            for index in range(arguments.warmups)
        ]
        result["warmup"] = [asdict(item) for item in warmup]
        if any(item.status != "success" for item in warmup):
            problems.append("Warmup failed; no warm measurements recorded.")
            return result
        observations = []
        started = time.perf_counter()
        for batch in range(arguments.repetitions):
            observations.extend(
                await asyncio.gather(
                    *(
                        observe(client, arguments, batch % len(PROMPTS))
                        for _ in range(arguments.concurrency)
                    )
                )
            )
        result["summary"] = summarize(observations, time.perf_counter() - started)
        result["observations"] = [asdict(item) for item in observations]
        successes = [item for item in observations if item.status == "success"]
        if not successes:
            problems.append("No successful completed inference requests.")
        unexpected = [
            item
            for item in observations
            if item.status != "success"
            and not (
                arguments.concurrency > 1
                and item.error_code in CAPACITY_ERRORS
                and item.http_status in {429, 503}
            )
        ]
        if unexpected:
            problems.append("One or more requests failed outside expected concurrent admission.")
        served = {item.provider_id for item in successes}
        if set(arguments.expected_provider) - served:
            problems.append("Not every expected provider completed a measured request.")
        async with asyncio.timeout(arguments.timeout):
            result["providers_after"] = await snapshot(client, arguments.provider_direct)
    except httpx.HTTPStatusError as error:
        problems.append(f"Provider inventory rejected (HTTP {error.response.status_code}).")
    except (httpx.HTTPError, ValueError, KeyError, TypeError, TimeoutError) as error:
        problems.append(f"Network inventory/authentication check failed ({type(error).__name__}).")
    return result


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinator-url", required=True, help="Existing loopback SSH-forward origin"
    )
    parser.add_argument(
        "--credentials-file", type=Path, required=True, help="Private pairing profile"
    )
    parser.add_argument(
        "--expected-provider", action="append", default=[], help="Require each to serve"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON file; never overwritten"
    )
    parser.add_argument("--model", default=MLX_MODEL_ID)
    parser.add_argument(
        "--provider-direct", action="store_true", help="Measure provider HTTP with its profile"
    )
    parser.add_argument(
        "--allow-simulation", action="store_true", help="Local tool validation only"
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Measured batches; use 30 for timing"
    )
    parser.add_argument("--warmups", type=int, default=0, help="Use 3 for warm timing comparisons")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--host-label", default="unspecified", help="Operator-provided host description"
    )
    parser.add_argument(
        "--network-note", default="unspecified", help="Operator-provided topology; no secrets"
    )
    arguments = parser.parse_args(argv)
    try:
        arguments.coordinator_url = validate_local_url(arguments.coordinator_url)
        request_for(0, arguments.model)
    except ValueError:
        parser.error("Use a numeric loopback HTTP origin and a valid model identifier")
    if not 1 <= arguments.repetitions <= 10000 or not 0 <= arguments.warmups <= 100:
        parser.error("repetitions must be 1–10000 and warmups 0–100")
    if not math.isfinite(arguments.timeout) or not 0 < arguments.timeout <= 120:
        parser.error("timeout must be finite, positive, and at most 120 seconds")
    if len(set(arguments.expected_provider)) != len(arguments.expected_provider):
        parser.error("expected-provider must not contain duplicates")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        profile = load_profile(arguments.credentials_file)
        expected_type = ProviderProfile if arguments.provider_direct else SenderProfile
        if not isinstance(profile, expected_type):
            raise ValueError("Wrong credential role")
        token = profile.token.get_secret_value()
        descriptor = os.open(arguments.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except (ValueError, OSError):
        print(
            "Cannot start: check credential role/permissions and use a new output file.",
            file=sys.stderr,
        )
        return 2

    async def execute() -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=arguments.coordinator_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=arguments.timeout,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            return await run(arguments, client)

    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        result = asyncio.run(execute())
        destination.write(json.dumps(result, indent=2) + "\n")
    problems = result["validation_problems"]
    print(f"{'FAILED' if problems else 'PASSED'} API checks; evidence: {arguments.output}")
    print("Validate cross-machine execution and SSH transport separately on the physical devices.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
