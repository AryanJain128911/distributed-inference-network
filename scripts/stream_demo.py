"""Launch cached MLX providers and verify genuine streaming through the coordinator."""

import argparse
import json
import socket
import sys
import time
from collections.abc import Callable
from typing import Any

import httpx

from inference_network.config import MLX_MODEL_ID
from inference_network.launcher import LocalNetwork

PROMPT = "Explain what machine learning is in two sentences."
CHAT = {"model": MLX_MODEL_ID, "messages": [{"role": "user", "content": PROMPT}]}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def providers(client: httpx.Client) -> dict[str, dict[str, Any]]:
    response = client.get("/providers")
    response.raise_for_status()
    return {provider["provider_id"]: provider for provider in response.json()["providers"]}


def wait_for(condition: Callable[[], bool], message: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise RuntimeError(message)


def stream(client: httpx.Client, network: LocalNetwork) -> dict[str, Any]:
    payload = dict(CHAT, stream=True, stream_options={"include_usage": True})
    content: list[str] = []
    request_id = None
    provider_id = None
    finish_reason = None
    usage = None
    done = False
    observed_active_generation = False
    event = "message"
    started = time.monotonic()
    first_text_seconds = None
    print("Generated text: ", end="", flush=True)
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        require(
            response.headers.get("content-type", "").startswith("text/event-stream"),
            "Expected an SSE response",
        )
        for line in response.iter_lines():
            if not line:
                event = "message"
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            require(not done, "Unexpected event after [DONE]")
            if data == "[DONE]":
                done = True
                continue
            chunk = json.loads(data)
            require(event != "error", f"Streaming failed: {chunk.get('error', {}).get('code')}")
            require(chunk["object"] == "chat.completion.chunk", "Invalid completion chunk")
            require(chunk["model"] == MLX_MODEL_ID, "Unexpected model")
            require(chunk["metadata"]["simulated"] is False, "Expected genuine MLX inference")
            if request_id is None:
                request_id = chunk["id"]
                provider_id = chunk["metadata"]["provider_id"]
                require(chunk["choices"][0]["delta"].get("role") == "assistant", "Missing role")
            require(chunk["id"] == request_id, "Completion ID changed during the stream")
            require(chunk["metadata"]["provider_id"] == provider_id, "Provider changed midstream")
            if not chunk["choices"]:
                require(finish_reason is not None, "Usage arrived before the terminal chunk")
                usage = chunk["usage"]
                continue
            choice = chunk["choices"][0]
            text = choice["delta"].get("content")
            if text:
                require(finish_reason is None, "Text arrived after generation finished")
                if first_text_seconds is None:
                    first_text_seconds = time.monotonic() - started
                print(text, end="", flush=True)
                content.append(text)
                if len(content) == 2:
                    endpoint = network.provider_urls[provider_id]
                    health = client.get(f"{endpoint}/health")
                    health.raise_for_status()
                    observed_active_generation = health.json()["active_requests"] == 1
            if choice["finish_reason"] is not None:
                finish_reason = choice["finish_reason"]
    elapsed = time.monotonic() - started
    print()
    require(
        done and finish_reason in {"stop", "length"}, "Stream ended without successful completion"
    )
    require(
        len(content) > 1 and observed_active_generation, "Did not observe incremental generation"
    )
    require(usage is not None and usage["completion_tokens"] > 0, "Missing actual token usage")
    require(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        "Inconsistent token usage",
    )
    print(
        f"Stream: {provider_id}; {len(content)} text chunks; "
        f"finish={finish_reason}; [DONE] received"
    )
    print(
        f"Observed active generation after second text chunk; "
        f"first text={first_text_seconds:.3f}s; complete={elapsed:.3f}s"
    )
    print(f"Actual tokenizer usage: {json.dumps(usage, sort_keys=True)}")
    return {
        "content": "".join(content),
        "provider_id": provider_id,
        "usage": usage,
        "finish_reason": finish_reason,
    }


def demonstrate(coordinator_port: int, provider_ports: tuple[int, int]) -> None:
    network = LocalNetwork(
        coordinator_port,
        provider_ports,
        startup_timeout=60,
        environment={
            "DIN_BACKEND": "mlx",
            "DIN_MAX_CONCURRENCY": "1",
            "DIN_DELAY_SECONDS": "0",
            "DIN_MAX_OUTPUT_TOKENS": "128",
            "DIN_REQUEST_TIMEOUT_SECONDS": "60",
            "DIN_HEARTBEAT_INTERVAL_SECONDS": "0.1",
            "DIN_HEARTBEAT_TTL_SECONDS": "0.6",
            "DIN_STRATEGY": "round_robin",
            "HF_HUB_OFFLINE": "1",
        },
    )
    with (
        network,
        httpx.Client(base_url=network.coordinator_url, timeout=60, trust_env=False) as client,
    ):
        states = providers(client)
        require(
            all(
                state["ready"] and state["online"] and state["streaming"] and not state["simulated"]
                for state in states.values()
            ),
            "Both real streaming providers must be ready",
        )
        print(f"Coordinator: {network.coordinator_url}")
        for provider_id, state in states.items():
            print(
                f"Ready: {provider_id}; PID={network.processes[provider_id].pid}; "
                f"{state['gpu_type']}; model={state['models'][0]}"
            )
        print(f"Prompt: {PROMPT}")
        output = stream(client, network)
        nonstream = client.post("/v1/chat/completions", json=dict(CHAT, stream=False))
        nonstream.raise_for_status()
        completion = nonstream.json()
        require(
            completion["choices"][0]["message"]["content"] == output["content"],
            "Greedy streamed text differs from the non-streaming result",
        )
        require(completion["usage"] == output["usage"], "Token usage differs across response modes")
        print(f"Non-streaming: {completion['metadata']['provider_id']}; exact text and usage match")

        stopped = output["provider_id"]
        process = network.processes[stopped]
        process.terminate()
        process.wait(timeout=10)
        wait_for(
            lambda: not providers(client)[stopped]["online"], "Provider did not become offline"
        )
        print(f"Stopped {stopped}; heartbeat lease expired; online=false")
        continued = stream(client, network)
        require(continued["provider_id"] != stopped, "Offline provider was selected")
        wait_for(
            lambda: all(state["in_flight"] == 0 for state in providers(client).values()),
            "Coordinator capacity was not released",
        )
        print(f"Offline routing: request served by {continued['provider_id']}")
    require(
        all(process.poll() is not None for process in network.processes.values()), "Process leak"
    )
    for port in (coordinator_port, *provider_ports):
        with socket.socket() as probe:
            probe.settimeout(0.5)
            require(probe.connect_ex(("127.0.0.1", port)) != 0, f"Port {port} remains open")
    print(
        "PASS: genuine MLX streaming, non-streaming, offline routing, and process cleanup verified."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-port", type=int, default=8000)
    parser.add_argument("--provider-ports", type=int, nargs=2, default=(8101, 8102))
    arguments = parser.parse_args()
    try:
        demonstrate(arguments.coordinator_port, tuple(arguments.provider_ports))
        return 0
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
