"""Exercise a running local network and print evidence from real HTTP responses."""

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

import httpx

from inference_network.schemas import validate_local_url

MODEL = "simulated-echo-v1"
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello, distributed network."}]}


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


def set_mode(client: httpx.Client, endpoint: str, mode: str) -> None:
    response = client.put(f"{endpoint}/simulation", json={"mode": mode, "delay_seconds": 0.05})
    response.raise_for_status()


def complete(client: httpx.Client) -> dict[str, Any]:
    response = client.post("/v1/chat/completions", json=CHAT)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    require(result["metadata"]["simulated"] is True, "Response must declare simulated inference")
    require(result["usage"] is None, "Simulator must not invent token usage")
    require(
        result["choices"][0]["message"]["content"]
        == "[SIMULATED simulated-echo-v1] Hello, distributed network.",
        "Response did not match the deterministic simulator contract",
    )
    return result


def demonstrate(url: str) -> None:
    with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        registered = providers(client)
        require(
            set(registered) == {"provider-a", "provider-b"}, "Launch the default two-provider demo"
        )
        endpoints = {
            name: validate_local_url(provider["endpoint"]) for name, provider in registered.items()
        }
        require(
            all(
                item["online"] and item["ready"] and item["healthy"] for item in registered.values()
            ),
            "Both providers must be healthy before running the walkthrough",
        )
        print(f"Coordinator: {url}; strategy: {health.json()['strategy']}")
        print(
            "Registered independently: provider-a, provider-b (simulated hardware advertisements)"
        )

        routes = [complete(client)["metadata"]["provider_id"] for _ in range(4)]
        require(
            routes.count("provider-a") == routes.count("provider-b") == 2,
            "Uneven sequential routing",
        )
        print(f"Four real HTTP completions: {' → '.join(routes)}")
        print(
            "Output verified: [SIMULATED simulated-echo-v1] Hello, distributed network.; usage=null"
        )

        unsupported = client.post("/v1/chat/completions", json=dict(CHAT, model="not-a-real-model"))
        require(unsupported.status_code == 404, "Unsupported model should return 404")
        require(unsupported.json()["error"]["code"] == "model_not_found", "Unexpected model error")
        print("Unsupported model: HTTP 404, model_not_found")

        try:
            set_mode(client, endpoints["provider-a"], "not_ready")
            wait_for(
                lambda: not providers(client)["provider-a"]["ready"],
                "Provider A readiness update did not arrive",
            )
            require(
                complete(client)["metadata"]["provider_id"] == "provider-b",
                "Unready provider selected",
            )
            print("Provider A unready: request served by provider-b")

            set_mode(client, endpoints["provider-b"], "error")
            failure = client.post("/v1/chat/completions", json=CHAT)
            require(failure.status_code == 502, "Controlled provider failure should return 502")
            require(
                failure.json()["error"]["code"] == "provider_error", "Unexpected provider error"
            )
            require(
                all(item["in_flight"] == 0 for item in providers(client).values()),
                "Capacity leaked",
            )
            print("Provider B controlled failure: HTTP 502, provider_error; reservations released")
        finally:
            for endpoint in endpoints.values():
                try:
                    set_mode(client, endpoint, "healthy")
                except httpx.HTTPError as error:
                    print(f"Could not restore {endpoint}: {type(error).__name__}", file=sys.stderr)

        wait_for(
            lambda: all(item["ready"] and item["healthy"] for item in providers(client).values()),
            "Providers did not recover",
        )
        recovered = [complete(client)["metadata"]["provider_id"] for _ in range(2)]
        require(
            set(recovered) == set(endpoints), "Both recovered providers should receive requests"
        )
        print(f"Recovered via heartbeats: {' → '.join(recovered)}")
        print(
            "PASS: local distributed simulation verified; no CUDA or real LLM inference executed."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=validate_local_url, default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    try:
        demonstrate(arguments.url)
        return 0
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
