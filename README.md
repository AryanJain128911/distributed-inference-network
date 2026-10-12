# Distributed Inference Network (DIN)

**Version 0.1.4** — an independent engineering prototype for routing LLM requests to available compute providers.

DIN explores model discovery, provider scheduling, capacity management, and failure handling. A coordinator selects a provider that supports the requested model and has a free execution slot. That provider runs the whole request on its own device; DIN does not combine devices into a virtual GPU.

```text
Sender ── request ──► Coordinator ── dispatch ──► Provider ──► MLX / Metal
       ◄────────────── streamed response ────────────────┘     or simulator
                     ◄── registration + heartbeats ────┘
```

Implemented capabilities include:

- Independent FastAPI services with model-aware round-robin or capacity-aware scheduling.
- Real Qwen inference on Apple Silicon, plus a labeled simulator without model weights.
- Incremental SSE streaming, actual token usage, deadlines, and cooperative cancellation.
- Sender/Provider terminal workflows and opt-in authenticated SSH networking.
- Reproducible direct-backend, provider-HTTP, and coordinator benchmarks.

All HTTP listeners bind to `127.0.0.1`; LAN mode carries traffic through verified SSH tunnels. See [architecture](docs/architecture.md) for admission, trust, and retry decisions.

## Quick start

From the repository root, using Python 3.12 (Python 3.11+ is supported):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e .
DIN_BACKEND=simulator din demo
```

This starts a coordinator and two simulator providers. In another activated terminal, run `python scripts/demo.py` to check routing and failure recovery. Ctrl+C stops the services.

Run `din` for the interactive menu: **Sender** creates or joins a network; **Provider** serves Qwen after [MLX setup](docs/setup.md#real-model-setup). Senders need no MLX or weights. Complete [secure two-Mac setup](docs/setup.md#secure-two-mac-setup) before using `din interactive --lan`.

## Measured results

Apple M4, 16 GiB unified memory, pinned 4-bit Qwen 0.5B, three prompts, 30 warm requests per path/mode:

| Streaming path | First text P50 | Total latency P50 | Completion tokens/s |
| --- | ---: | ---: | ---: |
| Direct backend | 28.27 ms | 505.45 ms | 248.25 |
| Provider HTTP | 29.28 ms | 505.82 ms | 247.67 |
| Coordinator HTTP | 30.46 ms | 507.65 ms | 246.86 |

These are local workload measurements, including request processing and generation. [Benchmarking](docs/benchmarking.md) provides definitions, reproduction commands, and raw evidence.


