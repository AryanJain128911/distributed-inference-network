# Benchmarking and verification

DIN measures the pinned Qwen model through three paths: **A — direct MLX backend**, **B — provider HTTP**, and **C — coordinator → provider HTTP**. The [benchmark harness](../scripts/benchmark.py) uses the production backend and API without changing scheduling or capacity. The direct path includes tokenization, chat templating, worker/queue handling, generation, and result collection; it is an application baseline, not a GPU-kernel benchmark.

## Measured results

Measured October 10, 2026, on Apple M4 (Mac16,12), 10 logical CPUs, 16 GiB unified memory, macOS 27.0.1 arm64, and Python 3.12.14. Runtime: MLX/MLX Metal 0.32.3 and MLX-LM 0.32.0. The model, workload, and timing boundaries are defined below.

| Run | Raw observations | Generated comparison |
| --- | --- | --- |
| Serial A → B → C | [Serial JSON](../benchmarks/v0.1.3/serial/results.json) | [Serial comparison](../benchmarks/v0.1.3/serial/comparison.md) |
| Reverse C → B → A | [Reverse JSON](../benchmarks/v0.1.3/reverse/results.json) | [Reverse comparison](../benchmarks/v0.1.3/reverse/comparison.md) |
| Two offered clients | [Concurrent JSON](../benchmarks/v0.1.3/concurrent/results.json) | [Concurrent comparison](../benchmarks/v0.1.3/concurrent/comparison.md) |

Directory names identify the version measured; these are historical results, not current-source fingerprints. Public copies redact personal hostnames/home paths with explicit metadata, preserving measurements and source hashes.

### Warm serial inference

Each row has **30/30 successful requests**, zero failures/timeouts, and **2,870 actual completion tokens**:

| Path | Mode | First text P50 (ms) | Total P50 / P95 (ms) | Completion tokens/s | Successful requests/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct backend | SSE | 28.27 | 505.45 / 508.43 | 248.25 | 2.59 |
| Provider HTTP | SSE | 29.28 | 505.82 / 509.43 | 247.67 | 2.59 |
| Coordinator HTTP | SSE | 30.46 | 507.65 / 510.15 | 246.86 | 2.58 |
| Direct backend | JSON | unavailable | 505.62 / 507.96 | 248.15 | 2.59 |
| Provider HTTP | JSON | unavailable | 505.99 / 508.92 | 247.28 | 2.58 |
| Coordinator HTTP | JSON | unavailable | 507.93 / 510.37 | 246.59 | 2.58 |

Differences between independent medians, calculated before rounding:

| Response / metric | Run order | Provider − direct (ms) | Coordinator − provider (ms) | Coordinator − direct (ms) |
| --- | --- | ---: | ---: | ---: |
| SSE total latency | A → B → C | 0.38 | 1.82 | 2.20 |
| SSE total latency | C → B → A | 0.37 | 1.69 | 2.05 |
| SSE first text | A → B → C | 1.01 | 1.18 | 2.19 |
| SSE first text | C → B → A | 1.03 | 1.07 | 2.10 |
| JSON total latency | A → B → C | 0.37 | 1.95 | 2.32 |
| JSON total latency | C → B → A | 1.11 | 1.67 | 2.77 |

Reverse order also completed 180/180 requests. Full-path SSE median increments were **2.05–2.20 ms (0.41–0.43%)** total and **2.10–2.19 ms** to first text; JSON increments were **2.32–2.77 ms**. These differences include serialization, client validation, thread/event-loop work, and device variation. They do not isolate routing CPU time. Reversing order checks sensitivity; it is not randomized statistical proof or a universal overhead bound.

### Startup, memory, and admission

Primary-run initialization and first-request observations were separate from warm summaries:

| Path | Initialization (s) | First request: first text / total (ms) |
| --- | ---: | ---: |
| Direct | 0.790 | 53.56 / 167.29 |
| Provider | 1.472 | 69.06 / 184.61 |
| Coordinator | 1.469 | 72.79 / 186.69 |

Direct initialization times `load()`, including vendor imports. Both HTTP values time the provider process to readiness, including interpreter startup, polling, and registration; subtracting these values does not isolate model-loading overhead. The first request uses the short prompt and is a single observation, not a cold-load percentile. Files were cached and the OS cache was not flushed.

Boundary RSS was approximately **530.1 MiB** for the direct harness, **538.9–539.4 MiB** for providers, and **56.7–57.9 MiB** for coordinators. Direct RSS includes harness/vendor imports; HTTP samples exclude their client process. These are before/after process snapshots, not comparable total-memory measurements, peak memory, or GPU allocations.

Two offered clients produced **30 successes / 60 attempts in each of six scenarios**: 180 successes, 180 expected capacity rejections, zero timeouts. Full-path SSE achieved **244.25 completion tokens/s**, **2.55 successful requests/s**, and **508.79 / 516.11 ms** successful latency P50/P95. MLX still executed one request at a time; this measures overload admission, not parallel GPU generation or scaling.

Across all three runs: **720 attempts, 540 successful warm measurements, 180 intentional capacity failures, zero timeouts**. All 18 summaries reproduce from raw observations; successful text, usage, and finish reasons match across runs. Validation lists are empty. All 12 HTTP service processes exited with expected Uvicorn SIGTERM status, no forced kills, and closed ports; direct backends closed between paths.

## Reproduce

Complete [MLX setup](setup.md#real-model-setup), cache the pinned model, and run from the repository root. Stop independently launched inference services and other GPU workloads. Run only one benchmark or real-model test at a time; record background activity, power mode, and thermal conditions when known. Each output directory must be new, including after interruption.

```bash
# Serial comparison: 30 measured requests per path and response mode.
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.benchmark \
  --output benchmarks/runs/serial \
  --repetitions 30 --warmups 3 --max-output-tokens 128 --timeout 60

# Reverse path order.
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.benchmark \
  --output benchmarks/runs/reverse --paths coordinator provider direct \
  --repetitions 30 --warmups 3 --max-output-tokens 128 --timeout 60

# Two offered clients; unchanged one-request execution capacity.
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.benchmark \
  --output benchmarks/runs/concurrent --concurrency 2 \
  --repetitions 30 --warmups 3 --max-output-tokens 128 --timeout 60
```

Defaults run streaming then non-streaming. Select either with `--modes streaming` or `--modes nonstreaming`; `--paths` selects/orders paths. `--concurrency` accepts 1 and/or 2, output limits 1–512, and finite positive timeouts up to 120 seconds. Repetitions/warmups must be positive. These commands retain the 2,048-token prompt limit, 128-token output limit, and single MLX slot.

## Workload and measurement method

All paths use `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, with greedy temperature-zero sampling. The [shared workload](../scripts/workloads.py) contains:

| Prompt | Actual prompt / completion tokens | Finish |
| --- | ---: | --- |
| Explain what machine learning is in two sentences. | 39 / 31 | `stop` |
| Explain how computers work in ten paragraphs. | 38 / 128 | `length` |
| List five practical ways to save water at home, with a short explanation of each. | 46 / 128 | `length` |

Batches cycle through prompts, giving ten batches per prompt at 30 repetitions. With one client this is 30 attempts; with two, each batch submits the same prompt twice and waits for both results: 60 attempts. This is a closed-loop workload without an arrival-rate generator, retries, or an added queue. Simultaneous submission does not guarantee service overlap.

Paths run sequentially with one loaded backend at a time. B/C each create a fresh coordinator and one provider; B bypasses coordinator dispatch but retains registration/heartbeats. A runs inside the harness and closes before HTTP measurements, although imports/driver state may remain resident. The [runtime helper](../scripts/benchmark_runtime.py) uses distinct ephemeral loopback ports, clears inherited `DIN_` settings, and requires matching ready health and exactly one healthy registered provider. A competing port bind fails startup.

Each path loads once, records a separate first streaming request, then shares its loaded model across scenarios. Three sequential warmups per scenario cycle through the prompts; failures abort the run. Initialization, first-after-load, and warmup observations are retained but excluded from warm summaries. “Cold” means new backend/process initialization, not a fresh boot, uncached storage, or a guaranteed cold GPU.

### Timing and token accounting

Timers use `time.perf_counter()`. Request timing excludes request-object construction but includes response validation and text hashing. Workload wall time includes construction, batch scheduling, and failed attempts; initialization, warmups, and RSS sampling are outside it.

| JSON field | Meaning |
| --- | --- |
| `first_text_seconds` | Request start to first nonempty generated text: backend chunks for A, decoded SSE content for B/C. Role events and empty deltas do not count. |
| `latency_seconds` | Request start to consumed/validated completion, or handled failure. SSE success requires terminal usage and `[DONE]`. |
| `latency_p50_seconds`, `latency_p95_seconds` | Successful-request total-latency percentiles. First-text percentile fields likewise use successful streams. |
| `completion_tokens_per_wall_second` | Actual successful completion tokens / workload wall time; used in the results tables. |
| `completion_tokens_per_request_second` | Actual successful completion tokens / summed successful request latencies, not the mean of individual rates. |
| `successful_requests_per_second` | Completed requests / workload wall time. |
| `prompt_tokens`, `completion_tokens` | Actual backend tokenizer/generator counts, retained per request and summed over successes. |
| `failure_rate`, `timeout_rate` | Failures / attempts and timeouts / attempts; timeouts are a subset of failures. |

JSON cannot expose first generated text, so first-text fields are `null`, not total response latency. Text chunks are not tokens. Both token rates include prompt processing, generation, cleanup, and applicable client/HTTP work; neither is decode-only throughput. Incomplete output contributes no invented usage. Errors retain code/status/elapsed time, and cooperative native cleanup can outlast a deadline.

Percentiles interpolate at sorted index `(n − 1) × percentile`. Failures are excluded from latency percentiles but included in rates/wall time. Even 30 successes provide few tail observations, with only ten per prompt and unequal output lengths; these are descriptive estimates, not confidence intervals or production p95 guarantees. Fewer than 30 successes trigger a caution.

### Validation and evidence

Capacity rejections remain failures: `backend_capacity` at A, `provider_capacity` at B, and `no_eligible_provider` at C. Validation permits these only for concurrency greater than one and requires a success in every scenario. Unexpected errors, timeouts, or serial failures invalidate a run.

For each prompt, validation compares exact text SHA-256, actual usage, and finish reason across successful repetitions, paths, and modes. This checks equivalent work, not answer quality. Natural stopping below the output limit is preserved.

`results.json` retains all observations/warmups, workload, hardware/OS, Python/dependency versions, model revision, Git status/commit, source hashes, startup, and cleanup. Preserve it with `comparison.md` and service logs; review local metadata before sharing. `partial.json` is incomplete evidence and is removed on completion. Initialization/warmup failures can leave fewer results or none; validation problems return a nonzero exit even if a report was written.

RSS uses `ps -o rss=` before/after scenarios, converting KiB to bytes; unavailable values are null. The harness does not measure accelerator utilization, temperatures, clocks, memory pressure, power, energy, or cost. Installed unified memory is hardware capacity, not inference consumption.

Cleanup stops owned providers before coordinators with SIGTERM, bounded wait, and kill/reap fallback, including after startup failures or interruption. Exit status, forced kills, and closed ports are recorded; unexpected exits, forced kills, or remaining listeners invalidate cleanup. Unrelated services are never stopped.

## Functional inference evidence

These single-run checks establish behavior separately from the warm performance distributions:

| Evidence | Observation |
| --- | --- |
| Instrumented vendor stream, October 9, 2026 | Client first text at 0.094330292 s preceded vendor terminal output at 1.967771375 s, both from request start. Reconstructed text matched the vendor exactly, demonstrating incrementality. |
| Long stream, supported 512-output-token test limit | Computer-explanation prompt: 486 nonempty text chunks, 38 prompt + 487 completion = 525 tokens. JSON text/usage/finish matched; disconnect, capacity release, recovery, and cleanup passed. |
| Short streaming demonstration | Machine-learning prompt: 30 text chunks, 39 + 31 = 70 tokens, `stop`, `[DONE]`, exact JSON match; first text 0.123 s, total 0.246 s. |
| [Authenticated interactive MLX smoke](../benchmarks/v0.1.4/local-auth-smoke.json) | Interactive provider and authorized sender; identity, text hash, actual usage, and cleanup checked. |
| [SSH plus MLX smoke](../benchmarks/v0.1.4/local-ssh-mlx-smoke.json) | OpenSSH forward/reverse tunnels, registration, and real inference through one host's private interface; unauthenticated access rejected. |

Each smoke artifact contains one successful 39-prompt/31-completion-token request and `physical_two_host_verification: false`. They establish neither a latency distribution nor two-machine/LAN performance. Provider advertisements are not hardware attestations; text-chunk counts alone do not prove incremental delivery.

Tests also cover SSH host trust, approved/rejected clients, occupied/wildcard reverse listeners, paired authorization, cancellation, expiry/restart, and cleanup. Multiple-provider/client process tests use authenticated simulators. Distinct machines still require [physical-device validation](setup.md#physical-device-validation).

## Matched local and LAN measurements

Complete [secure two-Mac setup](setup.md#secure-two-mac-setup) and physical functional checks first. Run sequentially on Studio with no Air provider. Match the pinned revision, greedy sampling, 2,048/128 token limits, one slot, three warmups, and 30 measurements. Record provider configuration: the smoke client neither changes nor remotely verifies generation settings.

Stop Studio's provider and measure its direct backend:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.benchmark \
  --paths direct --modes streaming --concurrency 1 \
  --repetitions 30 --warmups 3 --max-output-tokens 128 --timeout 60 \
  --output benchmarks/runs/studio-direct-backend
```

Restart the paired Studio provider, wait for readiness, then measure local provider HTTP on Studio using its private invitation:

```sh
.venv/bin/python -m scripts.lan_smoke \
  --provider-direct --coordinator-url http://127.0.0.1:8101 \
  --credentials-file "$HOME/.din/invitations/provider-studio.json" \
  --expected-provider studio --repetitions 30 --warmups 3 --concurrency 1 \
  --host-label mac-studio --network-note 'Same-host provider HTTP on Studio' \
  --output benchmarks/runs/studio-provider-http.json
```

On Air, set `DIN_PAIRING_DIR` to the private network directory printed at creation, then measure the coordinator → SSH → Studio path:

```sh
mkdir -p benchmarks/runs
.venv/bin/python -m scripts.lan_smoke \
  --coordinator-url http://127.0.0.1:8000 \
  --credentials-file "$DIN_PAIRING_DIR/sender.json" \
  --expected-provider studio --repetitions 30 --warmups 3 --concurrency 1 \
  --host-label macbook-air --network-note 'Air coordinator to Studio over verified SSH LAN' \
  --output benchmarks/runs/air-routed-studio.json
```

Optionally measure SSH/LAN without coordinator routing: on Air, use `--provider-direct` against `http://127.0.0.1:18101`, the existing reverse tunnel, with the creator's private `provider-studio.json` and a new output file. Keep provider credentials restricted to their intended recipients.

Compare prompt indices, hashes, usage, and finish reasons across files before comparing successful latency/first-text p50/p95, actual completion tokens per wall second, attempts, and failure rates. The smoke tool does not check cross-file equivalence. Warmups are excluded; failures remain in rates/wall time. Repeat in another order with new output paths. Differences include client scheduling/parsing and device variation, not solely network latency.

The tool requires existing services/tunnels and an existing output parent; it refuses overwrites and writes mode 0600. Results retain authentication, inventory, identity, workload, usage, hashes, failures, and declared topology. It cannot attest physical separation (`physical_two_host_verification` stays false) or measure remote memory, loading, or energy. `--allow-simulation` validates the harness only. Preserve failed runs and review metadata before publication.

## Interpretation limits

Measured performance covers one device, one small quantized model, and three prompts over loopback HTTP. Thermal/power state and background desktop activity were not controlled or logged. There is no accuracy study, verified two-physical-machine inference, energy/price measurement, or established cost reduction. NVIDIA CUDA/vLLM execution is not implemented; see the [project limitations](../README.md#current-limitations).
