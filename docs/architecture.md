# Architecture

DIN routes each complete request through **Sender → Coordinator → Provider → Backend**. Providers execute independently; the coordinator handles discovery and admission. [Setup](setup.md) covers operation, and the [README](../README.md) provides a project overview.

## Responsibilities and lifecycle

The coordinator maintains provider registrations, heartbeat freshness, model/streaming capabilities, and in-flight reservations. It selects a provider, dispatches the request, validates the response, and releases the reservation on success, failure, or cancellation. Providers enforce their own capacity and own model loading, execution, and cleanup.

An MLX provider loads the pinned cached model before registering. Each process has a boot ID, and heartbeats carry increasing sequence numbers. Identical registration retries do not renew freshness or reset reservations; changing capabilities within the same boot is rejected. A replacement boot can claim an ID only after the previous heartbeat expires and reservations reach zero. Provider endpoints must be unique.

Heartbeats normally arrive every 0.5 seconds and expire after 3 seconds, using a monotonic freshness clock. Graceful shutdown sends a best-effort not-ready notice; crashes are detected by expiry. Providers re-register after a coordinator restart. Discovery and counters are in memory, including retained stale records, and disappear when the coordinator stops.

`GET /providers` separates online status, readiness, health, and in-flight requests. `GET /v1/models` includes healthy, fresh, ready providers even when their capacity is full. Health and hardware are provider reports, not attestation; dispatch counters are not a billing ledger.

## Routing and capacity

A provider is eligible only if it supports the model and requested response mode, has a fresh heartbeat, is ready and healthy, and has capacity. Selection and reservation happen together:

- **Round robin** rotates through eligible provider IDs for each model.
- **Capacity aware** chooses the lowest `in_flight / max_concurrency` ratio, using round robin for ties.

The registry checks and increments capacity without an intervening `await`. This is atomic only within the configured **single event loop and single worker**. Replicated coordinators would need shared transactional admission.

Providers independently enforce local admission. This protects execution when coordinator accounting differs from actual work, especially during cancellation. MLX permits one active generation; excess requests are rejected immediately, with no queue. Capacity counts requests, not tokens, free VRAM, or GPU utilization. DIN does not schedule token batches or manage the engine's KV cache.

## Model execution and streaming

The MLX backend loads one pinned model/tokenizer and uses greedy generation within configured prompt/output limits. MLX operations stay on a dedicated worker thread; HTTP and heartbeats remain responsive on the event loop. Non-streaming responses aggregate the same generation path as SSE. Serving is offline, disables remote model code, and never falls back to simulation on failure.

The simulator produces explicitly labeled echo text and null usage. MLX reports actual token counters. Apple hardware memory is reported as unified memory; the compatibility field `vram_gb` does not describe dedicated or available GPU memory.

Streaming follows **assistant role → content → finish → optional usage → `[DONE]`**, with stable completion/model/provider identity. Finish is `stop` or `length`; requested usage has `choices: []`. Text chunks are not tokens. Internal transport always checks usage, even when the public client does not request it.

The provider prefetches an increment before sending headers, and the coordinator validates the upstream role event before committing. Content is forwarded immediately; finish and usage are withheld until upstream `[DONE]` and EOF validate. The SSE parser handles split UTF-8 and newline boundaries and bounds event size, total bytes, event count, and generated text.

An eight-item queue applies backpressure between the model thread and event loop. Cancellation signals the worker, closes its iterator, synchronizes native MLX work, and clears caches before returning provider capacity. Cleanup may outlast the request deadline. Repeated cancellation cannot detach cleanup, and direct backend callers must close abandoned iterators.

## HTTP contract and failures

`POST /v1/chat/completions` accepts `model`, text `messages`, optional `stream`, and streaming `stream_options.include_usage`. Messages use system/user/assistant roles and require at least one user message. Extra fields, tools, images, and sampling overrides are rejected. Responses have one choice and DIN provider metadata; full API/SDK compatibility is not claimed.

| Condition | HTTP status / code |
| --- | --- |
| Model never advertised | `404 / model_not_found` |
| Known model with no eligible capacity | `503 / no_eligible_provider` |
| Invalid request / oversized body | `422 / invalid_request`; `413 / payload_too_large` |
| Provider rejection or execution failure | `502 / provider_error` |
| Safe connection attempts exhausted | `502 / provider_connect_failed` |
| Provider / overall deadline | `504 / provider_timeout` or `request_timeout` |
| Missing/invalid credential / wrong role | `401` / `403` in LAN mode |

Errors use `{"error":{"message":"...","type":"network_error","code":"..."}}`; framework Host errors are separate. Direct provider capacity rejection is `429`; tokenized input over its limit returns `422 / prompt_too_long`. The coordinator maps both provider rejections to `502 / provider_error`.

Only connection failures **before a connection is established** are retried, excluding the failed provider and sharing one overall deadline (two attempts by default). Read/write errors, provider errors, malformed output, and committed streams may follow execution and are never replayed. Correlation IDs do not provide durable deduplication or exactly-once delivery.

Before response commitment, failures return HTTP JSON. After commitment, an SSE `event: error` is attempted and the stream closes without `[DONE]`; a broken socket may prevent even that error from arriving. Partial text is not success. The total deadline covers upstream reads and downstream sends, with a bounded error-notification grace. Streams have no keepalive event or resume protocol.

A disconnected client cancels dispatch and closes the provider connection. The coordinator may release its reservation before native cleanup finishes; provider-local admission keeps the model slot protected. A live heartbeat does not prove generation has stopped.

## SSH transport and trust

All HTTP listeners and approved provider origins use numeric loopback addresses. HTTP clients ignore proxy environment variables and redirects. Request buffering, upload duration, and inference duration are bounded separately; browser-origin writes are rejected, and logs omit prompts and credentials.

Explicit LAN mode combines OpenSSH encryption and host verification with application credentials, avoiding custom cryptography or certificate enrollment:

- A remote sender opens a local forward (`-L`) to the coordinator.
- A remote provider uses `-L` for registration/heartbeats and a reverse forward (`-R`) for coordinator dispatch to its approved loopback endpoint.
- A participant on the coordinator's Mac uses `--local-coordinator` without SSH.

SSH accepts an approved private IPv4 address, verified host keys, and public-key-only batch authentication. Proxy/jump commands and agent forwarding are disabled. [Secure setup](setup.md#secure-two-mac-setup) requires restricted keys and effective `GatewayPorts no`; the runtime reverse-port probe checks only the selected LAN interface, not every interface or firewall rule.

Application authorization runs before body buffering. Sender credentials allow inventory/inference; provider credentials are bound to one provider ID and endpoint. Senders cannot register or invoke providers directly, and a provider cannot impersonate another. LAN mode and its private profile must be enabled together.

Profiles are owner-only regular files, reject symlinks, and load at startup. Only recipient invitations are shared; the coordinator profile contains all credentials. Rotation replaces shared sender credentials. These controls assume trusted OS accounts and do not isolate hostile tenants.

Launchers stop only their own services/tunnels, including after failed startup. A joined sender owns no coordinator. Providers remain independent and must be stopped separately. A terminated SSH process requires restarting that participant's launcher.

## Future NVIDIA integration

CUDA/vLLM execution is not implemented. A Linux provider could reuse DIN's registry, schedulers, admission, and SSH transport, but needs a real backend adapter that:

- Pins a compatible GPU/driver/runtime stack and model/tokenizer revisions; the MLX quantized snapshot is not a CUDA checkpoint.
- Translates engine JSON/SSE into DIN's identity, usage, and termination contracts and establishes model readiness before registration.
- Defines engine process ownership, measures dedicated GPU memory and queue/KV-cache limits, and proves native cancellation before releasing capacity.
- Tests engine death, malformed output, out-of-memory recovery, and Linux SSH restrictions on real hardware, then repeats matched [benchmarks](benchmarking.md).

The five cases in [test_vllm_contract.py](../tests/test_vllm_contract.py) use synthetic HTTP responses only. They verify protocol assumptions, not vLLM or NVIDIA operation. Any future performance or cost claim needs measured hardware, workload, energy or billing inputs, and raw results.
