# Setup and operation

Run commands from the repository root, activate `.venv` in each terminal, and use the same repository revision on participating machines. See [architecture](architecture.md) for request and failure contracts.

## Install

Use Python 3.12 (Python 3.11+ is supported):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e .
```

The base package runs the coordinator, Sender, and simulator without MLX or weights. Installation needs internet access; simulation and cached inference can run offline. The distribution identifier `nvidia-distributed-inference-prototype`, import `inference_network`, and command `din` are retained for compatibility.

## Local simulation

Start the coordinator and two providers:

```sh
DIN_BACKEND=simulator din demo
```

Wait for `Local inference network ready`, then run in another activated terminal:

```sh
python scripts/demo.py
```

The client checks routing, missing models, unready providers, controlled failures, reservation release, and heartbeat recovery. It restores healthy simulator controls afterward. Output is labeled simulated and has no token usage. Ctrl+C in the supervising terminal stops its children; a child failure also stops the remaining services.

Inspect `/health`, `/providers`, and `/v1/models` at `http://127.0.0.1:8000`. Send a request or mark provider A unready:

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"simulated-echo-v1","messages":[{"role":"user","content":"Hello network"}],"stream":false}'
curl -sS -X PUT http://127.0.0.1:8101/simulation \
  -H 'Content-Type: application/json' \
  -d '{"mode":"not_ready","delay_seconds":0.05}'
```

After the next heartbeat, requests use B. Restore A with `"mode":"healthy"`; `"mode":"error"` tests execution failure. These controls are unavailable for MLX.

Use `din demo --strategy capacity_aware` for the second scheduler. Alternate ports use `din demo --coordinator-port 9000 --provider-ports 9101 9102`, then `python scripts/demo.py --url http://127.0.0.1:9000`. For independent process tests, run `din coordinator` and separate `din provider --provider-id NAME --port PORT` commands with unique IDs/ports.

## Real model setup

MLX providers require native Apple Silicon, Apple Metal, and macOS 26+. Install the pinned extra:

```sh
python -m pip install -c requirements-dev.lock -c requirements-mlx.lock -e '.[mlx]'
```

| Setting | Value |
| --- | --- |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Revision | `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` |
| Runtime | `mlx-lm==0.32.0`, `mlx==0.32.3` |
| Sampling | Greedy, temperature 0 |
| Default prompt / output limits | 2,048 tokens including chat template / 128 generated tokens |
| Maximum configurable limits | 4,096 prompt / 512 output tokens |
| Capacity | One active generation per provider |

The interactive Provider checks hardware, runtime versions, and cache completeness, then asks before downloading missing weights. Declining stops startup. To cache the model explicitly:

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/hf download \
  mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
```

Use the normal cache, not `--local-dir`, with the same account and cache environment (`HF_HOME`, if set) for download and serving. The snapshot is about 276 MiB on disk, not a runtime memory estimate. Serving loads only the cached revision, disables remote model code, and needs no inference API key. Unsupported hardware, dependencies, or cache fail startup.

Stop other services on ports 8000/8101/8102, then run the self-contained demonstration:

```sh
HF_HUB_OFFLINE=1 python scripts/stream_demo.py
```

It launches a coordinator and two MLX providers on the same accelerator, compares streamed and JSON output, stops one provider, checks routing to the other, and verifies cleanup. It accepts the same alternate-port options as `din demo`.

For manual serving, run these in **separate terminals**, leaving `DIN_MODELS`, `DIN_MAX_CONCURRENCY`, and `DIN_DELAY_SECONDS` unset for MLX defaults:

```sh
DIN_REQUEST_TIMEOUT_SECONDS=60 din coordinator
```

```sh
DIN_BACKEND=mlx DIN_REQUEST_TIMEOUT_SECONDS=60 din provider --provider-id provider-mlx --port 8101
```

Registration follows model loading. Inspect provider `/health` on port 8101 or coordinator `/providers`, then stream:

```sh
curl -N -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen2.5-0.5B-Instruct-4bit","messages":[{"role":"user","content":"Explain what machine learning is in two sentences."}],"stream":true,"stream_options":{"include_usage":true}}'
```

Expect generated text, provider identity, `simulated: false`, actual usage, and `[DONE]`. For JSON, use `"stream":false` and omit `stream_options`. Interrupting curl cancels the request; capacity returns after native cleanup. Partial output without `[DONE]` is a failure. See the [streaming contract](architecture.md#model-execution-and-streaming).

## Sender and Provider

Run `din` (or `python -m inference_network`) for the localhost menu:

- **Sender → Create a network:** start a coordinator and enter prompts.
- **Sender → Join an existing network:** connect a client without starting another coordinator or model.
- **Provider → Qwen2.5-0.5B-Instruct-4bit:** serve the cached model after the setup above.

Sender commands are `/providers`, `/models`, `/model MODEL_ID`, and `/quit`. Provider status shows connection, readiness, and active requests; very short requests may finish between polls. Ctrl+C during a response cancels that request and returns to the prompt. `/quit`, EOF, or Ctrl+C at the prompt stops owned services/tunnels. Independent providers must be stopped in their own terminals.

## Secure two-Mac setup

The example uses a MacBook Air as Sender/coordinator and Mac Studio as provider `studio`. Install the base package on both and MLX only on providers. DIN never changes macOS Sharing, firewall rules, SSH configuration, or authorized keys.

### 1. Approve SSH access

On Air, enable **System Settings → General → Sharing → Remote Login → Only these users** for the intended local account. Full disk access is unnecessary; see [Apple's controls](https://support.apple.com/guide/mac-help/allow-a-remote-computer-to-access-your-mac-mchlp1066/mac).

Use a trusted private LAN. Keep router forwarding/public relays disabled and inbound WAN access blocked, including IPv6; do not disable the firewall. Guest Wi-Fi/client isolation may prevent connectivity. Find Air's numeric private IPv4 in Network settings (`10/8`, `172.16/12`, or `192.168/16`); hostnames/discovery are unsupported. A DHCP reservation avoids address changes.

Each remote peer needs an approved key. On Studio, reuse a suitable key or create a dedicated one **without overwriting an existing key**. Set a passphrase and load it into the local agent:

```sh
ssh-keygen -t ed25519 -f "$HOME/.ssh/din_lan_ed25519" -C din-lan
ssh-add "$HOME/.ssh/din_lan_ed25519"
```

Transfer only the `.pub` file through a trusted channel. On Air, use mode 700 for `~/.ssh`; append the public key to the intended account's `authorized_keys` (mode 600), preserving existing entries. The private key stays on Studio. Prefix Studio's key line with these restrictions, a space, then the public key:

```text
restrict,port-forwarding,permitopen="127.0.0.1:8000",permitlisten="127.0.0.1:18101",command="/usr/bin/false"
```

Match ports to the profile. This restricts TCP destinations/listeners and denies shell commands, PTY allocation, agent forwarding, and X11 forwarding. For a sender key, use a separately reserved loopback listen port; its launcher opens only `-L`. An administrator can prohibit reverse forwarding with `AllowTcpForwarding local` for a dedicated sender account, which cannot also support remote providers. See [OpenSSH key options](https://man.openbsd.org/sshd.8#AUTHORIZED_KEYS_FILE_FORMAT).

These keys cannot transfer files. Use AirDrop in person or a separately approved transfer key for invitations; unrestricted keys grant normal account access.

### 2. Verify host identity and forwarding policy

On Air, display its public host-key fingerprint:

```sh
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

On Studio, substitute the actual Air account/private IP:

```sh
ssh -F none -o HostKeyAlgorithms=ssh-ed25519 -o StrictHostKeyChecking=ask \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -N -T AIR_USER@AIR_PRIVATE_IP
```

Compare the SHA256 fingerprint directly with Air's screen **before accepting**. A successful session stays quiet; stop it with Ctrl+C. Runtime SSH uses strict host verification and public-key-only batch authentication. Never bypass an unknown/changed key; verify it on Air. Custom SSH config, proxies, and jump commands are ignored, so load nonstandard keys into `ssh-agent`.

On Air, inspect effective policy for each peer, substituting the actual connection values. This read-only command may require an administrator password:

```sh
sudo /usr/sbin/sshd -T \
  -C user=AIR_USER,addr=STUDIO_PRIVATE_IP,host=STUDIO_PRIVATE_IP,laddr=AIR_PRIVATE_IP,lport=22 \
  | grep -E '^(gatewayports|allowtcpforwarding|disableforwarding|permitopen|permitlisten) '
```

Require **`gatewayports no`**, allowed TCP forwarding, and no conflicting deny rule, including applicable `Match` rules. `GatewayPorts yes` can force wildcard reverse listeners. DIN rejects a reverse port reachable on the selected LAN IP, but that probe does not verify every interface/firewall rule. See [GatewayPorts](https://man.openbsd.org/sshd_config.5#GatewayPorts).

### 3. Create and join the network

On Air, run `din interactive --lan`, choose **Sender → Create a network**, and confirm the Air address/account, coordinator port 8000, and a new private directory (default `~/.din/network-YYYYMMDD-HHMMSS`). Keep this terminal running. The launcher prints file paths, never tokens:

| Profile | Intended recipient |
| --- | --- |
| `coordinator.json` | Air coordinator only; contains all credentials |
| `sender.json` | Approved senders; inventory and inference |
| `provider-studio.json` | Studio, ID `studio`, reverse endpoint `127.0.0.1:18101` |
| `provider-air.json` | Optional Air provider, ID `air`, endpoint `127.0.0.1:18102` |

Keep directories mode 700 and profiles mode 600, owned by the current account, regular files, and not symlinks. Transfer only each recipient's invitation through the trusted channel. Keep `coordinator.json` on Air and profiles outside the checkout; never commit them or paste tokens into chats, screenshots, or logs.

Store Studio's invitation in `$HOME/.din/invitations` with those permissions, then run:

```sh
din interactive --lan --profile "$HOME/.din/invitations/provider-studio.json"
```

Choose **Provider → Qwen**, approve any needed download, and accept local port 8101. Wait for connected/ready status. On Air, use `/providers`, `/models`, and a prompt; require provider `studio`, real model output, and actual usage.

An additional remote sender needs its own approved SSH key, verified host trust, and only `sender.json`:

```sh
din interactive --lan --profile "$HOME/.din/invitations/sender.json"
```

Choose **Sender → Join**. On Air itself, add `--local-coordinator`. For an additional Air provider, install/cache MLX and run the following with the actual network directory:

```sh
din interactive --lan --local-coordinator --profile "$HOME/.din/NETWORK/provider-air.json"
```

Choose **Provider**; it binds the approved 18102 endpoint. Each provider has capacity one. To restart an existing coordinator, use `din interactive --lan --profile "$HOME/.din/NETWORK/coordinator.json"` and choose **Sender → Create**.

### 4. Revoke access and shut down

Profiles load at startup, without hot reload. To revoke access, stop the coordinator, generate profiles in a fresh directory, and restart approved participants. All sender copies share a token, so rotate every sender invitation. Remove revoked SSH public keys from `authorized_keys` and terminate their existing sessions. Regenerate profiles if Air's address/account changes.

Stop independent providers explicitly, then quit the creating sender to stop its coordinator. A joined sender does not own the coordinator. If Remote Login is no longer needed, disable it manually after stopping tunnels. Restart a participant's launcher if its SSH process has exited.

## Physical device validation

Two-device operation remains unverified. Record both machines' date, hardware, memory, OS/Python versions, Git revision, roles, and ports. Keep private addresses, network/account names, and fingerprints in a private run record. Useful commands: `sw_vers`, `sysctl -n hw.model machdep.cpu.brand_string hw.memsize`, `python --version`, `git rev-parse HEAD`, and `git status --short`.

With Air's coordinator and only Studio's provider running, verify model activity on Studio and streamed output/usage on Air. In another Air terminal, substitute the pairing directory:

```sh
DIN_PAIRING_DIR="$HOME/.din/network-YYYYMMDD-HHMMSS"
mkdir -p benchmarks/runs/two-mac
python -m scripts.lan_smoke \
  --coordinator-url http://127.0.0.1:8000 \
  --credentials-file "$DIN_PAIRING_DIR/sender.json" \
  --expected-provider studio --host-label macbook-air \
  --network-note 'Air coordinator; Studio provider over verified SSH' \
  --output benchmarks/runs/two-mac/remote-inference.json
```

Use a new output filename for every run, with an existing parent. Results are mode 0600. The script cannot attest host separation or encryption and retains `physical_two_host_verification: false`; labels are operator statements. Preserve failures and successes, exact commands, and physical observations; redact personal metadata before sharing.

| Check | Required observation |
| --- | --- |
| Stop/restart or idle network loss | Provider becomes unavailable/offline after heartbeat expiry (default 3 seconds); no-provider requests fail; restored provider serves a fresh request. Restart the launcher if SSH exited. |
| Both providers | Start Air's provider; use `--expected-provider studio --expected-provider air --repetitions 6 --concurrency 2` with a new output file. Both physical providers serve requests and release capacity. Simultaneous submission alone does not prove overlap. |
| Independent senders | Join a second terminal and send overlapping long prompts. Two free providers may both serve; one saturated provider rejects excess work. A `--concurrency 3` smoke run supplements this but remains one client process. |
| Cancellation/failure during streaming | Interrupt after text arrives or stop the selected provider/network. No successful `[DONE]` or replay on another provider; native work and reservations eventually clear, and a fresh request succeeds. Check IDs/logs. |
| Ownership | Quitting a joined sender leaves the coordinator alive; quitting its creator stops it. Stop providers/tunnels separately and verify exit. |

Also test an unavailable model with `/model unavailable-model`, then restore the pinned ID. Follow [matched LAN measurements](benchmarking.md#matched-local-and-lan-measurements) after functional checks succeed; smoke snapshots are not continuous monitoring.

## Configuration

Set `DIN_` variables before manual service startup; explicit CLI flags override their settings. Durations are seconds. `DIN_HOST` accepts only `127.0.0.1`.

| Variable | Default / constraint |
| --- | --- |
| `DIN_STRATEGY` | `round_robin`; alternative `capacity_aware` |
| `DIN_HEARTBEAT_TTL_SECONDS`, `DIN_HEARTBEAT_INTERVAL_SECONDS` | `3`, `0.5`; interval should remain comfortably below TTL |
| `DIN_MAX_PROVIDERS`, `DIN_MAX_ATTEMPTS` | `64` (maximum 256), `2` (range 1–3) |
| `DIN_REQUEST_TIMEOUT_SECONDS`, `DIN_CONNECT_TIMEOUT_SECONDS` | `5`, `0.5`; interactive/manual MLX walkthrough uses a 60-second request deadline |
| `DIN_MAX_BODY_BYTES`, `DIN_BODY_TIMEOUT_SECONDS` | `65536`, `5` |
| `DIN_LAN_MODE`, `DIN_CREDENTIALS_FILE` | `false`, unset; enable together with the matching profile |
| `DIN_BACKEND` | `simulator`; alternative `mlx` |
| `DIN_MAX_CONCURRENCY`, `DIN_DELAY_SECONDS` | Simulator: `2`, `0.05`; MLX requires `1`, `0` |
| `DIN_MODELS` | `simulated-echo-v1` for simulation; MLX accepts only its pinned ID. Comma-separated IDs. |
| `DIN_MAX_PROMPT_TOKENS`, `DIN_MAX_OUTPUT_TOKENS` | `2048`, `128`; maxima `4096`, `512` |

`DIN_PROVIDER_ID`, `DIN_PORT`, and `DIN_COORDINATOR_URL` configure standalone providers. `DIN_ENDPOINT` must match the local port in default mode or the profile's approved coordinator-side origin in LAN mode. Environment variables alone do not establish tunnels; the interactive launcher owns SSH and constructs its settings explicitly. Use manual service commands for environment-based configuration.

## Testing and troubleshooting

Install development tools, then run:

```sh
python -m pip install -c requirements-dev.lock -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pip check
```

With the pinned model cached and SSH prerequisites available:

```sh
DIN_RUN_MLX_TESTS=1 DIN_SSH_INTEGRATION=1 HF_HUB_OFFLINE=1 python -m pytest -q
```

Real-model tests require supported Apple hardware. SSH tests require OpenSSH client tools, `/usr/sbin/sshd`, and a private IPv4 interface; they use temporary synthetic keys, an isolated agent/daemon, and owned cleanup without changing system SSH settings. Missing prerequisites produce explicit skips. Run only one real-model suite or benchmark at a time.

| Symptom | Action |
| --- | --- |
| Unknown/changed SSH key or disconnected tunnel | Verify IP/account, host fingerprint, approved key, `ssh-add -l`, Remote Login, and LAN reachability. Never bypass verification. |
| Reverse listener reachable on LAN | Stop and correct effective `GatewayPorts` policy. |
| Occupied port | Identify its owner and stop only your old instance; reverse ports must be unique. |
| Invalid profile or HTTP 401/403 | Check recipient role, owner, mode 600, regular file/no symlink, and current credentials without printing them. |
| MLX/model startup failure | Check native arm64 macOS 26+, pinned dependencies, and cache account/environment. |
| Model unavailable/full or replacement rejected | Refresh `/providers`; wait for loading, old heartbeat expiry, or native cleanup. There is no queue. |
| Interrupted output | Require `[DONE]`; allow cleanup before a new explicit request. |
| Prompt too long | Reduce input or use a supported limit; direct provider returns `422 / prompt_too_long`, coordinator `502 / provider_error`. |

## Third-party licenses

The pinned [Qwen conversion](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/raw/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3/README.md) declares Apache-2.0; see the [upstream model license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE). [MLX](https://github.com/ml-explore/mlx/blob/main/LICENSE), MLX-Metal, [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE), FastAPI, and Pydantic use MIT; HTTPX and Uvicorn use BSD-3-Clause.

No weights or dependencies are vendored. Bundled distributions must retain applicable notices, [Apache requirements](https://www.apache.org/licenses/LICENSE-2.0), and [MPL covered-source availability](https://www.mozilla.org/en-US/MPL/2.0/FAQ/). The pinned set includes MPL material in certifi/pathspec/tqdm, mixed Apache/CNRI-Python terms in regex, PSF terms in typing_extensions, and NumPy notices. No project license has been selected; dependency/model licenses do not license DIN's source.
