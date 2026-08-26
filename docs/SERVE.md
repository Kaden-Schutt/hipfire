# Serve (OpenAI-compatible HTTP)

`hipfire serve` starts the native Rust HTTP control plane, owns one GPU daemon
process, and exposes an OpenAI-shaped chat API. Defaults come from typed TOML
configuration ([CONFIG.md](CONFIG.md)); the HTTP surface is implemented by
`crates/hipfire-cli` and the shared transport lives in `crates/hipfire-client`.

| Field | Default (source) |
|---|---|
| Bind host | `serve.host = "0.0.0.0"` |
| Port | `serve.port = 11435` |
| Pre-warm model | `serve.default_model = "qwen3.5:9b"` or a positional model arg |
| Idle unload | `serve.idle_timeout_seconds = 300` (`0` = never) |
| Max request body | `serve.max_request_bytes = 67108864` (64 MiB) |
| Admission queue | `serve.max_queue = 64`, `serve.queue_timeout_ms = 30000` |
| Pid / log | `~/.hipfire/serve.pid`, `~/.hipfire/serve.log` |

Truth state: **shipped / ref-pinned** for the HTTP contract and lifecycle
below. Branch-only model routes (for example LFM framing details) are not
implied by this page — see [MODELS.md](MODELS.md) and [VALIDATION.md](VALIDATION.md).

## Security (no auth / no TLS)

The native handler implements **no authentication and no TLS**.
Anyone who can reach the bind address can call every endpoint, including
chat completions. Default bind is `0.0.0.0` (all interfaces).

- Prefer loopback for local use: `hipfire serve 127.0.0.1:11435`
- Expose beyond localhost only behind a trusted network or an authenticated
  TLS reverse proxy you control. Do not publish the raw port to the internet.

## Start and stop

```bash
hipfire serve                         # foreground; Ctrl-C stops (default bind 0.0.0.0)
hipfire serve 127.0.0.1:11435         # loopback-only (preferred local bind)
hipfire serve -d                      # background (setsid/nohup); polls /health up to 300s
hipfire serve qwen3.5:9b -d           # pre-warm a specific tag this run
hipfire serve 0.0.0.0:11435 -d        # all-interfaces bind — unauthenticated; see Security
hipfire serve --no-prewarm -d         # bind first; load on first request
hipfire serve --kv-mode q8 --idle-timeout 0 -d
hipfire serve --tp 2 -d               # forwards params.tp; admission resolves dense TP (HFQ) or EP (MoE)

hipfire ps                            # daemons / quantize / uploads + port state (Linux)
hipfire stop                          # SIGTERM tracked pid in serve.pid (safe default)
# Destructive — inspect first (hipfire ps, curl /health, port owner):
hipfire stop --force                  # also pkill -x daemon + fuser -k <port>/tcp
hipfire stop --all                    # --force plus pkill orphan quantize jobs
hipfire restart -d                    # stop --force semantics, then serve again
```

**Destructive lifecycle flags.** Plain `hipfire stop` only SIGTERMs the tracked
pid after ownership checks — use that by default. `--force`, `--all`, and
`restart` call `reapOrphans`: system-wide `pkill -x daemon` (exact name), and
`fuser -k <port>/tcp` which **SIGKILLs whichever process owns the port**.
`--all` also runs `pkill -f` against release quantize binaries. Before using
them, run `hipfire ps`, `curl -s http://127.0.0.1:<port>/health`, and confirm
the port owner so you do not kill an unrelated daemon or listener.

Flags (also accepted by `restart`):

| Flag | Effect |
|---|---|
| `-d` / `--detach` / `--background` | Fork detached child; log → `~/.hipfire/serve.log` |
| `--kv-mode <m>` | Sets `HIPFIRE_KV_MODE` for this run |
| `--idle-timeout <s>` | Sets `HIPFIRE_IDLE_TIMEOUT` (0..86400) |
| `--no-prewarm` | Sets `HIPFIRE_NO_PREWARM=1` |
| `--tp N` | Forwards `params.tp` (1..64) in the load message; admission resolves dense TP for admitted dense HFQ cells or EP for supported MoE cells. No `HIPFIRE_TP` env read — that variable is retired with the pre-merge CLI |

Positional forms: `[model] [host] [port]`, or `host:port` / `[ipv6]:port`.
Without a model arg, pre-warm uses `HIPFIRE_MODEL` if set, else
`cfg.default_model`.

Config and env owners for bind, idle, queue, and body limits:
[CONFIG.md](CONFIG.md), [env-vars.md](env-vars.md).

## Lifecycle

1. **Bind + pid record.** Serve writes its numeric pid to
   `~/.hipfire/serve.pid`. `stop` verifies that `/proc/<pid>/cmdline` still
   identifies a native `hipfire serve` process before signaling it.
2. **Daemon.** Spawns the Rust `daemon` binary (`crates/hipfire-daemon`) over stdio JSON.
3. **Pre-warm (default).** Loads the chosen model asynchronously. Failures log
   and leave the process serving; the model loads on the first real request.
4. **HTTP.** The native server accepts traffic. Only one generation holds the
   daemon lock at a time (bounded queue).
5. **Idle eviction.** When `idle_timeout > 0`, an interval unloads the model
   after idle seconds **and** only when no generation is in flight and the
   serve lock is free. Next request reloads.
6. **Stop.** `hipfire stop` validates pid ownership before SIGTERM. Stale reused
   pids are **not** killed; the pidfile is removed instead.

Detached readiness: parent polls `GET /health` for up to **60 seconds**.
`/health` means the native process is answering; inspect `model` and
`loading_model` to distinguish ready, loading, and unloaded state.

## Endpoints

Implemented paths (anything else → `404`):

| Method | Path | Role |
|---|---|---|
| `GET` | `/health` | Liveness JSON: `status`, `model`, `loading_model`, `pid`, `native` |
| `GET` | `/v1/models` | `{ data: [{ id }, ...] }` from local model files |
| `GET` | `/stats` | Serve telemetry: uptime, queue depth, requests served, recent decode tok/s |
| `POST` | `/v1/chat/completions` | Chat completions (stream or non-stream) |

There is **no** `/v1/completions` route in the current CLI serve.

For the sealed MQ4R reproduction, default-model handoff, and copyable Hermes
Agent / Pi custom-provider configuration, see
[`GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md).

### `GET /health`

Always `200` while the HTTP server is up. `model` is the loaded tag/path or
`null` when idle/unloaded; `loading_model` names an asynchronous pre-warm.

```bash
# Loopback example (safe default for local smoke):
curl -s http://127.0.0.1:11435/health
```

### `POST /v1/chat/completions`

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": true
  }'
```

- **`stream: true`** (typical clients): SSE `data: {chat.completion.chunk}`
  lines until `data: [DONE]`.
- **`stream: false` / omitted falsey:** single `chat.completion` JSON body.
- Oversized body → **413** before the daemon lock (Content-Length or streamed
  cap at `max_request_bytes`).
- Saturated admission queue → **503** with `Retry-After`.
- Invalid JSON body → **400**.

Request fields honored by the serve layer (non-exhaustive; sampling falls
through to per-model / registry / daemon defaults when omitted):

| Field | Notes |
|---|---|
| `model` | Tag or path; triggers reload if different from resident model |
| `messages` | OpenAI chat messages (required for useful chat) |
| `messages[].content[].image_url` | One base64 PNG/JPEG data URI for VL models; remote URLs and multiple images are rejected |
| `stream`, `stream_options.include_usage` | Streaming + optional usage on stream end |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty` | Sampling; explicit request values win, otherwise per-model TOML / registry-card values are applied |
| `presence_penalty`, `frequency_penalty` | Forwarded natively to the daemon (≥ 0); `presence_penalty` also inherits per-model / registry defaults |
| `max_tokens` | Generation cap |
| `stop` | Up to 4 strings, each ≤ 64 chars |
| `tools` | Tool definitions (with structured `messages` when Jinja chat is on) |
| `chat_template_kwargs.enable_thinking` | Mode axis. `false` forces a no-think turn (Qwen native empty closed think). Independent of effort and cap. |
| `chat_template_kwargs.preserve_thinking` | Keep `<think>` in final non-stream content |
| `thinking` / `thinking.type` | Mode axis (`on`/`off`, or DeepSeek-style `enabled`/`disabled`). Thinking disabled wins and drops effort/cap with a warning. |
| `reasoning.effort` / `reasoning_effort` | **Semantic effort only** — prompt strength. Never mapped to a think-token budget. Family ladders differ; unsupported/cross-family values are dropped with a warning (see below). |
| `reasoning.max_tokens` / `max_think_tokens` | Explicit **integer** think-span cap on Qwen Jinja contracts. `0` or omitted = uncapped. DeepSeek, Gemma, Glimmer, and unsupported contracts drop it with a warning. Independent of effort. |
| `thinking_budget` / `reasoning.budget` | Legacy **named** cap preset only on non-effort-native Qwen templates that still accept it. Dropped+warned elsewhere. |

### Reasoning request contract

Mode, effort, and cap are three independent axes — full key table and budget
map in [`CONFIG.md`](CONFIG.md). HTTP accepts the same meanings as CLI/config.

**Normalization (warn + drop, not silent reinterpretation):**

- Recognizable unsupported or cross-family values are dropped or aliased with
  `[WARN: INVALID CONFIG]`. Warnings appear in the server log and, on OpenAI
  responses, under `hipfire.reasoning` / `hipfire.config_warnings` metadata.
- Malformed types and out-of-range integers remain hard request errors.
- Effort is **never** converted into a token cap (including on Qwen3.6 and other
  non-effort-native templates).
- Qwen3.8 defaults to an uncapped think span unless the request sets a positive
  integer cap. DeepSeek V4 and Muse Glimmer expose semantic effort but no
  independent parent-defined cap; cap fields are dropped+warned.

**Examples:**

```bash
# Qwen3.8 — disable thinking natively (empty closed think block)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'

# Qwen3.8 — semantic effort only (still uncapped unless max_think_tokens is set)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "plan a refactor"}],
  "reasoning_effort": "medium"
}'

# Qwen3.6 — no native effort: effort is dropped+warned; set an explicit cap if wanted
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "reasoning_effort": "low",
  "max_think_tokens": 2048
}'

# DeepSeek V4 — mode + effort; medium/xhigh alias to high; no cap from effort
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "hi"}],
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high"
}'

# Gemma4 — boolean thinking (official default off); unsupported effort/cap dropped
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma4:31b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": true}
}'

# Muse Glimmer — always reasons via Onyx; strength dial; off is dropped+warned
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "muse-glimmer",
  "messages": [{"role": "user", "content": "hi"}],
  "reasoning_effort": "high"
}'
```

| Family | Mode | Effort | Cap |
|---|---|---|---|
| Qwen3.8 | `enable_thinking=false` → empty closed think | `low` \| `medium` \| `xhigh` (default `xhigh`) | Integer only; named budget dropped |
| Qwen3.6 | on/off | Dropped+warned (never → cap) | Named preset or integer if set |
| DeepSeek V4 | `thinking.type` enabled/disabled (default on) | `low` \| `high` \| `max` (`medium`/`xhigh`→`high`) | Integer only; default uncapped |
| Gemma4 | Boolean; default **off** | Dropped+warned | Dropped+warned unless explicit engine integer |
| Muse Glimmer | Always on (off dropped) | `low` \| `medium` \| `high` \| `xhigh` | Integer only; default uncapped |

When `messages` contains no `system` or `developer` role, the serve layer inserts `prompt.system` from a per-model TOML override or the registry card's `recommended_settings.system_prompt`. A client-supplied system/developer message always wins.

`finish_reason` values emitted to clients: `stop`, `length`, `tool_calls`.

Prefix-cache capable arches (daemon `cache_capable`, or arch allowlist
`deepseek4` / `qwen3_5` / `qwen3_5_moe`) skip per-request `reset` so multi-turn
LCP can hit. Other arches reset every request (stateless OpenAI shape).

## Auto-routing from `hipfire run`

```bash
hipfire serve -d
hipfire run qwen3.5:9b "..."                 # uses HTTP when /health is up
HIPFIRE_LOCAL=1 hipfire run qwen3.5:9b "..." # force one-shot local daemon
```

`run` probes `http://<probe-host>:<port>/health` (500 ms). Probe host maps
`0.0.0.0` / `::` → `127.0.0.1`. If serve is up, `run` POSTs
`/v1/chat/completions` and does **not** spawn a second daemon. If serve is up
but the request fails, `run` exits rather than colliding on the GPU lock.
`--json` / `--no-stream` also force local control.

Model mismatch: serve reloads to the requested model on the chat path (cold
start cost on that first switched request).

## Production smoke (GPU)

User-facing serve behavior is a **manual GPU** route — not proven by
container builds or no-GPU CI.

| Claim | Route | Owner |
|---|---|---|
| Generic serve semantics (finish reasons, empty/runaway, cache, timing hooks) | `scripts/serve_harness.py --model <path>` | [VALIDATION.md](VALIDATION.md) |
| Optional wrapper (serve + optional Redline/perf arms) | `scripts/gates.sh --model <path>` | same |

### Branch-only: LFM framing harness

**Branch-implemented** at audited `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`; **absent** from comparison base `origin/beta@202282de8759dfa6963ea5184ad2bf2b9259cef6`. Not a shipped/ref-pinned HTTP contract on this page.

| Claim | Route | Owner |
|---|---|---|
| LFM2.5 thinking / framing smoke | `scripts/serve_harness.py` with an exact `lfm2.5:*` tag | [VALIDATION.md](VALIDATION.md) |

Example (harness spawns its own serve by default on port `11520`):

```bash
python3 scripts/serve_harness.py --model ~/.hipfire/models/<file> --tag qwen3.5:9b --mode battery
```

No-GPU CI (`.github/workflows/no-gpu-ci.yml` → `scripts/no-gpu-ci.sh`) never
certifies GPU serve, model coherence, or throughput.

## Multi-process notes

- One listener per bind; second serve on the same host/port exits.
- Prefer plain `hipfire stop` when the pidfile is healthy. Reach for
  `hipfire stop --force` / `restart` only after `hipfire ps` and `/health`
  show a stuck orphan or port holder — those paths are destructive
  (system-wide `pkill -x daemon` + `fuser -k` on the port). Avoid ad-hoc
  `pkill` unless you know the exact process tree.
- `hipfire chat` starts a tracked detached native service if none is healthy;
  stop it explicitly with `hipfire stop` when the session is done.

## Logs and state files

| Path | Role |
|---|---|
| `~/.hipfire/serve.log` | Detached stdout/stderr |
| `~/.hipfire/serve.pid` | Ownership record for stop/ps/restart |
| `~/.hipfire/config.toml` | Sparse typed global configuration |
| `~/.hipfire/models.toml` | Local catalog, aliases, registry identities, and per-model overrides |
| `~/.hipfire/{config,models}.json`, `per_model_config.json` | Legacy migration inputs only |

```bash
tail -f ~/.hipfire/serve.log
```

## Related

- CLI inventory: [CLI.md](CLI.md)
- Chat TUI attach behavior: [CHAT.md](CHAT.md)
- Containers: [CONTAINER.md](CONTAINER.md)
- NixOS service: [NIXOS.md](NIXOS.md)
