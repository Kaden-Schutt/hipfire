# Rust CLI + Server Port Plan (Bun → clap + axum)

Replaces `cli/index.ts` (10 K LOC Bun) with two new Rust crates:
`crates/hipfire-cli` (clap) and `crates/hipfire-server` (axum).
Supersedes the CLI/server section of `v1-architecture-roadmap.md`.

---

## Guiding Constraints

- **Do not touch `qwen35.rs`, `dispatch.rs`, or `daemon.rs`** during this work.
  The v1-architecture-roadmap mandates stabilize-before-extraction; this port
  operates strictly above the daemon boundary.
- The daemon JSON-over-stdin/stdout protocol is frozen. The Rust server becomes
  another client of the daemon, the same way the Bun CLI is today.
- Behaviour parity with the Bun CLI is the acceptance criterion. No new features
  during the port.
- The TUI commands (`chat`, `config`) are deferred — they are complex, TTY-bound,
  and not required to unblock server parity. They remain in Bun (or a thin shim)
  until after the server is green.

---

## What Needs Porting

### Commands (clap subcommands)

| Command      | Complexity | Notes |
|--------------|------------|-------|
| `serve`      | High       | Core deliverable — starts axum + manages daemon subprocess |
| `run`        | Medium     | Single-shot generate; reuses daemon management |
| `pull`       | Medium     | HTTP download from HF; progress bar |
| `list`       | Low        | Scan `~/.hipfire/models/`; read aliases from `models.json` |
| `ps`         | Low        | Shell out to `ps` or read `/proc` |
| `stop`       | Low        | Send SIGTERM to PID in `serve.pid` |
| `rm`         | Low        | Unlink model file |
| `diag`       | Low        | GPU/daemon probe; mostly reads |
| `quantize`   | Medium     | Spawn `hipfire-quantize`; optional HF upload |
| `sidecar-gen`| Medium     | Spawn `triattn_validate` with calibration args |
| `update`     | Medium     | `git pull` + `cargo build`; kernel precompile |
| `bench`      | Medium     | Spawn daemon; send `bench_prefill` message |
| `eval`       | Low        | Forward to `hipfire-eval` binary |
| `host-profile`| Low       | Forward to `hipfire-host-profile` binary |
| `chat`       | **Deferred** | TTY TUI; stays in Bun initially |
| `config`     | **Deferred** | TUI config editor; stays in Bun initially |
| `profile`    | Medium     | Wraps bench with kernel tracing |

### HTTP Routes (axum handlers)

| Method | Path | Notes |
|--------|------|-------|
| GET    | `/health` | Worker status, batch health, state cache stats |
| GET    | `/v1/models` | Local model list |
| POST   | `/v1/chat/completions` | Primary path; SSE streaming |
| POST   | `/v1/responses` | Chains previous_response_id; delegates to chat handler |
| GET    | `/v1/files` | List batch files |
| POST   | `/v1/files` | Upload JSONL (multipart) |
| GET    | `/v1/files/{id}` | File metadata |
| GET    | `/v1/files/{id}/content` | Download JSONL |
| DELETE | `/v1/files/{id}` | Delete file |
| GET    | `/v1/batches` | List batch jobs |
| POST   | `/v1/batches` | Create batch job |
| GET    | `/v1/batches/{id}` | Batch status |
| POST   | `/v1/batches/{id}/cancel` | Cancel batch |

### Daemon Message Types to Implement

`load`, `unload`, `unload_worker`, `ping`, `diag`, `generate`,
`generate_batch_prefill`, `generate_batch_decode_step`,
`prefix_hash_preflight`, `reserve_session_state`,
`release_session_state_reservation`, `release_sessions`, `reset`,
`bench_prefill`.

All line-delimited JSON over stdin/stdout.

---

## Crate Structure

```
crates/
  hipfire-server/        # axum HTTP server + daemon subprocess manager
    src/
      main.rs            # entrypoint for `hipfire serve` embedded mode
      lib.rs             # pub API (for hipfire-cli to embed)
      daemon/
        engine.rs        # spawn, stdin writer, stdout reader (tokio tasks)
        protocol.rs      # serde types for every daemon message/response
      routes/
        chat.rs          # POST /v1/chat/completions + /v1/responses
        models.rs        # GET /v1/models
        health.rs        # GET /health
        files.rs         # /v1/files CRUD
        batches.rs       # /v1/batches CRUD + async batch runner
      state/
        app_state.rs     # Arc<AppState>: all Maps, daemon handle, config
        worker.rs        # ResidentModelWorker, worker routing
        session.rs       # RequestSessionDraft, worker key ID generation
        state_cache.rs   # PrefixCheckpointManifest, fingerprint, eviction
        scheduler.rs     # PriorityPrefillScheduler, PriorityDecodeScheduler
        batch.rs         # BatchFileRecord, BatchJobRecord, async runner
      config.rs          # HipfireConfig, per-model overrides, validation, load/save
      model/
        discovery.rs     # findModel() 6-level fallback chain
        registry.rs      # parse registry.json, tag→filename mapping
        sidecar.rs       # triattn/dflash/mtp sidecar discovery
      error.rs           # AppError → axum IntoResponse

  hipfire-cli/           # clap binary
    src/
      main.rs            # clap command dispatch
      commands/
        serve.rs         # start server (embeds hipfire-server)
        run.rs           # single-shot generate
        pull.rs          # HF download
        list.rs          # local model list
        ps.rs, stop.rs, rm.rs, diag.rs
        quantize.rs
        sidecar_gen.rs
        update.rs
        bench.rs
        eval.rs, host_profile.rs
```

---

## Global State (`Arc<AppState>`)

All 10+ global Maps from Bun become fields on a single `AppState` struct,
wrapped in `Arc<RwLock<...>>` or `Arc<Mutex<...>>` per field.

```rust
pub struct AppState {
    pub config: RwLock<HipfireConfig>,
    pub engine: Mutex<DaemonEngine>,         // stdin/stdout to daemon
    pub resident_workers: RwLock<HashMap<String, ResidentModelWorker>>,
    pub worker_state_caches: RwLock<HashMap<String, HashMap<String, PrefixCheckpointManifest>>>,
    pub batch_files: RwLock<HashMap<String, BatchFileRecord>>,
    pub batch_jobs: RwLock<HashMap<String, BatchJobRecord>>,
    pub responses_contexts: RwLock<HashMap<String, StoredResponsesContext>>,
    pub resident_decode_sessions: RwLock<HashMap<String, ActiveDecodeSession>>,
    pub pending_prefill: RwLock<HashMap<String, PendingPrefillRequest>>,
    pub worker_prefill_schedulers: RwLock<HashMap<String, PriorityPrefillScheduler>>,
    pub worker_decode_schedulers: RwLock<HashMap<String, PriorityDecodeScheduler>>,
    pub metrics: Mutex<ServerMetrics>,
}
```

Lock ordering: always acquire `engine` last (only when sending to daemon),
never while holding any worker/session lock.

---

## Daemon I/O Architecture

The daemon subprocess is managed by a `DaemonEngine` owned by `AppState`.

```
tokio task A: reads daemon stdout line-by-line
              → routes each JSON line to a per-request oneshot channel
              → uses `id` field to demux

request handler: acquires engine lock, writes one JSON line to stdin,
                 registers its oneshot receiver, releases lock,
                 then awaits lines until done/error

streaming: response lines forwarded to axum Body stream via mpsc channel
```

The `id` field in daemon responses is the request ID registered at send
time. The demux table is `HashMap<String, mpsc::Sender<DaemonResponse>>`
protected by a Mutex separate from the engine write lock, so the reader
task never needs the write lock.

---

## Key Sharp Edges to Handle

1. **SSE streaming**: Use `axum::response::sse::Sse` with `tokio_stream`.
   Client disconnect via `futures::StreamExt::takeWhile` on
   `axum::extract::ConnectInfo` or a `tokio::select!` with the connection
   drop signal.

2. **Serialization fidelity**: Daemon protocol uses snake_case JSON.
   All `serde` structs must use `#[serde(rename_all = "snake_case")]`
   or explicit `rename` annotations. The `id` field in responses must
   match the `id` sent — test this carefully.

3. **Batch job async runner**: `runBatchJob()` in Bun runs in the
   background without blocking the POST /v1/batches response.
   Use `tokio::spawn` from the handler; share `AppState` via `Arc`.

4. **Model discovery ordering**: Port the 6-level chain exactly.
   The fuzzy filesystem walk prefers `mq4 > hf4 > legacy > mq3 > mq2lloyd
   > mq6 > hf6 > q8`. Wrong ordering changes which model loads.

5. **Config validation**: Port `validateConfigValue` switch exactly.
   Per-model overrides stored nested under `model_overrides.{tag}` in JSON.

6. **HF download**: Use `reqwest` with streaming body + `indicatif`
   progress bar. Respect `HIPFIRE_HF_TOKEN` env var for auth.

7. **Idle timeout**: Start a `tokio::time::interval` on serve start;
   on each tick, if `now - last_request_time > idle_timeout_secs`,
   send `unload` to daemon.

8. **Background serve (`serve -d`)**: `hipfire-cli serve --detach` forks
   a new process with stdin/stdout/stderr redirected to `serve.log`,
   writes PID to `serve.pid`. Use `nix::unistd::daemon()` or
   `std::process::Command` with piped stdio on Linux.

---

## Build & Dependency Plan

New dependencies (add to workspace `Cargo.toml`):

```toml
# Server
axum = { version = "0.8", features = ["macros", "multipart"] }
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.6", features = ["cors"] }
tokio-stream = "0.1"

# CLI
clap = { version = "4", features = ["derive", "cargo"] }
indicatif = "0.17"
reqwest = { version = "0.12", features = ["stream", "json"] }

# Shared
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

The existing `hipfire-runtime` dependency chain is **not** a dependency of
`hipfire-server` or `hipfire-cli`. The server is a client of the daemon
binary, not a library consumer of the runtime.

---

## Phased Delivery

### Phase A — Daemon plumbing + `/v1/chat/completions` (foundation)
1. `crates/hipfire-server`: `DaemonEngine`, `protocol.rs`, `AppState` skeleton,
   `config.rs`, `model/discovery.rs`.
2. `POST /v1/chat/completions` (non-streaming, single model, no tools, no images).
3. `hipfire-cli serve` starts axum on the configured port.
4. `hipfire-cli run <model> <prompt>` works end-to-end.
5. Gate: `hipfire run qwen3.6:35b-a3b "Hello"` produces output.

### Phase B — Full chat completions + models + health
1. SSE streaming for `/v1/chat/completions`.
2. Tool calling (pass-through to daemon `tools` field).
3. Vision (`image` field in generate).
4. `/v1/models`, `/health`.
5. `/v1/responses` (chains to chat handler).
6. Gate: OpenAI-compatible client (curl / opencode) works against hipfire-server.

### Phase C — Pull, list, ps, stop, rm, diag
Simple commands with no daemon interaction (or trivial ping/diag).
Gate: `hipfire list` and `hipfire pull` work.

### Phase D — Batch API + state cache + prefill batching
`/v1/files`, `/v1/batches`, async batch runner, state cache eviction,
prefill batch eligibility + scheduler.
Gate: batch job runs to completion.

### Phase E — Remaining commands + Bun retirement
`quantize`, `sidecar-gen`, `update`, `bench`, `eval`, `host-profile`.
Once all routes and commands pass parity testing, `cli/index.ts` is
retired. `chat` and `config` ported last or kept as thin Bun shims.

---

## Open Questions (resolve before Phase A)

1. **Where does `hipfire-server` live in the binary?** Options:
   (a) `crates/hipfire-server` is a library; `hipfire-cli` embeds it via
       `hipfire_server::serve(state).await`. No separate `hipfire-server`
       binary needed — simpler.
   (b) Separate binary, `hipfire-cli serve` exec's it.
   Recommendation: (a) — one binary, no IPC between cli and server layers.

2. **Registry format**: `registry.json` is a Bun/JSON file. Port to a
   compiled-in `include_str!` or keep as a runtime-read file?
   Recommendation: keep as runtime-read file in `cli/` dir alongside the
   binary; avoids baking stale registry into the binary.

3. **TUI shim during transition**: During Phases A–D, `chat` and `config`
   still need to work. Simplest: `hipfire-cli chat` shells out to
   `bun cli/index.ts chat` if bun is available, failing with a clear
   error if not. This decouples TUI port from server parity.
