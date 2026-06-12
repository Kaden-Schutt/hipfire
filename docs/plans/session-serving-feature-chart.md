# Feature Chart - Session Serving and Priority Microbatching

Source plans:

- `docs/plans/priority-microbatching-scheduler.md`
- `docs/plans/multi-model-session-state-serving.md`

Status legend:

- `COMPLETE`: implemented and covered by focused tests or smoke.
- `IN PROGRESS`: partially implemented and used, but not yet complete end-to-end.
- `BLOCKED`: depends on a missing lower-level runtime/architecture capability.
- `DEFERRED`: intentionally out of the first implementation slice.
- `NOT STARTED`: planned but no meaningful implementation exists yet.

## Executive Status

| Area | Status | Current state | Primary evidence | Next work |
|---|---|---|---|---|
| Priority policy surface | COMPLETE | 256-level priority model, deterministic classes, env parsing, wait/batch/quantum policy. | `cli/scheduler_policy.ts`, `cli/scheduler_policy.test.ts` | None for V1. |
| Worker-local prefill scheduler | COMPLETE | Per-worker queue with enqueue, cancel, preview, priority scan, compatibility filtering, opportunistic dispatch, deadline-aging, queue backpressure, and bounded default pending fan-in. | `cli/worker_scheduler.ts`, `cli/worker_scheduler.test.ts` | Add disk deferral once state spill exists. |
| Request/session draft model | COMPLETE | `ModelWorkerKey`, `RequestSessionDraft`, `SessionStateHandle`, state-kind compatibility, suffix split, and accelerator placement identity. | `cli/session_state.ts`, `cli/session_state.test.ts` | Replace draft-only CLI model with runtime-owned session objects. |
| Server prefill eligibility | COMPLETE | Text-only same-worker AR batching gate rejects reloads, tools, images, PFlash, CASK, max-seq growth. | `cli/server_prefill_batch.ts`, `cli/server_prefill_batch.test.ts` | Extend only after V1 correctness is stable. |
| Request-path exclusion for streaming and unsupported Responses modes | COMPLETE | Streaming requests remain excluded; non-streaming text-only `/v1/responses` can share `generate_batch_prefill` after normalization. | `cli/server_prefill_request_path.ts`, `cli/server_prefill_request_path.test.ts`, `cli/index.ts` | Keep tools/images/streaming excluded until session output routing supports them. |
| Daemon `generate_batch_prefill` protocol | COMPLETE | Probe, validation, ready/unsupported/error handling, session-done/done events. | `crates/hipfire-daemon/src/main.rs`, `cli/generate_batch_prefill_protocol.ts`, protocol tests | Keep protocol stable while expanding architectures. |
| Dense Qwen35 fused prefill microbatching | COMPLETE | Auto backend now selects `fused_dense` for eligible dense Qwen35 multi-session batches; serial reference remains explicit fallback. | `qwen35::forward_prefill_dense_session_batch`, daemon backend selection, `tests/smoke-generate-batch-prefill.sh`, `tests/smoke-server-prefill-batch.sh` | Broaden perf/latency coverage beyond smoke prompts. |
| Dense Qwen35 correctness gate | COMPLETE | 2/4/8 full-prompt and cached-prefix suffix cases compare greedy continuation/debug sample against serialized reference. | `tests/smoke-generate-batch-prefill.sh` | Add richer logit-level gate if needed. |
| Server prefill queue flush/coalescing | COMPLETE | Singleton pending requests flush through daemon prefill after the coalesce window, compatible non-streaming HTTP requests coalesce, daemon prefill runs, both decode with `prefill_already_done`, and runtime sessions are released. | `tests/smoke-server-prefill-batch.sh` | Add live client-abort coverage when the harness can do it without flake. |
| Qwen35 MoE/A3B grouped fused prefill | COMPLETE | Grouped MoE candidate and fused grouped backend pass 2/4/8 MQ4 control smokes including generated-suffix replay. Auto backend selects `fused_grouped_moe` for eligible grouped-MoE candidate plans. | `qwen35::forward_prefill_grouped_moe_session_batch`, daemon backend selection, `tests/smoke-generate-batch-prefill.sh` | Add BF16/MQ variant coverage. |
| Unsupported architecture fallback | COMPLETE | Non-Qwen35 architectures report unsupported cleanly for `generate_batch_prefill`. | `tests/smoke-generate-batch-prefill.sh` | Add arch-specific ports when their session-state contracts exist. |
| Prefix checkpoint attach/fork | COMPLETE FOR QWEN35 V1 | Qwen35 single-GPU prefill returns attachable checkpoint handles with daemon-authoritative `xxh128` prefix hashes computed over the rendered/tokenized runtime stream. Cache lookup can preflight candidate prompt hashes through the daemon before dispatch, exact rendered-input hits attach with full-prompt retokenization, and serial Qwen35 prefill now stores intermediate semantic-boundary checkpoints for partial-prefix reuse at completed chat-template boundaries. Chat requests can opt in with `prompt_cache_retention`; `/v1/responses` follow-ups using `previous_response_id` replay the stored transcript into the same resident prefix path. Exact same-wave fanout timer-flushes pending identical cold prompts, prefills one leader, and attaches compatible followers. CLI invalidates resident manifests on reset/unload/reload/drain and on stale attach/hash mismatch failure. | `cli/state_cache.ts`, `cli/index.ts`, `crates/hipfire-daemon/src/main.rs`, `tests/smoke-server-prefix-checkpoint-reuse.sh`, `tests/smoke-server-prefix-hash-preflight.sh`, `tests/smoke-server-prefix-boundary-reuse.sh`, `tests/smoke-server-responses-prefix-reuse.sh`, `tests/smoke-server-shared-prefix-fanout.sh` | Genericize beyond Qwen35 resident checkpoints and expose equivalent boundary snapshots from fused backends through the generic state arena. |
| Prefix-cache health/telemetry shell | COMPLETE FOR QWEN35 V1 | `/health` exposes batching/cache metadata, metadata-only hits, runtime hits, resident decode sessions, resident checkpoints, resident cap, eviction/recompute counters, disk-vs-resident enablement, daemon-prefix-hash authority, semantic-boundary checkpoint presence/count, prefix-hash preflight boundary matches, Responses `previous_response_id` hit/miss/context counters, and shared-prefix fanout group/follower counters. | `cli/index.ts`, `cli/prefill_batch_health.ts`, `cli/prefill_batch_health.test.ts` | Add cached-token, rehydrate, and disk-spill counters when serialization exists. |
| Startup accelerator inventory | COMPLETE | Server startup probes HIP GPU inventory and best-effort NPU/XDNA presence, logs counts, and exposes inventory through `/health`. | `cli/index.ts` | Replace best-effort sysfs inventory with daemon/HIP-owned placement when multi-worker residency starts. |
| Multi-model worker registry | IN PROGRESS FOR V1 | One daemon can hold multiple resident `LoadedModel` workers keyed by server worker id when `HIPFIRE_MAX_RESIDENT_WORKERS>1`; requests route per worker and batching remains worker-local. `/health.runtime_workers` lists resident models and V1 memory metrics. | `cli/index.ts`, `cli/session_state.ts`, `crates/hipfire-daemon/src/main.rs`, `tests/smoke-server-multi-model-workers.sh` | Add allocator-precise memory accounting and fine-grained state/chunk eviction policy. |
| Runtime per-session state arena | IN PROGRESS | Qwen35 session state exists for current generate-batch-prefill path, top-level attach/fork/activate/reset/release/count operations route through the backend-neutral arena wrapper, fused final-checkpoint creation uses the checkpoint hook, and `/health.runtime_workers` reports Qwen35 state-page descriptor counts/bytes plus resident worker memory totals. V1 model bytes are HFQ payload/file bytes, not exact allocator-reserved bytes. | `crates/hipfire-daemon/src/main.rs`, `cli/session_state.ts`, `/health.runtime_workers` | Add allocator-precise memory accounting and replace wrapped Qwen35 maps with generic state pages across architectures. |
| Resident session memory-pressure admission | COMPLETE FOR V1 | Before prefill enqueue, the server asks the daemon to reserve session state against worker memory/descriptor telemetry and `HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB`. The request is accepted only after `reserve_session_state_done`; daemon rejection returns HTTP 503, and reservations are released on timeout/failure or after prefill materializes resident state. `/health.runtime_workers` exposes pressure counters. | `cli/index.ts`, `cli/worker_scheduler.ts`, `crates/hipfire-daemon/src/main.rs`, `scripts/stress-server-concurrency.sh` | Replace estimates with allocator-backed page reservations and add spill-to-disk deferral mode. |
| Resident state-cache cap | COMPLETE FOR QWEN35 V1 | `HIPFIRE_SERVER_PREFILL_STATE_CACHE=1` enables in-memory checkpoint reuse; `HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS` caps attachable resident checkpoints and evicts LRU handles through daemon `release_sessions`. | `cli/scheduler_policy.ts`, `cli/index.ts`, `cli/state_cache.ts` | Tune defaults after real workload measurements. |
| Disk state-cache spill | BLOCKED | Spill eligibility metadata exists only in CLI helper code. `HIPFIRE_SCHED_STATE_CACHE_DISK` remains reserved for future serialization/rehydrate and does not gate resident checkpoint reuse. | `cli/state_cache.ts`, `cli/scheduler_policy.ts` | Requires runtime state arena and complete prefix checkpoint snapshots. |
| Decode batching | IN PROGRESS FOR QWEN35 V1 | Non-streaming text-only greedy Qwen35/Qwen35-MoE requests that already passed server prefill can enter a worker-local decode active set. The daemon exposes `generate_batch_decode_step`; `serial_reference` remains the fallback oracle. `HIPFIRE_QWEN35_DECODE_BATCH=auto` now selects dense Qwen35 FP32-state `fused_dense_layer_chunked` when resident session capability checks pass, and otherwise remains serial. Dense native multi-row chunks are enabled with `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW=1`; batch-size 2/4/8, token-count 4, chunk-cap 1/2/8 parity smokes and 128-request stress pass. Qwen35-MoE `fused_grouped_moe_layer_chunked` remains explicit and now advances multi-session chunks through the grouped-MoE native row worker, with `serial_reference` available as the oracle and internal parity gate. Auto grouped-MoE promotion stays disabled until parity and latency gates pass on real artifacts. Quantized dense KV/state, compacted dense KV, streaming routing, tools/images, sampling, DFlash/MTP, and non-Qwen35 backends remain deferred. | `cli/worker_scheduler.ts`, `cli/index.ts`, `crates/hipfire-daemon/src/main.rs`, `tests/smoke-server-decode-batch.sh`, `tests/stress-server-concurrency.sh` | Run grouped-MoE decode parity/latency smokes on real Qwen35-MoE artifacts, then consider conservative auto promotion. |
| MTP/DFlash verify batching | DEFERRED | Planned but intentionally disabled for multi-request batching. | Source plans only | Requires rollback/state parity tests. |
| Streaming prefill batching | DEFERRED | Streaming requests are explicitly excluded from `generate_batch_prefill`. | `cli/server_prefill_request_path.ts` | Revisit only after stream-safe session state/output routing exists. |
| `/v1/responses` prefill batching | COMPLETE FOR TEXT-ONLY NON-STREAMING | Responses input is normalized to the chat-shaped execution path, carries `prompt_cache_key` / `prompt_cache_retention`, and can use the same prefill scheduler. Completed Responses store a bounded in-memory transcript by `response.id`; follow-up requests with `previous_response_id` prepend that transcript, opt into in-memory resident prefix reuse, and attach the shared daemon checkpoint when hashes match. | `cli/index.ts`, `cli/batch_api.ts`, `cli/generate_batch_prefill_protocol.ts`, `cli/server_prefill_request_path.ts`, `tests/smoke-server-responses-prefix-reuse.sh` | Streaming, tools, images, richer Responses item types, and durable conversation storage remain deferred. |
| Tools/images/PFlash/CASK batching | DEFERRED | Eligibility gate rejects these feature combinations. | `cli/server_prefill_batch.ts` | Revisit individually after V1 text-only AR path is stable. |
| Multi-GPU request batching | DEFERRED | Not part of V1 batching. | Source plans only | Requires per-worker multi-GPU state ownership contract. |

## Priority Microbatching Scheduler Breakdown

| Feature | Status | Notes |
|---|---|---|
| Priority classes `0..255` | COMPLETE | `0` realtime, `64` default interactive, `255` opportunistic. |
| Env/config controls | COMPLETE | `HIPFIRE_SCHED_*` controls plus legacy `HIPFIRE_SERVER_PREFILL_BATCH*` compatibility. |
| Realtime dispatch behavior | COMPLETE | Scheduler can dispatch realtime immediately/singleton according to policy tests. |
| Interactive coalescing | COMPLETE | Wait-window and max-batch behavior covered by scheduler tests and server smoke. |
| Background/opportunistic policy | COMPLETE | Opportunistic requires paired work or clear schedule in current scheduler model. |
| Per-worker compatibility filtering | COMPLETE | Compatibility is same worker key plus same state-kind set. |
| Queue preview without mutation | COMPLETE | Used for incoming-session scheduling decisions. |
| Cancellation from queue | COMPLETE | Queue-level cancel implemented and tested. |
| Deadline aging/starvation hardening | COMPLETE | `HIPFIRE_SCHED_DEADLINE_AGING_MS` lets aged work bypass an unready higher-priority bucket. |
| Backpressure tests | COMPLETE | `HIPFIRE_SCHED_PREFILL_MAX_QUEUED` defaults to 256 and rejects new queue entries once the worker queue is full; resident session memory pressure can also reject before enqueue. |
| Decode active set | IN PROGRESS | Worker-local active decode scheduler exists for non-streaming greedy Qwen35 V1; serial fallback, auto dense FP32-state native chunked, and explicit Qwen35-MoE chunked daemon step backends are wired. Dense native multi-row chunks are parity/stress-smoked; native grouped-MoE routed kernels and streaming-safe routing remain future work. |
| Verify job scheduling | NOT STARTED | Deferred until MTP/DFlash rollback parity exists. |

## Multi-Model Session-State Serving Breakdown

| Feature | Status | Notes |
|---|---|---|
| OpenAI-compatible `/v1/chat/completions` shape | COMPLETE | Public API remains stateless and compatible. |
| Request-to-session draft conversion | COMPLETE | Prompt tokens, cached prefix length, suffix tokens, priority, worker key, and state kinds are represented. |
| Same-model batching only | COMPLETE | No cross-worker batching by construction. |
| Current server queue and pending wait flow | COMPLETE | Compatible requests can wait pending and be selected into a batch. |
| Daemon session prefill continuation | COMPLETE | After prefill, decode uses `session_id` and `prefill_already_done`. |
| Qwen35 dense per-session state isolation for fused prefill | COMPLETE | Dense fused worker receives per-session rows and isolated KV/DeltaNet/logits surfaces. |
| Qwen35 MoE/A3B per-session grouped prefill | COMPLETE | Grouped expert path uses per-session row routing and grouped fused backend. |
| Generic `ModelWorker` abstraction | BLOCKED | Trait exists only in plan text; worker keys now carry placement identity but no multi-worker runtime registry exists. |
| Multiple resident workers | IN PROGRESS FOR V1 | Server/daemon can keep more than one model resident when `HIPFIRE_MAX_RESIDENT_WORKERS>1`; health lists resident model paths and V1 memory totals. |
| Generic sequence-state arena | IN PROGRESS | Single-worker scaffold now names the wrapped Qwen35 arena and routes top-level session operations through backend-neutral arena methods, but generic paged state pages are still needed for reusable prefix cache, multi-model sessions, Mamba hybrids, and decode batching. |
| Qwen35 runtime prefix checkpoint attach | COMPLETE FOR V1 | Single-GPU Qwen35 prefill checkpoints can be forked from resident daemon state for exact rendered-input cache hits. The daemon now returns, preflights, and validates canonical `xxh128` prefix hashes for attachable checkpoints, and serial Qwen35 prefill can fork semantic-boundary checkpoints for partial-prefix hits. Generic page attach/copy-on-write remains absent. |
| Mamba/Nemotron-H state support | BLOCKED | Requires state layout derivation and recurrent state arena. |
| Native stateful surfaces | IN PROGRESS | `/v1/responses.previous_response_id` now maps to bounded in-memory transcript replay plus resident prefix reuse. Conversations, durable response state, and native sessions remain additive future work. |

## Current Validation Evidence

Recent focused validation for this status:

| Command | Result | Coverage |
|---|---|---|
| `cargo fmt --package hipfire-runtime --package hipfire-arch-qwen35 --package rdna-compute --check` | PASS | Rust formatting for touched runtime/arch crates. |
| `cargo build --release -p hipfire-daemon --bin hipfire-daemon` | PASS | Release daemon build. |
| `cargo test -p hipfire-daemon --bin hipfire-daemon generate_batch_prefill_tests` | PASS | Daemon protocol/backend planning and preflight unit tests. |
| `cd cli && bun test worker_scheduler.test.ts server_prefill_batch.test.ts generate_batch_prefill_protocol.test.ts session_state.test.ts server_prefill_request_path.test.ts prefill_batch_health.test.ts` | PASS | Scheduler, session, request-path, protocol, health tests. |
| `./tests/smoke-generate-batch-prefill.sh` | PASS | Dense 2/4/8 fused default, suffix replay, MoE serial reference 2/4/8, MoE fused grouped 2/4/8, unsupported arch. |
| `./tests/smoke-server-prefill-batch.sh` | PASS | Singleton pending request timer-flushes through daemon prefill; two HTTP non-streaming requests coalesce and use daemon `backend=fused_dense`; resident runtime sessions return to zero. |
| `./tests/smoke-server-prefix-checkpoint-reuse.sh` | NEW | First HTTP request creates a daemon-hashed resident checkpoint; second identical request attaches it and reports a runtime cache hit; a corrupted-prefix-hash attach is rejected and invalidates the manifest without repeated stale-hit telemetry. |
| `./tests/smoke-server-prefix-hash-preflight.sh` | NEW | Reuses the checkpoint smoke with preflight counters required, proving cache lookup used daemon prefix-hash preflight before attach. |
| `./tests/smoke-server-prefix-boundary-reuse.sh` | NEW | First HTTP request creates semantic-boundary resident checkpoints; second extended conversation attaches the longest shared boundary through daemon prefix-hash preflight and reports a boundary-match counter. |
| `./tests/smoke-server-responses-prefix-reuse.sh` | NEW | First `/v1/responses` request stores a response transcript and resident checkpoint; second request sends only `previous_response_id` plus a query, replays the transcript, attaches the resident prefix checkpoint, and reports Responses hit telemetry. |
| `HIPFIRE_SHARED_PREFIX_REQUESTS=16 ./tests/smoke-server-shared-prefix-fanout.sh` | NEW | Sixteen identical cold Chat requests timer-flush into one selected wave, prefill one leader, attach fifteen followers from a daemon-preflight-matched leader checkpoint, and report shared-prefix fanout counters. |
| `./tests/smoke-server-decode-batch.sh` | NEW | Two concurrent non-streaming greedy no-think HTTP requests coalesce through server prefill, then run through the Qwen35 `generate_batch_decode_step` backend with decode-batch telemetry and no leaked resident decode sessions. Default mode checks two-token `serial_reference`; `HIPFIRE_QWEN35_DECODE_BATCH=fused` checks dense FP32-state `fused_dense_layer_chunked`; `HIPFIRE_QWEN35_DECODE_BATCH=fused_grouped_moe` checks explicit Qwen35-MoE grouped native chunks when paired with a grouped-MoE model. Add `HIPFIRE_DECODE_BATCH_PARITY=1` to compare fused response text against a serial baseline, and set `HIPFIRE_QWEN35_DECODE_BATCH_MAX=1` for dense or `2` for grouped-MoE to require multiple reported chunks. |

## Recommended Next Slices

| Priority | Slice | Why |
|---|---|---|
| 1 | Run grouped-MoE decode parity/latency gates on real Qwen35-MoE artifacts and decide whether auto promotion is safe. | Explicit grouped-MoE decode now uses the native grouped row worker; auto remains serial until evidence is strong. |
| 2 | Add fused executor capture points for prefill interior boundaries. | Final prefill checkpoints and serial semantic-boundary checkpoints now share one typed Qwen35 prefill checkpoint hook over the arena wrapper. Fused dense/grouped executors still need backend-native interior capture points before they can emit mid-prefill semantic-boundary snapshots. |
| 3 | Split Qwen35 wrapped state into generic state-page descriptors. | The arena wrapper is still backed by Qwen35-owned KV/DeltaNet/logits structs rather than backend-neutral pages. |

## Retired Goal File Coverage

The former root `goal.md` tracked the bridge between batch-prefill protocol,
batch/Responses request adaptation, scheduler/session execution, unsupported
fallback behavior, and `/health` capability reporting. Its live content is now
split across:

- this feature chart for current status and validation evidence,
- `docs/plans/priority-microbatching-scheduler.md` for scheduler policy,
  lifecycle evidence, and V1 scope boundaries,
- `docs/plans/multi-model-session-state-serving.md` for session-state,
  runtime ownership, and compatibility-track dependencies.

Deletion of the root file does not remove scope. The remaining open items are
the explicit `IN PROGRESS`, `BLOCKED`, `DEFERRED`, and `NOT STARTED` rows above.
