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
| Worker-local prefill scheduler | COMPLETE | Per-worker queue with enqueue, cancel, preview, priority scan, compatibility filtering, opportunistic dispatch, deadline-aging, and queue backpressure. | `cli/worker_scheduler.ts`, `cli/worker_scheduler.test.ts` | Add decode active set only after runtime session ownership matures. |
| Request/session draft model | COMPLETE | `ModelWorkerKey`, `RequestSessionDraft`, `SessionStateHandle`, state-kind compatibility, suffix split, and accelerator placement identity. | `cli/session_state.ts`, `cli/session_state.test.ts` | Replace draft-only CLI model with runtime-owned session objects. |
| Server prefill eligibility | COMPLETE | Text-only same-worker AR batching gate rejects reloads, tools, images, PFlash, CASK, max-seq growth. | `cli/server_prefill_batch.ts`, `cli/server_prefill_batch.test.ts` | Extend only after V1 correctness is stable. |
| Request-path exclusion for streaming and unsupported Responses modes | COMPLETE | Streaming requests remain excluded; non-streaming text-only `/v1/responses` can share `generate_batch_prefill` after normalization. | `cli/server_prefill_request_path.ts`, `cli/server_prefill_request_path.test.ts`, `cli/index.ts` | Keep tools/images/streaming excluded until session output routing supports them. |
| Daemon `generate_batch_prefill` protocol | COMPLETE | Probe, validation, ready/unsupported/error handling, session-done/done events. | `crates/hipfire-runtime/examples/daemon.rs`, `cli/generate_batch_prefill_protocol.ts`, protocol tests | Keep protocol stable while expanding architectures. |
| Dense Qwen35 fused prefill microbatching | COMPLETE | Auto backend now selects `fused_dense` for eligible dense Qwen35 multi-session batches; serial reference remains explicit fallback. | `qwen35::forward_prefill_dense_session_batch`, daemon backend selection, `scripts/smoke-generate-batch-prefill.sh`, `scripts/smoke-server-prefill-batch.sh` | Broaden perf/latency coverage beyond smoke prompts. |
| Dense Qwen35 correctness gate | COMPLETE | 2/4/8 full-prompt and cached-prefix suffix cases compare greedy continuation/debug sample against serialized reference. | `scripts/smoke-generate-batch-prefill.sh` | Add richer logit-level gate if needed. |
| Server two-request coalescing | COMPLETE | Two compatible non-streaming HTTP requests coalesce, timeout fallback cleans pending waiters, daemon prefill runs, both decode with `prefill_already_done`, and runtime sessions are released. | `scripts/smoke-server-prefill-batch.sh` | Add live client-abort coverage when the harness can do it without flake. |
| Qwen35 MoE/A3B grouped fused prefill | COMPLETE | Grouped MoE candidate and fused grouped backend pass 2/4/8 MQ4 control smokes including generated-suffix replay. Auto backend selects `fused_grouped_moe` for eligible grouped-MoE candidate plans. | `qwen35::forward_prefill_grouped_moe_session_batch`, daemon backend selection, `scripts/smoke-generate-batch-prefill.sh` | Add BF16/MQ variant coverage. |
| Unsupported architecture fallback | COMPLETE | Non-Qwen35 architectures report unsupported cleanly for `generate_batch_prefill`. | `scripts/smoke-generate-batch-prefill.sh` | Add arch-specific ports when their session-state contracts exist. |
| Prefix checkpoint metadata | IN PROGRESS | Fingerprint, prefix hash, manifest, compatibility, runtime-state classification, touch, and spill guardrails exist. Metadata-only hits are counted but not reused. | `cli/state_cache.ts`, `cli/state_cache.test.ts` | Add real attach/fork for runtime checkpoints. |
| Prefix-cache health/telemetry shell | IN PROGRESS | `/health` exposes batching/cache metadata, metadata-only hits, runtime hits, and resident runtime session count when server prefill batching is enabled. | `cli/index.ts`, `cli/prefill_batch_health.ts`, `cli/prefill_batch_health.test.ts` | Add cached-token, rehydrate, and spill counters after runtime attach exists. |
| Startup accelerator inventory | COMPLETE | Server startup probes HIP GPU inventory and best-effort NPU/XDNA presence, logs counts, and exposes inventory through `/health`. | `cli/index.ts` | Replace best-effort sysfs inventory with daemon/HIP-owned placement when multi-worker residency starts. |
| Multi-model worker registry | BLOCKED | Request routing still uses one current loaded model and reloads globally. Worker-key metadata exists, but no resident worker registry. | `cli/index.ts`, `cli/session_state.ts` | Implement daemon/server model worker registry and residency policy. |
| Runtime per-session state arena | BLOCKED | Qwen35 session state exists for current generate-batch-prefill path, but there is no generic paged state arena for attention, DeltaNet, Mamba, and architecture-specific state. | `crates/hipfire-runtime/examples/daemon.rs`, `cli/session_state.ts` | Split model residency from runtime state pages across architectures. |
| Disk state-cache spill | BLOCKED | Spill eligibility metadata exists only in CLI helper code. No runtime checkpoint serialization or rehydrate path. | `cli/state_cache.ts` | Requires runtime state arena and complete prefix checkpoint snapshots. |
| Decode batching | DEFERRED | Planned but not implemented. Decode remains per request. | Source plans only | Start after prefill session-state ownership is mature. |
| MTP/DFlash verify batching | DEFERRED | Planned but intentionally disabled for multi-request batching. | Source plans only | Requires rollback/state parity tests. |
| Streaming prefill batching | DEFERRED | Streaming requests are explicitly excluded from `generate_batch_prefill`. | `cli/server_prefill_request_path.ts` | Revisit only after stream-safe session state/output routing exists. |
| `/v1/responses` prefill batching | COMPLETE FOR TEXT-ONLY NON-STREAMING | Responses input is normalized to the chat-shaped execution path, carries `prompt_cache_key` / `prompt_cache_retention`, and can use the same prefill scheduler. | `cli/index.ts`, `cli/batch_api.ts`, `cli/generate_batch_prefill_protocol.ts`, `cli/server_prefill_request_path.ts` | Streaming, tools, images, and richer Responses item types remain deferred. |
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
| Backpressure tests | COMPLETE | `HIPFIRE_SCHED_PREFILL_MAX_QUEUED` rejects new queue entries once the worker queue is full. |
| Decode active set | NOT STARTED | Deferred until after prefill path. |
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
| Multiple resident workers | BLOCKED | Server/daemon still center around one loaded model at a time. |
| Generic sequence-state arena | BLOCKED | Needed for reusable prefix cache, multi-model sessions, Mamba hybrids, and decode batching. |
| Runtime prefix checkpoint attach | BLOCKED | Metadata exists and cannot be used unless classified as attachable; no runtime page attach/copy-on-write machinery exists yet. |
| Mamba/Nemotron-H state support | BLOCKED | Requires state layout derivation and recurrent state arena. |
| Native stateful surfaces | DEFERRED | `/v1/responses`, conversations, or native sessions are additive future work. |

## Current Validation Evidence

Recent focused validation for this status:

| Command | Result | Coverage |
|---|---|---|
| `cargo fmt --package hipfire-runtime --package hipfire-arch-qwen35 --package rdna-compute --check` | PASS | Rust formatting for touched runtime/arch crates. |
| `cargo build --release -p hipfire-runtime --example daemon` | PASS | Release daemon build. |
| `cargo test -p hipfire-runtime --example daemon generate_batch_prefill_tests` | PASS | Daemon protocol/backend planning and preflight unit tests. |
| `cd cli && bun test worker_scheduler.test.ts server_prefill_batch.test.ts generate_batch_prefill_protocol.test.ts session_state.test.ts server_prefill_request_path.test.ts prefill_batch_health.test.ts` | PASS | Scheduler, session, request-path, protocol, health tests. |
| `./scripts/smoke-generate-batch-prefill.sh` | PASS | Dense 2/4/8 fused default, suffix replay, MoE serial reference 2/4/8, MoE fused grouped 2/4/8, unsupported arch. |
| `./scripts/smoke-server-prefill-batch.sh` | PASS | Timeout fallback cleans pending waiters; two HTTP non-streaming requests coalesce and use daemon `backend=fused_dense`; resident runtime sessions return to zero. |

## Recommended Next Slices

| Priority | Slice | Why |
|---|---|---|
| 1 | Add Qwen35 runtime checkpoint attach/fork or explicitly keep cache metadata-only. | The server now refuses metadata-only prefix reuse; real reuse needs a safe runtime state operation. |
| 2 | Start generic runtime `ModelWorker`/state arena design in code. | This unblocks true multi-model residency, disk spill, Mamba hybrids, and decode batching. |
| 3 | Replace CLI-only accelerator placement with daemon-owned device placement. | Worker keys are device-aware, but execution is still one selected HIP GPU. |
| 4 | Add live client-abort server coverage when practical. | Timeout cleanup is covered; abort still needs a reliable harness. |
| 5 | Add BF16/MQ variant coverage for MoE/A3B grouped fused prefill. | MQ4 control is covered at 2/4/8; dtype/format breadth remains. |

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
