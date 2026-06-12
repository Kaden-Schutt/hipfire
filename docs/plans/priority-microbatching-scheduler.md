# Plan - Priority-Aware Microbatching Scheduler

## Summary

Hipfire should add a per-worker scheduler with 256 priority levels, where
priority `0` is realtime and priority `255` is opportunistic. This scheduler is
the execution policy layer for the broader multi-model/session-state serving
plan in `docs/plans/multi-model-session-state-serving.md`.

The first implementation target was text-only AR prefill microbatching. The
current follow-up slice adds a Qwen35-only decode active set and daemon
`generate_batch_decode_step` protocol for non-streaming greedy no-think
requests. The default path uses a serial reference backend, and explicit
`HIPFIRE_QWEN35_DECODE_BATCH=fused` enables dense Qwen35 FP32-state native
chunked decode dispatch, while
`HIPFIRE_QWEN35_DECODE_BATCH=fused_grouped_moe` enables Qwen35-MoE chunked
dispatch through the grouped-MoE native row worker. Both explicit backends
report `HIPFIRE_QWEN35_DECODE_BATCH_MAX` telemetry; dense native multi-row chunks are enabled with
`HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW=1`. Dense Qwen35 FP32-state `auto`
promotion is capability-gated in the daemon: supported resident dense batches
route to `fused_dense_layer_chunked`, while unsupported dense state stays on
`serial_reference`. Grouped-MoE `auto` promotion stays disabled until explicit
parity and latency gates pass. MTP/DFlash verify batching and state-cache
eviction to disk remain follow-up phases.

External `/v1/chat/completions` remains OpenAI-compatible. Priority is optional
Hipfire metadata/config and defaults to interactive user traffic.

## Implementation Status

| Goal item | Status | Evidence | Notes |
|---|---|---|---|
| 1. 256-level policy surface + deterministic classes + env controls | DONE | `cli/scheduler_policy.ts`, `cli/scheduler_policy.test.ts` | Classes, clamp/parse logic, and env names are implemented. |
| 2. Per-worker queueing, enqueue/dequeue/cancel, policy selection | DONE | `cli/worker_scheduler.ts`, `cli/worker_scheduler.test.ts` | Priority buckets, compatibility filtering, opportunistic dispatch policy in one worker-local scheduler. |
| 3. Model/session foundations and request/session compatibility | DONE | `cli/session_state.ts`, `cli/session_state.test.ts` | Includes `ModelWorkerKey`, accelerator/device placement identity, `RequestSessionDraft`, `SessionStateHandle`, and same-worker/state-kind compatibility. |
| 4. Server-side prefill batching integration (same-worker/session compatibility, no cross-model batching) | DONE FOR QWEN35 | `cli/server_prefill_batch.ts`, `cli/server_prefill_batch.test.ts`, `cli/worker_scheduler.ts`, `cli/index.ts`, `crates/hipfire-daemon/src/main.rs`, `tests/smoke-generate-batch-prefill.sh`, `tests/smoke-server-prefill-batch.sh` | Policy parsing, eligibility gate, scheduler selection, session adapter, daemon `generate_batch_prefill` dispatch, Qwen35 resident state handles, session release, dense fused prefill, grouped-MoE fused prefill, and non-streaming text-only `/v1/responses` normalization are implemented. Remaining work is generic worker residency beyond Qwen35. |
| 5. Prefix/state cache metadata + safety telemetry | DONE FOR QWEN35 V1 | `cli/state_cache.ts`, `cli/state_cache.test.ts`, `cli/index.ts`, `/health.prefill_batch`, `/health.state_cache`, `tests/smoke-server-prefix-checkpoint-reuse.sh`, `tests/smoke-server-prefix-hash-preflight.sh`, `tests/smoke-server-prefix-boundary-reuse.sh`, `tests/smoke-server-responses-prefix-reuse.sh`, `tests/smoke-server-shared-prefix-fanout.sh` | Fingerprint, manifest keying, compatibility, `prompt_cache_key` namespace support, spill guardrails, metadata/runtime-hit telemetry, Qwen35 resident attach/fork, daemon-authoritative `xxh128` checkpoint identity, daemon prefix-hash preflight, lifecycle invalidation, capped in-memory checkpoint residency, serial semantic-boundary checkpoint reuse, Responses `previous_response_id` transcript replay onto resident prefix reuse, timer-flushed exact same-wave fanout for identical cold prefixes, and arena-hooked fused final checkpoints are wired. Interior fused-backend boundary snapshots still need backend-native capture points. |
| 6. Scheduler starvation/backpressure hardening | DONE | `cli/worker_scheduler.ts`, `cli/worker_scheduler.test.ts`, `cli/index.ts` | Default queue cap, resident session memory-pressure admission, and deadline-aging selection prevent unbounded queue/state growth and strict-priority starvation. |
| 7. Decode active set and dense/grouped chunked step backends | IN PROGRESS FOR QWEN35 V1 | `cli/worker_scheduler.ts`, `cli/index.ts`, `crates/hipfire-daemon/src/main.rs`, `tests/smoke-server-decode-batch.sh`, `tests/stress-server-concurrency.sh` | Non-streaming text-only greedy Qwen35 requests that already passed server prefill can batch one-token decode steps through `generate_batch_decode_step`. Dense FP32-state `auto` uses native chunked state advance and can enable multi-row chunks behind `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW=1`; unsupported dense state remains serial; explicit grouped-MoE mode uses native grouped-MoE row-worker chunks with serial parity support. Grouped-MoE auto promotion, sampling, streaming, and non-Qwen35 support remain future work. |
| Blocker | PARTIAL | `crates/hipfire-daemon/src/main.rs` implements Qwen35 fused prefill plus state-handle lifecycle and release protocol; top-level attach/fork/activate/reset/release/count and fused final-checkpoint creation route through the backend-neutral arena wrapper around the Qwen35 session map. `/health.runtime_workers` exposes descriptor counts/bytes for wrapped Qwen35 KV/DeltaNet/logits state, resident worker lists, and V1 memory totals for pressure-aware model eviction. | Allocator-precise memory accounting, backend-neutral state-page allocation, and non-Qwen35 worker-owned session arenas remain future work. |

### SKIPPED Slice Notes

- Slice 4/5 now has queueing, compatible-batch selection, Qwen35 dense fused
  prefill, Qwen35 grouped-MoE fused prefill, Qwen35 state handles, and release
  telemetry. Generic fused worker APIs for other architectures remain future
  work.

### Current Scope Boundary

The active implementation slice covers compatible same-worker text-only AR
prefill batches, plus the scheduler and telemetry scaffolding needed to share
that path with future OpenAI-style batch jobs. Multi-resident model serving is
now V1 worker-local and reports load-time HFQ payload/file bytes plus known
runtime-state bytes. `max_seq` is the logical context-window limit; Qwen35
workers report and allocate a separate `physical_cap` that can start smaller
and grow by loading a larger compatible worker up to `max_seq`. Before prefill
enqueue, the server now asks the daemon to reserve session state; only
`reserve_session_state_done` admits the request, while daemon memory-pressure
rejection returns HTTP 503. Disk-backed deferral is still future work. It does
not yet claim allocator-precise reserved-memory accounting or
fine-grained memory-pressure state/chunk eviction. It also does
not claim native fused multi-step decode state advance, MTP/DFlash verify
batching, generic cross-architecture session arenas, or disk state-cache
spill/reload. The decode batching slice currently claims only Qwen35
non-streaming greedy no-think requests through serial reference or explicit
dense FP32-state chunked daemon step backends.

Completion evidence for this plan should stay tied to observable request
lifecycle behavior:

- contract tests for `generate_batch_prefill` envelopes and stable session IDs,
- fallback tests proving deterministic rejection reasons and telemetry counting,
- mixed-mode batch tests proving invalid work does not disturb valid work,
- execution-path tests proving queued requests move through selected, prefilled,
  decoded, released, and failed/cancelled states correctly,
- `/health` tests proving selected backend, fallback reason, unsupported counters,
  state-cache counters, and resident-session counts are coherent.

## Defined Goals

1. Provide deterministic scheduling policy before adding true request
   microbatching.
2. Keep realtime traffic responsive even when background or opportunistic work
   is queued.
3. Allow opportunistic work to run only when it can improve batching efficiency
   or when the schedule is otherwise clear.
4. Make priority control both queue wait and maximum processing quantum.
5. Keep all batching per `ModelWorker`; never batch across models.
6. Batch compute only after `RequestSession` state isolation exists.
7. Plan state-cache disk eviction as generalized sequence-state spill, not
   KV-only spill.

## Priority Model

Priority is a `u8`:

```text
0      realtime
1-63   high
64-127 interactive/default
128-191 background
192-254 low/background-bulk
255    opportunistic
```

Default priority is `64`. Realtime requests dispatch immediately or after the
smallest possible coalescing window. Opportunistic requests dispatch only when
there is enough compatible paired work, unless the worker has no runnable
higher-priority work.

Priority controls:

- prefill queue order,
- maximum coalescing wait,
- target batch size/tokens,
- maximum processing quantum,
- disk-cache rehydrate preference.

## Initial Controls

```text
HIPFIRE_SCHED_PRIORITY_DEFAULT=64
HIPFIRE_SCHED_PREFILL_BATCH_MAX=8
HIPFIRE_SCHED_PREFILL_MAX_QUEUED=256
HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB=16384
HIPFIRE_SCHED_DEADLINE_AGING_MS=0
HIPFIRE_SCHED_PREFILL_WAIT_MS_REALTIME=0
HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE=5
HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND=25
HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS=256
HIPFIRE_SERVER_PREFILL_STATE_CACHE=0
HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS=4
HIPFIRE_SCHED_STATE_CACHE_DISK=0
```

The existing `HIPFIRE_SERVER_PREFILL_BATCH*` knobs can remain as compatibility
aliases during migration, but new scheduler code should use the
`HIPFIRE_SCHED_*` names.

Resident in-memory prefix checkpoint reuse is intentionally controlled by
`HIPFIRE_SERVER_PREFILL_STATE_CACHE`, not by the disk-spill flag. Requests can
also opt into the resident cache with `prompt_cache_retention=in_memory` or
`prompt_cache_retention=24h`; `24h` is accepted as OpenAI-compatible intent but
is treated as in-memory until serialization/rehydrate exists.

For Chat Completions, clients resend the full prompt and opt into resident
reuse with `prompt_cache_retention` or the global resident-cache flag. For
Responses, completed non-streaming text-only calls store a bounded in-memory
transcript keyed by `response.id`; a follow-up with `previous_response_id`
prepends that transcript, opts into in-memory reuse, and then goes through the
same daemon prefix-hash preflight and attach path as Chat. This supports the
system-prompt plus document plus multiple-query use case after the first
request has produced a resident checkpoint. Exact same-wave fanout now handles
identical cold prompts by timer-flushing the pending queue, prefilling one
leader, and attaching followers from a daemon-preflight-matched leader
checkpoint.

### Disk-Eviction Policy Note

The scheduler should eventually separate execution batch limits from state
residency limits. A selected prefill batch may be safe to execute at a larger
size than the number of completed session states we want to keep resident.

Planned distinction:

- `prefill_batch_max`: maximum sessions sent through one
  `generate_batch_prefill` call.
- `resident_state_max`: maximum completed session states kept resident after
  prefill.
- `spillable_batch_max`: larger low-priority/background/opportunistic batch
  limit when disk state-cache spill is enabled.

Disk spill should be priority-gated. Realtime, high, and default interactive
requests should remain resident-only by default. Background, bulk, and
opportunistic requests may opt into larger batches where overflow states are
eligible for disk spill or recompute. This keeps interactive latency stable
while still allowing low-priority work to harvest larger prefill batches.

Candidate future controls:

```text
HIPFIRE_SCHED_RESIDENT_STATE_MAX=8
HIPFIRE_SCHED_SPILLABLE_BATCH_MAX=32
HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY=128
```

The selected batch plan and `/health.prefill_batch` telemetry should surface
the effective resident limit, spillable session count, and disk-cache decision
so large low-priority batches are auditable rather than implicit.

This policy is for session state, not model weight modules. Chaingun routed
expert residency needs a separate model-module cache with pinned router/shared
components, hot routed experts, warm GTT/UMA expert modules, and cold disk
fallback. See [Chaingun MoE Module Layout Notes](chaingun-moe-module-layout.md).

### Resident Checkpoint Policy Note

Qwen35 V1 resident prefix reuse keeps attachable checkpoints in daemon memory
only. The CLI tracks resident checkpoint handles separately from active decode
session handles, releases decode sessions after generation, and evicts LRU
checkpoints above `HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS` with `release_sessions`.
Daemon reset, unload, reload, interrupted-generation drain, and stale attach
failures clear the corresponding CLI manifests so a dead handle is not retried
as a cache hit.

Serial Qwen35 prefill also stores intermediate resident checkpoints at
daemon-authoritative chat-template boundaries such as completed message, vision,
tool-call, and tool-response sentinels. The daemon returns these handles in
`state_handle.prefix_checkpoints[]` with canonical `xxh128` hash metadata, and
the CLI stores them as normal attachable resident manifests. `/health.state_cache`
reports `semantic_boundary_checkpoints`, `semantic_boundary_checkpoint_entries`,
and `prefix_hash_preflight_boundary_matches`. Final checkpoints, serial
semantic-boundary checkpoints, and synchronized multi-session fused dense/grouped
semantic-boundary checkpoints now route through typed Qwen35 prefill checkpoint
hooks carrying the session id, source state handle, logical token position,
boundary kind, and prefix hash. Fused boundary layouts that would leave a
single-session interior segment fall back to `serial_reference` so the serial
oracle remains the correctness path for unsupported layouts.

The daemon now routes top-level Qwen35 attach, fork, activate, reset, release,
resident-count, and logical-position operations through `SequenceStateArenaBackend`
methods, and reports descriptor counts/bytes for the wrapped Qwen35
KV/DeltaNet/logits state in `model_worker` responses and `/health.runtime_workers`.
Descriptors now carry typed page kinds, shape metadata, and a sequence-state
handle identity for active sessions and checkpoints. This does not create
allocator-owned generic state pages yet; it makes the current Qwen35-owned state
map observable and checkpointable through one operation surface so later
eviction policy and non-Qwen35 backends can share it.

This is deliberately narrower than disk spill. `HIPFIRE_SCHED_STATE_CACHE_DISK`
is reserved for future checkpoint serialization and rehydrate; it must not be
used as the in-memory resident reuse flag.

## Scheduler Data Model

```text
RequestSession {
  id,
  model_worker_key,
  priority,
  enqueue_time,
  deadline,
  prompt_suffix_tokens,
  state_handle,
  sampling_params,
  stream_sink,
  cancellation_token,
}

PriorityClassPolicy {
  coalesce_wait_ms,
  max_batch_size,
  target_pair_tokens,
  max_processing_ms,
}

WorkerScheduler {
  priority_buckets[256],
  active_decode_set,
  state_cache_pressure,
  cancellation_index,
}
```

The scheduler chooses work for one worker only. It does not own model weights,
GPU scratch, or per-session state pages.

## Prefill Microbatching V1

Eligible sessions:

- same `ModelWorker`,
- text-only,
- AR path,
- same state/KV mode,
- no tools,
- no images,
- no PFlash,
- no DFlash,
- no MTP,
- no active CASK/eviction,
- no multi-GPU request batching.

Dispatch policy:

- scan priorities from `0` to `255`,
- dispatch realtime immediately,
- allow high/interactive priorities a short coalescing wait,
- allow background priorities longer waits and larger batches,
- dispatch opportunistic only if:
  - enough compatible paired suffix tokens are available, or
  - the schedule is clear.

In this context, "schedule is clear" means there is no runnable higher-priority
work, no same-priority work with an earlier deadline, and no expected compatible
pairing opportunity before the opportunistic request's next scheduling check.

Batch construction:

- prefer compatible sessions with earliest deadlines inside the selected
  priority class,
- cap by `max_batch_size`,
- cap by worker scratch/state capacity,
- never concatenate sessions into one logical sequence,
- pass per-session state handles to worker prefill APIs.

## Decode Batching V1

Decode batching now has a conservative V1 control plane:

- a worker-local active decode scheduler,
- bounded active decode admission via `HIPFIRE_SCHED_DECODE_MAX_ACTIVE`
  (default 256),
- `/health.decode_batch` counters,
- daemon validation for `generate_batch_decode_step`,
- Qwen35/Qwen35-MoE serial reference stepping for already-prefilled sessions,
- explicit dense Qwen35 FP32-state chunked native dispatch with one token per
  session per daemon step and `HIPFIRE_QWEN35_DECODE_BATCH_MAX` as the maximum
  native chunk rows,
- release-path preservation for resident decode handles.

The dense native backend defaults to native singleton chunks for conservative
compatibility. The Qwen35 session-batched dense row worker is wired for decode
chunks with two or more rows behind `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW=1`.
Batch-size 2/4/8, token-count 4, chunk-cap 1/2/8 parity smokes now pass, and
the 128-request stress gate reports `fused_dense_layer_chunked` under `auto`
with FP32 dense state.

Eligible V1 sessions:

- Qwen35 or Qwen35-MoE,
- non-streaming `/v1/chat/completions`,
- text-only,
- greedy sampling (`temperature=0`, `top_p=1`),
- no tools,
- no images,
- no DFlash/MTP/PFlash/CASK/eviction,
- no-think assistant prefix,
- already owns a runtime session handle from server prefill.

Decode scheduler requirements:

- one input token per active session per decode step,
- per-session logits and sampler state,
- per-session stop/filter/tool state,
- stream-safe output routing,
- cancellation-safe state release.

High fan-in coverage lives in `scripts/stress-server-concurrency.sh`. The stress
gate runs 100+ non-streaming requests against one local model or two resident
local models, then asserts that pending prefill, active decode, and resident
decode-session counters return to idle.

The default daemon backend is intentionally named `serial_reference`. It proves
protocol, scheduler, session ownership, stop handling, response assembly, and
cleanup. `HIPFIRE_QWEN35_DECODE_BATCH=auto` now selects
`fused_dense_layer_chunked` for capability-compatible dense Qwen35 FP32-state
batches. The backend honors `HIPFIRE_QWEN35_DECODE_BATCH_MAX` by splitting
selected sessions into bounded chunks and reporting `chunk_count` / effective
`chunk_size`; with `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW=1`, multi-row dense
chunks advance state through the native dense row worker. Explicit
`HIPFIRE_QWEN35_DECODE_BATCH=fused_grouped_moe` uses the same bounded chunk
telemetry and advances Qwen35-MoE chunks through the grouped-MoE native row
worker, while `auto` keeps grouped-MoE on `serial_reference` until parity and
latency gates pass. Quantized dense KV/state and compacted KV remain on
`serial_reference`.

Priority policy is stricter for decode than prefill:

- realtime and interactive sessions should not wait behind large background
  decode batches,
- max processing quantum matters more than batch size,
- opportunistic decode runs only when it can pair with compatible decode work
  or the worker is idle.

## MTP And DFlash Verify Outline

MTP/DFlash verify jobs are internal scheduler work items linked to a parent
session.

Rules:

- parent session priority propagates to verify jobs,
- verify batching stays disabled until AR prefill/decode coherence is proven,
- verify rollback/state repair must be per-session,
- DFlash/MTP must not share a verify batch across sessions until first-token
  parity and rollback parity tests exist,
- MoE verify follows MQ4 control first before expanding MQ3/MQ6 or Paro lanes.

The scheduler can eventually batch verify spines because existing code already
has internal batched verify machinery, but multi-request verify batching must
wait for per-session state handles.

## State Cache Disk Eviction Outline

State-cache eviction to disk should spill generalized sequence-state
checkpoints, not KV alone.

Spillable objects:

- inactive prefix checkpoints,
- unpinned attention KV pages,
- unpinned DeltaNet recurrent snapshots,
- unpinned Mamba SSM snapshots,
- unpinned Mamba conv-state snapshots.

Not spillable:

- active session state,
- pages pinned by an in-flight batch,
- state with unknown architecture compatibility,
- state lacking fingerprint metadata.

Disk checkpoint metadata must include:

- model artifact digest,
- architecture id,
- tokenizer hash,
- chat template hash,
- runtime config fingerprint,
- state quantization mode,
- prefix token hash and length,
- state kind manifest,
- checksum for each spilled blob.

Realtime requests should prefer recompute over cold disk restore unless the
state is already being rehydrated. Background and opportunistic requests may
wait for disk restore to improve reuse.

## Implementation Slices

### Slice 1 - Policy Module

- Add pure priority parsing and policy helpers.
- Add no-GPU tests for priority classes, waits, quanta, and opportunistic
  dispatch rules.
- Do not change serve behavior yet.

### Slice 2 - Scheduler Skeleton

- Add per-worker queue types.
- Add enqueue/dequeue/cancel tests.
- Route current single-worker serve path through the scheduler in serialized
  mode.

### Slice 3 - Prefill Queue Integration

- Add worker prefill queues.
- Use policy helpers to choose batches.
- Keep batching disabled by default.
- Emit batch telemetry.

### Slice 4 - True Prefill Microbatching

- Require `RequestSession` state handles.
- Batch same-worker AR text suffixes.
- Start with Qwen35 dense/A3B.
- Preserve serialized fallback for all incompatible requests.

### Slice 5 - Priority And Fairness Hardening

- Add starvation prevention.
- Add deadline aging for background work.
- Add opportunistic "schedule clear" checks.
- Add cancellation and backpressure tests.

### Slice 6 - Decode Batching

- Add decode active set and one-token microbatching. DONE FOR QWEN35 SERIAL V1.
- Preserve per-session runtime state handles and release semantics. DONE FOR
  QWEN35 SERIAL V1.
- Add explicit dense FP32-state chunked decode backend. DONE FOR DENSE QWEN35
  V1 CONTROL PATH.
- Add explicit grouped-MoE chunked decode backend. DONE FOR QWEN35-MOE V1
  CONTROL PATH.
- Broaden dense native perf measurements beyond smoke prompts. NEXT.
- Promote dense fused decode to `auto`. DONE FOR DENSE QWEN35 FP32 V1.
- Run grouped-MoE decode parity/latency gates on real Qwen35-MoE artifacts before auto promotion. NEXT.
- Add per-session sampler and stream state. DEFERRED.
- Gate fused backend by correctness tests and latency measurements. NEXT.

### Slice 7 - Verify Batching

- Add internal job kind for MTP/DFlash verify.
- Propagate parent priority.
- Keep disabled until rollback/state parity tests pass.

### Slice 8 - State Cache Disk Spill

- Add disk checkpoint manifest.
- Spill inactive prefix checkpoints only.
- Rehydrate for background/opportunistic first.
- Keep disabled by default.

## Test Plan

No-GPU tests:

- all 256 priorities map to deterministic classes,
- invalid priorities clamp to `0..255`,
- realtime wait is zero,
- interactive default priority is `64`,
- opportunistic requires paired work unless schedule is clear,
- max processing quantum decreases with urgency,
- cancellation removes queued sessions.

GPU tests after session-state refactor:

- 2/4/8 compatible prefill requests batch and complete independently,
- first-token logits or greedy first token matches serialized path,
- Qwen35 A3B MQ4 grouped MoE path remains active on supported RDNA,
- two compatible no-think greedy decode requests enter
  `generate_batch_decode_step`, report decode-batch telemetry, and release all
  resident decode sessions,
- the same decode smoke passes with `HIPFIRE_QWEN35_DECODE_BATCH=fused` and
  reports `backend=fused_dense_layer_chunked`,
- the fused decode smoke passes with `HIPFIRE_DECODE_BATCH_PARITY=1`, comparing
  response text against a serial baseline,
- mixed-priority batches preserve realtime latency.

State-cache tests:

- memory prefix hit parity,
- disk spill/reload parity,
- active pages are never evicted,
- Qwen35 DeltaNet state is restored with KV,
- future Mamba state restores SSM plus conv state.

## Assumptions

- Priority `0` is highest and `255` is lowest.
- Priority default is `64`.
- "Maximum processing time" means scheduler wait budget plus per-dispatch
  compute quantum, not total HTTP request timeout.
- Prefill microbatching lands before decode batching.
- MTP/DFlash verify batching remains deferred until per-session rollback is
  proven.
- Disk state-cache eviction remains disabled by default until parity tests pass.
