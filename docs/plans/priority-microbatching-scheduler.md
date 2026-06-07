# Plan - Priority-Aware Microbatching Scheduler

## Summary

Hipfire should add a per-worker scheduler with 256 priority levels, where
priority `0` is realtime and priority `255` is opportunistic. This scheduler is
the execution policy layer for the broader multi-model/session-state serving
plan in `docs/plans/multi-model-session-state-serving.md`.

The first implementation target is text-only AR prefill microbatching. Decode
batching, MTP/DFlash verify batching, and state-cache eviction to disk are
planned as follow-up phases once per-session state isolation is in place.

External `/v1/chat/completions` remains OpenAI-compatible. Priority is optional
Hipfire metadata/config and defaults to interactive user traffic.

## Implementation Status

| Goal item | Status | Evidence | Notes |
|---|---|---|---|
| 1. 256-level policy surface + deterministic classes + env controls | DONE | `cli/scheduler_policy.ts`, `cli/scheduler_policy.test.ts` | Classes, clamp/parse logic, and env names are implemented. |
| 2. Per-worker queueing, enqueue/dequeue/cancel, policy selection | DONE | `cli/worker_scheduler.ts`, `cli/worker_scheduler.test.ts` | Priority buckets, compatibility filtering, opportunistic dispatch policy in one worker-local scheduler. |
| 3. Model/session foundations and request/session compatibility | DONE | `cli/session_state.ts`, `cli/session_state.test.ts` | Includes `ModelWorkerKey`, `RequestSessionDraft`, `SessionStateHandle`, and same-worker/state-kind compatibility. |
| 4. Server-side prefill batching integration (same-worker/session compatibility, no cross-model batching) | PARTIAL | `cli/server_prefill_batch.ts`, `cli/server_prefill_batch.test.ts`, `cli/index.ts` | Policy parsing, eligibility gate, and session adapter are implemented; runtime dispatch remains serialized. |
| 5. Prefix/state cache metadata + safety telemetry | PARTIAL | `cli/state_cache.ts`, `cli/state_cache.test.ts`, `cli/index.ts` | Fingerprint, manifest keying, compatibility, and spill guardrails exist; daemon/runtime hookup and telemetry counters not yet wired. |
| Blocker | SKIPPED | `crates/hipfire-runtime/examples/daemon.rs` lacks a session-batched protocol | True multi-request dispatch requires `generate_batch_prefill` and per-request state isolation in runtime worker before enabling microbatch execution. |

### SKIPPED Slice Notes

- Slice 4/5 true multi-request prefill batching is explicitly blocked until the daemon exposes batched prefill execution and per-session KV/state ownership. Current server code only sends metadata and preserves current serialized behavior.

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
HIPFIRE_SCHED_PREFILL_WAIT_MS_REALTIME=0
HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE=5
HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND=25
HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS=256
HIPFIRE_SCHED_STATE_CACHE_DISK=0
```

The existing `HIPFIRE_SERVER_PREFILL_BATCH*` knobs can remain as compatibility
aliases during migration, but new scheduler code should use the
`HIPFIRE_SCHED_*` names.

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

## Decode Batching Outline

Decode batching is later than prefill batching.

Decode scheduler requirements:

- one input token per active session per decode step,
- per-session logits and sampler state,
- per-session stop/filter/tool state,
- stream-safe output routing,
- cancellation-safe state release.

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

- Add decode active set and one-token microbatching.
- Preserve per-session sampler and stream state.
- Gate by correctness tests and latency measurements.

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
