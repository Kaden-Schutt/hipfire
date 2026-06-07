# Plan - Multi-Model Serving, Session State, Prefix Cache, and Batching

## Summary

Hipfire's current OpenAI-compatible serve path is built around one loaded
model and one mutable conversation state. That works for serialized
single-request serving, but it is the wrong foundation for:

- concurrent `/v1/chat/completions` requests,
- prefill microbatching,
- decode batching,
- multi-model serving,
- reusable prefix/KV caching,
- hybrid recurrent models with DeltaNet or Mamba state,
- future MTP/DFlash batching.

The required structural change is to split **model residency** from
**request/session state**.

## Implementation Status

| Goal item | Status | Evidence | Notes |
|---|---|---|---|
| 1) Keep OpenAI-compatible request shape | DONE | `cli/index.ts` existing request parsing and response path | No external API change introduced by these slices. |
| 2) multi-model worker selection by request model | SKIPPED | `cli/index.ts` still routes to a single currently-loaded model path | Current model routing is `findModel(...)` + global reload; per-worker registry not yet added in daemon architecture. |
| 3) `RequestSession` extraction with request/session state handles | DONE | `cli/session_state.ts`, `cli/session_state.test.ts`, `cli/server_prefill_batch.ts` | Session drafts now carry worker keys, priorities, suffix split, and state-kinds. |
| 4) Per-session state arena (attention + recurrent + Mamba) | SKIPPED | Runtime model state remains process-global in `crates/hipfire-runtime/examples/daemon.rs` | Requires splitting seq_pos/conversation/KV/DN/mamba into per-session handles. |
| 5) Prefix cache foundations (token/state manifest + compatibility) | PARTIAL | `cli/state_cache.ts`, `cli/state_cache.test.ts` | In-memory fingerprint/caching metadata is present; runtime cache index/integration still pending. |
| 6) Same-model prefill batching integration | PARTIAL | `cli/server_prefill_batch.ts`, `cli/index.ts` | Eligibility + metadata is emitted, but no dispatch path to batched daemon call yet. |
| 7) No-cross-model batching guarantees | DONE | `sessionsCompatibleForPrefill` in `cli/session_state.ts` | Compatibility uses `ModelWorkerKey` identity and state kind equality. |
| 8) State cache spill/touch telemetry | SKIPPED | `cli/state_cache.ts` defines spill guardrails only | End-to-end telemetry emission not wired to request loop yet. |
| Blocker | SKIPPED | `crates/hipfire-runtime/examples/daemon.rs` has only single-request `generate` flow | `generate_batch_prefill` protocol and session API do not exist. |

### SKIPPED Slice Notes

- Slice 2/3 (single-worker registry + state arena) are intentionally blocked by missing runtime session state ownership in daemon.
- Slice 5 (runtime prefix cache) cannot be completed end-to-end until daemon receives per-request state handles and prefix manifest attachment.
- Slice 6 (worker-local prefill microbatching) is deferred until the above runtime primitives are available.

External API compatibility should stay simple: `/v1/chat/completions`
remains stateless and OpenAI-compatible. Internally, every request becomes a
`RequestSession` routed to a `ModelWorker`. Workers batch compatible sessions
while preserving isolated per-request state.

## Problem

Today the Qwen35 daemon path stores mutable serving state on the loaded model:

- one `seq_pos`,
- one `conversation_tokens`,
- one KV cache,
- one DeltaNet state,
- one scratch/logits surface,
- one generation stream in flight.

This makes true concurrent prefill unsafe. Concatenating prompt suffixes into
one `forward_prefill_batch` call would cause attention, RoPE position, DeltaNet
state, sampler history, and output accounting to bleed across requests.

Newer model families also make "KV cache" too narrow as the central concept.
For example, `stale/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.config.json`
declares a hybrid layer schedule with `mamba`, `moe`, and periodic
`attention` blocks, plus Mamba-specific recurrent state (`conv_kernel`,
`mamba_num_heads`, `mamba_head_dim`, `ssm_state_size`). Reusing attention KV
without the matching Mamba recurrent state would be silent corruption.

The serving core should therefore manage **sequence state**, not just KV.

## Goals

1. Keep Chat Completions externally stateless and OpenAI-compatible.
2. Support multiple resident model workers selected by request `model`.
3. Represent each in-flight request as an isolated `RequestSession`.
4. Add a generic per-session state arena that can hold:
   - attention KV pages,
   - DeltaNet recurrent state,
   - Mamba SSM state,
   - Mamba convolution state,
   - architecture-specific position metadata.
5. Add prefix caching over complete architecture state checkpoints, not just
   attention KV.
6. Enable same-model prefill batching over request suffixes.
7. Preserve existing fallbacks and experimental feature gates.
8. Make correctness easier to test than the current single mutable-state path.

## Non-Goals For The First Implementation

- No public session API requirement.
- No batching across different model workers.
- No batching across incompatible runtime features.
- No MTP/DFlash batching in v1.
- No tools/images/PFlash/CASK/eviction batching in v1.
- No multi-GPU request batching in v1.
- No attempt to support every architecture in the first slice.

## External API Shape

### `/v1/chat/completions`

Keep the OpenAI-compatible request contract:

- client sends `model`,
- client sends full `messages`,
- server returns one completion or an SSE stream,
- no client-visible session id is required.

Internally, the server may use prefix-cache hits and per-request sessions, but
the client does not need to know.

Usage accounting should report cache reuse when available:

```json
{
  "usage": {
    "prompt_tokens": 4096,
    "completion_tokens": 128,
    "total_tokens": 4224,
    "prompt_tokens_details": {
      "cached_tokens": 3072
    }
  }
}
```

### Optional Native Surface Later

After the internal architecture is proven, Hipfire can add a native stateful
surface such as:

- `/v1/responses`,
- `/v1/conversations`,
- `/v1/hipfire/sessions`.

That should be additive. OpenAI-compatible Chat Completions must not depend on
it.

## Core Architecture

```text
HTTP server
  -> RequestRouter
       selects ModelWorker by model/config
  -> RequestSession
       owns prompt tokens, state refs, sampler state, stream sink
  -> WorkerScheduler
       forms compatible prefill/decode batches
  -> ModelWorker
       owns weights, tokenizer, GPU handles, scratch pools, cache arenas
```

### ModelRegistry

Tracks resident and loadable workers.

Responsibilities:

- resolve model tags to artifact paths,
- normalize per-model config,
- reuse already loaded workers,
- load/unload workers based on VRAM policy,
- expose worker health and model metadata,
- prevent duplicate loads of byte-identical artifacts.

Suggested key:

```text
ModelWorkerKey {
  artifact_path,
  artifact_digest,
  arch_id,
  quant_family,
  kv_mode,
  max_seq_bucket,
  feature_flags
}
```

Do not include request sampling params in the worker key; those belong to
`RequestSession`.

### ModelWorker

Owns model-resident resources:

- weights,
- tokenizer and chat template metadata,
- architecture config,
- GPU handle(s),
- scratch pools,
- sequence-state arenas,
- prefix-cache index,
- scheduler queue,
- telemetry counters.

Minimal trait shape:

```rust
trait ModelWorker {
    fn model_id(&self) -> &str;
    fn architecture(&self) -> ArchitectureKind;
    fn prepare_session(&mut self, request: GenerateRequest) -> Result<RequestSession>;
    fn prefill_suffix_batch(&mut self, sessions: &mut [&mut RequestSession]) -> Result<()>;
    fn decode_batch(&mut self, sessions: &mut [&mut RequestSession]) -> Result<()>;
    fn release_session(&mut self, session: RequestSession);
}
```

The first implementation can keep `decode_batch` as a loop over sessions while
still moving all state ownership to `RequestSession`.

### RequestSession

Owns one client request.

Fields:

```text
RequestSession {
  id,
  model_key,
  rendered_prompt_tokens,
  cached_prefix_len,
  suffix_tokens,
  logical_position,
  max_tokens,
  sampling_params,
  state_handle,
  sampler_state,
  output_filter_state,
  stream_sink,
  cancellation_token,
  timing,
}
```

State must be isolated even when compute is batched.

### SequenceState

This is the central abstraction. "KV cache" is one state kind, not the whole
system.

```text
SequenceState {
  attention_kv_pages,
  deltanet_recurrent_pages,
  mamba_ssm_pages,
  mamba_conv_pages,
  position_metadata,
  token_history,
}
```

Architecture-specific workers decide which fields are populated.

### LayerStateKind

Each architecture should declare its persistent state layout from config:

```text
LayerStateKind:
  AttentionKv
  DeltaNetRecurrent
  MambaRecurrent
  StatelessMoe
  StatelessMlp
  OtherArchitectureSpecific
```

Examples:

- Qwen35 dense/A3B:
  - FullAttention layers: `AttentionKv`
  - DeltaNet layers: `DeltaNetRecurrent`
  - MoE FFN: `StatelessMoe`
- Nemotron-H style:
  - attention blocks: `AttentionKv`
  - mamba blocks: `MambaRecurrent`
  - moe blocks: `StatelessMoe`
- Plain transformer:
  - all attention layers: `AttentionKv`
  - MLP/MoE layers: stateless between tokens.

## Prefix Cache

### Principle

A prefix-cache hit is valid only when **all state needed at the prefix boundary**
can be reused.

For attention-only models, that mostly means KV pages plus position metadata.
For Qwen35, it also means DeltaNet recurrent state. For Mamba hybrids, it means
Mamba SSM and convolution state. Reusing only KV is wrong for recurrent
architectures.

### PrefixCheckpoint

```text
PrefixCheckpoint {
  model_fingerprint,
  tokenizer_fingerprint,
  chat_template_fingerprint,
  runtime_config_fingerprint,
  token_prefix_hash,
  prefix_len,
  state_refs,
  rope_or_position_metadata,
  created_at,
  last_used_at,
  hit_count,
  bytes,
}
```

`state_refs` can include:

- attention KV page refs,
- DeltaNet state snapshot refs,
- Mamba SSM state snapshot refs,
- Mamba conv state snapshot refs.

### Cache Lookup

1. Render request messages into tokens.
2. Compute rolling hashes over token prefixes.
3. Find the longest compatible checkpoint.
4. Attach or copy-on-write the checkpoint state into the session.
5. Prefill only `tokens[cached_prefix_len..]`.
6. Report `cached_tokens`.

### Compatibility Key

Prefix checkpoints must match:

- model artifact digest,
- architecture id,
- tokenizer hash,
- chat template hash,
- prompt normalization settings,
- KV/state quantization mode,
- RoPE/position policy,
- feature flags affecting forward semantics.

Sampling params do not affect prefix cache validity.

### Refcounting

Prefix pages should be refcounted:

- active sessions pin pages,
- prefix-cache entries hold weak or strong refs according to policy,
- eviction cannot reclaim pinned pages,
- copy-on-write is required if a session mutates a shared page.

## State Arena Design

Use paged arenas per worker, not per request allocations.

```text
StateArena {
  attention_kv_pool,
  recurrent_state_pool,
  conv_state_pool,
  page_table,
  free_lists,
  refcounts,
  lru_metadata,
}
```

The arena must support:

- allocate session state,
- attach prefix checkpoint,
- clone/copy recurrent snapshots,
- free session pages,
- evict inactive prefix pages,
- compact or reuse pages without changing logical positions.

## Batching Model

### Prefill Batching

Batch suffix compute across compatible sessions:

```text
prefill_suffix_batch([session_a, session_b, ...])
```

The input is ragged:

```text
[
  { session_id, suffix_tokens, start_pos, state_handle },
  ...
]
```

The worker may process this as:

- one ragged batch if kernels support it,
- bucketed chunks by length,
- a loop initially, preserving the public scheduler contract.

For Qwen35 A3B, grouped MoE prefill can then batch routed expert slots across
sessions once the forward path accepts per-session state handles.

### Decode Batching

Decode batching is a later stage. It requires:

- one next-token input per active session,
- per-session logits,
- per-session sampler state,
- per-session stop/filter/tool state,
- stream-safe output routing.

Do not block prefill batching on decode batching.

### Compatibility Rules For V1

Eligible:

- same `ModelWorker`,
- text-only,
- AR path,
- same KV/state quantization mode,
- no active eviction/CASK,
- no PFlash,
- no DFlash,
- no MTP,
- no tools,
- no images,
- no multi-GPU request batching.

Ineligible requests keep the existing serialized path or run through the same
worker one at a time.

## Multi-Model Serving

### Worker Residency

Multiple models can be resident if VRAM allows:

```text
qwen3.6:27b        -> Worker A
qwen3.6:35b-a3b    -> Worker B
qwen3.5:9b         -> Worker C
```

Requests route by `body.model`.

Batching never crosses workers. This keeps architecture-specific state and
kernel assumptions local to the worker.

### VRAM Admission Policy

Start with a conservative policy:

- keep the default model resident,
- load requested model if enough free VRAM remains,
- unload idle workers by LRU if needed,
- reject with a clear error if the requested model cannot fit,
- never unload a worker with active sessions.

Later policy improvements:

- model pinning,
- max resident model count,
- per-model idle timeout,
- weighted LRU by load cost and hit rate,
- background prewarm.

## HTTP Server Scheduling

The Bun server should not hold one global FIFO lock forever. Replace it with:

```text
RequestRouter:
  parse request
  resolve model/config
  get ModelWorker
  create RequestSession
  enqueue session on worker scheduler
```

Each worker scheduler owns:

- pending prefill queue,
- active decode set,
- cancellation handling,
- microbatch timer,
- max batch size,
- backpressure.

Initial controls:

```text
HIPFIRE_SERVER_PREFILL_BATCH=0|1      default 0
HIPFIRE_SERVER_PREFILL_BATCH_MAX=8    default 8
HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS=5 default 5
HIPFIRE_SERVER_MAX_RESIDENT_MODELS=1  default 1 initially
HIPFIRE_SERVER_PREFIX_CACHE=0|1       default 0 initially
HIPFIRE_SERVER_PREFIX_CACHE_MB=<N>    default conservative
```

## Daemon Protocol

The current JSON-lines daemon protocol can evolve in two steps.

### Scheduler-to-Daemon Contract (Scaffolded `generate_batch_prefill`)

The scheduler currently targets a scaffold-only `generate_batch_prefill` protocol
to avoid GPU hot-path changes in this goal.

- **Request envelope** (`type: "generate_batch_prefill"`):
  - `id` (optional): client correlation ID returned on protocol responses
  - `batch_id` (required): non-empty batch identifier generated by the scheduler
  - `worker_key_id` (preferred) or `model` (required at least one): worker
    identity for batch compatibility checks
    - `worker_key_id` is the canonical worker identity string generated from:
      `artifact_path` or `artifact_digest`, `arch_id`, `quant_family`,
      `state_mode`, `max_seq_bucket`, and sorted `feature_flags`
      (the same fields consumed by `modelWorkerKeyId` when batching is built in
      CLI).
    - If `worker_key_id` is unavailable, `model` may be used as a fallback
      minimum identity check, but this cannot encode scheduler-only constraints
      (e.g. state-kind compatibility) as precisely.
  - `sessions` (required): non-empty array of per-request sessions

- **Session payload (`sessions[]`)**:
  - `id` (required): session/request identifier, non-empty and unique inside the
    same batch
  - exactly one of:
    - `prompt` (string), or
    - `suffix_tokens` (`u32[]`, non-empty)
  - `state_handle` (required):
    - `state_kinds` (`string[]`, non-empty) from:
      `attention_kv`, `deltanet_recurrent`, `mamba_ssm`, `mamba_conv`,
      `architecture_specific`
    - `logical_position` (integer >= 0)
    - `cached_prefix_tokens` (integer >= 0, optional)
  - `params` (optional object): implementation-defined sampling/dispatch params

- **Scheduler eligibility checks (before envelope creation)**:
  - text-only request: no tools and no image content
  - same worker state (model/artifact/arch/quant/state mode/flags)
  - no active features that are excluded from V1 prefill batching
    (`pflash`, `cask`, unsupported max-seq growth, etc.)
  - request-level routing compatibility with current loaded worker

- **Fallback/unsupported metadata path (currently active behavior)**:
  - Ineligible or unsupported requests are still sent through
    serialized `generate`.
  - `generate_batch_prefill_runtime` dispatch reasons are surfaced via
    `genParams.server_prefill_batch` and `/health` as:
    - `queue_wait_reason` (`selected|waiting|insufficient_queue|not_eligible|disabled`)
    - `fallback_reason`
    - `selected_batch_size`
    - `runtime_dispatch_skipped_reason`
  - `/health` publishes aggregate counters and last observed queue/runtime/fallback
    values through `prefill_batch`.

- **Unsupported contract response shape** (daemon-side scaffold):

```json
{
  "type": "generate_batch_prefill_unsupported",
  "id": "<request id or 0>",
  "batch_id": "<batch id>",
  "sessions": 0,
  "supported": false,
  "reason": "per_session_runtime_state_unavailable"
}
```

CLI interprets this as a stable capability state and keeps execution serialized.

### Step 1 - Worker-Aware Single Generate

Keep existing request shape, but route through a session internally:

```json
{"type":"generate","id":"r1","prompt":"...","model":"..."}
```

Daemon creates a `RequestSession`, runs one-session prefill/decode, releases
the session.

### Step 2 - Batched Prefill Message

Add an internal message for the Bun server or future in-process scheduler:

```json
{
  "type": "generate_batch_prefill",
  "requests": [
    {
      "id": "r1",
      "prompt": "...",
      "system": "...",
      "params": { "temperature": 0.0, "max_tokens": 128 }
    }
  ]
}
```

Response stream:

```json
{"type":"prefill_done","id":"r1","prompt_tokens":1024,"cached_tokens":768,"prefill_ms":12.3}
{"type":"token","id":"r1","text":"..."}
{"type":"done","id":"r1",...}
```

This message must not batch by concatenating prompt tokens into one sequence.
It must create separate `RequestSession` objects and call worker batch APIs.

## Architecture-Specific Work

### Qwen35 Dense/A3B

Required:

- move `seq_pos`, `conversation_tokens`, KV, DeltaNet state into
  `RequestSession`,
- keep weights/scratch pools on `ModelWorker`,
- teach `forward_prefill_batch` or a sibling API to accept per-session state,
- preserve grouped MoE path:
  - scatter routed slots by expert,
  - grouped gate/up GEMM,
  - unscatter SwiGLU inputs,
  - grouped down GEMM,
  - weighted combine into the correct session row.

Initial state layout:

```text
Qwen35SessionState {
  kv_cache_handle,
  deltanet_state_handle,
  seq_pos,
  compact_offset,
  conversation_tokens,
}
```

### Nemotron-H / Mamba-Hybrid

Required before first-class support:

- parse `layers_block_type`,
- map each layer to `LayerStateKind`,
- define Mamba recurrent state allocation:
  - SSM state: `[layers, heads, ssm_state_size, ...]` per session,
  - conv state: `[layers, conv_kernel - 1, hidden/expanded channels]`,
- snapshot recurrent state at prefix checkpoints,
- validate prefill split equivalence.

The config example has:

- `hidden_size = 8192`,
- `layers_block_type` with `mamba`, `moe`, and `attention`,
- `mamba_num_heads = 256`,
- `mamba_head_dim = 64`,
- `ssm_state_size = 128`,
- `conv_kernel = 4`,
- `num_key_value_heads = 2`,
- `num_experts_per_tok = 22`,
- `n_routed_experts = 512`.

That means persistent session state is dominated by recurrent Mamba state plus
attention KV for periodic attention layers. MoE routing is compute-heavy but
not persistent between tokens.

## Correctness Tests

### No-GPU Tests

- state-layout derivation from config:
  - Qwen35,
  - Qwen35-MoE,
  - Nemotron-H example,
  - plain transformer.
- prefix-cache compatibility key equality/inequality.
- longest-prefix lookup.
- refcount pin/unpin semantics.
- batching eligibility:
  - same model accepted,
  - different model rejected,
  - tools/images/PFlash/DFlash/CASK rejected,
  - max-seq reload rejected.

### GPU Microtests

For each architecture:

1. Full prefill in one call.
2. Prefix prefill + suffix prefill.
3. Prefix checkpoint attach + suffix prefill.
4. Compare first-token logits or greedy first token.
5. Decode a short continuation.

For recurrent architectures, include split points inside:

- DeltaNet regions,
- Mamba regions,
- attention regions,
- immediately before/after MoE layers.

### Server Tests

- 2, 4, 8 simultaneous text-only requests to same model.
- mixed model requests route to separate workers.
- incompatible requests do not enter batch.
- streaming heartbeats continue while queued/prefilling.
- cancellation releases session state.
- `cached_tokens` reports prefix hits.

## Performance Telemetry

Log one concise line per worker batch:

```text
[hipfire] prefill_batch model=qwen3.6:35b-a3b n=8 total_prompt=12288 max_prompt=2048 cached=8192 prefill_ms=43.2 tok_s=284444
```

Track counters:

- worker loads/unloads,
- active sessions,
- queued sessions,
- prefix-cache hits/misses,
- cached tokens,
- batch size histogram,
- prefill tokens/s,
- decode tokens/s,
- evictions,
- cancellation count.

## Implementation Slices

### Slice 0 - Documentation And Guardrails

- Land this plan.
- Keep existing server batching feature off by default.
- Add explicit docs that true batching requires per-session state isolation.

### Slice 1 - RequestSession Extraction

- Extract prompt rendering/tokenization/budget checks from Qwen35 `generate`.
- Introduce `Qwen35SessionState`.
- Keep behavior serialized.
- Verify existing `hipfire run` and `/v1/chat/completions` behavior.

Success criteria:

- no output change for single request,
- existing no-GPU tests pass,
- short GPU smoke passes.

### Slice 2 - ModelWorker Registry

- Add single-worker registry behind current serve path.
- Move loaded-model resources into `ModelWorker`.
- Route requests through worker API.
- Keep max resident models = 1.

Success criteria:

- no external API change,
- reload behavior matches current behavior,
- worker health exposed in `/health`.

### Slice 3 - State Arena

- Add paged state allocator for Qwen35 KV + DeltaNet.
- Attach session state handles to `RequestSession`.
- Remove global `m.seq_pos` dependency from Qwen35 text path.

Success criteria:

- two sessions can be created and advanced independently in one daemon process,
- split-prefill equivalence passes,
- cancellation frees state.

### Slice 4 - Prefix Cache

- Add token-prefix rolling hash.
- Add checkpoint creation at configured boundaries.
- Reuse checkpoint for exact-prefix requests.
- Emit `cached_tokens`.

Success criteria:

- repeated identical prompt reports full prompt cache hit,
- prompt with shared system/history reports partial hit,
- logits/first-token parity vs no-cache.

### Slice 5 - Prefill Microbatching

- Add worker prefill queue.
- Batch same-model text-only AR sessions.
- Start with Qwen35 dense/A3B.
- Preserve fallback for incompatible requests.

Success criteria:

- 2/4/8 simultaneous requests complete independently,
- batch log shows one batch for compatible arrivals,
- first-token parity vs serialized path,
- grouped MoE path stays active for A3B MQ4 on supported RDNA.

### Slice 6 - Multi-Model Residency

- Allow multiple resident workers when VRAM allows.
- Add per-worker queues.
- Add LRU unload for idle workers.

Success criteria:

- simultaneous requests to two models do not corrupt state,
- same-model requests batch within each worker,
- cross-model requests do not batch together,
- idle unload never kills active sessions.

### Slice 7 - Mamba-Hybrid State Support

- Add config parser for Mamba/hybrid state layout.
- Add Mamba recurrent state arena.
- Add prefix checkpoint support for Mamba state.
- Bring up a no-GPU Nemotron-H state-layout test first.

Success criteria:

- config-derived state layout matches expected layer schedule,
- split-prefill recurrent-state parity passes on smallest available Mamba
  model before any 550B-scale claims.

### Slice 8 - Decode Batching

- Add per-worker decode scheduler.
- Batch one decode step across active sessions.
- Keep per-session sampler/filter/output state isolated.

Success criteria:

- throughput improves under concurrent load,
- streaming order remains correct per request,
- cancellation does not poison other sessions.

### Slice 9 - Advanced Features

Re-enable feature combinations only after dedicated tests:

- tools,
- images/VL,
- PFlash,
- DFlash,
- MTP,
- CASK/eviction,
- multi-GPU batching.

Each gets its own compatibility predicate and correctness gate.

## Main Risks

### Silent State Corruption

Highest risk. Any missing state field in `RequestSession` or
`PrefixCheckpoint` can produce fluent wrong answers.

Mitigation:

- split-prefill parity tests,
- first-token logits parity where feasible,
- per-architecture state-layout tests,
- cache disabled by default initially.

### VRAM Fragmentation

Paged state arenas and multiple resident models can fragment memory.

Mitigation:

- fixed-size page classes,
- per-worker arenas,
- explicit session release,
- conservative resident-model defaults.

### Scheduler Complexity

Combining streaming, cancellation, prefix cache, and batching can become
fragile.

Mitigation:

- one scheduler per worker,
- prefill batching before decode batching,
- incompatible requests fall back cleanly,
- keep external API stateless.

### Architecture Drift

Qwen35, Nemotron-H, DeepSeek, MiniMax, Dots OCR, and future architectures have
different persistent state.

Mitigation:

- architecture declares `LayerStateKind`,
- scheduler sees only `RequestSession` handles,
- worker owns architecture-specific implementation.

## Recommended First Concrete PR

Do not start with true batching. Start by extracting state:

1. Add `RequestSession` and `Qwen35SessionState`.
2. Move `seq_pos`, `conversation_tokens`, KV cache handle, and DeltaNet state
   under the session.
3. Keep one request at a time.
4. Add split-prefill parity tests.
5. Only then add prefix cache and prefill batching.

This is the smallest step that reduces risk instead of adding another layer on
top of the current single mutable-state design.
