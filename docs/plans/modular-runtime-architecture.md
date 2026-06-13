# Modular Runtime Architecture Plan

This plan consolidates the older `stabilize-before-extraction.md` and
`v1-architecture-roadmap.md` notes into one implementation-oriented roadmap.
It also incorporates the newer serving, scheduler, state-cache, and NPU module
direction.

## Summary

Hipfire should move toward a modular runtime where `run`, `serve`, `eval`, and
the legacy daemon are thin adapters over shared Rust libraries.

The important split is:

- typed library APIs for the hot path,
- IPC only at external process/protocol boundaries,
- backend engines behind explicit module contracts,
- state ownership separated from model residency,
- scheduler policy separated from execution.

This structure should support:

- CLI single-shot generation,
- Axum serving,
- eval evidence collection,
- daemon/JSONL compatibility,
- prefill and decode microbatching,
- resident prefix/state cache reuse,
- CPU/GPU/NPU module substitution,
- future multi-model workers.

## Stabilize First

Large runtime files such as `crates/hipfire-arch-qwen35/src/qwen35.rs`,
`crates/rdna-compute/src/dispatch.rs`, and `crates/hipfire-daemon/src/main.rs`
should not be split mechanically while correctness gates are unstable.

Extraction should be staged:

1. Land invariants and focused tests around the behavior being moved.
2. Add typed seams around the behavior in place.
3. Move code behind those APIs without changing behavior.
4. Only then broaden backends, batching, or protocol support.

Do not combine behavior-preserving extraction with MQ3/MQ6/MTP admission
changes, DFlash verifier changes, or new kernel format work.

Minimum gates for extraction work:

- focused crate `cargo check` and relevant unit tests,
- `./tests/no-gpu-ci.sh` for workflow-only changes,
- `./tests/coherence-gate-dflash.sh` after changes touching kernels, dispatch,
  quant formats, fusion, rotation, RMSNorm, or speculative decode,
- targeted `hipfire-eval` rows when eval/runtime behavior changes.

## Target Crates

### `hipfire-model`

Owns model and artifact loading:

- model path and package resolution,
- HFQ/GGUF metadata inspection,
- architecture selection,
- tensor index and weight residency,
- model worker identity and placement metadata,
- load-time compatibility checks.

This crate should not own prompt rendering, request scheduling, or generation
policy.

### `hipfire-prompt`

Owns prompt construction:

- tokenizer-facing prompt assembly,
- ChatML/Jinja/messages/tools rendering,
- prompt hashing and provenance,
- semantic boundary markers for state-cache checkpoints,
- prompt/template validation shared by `serve`, `run`, and `eval`.

Tool-call failures should be attributable to either this crate's rendered input
or to model output, not to server-specific plumbing.

### `hipfire-state`

Owns sequence state:

- per-session state handles,
- attention KV state,
- DeltaNet recurrent state,
- future Mamba/SSM and convolution state,
- logits snapshots,
- prefix checkpoints,
- attach/fork/reset/release operations,
- resident checkpoint limits,
- future disk spill/rehydrate policy.

The core abstraction is a complete sequence-state arena, not a KV-cache-only
store. Prefix caching must include all architecture state required to resume
generation correctly.

### `hipfire-generate`

Owns generation orchestration:

- prefill/decode/session loop,
- module substitution graph,
- backend selection and fallback,
- single-session and batched execution APIs,
- sampling integration,
- runtime timing hooks,
- logits drift and module comparison hooks.

This crate should decide *what* module runs next and *which backend* is
selected. It should not contain the backend implementation details.

### `hipfire-rocm`

Owns HIP/RDNA/CDNA backend implementation:

- ROCm module runners,
- ROCm tensor views/adapters,
- calls into `rdna-compute`,
- kernel dispatch wrappers,
- graph/capture details,
- ROCm-specific cache invalidation.

`rdna-compute` can remain the lower-level HIP kernel/runtime crate under this
backend. Do not rename or split it wholesale before the wrapper boundary is
proven. Keep this crate name backend-specific so future wrappers such as
`hipfire-cuda` or `hipfire-gaudi` can implement the same module contracts
without overloading a generic GPU crate name.

### `hipfire-cpu`

Owns CPU reference and fallback implementations:

- scalar/SIMD module runners,
- deterministic oracle paths,
- debug-only and validation module implementations,
- CPU fallback for unsupported backend modules.

For NPU work, CPU remains the correctness oracle.

### `hipfire-npu`

Owns XDNA/NPU backend implementation:

- XDNA/XRT or XDNA-IE integration,
- artifact discovery and readiness checks,
- NPU module contracts,
- buffer layout and transfers,
- per-module NPU dispatch,
- drift and timing metadata.

NPU modules should be opt-in per module until isolated module comparison,
mixed-pipeline comparison, and logits drift checks pass.

### `hipfire-scheduler`

Owns CPU-side scheduling policy:

- 256-level priority model,
- worker-local queues,
- compatibility filtering,
- prefill microbatch selection,
- decode active-set selection,
- coalescing windows,
- deadline aging,
- backpressure,
- memory-pressure admission hooks.

This crate should not own GPU/NPU state. It should form batch plans over
session drafts and state handles, then call `hipfire-generate`.

### `hipfire-coherence`

Owns runtime coherence detection:

- detector profiles,
- detector implementations,
- coherence report schema,
- generated-text and token-stream checks.

This should be usable by `hipfire-eval`, server smoke tests, and direct runtime
diagnostics without launching a daemon unless the daemon protocol itself is
under test.

### `hipfire-evidence`

Owns eval and runtime evidence:

- eval rows,
- manifests,
- artifact writers,
- cache keys,
- model/prompt/binary/provenance hashing,
- performance, coherence, memory, launch-count, and profiling artifacts.

`hipfire-eval` should use this crate directly. Daemon-backed eval rows should
be reserved for daemon protocol coverage, not basic model execution.

## Binary Adapters

### `hipfire-run`

Thin CLI adapter:

```text
args -> hipfire-model -> hipfire-prompt -> hipfire-generate -> stdout/evidence
```

It should share the same prompt and generation libraries as serving and eval.

### `hipfire-serve`

Axum/OpenAI-compatible adapter:

```text
HTTP request
  -> request normalization
  -> hipfire-prompt
  -> hipfire-scheduler
  -> hipfire-state
  -> hipfire-generate
  -> HTTP/SSE response
```

The current Bun/daemon split should be retired when the Axum port is complete.
External `/v1/chat/completions` remains stateless and OpenAI-compatible.

### `hipfire-daemon`

Legacy JSONL/process adapter over the same libraries. Keep it for compatibility
and daemon-specific tests, but do not make `hipfire-eval` depend on it for
ordinary model-backed rows.

### `hipfire-eval`

Evidence adapter:

```text
eval config
  -> hipfire-model/prompt/generate/coherence/evidence
```

The default executor should be direct library execution. Optional daemon/server
executors should test protocol behavior and process integration.

## Libraries vs IPC

Use libraries for communication between:

- `hipfire-server`,
- `hipfire-prompt`,
- `hipfire-state`,
- `hipfire-generate`,
- `hipfire-scheduler`,
- backend crates.

Use IPC only at external boundaries:

- OpenAI HTTP,
- legacy JSONL daemon protocol,
- optional per-worker process isolation,
- remote/distributed workers,
- crash or privilege isolation.

The hot path shares GPU handles, state handles, token buffers, allocator
state, and batch plans. Serializing those through IPC by default would make
prefix caching and decode batching harder to reason about.

## Microbatching Placement

Microbatching belongs in `hipfire-scheduler` plus `hipfire-generate`, not in
the server adapter.

Responsibilities:

- `hipfire-prompt`: render prompt tokens and boundary markers.
- `hipfire-scheduler`: decide which compatible sessions form a batch.
- `hipfire-state`: attach cached prefixes or allocate isolated session state.
- `hipfire-generate`: execute batched prefill/decode against backend modules.
- `hipfire-evidence`: record batch size, cached tokens, backend, timing, and
  fallback reasons.

Batching remains worker-local. Do not batch across different model workers,
accelerators, quant/state modes, incompatible prompt features, or unsupported
runtime features.

V1 scope:

- text-only non-streaming AR prefill batching,
- Qwen35 dense/grouped-MoE fused prefill,
- Qwen35 greedy decode active set,
- resident prefix checkpoint reuse.

Deferred:

- streaming batching,
- tools/images batching,
- PFlash/CASK batching,
- MTP/DFlash verify batching,
- multi-GPU request batching,
- generic non-Qwen35 session arenas.

## State Cache Placement

State caching belongs in `hipfire-state`, with policy hooks exposed to
`hipfire-scheduler` and telemetry exposed to `hipfire-evidence`.

Required state-cache concepts:

- `SessionStateHandle`: active decode/prefill state.
- `PrefixCheckpointHandle`: attachable complete sequence-state checkpoint.
- `StateKind`: attention KV, DeltaNet, Mamba/SSM, logits snapshot, backend
  private state.
- `PrefixHash`: daemon/runtime-authoritative hash of rendered token prefix.
- `StateResidency`: active, resident checkpoint, spillable, spilled, invalid.

Resident checkpoints should be capped independently from prefill batch size.
Future disk spill should be priority-gated and explicit in telemetry.

## Backend Module Contracts

Module substitution should use explicit contracts:

```text
ModuleKind
TensorContract
StateContract
BackendPreference
ModuleInvocation
ModuleOutput
DriftTolerance
```

Example qwen3.5 migration path:

```text
GPU production path
  rmsnorm -> qkv/z/alpha/beta -> recurrent state -> swiglu/down -> residual/logits

Step 1
  replace one FFN/SwiGLU/down module with hipfire-cpu oracle

Step 2
  route the same module to hipfire-npu behind an opt-in flag

Step 3
  compare CPU/GPU/NPU module output and final logits drift

Step 4
  repeat for the next module boundary
```

Default production behavior should not change until the replacement backend is
validated.

## NPU Policy

The NPU engine belongs in `hipfire-npu`, integrated through
`hipfire-generate`.

Rules:

- CPU module remains the oracle.
- NPU is opt-in per module.
- Artifact readiness is checked at model/backend selection time.
- Module-level drift and final logits drift are recorded separately.
- Evidence must state which modules ran on NPU.
- Host bookkeeping, cache append, residual add, and downstream fallback stay
  outside the NPU module unless the module explicitly owns them.

## CLI, Server, UI, Metrics

### Rust CLI

`hipfire-cli` should become the Clap-powered command surface for:

- `pull`,
- `run`,
- `serve`,
- `list`,
- future package/model helpers.

Model discovery and sidecar matching should move into Rust libraries used by
both CLI and server.

### Axum Server

`hipfire-server` should own Axum routing:

- `/v1/chat/completions`,
- `/v1/models`,
- `/v1/responses` where supported,
- `/health`,
- `/metrics`,
- static WebUI routes if enabled.

Avoid global mutable singleton state where request-scoped or worker-scoped
state can be injected through typed server state.

### Metrics and Accounting

Metrics should be recorded through shared evidence/telemetry structures:

- prefill tokens/sec,
- decode tokens/sec,
- TTFT,
- speculative accept rate,
- cached prompt tokens,
- state-cache hits/misses,
- batch size and coalescing delay,
- backend selection,
- VRAM/GTT/NPU memory where available.

Prometheus export and UI rendering are server concerns; metric definitions and
collection points should be library concerns.

## Distributed and Multi-Worker Direction

Multi-worker serving should build on the same model/state/generate/scheduler
split:

- worker identity includes model, backend, accelerator kind, device id, and
  state capability,
- batching remains worker-local,
- remote workers are an IPC/network adapter over the same typed worker API,
- cluster discovery/proxying is additive and should not leak into generation
  internals.

## Staged Migration

### Phase 0 - Guardrails

- Keep old behavior intact.
- Add direct eval paths where daemon process launch is not the thing being
  tested.
- Keep old docs as redirects to this plan.
- Continue focused checks before and after each extraction.

### Phase 1 - Prompt and Evidence

- Extract prompt rendering/hash helpers into `hipfire-prompt`.
- Extract coherence detectors into `hipfire-coherence`.
- Extract eval artifact/result helpers into `hipfire-evidence`.
- Make `hipfire-eval` call direct libraries for basic rows.

### Phase 2 - CPU Module Oracle

- Add `hipfire-cpu`.
- Define module contracts for one qwen3.5 dense FFN/SwiGLU/down path.
- Route the module through `hipfire-generate` with CPU oracle support.
- Add per-module and final-logits drift evidence.

### Phase 3 - ROCm Backend Wrapper

- Add `hipfire-rocm` as a wrapper over existing `rdna-compute` and arch calls.
- Do not move every kernel at once.
- Route the same module contract to ROCm production behavior.
- Keep `rdna-compute` as low-level HIP kernel/runtime support.

### Phase 4 - State Arena

- Extract Qwen35 wrapped state operations into `hipfire-state` APIs.
- Generalize handles for KV, DeltaNet, logits snapshots, and future Mamba
  state.
- Move prefix attach/fork/reset/release behind the state arena.
- Preserve existing Qwen35 checkpoint smokes.

### Phase 5 - Scheduler

- Move queue policy and compatibility checks into `hipfire-scheduler`.
- Keep server-specific request normalization in `hipfire-server`.
- Make batch plans typed inputs to `hipfire-generate`.
- Preserve prefill/decode batch smokes.

### Phase 6 - NPU Backend

- Add `hipfire-npu`.
- Integrate one opt-in module using the same module contract.
- Validate CPU vs NPU and mixed-pipeline drift.
- Record backend selection in eval/server evidence.

### Phase 7 - Axum Serve Consolidation

- Port remaining server path to `hipfire-server`.
- Keep daemon as legacy adapter.
- Make `hipfire-run`, `hipfire-serve`, and `hipfire-eval` share prompt,
  generate, state, and evidence libraries.

### Phase 8 - UI, Metrics, and Distributed Workers

- Add WebUI and Prometheus routes on top of `hipfire-server`.
- Add worker registry and optional remote worker adapters.
- Keep remote/process IPC outside the core generation libraries.

## Dependency Sketch

```text
hipfire-run
hipfire-serve
hipfire-daemon
hipfire-eval
    |
    +-- hipfire-model
    +-- hipfire-prompt
    +-- hipfire-scheduler
    +-- hipfire-state
    +-- hipfire-generate
    |       |
    |       +-- hipfire-cpu
    |       +-- hipfire-rocm
    |       |       +-- rdna-compute
    |       |               +-- hip-bridge
    |       +-- hipfire-cuda      (future)
    |       +-- hipfire-gaudi     (future)
    |       +-- hipfire-npu
    +-- hipfire-coherence
    +-- hipfire-evidence
```

Avoid cycles by keeping adapters above libraries and backend implementations
below `hipfire-generate`.

## Open Questions

- Which crate should own tokenizer internals long term: `hipfire-prompt` or a
  lower-level `hipfire-tokenizer`?
- Should `hipfire-model` own model package discovery, or should that stay in
  `hipfire-cli` until packaging stabilizes?
- What is the smallest stable `ModuleKind` set for qwen3.5 NPU bring-up?
- How much of `rdna-compute::Gpu` should be exposed through `hipfire-rocm`
  versus hidden behind backend tensor handles?
- What evidence tolerance should be accepted for CPU/ROCm/CUDA/Gaudi/NPU mixed
  pipelines?
