# Session-serving backend hoist — qwen35 + lfm2 onto a shared trait

Status: design / not started. Owner: prefill-seam effort. GPU validation box:
gfx1151 (lfm2.5-8b-a1b-mq4, qwen3.5 dense + MoE).

## Context

hipfire has **two serving tiers** today:

- **Simple tier** — llama, qwen2, gemma3, zaya, nemotron — implement `SimpleAr`
  and serve through the shared `run_simple_ar` → prefill → `decode_loop`
  (stateless, one-shot, no sessions). This seam is done.
- **Rich tier** — qwen35 (5/6) and lfm2-moe (11) — a stateful protocol:
  multi-session KV, prefix-hash prompt-cache, semantic checkpoints, session
  fork/save, DFlash spec-decode, CASK eviction. This tier is **inline in the
  18k-line daemon** as an `if is_qwen35_family {...} else if is_lfm2 {...}`
  ladder, with the per-session state held in `LoadedModel` as parallel
  Option-soup and the logic in parallel `qwen35_*` / `lfm2_*` serving-core
  functions.

The rich tier is **duplicated, not shared**: qwen35 and lfm2 have a near-identical
~11-method session API and identical-shaped `LoadedModel` state. That duplication
— not redundant legacy — is what this plan converges. (Discovered while trying to
"migrate lfm2 to `run_simple_ar` and delete legacy": that would have *regressed*
lfm2, since `run_simple_ar` has no sessions/cache/DFlash, and diverged it from the
qwen35 flagship.)

**Goal:** make the rich protocol a first-class `SessionServingBackend` capability
that qwen35 and lfm2 both implement; relocate the per-arch session state out of
`LoadedModel` into each backend; collapse the daemon `if-arch` ladder + the
duplicated `qwen35_*`/`lfm2_*` functions into one generic driver; delete the
Option-soup. No feature regression; both archs validated on GPU.

## SALVAGE — most of the foundation already exists (2026-06-29)

This is NOT a greenfield build. A prior pass already built the
scheduler-visible session-state layer; this plan **formalizes a trait over it**,
it does not invent the arena. Reuse, do not reinvent:

- **`crates/hipfire-state`** — the shared, arch-agnostic type system + arena:
  - `GenericSequenceStateArena` (`reserve` / `describe` / `release` /
    `purge_expired` / `release_worker`) — the scheduler/batcher-visible
    session-state arena.
  - **`SequenceStateArenaBackend { Qwen35Wrapped, BackendOwned, Unsupported }`**
    with `owns_state_pages()`, `supported_operations()`, and
    `for_worker_parts(arch_id, pp)` — **this already answers the gating ownership
    question** (see below).
  - Request/handle/descriptor types: `SequenceStateForkRequest`,
    `SequenceStateCheckpointRequest`, `SequenceStateReservationRequest/Plan`,
    `SequenceStateHandle`, `SequenceStatePageDescriptor`,
    `SequenceStatePageKind/Ownership`, `SequenceStateEvictionPolicy/SpillTarget`,
    per-arch `*_state_kind_labels()`. These become the trait's param/return types.
- **`crates/hipfire-scheduler`** — the multi-session microbatcher already exists:
  `RequestSessionDraft`, `PrefillBatchSelection`, `next_prefill_batch`,
  `sessions_compatible_for_prefill`, and `worker_key_is_state_arena_conservative`
  (it already gates multi-session prefill batching on the arena backend). The
  future cross-session scheduler concern is already wired to the arena taxonomy.
- **`crates/hipfire-serving-core/src/session.rs`** — parallel per-session structs
  with a uniform API and generic dispatch already partly done:
  - `Qwen35RequestSessionState` / `Lfm2RequestSessionState` both expose
    `new` / `take_from_loaded` / `restore_into_loaded` / `reset` / `fork_from`.
  - Generic (arena-backend-dispatched, NOT if-arch) helpers already exist:
    `loaded_model_state_arena_backend`, `loaded_model_worker_runtime_view`,
    `describe_loaded_model_sequence_state`,
    `release_loaded_model_sequence_state_handles`, `backend_owned_session_id`,
    `backend_owned_state_page_descriptors`.

**What's left** (the actual hoist) is therefore much smaller than a from-scratch
build: (a) unify the two parallel `*RequestSessionState` structs + the
`qwen35_*`/`lfm2_*` *operation* free functions (activate/save/prefill/checkpoint/
generate/DFlash) behind one `SessionServingBackend` trait; (b) break the
`take_from_loaded`/`restore_into_loaded` coupling so state lives in the backend
(or arena) rather than `LoadedModel`; (c) replace the daemon `if-arch` ladder with
trait dispatch; (d) delete the Option-soup + the now-duplicate free functions.

## The duplicated surface (evidence)

Parallel serving-core functions (one pair per row) → trait methods:

| protocol op | qwen35 fn | lfm2 fn |
|---|---|---|
| activate session | `qwen35_activate_session` | `lfm2_activate_session` |
| logical position | `qwen35_active_logical_position` | `lfm2_active_logical_position` |
| allocate state | `qwen35_allocate_session_state` | `lfm2_allocate_session_state` |
| checkpoint | `qwen35_checkpoint_session_state` | `lfm2_checkpoint_session_state` |
| fork | `qwen35_fork_session_state` | `lfm2_fork_session_state` |
| reset active | `qwen35_reset_active_session` | `lfm2_reset_active_session` |
| save active | `qwen35_save_active_session` | `lfm2_save_active_session` |
| release | `qwen35_release_sessions` | `lfm2_release_sessions` |
| session count | `qwen35_request_session_count` | `lfm2_request_session_count` |
| materialize prefill | `qwen35_materialize_batch_prefill_prompt` | `lfm2_materialize_prefill_tokens` |
| prefill (suffix/checkpoints) | `qwen35_prefill_suffix_batch*` | `lfm2_prefill_with_boundary_checkpoints` |

Parallel `LoadedModel` state (`crates/hipfire-serving-core/src/model.rs`):

```
sequence_state: Option<SequenceState>           // shared KV/recurrent pages
eviction:       Option<Eviction>                // shared CASK
q35_sessions:   HashMap<String, Qwen35RequestSessionState>     ┐
q35_active_session_id: Option<String>                          │ identical
q35_active_state_allocation_epoch: u64                         │ shape
lfm2_sessions:  HashMap<String, Lfm2RequestSessionState>       │
lfm2_active_session_id: Option<String>                         │
lfm2_active_state_allocation_epoch: u64                        ┘
lfm2_dflash:    Option<Lfm2DflashState>         // + per-arch DFlash
```

→ a generic `SessionRegistry<S> { sessions: HashMap<String, S>, active_id,
allocation_epoch }` parameterized by the arch's per-session state `S`.

## Trait shape (to finalize against the exact signatures)

Object-safe; the daemon holds `&mut dyn SessionServingBackend`. Methods take
`&mut self`, `&mut Gpu`, ids/slices, return simple owned types — same discipline
as `ServingBackend`. The per-session state type and DFlash state stay concrete
inside the implementor.

Param/return types (`SessionForkRequest`, `BoundaryCheckpoint`, page descriptors,
etc.) are the **existing `hipfire-state` types** (`SequenceStateForkRequest`,
`SequenceStateCheckpointRequest`, `SequenceStatePageDescriptor`, …), not new ones.

```rust
pub trait SessionServingBackend: ServingBackend {
    /// Which arena ownership mode this backend uses (drives supported ops).
    fn state_arena_backend(&self) -> hipfire_state::SequenceStateArenaBackend;
    fn activate_session(&mut self, gpu: &mut Gpu, id: &str) -> Result<bool, String>; // created?
    fn reset_active_session(&mut self, gpu: &mut Gpu) -> Result<(), String>;
    fn save_active_session(&mut self) -> Result<(), String>;
    fn active_logical_position(&self) -> Result<usize, String>;
    fn fork_session_state(&mut self, gpu: &mut Gpu, req: SessionForkRequest) -> Result<(), String>;
    fn checkpoint_session_state(&mut self, gpu: &mut Gpu, /* … */) -> Result<(), String>;
    fn release_sessions(&mut self, gpu: &mut Gpu, ids: &[String]) -> Result<usize, String>;
    fn request_session_count(&self) -> usize;
    fn materialize_prefill_tokens(&mut self, session: &SessionPrefillSpec)
        -> Result<(Vec<u32>, Vec<BoundaryCheckpoint>), String>;
    fn prefill_session(&mut self, gpu: &mut Gpu, batch_id: &str, session: &SessionPrefillSpec,
        tokens: &[u32], checkpoints: &mut Vec<BoundaryCheckpoint>) -> Result<(), String>;
    // DFlash spec-decode fast path (caps().dflash); None ⇒ no drafter loaded.
    fn dflash_spec_step(&mut self, gpu: &mut Gpu, /* … */) -> Option<Result<SpecStep, String>>;
}
```

The shared envelope/result types (`SessionForkRequest`, `SessionPrefillSpec`,
`BoundaryCheckpoint`, `SpecStep`) are arch-agnostic and already partly exist
(`SequenceStateForkRequest`, `*PrefillSessionResult`, `SequenceStatePageKind`) —
lifted to `hipfire_runtime::arch` (or a `serving` submodule).

`run_generate_batch_prefill_serial_{qwen35,lfm2}` collapse into ONE
`run_generate_batch_prefill_serial(backend: &mut dyn SessionServingBackend, …)`.

## Phased, GPU-validated plan

Each phase compiles and keeps the daemon behavior-identical until the final
delete; validate on gfx1151 before advancing.

- **S0 — generic state (mostly salvage).** The shared types + arena already exist
  in `hipfire-state`; the per-session structs already share an API in `session.rs`.
  S0 = factor the `q35_*`/`lfm2_*` `LoadedModel` fields into a generic
  `SessionRegistry<S>` (sessions map + active_id + allocation_epoch — both archs are
  identical-shaped) and route `describe`/`release`/worker-view through the *existing*
  arena dispatch (they already are, partly). Mechanical, behavior-preserving. Gate:
  workspace build + `no-gpu-ci.sh`.
- **S1 — define the trait** in `hipfire-runtime::arch` (compiles, no impls yet),
  reusing `hipfire-state` types as method params; the trait advertises
  `state_arena_backend()` so dispatch follows the existing taxonomy.
- **S2 — lfm2 impl.** Implement `SessionServingBackend` for the lfm2 backend by
  *moving* the `lfm2_*` serving-core bodies onto it (state now backend-owned).
  Daemon still calls the same entry points (thin shims). Validate: lfm2 sessions
  + prefix-cache + DFlash spec parity vs current `main` on gfx1151
  (`coherence-gate-dflash.sh` + a session/cache smoke).
- **S3 — qwen35 impl.** Same move for qwen35. Validate qwen35 dense + MoE +
  DFlash/MTP on gfx1151.
- **S4 — daemon dispatch.** Replace the `if is_qwen35 {} else if is_lfm2 {}`
  ladder (main.rs ~3446, 4094) with `&mut dyn SessionServingBackend`; collapse the
  two `run_generate_batch_prefill_serial_*` into one generic driver.
- **S5 — delete duplication.** Remove the per-arch `qwen35_*`/`lfm2_*` functions
  superseded by trait methods, the `LoadedModel` Option-soup, and the `lfm2_prefill`
  legacy re-exports. Final full GPU gate on both archs.

## Risks / open questions

- **Trait granularity:** the qwen35 prefill has many specialized variants
  (`_fused_dense`, `_fused_grouped_moe`, `_serial_reference`); these stay *inside*
  the qwen35 `prefill_session` impl, not on the trait.
- **DFlash + CASK exclusion:** today `DFlash + eviction` is rejected; preserve that
  guard in the trait impl.
- **`SequenceState` ownership — RESOLVED by the existing `SequenceStateArenaBackend`.**
  The worry (backend must not own pages opaquely, because EP/TP/GPU-parallel batching
  + the future cross-session microbatcher need a global view) is **already designed
  for**: `GenericSequenceStateArena` is the scheduler/batcher-visible layer, and
  `SequenceStateArenaBackend` classifies ownership per (arch, pp):
  - **`Qwen35Wrapped`** (qwen35, pp=1): arena owns/manages the pages — full
    reserve/attach/fork/release/describe. The scheduler sees every session.
  - **`BackendOwned`** (lfm2/minimax/nemotron, pp=1): the backend owns its pages;
    the arena only `describe`s them (so the scheduler still has visibility for
    accounting/batchability via `worker_key_is_state_arena_conservative`).
  - **`Unsupported`**: pp>1 etc.

  So the trait does NOT take an ownership stance: `SessionServingBackend` advertises
  its arena backend (`fn state_arena_backend(&self) -> SequenceStateArenaBackend`)
  and its supported ops follow `SequenceStateArenaBackend::supported_operations`. The
  arena stays where it is (scheduler/daemon-visible); the backend borrows handles.
  This unblocks S0 — no new ownership model needed.
- **Scope:** this is its own project, larger than the prefill-seam work that
  surfaced it. The prefill-seam deliverables (transformer seam, N-D matrix,
  `Lfm2Backend` SimpleAr for one-shot) are independent and already landed/validated.

## Verification

Per phase: workspace build + `no-gpu-ci.sh`; then on gfx1151 under `hipfire lock`:
`coherence-gate-dflash.sh`, a multi-session + prefix-cache reuse smoke, and a
DFlash spec-decode acceptance/parity run (`lfm2_dflash_acceptance_eval`,
qwen35 equivalent) diffed against pre-refactor `main`.
