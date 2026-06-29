# Concurrent session execution — per-session slots + microbatching

Status: design / not started. Supersedes the single-resident-slot model. Builds
on the SessionServingBackend hoist (docs/plans/2026-06-29-session-serving-backend.md,
S0+S1 done). GPU box: gfx1151.

## Context

Today the daemon is **single-resident-slot**: one model resident
(`max_resident_workers: 1`), one shared cursor (`m.seq_pos` — 172 sites;
`m.conversation_tokens` — 103 sites), and sessions (qwen35 *or* lfm2) **swap**
their per-session state in and out of that one slot
(`Qwen35/Lfm2RequestSessionState::take_from_loaded`/`restore_into_loaded`).

The **planning** layer for multi-session already exists in `hipfire-scheduler`:
`RequestSessionDraft`, `PrefillBatchSelection { sessions: Vec<…> }`,
`next_prefill_batch`, `sessions_compatible_for_prefill`, priority policy. But the
**execution** is serial — the daemon consumes a selection via
`run_generate_batch_prefill_serial_{qwen35,lfm2}` (note *serial*): each session is
swapped through the single slot in turn. The shared cursor is the reason it can't
run them concurrently.

**Goal (user decision 2026-06-29):** restructure to **per-session slots** — each
session carries its own cursor + resident KV/recurrent pages, multiple sessions
resident concurrently, executed as a **microbatch** (one forward over several
sessions' tokens) driven by the scheduler's existing selection. Kill the shared
`m.seq_pos`/`m.conversation_tokens` resident cursor.

This is the "future multi-session scheduler/microbatching" referenced when we hit
the `SequenceState` ownership question — we are building it now, not deferring.

## Salvage (reuse, do not reinvent)

- **`hipfire-scheduler`** — the selection/priority layer: `PrefillBatchSelection`,
  `next_prefill_batch`, `sessions_compatible_for_prefill`,
  `worker_key_is_state_arena_conservative`, priority classes. The executor consumes
  these; the policy stays.
- **`hipfire-state`** — `GenericSequenceStateArena` (`reserve`/`describe`/`release`,
  per-session `SessionStateReservation` keyed by id) + `SequenceStateArenaBackend`
  ownership taxonomy. The arena already models *multiple* concurrent reservations;
  the single-slot executor just never used more than one at a time.
- **`SessionServingBackend` trait (S1, done)** — the per-arch session-op surface;
  it evolves into the per-session execution interface (its methods already key on a
  session, not the shared slot).
- **`Session{Qwen35,Lfm2}RequestSessionState`** — already carry per-session
  `seq_pos` + `conversation_tokens` (the swap snapshots them). Per-session slots
  makes those the *primary* home instead of a swapped-out copy.

## Core changes

1. **Per-session cursor.** Move `seq_pos` + `conversation_tokens` out of the shared
   `LoadedModel` slot into the per-session state (where a copy already lives).
   The ~275 generic-loop / repeat-penalty / logical-position sites read the
   *active request's* session cursor, not a model-global one. This is the largest
   mechanical change and the one that unblocks concurrency.
2. **Concurrent resident state.** Lift `max_resident_workers`/the swap model so N
   sessions' KV/recurrent pages are resident at once, allocated via the arena's
   per-session `reserve` (already keyed by session id). Bounded by a VRAM budget
   (`resident_state_reservation_budget_bytes`, exists).
3. **Microbatched forward.** A forward that advances K sessions' next tokens in one
   launch — block-diagonal attention (each session attends only its own KV), per-
   session positions. This is the real kernel work and the central risk: today's
   batched forwards (`lfm2::prefill_batch`, qwen35 prefill) batch tokens of **one**
   prompt, not across sessions. Phase it: (a) concurrent residency with serial
   step (correctness, no shared cursor), then (b) true microbatched step.
4. **Executor.** Replace `run_generate_batch_prefill_serial_*` with a generic
   concurrent driver over `&mut dyn SessionServingBackend` consuming a
   `PrefillBatchSelection`, owning the resident set + per-session decode loops.

## Key design decisions / open questions

- **Block-diagonal attention** across sessions in the microbatch — kernel support
  per arch (full-attn, lfm2 conv+GQA, qwen35 DeltaNet recurrent). DeltaNet/SSM
  recurrent mixers don't have a KV to mask — concurrent recurrent state is N
  independent states advanced in lockstep; feasible but a distinct kernel shape.
- **Simple-tier arches** (llama/qwen2/gemma3/zaya/nemotron, `SimpleAr`): do they
  get concurrency too, or stay one-at-a-time? Recommend: per-session slots are
  arch-agnostic at the cursor level; microbatched *forward* is opt-in per arch
  (cap flag), others run concurrent-resident + serial-step.
- **VRAM budget vs concurrency** — N concurrent sessions × KV pages on a 128 GB
  APU; the arena budget + scheduler admission gate K.
- **Memory `[[project_apu_gtt_memory]]`** — large contiguous hipMalloc is unsafe on
  the APU; per-session paged reservations must respect that.

## Phased plan (each compiles + GPU-validates on gfx1151 before advancing)

- **C0** — finish the SessionServingBackend hoist S2–S5 first (single-slot still),
  so the protocol is behind the trait before concurrency changes the executor.
  *(S2 was the entry point that surfaced the shared-cursor blocker.)*
- **C1 — per-session cursor.** Move `seq_pos`/`conversation_tokens` into the
  session state; thread the active session's cursor through the generation loop.
  No concurrency yet — behavior-identical, single active session. Validate
  coherence + sessions unchanged.

  **Scope finding (2026-06-29):** `m.seq_pos` is the **universal generation
  cursor**, not a rich-tier-only field. It is written by *every* arch's
  generate/prefill path — `generate_vl.rs` (VL decode, `m.seq_pos += 1`),
  `lfm2_prefill.rs`, `qwen35_prefill.rs`, `generate_arch.rs` (minimax etc.) and
  the generic `generate.rs` — ~91 `m.seq_pos` sites plus `conversation_tokens`.
  So C1 touches every arch's live hot path and needs multi-arch GPU validation
  (not just qwen35/lfm2). Sub-steps (as built):
  - **C1a — group into `SessionCursor` (DONE, 4a7619af8).** NOT accessor methods:
    those borrow the whole `*m` and break the pervasive disjoint field borrows the
    generation loop holds (`let tok = m.tokenizer…` across a seq_pos write) — 29
    borrow conflicts, reverted. Instead grouped `LoadedModel::{seq_pos,
    conversation_tokens}` → `cursor: SessionCursor`; `m.cursor.seq_pos` is a plain
    field path, preserving disjoint borrows. 0 conflicts.
  - **C1b-1 — unify the per-session structs to `cursor: SessionCursor`
    (DONE, 6c84bbf32).** Qwen35/Lfm2RequestSessionState carry the same cursor
    type; the swap moves one value. coherence-gate green (qwen35 + lfm2).
  - **C1b-2 — eliminate the `m.cursor` working copy + swap: FOLDS INTO C2.** The
    active session is decomposed into `LoadedModel` fields on activate
    (`restore_into_loaded`: session.cursor→m.cursor, session.sequence_state→
    m.sequence_state) and recomposed on save. Eliminating the working copy means
    the forward reads the active session's cursor DIRECTLY — i.e. the active
    session stays a cohesive resident struct (`m.<active>.cursor` /
    `…sequence_state`) instead of decomposed. That forward-hot-path restructure
    IS C2 (residency). So C1a + C1b-1 are the cursor PREP (done); the elimination
    lands with C2.
- **C2 — concurrent residency, serial step.** N sessions resident (arena reserve),
  decode them round-robin (no shared slot). Validate two concurrent sessions
  produce identical output to two serial runs.

  **C2 design — `m.active` shape + borrow implications (2026-06-29).** The active
  session is today DECOMPOSED across four `LoadedModel` fields — `cursor:
  SessionCursor` (163 access sites), `sequence_state: Option<SequenceState>`
  (qwen35 KV+DeltaNet, 15), `q35_active_prefilled_generated_suffix_len: usize`
  (3), and `lfm2moe_state: Option<Lfm2MoeState>` (cfg, 18). Activate spreads a
  saved session into these (`restore_into_loaded`); save recomposes
  (`take_from_loaded`) — that field-by-field copy IS the cursor *working copy*.

  **Shape decision: group the four into one cohesive `pub active: ResidentSession`
  field (NOT an enum, NOT accessor methods).** Rationale follows the C1a finding:
  - *Not accessor methods* — `m.set_seq_pos(x)` borrows all of `*m`, breaking the
    pervasive disjoint field borrows the generation loop holds (`let tok =
    m.tokenizer…` alive across a cursor write; `let ss = m.sequence_state…`). C1a
    hit 29 borrow conflicts this way and reverted.
  - *Not an enum* (`enum ActiveSession { Qwen35(..), Lfm2(..) }`) — an enum
    reintroduces the borrow problem *inside* the rich tier: the qwen35 hot path
    needs `cursor` AND `sequence_state` borrowed disjointly, which an enum variant
    can't yield without a single `match` binding threaded through ~100 sites.
  - *Field grouping wins* — `m.active.cursor.seq_pos` and
    `m.active.sequence_state.kv` are plain field paths; Rust grants disjoint
    borrows at any nesting depth, so `m.active.cursor` ⟂ `m.active.sequence_state`
    ⟂ `m.tokenizer` all coexist exactly as the four flat fields do today. The
    change is a pure field-path regroup — the same mechanical class as C1a
    (`{seq_pos,conversation_tokens}` → `cursor`) and C1b-1, both GPU-validated.

  ```rust
  #[derive(Default)]
  pub struct ResidentSession {
      pub cursor: SessionCursor,
      pub sequence_state: Option<SequenceState>,        // qwen35 KV + DeltaNet
      pub q35_active_prefilled_generated_suffix_len: usize,
      #[cfg(feature = "arch-lfm2moe")]
      pub lfm2moe_state: Option<lfm2moe::lfm2moe::Lfm2MoeState>,
  }
  ```

  `Option<T>: Default` regardless of `T`, so `derive(Default)` holds even though
  `SequenceState`/`Lfm2MoeState` aren't `Default`; the cfg-gated field is fine in
  a derived `Default`. Field names are kept identical (even the redundant
  `active.q35_active_prefilled_…`) so every access is a pure `m.X → m.active.X`
  prefix-insert (compiler-backstopped: a non-`LoadedModel` `m` — `m` is overloaded
  for messages/iterators — would fail to find `.active`).

  **Sub-steps (mirrors C1a/C1b discipline):**
  - **C2a — group into `ResidentSession` (DONE 2026-06-29).** Pure regroup:
    defined the struct (model.rs), replaced the 4 `LoadedModel` fields
    (`cursor`/`sequence_state`/`q35_active_prefilled_generated_suffix_len`/
    `lfm2moe_state`) with `pub active: ResidentSession`, prefix-inserted the access
    sites (163 `m.cursor` + 15 `m.sequence_state` + 18 `m.lfm2moe_state` + 3
    suffix-len + the `impl LoadedModel` `self.sequence_state`) across
    generate/generate_arch/generate_vl/qwen35_prefill/lfm2_prefill/session/load +
    daemon main.rs, fixed the 15 construction literals (11 all-default →
    `ResidentSession::default()`, 3 qwen35 carry `sequence_state` shorthand, 1 lfm2
    carries `Some(state)`). GOTCHA repeated from C1b-1: single-line perl misses
    multi-line `m\n  .field` chains (18 of them) — caught with a `-0777` slurp
    pass. Compiles clean w/ & w/o `arch-lfm2moe` + workspace. **GPU-validated
    (gfx1151):** coherence-gate.sh exit 0 16/16 OK; baseline-vs-changes Output
    diff is byte-IDENTICAL across qwen3.5 dense 0.8b–27b (mq3/mq4/mq6/lloyd) +
    lfm2.5-8b-a1b → behavior-preserving confirmed (lfm2 greedy repetition is
    pre-existing).
  - **C2b — collapse the swap spread (DONE 2026-06-29).** Scope finding: C2a
    already made the active slot cohesive (`m.active`), so the C1b-2 "decomposed
    active session" is *already* gone — what remained was the field-by-field
    *spread* in `restore_into_loaded` (`m.active.cursor.seq_pos = self.cursor…;
    restore_sequence_state_into_model(m, …); m.active.q35_…suffix = …`). Replaced
    both arches' restore with a single wholesale move
    `m.active = ResidentSession { … }` (activate = move the resident in); deleted
    the now-dead `restore_sequence_state_into_model` helper (converge-and-delete).
    `take_from_loaded` left as-is — it is extraction, not a spread, and lfm2's
    take intentionally derives `seq_pos` from `state.n_tokens`. Did NOT nest a
    `ResidentSession` inside the session structs (would be ~130 hot-path edits +
    borrow surgery for the disjoint `state.sequence_state.kv`/`.recurrent` sites,
    and C2c dissolves the swap anyway) — the session structs keep flat fields as
    the parked snapshot. Behavior-preserving: coherence-gate.sh exit 0 16/16,
    output BYTE-IDENTICAL to the C2a commit across the full qwen3.5+lfm2 matrix.
  - **C2c — N concurrent residents**: `m.active` → a resident set keyed by session
    id (arena `reserve`), serial round-robin decode (no microbatch yet).
- **C3 — microbatched prefill** for one arch (lfm2 or a dense arch) via the
  scheduler selection. Validate vs serial parity + measure throughput.
- **C4 — microbatched decode** + wire the scheduler's `PrefillBatchSelection` to
  the concurrent executor; delete `run_generate_batch_prefill_serial_*`.
- **C5 — extend** to the remaining rich arch + simple-tier opt-in; delete the
  single-slot remnants.

## Ground-truth finding (2026-06-29, after C2a/C2b) — plan premise partly stale

Investigating the C2c entry surfaced that the "single resident slot / execution is
serial / build concurrency now" premise is **only partly true**. The actual state:

- **qwen35 already has concurrent multi-session kernels.** Decode:
  `run_generate_batch_decode_step_qwen35` (qwen35_decode.rs:343) selects
  `Qwen35DecodeBatchBackend::{SerialReference, FusedDenseLayerChunked,
  FusedGroupedMoeLayerChunked}` — the two *Fused* backends run a true microbatched
  forward over several **registry-resident** sessions (they call
  `qwen35_save_active_session` first to park `m.active`, then operate on the
  resident set via `validate_qwen35_decode_resident_sessions`). Prefill:
  `qwen35_prefill_suffix_batch` likewise has a fused-grouped-MoE batched backend
  plus a serial-reference fallback. So C3/C4-level **kernels** largely **exist**
  for qwen35, and each resident session carries its own cursor — the
  "shared-cursor blocks concurrency" framing is moot on the batch path.
- **These batch paths are NOT wired to live traffic.** `hipfire-server` never
  emits `generate_batch_decode` / `generate_batch_prefill` envelopes for organic
  HTTP requests (only the daemon's test harness + the explicit batch protocol do;
  cf. the health.rs `generate_batch_prefill_not_used_for_file_batches` note). The
  scheduler *plans* batches (`PrefillBatchSelection`/`next_prefill_batch`) but the
  server still drives the daemon one session at a time through `generate()` /
  `m.active`. **The real gap is the executor wiring** (server scheduler selection
  → batch envelopes → the daemon's existing fused backends), not a hot-path
  kernel rewrite.
- **lfm2 is serial-only** — no fused batch decode (only per-token
  `lfm2moe::forward::decode_step` + `run_generate_batch_prefill_serial_lfm2`).
- **Known bug:** the fused-dense batch-prefill KV-quant path is buggy (worked
  around by `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial` in
  smoke-generate-batch-prefill).

**Re-scope implication.** C2c-as-written (single-slot → resident-set hot-path
rewrite) is largely unnecessary: residency already works on the batch path; C2a/C2b
cleaned up the single-session `generate()` slot. The genuine remaining work is one
of: (A) wire the server scheduler → `generate_batch_*` → daemon fused backends so
**organic** concurrent traffic uses them; (B) fix the fused-dense KV-quant bug; (C)
add fused batch decode to lfm2; (D) build the two-session parity smoke + a grounded
"current concurrency state" doc first.

**USER DIRECTION + PROGRESS (2026-06-29):** chose (A) full quant-dense port + then
wire live traffic.
- **Dense fused PREFILL quant port — DONE + VALIDATED.** Made the dense fused
  prefill quant-complete: (1) KV contract relaxed to accept plain Q8 + Q8 KV
  write/attention branch (shared `prefill_session_batch_{write_q8_kv,attention_q8}_layer`
  helpers, renamed from `grouped_moe_*`); (2) `dense_session_prefill_gemm_full_precision`
  + `_residual` now dispatch Q8_0 + MQ4G256 (FWHT pre-rotation via internally-alloced
  scratch; MQ6G256 + other quant fall back to serial via the contract); (3) dense
  lm_head final-logits dispatch Q8_0 + MQ4G256; (4) weights contract predicate allows
  Q8_0 + MQ4G256. **Validation:** `smoke-generate-batch-prefill.sh` dense path now
  `backend=fused_dense ... ok` for size 2/4/8 + boundary + explicit (fused == serial
  parity) on qwen3.5-0.8b-mq4 (mq4 weights + Q8 KV + Q8_0 lm_head); coherence-gate
  16/16 OK + normal-path output BYTE-IDENTICAL to baseline (fused path is batch-only,
  doesn't touch single generation). No HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial
  workaround needed for prefill.
- **Separate PRE-EXISTING finding (NOT from this change):** the grouped-MoE fused
  prefill requires Q8 DeltaNet state, but the daemon auto-upgrades DeltaNet to FP32
  for the MoE test model → `grouped MoE session fused prefix row 0 has FP32 DeltaNet
  state; first MoE target is Q8 DeltaNet`. This was masked before (smoke failed earlier
  at the dense gate). Fix candidate: branch the grouped-MoE DeltaNet FP32/Q8 like the
  dense KV branch (needs FP32 DeltaNet layer in the grouped-MoE loop). Separate from
  the dense port; needed for MoE live concurrency.
- **REMAINING:** dense fused DECODE quant port (qwen35_decode_step_fused_dense_layer_chunked
  + capability gate requires kv_mode=fp32 today), then live-traffic wiring.

## Verification

Per phase on gfx1151 under `hipfire lock`: `coherence-gate-dflash.sh`; a
two-concurrent-session parity smoke (concurrent output == serial output token-for-
token at temp 0); DFlash spec parity; throughput vs the serial baseline.

## Scope note

This is the largest project in the prefill-seam line and effectively the daemon's
concurrent-execution engine. The prefill-seam deliverables (transformer seam, N-D
matrix, Lfm2Backend SimpleAr) and the hoist S0/S1 are landed/committed and
independent. C0 (hoist S2–S5) is the immediate prerequisite.
