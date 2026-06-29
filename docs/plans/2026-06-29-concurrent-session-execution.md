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
  (not just qwen35/lfm2). Sub-steps:
  - **C1a — accessor indirection (safe, behavior-preserving).** Add
    `seq_pos()` / `set_seq_pos()` / `bump_seq_pos()` (+ `conversation_tokens*`)
    to `LoadedModel`; migrate the ~91 `m.seq_pos` sites (S0-style mechanical
    rename). The field stays; call sites are decoupled from it. Gate: workspace
    build + coherence on ≥2 arches.
  - **C1b — relocate behind the accessor.** Change the accessor internals to
    read/write the *active session's* cursor (the saved session structs already
    carry `seq_pos`/`conversation_tokens`), eliminating the working-copy field +
    the take/restore swap of it. Validate coherence + multi-turn across
    qwen35 / lfm2 / minimax / VL.
- **C2 — concurrent residency, serial step.** N sessions resident (arena reserve),
  decode them round-robin (no shared slot). Validate two concurrent sessions
  produce identical output to two serial runs.
- **C3 — microbatched prefill** for one arch (lfm2 or a dense arch) via the
  scheduler selection. Validate vs serial parity + measure throughput.
- **C4 — microbatched decode** + wire the scheduler's `PrefillBatchSelection` to
  the concurrent executor; delete `run_generate_batch_prefill_serial_*`.
- **C5 — extend** to the remaining rich arch + simple-tier opt-in; delete the
  single-slot remnants.

## Verification

Per phase on gfx1151 under `hipfire lock`: `coherence-gate-dflash.sh`; a
two-concurrent-session parity smoke (concurrent output == serial output token-for-
token at temp 0); DFlash spec parity; throughput vs the serial baseline.

## Scope note

This is the largest project in the prefill-seam line and effectively the daemon's
concurrent-execution engine. The prefill-seam deliverables (transformer seam, N-D
matrix, Lfm2Backend SimpleAr) and the hoist S0/S1 are landed/committed and
independent. C0 (hoist S2–S5) is the immediate prerequisite.
