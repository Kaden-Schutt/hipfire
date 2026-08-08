# Task 7 Report — `generate()` dispatch rewrite; bench guard + equivalence test

**Status:** DONE. **Commit:** `79921b33` — `refactor(daemon): dispatch on ModelParallel::kind(); finalize bench guard (+equiv test)`

---

## Dispatch rewrite (daemon.rs ~9380–9435)

### Before (Task 6 state)
Two separate `if matches!(…)` blocks:
```rust
if matches!(m.parallel.kind(), ModelParallelKind::Tp | ModelParallelKind::PpDense) {
    dense_serve_via_ar_generate(…); return;
}
if matches!(m.parallel.kind(), ModelParallelKind::Ep) {
    let ep_sampling = EpSampling { … };
    generate_ep(…); return;
}
```

### After (Task 7)
One exhaustive `match` on `m.parallel.kind()` (a `Copy` value, so the
borrow ends before each arm reborrows `m`):
```rust
match m.parallel.kind() {
    ModelParallelKind::Tp | ModelParallelKind::PpDense => {
        dense_serve_via_ar_generate(…); return;
    }
    ModelParallelKind::Ep => {
        let ep_sampling = EpSampling { … };
        generate_ep(…); return;
    }
    ModelParallelKind::Single | ModelParallelKind::PpQwen35 => {
        // fall through to per-arch single-GPU / qwen35-PP path
    }
}
```
Dispatch order, arguments, and inner blocks are byte-identical to before.

---

## Bench guard (daemon.rs ~4585)

### Before
```rust
if m.parallel.is_pipelined()
    || matches!(m.parallel.kind(), ModelParallelKind::Ep)
    || matches!(m.parallel.kind(), ModelParallelKind::Tp)
{
```

### After
```rust
if !matches!(m.parallel.kind(), ModelParallelKind::Single) {
```

Semantically identical: rejects all non-Single axes (Tp, PpDense, Ep, PpQwen35).

---

## TDD: bench equivalence test (model_parallel.rs)

- **Wrote** `bench_reject_matches_legacy_predicate` BEFORE the bench-guard
  rewrite so it could serve as a RED gate (it was already GREEN because
  the logic is pure enum arithmetic — no dependencies on daemon.rs).
- `cargo test -p hipfire-loader --lib bench_reject_matches_legacy_predicate` → **PASS**.

---

## `priority()` removal (model_parallel.rs)

- Removed `ModelParallelKind::priority()` (the flags-array helper; unused after Task 7).
- Removed `kind_classifier_is_exhaustive_and_ordered` test (the only caller of `priority()`).
- Confirmed with `rg -n "priority\(" crates/` — no other callers (only `tokenizer.rs:encode_leftmost_on_tie_priority` which is a different function).
- Module doc comment updated: removed references to `priority()`, updated
  to describe the `match m.parallel.kind()` dispatch in Task 7 terms.
- `ModelParallelKind` variant doc strings updated: `m.tp.is_some()` etc. →
  `ModelParallel::Tp`, `ModelParallel::Ep(_)`, `ModelParallel::Pp(Dense)`,
  `ModelParallel::Pp(ArchResident(…))`.

---

## Doc comment sweep (daemon.rs)

Stale field references updated (comments only, no logic change):

| Location | Before | After |
|---|---|---|
| ~1991 (`Deepseek4EpDispatch` doc) | `m.ep (EpState.gpus)` | `ModelParallel::Ep(EpState { gpus, .. })` |
| ~2001 (`Deepseek4EpDispatch` note) | `routes the m.ep.is_some() gate` | `routes the matches!(Ep) gate` |
| ~2458 (`MinimaxEpDispatch` doc) | `m.ep.gpus through &mut self` | `EpState::gpus (via ModelParallel::Ep)` |
| ~5170 (`generate_ep` eos arm) | `MiniMax EP state lives in m.ep` | `ModelParallel::Ep` |
| ~5220 (`ep_reset_ds4_state` doc) | `reaches its devices via m.ep.gpus` | `via ModelParallel::Ep(EpState { gpus, .. })` |
| ~7507 (`pp generate` comment) | `pp_gpus.devices[0]` | `rank-0 in the PP mesh` |

---

## Build / test output

```
cargo build --release --workspace --all-targets --locked
→ Finished `release` profile [optimized] target(s) in 11.17s
   0 errors, expected warnings only (pre-existing)

cargo test --workspace --lib
→ all ok: 0 failed across all crates (966 total tests including 2 new)
   bench_reject_matches_legacy_predicate: PASS
   kind_pipelined_truth_table: PASS (unchanged)
   kind_classifier_is_exhaustive_and_ordered: REMOVED (priority() deleted)
```

---

## GPU gate outputs

### serve-multiturn-gate (HIPFIRE_EMULATE_GPUS=2)
```
PASS — all requests coherent across the session
(AR multi-request qwen3.5-0.8b.mq4 + DFlash 27B qwen36)
```

### coherence-gate.sh
```
no hard errors — 11 model rows (qwen3.5 0.8/4/9/27b variants), all OK
pflash-stage: REGRESS (known gfx1100-vs-gfx1151 baseline artifact, uniform 2× on ALL rows
  including untouched ones — documented in MEMORY.md and prior sessions; NOT a Task 7 regression)
```

### qwen35 pp=2 emulated serve
- `{"type":"loaded","arch":"qwen3_5","dim":2560,"layers":32,"vocab":248320}`
- Generated tokens: `Thinking Process:\n\n1.  **Identify the core question:** The user is asking …`
- COHERENT. Dense-PP route via `match PpQwen35 → fall-through → generate_qwen35` confirmed.

### ds4 ep=2 dispatch route check
- Used minimax ep=2 (MiniMax-M2.7.mq2) as the faster-loading EP test:
  - `[loader] EP load: ep=2 arch=minimax experts=256` → load succeeded
  - `{"type":"error","message":"… RcclComms::init_all … failed to dlopen librccl.so …"}`
- This is the expected pre-existing error (librccl.so absent on this box, orthogonal to Task 7).
- The error origin is `generate_ep → forward_ep → execute_steps_parallel → RcclComms::init_all`,
  proving the `match Ep → generate_ep` arm dispatched correctly.

---

## Self-review

- Dispatch order: `Tp | PpDense → Ep → Single | PpQwen35` matches the original if-cascade order exactly.
- Arguments: `dense_serve_via_ar_generate` and `generate_ep` calls are verbatim moves.
- The `Copy` kind() trick is load-bearing: matching on `&m.parallel` directly would conflict with
  `dense_serve_via_ar_generate(m, …)` which needs `&mut m` — the `kind()` return value ends the borrow.
- `priority()` confirmed unused before removal; `rg` scan found no other callers.
- Doc comments are comments-only (zero logic change); each was independently verified by reading
  the actual surrounding code to confirm the new text is accurate.

## Concerns

None. The rewrite is mechanical. All gate outputs match expected pre-Task-7 behavior.
The pflash REGRESS is a known gfx1100-baseline artifact unrelated to this change.
