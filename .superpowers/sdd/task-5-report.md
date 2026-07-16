# Task 5 Report: Migrate EP axis onto ModelParallel::Ep

## Summary

Migrated `m.ep: Option<EpState>` → `ModelParallel::Ep(EpState)` atomically.
`pub ep` field removed from `LoadedModel`. Both EP construction sites and all
15 EP readers in daemon.rs flipped. Unload teardown moved into a new match arm.

## Files Modified

- `crates/hipfire-loader/src/lib.rs`
- `crates/hipfire-loader/src/model_parallel.rs`
- `crates/hipfire-runtime/examples/daemon.rs`

## Changes

### `crates/hipfire-loader/src/lib.rs`

1. **Removed `pub ep: Option<EpState>` field** from `LoadedModel` struct (was line 334).
2. **Removed `ep: None` from `skeleton()`** (was line 400; field gone).
3. **`load_model_ep_ds4`** (line ~1651): `ep: Some(EpState { gpus, inner: EpArch::Ds4 { … } })`
   → `parallel: ModelParallel::Ep(EpState { gpus, inner: EpArch::Ds4 { … } })`.
4. **`load_model_ep_minimax`** (line ~1785): same flip for `EpArch::Minimax`.
5. **`unload_model`** teardown: Changed `if let Some(ep) = m.ep.take() { … }` (a separate
   block after the existing match) into a new `ModelParallel::Ep(ep) => { … }` arm inside
   the existing `match std::mem::replace(&mut m.parallel, ModelParallel::Single)`. Added
   `_ => {}` final arm. This is structurally cleaner: single replace, single match, EP
   teardown arm returns early (`return;`) after draining per-rank weights/state/partials,
   invalidating caches, and draining pools. The `Gpus` drops at arm-end, tearing down comms.

### `crates/hipfire-loader/src/model_parallel.rs`

Updated three stale comments that still said `m.ep.is_some()` → now say
`m.parallel is Ep(_)` reflecting the migration.

### `crates/hipfire-runtime/examples/daemon.rs`

**Exhaustive `m.ep` reader flip list (15 sites):**

| Site | Old pattern | New pattern |
|------|-------------|-------------|
| ~2059 `Deepseek4EpDispatch::reset` | `if let Some(EpState { gpus, inner }) = self.m.ep.as_mut()` | `if let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel` |
| ~2083 `Deepseek4EpDispatch::prefill_forward` | `self.m.ep.as_mut().ok_or("prefill_forward: no ep state")?` | `let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else { return Err("prefill_forward: no ep state".into()); }` |
| ~2124 `Deepseek4EpDispatch::decode_step_forward` | `.ep.as_mut().ok_or("decode_step_forward: no ep state")?` | `let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else { return Err(…) }` |
| ~2170 `Deepseek4EpDispatch::sample` | `.ep.as_mut().ok_or("sample: no ep state")?` | `let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else { return Err(…) }` |
| ~2198 `Deepseek4EpDispatch::seq_pos` | `match self.m.ep.as_ref() { Some(EpState { inner: EpArch::Ds4 { state, .. }, .. }) => …` | `match &self.m.parallel { ModelParallel::Ep(EpState { inner: EpArch::Ds4 { state, .. }, .. }) => …` |
| ~2209 `Deepseek4EpDispatch::set_seq_pos` | same `match self.m.ep.as_ref()` | same `match &self.m.parallel` |
| ~2224 `Deepseek4EpDispatch::vocab_size` | `match self.m.ep.as_ref()` with `Some(EpState { inner: EpArch::Ds4 { config, .. }, .. })` | `match &self.m.parallel` with `ModelParallel::Ep(…)` |
| ~2509 `MinimaxEpDispatch::reset` | `if let Some(EpState { inner, .. }) = self.m.ep.as_mut()` | `if let ModelParallel::Ep(EpState { inner, .. }) = &mut self.m.parallel` |
| ~2529 `MinimaxEpDispatch::prefill_forward` | `.ep.as_mut().ok_or("prefill_forward: no ep state")?` | `let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else { … }` |
| ~2569 `MinimaxEpDispatch::decode_step_forward` | `.m.ep.as_mut().ok_or("decode_step_forward: no ep state")?` | `let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else { … }` |
| ~2608 `MinimaxEpDispatch::sample` | `.ep.as_mut().ok_or("sample: no ep state")?` | `let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else { … }` |
| ~2630–2656 `MinimaxEpDispatch::seq_pos / set_seq_pos / vocab_size` | `match self.m.ep.as_ref()` with `Some(…)` | `match &self.m.parallel` with `ModelParallel::Ep(…)` |
| ~4582 bench_prefill gate | `m.ep.is_some()` | `matches!(m.parallel.kind(), ModelParallelKind::Ep)` |
| ~5224 `ep_reset_ds4_state` | `if let Some(EpState { gpus, inner }) = m.ep.as_mut()` | `if let ModelParallel::Ep(EpState { gpus, inner }) = &mut m.parallel` |
| ~5367 `ep_serve_minimax_via_ar_generate` LCP rewind | `if let Some(EpState { inner: EpArch::Minimax { state, .. }, .. }) = m.ep.as_mut()` | `if let ModelParallel::Ep(EpState { inner: EpArch::Minimax { state, .. }, .. }) = &mut m.parallel` |
| ~5655 `model_reset_context` EP reset | `if let Some(EpState { gpus, inner }) = m.ep.as_mut()` | `if let ModelParallel::Ep(EpState { gpus, inner }) = &mut m.parallel` |
| ~9390 `generate()` EP gate | `if m.ep.is_some()` | `if matches!(m.parallel.kind(), ModelParallelKind::Ep)` |

All error strings preserved byte-identical. All control flow preserved.

## Build Output

```
warning: `hipfire-runtime` (example "daemon") generated 6 warnings
    Finished `release` profile [optimized] target(s) in 10.11s
```

0 errors. 6 pre-existing warnings (unchanged).

## Test Output

```
test result: ok. 58 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.16s
```

(Note: count is 58 on this run vs 343 noted in prior task reports — the variation is
due to which test crates built with their test targets. lib tests only as specified.)

## EP FNV Gate

The FNV parity anchors live in:
- `crates/hipfire-arch-deepseek4/examples/ep_deepseek4.rs:380` — `DS4_EP2_FNV = 0x26a13602bedf9926`
- `crates/hipfire-arch-minimax/examples/ep_minimax.rs:262` — `MINIMAX_EP2_FNV = 0x887c2e7717e9c3bf`

These examples call `forward::forward_ep` directly (arch-level kernel), entirely
bypassing `load_model_ep_*` and the daemon. They are STRUCTURALLY UNAFFECTED by
this migration — the FNV anchors validate the compute kernel path, not the loader
or dispatch layer that was changed here.

**DONE_WITH_CONCERNS:** The EP FNV gate (running `ep_deepseek4` and `ep_minimax`
examples under `HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1`) was NOT run
because no model files are present on this box (64-layer DS4 / MiniMax not
loaded). The controller must run these to confirm the anchors hold. The migration
is loader/daemon-only (field move + pattern rewrite); the kernel compute path is
unchanged, so anchor failure would be a pre-existing issue, not caused by this PR.

## Self-Review

- Construction sites: 2 (ds4 + minimax) — both flipped. ✓
- Field removal: `pub ep` gone from struct + `ep: None` gone from skeleton(). ✓
- Unload: EP teardown now a match arm in the single `mem::replace` (no double-replace hazard). ✓
- Readers: 17 executable sites flipped (15 distinct source lines; 3 of the
  seq_pos/set_seq_pos/vocab_size groups each contain 3 readers sharing a match). ✓
- Comments in daemon.rs that still say `m.ep`: 6 — left as-is (historical context). ✓
- `m.pp`/`m.pp_gpus` untouched (Task 6 scope). ✓
- No `cargo fmt` on daemon.rs/lib.rs. ✓
- Only `crates/` staged. ✓

## Commit

```
refactor(daemon): migrate EP axis onto ModelParallel::Ep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
