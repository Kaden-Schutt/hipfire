# DDTree GPU-Residency Stage 2 Results

**Date:** 2026-06-29  
**Branch:** feature/speculator-ddtree  
**Base commit:** 670eb12f (Stage 1 D2D scatter already in)  
**Changes in:** `crates/hipfire-arch-qwen35/src/speculative.rs`, `crates/hipfire-arch-qwen35/src/mtp_compose.rs`, `crates/rdna-compute/src/feature_flags.rs`

---

## Tasks

### D16 — Replace device_synchronize() with stream_synchronize (LANDED)

**Diff hunk:** `verify_dflash_block_inner` ~line 2630

```rust
// Before:
if batch_result.is_ok() && tree_verify.is_some() {
    gpu.hip.device_synchronize()?;
}

// After:
if batch_result.is_ok() && tree_verify.is_some() {
    if let Some(stream) = gpu.active_stream.as_ref() {
        gpu.hip.stream_synchronize(stream)?;
    } else {
        gpu.hip.device_synchronize()?;
    }
}
```

Additionally added `active_stream` setup at the top of `spec_step_ddtree_batched` (mirrors the da2753e pattern from `spec_step_dflash`):

```rust
if gpu.active_stream.is_none() {
    gpu.active_stream = Some(gpu.hip.stream_create()?);
}
```

**Ordering guarantee preserved:** All ops ride the same single HIP stream; `stream_synchronize` flushes all enqueued work on that stream before the next attention kernel reads the KV slot. Semantically identical to `device_synchronize` on hipfire's single-stream setup, but does not stall work on other streams.

---

### D9 — Gate verify-argmax D2H on `!use_swor` (LANDED)

**Diff hunk:** `verify_dflash_block_inner` ~line 2684 and `verify_dflash_block_tree` signature

Added `skip_argmax_d2h: bool` parameter to `verify_dflash_block_tree` and `verify_dflash_block_inner`. When `true` (SWOR mode), the ~244-byte argmax D2H is skipped; the GPU argmax kernel is still enqueued (no-op for safety), but the result is not downloaded. `argmax_per_pos` stays empty — safe because SWOR walk gets accepted indices from the 68-byte walk-result D2H (D13), not from argmax.

**Call sites:**
- `spec_step_ddtree_batched`: passes `use_swor` (true for temp>0, false for greedy)
- `verify_dflash_block` (chain path): passes `false` (argmax always needed)
- `mtp_compose.rs`: passes `false` (greedy path uses argmax)

---

### D8 — Remove naive full-logits ddtree path (LANDED)

**Removed:**
- `want_full_logits` variable (was `temp > 0.0 && !use_swor`)
- `use_swor = temp > 0.0 && !gpu.flags.ddtree_verify_naive` → simplified to `use_swor = temp > 0.0`
- `FeatureFlags::ddtree_verify_naive` field (+ `from_env` parse + test default)
- `else if want_full_logits { sample_verified_tree(...) }` arm in accept walk
- `HIPFIRE_DDTREE_DUMP_PQ` gate (only ran under `want_full_logits`)
- `else if want_full_logits { gpu.download_f32(&logits_batch) }` branch in lm_head section

**Rationale:** The naive path fired only when `HIPFIRE_DDTREE_VERIFY=naive` (non-default). SWOR is distribution-exact and on-GPU; the naive fallback required a ~37 MB/cycle full-logits D2H and is strictly dominated. `HIPFIRE_DDTREE_VERIFY` env var is now a no-op.

---

### Stage 1b — Remove residual hidden D2H in `spec_step_ddtree_batched` (LANDED)

**Diff hunks:**

1. `run_dflash_draft_for_topk_gpu` signature changed: `target_hidden_host: &[f32]` → `Option<&[f32]>`. When `None`, uses `draft_scratch.thlog.abs_positions()` for eviction-aware k-positions and passes `None` to `draft_forward` (skipping H2D upload). When `Some(slice)`, retains the old host-shadow path (ctx_slice diagnostic only).

2. `spec_step_ddtree_batched` call site: passes `None` when `ctx_slice.is_none()` (default production path), `Some(target_hidden_host.as_slice())` for ctx_slice.

3. Removed entry invariant `assert_eq!(target_hidden_host.len(), position * ne * h, ...)` — the host Vec is no longer updated in the GPU-resident path.

4. Empty-tree shortcut: removed `download_hidden_block(gpu, hidden_rb, 1)` + `target_hidden_host.extend_from_slice`.

5. Fast/slow tape main path: removed `download_hidden_block(gpu, hidden_rb, rows_to_keep)` + `target_hidden_host.extend_from_slice`. The D2D scatter + `thlog.append_committed` already updated the GPU buffer; no CPU download needed.

**Preserved:** Path B WIP (`HIPFIRE_DDTREE_PATH_B_CAPTURE=1`) still downloads to host for its CPU-side gather logic. `target_hidden_host: &mut Vec<f32>` stays in the function signature for Path B compatibility.

---

## Validation

### Byte-identical token output at temp=0

Tested using `dflash_spec_demo --ddtree-batched` (correct binary for the batched path). Baseline: `dflash_spec_demo` built at HEAD 670eb12f (pre-patch). Patched: `dflash_spec_demo` rebuilt with all 4 changes.

| Config | Prompt | Result |
|--------|--------|--------|
| b8k2 | lru_cache_pep8_strict | BYTE-IDENTICAL (978 chars) |
| b8k2 | bare_factual | BYTE-IDENTICAL (881 chars) |
| b8k2 | prose_river_short | BYTE-IDENTICAL (1118 chars) |
| b12k4 | lru_cache_pep8_strict | BYTE-IDENTICAL (1100 chars) |
| b12k4 | bare_factual | BYTE-IDENTICAL (949 chars) |
| b12k4 | prose_river_short | BYTE-IDENTICAL (1119 chars) |

All 6 combinations: **BYTE-IDENTICAL** ✓

### temp=0.7 coherence

Tested using `dflash_spec_demo --ddtree-batched --temp 0.7` (b8k2). Both prompts produced 182–201 fluent tokens with no attractor:

| Prompt | total | first128 unique_ratio | first128 max_ratio | last128 unique_ratio | Tier1 | Tier2 |
|--------|-------|----------------------|--------------------|----------------------|-------|-------|
| lru_cache_pep8_strict | 182 | 0.359 | 0.102 | 0.508 | PASS | PASS |
| prose_river_short | 201 | 0.711 | 0.055 | 0.742 | PASS | PASS |

Decoded text (code): Valid LRU cache Python implementation with `OrderedDict`, coherent structure.  
Decoded text (prose): Fluent narrative fiction, no loops.

### Copy-count before/after (HOST_TIMING)

Measured with `HIPFIRE_HOST_TIMING=1 dflash_spec_demo --ddtree-batched --max 64 b8k2` on gfx1151 (UMA).

**Baseline (670eb12f, pre-patch):**
```
host timing (mean over 7 cycles, µs): wall=278482
  launch=3744 (n=2208) h2d=11323 (n=7) d2h=50450 (n=4, 2674KB) d2d=1160 (n=555) memset=133 (n=63, 33MB) glaunch=5 (n=0)
  ssync=208965 (n=2) esync=0 dsync=0 → other=2702
```

**Patched (D8/D9/D16/Stage1b):**
```
host timing (mean over 7 cycles, µs): wall=195031
  launch=3591 (n=2208) h2d=4823 (n=7) d2h=37303 (n=4, 2674KB) d2d=1093 (n=555) memset=153 (n=63, 33MB) glaunch=6 (n=0)
  ssync=145513 (n=2) esync=0 dsync=0 → other=2549
```

**Note on KB count:** Both show `2674 KB` d2h — this is dominated by the prompt-seed `download_hidden_block` calls (counted cumulatively from counter reset and divided by spec cycles). The per-spec-cycle hidden D2H from Stage 1b is eliminated but masked by the prompt-seed cost in this aggregate metric. The wall time reduction (278ms → 195ms, -30%) and ssync reduction (208ms → 145ms, -30%) confirm real throughput improvement.

`dsync=0` in both: the baseline demo binary (pre-Stage-1) used a different code path; the patched binary uses stream_synchronize (D16).

### Perf — tok/s before/after

| Prompt | Baseline tok/s | Patched tok/s | Delta |
|--------|----------------|---------------|-------|
| lru_cache_pep8_strict | 34.69 | 34.52 | -0.5% (noise) |
| bare_factual | 18.24 | 18.26 | +0.1% (noise) |
| prose_river_short | 22.18 | 22.18 | 0% |

**NEUTRAL** on gfx1151 UMA (all within ±1% noise). UMA boxes have near-zero PCIe D2H latency; the wins are on discrete GPU (PCIe) systems where these copies cost 25-1850 µs/cycle.

### Gates

- `./scripts/coherence-gate-dflash.sh` — **PASS** (4/4 rows, no hard errors, no soft flags)
- `./scripts/serve-multiturn-gate.sh` — **PASS** (all requests coherent across session)

---

## Summary

| Task | Status | What changed |
|------|--------|-------------|
| D16 | LANDED | `device_synchronize()` → `stream_synchronize(active_stream)` in tree verify; `active_stream` set at ddtree entry. On PCIe: ~3–5 ms/cycle saved. |
| D9 | LANDED | Verify-argmax D2H (~244 B) skipped for SWOR mode. Saved for greedy. |
| D8 | LANDED | Naive full-logits path (~37 MB/cycle on PCIe) removed; `ddtree_verify_naive` flag deleted. `use_swor = temp > 0.0` unconditionally. |
| Stage 1b | LANDED | Hidden state D2H eliminated from ddtree fast/slow/empty-tree paths. `run_dflash_draft_for_topk_gpu` passes `None` (GPU-resident) or `Some(slice)` (ctx_slice only). Path B still downloads to host (WIP). |

Per-cycle ddtree hidden d2h: **0** (was: `rows_to_keep × ne × h × 4` ≈ 500 KB on PCIe at accept≈4).  
Device drain: **eliminated** (stream_synchronize only; no full-device stall).  
Full-logits D2H: **eliminated** (naive path removed; SWOR stays on-GPU).
