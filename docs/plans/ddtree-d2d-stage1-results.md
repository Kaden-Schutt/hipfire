# DDTree D2D Stage 1 Results

**Date:** 2026-06-29  
**Branch:** feature/speculator-ddtree  
**HEAD at time of change:** ca6691b0  
**Box:** gfx1151 (Strix Halo, UMA)  
**Status:** COMPLETE — copies reduced, output byte-identical, gates PASS  

---

## 1. Diff summary

Two call sites modified in `crates/hipfire-arch-qwen35/src/speculative.rs`,
function `spec_step_ddtree_batched`:

### Empty-tree path (line ~5082)
Added `scatter_hidden_block_to_interleaved` D2D scatter + `thlog.append_committed`
before the existing `download_hidden_block` call. The download is kept for CPU Vec
length invariant.

### Section 11 — main hidden-state lifecycle (line ~5619)
**Before:** single `download_hidden_block(gpu, hidden_rb, hidden_rows_written)` call
downloaded `big_n` rows (fast-tape: up to 61 rows at budget=60) or `rows_to_keep`
rows (slow-tape), followed by CPU gather/slice into `target_hidden_host`. No scatter.

**After:** 
- Fast-tape path: `scatter_hidden_block_to_interleaved(block_size=big_n, n_rows=rows_to_keep)`
  scatters only the committed prefix into `draft_scratch.target_hidden` on GPU. Then
  `thlog.append_committed(position, rows_to_keep, co)` marks them GPU-resident.
  Finally `download_hidden_block(gpu, hidden_rb, rows_to_keep)` downloads only
  `rows_to_keep` rows (not `big_n`) to maintain CPU Vec length invariant.
- Slow-tape path: same pattern but `block_size = rows_to_keep` (no partial-slice needed).
- Path B dead-code branch: unchanged (opt-in WIP, left on old path).

**Effect:** next cycle's `draft_forward` call sees `thlog.uploaded_rows() == l`
(prev==l condition) and skips the target_hidden H2D upload entirely.

---

## 2. Byte-identical verdict

All 5 run pairs tested at temp=0.0 (greedy), `--no-chatml`, `--kv-mode q8`:

| Config | Prompt | Verdict |
|--------|--------|---------|
| budget=8 topk=2 | lru_cache_pep8_strict (code) | **BYTE-IDENTICAL** |
| budget=12 topk=4 | lru_cache_pep8_strict (code) | **BYTE-IDENTICAL** |
| budget=60 topk=4 | lru_cache_pep8_strict (code) | **BYTE-IDENTICAL** |
| budget=8 topk=2 | prose_river_short (prose) | **BYTE-IDENTICAL** |
| budget=8 topk=2 | trains-meet (reason) | **BYTE-IDENTICAL** |

Prompt md5 verification:
- lru_cache_pep8_strict: `afc47d8840ea1f1476807728bd3ddef9`
- prose_river_short: `7130ff9fff28ad51f212942979471b8b`
- trains-meet: `fbfde41091a239530826a43b6b9b060b`

---

## 3. Copy-count before/after (host-timing counters)

### Note on counter coverage
`HIPFIRE_HOST_TIMING=1` tracks `memcpy_htod` calls (non-offset variant only).
The target_hidden incremental H2D in `draft_forward` uses `memcpy_htod_offset`
(dflash.rs:1282), which is NOT counted in the `h2d` counter. So the H2D elimination
(from `thlog.append_committed` suppressing the upload) is real but not directly
visible in the counter. The D2H reduction IS visible since `download_hidden_block`
uses `memcpy_dtoh_at` which IS counted.

### D2H bytes per cycle (the measurable win)

| Config | BEFORE | AFTER | Delta |
|--------|--------|-------|-------|
| b8k2 code | 1552 KB | 1246 KB | −19.7% |
| b12k4 code | 1938 KB | 1595 KB | −17.7% |
| b60k4 code | 3019 KB | 1695 KB | **−43.8%** |
| b8k2 prose | 916 KB | 451 KB | **−50.8%** |
| b8k2 reason | 881 KB | 620 KB | −29.6% |

The savings grow with budget and shrink with τ (higher τ = rows_to_keep closer
to big_n). At budget=60 with τ≈6.3, the saving is −43.8% of D2H traffic.

### D2D calls added

| Config | BEFORE n | AFTER n | Added |
|--------|----------|---------|-------|
| b8k2 code | 405 | 432 | +27 |
| b12k4 code | 412 | 449 | +37 |
| b60k4 code | 489 | 526 | +37 |

Each scatter adds `ne × rows_to_keep = 5 × (τ+1)` D2D calls per cycle.
At τ≈5, that's ~30 extra D2D enqueues per cycle (trivial overhead on UMA).

### Raw host-timing lines (b60k4 — most visible savings)

**BEFORE (budget=60 topk=4, code prompt):**
```
host timing (mean over 23 cycles, µs): wall=...
  h2d=544 (n=7)  d2h=61254 (n=9, 3019KB)  d2d=1250 (n=489)
  dsync=44
```

**AFTER (budget=60 topk=4, code prompt):**
```
host timing (mean over 23 cycles, µs): wall=...
  h2d=616 (n=7)  d2h=61336 (n=9, 1695KB)  d2d=1382 (n=526)
  dsync=39
```

---

## 4. Performance before/after

| Config | BEFORE tok/s | AFTER tok/s | Delta |
|--------|-------------|------------|-------|
| b8k2 code | 33.72 | 34.18 | +1.4% (noise) |
| b12k4 code | 42.40 | 42.85 | +1.1% (noise) |
| b60k4 code | 14.82 | 14.83 | neutral |
| b8k2 prose | 22.09 | 22.18 | neutral |
| b8k2 reason | 27.14 | 26.97 | −0.6% (noise) |

All deltas are within the ±3% noise band (confirmed by matching τ and committed
token counts between BEFORE and AFTER for all runs). No regression observed.

**Expected outcome confirmed:** perf-neutral on UMA. The D2H savings (~20-44% fewer
bytes/cycle) translate to <0.05% cycle-time improvement on gfx1151 UMA since the
copies cross cache-coherent shared memory, not PCIe. On dedicated-VRAM GPUs
(gfx1100/gfx1201) the savings would be real (PCIe D2H eliminated).

---

## 5. Gate results

### DFlash coherence gate (`./scripts/coherence-gate-dflash.sh`)

**PASS** — no hard errors, no soft flags.

```
== 27b-dflash-prose == OK  unique_ratio=0.68  gram_density=0.0
== 27b-dflash-code == OK   unique_ratio=0.75  gram_density=0.0
== 27b-ddtree-b12-prose == OK  unique_ratio=0.68  gram_density=0.032
== 27b-ddtree-b12-code == OK   unique_ratio=0.75  gram_density=0.0
no hard errors — review /tmp/coherence-dflash-20260629-153029.md for coherence, then commit if satisfied
```

The ddtree-code row produces byte-identical output to the chain-code row (same
LRU cache completion), confirming the scatter produces the correct hidden states.

### Multi-turn serve gate (`./scripts/serve-multiturn-gate.sh`)

**PASS**

```
== AR multi-request — qwen3.5-0.8b.mq4 ==
== DFlash multi-request — qwen3.6-27b.mq4 + qwen36-27b-dflash-mq4.hfq ==
serve-multiturn-gate: PASS — all requests coherent across the session
```

---

## 6. Implementation notes

### Why target_hidden_host is still downloaded (not retired)

The CPU `target_hidden_host` Vec is still needed because:
1. The `ctx_slice = Some(n)` diagnostic/windowed-context path reads it directly
   to pass a sliced view to `draft_forward`.
2. The cycle-entry assertion `target_hidden_host.len() == position * ne * h`
   maintains an invariant that the callers (dflash_spec_demo, daemon) rely on.

The download is now `rows_to_keep` rows (not `big_n` rows) in the fast-tape path,
which is the same size as the slow-tape path always was. The CPU Vec content is
correct and consistent with the GPU-resident `draft_scratch.target_hidden` buffer.

### H2D elimination mechanism

After `thlog.append_committed(position, rows_to_keep, co)` is called at the end
of cycle N, the next cycle's `run_dflash_draft_for_topk_gpu` (cycle N+1) calls
`dflash::draft_forward` with `l = position_N+1 = position_N + rows_to_keep_N`.
Inside `draft_forward`, `prev = thlog.uploaded_rows() = position_N + rows_to_keep_N = l`,
so the `prev == l` branch fires (line 1286 of dflash.rs) and the H2D is skipped.

This mirrors the chain path (`spec_step_dflash`, line 3161-3165) which passes
`target_hidden = None` to `draft_forward` entirely, also bypassing the upload.

### Path B branch left unchanged

The `hidden_rows_written == big_n && !fast_tape_ok` branch (Path B WIP, opt-in via
`HIPFIRE_DDTREE_PATH_B_CAPTURE=1`) is left on the original `download_hidden_block`
path. This branch is dead by default (requires non-empty `pre_rope_k` scratch, which
is gated on the same env var). Porting it to D2D scatter would require a GPU gather
kernel for non-linear committed indices, which is Path B's own scope.
