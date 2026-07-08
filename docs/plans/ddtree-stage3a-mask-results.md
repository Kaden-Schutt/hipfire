# DDTree Stage 3a: On-GPU attn_bias Mask — Results

**Branch:** feature/speculator-ddtree  
**Base commit:** 1317d16f  
**Date:** 2026-06-30  
**Box:** gfx1151 (Strix Halo UMA, ~96 GB)

---

## 1. Change Summary

Eliminated the per-cycle `attn_bias` H2D copy (D4 in the scope doc, ~15 KB at
`big_n=61`) by replacing it with a new GPU kernel `ddtree_build_attn_mask_f32`
that reads the already-uploaded `parent_indices` and writes `attn_bias` directly
on-device.

**Files changed:**
- `kernels/src/ddtree_build_attn_mask.hip` — NEW 50-line HIP kernel
- `crates/rdna-compute/src/kernels.rs` — added `DDTREE_BUILD_ATTN_MASK_SRC` constant
- `crates/rdna-compute/src/sampling.rs` — added `Gpu::ddtree_build_attn_mask_f32()` dispatch wrapper
- `crates/hipfire-arch-qwen35/src/speculative.rs` — removed mask H2D; always upload `parent_indices`, launch kernel; added `HIPFIRE_DDTREE_ASSERT_MASK=1` dual-path check

**Not changed:** `linearize_tree_with_parents` is still called (tokens/positions/parent_host still needed); its `mask_host` return value is only used by `HIPFIRE_DDTREE_ASSERT_MASK=1`. The host tree build is entirely intact.

**Kernel algorithm (per thread i, grid=[big_n,1,1], block=[big_n,1,1]):**
1. Fill row `i` with `-INFINITY`
2. Walk `j = i; while j >= 0: row[j] = 0.0f; j = parent_indices[j]`

Exactly mirrors the host `visibility[i][j] = visibility[parent_slot][j] || (j==i)` bottom-up pass + row-major flatten.

---

## 2. Mask Byte-Equality (HIPFIRE_DDTREE_ASSERT_MASK=1)

Ran dual-path mode on:
- `lru_cache_pep8_strict.txt`, budget=8 topk=2: **20+ cycles — ALL byte-identical**
  (big_n=9, 81 floats each)
- `trains-meet.txt`, budget=12 topk=4: **38 cycles — ALL byte-identical**
  (big_n=13, 169 floats each)

No panics, no mismatches at any cycle across all runs.

---

## 3. Byte-Identical ddtree Output at temp=0 (BEFORE vs AFTER)

Target: qwen3.6-27b.mq4  
Draft: qwen36-27b-dflash-mq4.hfq  
Flags: `--no-chatml --kv-mode q8 --max 200 --temp 0.0`

### Budget=8, topk=2

| Genre | BEFORE (HEAD 1317d16f) | AFTER | Match |
|-------|------------------------|-------|-------|
| code (lru_cache_pep8_strict) | [260, 413, 1328, 303, ...] (160 tokens) | identical | PASS |
| reason (trains-meet) | [2592, 264, 5257, 10583, ...] (160 tokens) | identical | PASS |
| prose (fiction_lighthouse) | [271, 248068, 198, 90700, ...] (166 tokens) | identical | PASS |

### Budget=12, topk=4

| Genre | BEFORE | AFTER | Match |
|-------|--------|-------|-------|
| code | [260, 413, 1328, 303, ...] (160 tokens) | identical | PASS |
| reason | [2592, 264, 5257, 10583, ...] (160 tokens) | identical | PASS |
| prose | [271, 248068, 198, 90700, ...] (166 tokens) | identical | PASS |

**All 6 combos: byte-identical. Zero divergences.**

---

## 4. H2D Copy Count (attn_bias D4 gone)

Config: `--ddtree-batched --ddtree-budget 12 --ddtree-topk 4 --temp 0.7 --max 64`  
`HIPFIRE_HOST_TIMING=1`

| State | n_htod (per cycle) | h2d (µs) |
|-------|-------------------|----------|
| BEFORE (1317d16f) | **n=10** | 11344 |
| AFTER (Stage 3a) | **n=9** | 11479 |

The `attn_bias` H2D is gone: exactly 1 H2D copy removed per cycle.

Note: timing difference is noise on UMA (all H2D ops are effectively free — the
dominant cycle cost ~218 ms is the verify forward + device_sync, unchanged by this
stage). On dGPU PCIe, the 15 KB H2D would save ~0.75 µs/cycle (negligible vs
the 3-5 ms device_sync).

---

## 5. Gate Results

| Gate | Result |
|------|--------|
| `./scripts/coherence-gate-dflash.sh` | **PASS** — no hard errors |
| `./scripts/serve-multiturn-gate.sh` | **PASS** — all requests coherent across session |

---

## 6. Notes

- `HIPFIRE_DDTREE_ASSERT_MASK=1` is left in the code behind the env gate (off by default).
  Cost when enabled: one `device_synchronize` + 1 D2H per cycle = adds ~2-5 ms/cycle.
  Safe to leave for debugging; zero cost in production.
- `parent_indices` is now always uploaded (was only when `ddtree_tree_la=true`).
  This adds 244 B H2D on cycles where `ddtree_tree_la=false`, which is a no-op on UMA.
  On PCIe this is sub-microsecond. The mask kernel requires `parent_indices` to be
  resident regardless of tree_la.
- Stage 3b (full GPU tree build, D5 + D4 + D6/D7/D10-D12) remains deferred per plan.
