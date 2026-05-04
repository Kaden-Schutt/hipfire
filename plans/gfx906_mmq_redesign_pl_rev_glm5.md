# Adversarial review: gfx906 MMQ kernel redesign plan

Reviewer: glm-5-turbo adversarial pass
Plan reviewed: `plans/gfx906_mmq_redesign.md` (291 lines)
Date: 2026-05-04

---

## Summary verdict: conditionally approve with 6 required changes

The structural thesis (switch to nwarps=4, template on mmq_x) is sound and
well-supported by the stock comparison data. The plan correctly identifies
that spill avoidance was a red herring and that tile width / arithmetic
intensity is the real lever. However, the plan has several specific errors
in its dispatch logic, under-examines format-level differences, and
underestimates validation complexity. The items below are ordered by
severity.

---

## CRITICAL (must fix before implementation)

### C1. Dispatch formula is wrong

**Plan (line 228):** `mmq_x = next_pow2(batch_size).clamp(8, 64)`

**Stock's actual dispatch** (`mmq.cuh:4069-4082`): a greedy loop stepping by
`granularity=8` (gfx906 non-MFMA path, line 292), picking the largest
`mmq_x <= ncols_max` that produces the fewest column tiles:

```cpp
for (int mmq_x = 8; mmq_x <= mmq_x_max && ntiles_x_best > 1; mmq_x += 8) {
    const int ntiles_x = (args.ncols_max + mmq_x - 1) / mmq_x;
    if (ntiles_x < ntiles_x_best) { mmq_x_best = mmq_x; ntiles_x_best = ntiles_x; }
}
```

This produces different results than `next_pow2`:

| ncols_max | Stock picks | Plan's `next_pow2` | Tile utilization |
|---|---|---|---|
| 128 | 64 | 128 (clamped to 64) | 100% vs 100% |
| 64 | 64 | 64 | 100% vs 100% |
| 48 | **24** | **32** | **100% vs 67%** |
| 32 | 32 | 32 | 100% vs 100% |
| 16 | 16 | 16 | 100% vs 100% |
| 40 | **40** | **64** | **100% vs 63%** |
| 56 | **56** | **64** | **100% vs 88%** |

For N=48 (the alpha/beta matrices in qkvza), stock picks mmq_x=24, which
isn't even in hipfire's candidate set {8, 16, 32, 64}. The plan's
`next_pow2` picks 32, wasting 33% of compute on zero-padding.

**Fix:** Either (a) compile all 8 multiples of 8 in {8, 16, 24, 32, 40,
48, 56, 64} (stock does this — binary size cost is real but compile-kernels.sh
parallelizes), or (b) document the utilization loss and explicitly opt for
simpler dispatch at the cost of qkvza efficiency. Option (a) is strongly
recommended since Q10's whole premise is that qkvza becomes feasible with
small mmq_x support.

### C2. Stock mmq_x_max on gfx906 is 64, not 128

**Plan (line 268):** asks "include {128} from day one?"

**Stock (mmq.cuh:124-125):**
```cpp
#if defined(GGML_USE_HIP)
    return 64;
```

gfx906 has no WMMA/MFMA, so `AMD_WMMA_AVAILABLE` is false and the HIP
path returns 64. The host-side `get_mmq_x_max_host` (line 109-117) also
returns 64 for gfx906 (neither `amd_wmma_available` nor
`turing_mma_available` is true).

**Impact:** The Q1 decision to exclude 128 is correct, but for the wrong
reason. The plan should cite the actual cap from `mmq.cuh` rather than
framing it as an LDS budget question. The 52.7 KiB LDS estimate at
mmq_x=128 (line 119) is moot since stock won't dispatch mmq_x=128 on
this arch.

### C3. Spill count assumption needs empirical grounding

**Plan (line 48):** "Accept vgpr_count ~128, vgpr_spill_count ~144,
private_segment_fixed_size ~500 B"

The 144-spill number is copied from stock's ELF metadata for
`mul_mat_q<Q4_K, mmq_x=64>`. But stock uses **Q4_K** dequantization
(HIP kernel `load_tiles_q4_K` at mmq.cuh:~1100), which has a different
intermediate live-range topology than hipfire's HFQ4 dequantization
(`load_hfq4_tile_dp4a`).

Specifically:
- Q4_K dequant: reads 32 bytes per group, produces int8s via lookup
  tables (`kmask`, `kscale`). Fewer live intermediates.
- HFQ4 dequant: reads 4 bytes at a time, unpacks 8 nibbles through 7
  shift+mask operations into 2 int32s, then subtracts bias constant.
  More live intermediates per iteration (n0..n7, int_a, int_b all live
  simultaneously at lines 189-202 of the current kernel).

At nwarps=4 with full j0 unroll, the compiler must keep the entire
dequant chain live across all j0 iterations simultaneously. Stock's
Q4_K dequant has fewer intermediates, so the compiler may spill less.
Hipfire's HFQ4 dequant could produce **more** than 144 spills.

**Fix:** Change the ELF check from a "verify" to a "measure first." If
spills exceed 200 (not 250 — the 250 threshold is arbitrary), consider:
(a) keeping `#pragma unroll 1` on j0 for HFQ4 specifically (stock's
Q4_K can afford full unroll, HFQ4 may not), or (b) reordering the
dequant to reduce live-range overlap (pipeline the nibble unpack into
the dp4a call instead of materializing all intermediates first).

---

## HIGH (strongly recommended before implementation)

### H1. HFQ4 vs Q4_K format difference is handwaved

**Plan (line 180):** "we should land at the same or better (we have
HFQ4-G256 vs their Q4_K_M; the per-byte accuracy and dequant cost differ
slightly)"

"Slightly" is not quantified. The concrete differences:

| Property | Q4_K (stock) | HFQ4-G256 (hipfire) |
|---|---|---|
| Scale storage per group | 2 bytes (q4_K super-block) | 4 bytes (f32 scale) |
| ZP storage per group | 0 bytes (symmetric) | 4 bytes (f32 zp) |
| Total metadata per 256 K | ~2 B | 8 B |
| Dequant intermediates | 4 (kmask, kscale, sc, m) | 7 (n0..n6, plus int_a, int_b) |
| Dequant ops per 8 elements | 1 table lookup + 1 shift | 7 shifts + 7 masks + 2 subs + 2 packs |

The 6 extra bytes of scale+zp per group means the X-tile loader reads
6 × 128 = 768 extra bytes from HBM per kg iteration vs stock's Q4_K.
At K=4096, that's 16 groups × 768 B = 12 KB extra HBM traffic — small
relative to the 33 KB x_qs read, but non-zero.

More importantly, the 7 extra shift+mask ops per 8 elements in the
dequant chain directly compete with dp4a for VALU issue slots. At
nwarps=4, the compiler may not be able to hide this overhead behind
the memory latency of the next X-tile load.

**Fix:** Before committing to the redesign, run a microbenchmark:
time `load_hfq4_tile_dp4a` alone (no compute) at nwarps=4 with rocprof
`VALUBusy` and `VMEMWrBusy`. If VALU is already saturated by dequant,
the wider tile won't help as much as the plan predicts. If so, consider
fusing the nibble unpack into the dp4a loop body (avoid materializing
int_a/int_b into LDS at all — unpack directly into the sdot4 source
operand).

### H2. Coherence gate gap for gfx906

**Plan (Phase 4, line 233):** "Coherence gate"

The coherence-gate.sh test matrix (lines 83-99) exercises:
`qwen3.5-{0.8b, 4b, 9b, 27b, 35b-a3b, mq3-9b, mq3-27b}` — all loaded
and run through the daemon with models in `~/.hipfire/models/`. These
models are MQ4/MQ3 format for gfx11xx/gfx12xx.

The gfx906 MMQ redesign changes the GEMM kernel used during **prefill**.
The coherence gate tests model loading + decode, not prefill. The gate
would catch a catastrophic kernel bug (wrong output from prefill → bad
KV cache → garbage decode), but it wouldn't catch a subtle numerical
regression that only manifests at specific batch sizes (e.g., mmq_x=32
produces different rounding than mmq_x=8 at the boundary).

**Fix:** Add a gfx906-specific prefill correctness test to the
validation phase:
1. Run the existing `test_gfx906_mmq_correctness` synthetic test at
   each mmq_x value (8, 16, 24, 32, 40, 48, 56, 64) and compare NRMSE
   against the FP16 wave64 reference.
2. Run `test_gfx906_mmq_realdata` with dumped real weights/activations
   at pp128.
3. Run coherence-gate.sh on a machine with a gfx906 GPU (MI50) using
   a gfx906-compiled binary. If no MI50 is available, document this
   as a known gap and gate the merge on CI once gfx906 CI exists.

### H3. gate_up should be co-designed, not deferred

**Plan (Phase 5, line 238-240):** gate_up port is "Commit 2: re-port
gate_up to use the new dispatch"

The comparison doc (line 111-118) shows:
- Residual MMQ: 1.3 ms/call vs stock 0.85 ms → **1.5× gap**
- gate_up FP16 wave64: **12 ms/call** vs stock's MMQ on same shape → **~14× gap**
- Total hipfire GEMM time: 944 ms (256 calls) vs stock 451 ms (528 calls)

**gate_up accounts for most of the end-to-end gap.** Shipping the
residual redesign alone (closing the 1.5× gap) improves residual from
1.3 ms to ~0.85 ms, saving maybe 0.45 ms × 128 residual calls = 57 ms.
But gate_up at 12 ms × ~64 calls = 768 ms dominates the 944 ms total.
Even halving gate_up (to 6 ms via MMQ) saves 384 ms — 6.7× more than
the residual fix.

The dispatch code (dispatch.rs:6304-6383) for `gemm_hfq4g256_mmq_set_gfx906`
shares the same MMQ_X, MMQ_Y, and LDS constants as the residual path.
If the residual redesign changes these constants (it does — MMQ_X
becomes dynamic, block dim changes), the set-mode dispatch breaks
silently unless updated simultaneously.

**Fix:** Include gate_up dispatch update in Phase 3 (not Phase 5).
The kernel itself can be shared (same `mmq_body` with `add=0`), so
the only additional work is the dispatch symbol lookup change — which
must happen at the same time as the residual dispatch change to avoid
a broken intermediate state.

### H4. Phase 0 nwarps contradiction is unresolved

The original MMQ plan (`plans/gfx906_mmq_plan.md`, Phase 0 complete)
recommended `nwarps=2` based on `llama.cpp-gfx906`'s `gfx906-config.h`
setting. The redesign adopts `nwarps=4` based on runtime observation
that stock's actual kernel uses 4 warps.

The Phase 0 document was approved as "GO" and shipped. The redesign
overturns a foundational assumption of that approved plan without
acknowledging the contradiction.

**Fix:** Add a section to the redesign plan: "Relationship to Phase 0
plan." State explicitly: (a) Phase 0's nwarps=2 recommendation was
based on a build-time config file that doesn't reflect the runtime
kernel selection, (b) the stock comparison doc (`2026-05-04-llamacpp-stock-comparison.md`)
is the authoritative source, (c) Phase 0's other conclusions (wave64
indexing, LDS bank conflict padding) remain valid.

---

## MEDIUM (recommended to address)

### M1. j0 full-unroll reversal lacks analysis

**Plan (Q7, line 174):** "full unroll on j0. Match stock; accept the
spills the unroll causes."

Commit 17c05f3 showed that `#pragma unroll 1` on j0 brought +16.2%
(125.2 → 145.5 tok/s) by cutting live-range pressure from 64 to 16
concurrent sites. The plan now recommends reverting this at nwarps=4.

The argument is "stock does it." But stock's Q4_K dequant has fewer
intermediates (see H1), so stock can afford the unroll. HFQ4's heavier
dequant chain may make the unroll catastrophic at nwarps=4.

**Recommendation:** Implement both variants. Compile with `#pragma
unroll 1` as the default for HFQ4, and add a `#pragma unroll` variant
behind a compile-time flag. Benchmark both. Only ship full unroll if
the ELF shows spills < 200 AND the benchmark shows improvement.

### M2. ds_read_b128 prior negative result is underweighted

**Plan (Q8, line 185):** "emit ds_read_b128 from day one"

The junroll doc (Follow-up 2) showed ds_read_b128 regressed wallclock
by 2.7% on the current nwarps=2 mmq_x=8 kernel. The plan argues the
larger kernel body at nwarps=4 mmq_x=64 changes the bottleneck. This
is plausible but untested.

**Recommendation:** Implement ds_read_b128 as a separate commit, not
"from day one." Ship the redesign without it first, benchmark, then
add ds_read_b128 and benchmark again. The 2.7% regression on a small
kernel doesn't prove it'll help on a large kernel — it only proves the
compiler's LDS-to-VGPR forwarding path is good on this ROCm version.

### M3. X-tile loader threading analysis is incomplete

**Plan (Q4, line 138):** "Load phase is ~250 chunks × 4 B per row × 128
rows = 128 KB of HBM read per kg, dwarfed by the 4 ms of compute."

This calculation is for the current nwarps=2 kernel. At nwarps=4, the
X-tile load is the same (33,280 B of x_qs + 1,024 B of x_dm = 34.3 KB
per kg), but:
- 128 of 256 threads are idle during X load (50% waste)
- The X load happens once per kg iteration (not once per Y-half)
- At K=4096 with 256-K groups, there are 16 kg iterations → 16 × 34.3
  KB = 549 KB of HBM read for X alone

With 128 active threads, that's 549 KB / 128 threads = 4.3 KB/thread
of HBM reads during X load. At MI50's ~1 TB/s HBM bandwidth, that's
~4.3 us per kg for X load. Across 16 kg iterations: 69 us. The
current per-call time is 1.3 ms, so X load is ~5% — indeed small.

But at nwarps=4 mmq_x=64, the per-call time should drop to ~0.85 ms
(target). 69 us / 850 us = 8%. Still small, but the "dwarfed" framing
is weaker. The idle-thread waste during X load is real and grows
relative to the shrinking compute time.

**Recommendation:** Keep option (a) for v1, but add a TODO to measure
X-load contribution with rocprof `VALUBusy` during the first barrier
after X load. If it exceeds 15% of per-call time in v1, prioritize
option (b) for v2.

### M4. Existing test infrastructure needs updating

The plan's Phase 2 emits 12 entry symbols (3 modes × 4 mmq_x values).
But:
- `test_gfx906_mmq_correctness.rs` references the old symbol names
  (`gemm_hfq4g256_residual_mmq_gfx906` etc.)
- `test_gfx906_mmq_realdata.rs` similarly references old symbols
- `dispatch.rs` (lines 6202-6297) hardcodes MMQ_X=8, block dim
  [64, 2, 1], and LDS=35,456

All three must be updated as part of Phase 2/3, not left as "will
update later." The plan's Phase 3 estimate of "0.5 day" doesn't
account for updating the test harnesses.

### M5. Timeline is aggressive

Phase 2 (kernel rewrite): 1-2 days. Phase 3 (dispatch): 0.5 day.
Phase 4 (validation): 0.5 day. Total: 2-3 days.

For comparison, the current kernel (455 lines) took multiple sessions
across multiple plans to reach its current state. The redesign:
- Changes topology (nwarps=2→4, block dim, launch bounds)
- Adds template parameterization (12 entry symbols)
- Rewrites accumulator indexing
- Changes tile loader threading
- Potentially adds ds_read_b128
- Requires correctness validation at 8 mmq_x values
- Requires dispatch update for both residual and set-mode paths

3-4 days is more realistic. The plan should budget for the case where
the first ELF check shows spills > 200 (see C3), which would require
a design pivot.

---

## LOW (nice to have)

### L1. Q8_1 tile loader edge case at mmq_x=8

**Plan (Q5, line 151):** "distribute by tid, each thread loads
(mmq_x * Y_STRIDE) / total_threads ints"

At mmq_x=8, nwarps=4: 8 × 36 = 288 ints / 256 threads = 1.125
ints/thread. 32 threads get 2 ints, 224 threads get 1 int, and the
last 0 threads get 0. The plan doesn't address the non-uniform
distribution. The current code (line 231) handles this with an
`if (tid < 32)` guard. The new code needs equivalent guarding.

### L2. MMQ_TILE_Y_K terminology

**Plan (line 123):** "Adopt stock's MMQ_TILE_Y_K = 36 ints per col"

Stock doesn't use the term "MMQ_TILE_Y_K." Stock uses `MMQ_TILE_Y_K`
for the K-dimension of the Y tile (32 elements per sub-block), not the
per-column stride. The 36-int stride in stock is the Y_STRIDE (4 ds
+ 32 qs). The plan's current kernel already uses Y_STRIDE = 36. This
is a terminology confusion, not a design error.

### L3. Risk register incompleteness

Missing risks:
- **Correctness regression from template parameterization.** The
  current bounds-checked kernel has battle-tested edge handling
  (row0 + i >= M, col0 + j >= N). Template instantiation with
  different mmq_x values may expose edge cases not hit at mmq_x=8.
- **Symbol name collision.** 12 new symbols × potential for typos in
  the suffix (`_x8` vs `_x08`, `_full_add` vs `_fulladd`). Add a
  compile-time or link-time assertion that all expected symbols exist.
- **ROCm version sensitivity.** The plan targets ROCm 6.4.3 (per the
  comparison doc). The `__launch_bounds__(256, 2)` behavior and spill
  heuristics may differ on ROCm 6.3 or 6.5. Document the minimum
  ROCm version.
- **Occupancy regression.** At 256 threads/WG with `__launch_bounds__(256, 2)`,
  max occupancy is 2 WGs/CU. With spills, the actual occupancy may
  drop to 1 WG/CU on some CUs (e.g., if scratch memory per WG exceeds
  the per-CU budget). Verify occupancy via rocprof, not just ELF.

### L4. The plan doesn't address the nwarps=2 kernel's fate

Should the current `gemm_hfq4g256_residual_mmq_gfx906.hip` be:
- Replaced entirely by the new kernel?
- Kept as a fallback behind a config flag?
- Deleted and the new file takes its place?

The compile-kernels.sh script (which selects kernel files by arch)
needs updating. If both files exist, the compiler may pick the wrong
one.

### L5. 400 tok/s target framing

**Plan (Goal, line 16):** "End-to-end target: prefill within 20% of
stock, i.e. >=400 tok/s on Qwen 3.5 9B pp128."

Stock's 500.5 tok/s is measured with Q4_K_M quant. Hipfire uses
HFQ4-G256. If the format difference costs even 5-10% (see H1), the
ceiling is 450-475 tok/s. The 400 tok/s target is achievable but
framing it as "within 20% of stock" is misleading because the formats
are different. A more honest target: "match stock's kernel-level
per-call latency (0.85 ms at K=4096), which projects to ~480 tok/s if
format overhead is negligible, ~430 tok/s with 10% format overhead."

---

## Required changes before implementation

1. **Fix dispatch formula (C1):** Match stock's greedy loop, not
   `next_pow2`. Compile all 8 multiples of 8 from 8 to 64. This
   changes the symbol count from 12 to 24 (3 modes × 8 mmq_x values).
   If 24 symbols is too many, start with {8, 16, 24, 32, 48, 64}
   (skip 40 and 56 — they're rarely the optimal pick).
2. **Remove mmq_x=128 question (C2):** Stock caps at 64 on gfx906.
   Update Q1 and the decision points section.
3. **Add empirical spill budget (C3):** ELF check after first build
   is a hard gate, not a soft verify. If spills > 200, pivot to
   `#pragma unroll 1` on j0 before proceeding.
4. **Quantify HFQ4 vs Q4_K dequant cost (H1):** Run a standalone
   dequant microbenchmark before committing to full unroll.
5. **Add gfx906-specific coherence validation (H2):** Update Phase 4
   to include per-mmq_x NRMSE tests and a coherence gate run on
   gfx906 hardware.
6. **Co-design gate_up dispatch (H3):** Include set-mode dispatch
   update in Phase 3, not Phase 5.

---

## What the plan gets right

- The nwarps=4 structural thesis is correct and well-evidenced.
- The non-negotiables (items 1-9) are all reasonable given the stock
  comparison data.
- The LDS budget analysis (Q3) is correct and shows comfortable
  headroom under 64 KiB.
- The Y-twice (X-once) pattern (Q6) is correctly inherited from stock.
- The "what this plan is NOT doing" section (lines 255-264) is
  well-scoped and prevents scope creep.
- The risk register exists (many plans skip this entirely).
- The implementation phases are clearly separated.

---

*Review based on: plan text (291 lines), current kernel (455 lines),
stock mmq.cuh (4176 lines, read in full), comparison doc (195 lines),
attribution/junroll/spill-reduction docs, original MMQ plan (439 lines),
CLAUDE.md (350 lines), dispatch.rs (gfx906 sections), test harnesses,
and coherence-gate.sh.*
