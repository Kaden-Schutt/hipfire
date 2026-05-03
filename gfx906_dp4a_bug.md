# gfx906 dp4a MMQ — bug analysis

Status: **bug not localized** despite extensive investigation. Kernel produces
~5× higher NRMSE on real Qwen weights (0.60% vs 0.12% on synthetic) and
this 0.6% per-layer noise compounds catastrophically through 32 layers.
Output is gibberish.

This document captures all analysis, hypotheses tested, and remaining
candidates for future debug sessions.

## TL;DR

- **Standalone correctness test**: PASSES at all production shapes
  (M=4096, K∈{4096, 12288}, N∈{21,32,36,64,128,200}). NRMSE 0.12-0.6%.
- **Real Qwen 9B weights, in-process MMQ vs FP16 wave64**: NRMSE 0.60%
  (= 5× synthetic's 0.12% at the same shape).
- **Production with `HIPFIRE_MMQ=1`**: gibberish output. Even firing
  MMQ for **just one** of the 64 residual GEMM calls per prefill is
  enough to break model coherence.
- **Diagnostic flags localized the bug to the dp4a kernel body**
  (`HIPFIRE_MMQ_DIAG_PASSTHROUGH=1` and `_QUANTIZE_ONLY=1` both
  produce coherent output).
- **All 9 cells with abs error >1e-2 are in row 3994** (out of 4096).
  All other rows have max abs error ~2e-3, which is normal Q8_1 noise.

## Setup

- Hardware: MI50 (gfx906)
- Model: Qwen 3.5 9B (HFQ4-G256 quant)
- Reference path: `gemm_hfq4g256_residual_fp16_wave64` (verified
  correct via existing coherence battery).
- New path: `gemm_hfq4g256_residual_mmq_gfx906` (this bug).
- Files:
  - Kernel: `kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip`
  - Dispatch: `crates/rdna-compute/src/dispatch.rs::gemm_hfq4g256_residual_mmq_gfx906`
  - Standalone test: `crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs`
  - Real-data test: `crates/rdna-compute/examples/test_gfx906_mmq_realdata.rs`

## Reproduction

```bash
# Coherent baseline:
HIPFIRE_KV_MODE=asym3 /tmp/coh_test.sh
# Output: clean reasoning answer.

# Gibberish (MMQ on, full kernel):
HIPFIRE_KV_MODE=asym3 HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=0 /tmp/coh_test.sh
# Output: e.g. "15 16 17 9 9 9 9 9 9 ..." or just "<|im_end|>"

# Gibberish even for just call 0:
HIPFIRE_KV_MODE=asym3 HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=0 \
  HIPFIRE_MMQ_CALL_FILTER=0:1 /tmp/coh_test.sh

# Coherent (kernel skipped):
HIPFIRE_KV_MODE=asym3 HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=0 \
  HIPFIRE_MMQ_DIAG_PASSTHROUGH=1 /tmp/coh_test.sh
```

`/tmp/coh_test.sh` is a one-shot daemon test for the 9B reasoning prompt.

## Smoking gun: row 3994

Real-data test result (M=4096 K=4096 N=36, dumped from production
call 0):

| Stat | Value |
|---|---|
| NRMSE | 0.6036% |
| max_abs_err | 1.4032e-2 (at col=7, row=3994) |
| max_rel_err | 40895% (worst-case ratio on tiny ref values) |
| 80% of cells | abs_err ∈ [1e-4, 1e-3] (Q8_1 quantization noise) |
| 9 cells | abs_err > 1e-2 — **ALL in row 3994** |

Top 20 worst rows by max abs error:
```
#0: row=3994 max_err=1.4032e-2  ← outlier
#1: row=676  max_err=2.2253e-3
#2: row=2384 max_err=2.0971e-3
#3-19: rows scattered, max_err ~1.8-2.1e-3
```

**Row 3994 has 7× the max abs error of the next-worst row.** All other
rows show typical Q8_1 noise (~2e-3). Row 3994 is anomalous.

But: row 3994 is also the row with the **largest output magnitude**
(~5.0 vs typical ~0.05). Per-cell relative error at row 3994 (~0.3%)
is actually *lower* than at row 0 (~2.3% on tiny values).

So it's not "row 3994 is broken." It's more like:
- All rows have ~1-3% relative error per cell (Q8_1 noise).
- Row 3994 happens to have large absolute values, so relative error
  gives large absolute error.
- In an absolute sense the model probably *also* fails on small-output
  rows because the relative error is the same.
- 1-3% relative error per cell, summed/multiplied through 32 layers,
  blows up.

## The compounding question

If 0.6% per-layer NRMSE breaks the model in 32 layers, **why does the
RDNA3 i8-WMMA MMQ work?** It uses the same Q8_1 quantize and same
int8×int8 → int32 dot product mathematics. Both kernels should have
identical per-layer NRMSE in theory.

Possibilities:
1. **My dp4a path has a subtle bias** that the i8-WMMA path doesn't.
   E.g. order of operations differs and FP16 conversion artifacts
   accumulate differently.
2. **Strix Halo is also fragile** but the gfx1100 user community
   doesn't notice because they use the path with HIPFIRE_MMQ_SCREEN=1
   default, which checks per-row max error. **Possibly the screening
   is rejecting most weights in real models.** Untested.

## What we've ruled out

- **Dispatch wiring**: `HIPFIRE_MMQ_DIAG_PASSTHROUGH=1` (skip the
  kernel entirely) produces coherent output. Wiring is correct.
- **Quantize kernel**: `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1` (run quantize
  then forward to FP16 wave64) produces coherent output. Quantize OK.
- **HipGraph**: daemon doesn't use hipGraph; gibberish reproduces
  without it.
- **Cross-process state corruption**: in-process MMQ produces same
  numerical result as out-of-process (NRMSE=0.6036% in both).
- **Boundary handling for OOB columns**: tried zeroing OOB col data
  in `load_q8_1_tile` (was clamping to N-1). Made no difference to
  the production bug or the row-3994 error.
- **Wave64 indexing**: Phase 0 verified correct (0 errors over 256
  thread probes).
- **VGPR overflow**: kernel uses 128 VGPRs/thread = exactly the
  occupancy=2 max for gfx906. Not over.
- **LDS overflow**: 43 KB used / 64 KB cap. All writes verified within
  bounds (max x_qs index 8318/8320, max tile_y index 2303/2304).
- **Single-call vs cumulative**: the bug fires on **just one call** of
  64. Cumulative state isn't the issue.
- **K size**: bug reproduces with `HIPFIRE_MMQ_K_FILTER=4096` AND
  `HIPFIRE_MMQ_K_FILTER=12288`. Not K-specific.

## What we've not yet ruled out

### 1. Sign-extension in nibble unpack

```cpp
const unsigned int n0 = (qs0 >>  0) & 0xFu;
...
const int int_a = (int)(n0 | (n1 << 8) | (n2 << 16) | (n3 << 24));
```

`__builtin_amdgcn_sdot4` interprets each byte as **signed int8**. My
nibbles are 0..15 stored in the low 4 bits with the high 4 bits zero,
so each byte is 0..15 = positive int8. **Should be safe.**

But: **the compiler might emit `v_or_b32` with sign-extending shifts**
or rearrange the unpack in a way that produces sign-extended values.
**Untested via ISA dump.**

To verify:
```bash
hipcc -O3 --offload-arch=gfx906 -S -o /tmp/k.s \
  kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip
# Look at load_hfq4_tile_dp4a body for unpack instructions.
```

### 2. Per-element scale/zp formula divisor

llama.cpp's Q4_0 path: `return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);`
llama.cpp's Q8_0×Q8_1: `return sumi*d8d8 + m8s8 / (QI8_1 / vdr);`
Mine: `sum[idx] += scale_w * d_x * (float)sumi + zp_w * sum_x;`

For Q8_1 with `vdr=8` and `QI8_1=8`, the divisor `QI8_1/vdr = 1`. So
no divisor needed. **My formula matches llama.cpp's exactly when vdr=QI8_1.**

But: if I miscount what "vdr" means here vs what llama.cpp counts,
the divisor could be off by 4 or 8. **Worth re-verifying carefully.**

In particular, llama.cpp's `vec_dot_q4_0_q8_1_dp4a` calls `vec_dot_q4_0_q8_1_impl`
with `<VDR_Q4_0_Q8_1_MMQ>` = 4, but each impl call processes both low
and high nibbles → 8 dp4a calls total per impl invocation, covering
2*vdr*4 = 32 K-elements. My kernel does `vdr=8` dp4a calls covering
8*4 = 32 K-elements per inner block. **Same K-coverage but different
"vdr" semantics.**

If "vdr" in the bias divisor is actually `VDR_Q4_0_Q8_1_MMQ=4`
(the "K-block stride per impl call divided by QR=2") rather than my
8 dp4a calls per impl... then divisor `QI8_1 / vdr` could be `8/4=2`
not `8/8=1`. **Possible factor-of-2 error in the bias term.**

**Action**: re-derive the bias term from first principles using the
algebraic definition rather than copying llama.cpp's expression.

### 3. K=4096 corner case

Production has K=4096. `groups_per_row = K/256 = 16` HFQ4 groups per
row. The kg loop runs 16 times. Each group's X tile is loaded once,
Y is loaded twice (kb=0,1). 16 groups × 2 Y reloads = 32 Y reloads
per output tile. 32 layers × 2 residual GEMMs × 32 Y reloads × ... lots
of LDS pressure but no obvious correctness issue.

### 4. Real weights have specific edge cases not in synthetic

Row 3994 weights have scale ~6e-2 and zp ~-5e-1, similar to other rows.
Synthetic test passes with similar weight magnitudes. No obvious data
edge case identified.

### 5. ALL rows have ~1-3% per-cell relative error and that's enough

This is the most-likely scenario based on observation. The 0.6% NRMSE
is "Q8_1 quantization noise that compounds through 32 layers."
The fix would be **better quantization** (Q8_0 instead of Q8_1, or
F16 activations) — not a kernel bug.

Test: run the same daemon test with the **RDNA3 i8-WMMA MMQ path** on
gfx1100 and see if it produces coherent output despite similar
quantization noise. If yes, the bug IS in my kernel. If no, the
issue is structural to Q8_1 MMQ in this model and we need a
different approach.

We can't test this directly without gfx1100 hardware in the loop,
but we can examine the per-row error distribution of the existing
RDNA3 MMQ on gfx906 (using `mmq_screen_weight` extended to gfx906).

## Suggested debug paths (ranked)

1. **Dump kernel ISA for `load_hfq4_tile_dp4a`** to verify nibble
   unpack is zero-extended, not sign-extended.

2. **Re-derive the bias correction divisor** from first principles
   for HFQ4×Q8_1 (NOT Q4_0×Q8_1). Verify by hand on a tiny example.

3. **Layer-bisect with `HIPFIRE_MMQ_LAYER_FILTER=N`**: enable MMQ for
   only layer N, FP16 for the rest. Test layer 0, 1, 2, ... and see if
   the model survives single-layer noise injection. If layer 0 breaks
   it, the issue is sensitivity to layer-0 perturbation. If only
   layer 5+ breaks it, suspect compounding.

4. **Compare RDNA3 MMQ noise**: extend `mmq_screen_weight` to use my
   gfx906 MMQ kernel for the "MMQ" side and FP16 wave64 for the
   reference. Compare per-weight max-abs-err distribution to RDNA3's
   typical error. If similar magnitude, gfx906 is fine and Strix Halo
   also has 0.6% NRMSE → this is structural to Q8_1.

5. **Try a higher-precision activation format**: skip Q8_1, use FP16
   activations instead. Just a sanity test to confirm whether removing
   activation quantization fixes coherence. If so, the bug is in
   either the Q8_1 quantization OR how the kernel uses it.

## Instrumentation already in place

- `HIPFIRE_MMQ_TRACE=1` — print residual MMQ calls with shapes
- `HIPFIRE_MMQ_DIAG_PASSTHROUGH=1` — skip the dp4a kernel entirely,
  forward to FP16 wave64
- `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1` — run Q8_1 quantize, forward
  to FP16 wave64 for the GEMM
- `HIPFIRE_MMQ_K_FILTER=N` — restrict MMQ to k==N calls
- `HIPFIRE_MMQ_CALL_FILTER=lo:hi` — restrict MMQ to call indices [lo, hi)
- `HIPFIRE_MMQ_DUMP=N` — dump inputs/outputs of the Nth residual GEMM
  to `/tmp/mmq_dump_N/{a_raw.bin, x.f32, y_in.f32, y_out.f32, y_mmq.f32, shape.txt}`
  for offline analysis with `test_gfx906_mmq_realdata`.
- Standalone test: `target/release/examples/test_gfx906_mmq_correctness M K N`
- Real-data test: `target/release/examples/test_gfx906_mmq_realdata <dump_dir>`

## What I tried that didn't help

- Changing OOB col handling in `load_q8_1_tile` from clamp-to-N-1 to
  zero-out: no change to row 3994 error or production gibberish.
- Multi-iteration test on same Y buffer: passes, no degradation.

## Open questions

1. Why does my kernel have **5×** more NRMSE on real data vs synthetic
   at the same shape (0.60% vs 0.12%)?
2. Why is **row 3994 specifically** showing 7× the max abs error of
   any other row, when synthetic data shows uniform error distribution?
3. Does the RDNA3 i8-WMMA MMQ path actually work cleanly in production,
   or does it have similar per-layer NRMSE that the model tolerates
   for some reason we don't understand?
