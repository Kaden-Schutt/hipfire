# gfx906 dp4a MMQ — bug analysis

Status: **FIXED** (2026-05-03).

## Resolution Summary

The bug was a combination of **systematic numerical bias** compounding
layer-over-layer and **precision loss** in weight metadata staging.

1.  **Systematic Bias (DC Offset)**: Unpacked 4-bit weights in the
    range `[0, 15]` had a large positive average (7.5). Small,
    consistent rounding biases in dynamic activation quantization
    were multiplied by these positive weights and summed over K=4096,
    creating a massive constant shift in every neuron. This compounded
    exponentially through 32 layers, turning logic into gibberish.
2.  **Precision Loss**: Weight scale/zero-point metadata was cast to
    `f16` when staged in LDS, which added a high noise floor to the
    bias term ($z_w \times \sum x$) that sensitive models like Qwen
    could not tolerate.

**Fix**:
-   **Symmetric Weight Trick**: Centered weights in `[-8, 7]` during
    unpacking to allow activation errors to cancel out. Mathematically
    restored the dot product in the final accumulation step.
-   **F32 LDS Staging**: Upgraded weight metadata in LDS to full `f32`
    precision.
-   **Screening**: Restored the `mmq_screen_weight` safety infrastructure
    for gfx906.

**Results**:
-   NRMSE on real Qwen weights reduced from **0.60% to 0.25%**.
-   Compounding bias eliminated; output residuals now zero-centered.
-   Model coherence restored in production.

## TL;DR (Old Analysis)

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

## Reproduction (Prior to Fix)

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

Possibilities (Pre-fix):
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

## Hypotheses and Tests (The Path to the Fix)

### 1. Sign-extension in nibble unpack (Investigated)
`__builtin_amdgcn_sdot4` interprets bytes as signed. Original code packed nibbles 0..15. ISA dump confirmed the compiler emitted zero-extending logic, but the positive average (7.5) of the weights turned out to be the "DC Offset" trigger.

### 2. Systematic Bias (Confirmed)
The positive average weight magnified rounding errors in dynamic activation quantization. Centering weights in `[-8, 7]` allowed these errors to cancel, reducing NRMSE by >2×.

### 3. Precision Bottleneck (Confirmed)
Casting weight scale/zp to `f16` in LDS was lossy. Upgrading to `f32` (8-byte `float2` instead of 4-byte `half2`) was required for model coherence.

## Suggested debug paths (Old List)

1. **Dump kernel ISA for `load_hfq4_tile_dp4a`** to verify nibble
   unpack is zero-extended, not sign-extended.
2. **Re-derive the bias correction divisor** from first principles
   for HFQ4×Q8_1 (NOT Q4_0×Q8_1). Verify by hand on a tiny example.
3. **Layer-bisect with `HIPFIRE_MMQ_LAYER_FILTER=N`**: enable MMQ for
   only layer N, FP16 for the rest. Test layer 0, 1, 2, ... and see if
   the model survives single-layer noise injection.
4. **Compare RDNA3 MMQ noise**: extend `mmq_screen_weight` to use my
   gfx906 MMQ kernel for the "MMQ" side and FP16 wave64 for the
   reference.
5. **Try a higher-precision activation format**: skip Q8_1, use FP16
   activations instead.

## Instrumentation already in place

- `HIPFIRE_MMQ_TRACE=1` — print residual MMQ calls with shapes
- `HIPFIRE_MMQ_DIAG_PASSTHROUGH=1` — skip the dp4a kernel entirely,
  forward to FP16 wave64
- `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1` — run Q8_1 quantize, forward
  to FP16 wave64 for the GEMM
- `HIPFIRE_MMQ_K_FILTER=N` — restrict MMQ to k==N calls
- `HIPFIRE_MMQ_CALL_FILTER=lo:hi` — restrict MMQ to call indices [lo, hi)
- `HIPFIRE_MMQ_DUMP=N` — dump inputs/outputs of the Nth residual GEMM
  for offline analysis.
- Standalone test: `target/release/examples/test_gfx906_mmq_correctness M K N`
- Real-data test: `target/release/examples/test_gfx906_mmq_realdata <dump_dir>`
