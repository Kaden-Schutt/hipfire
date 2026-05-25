# Tier 1 BF16 MFMA — foundation POC (2026-05-19)

Foundation POC for the **hipfire-native BF16 calibration path** that will
eventually replace the Tier 2 subprocess wrapper (llama.cpp
`llama-imatrix` + PyTorch `collect_hessian.py`) with native Rust binaries
consuming BF16 weights directly through hipfire's MFMA-optimized kernels.

Target speedup: 20× (~8h → ~25min for a Hessian-collection pass on a
27B-class model).

## Why this exists

The existing imatrix path (`crates/hipfire-runtime/examples/imatrix_collect.rs`)
shells out to llama.cpp at a pinned commit (`9dcf83552887bb...`) for
activation-magnitude collection. The Hessian path (`scripts/collect_hessian.py`)
shells out to PyTorch with HF transformers. Both paths re-implement the
forward pass outside hipfire:

- `llama-imatrix` runs its own llama.cpp forward (no MFMA on gfx942; no
  rocBLAS Tensile auto-routing; tokenizer disagrees with hipfire's at
  ~46% of positions per `docs/plans/issue-113-quant-quality-eval.md:126`).
- `collect_hessian.py` runs HF transformers eager mode (no MFMA, Python
  interp overhead, host-side rayon-equivalent rebuilds per `nn.Linear`).

For a 27B model on MI300x both currently bottleneck at single-digit
tokens/sec wall time on the calibration corpus. hipfire's HFQ4 prefill
on the same hardware hits 1284 tok/s with the v3 MFMA kernel
(`feat/mtp-mi300x` commit `496edfbc`, 97.6% of rocBLAS). Routing
calibration through the same MFMA path closes the gap.

## What this POC validates

The Tier 1 forward pass will consume BF16 weights end-to-end (no
HFQ4 dequant, no FP16 shadow). This POC verifies:

1. **Intrinsic resolution.** The gfx942 BF16 MFMA intrinsic is
   `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` — note the `_1k` suffix.
   Naively trying `..._bf16` (without `_1k`) or `..._16x16x16_bf16`
   (extra underscore) both fail compile. The `_1k` is the gfx90a name
   that gfx942 inherited.

2. **Operand bit-cast.** LLVM models BF16 lane bits as `i16` for the
   MFMA intrinsic. The wrapper `mfma_bf16(bf16x4, bf16x4) → vfloat4`
   bit-casts `__bf16x4 → short x4` before the call. The compiler folds
   the cast away; no data movement.

3. **Lane layout.** Identical to the F16 MFMA F32_16x16x16_F16 layout
   verified at `docs/investigations/2026-05-19-mfma-hfq4/mfma_test.cpp`:

   ```
   A:  lane l holds a[r] = A[m = l%16,        k = (l/16)*4 + r]
   B:  lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]
   D:  lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]   ← strip-major
   ```

   The strip-major output (different from A's m-major-by-lane input)
   was the first bug when scaffolding the HFQ4 v1 kernel; same trap here,
   resolved the same way.

4. **LDS B-tile geometry.** Replicates the v3 sweet-spot from
   `gemm_hfq4g256_residual_mfma.gfx942.hip` — 256-thread WG (4 wave64),
   32×32 output tile per WG, cooperative B-tile load to LDS with
   K_CHUNK=128. The 4-wave arrangement is 2×2 over the output tile
   (wave_id bit 1 selects m-half, bit 0 selects n-half).

## Files

- `mfma_bf16_test.cpp` — Two-stage POC.
  - **Stage 1**: single 16×16×16 MFMA tile against an FP32 reference
    (mirror of the F16 POC). Tight tolerance `max_err < 1e-2` because
    K=16 → small accumulator-error budget.
  - **Stage 2**: full LDS-tiled GEMM at M=128 K=128 batch=64 (one
    outer-K step at K_CHUNK=128 — exercises the LDS cooperative load,
    inter-wave m/n tiling, output strip-major write). Looser tolerance
    `max_abs < 1e-1` and `max_rel < 5e-3` since BF16's 7 mantissa bits
    accumulate ~K × 2^-7 ≈ 1e-2 abs over 128-K dot products.
- `../../../kernels/src/gemm_bf16_mfma.gfx942.hip` — Production kernel
  (the file the dispatcher will eventually call).

## Build & run

```bash
hipcc --offload-arch=gfx942 -O3 mfma_bf16_test.cpp -o mfma_bf16_test
./mfma_bf16_test
```

## Status

**Compile-validated locally on ROCm 7.2 / gfx1100 dev host (2026-05-19):**

- `mfma_bf16_test.cpp` compiles to a binary with `__bf16` typing,
  intrinsic resolution, and lane math typechecking. The host has no
  CDNA3, so runtime PASS depends on the MI300x droplet rerun.
- `kernels/src/gemm_bf16_mfma.gfx942.hip` compiles with
  `hipcc --offload-arch=gfx942 -O3 -c`. Disassembly verifies the
  emitted instruction is `v_mfma_f32_16x16x16_bf16` (visible in
  `llvm-objdump -d` of the unbundled gfx942 device object at offset
  `~0x1DD0` of the kernel).

**Runtime validation pending MI300x droplet rerun** (next session). Expected:

- Stage 1: `max_err < 1e-2`, `fails = 0/256` — exact same shape as the
  F16 POC's PASS (`max_err = 8.9e-08` at K=16 is the F16 fingerprint;
  BF16 should be ~3 orders of magnitude wider but still well below the
  `1e-2` gate).
- Stage 2: `max_rel_err < 5e-3` on K=128, batch=64. If above, the LDS
  layout or the 4-wave output offset math has a bug worth bisecting —
  the v3 HFQ4 channel test got `max_rel_err = 2e-5` so BF16's mantissa
  loss shouldn't push past 5e-3.

## Theoretical performance ceiling

CDNA3 BF16 MFMA throughput at gfx942 base clock (2.1 GHz, 304 CUs):
- `v_mfma_f32_16x16x16_bf16` = 16×16×16 = 4096 FMAs per call
- 8 cycles per call on gfx942 → 512 FMAs/cycle/CU = 1024 FLOPs/cycle/CU
- 304 CUs × 1024 × 2.1 GHz = **653 TFLOPs/sec BF16 peak**

The 16×16×16 BF16 has the same FMAs/cycle ratio as F16, so the
prefill-equivalent throughput should be in the same band as the v3 HFQ4
result (1284 tok/s = 97.6% of rocBLAS BF16 Tensile). The Tier 1 forward
pass doesn't dequantize, so the per-call cost may even be lower — no
unpack overhead in the inner loop.

## Comparison to rocBLAS BF16

rocBLAS exposes BF16×BF16→F32 GEMM via `rocblas_gemm_ex` with
`a/b_type = rocblas_datatype_bf16_r`, `c/d_type = rocblas_datatype_f32_r`,
`compute_type = rocblas_datatype_f32_r`. The v3 HFQ4 measurement showed
the LDS B-tile MFMA hit 97.6% of rocBLAS Tensile for HFQ4 → F32. For
the actual BF16 path, rocBLAS itself is the comparison target — Tier 1
should land within 5-10% of `rocblas_gemm_ex` for the calibration
shapes (M ∈ {4096, 5120, 6144}, K ∈ {4096, 5120, 6144}, batch ≈ ctx_len
= 2048).

## Why a separate kernel (not just rocBLAS)?

Two reasons:
1. **Capture hooks.** Tier 1 needs to feed activations to the on-GPU
   Σx²-style and outer-product reductions (Hessian). rocBLAS's
   `gemm_ex` is opaque — we can't inject a capture hook into the
   middle. Our kernel can have the capture call site at known offset.
2. **Custom output handling.** The Hessian / Σx² kernels read each
   layer's input activations. Tying the GEMM directly to the capture
   point removes a roundtrip to HBM (the capture hook can be inserted
   right where the activations are live in registers).

For pure forward-throughput-without-capture, we'd still use rocBLAS;
the Tier 1 kernel is for the calibration path where the capture hook
matters.

## What's next (deliberately out of scope here)

- **Phase 2** (separate task): on-GPU Σx² reduction kernel (sums per
  input channel) — consumed by the AWQ activation-magnitude path.
- **Phase 2** (separate task): on-GPU outer-product Hessian kernel
  (K×K rank-1 update per token) — consumed by the GPTQ Cholesky
  pass. Storage: write back to HBM in K×K FP32 blocks.
- **Phase 2** (separate task): GGUF imatrix writer in Rust (mirrors
  `llama.cpp/src/imatrix.cpp::write_to_file` byte-for-byte to keep the
  produced file consumable by the same `gguf_input.rs` reader).
- **Phase 3** (separate task): BF16 weight loader from safetensors,
  attention/MLP forward path wiring through this GEMM, capture-hook
  threading at every linear-layer dispatch site.

This POC delivers just the GEMM building block. Everything above sits
on top.
