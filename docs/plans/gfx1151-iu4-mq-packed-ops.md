# gfx1151 IU4/U4 Packed-Ops Investigation for MQ2/MQ3/MQ4

**Date:** 2026-06-01
**Scope:** gfx1151 first. Other RDNA/CDNA archs are intentionally out of scope.
**Formats:** MQ2, MQ3, MQ4, and their Lloyd variants.

## Short Answer

gfx1151 can execute packed 4-bit matrix ops. A local ROCm/LLVM probe
compiled `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` for `gfx1151`
and disassembled to:

```text
v_wmma_i32_16x16x16_iu4 v[8:15], v[1:2], v[3:4], v[8:15] neg_lo:[1,1,0]
```

AMD's `amd_matrix_instruction_calculator` agrees that `gfx1151` maps to
the RDNA3 WMMA table. For RDNA3 it reports:

| Instruction | M | N | K | Ops | Cycles | Ops/WGP/cycle |
|---|---:|---:|---:|---:|---:|---:|
| `v_wmma_i32_16x16x16_iu8` | 16 | 16 | 16 | 8192 | 32 | 1024 |
| `v_wmma_i32_16x16x16_iu4` | 16 | 16 | 16 | 8192 | 16 | 2048 |

So, for gfx1151's native RDNA3 WMMA instructions, IU4 has **2x the
modeled raw matrix-op throughput** of IU8. The wider
`v_wmma_i32_16x16x32_iu4` shape is RDNA4-only in this calculator and is
not available for the gfx1151 scope.

The catch is that this is an **int4-by-int4 -> int32 matrix op**. It is
not a packed 4-bit weight by fp16/fp32 activation op. Using it for MQ
requires an activation-quantized Q4/U4 scratch path and a numerical
contract distinct from the current scalar GEMV and Q8_1 MMQ paths.

The first viable target is therefore not scalar decode GEMV. It is
batched prefill, batched verify, or per-expert microbatching where a
16x16 WMMA tile has enough useful work to amortize activation
quantization and sub-byte unpack.

## Current hipfire Paths

Current gfx1151 MQ decode and prefill paths are not using IU4:

- scalar MQ4/HFQ4 decode unpacks nibbles and does scalar fp32 FMA.
- scalar MQ3/MQ2 unpack their dense 3-bit/2-bit streams and do scalar
  fp32 FMA.
- MQ4/HFQ4 MMQ prefill quantizes activations to Q8_1 and uses
  `v_wmma_i32_16x16x16_iu8`.
- Lloyd WMMA prefill expands codebook entries to fp16 and uses fp16
  WMMA, because the reconstruction value is not affine in the index.

The existing `block_q8_1_mmq` path in
`kernels/src/gemm_hfq4g256_residual_mmq.hip` is the right structural
template: prequantize activations once, then consume a compact activation
scratch from matrix kernels. IU4 needs the same architecture with a Q4
activation block.

## Format Fit

| Format | IU4 fit | Why |
|---|---|---|
| MQ4/HFQ4 uniform | Best first candidate | Reconstruction is affine: `w = scale * q + zero`. A Q4 activation scratch can turn the main term into an IU4 WMMA dot, with scale and zero corrections around it. |
| MQ3 uniform | Possible | Values `0..7` fit in U4 lanes. Packed 3-bit storage must be widened into U4 WMMA fragments or stored in an optional secondary prepacked layout. |
| MQ2 uniform | Possible but overhead-sensitive | Values `0..3` fit in U4 lanes, but the unpack/widen cost is a larger fraction of the work. It likely needs batching or per-expert microbatching before it wins clearly. |
| MQ4-Lloyd | Poor exact fit | Each 4-bit index maps through an arbitrary 16-entry codebook. `dot(codebook[q], x)` cannot be represented by one affine IU4 dot without changing the math. |
| MQ3-Lloyd | Poor exact fit | Same issue with an 8-entry codebook. Current LUT-to-fp16 then fp16 WMMA is the exact batched shape. |
| MQ2-Lloyd | Poor exact fit | Same issue with a 4-entry codebook. The small codebook makes bitplane experiments possible, but multiple IU4 dots plus masks are unlikely to beat the direct LUT path unless microbatching changes the economics. |

## Numerical Contract

The uniform path can be made approximate but structured:

```text
w = sw * qw + zw
x ~= sx * qx + zx

dot(w, x) ~= sw * sx * dot(qw, qx)
          + sw * zx * sum(qw)
          + zw * sx * sum(qx)
          + K * zw * zx
```

A symmetric activation quantizer can remove the `zx` terms and simplify
the correction path. That is the lowest-risk first prototype because MQ
weights already carry per-group affine metadata and the current MMQ path
already computes activation scales.

This is still a quality/perf tradeoff: Q4 activations reduce bandwidth
inside the MMQ path and unlock IU4 WMMA, but they introduce activation
quantization error that the current Q8_1 MMQ path does not have.

## Why Lloyd Should Not Be First

Lloyd codebooks are deliberately non-uniform. Treating the Lloyd index as
a numeric U4 value would approximate the codebook with an affine ramp and
throw away the reason the Lloyd quantizer exists.

Exact Lloyd acceleration options are:

- keep the existing table lookup into fp16/fp32 values, then use fp16
  WMMA for batched paths;
- use a codebook-specific decomposition with multiple bitplane or
  one-hot integer dots, which costs several matrix ops per logical dot;
- convert Lloyd-coded weights to an affine U4-compatible stream before
  they reach the GPU, accepting a new approximation boundary.

The first option is what hipfire already does for Lloyd prefill. The
second is probably too expensive. The third may become interesting if the
NPU is already converting and streaming weights, but it should be treated
as a mixed-format transform, not as a drop-in Lloyd kernel replacement.

## Proposed gfx1151 Implementation Order

1. **Add a tiny IU4 probe/microbench.** Keep it synthetic and isolated:
   prove packed operands, signedness flags, accumulator layout, and
   disassembly metadata on gfx1151.
2. **Implement Q4 activation scratch.** Add a `block_q4_*_mmq` sibling
   to the existing Q8_1 scratch. Start with symmetric Q4 unless a zero
   point is needed for quality.
3. **Prototype MQ4/HFQ4 uniform residual MMQ with IU4.** This is the
   cleanest affine case and the best comparison against the existing
   Q8_1 + IU8 MMQ kernel.
4. **Run gold comparisons before routing.** Compare against the current
   MQ4 control path and explicitly measure activation-Q4 drift. Do not
   make this the default until quality gates accept the approximation.
5. **Extend to uniform MQ3 and MQ2.** Widen dense 3-bit/2-bit packed
   weights into U4 fragments at tile load time first. Only add secondary
   prepacked artifacts if the unpack cost dominates.
6. **Revisit Lloyd only after microbatching or NPU streaming exists.**
   Without more reuse per loaded codebook, exact Lloyd IU4 is not the
   promising path.

## Decision

For gfx1151, IU4 is real and worth prototyping for **uniform**
MQ4/MQ3/MQ2 batched paths. It is not an immediate fix for current
single-token decode speed, and it is not a direct exact acceleration for
MQ2/MQ3/MQ4-Lloyd.

The next engineering step should be a Q4-activation MQ4/HFQ4 MMQ
prototype plus a resident synthetic benchmark. MQ2 should inherit that
path after the packed/unpack overhead is measured under per-expert
microbatching.
