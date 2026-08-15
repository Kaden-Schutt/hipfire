// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// BF16-weight × F32-input GEMV: y[m] = W_bf16[m, k] @ x_f32[k].
//
// Native-bf16 reference path (the KLD/PPL oracle). The model ships in bf16;
// this keeps the EXACT downloaded weight values rather than re-quantizing to
// f16 (a different exponent/mantissa split). bf16 is the high 16 bits of an
// f32, so the widen is a lossless 16-bit left shift — no conversion error.
// F32 input precision is preserved (input is read as f32, accumulate in f32),
// matching gemv_f16_xf32 / gemv_q8_0 / gemv_f32.
//   grid = [M, 1, 1], block = [32, 1, 1]  (1 warp per output row)
//
// Batch-1 GEMV is pure weight streaming, so the figure of merit is achieved
// GB/s. At small M there are too few waves to hide memory latency (M=1024 is
// 4 waves/SIMD on gfx1201), so per-wave memory-level parallelism has to come
// from wider loads: each thread pulls 8 bf16 (one 16-byte dwordx4) per step
// instead of a single 2-byte scalar, and 4 accumulators keep the FMAs off one
// dependency chain. Mirrors the unroll + occupancy hint that makes
// `gemv_q8_0` measure at the roof.

// bf16 -> f32 is a lossless 16-bit shift. A dwordx4 holds 8 bf16: within each
// u32 the LOW half is the even element, the HIGH half the odd one (LE).
#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bf16_xf32(
    const unsigned short* __restrict__ W,  // [M, K] bf16 (raw u16)
    const float*          __restrict__ x,  // [K] F32
    float*                __restrict__ y,  // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const unsigned short* w_row = W + (size_t)row * K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    // Wide path: 8 bf16 per thread per step => 256 elements per wave step.
    // Row base is 16B-aligned whenever K % 8 == 0, which every hipfire decode
    // shape satisfies; the scalar tail below covers the rest.
    if ((K & 7) == 0) {
        for (int i = tid * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa = *(const float4*)(x + i);
            const float4 xb = *(const float4*)(x + i + 4);
            a0 += HF_BF16_LO(wv.x) * xa.x;
            a1 += HF_BF16_HI(wv.x) * xa.y;
            a2 += HF_BF16_LO(wv.y) * xa.z;
            a3 += HF_BF16_HI(wv.y) * xa.w;
            a0 += HF_BF16_LO(wv.z) * xb.x;
            a1 += HF_BF16_HI(wv.z) * xb.y;
            a2 += HF_BF16_LO(wv.w) * xb.z;
            a3 += HF_BF16_HI(wv.w) * xb.w;
        }
    }

    // Scalar tail: the final K % 8 elements (and all of K when K % 8 != 0).
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        a0 += __uint_as_float(((unsigned int)w_row[i]) << 16) * x[i];
    }

    float sum = (a0 + a1) + (a2 + a3);

    // Wave-shuffle reduction (wave32).
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
