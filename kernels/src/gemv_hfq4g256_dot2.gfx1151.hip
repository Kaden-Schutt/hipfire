// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 decode-path HFQ4-G256 GEMV using v_dot2_f32_f16.
//
// This is deliberately a new gfx1151-only code object and symbol.  It does
// not change the gfx1100 or gfx12 source, JIT cache key, or retained tape.
// The hot LM-head shape is fixed at K=2048 (eight HFQ groups), allowing the
// compiler to fully remove loop/tail machinery.
//
// The scalar kernel expands each nibble to F32, applies scale/zero in F32,
// and performs eight F32 products per group.  Here the eight dequantized
// weights and F32 activations are packed to FP16 inline, four independent
// dot2 instructions form the group partial, and accumulation/reduction stays
// in F32.  This follows the production HFP4 gfx11 dot2 strategy: inline
// conversion avoids a separate activation-conversion launch, while four
// independent dot products expose latency-hiding ILP.
//
// Numerical contract: replay-exact between HIP and PM4 for this code object,
// but not bit-identical to the all-F32 HFQ4 kernel because the dot operands
// are rounded to FP16.  It therefore remains opt-in until the multi-turn
// quality battery passes.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_lm_head_dot2_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    constexpr int GROUPS_PER_ROW = 8;
    constexpr int ROW_STRIDE = GROUPS_PER_ROW * 136;
    const char* row_ptr = A + (long long)row * ROW_STRIDE;
    const unsigned int row_weight_offset = (unsigned int)((long long)row * ROW_STRIDE);
    const unsigned int boff = (unsigned int)tid * 4u;
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Do not unroll the eight groups. A fully expanded source made LLVM keep
    // temporaries from multiple groups live, hit the 96-VGPR launch plateau,
    // and spill eleven VGPRs. The loop index is wave-uniform, so the four-way
    // accumulator routing below uses scalar control flow while each group's
    // packed state is recycled before the next iteration.
    #pragma unroll 1
    for (int g = 0; g < GROUPS_PER_ROW; ++g) {
        const char* gp = row_ptr + g * 136;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned int pk = HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, goff + 8u + boff);

        const __half2 sc2 = __half2half2(__float2half(sc));
        const __half2 zp2 = __half2half2(__float2half(zp));
        const __half2 q01 = __floats2half2_rn(
            (float)((pk      ) & 0xFu), (float)((pk >>  4) & 0xFu));
        const __half2 q23 = __floats2half2_rn(
            (float)((pk >>  8) & 0xFu), (float)((pk >> 12) & 0xFu));
        const __half2 q45 = __floats2half2_rn(
            (float)((pk >> 16) & 0xFu), (float)((pk >> 20) & 0xFu));
        const __half2 q67 = __floats2half2_rn(
            (float)((pk >> 24) & 0xFu), (float)((pk >> 28) & 0xFu));
        const __half2 w01 = __hfma2(q01, sc2, zp2);
        const __half2 w23 = __hfma2(q23, sc2, zp2);
        const __half2 w45 = __hfma2(q45, sc2, zp2);
        const __half2 w67 = __hfma2(q67, sc2, zp2);

        const int base = g * 256 + tid * 8;
        const float4 xv0 = *(const float4*)(x + base);
        const float4 xv1 = *(const float4*)(x + base + 4);
        const __half2 x01 = __floats2half2_rn(xv0.x, xv0.y);
        const __half2 x23 = __floats2half2_rn(xv0.z, xv0.w);
        const __half2 x45 = __floats2half2_rn(xv1.x, xv1.y);
        const __half2 x67 = __floats2half2_rn(xv1.z, xv1.w);

        const float p0 = amd_mixed_dot(w01, x01, 0.0f, false);
        const float p1 = amd_mixed_dot(w23, x23, 0.0f, false);
        const float p2 = amd_mixed_dot(w45, x45, 0.0f, false);
        const float p3 = amd_mixed_dot(w67, x67, 0.0f, false);
        const float partial = (p0 + p1) + (p2 + p3);
        switch (g & 3) {
            case 0: acc0 += partial; break;
            case 1: acc1 += partial; break;
            case 2: acc2 += partial; break;
            default: acc3 += partial; break;
        }
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) y[row] = acc;
}
