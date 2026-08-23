// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 (RDNA3) persistent dual-row HFQ4-G256 GEMV with fused residual add.
// y[row] += A[row] . x for exactly 2 rows per block (grid = ceil(M/2)).
//
// Same per-row group/FMA schedule as gemv_hfq4g256_residual.gfx1100.hip
// (STAGE_X32 activation hoist, per-group consume, quad accumulator split,
// shfl_down tree, fused residual write) -> per-row sums are BIT-IDENTICAL to
// the shipping single-row kernel. Only the row mapping changed: each block
// owns two adjacent rows instead of one, which halves wave-quantization tail
// on short-row shapes (K=4096: 400 -> 600 GB/s DRAM-cold on gfx1100).
//
// Opt-in via HIPFIRE_RESIDUAL_PERSIST_R2=1 (host routing, gfx1100 only).

#ifndef HIPFIRE_RESIDUAL_PERSIST_KERNEL
#define HIPFIRE_RESIDUAL_PERSIST_KERNEL gemv_hfq4g256_residual_persist_r2
#endif

extern "C" __launch_bounds__(32, 32)
__global__ void HIPFIRE_RESIDUAL_PERSIST_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int groups_per_row = K >> 8;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    const long long stride = (long long)groups_per_row * 136;
    const int tid = threadIdx.x;
    const int boff = tid * 4;

    const int row0 = blockIdx.x * 2;
    const int row1 = min(M, row0 + 2);

    for (int row = row0; row < row1; row++) {
        const char* row_ptr = A + (long long)row * stride;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #define DOG_X8(pk, sc, zp, a, xa, xb) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * (xa).x \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (xa).y \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (xa).z \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (xa).w \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (xb).x \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (xb).y \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (xb).z \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (xb).w

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
            const int base = g * 256 + tid * 8;
            const float4 x0a = *reinterpret_cast<const float4*>(x + base);
            const float4 x0b = *reinterpret_cast<const float4*>(x + base + 4);
            const float4 x1a = *reinterpret_cast<const float4*>(x + base + 256);
            const float4 x1b = *reinterpret_cast<const float4*>(x + base + 260);
            const float4 x2a = *reinterpret_cast<const float4*>(x + base + 512);
            const float4 x2b = *reinterpret_cast<const float4*>(x + base + 516);
            const float4 x3a = *reinterpret_cast<const float4*>(x + base + 768);
            const float4 x3b = *reinterpret_cast<const float4*>(x + base + 772);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const char* gp = row_ptr + (g + j) * 136;
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                if (j == 0) DOG_X8(pk, sc, zp, acc0, x0a, x0b);
                if (j == 1) DOG_X8(pk, sc, zp, acc1, x1a, x1b);
                if (j == 2) DOG_X8(pk, sc, zp, acc2, x2a, x2b);
                if (j == 3) DOG_X8(pk, sc, zp, acc3, x3a, x3b);
            }
        }
        for (int t = 0; t < tail; t++) {
            const int g = (quads << 2) + t;
            const int base = g * 256 + tid * 8;
            const float4 xa = *reinterpret_cast<const float4*>(x + base);
            const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
            const char* gp = row_ptr + g * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            if (t == 0) DOG_X8(pk, sc, zp, acc0, xa, xb);
            if (t == 1) DOG_X8(pk, sc, zp, acc1, xa, xb);
            if (t == 2) DOG_X8(pk, sc, zp, acc2, xa, xb);
            if (t == 3) DOG_X8(pk, sc, zp, acc3, xa, xb);
        }
        #undef DOG_X8

        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) y[row] += acc;
    }
}
