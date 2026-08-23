// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_RESIDUAL_LUT_KERNEL
#define HIPFIRE_RESIDUAL_LUT_KERNEL gemv_hfq4g256_residual_lut
#endif

// gfx1100 (RDNA3) HFQ4-G256 residual GEMV with a per-group 16-entry LDS
// dequant LUT. Within one 256-value group every lane's nibbles map to the
// same 16 possible values v[i] = sc*i + zp; lanes 0-15 publish the table to
// LDS once per group and all lanes then read their values back, replacing
// per-element extract+cvt+MAD chains with LDS lookups. The published value
// is computed by the SAME expression as the shipping kernel (sc*(float)nib +
// zp, f32) so per-element results and summation order are BIT-IDENTICAL.
//
// LUT rows: lut[j] holds group (g+j)'s table while the quad loop walks four
// groups per iteration; a trailing __syncthreads guards reuse.

extern "C" __launch_bounds__(32, 16)
__global__ void HIPFIRE_RESIDUAL_LUT_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    __shared__ float lut[4][16];

    const int groups_per_row = K >> 8;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    const long long stride = (long long)groups_per_row * 136;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int boff = tid * 4;
    const char* row_ptr = A + (long long)row * stride;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    #define DOG_LUT(lrow, gp, a, xa, xb) do { \
        const unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (lrow)[((pk)      ) & 0xFu] * (xa).x \
             + (lrow)[(((pk) >>  4) & 0xFu)] * (xa).y \
             + (lrow)[(((pk) >>  8) & 0xFu)] * (xa).z \
             + (lrow)[(((pk) >> 12) & 0xFu)] * (xa).w \
             + (lrow)[(((pk) >> 16) & 0xFu)] * (xb).x \
             + (lrow)[(((pk) >> 20) & 0xFu)] * (xb).y \
             + (lrow)[(((pk) >> 24) & 0xFu)] * (xb).z \
             + (lrow)[(((pk) >> 28) & 0xFu)] * (xb).w; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const char* gp = row_ptr + (g + j) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            if (tid < 16) {
                lut[j][tid]      = sc * (float)tid + zp;
                lut[j][tid + 16] = sc * (float)(tid + 16) + zp;
            }
        }
        __syncthreads();
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
            if (j == 0) DOG_LUT(lut[j], gp, acc0, x0a, x0b);
            if (j == 1) DOG_LUT(lut[j], gp, acc1, x1a, x1b);
            if (j == 2) DOG_LUT(lut[j], gp, acc2, x2a, x2b);
            if (j == 3) DOG_LUT(lut[j], gp, acc3, x3a, x3b);
        }
        __syncthreads();
    }
    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        const char* gp = row_ptr + g * 136;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        if (tid < 8) {
            lut[0][tid]       = sc * (float)tid + zp;
            lut[0][tid + 8]   = sc * (float)(tid + 8) + zp;
        }
        __syncthreads();
        if (t == 0) DOG_LUT(lut[0], gp, acc0, xa, xb);
        if (t == 1) DOG_LUT(lut[0], gp, acc1, xa, xb);
        if (t == 2) DOG_LUT(lut[0], gp, acc2, xa, xb);
        if (t == 3) DOG_LUT(lut[0], gp, acc3, xa, xb);
        __syncthreads();
    }
    #undef DOG_LUT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] += acc;
}
