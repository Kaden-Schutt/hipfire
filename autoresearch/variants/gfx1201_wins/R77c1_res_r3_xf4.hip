// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
// Default (arch-agnostic) kernel. See gemv_hfq4g256.hip for the full
// writeup on why this had to be re-ported from the gfx1100 variant to
// produce byte-exact output against the MQ4 quality-gate baselines.
// TL;DR: single-accumulator sequential summation drifts from the
// 4-accumulator pairwise-combine ordering used to generate the
// baselines on gfx1100, breaking cross-arch exactness.
//
// Identical math to gemv_hfq4g256 except the final write accumulates
// into y instead of overwriting. Used for wo and w_down projections
// in the Qwen3.5 forward path to eliminate one add_inplace_f32 launch
// per residual (~10 µs per call).

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    // R77c1_res_r3_xf4: keep the confirmed three-row residual traffic cut
    // from R17, then add R18's two-float4 x-slice loads. This tests whether
    // the x-load access improvement that helped r2 still pays when the
    // block already carries a third row's weight stream.
    const int row0 = blockIdx.x * 3;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    float ccc0 = 0.0f, ccc1 = 0.0f, ccc2 = 0.0f, ccc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8(b) \
        const float4 xv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 xv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float x0 = xv0.x; \
        const float x1 = xv0.y; \
        const float x2 = xv0.z; \
        const float x3 = xv0.w; \
        const float x4 = xv1.x; \
        const float x5 = xv1.y; \
        const float x6 = xv1.z; \
        const float x7 = xv1.w

    #define DOG_X8(gp, a) do { \
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        {
            LOAD_X8(base0);
            DOG_X8(row_ptr0 + g * 136, acc0);
            if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
            if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc0);
        }
        {
            LOAD_X8(base0 + 256);
            DOG_X8(row_ptr0 + (g + 1) * 136, acc1);
            if (have_row1) DOG_X8(row_ptr1 + (g + 1) * 136, bcc1);
            if (have_row2) DOG_X8(row_ptr2 + (g + 1) * 136, ccc1);
        }
        {
            LOAD_X8(base0 + 512);
            DOG_X8(row_ptr0 + (g + 2) * 136, acc2);
            if (have_row1) DOG_X8(row_ptr1 + (g + 2) * 136, bcc2);
            if (have_row2) DOG_X8(row_ptr2 + (g + 2) * 136, ccc2);
        }
        {
            LOAD_X8(base0 + 768);
            DOG_X8(row_ptr0 + (g + 3) * 136, acc3);
            if (have_row1) DOG_X8(row_ptr1 + (g + 3) * 136, bcc3);
            if (have_row2) DOG_X8(row_ptr2 + (g + 3) * 136, ccc3);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc0);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc1);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc1);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc2);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc2);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc2);
    }
    #undef DOG_X8
    #undef LOAD_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    float ccc = (ccc0 + ccc1) + (ccc2 + ccc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        ccc += __shfl_down(ccc, offset);

    if (tid == 0) {
        y[row0] += acc;
        if (have_row1) y[row1] += bcc;
        if (have_row2) y[row2] += ccc;
    }
}
