// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HFQ4-G256 GEMV, sigmoid-scaled, WRITE (non-residual) variant.
//
//   y[row] = sigmoid(c_buf[0]) * (A[row] · x)
//
// Bit-for-bit copy of the accumulator body of
// `gemv_hfq4g256_residual_sigmoid_scaled_gpu` (see
// kernels/src/gemv_hfq4g256_residual_scaled.hip) — same two-row x/gate reuse,
// same reduction, same `gate = 1/(1+exp(-c))` — but the final write ASSIGNS
// `gate*acc` into `y` instead of `+=`-ing it into a residual stream.
//
// This is the `HIPFIRE_FOLD_SHARED_EXPERT` decode micro-lever: the MoE
// shared-expert down projection writes its scaled result into a scratch
// buffer here, and the routed-expert combine
// (`moe_down_combine_k8_shared_batched`) folds that scratch into the residual
// together with the routed sum — one residual read-modify-write instead of the
// baseline's separate shared-expert residual GEMV followed by the routed
// combine. Because the emitted value `gate*acc` is bit-identical to the term
// the residual variant would have added, and the fold kernel adds it into the
// residual BEFORE the routed sum (matching the baseline launch order), the
// composed result is byte-identical to the baseline path.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_sigmoid_scaled_write_gpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ c_buf,
    int M, int K
) {
    const int row0 = blockIdx.x * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8(b) \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]

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
        }
        {
            LOAD_X8(base0 + 256);
            DOG_X8(row_ptr0 + (g + 1) * 136, acc1);
            if (have_row1) DOG_X8(row_ptr1 + (g + 1) * 136, bcc1);
        }
        {
            LOAD_X8(base0 + 512);
            DOG_X8(row_ptr0 + (g + 2) * 136, acc2);
            if (have_row1) DOG_X8(row_ptr1 + (g + 2) * 136, bcc2);
        }
        {
            LOAD_X8(base0 + 768);
            DOG_X8(row_ptr0 + (g + 3) * 136, acc3);
            if (have_row1) DOG_X8(row_ptr1 + (g + 3) * 136, bcc3);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc0);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc1);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc2);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc2);
    }
    #undef DOG_X8
    #undef LOAD_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_buf[0]));
        y[row0] = gate * acc;                 // WRITE (non-residual)
        if (have_row1) y[row1] = gate * bcc;  // WRITE (non-residual)
    }
}
