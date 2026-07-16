// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_QKV_KERNEL_NAME
#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_q8_0
#endif

#if defined(HIPFIRE_QKV_WITH_BIAS)
#define HIPFIRE_QKV_BIAS_ARGS \
    , const float* __restrict__ bias_q \
    , const float* __restrict__ bias_k \
    , const float* __restrict__ bias_v
#define HIPFIRE_QKV_BIAS_DECL const float* bias;
#define HIPFIRE_QKV_BIAS_Q bias = bias_q;
#define HIPFIRE_QKV_BIAS_K bias = bias_k;
#define HIPFIRE_QKV_BIAS_V bias = bias_v;
#else
#define HIPFIRE_QKV_BIAS_ARGS
#define HIPFIRE_QKV_BIAS_DECL
#define HIPFIRE_QKV_BIAS_Q
#define HIPFIRE_QKV_BIAS_K
#define HIPFIRE_QKV_BIAS_V
#endif

// 3-way fused Q8_0 GEMV for full-attention layers (DECODE path, n=1).
// One kernel launch performs all three projections from a single shared
// x_normed input:
//
//   y_q = W_q · x   (M=q_m)
//   y_k = W_k · x   (M=k_m)
//   y_v = W_v · x   (M=v_m)
//
// Grid = q_m + k_m + v_m workgroups × 32 threads. Each workgroup routes
// to its projection via blockIdx.x range check, then runs the same Q8_0
// inner loop as fused_qkvza_q8_0.hip / gemv_q8_0.hip.
//
// Q8_0 block layout: 34 bytes per 32-element block
//   [fp16 scale (2 B) | int8[32] weights (32 B)]
//
// x is pre-normed F32. Bit-exact with three sequential gemv_q8_0 calls.
// Arch coverage: works on every RDNA generation. No dp4a dependency.

__launch_bounds__(32, 20)
extern "C" __global__ void HIPFIRE_QKV_KERNEL_NAME(
    const unsigned char* __restrict__ A_q,
    const unsigned char* __restrict__ A_k,
    const unsigned char* __restrict__ A_v,
    const float*         __restrict__ x,
    float*               __restrict__ y_q,
    float*               __restrict__ y_k,
    float*               __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K HIPFIRE_QKV_BIAS_ARGS
) {
    const int gid = blockIdx.x;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    // Route to one of the three projections.
    const unsigned char* A;
    float* y;
    int row;
    HIPFIRE_QKV_BIAS_DECL
    if (gid < q_m) {
        A = A_q; y = y_q; row = gid; HIPFIRE_QKV_BIAS_Q
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; row = gid - q_m; HIPFIRE_QKV_BIAS_K
    } else {
        A = A_v; y = y_v; row = gid - q_m - k_m; HIPFIRE_QKV_BIAS_V
    }

    const int tid = threadIdx.x;
    const int nblocks = K / 32;
    const unsigned char* row_data = A + (size_t)row * nblocks * 34;

    float sum = 0.0f;

    const int outer_iters = nblocks / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    for (int bi = outer_iters * 8; bi < nblocks; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
#if defined(HIPFIRE_QKV_WITH_BIAS)
    if (tid == 0) y[row] = sum + (bias ? bias[row] : 0.0f);
#else
    if (tid == 0) y[row] = sum;
#endif
}
