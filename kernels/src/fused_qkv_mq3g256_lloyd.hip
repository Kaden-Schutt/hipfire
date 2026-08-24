// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Fused QKV MQ3-Lloyd: 3 GEMVs in one launch.
// Per-row math identical to gemv_mq3g256_lloyd.hip (LDS codebook path).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_QKV_KERNEL_NAME
#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq3g256_lloyd
#endif
#if defined(HIPFIRE_QKV_WITH_BIAS)
#define HIPFIRE_QKV_BIAS_ARGS , const float* __restrict__ bias_q, const float* __restrict__ bias_k, const float* __restrict__ bias_v
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

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_QKV_KERNEL_NAME(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const float* __restrict__ x,
    float*       __restrict__ y_q,
    float*       __restrict__ y_k,
    float*       __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K HIPFIRE_QKV_BIAS_ARGS
) {
    const int gid = blockIdx.x;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int row;
    HIPFIRE_QKV_BIAS_DECL
    if (gid < q_m) {
        A = A_q;  y = y_q;  row = gid; HIPFIRE_QKV_BIAS_Q
    } else if (gid < q_m + k_m) {
        A = A_k;  y = y_k;  row = gid - q_m; HIPFIRE_QKV_BIAS_K
    } else {
        A = A_v;  y = y_v;  row = gid - (q_m + k_m); HIPFIRE_QKV_BIAS_V
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 112;
    const int boff = tid * 3;

    __shared__ float cbt[8];

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 112;

        if (tid < 8) {
            cbt[tid] = __half2float(((const __half*)gptr)[tid]);
        }
        __syncthreads();

        const unsigned char* data = (const unsigned char*)(gptr + 16);

        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        unsigned int q0 = pk & 7u;
        unsigned int q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u;
        unsigned int q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u;
        unsigned int q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u;
        unsigned int q7 = (pk >> 21) & 7u;

        int base_idx = g * 256 + tid * 8;

        acc += cbt[q0] * x[base_idx]
             + cbt[q1] * x[base_idx + 1]
             + cbt[q2] * x[base_idx + 2]
             + cbt[q3] * x[base_idx + 3]
             + cbt[q4] * x[base_idx + 4]
             + cbt[q5] * x[base_idx + 5]
             + cbt[q6] * x[base_idx + 6]
             + cbt[q7] * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

#if defined(HIPFIRE_QKV_WITH_BIAS)
    if (tid == 0) y[row] = acc + (bias ? bias[row] : 0.0f);
#else
    if (tid == 0) y[row] = acc;
#endif
}
