// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 3-way fused HFQ6-G256 GEMV for the qwen2-family FullAttention preamble.
// One launch performs all three projections from a single shared x:
//
//   y_q = W_q · x   (M=q_m)
//   y_k = W_k · x   (M=k_m)
//   y_v = W_v · x   (M=v_m)
//
// Per-row body is byte-identical to gemv_hfq6g256.hip (200 B / G256 group,
// 6-bit packed 4-weights-per-3-bytes). Grid = q_m + k_m + v_m. Covers both
// HFQ6G256 and MQ6G256 (same 6-bit on-disk layout).
//
// Bit-exact with three sequential gemv_hfq6g256 calls (+ the optional Q/K/V
// bias add). NULL bias = pure no-op, so it is bit-exact with the unfused
// 3×gemv_hfq6g256 path; non-null folds qwen2's QKV bias into the lane-0 store
// (HIPFIRE_FUSE_QKV_BIAS), eliminating 3 separate bias_add launches.

__launch_bounds__(32, 20)
extern "C" __global__ void fused_qkv_hfq6g256(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const float* __restrict__ x,
    float*       __restrict__ y_q,
    float*       __restrict__ y_k,
    float*       __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K,
    // Optional Q/K/V bias (HIPFIRE_FUSE_QKV_BIAS). NULL = pure no-op, so the
    // unfused callers that pass null stay byte-identical to 3×gemv_hfq6g256.
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v
) {
    const int gid = blockIdx.x;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    // Route to one of the three projections (and its matching bias).
    const char* A;
    float* y;
    int row;
    const float* bias;
    if (gid < q_m) {
        A = A_q; y = y_q; row = gid; bias = bias_q;
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; row = gid - q_m; bias = bias_k;
    } else {
        A = A_v; y = y_v; row = gid - q_m - k_m; bias = bias_v;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 4 weights per 3 bytes → 8 weights = 6 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 6;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];
        unsigned char b5 = data[byte_off + 5];

        // 4 weights from first 3 bytes
        int q0 = b0 & 63;
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        int q3 = b2 >> 2;
        // 4 weights from next 3 bytes
        int q4 = b3 & 63;
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        int q7 = b5 >> 2;

        acc += (scale * (float)q0 + zero) * x[base_idx]
             + (scale * (float)q1 + zero) * x[base_idx + 1]
             + (scale * (float)q2 + zero) * x[base_idx + 2]
             + (scale * (float)q3 + zero) * x[base_idx + 3]
             + (scale * (float)q4 + zero) * x[base_idx + 4]
             + (scale * (float)q5 + zero) * x[base_idx + 5]
             + (scale * (float)q6 + zero) * x[base_idx + 6]
             + (scale * (float)q7 + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    // NULL bias is a pure no-op: callers passing null stay byte-identical.
    if (tid == 0) y[row] = acc + (bias ? bias[row] : 0.0f);
}
