// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// dp4a-port of fused_qkv_hfq6g256_wave64. Same math + lane-mapping
// invariants as fused_gate_up_hfq6g256_wave64_dp4a.hip — see that file
// for the derivation. This kernel adds Q+K+V routing (3 output
// matrices instead of 2). Activations pre-quantized to Q8_1
// (block_q8_1_mmq layout, see kernels/src/gemm_hfq4g256_residual_mmq.hip:31).
//
// Plan §2.2 Option A: HFQ6 weights are unsigned q ∈ [0, 63], no
// shift. zp_eff = zp (no +8*sc correction).

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void fused_qkv_hfq6g256_wave64_dp4a(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const block_q8_1_mmq* __restrict__ Xq,
    float*       __restrict__ y_q,
    float*       __restrict__ y_k,
    float*       __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int row;
    if (gid < q_m) {
        A = A_q; y = y_q; row = gid;
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; row = gid - q_m;
    } else {
        A = A_v; y = y_v; row = gid - q_m - k_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int byte_off = lane * 6;

    const int sub_in_block       = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off        = 8 * (lane % 16);

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 200;

        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const unsigned char* dp = (const unsigned char*)(gp + 8);
        const unsigned char b0 = dp[byte_off];
        const unsigned char b1 = dp[byte_off + 1];
        const unsigned char b2 = dp[byte_off + 2];
        const unsigned char b3 = dp[byte_off + 3];
        const unsigned char b4 = dp[byte_off + 4];
        const unsigned char b5 = dp[byte_off + 5];

        const unsigned int q0 = b0 & 63;
        const unsigned int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        const unsigned int q2 = (b1 >> 4) | ((b2 & 3)  << 4);
        const unsigned int q3 = b2 >> 2;
        const unsigned int q4 = b3 & 63;
        const unsigned int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        const unsigned int q6 = (b4 >> 4) | ((b5 & 3)  << 4);
        const unsigned int q7 = b5 >> 2;

        const int int_a = (int)((q0 & 0xFF)
                              | ((q1 & 0xFF) << 8)
                              | ((q2 & 0xFF) << 16)
                              | ((q3 & 0xFF) << 24));
        const int int_b = (int)((q4 & 0xFF)
                              | ((q5 & 0xFF) << 8)
                              | ((q6 & 0xFF) << 16)
                              | ((q7 & 0xFF) << 24));

        const int kblock = 2 * g + block_within_group;
        const block_q8_1_mmq* xb = Xq + kblock;

        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        const int xq_a = xqs[0];
        const int xq_b = xqs[1];

        int sumi = 0;
        sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
        sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

        const half2 ds = xb->ds4[sub_in_block];
        const float2 dsf = __half22float2(ds);
        const float d_x = dsf.x;
        const float sum_x = dsf.y;

        acc += sc * d_x * (float)sumi + zp * sum_x * 0.25f;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] = acc;
}
