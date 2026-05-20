// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// dp4a-port of fused_gate_up_hfq6g256_wave64. Activations are
// pre-quantized to Q8_1 (block_q8_1_mmq layout, see
// kernels/src/gemm_hfq4g256_residual_mmq.hip:31). Weights stay
// HFQ6-G256 (6-bit unsigned + per-256-group fp32 scale + zp).
//
// Plan §2.2 Option A: HFQ6 weights are unsigned q ∈ [0, 63]. No
// (n - 8) shift — the values fit in signed int8 cleanly because
// q < 128. So the dp4a path uses sdot4 with unsigned-positive int8
// lanes on the weight side and signed int8 on the activation side,
// and the math identity is:
//
//   sum_k (sc * q_k + zp) * x_k
// = sc * sum_k(q_k * x_k) + zp * sum_k(x_k)
// =      ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
//        sdot4-able        zp * sum_x (no shift correction)
//
// And with x quantized as x_k ≈ d_x * x_int8_k per Q8_1 sub-block:
//   = sc * d_x * sum_k(q_k_packed_int8 · x_int8_packed) + zp * sum_x
//
// Per-sub-block contribution:
//   acc += sc * d_x * sumi_dp4a + zp * sum_x_subblock * 0.25
//
// (The 0.25 weight on sum_x is because each Q8_1 sub-block's sum_x
// covers 32 K-elements while each lane only covers 8 — the 32-lane
// reduction recovers the per-sub-block term exactly.)
//
// Lane mapping inside one HFQ6-G256 group (256 K-elements):
//   lane l owns K-elements [8l, 8l+7]
//   sub-block s(l) = l/4   (8 sub-blocks total: 4 in block A, 4 in B)
//   block      b(l) = l/16   (0 = block A, 1 = block B)
//   in-block sub  = (l/4) % 4
//
// HFQ6 weight: 8 weights = 6 bytes per thread per group. Bytes are
// loaded individually (not as a single u32) because byte_off = lane*6
// is not dword-aligned for half the lanes. Compiler coalesces the
// 6 byte-loads into the natural alignment.

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];           // (d, sum_x) per 32-K sub-block, 4 sub-blocks/block
    int8_t qs[4 * QK8_1];   // 128 int8 K-elements per block
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void fused_gate_up_hfq6g256_wave64_dp4a(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,   // K/128 blocks
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)local_row * groups_per_row * 200;
    const int byte_off = lane * 6;

    const int sub_in_block = (lane / 4) % 4;     // ds4[i] index, 0..3
    const int block_within_group = lane / 16;    // 0 (A) or 1 (B)
    const int qs_byte_off = 8 * (lane % 16);     // 0,8,16,...,120

    float acc = 0.0f;

    // One HFQ6 group per outer iter.
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 200;

        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // Load this lane's 6 weight bytes.
        const unsigned char* dp = (const unsigned char*)(gp + 8);
        const unsigned char b0 = dp[byte_off];
        const unsigned char b1 = dp[byte_off + 1];
        const unsigned char b2 = dp[byte_off + 2];
        const unsigned char b3 = dp[byte_off + 3];
        const unsigned char b4 = dp[byte_off + 4];
        const unsigned char b5 = dp[byte_off + 5];

        // 6-bit unpack: 8 weights from 6 bytes (4 weights per 3 bytes).
        const unsigned int q0 = b0 & 63;
        const unsigned int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        const unsigned int q2 = (b1 >> 4) | ((b2 & 3)  << 4);
        const unsigned int q3 = b2 >> 2;
        const unsigned int q4 = b3 & 63;
        const unsigned int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        const unsigned int q6 = (b4 >> 4) | ((b5 & 3)  << 4);
        const unsigned int q7 = b5 >> 2;

        // Pack 4 weights per int (each weight ∈ [0, 63] fits in lower
        // 6 bits of an int8 lane; high bit zero → reads as positive
        // signed int8 for sdot4).
        const int int_a = (int)((q0 & 0xFF)
                              | ((q1 & 0xFF) << 8)
                              | ((q2 & 0xFF) << 16)
                              | ((q3 & 0xFF) << 24));
        const int int_b = (int)((q4 & 0xFF)
                              | ((q5 & 0xFF) << 8)
                              | ((q6 & 0xFF) << 16)
                              | ((q7 & 0xFF) << 24));

        // Pick the right Q8_1 block (A or B) for this lane.
        // HFQ6 group g spans Q8_1 blocks (2g, 2g+1) — same as HFQ4
        // since the activation Q8_1 layout is keyed off K-element
        // count, not weight quant format.
        const int kblock = 2 * g + block_within_group;
        const block_q8_1_mmq* xb = Xq + kblock;

        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        const int xq_a = xqs[0];
        const int xq_b = xqs[1];

        // sdot4: signed int8 lanes. Weights are in [0, 63] (positive
        // signed) so the multiplication is mathematically identical
        // to an unsigned 6-bit × signed int8 product.
        int sumi = 0;
        sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
        sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

        const half2 ds = xb->ds4[sub_in_block];
        const float2 dsf = __half22float2(ds);
        const float d_x = dsf.x;
        const float sum_x = dsf.y;

        // Per-sub-block contribution: HFQ6 has no zp shift correction
        // (vs HFQ4's zp + 8*sc) because weights are unsigned-positive.
        acc += sc * d_x * (float)sumi + zp * sum_x * 0.25f;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (lane == 0) y[local_row] = acc;
}
