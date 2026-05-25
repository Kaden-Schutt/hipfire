// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave64+dp4a port of gemm_hfq6g256_residual. Activations pre-quantized
// to Q8_1 (block_q8_1_mmq layout, kblock-major: [K/128 * batch_size]).
// Weights stay HFQ6-G256 (6-bit unsigned + per-256-group fp32 scale + zp).
//
// Combines three reference patterns:
//   - Wave64 topology + Q8_1 lookup pattern from gemm_hfq4g256_wave64_dp4a.hip
//   - 6-bit unpack from fused_gate_up_hfq6g256_wave64_dp4a.hip
//   - Residual `+=` write semantic from gemm_hfq6g256_residual.hip
//
// Plan §3.1 v3.2.3 §5.1 item 2 (Phase A.2). Targets the per-layer wo +
// w_down at B>1, the prefill batched-residual surface that doesn't
// route through the (not-yet-ported-for-HFQ6) MMQ family.
//
// Math identity is the same as the fused dp4a GEMVs (HFQ6 unsigned,
// no zp shift):
//   acc[b] += sc * d_x[b] * sumi_dp4a + zp * sum_x[b] * 0.25f
//
// Topology mirrors gemm_hfq4g256_wave64_dp4a:
//   block = [64, 1, 1]   = 2 warps, each handles 1 row (warp_id = row offset)
//   grid  = [(M+1)/2, ceil(N/BATCH_TILE), 1]
//   BATCH_TILE = 8 tokens per WG, accumulators acc[BATCH_TILE]
//
// Output write semantic: `Y[...] += val` (residual fuse), unlike the
// HFQ4 sibling's `Y[...] = val` (overwrite).

#define QK8_1 32
#define BATCH_TILE 8

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemm_hfq6g256_residual_wave64_dp4a(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,   // kblock-major: [K/128 * N]
    float* __restrict__ y,                   // [batch_size, M]
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int byte_off = lane * 6;

    const int sub_in_block       = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off        = 8 * (lane % 16);

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 200;

        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // Lane's 8 weights from 6 bytes (HFQ6 6-bit unpack).
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

        // Pack 4 weights per int (each ∈ [0, 63] fits signed int8).
        const int int_a = (int)((q0 & 0xFF)
                              | ((q1 & 0xFF) << 8)
                              | ((q2 & 0xFF) << 16)
                              | ((q3 & 0xFF) << 24));
        const int int_b = (int)((q4 & 0xFF)
                              | ((q5 & 0xFF) << 8)
                              | ((q6 & 0xFF) << 16)
                              | ((q7 & 0xFF) << 24));

        const int kblock = 2 * g + block_within_group;

        // Per-batch-token loop. Each iteration loads its own activation
        // slice from kblock-major Xq.
        for (int b = 0; b < local_bs; b++) {
            const int token = batch_start + b;
            const block_q8_1_mmq* xb = Xq + (long long)kblock * batch_size + token;

            const int* xqs = (const int*)(xb->qs + qs_byte_off);
            const int xq_a = xqs[0];
            const int xq_b = xqs[1];

            int sumi = 0;
            sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
            sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

            const half2 ds = xb->ds4[sub_in_block];
            const float2 dsf = __half22float2(ds);
            const float d_x   = dsf.x;
            const float sum_x = dsf.y;

            acc[b] += sc * d_x * (float)sumi + zp * sum_x * 0.25f;
        }
    }

    // Reduce + residual write-back.
    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (lane == 0) y[(long long)(batch_start + b) * M + row] += val;
    }
}
