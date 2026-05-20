// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave64+dp4a batched 2-way fused gate_up GEMM for HFQ4/MQ4, gfx906.
// FFN preamble (gate + up) at B>1.
//
// HFQ4 sibling of gemm_gate_up_hfq6g256_wave64_dp4a (HFQ6 Phase A.3,
// commit c070dbb3 → merged via #187). Issue #276 Gap 2.
//
// Closes the dispatch fallthrough where MQ4 at gfx906 batched prefill
// drops to gemm_gate_up_hfq4g256_fp16_wave64. Wins on per-call ALU
// (sdot4 4 mul + 4 acc-add per cycle vs hfma2 2 mul + 2 add ≈ 2×
// FLOPs/cycle) and reuses the existing Q8_1 scratch.
//
// Math identity (signed HFQ4) — see gemm_qkvza_hfq4g256_wave64_dp4a.hip
// for the full derivation. Same per-row inner loop; differs only in the
// row-routing prologue (2-way gate/up vs 4-way qkv/z/beta/alpha).
//
// BATCH_TILE = 16 from start per HFQ6 Phase B.1.1 measurement
// (commit 2bee6e6b: BT=8→16 measured -16.7% per-call on the HFQ6
// gate_up sibling — the largest of the 4 BT improvements).
//
// Output semantics: y[b][row] = acc (overwrite).

#define QK8_1 32
#define BATCH_TILE 16

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemm_gate_up_hfq4g256_wave64_dp4a(
    const char*  __restrict__ A_gate,
    const char*  __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,    // kblock-major: [K/128 * batch_size]
    float*       __restrict__ Y_gate,          // [batch_size, gate_m]
    float*       __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < gate_m) {
        A = A_gate; Y = Y_gate; row = gid;            out_stride = gate_m;
    } else {
        A = A_up;   Y = Y_up;   row = gid - gate_m;   out_stride = up_m;
    }

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = lane * 4;

    const int sub_in_block       = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off        = 8 * (lane % 16);

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 136;

        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const float zp_eff = zp + 8.0f * sc;

        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        const unsigned int n0 = (pk >>  0) & 0xFu;
        const unsigned int n1 = (pk >>  4) & 0xFu;
        const unsigned int n2 = (pk >>  8) & 0xFu;
        const unsigned int n3 = (pk >> 12) & 0xFu;
        const unsigned int n4 = (pk >> 16) & 0xFu;
        const unsigned int n5 = (pk >> 20) & 0xFu;
        const unsigned int n6 = (pk >> 24) & 0xFu;
        const unsigned int n7 = (pk >> 28) & 0xFu;
        const int int_a = (int)(((n0 - 8) & 0xFF)
                              | (((n1 - 8) & 0xFF) << 8)
                              | (((n2 - 8) & 0xFF) << 16)
                              | (((n3 - 8) & 0xFF) << 24));
        const int int_b = (int)(((n4 - 8) & 0xFF)
                              | (((n5 - 8) & 0xFF) << 8)
                              | (((n6 - 8) & 0xFF) << 16)
                              | (((n7 - 8) & 0xFF) << 24));

        const int kblock = 2 * g + block_within_group;

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

            acc[b] += sc * d_x * (float)sumi + zp_eff * sum_x * 0.25f;
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (lane == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
