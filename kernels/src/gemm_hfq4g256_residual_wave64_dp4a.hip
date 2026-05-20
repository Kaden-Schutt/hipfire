// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave64+dp4a port of gemm_hfq4g256_residual at GEMM-batched shape.
// Activations pre-quantized to Q8_1 (block_q8_1_mmq layout, kblock-major:
// [K/128 * batch_size]). Weights stay HFQ4-G256 (signed 4-bit nibbles +
// per-256-group fp32 scale + zp).
//
// HFQ4 sibling of gemm_hfq6g256_residual_wave64_dp4a (HFQ6 Phase A.2,
// commit 1b9f3747 on feat/gfx906-hfq6-hfq8-analysis). Closes the
// dispatch gap where MQ4 at batch_size > 1 but below the MMQ cutover
// (B ∈ [2, 7] on gfx906) falls to gemm_hfq4g256_residual_fp16_wave64;
// the dp4a path wins on per-call ALU (sdot4 issues 4 int8 mul + 4
// acc-add per cycle vs hfma2's 2 mul + 2 add → ~2× FLOPs/cycle) and
// reuses the existing Q8_1 scratch.
//
// Differs from gemm_hfq4g256_wave64_dp4a only in writeback semantic:
//   gemm_hfq4g256_wave64_dp4a:          y[...]  = val  (set/lm-head)
//   gemm_hfq4g256_residual_wave64_dp4a: y[...] += val  (residual fuse)
//
// Math identity (signed 4-bit HFQ4, see gemm_hfq4g256_wave64_dp4a.hip
// header for the full derivation):
//   acc[b] += sc * d_x[b] * sumi_dp4a + zp_eff * sum_x[b] * 0.25f
// where zp_eff = zp + 8 * sc folds the unsigned-bias correction into a
// single per-block term. The 0.25 factor distributes Q8_1's per-sub-
// block sum_x across the 4 lanes that share the sub-block.
//
// Topology mirrors gemm_hfq4g256_wave64_dp4a:
//   block = [64, 1, 1]   = 2 warps, each handles 1 row (warp_id = row offset)
//   grid  = [(M+1)/2, ceil(N/BATCH_TILE), 1]
//   BATCH_TILE = 16 — per HFQ6 Phase B.1.1 (commit ff9e2105): doubling
//     BATCH_TILE 8→16 halved A-reload trips per row across grid_y
//     batch tiles, recovering 7-17% per-call vs BT=8 on the structurally
//     identical HFQ6 sibling. Shipping BT=16 from the start rather than
//     repeating HFQ6's bisect.

#define QK8_1 32
#define BATCH_TILE 16

struct block_q8_1_mmq {
    half2 ds4[4];           // (d, sum_x) per 32-K sub-block, 4 sub-blocks/block
    int8_t qs[4 * QK8_1];   // 128 int8 K-elements per block
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemm_hfq4g256_residual_wave64_dp4a(
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
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = lane * 4;

    // Per-lane mapping inside one HFQ4-G256 group (256 K-elements):
    //   K range: [8*lane, 8*lane+7]
    //   sub-block index s(l) = l/4 (0..7)
    //   Q8_1 block index within group b(l) = l/16 (0=A, 1=B)
    //   ds4 slot within block i(l) = (l/4) % 4
    //   qs byte offset within block: 8 * (lane % 16)
    const int sub_in_block       = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off        = 8 * (lane % 16);

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 136;

        // Wave-uniform-per-warp scale + zp.
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const float zp_eff = zp + 8.0f * sc;

        // Lane's 8 nibbles → 2 int8-packs. (n - 8) shifts unsigned [0,15]
        // to signed [-8,7] so sdot4 consumes the same range as Q8_1's
        // signed activations.
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

        // Pick the lane's Q8_1 block (A or B). HFQ4 group g spans
        // Q8_1 kblocks (2g, 2g+1).
        const int kblock = 2 * g + block_within_group;

        // Per-batch-token loop. Each iteration loads its own
        // (d_x, sum_x, xq_a, xq_b) for this lane's sub-block view.
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

    // Reduce + residual write-back. Same pyramid as the FP kernel.
    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (lane == 0) y[(long long)(batch_start + b) * M + row] += val;
    }
}
