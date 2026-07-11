// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>

// Fused MoE down GEMV + K_TOP weighted-accumulate + residual add, with NUM_ROWS
// register row-tiling.
//
// Replaces THREE kernels of the atomic-free down path:
//   gemv_hfq4g256_moe_down_k8_indexed_batched_expanded  (down -> [N×K_TOP×M])
//   moe_down_combine_k8_batched                          (weighted sum over K_TOP)
//   the residual add                                     (x_residual += combined)
//
// Each block owns NUM_ROWS output rows of ONE token. For each of the K_TOP=8
// experts it reads that expert's activation X ONCE and computes NUM_ROWS down
// dots against it (the NUM_ROWS weight rows) — so the per-expert X slice is
// reused across NUM_ROWS output rows (stays L1-resident) and the workgroup count
// is cut by NUM_ROWS. Weights are weight-scaled + register-accumulated across
// experts; a single race-free  x_residual[token][row] += sum_k w[k]*down_k(row)
// closes each row (each cell owned by exactly one block -> no atomic, no
// N×K_TOP×M expanded intermediate).
//
// Algorithm (WG-owns-row-tile, per-WG X-reuse, register-accumulate over experts)
// learned from ZINC's dmmv_q4k_moe_fused_down_acc.comp (MIT,
// github.com/zolotukhin/zinc, NUM_ROWS=4); reimplemented in HIP for HFQ4-G256
// (MQ4). See NOTICE.
//
// NUMERICAL ORDER is identical per output row to the (expanded + combine) path:
//   per expert : 4-accumulator pairwise (acc0..3) -> __shfl_down warp reduce
//   over experts: sequential  weighted_acc[r] += w[k] * per_expert_result[r]
//   -> x_residual[row] += weighted_acc[r]
// The row-tile changes only WHICH rows a block owns + X-reuse; each row's math is
// bit-identical to NUM_ROWS=1, so the kernel stays token-id exact.
//
// Shapes:
//   expert_ptrs  : [n_exp]        device pointer table (per-expert down weight)
//   topk_indices : [N × K_TOP]
//   topk_weights : [N × K_TOP]
//   rot_batch    : [N × K_TOP × K] (silu*mul output, per (token,krank))
//   x_residual   : [N × M]        (in-place +=)
// MQ4 group = 136 bytes (4 scale + 4 zp + 128 packed 4-bit), 256 elems/group.
//
// Grid: (ceil(M/NUM_ROWS), N). Block = 32 threads (one warp).

// NUM_ROWS row-tile width. ZINC uses 4 (wave64); hipfire is wave32, so register
// pressure per row is effectively 2x and the optimum shifts down. Measured sweep
// on R9700/gfx1201 a3b mq4r decode (interleaved, ±0.1%): N=1 -0.2%, N=2 +3.5%
// (VGPR 64), N=4 +1.4% (VGPR ~80), N=8 -9.5% (VGPR 96, occupancy crash). N=2 is
// the reuse-vs-occupancy optimum. Grid divisor in gemv.rs must match.
#ifndef MOE_DOWN_FUSED_NUM_ROWS
#define MOE_DOWN_FUSED_NUM_ROWS 2
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_down_k8_indexed_fused_acc(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ topk_weights,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ x_residual,                // [N × M] += result
    int M, int K, int K_TOP
) {
    constexpr int NR = MOE_DOWN_FUSED_NUM_ROWS;
    const int row_base = blockIdx.x * NR;
    if (row_base >= M) return;
    const int bid = blockIdx.y;   // token id
    const int tid = threadIdx.x;

    const int routing_base   = bid * K_TOP;
    const int groups_per_row = K / 256;
    const int boff           = tid * 4;
    const int quads          = groups_per_row >> 2;
    const int tail           = groups_per_row & 3;
    const int nrow           = min(NR, M - row_base);   // rows this block actually commits

    float weighted_acc[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) weighted_acc[r] = 0.0f;

    for (int krank = 0; krank < K_TOP; krank++) {
        const int expert_id = topk_indices[routing_base + krank];
        const char* __restrict__ A =
            reinterpret_cast<const char*>(expert_ptrs[expert_id]);
        const float* __restrict__ x =
            rot_batch + ((long long)bid * K_TOP + krank) * K;

        float acc0[NR], acc1[NR], acc2[NR], acc3[NR];
        #pragma unroll
        for (int r = 0; r < NR; r++) { acc0[r]=0.0f; acc1[r]=0.0f; acc2[r]=0.0f; acc3[r]=0.0f; }

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
            const int base = g * 256 + tid * 8;
            // NR weight rows share these X reads (L1-resident across the tile).
            #pragma unroll
            for (int r = 0; r < NR; r++) {
                const int rr = min(row_base + r, M - 1);   // OOB-safe (result discarded if r>=nrow)
                const char* row_ptr = A + (long long)rr * groups_per_row * 136;
                const char* gp0 = row_ptr + g * 136;
                const char* gp1 = gp0 + 136;
                const char* gp2 = gp1 + 136;
                const char* gp3 = gp2 + 136;
                float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
                float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
                float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
                float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
                float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
                float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
                float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
                float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));
                unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
                unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
                unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
                unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);
                DOG(pk0, sc0, zp0, base,       acc0[r]);
                DOG(pk1, sc1, zp1, base + 256, acc1[r]);
                DOG(pk2, sc2, zp2, base + 512, acc2[r]);
                DOG(pk3, sc3, zp3, base + 768, acc3[r]);
            }
        }

        // tail groups (<4): fold into acc0/acc1/acc2 exactly like the NUM_ROWS=1 path.
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const int rr = min(row_base + r, M - 1);
            const char* row_ptr = A + (long long)rr * groups_per_row * 136;
            if (tail >= 1) {
                const int g = quads << 2; const char* gp = row_ptr + g * 136;
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                int base = g * 256 + tid * 8; DOG(pk, sc, zp, base, acc0[r]);
            }
            if (tail >= 2) {
                const int g = (quads << 2) + 1; const char* gp = row_ptr + g * 136;
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                int base = g * 256 + tid * 8; DOG(pk, sc, zp, base, acc1[r]);
            }
            if (tail >= 3) {
                const int g = (quads << 2) + 2; const char* gp = row_ptr + g * 136;
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                int base = g * 256 + tid * 8; DOG(pk, sc, zp, base, acc2[r]);
            }
        }
        #undef DOG

        const float w = topk_weights[routing_base + krank];
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
            for (int offset = 16; offset > 0; offset >>= 1)
                acc += __shfl_down(acc, offset);
            if (tid == 0) weighted_acc[r] += w * acc;
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r < nrow)
                x_residual[(long long)bid * M + row_base + r] += weighted_acc[r];
        }
    }
}
