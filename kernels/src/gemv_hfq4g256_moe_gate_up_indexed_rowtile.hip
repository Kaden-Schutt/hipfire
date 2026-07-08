// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>

// NUM_ROWS register row-tile of gemv_hfq4g256_moe_gate_up_k8_indexed (single-token
// decode MoE gate_up GEMV). Each block owns NUM_ROWS output rows for ONE expert
// rank, reusing that expert's activation x across the NUM_ROWS rows (x stays
// L1-resident) and cutting the workgroup count by NUM_ROWS.
//
// gate_up is the LARGER MoE decode GEMV (2*MI output rows/expert, ~2x the down
// weight traffic on A3B) so it has more compute headroom than the down kernel.
// Unlike the fused down (which loops K_TOP experts + weight-accumulates + writes
// one residual/row), gate_up keeps krank in the grid (blockIdx.y), applies no
// routing weight, and writes each row to y_gate OR y_up by row<MI — the per-row
// split already handles a tile straddling the gate/up boundary (row_base=MI-1),
// and M = 2*MI is even so row_base+NR-1 < M for NR=2 (guarded anyway).
//
// Per-row math is IDENTICAL to the NUM_ROWS=1 kernel (same 4-acc pairwise DOG +
// __shfl_down reduce + same output slot), so the kernel is token-id exact. The
// tile changes only WHICH rows a block owns + the shared x reads.
//
// Mechanism (WG-owns-row-tile, per-WG x-reuse) mirrors the committed fused-down
// row-tile (project_fused_moe_down_rowtile_win, +3.5%); note gate_up's x is the
// shared post-rmsnorm activation (already cache-hot), so the x-locality half of
// down's win transfers weakly — the gain rests on ILP + halved workgroup count.
//
// Grid: (ceil(M/NUM_ROWS), K_TOP, 1). Block = 32 threads. Grid divisor in gemv.rs
// must match MOE_GATE_UP_NUM_ROWS.

#ifndef MOE_GATE_UP_NUM_ROWS
#define MOE_GATE_UP_NUM_ROWS 2
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed_rowtile(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    constexpr int NR = MOE_GATE_UP_NUM_ROWS;
    const int row_base = blockIdx.x * NR;
    if (row_base >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const int boff = tid * 4;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    const int mi = M >> 1;

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
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const int rr = min(row_base + r, M - 1);   // OOB-safe (result discarded if row>=M)
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

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) {
            const int row = row_base + r;
            if (row < M) {
                if (row < mi) y_gate[krank * mi + row] = acc;
                else          y_up[krank * mi + (row - mi)] = acc;
            }
        }
    }
}
