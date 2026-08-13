// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for MQ2-GL — "global Lloyd": 2-bit codes
// against ONE codebook shared by the whole tensor, plus a per-block fp16 scale.
// Drop-in sibling of gemv_mq2g256_lloyd_moe_gate_up_k8_indexed (72 B/group,
// per-block 4-entry codebook).
//
// Why: post-FWHT blocks are Gaussian by CLT, so the optimal LEVEL SHAPE is
// identical in every block — a per-block fit re-derives the same shape ~4000×
// per tensor and differs only by scale. Measured on 28.3M real a3b expert
// weights: dropping to a global codebook costs +2.35% NRMSE and saves
// 0.1875 bpw. The fitted global codebook lands on textbook Lloyd–Max Gaussian
// levels (±0.4528, ±1.5104) to three decimals.
//
// X must be FWHT-pre-rotated by the caller — identical convention to the
// per-block kernel; the rotation path is untouched by this format.
//
// Expert blob layout (SoA, two regions, both naturally aligned):
//   [0 .. M*gpr*64)                     : packed 2-bit indices, 64 B/group
//   [M*gpr*64 .. M*gpr*64 + M*gpr*2)    : fp16 per-block scales
// vs the per-block format's interleaved 72 B/group. 64 B/group means each
// group base is 64 B aligned (the 72 B stride was only 8 B aligned).
//
// The three wins over the per-block kernel, in order of expected value:
//   1. NO __syncthreads() in the main loop. The per-block kernel reloads a
//      16-slot LDS codebook every K4 iteration and barriers each time; here
//      the codebook is loaded ONCE before the loop.
//   2. 64 B/group of weight traffic instead of 72 B — 11.1% less on a
//      BW-bound decode path.
//   3. The scale factors out of the 8-term dot product (one multiply per
//      group instead of folding into every lookup).
//
// Grid: [M, k_top, 1], block [32]. Output split: row<M/2 → y_gate, ≥M/2 → y_up.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256gl_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x_rot,                     // [K] (FWHT-rotated)
    float* __restrict__ y_gate,                          // [k_top × mi]
    float* __restrict__ y_up,                            // [k_top × mi]
    float cb0, float cb1, float cb2, float cb3,          // global codebook
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;

    // Global codebook → LDS ONCE. Dynamic indexing of a 4-element register
    // array compiles to a v_cndmask chain; LDS gives a single ds_read_b32
    // shared by all 8 weights. Unlike the per-block kernel this happens
    // exactly once, so the main loop carries no barrier.
    __shared__ float cb_lds[4];
    if (tid == 0) {
        cb_lds[0] = cb0; cb_lds[1] = cb1; cb_lds[2] = cb2; cb_lds[3] = cb3;
    }
    __syncthreads();

    const char* idx_base   = A + (long long)row * groups_per_row * 64;
    const __half* scale_base = reinterpret_cast<const __half*>(
        A + (long long)M * groups_per_row * 64) + (long long)row * groups_per_row;

    const int boff = tid * 2;  // 2 bytes of indices per group per thread

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOT8(pk, b, sub)                                       \
        (sub) = cb_lds[ (pk)        & 3u] * x_rot[(b)]             \
              + cb_lds[((pk) >>  2) & 3u] * x_rot[(b) + 1]         \
              + cb_lds[((pk) >>  4) & 3u] * x_rot[(b) + 2]         \
              + cb_lds[((pk) >>  6) & 3u] * x_rot[(b) + 3]         \
              + cb_lds[((pk) >>  8) & 3u] * x_rot[(b) + 4]         \
              + cb_lds[((pk) >> 10) & 3u] * x_rot[(b) + 5]         \
              + cb_lds[((pk) >> 12) & 3u] * x_rot[(b) + 6]         \
              + cb_lds[((pk) >> 14) & 3u] * x_rot[(b) + 7]

    // ----- main K4 loop — no barrier, no per-group codebook load -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const unsigned char* d0 = (const unsigned char*)(idx_base + (g    ) * 64) + boff;
        const unsigned char* d1 = (const unsigned char*)(idx_base + (g + 1) * 64) + boff;
        const unsigned char* d2 = (const unsigned char*)(idx_base + (g + 2) * 64) + boff;
        const unsigned char* d3 = (const unsigned char*)(idx_base + (g + 3) * 64) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8);

        const float s0 = __half2float(scale_base[g]);
        const float s1 = __half2float(scale_base[g + 1]);
        const float s2 = __half2float(scale_base[g + 2]);
        const float s3 = __half2float(scale_base[g + 3]);

        const int base = g * 256 + tid * 8;
        float sub0, sub1, sub2, sub3;
        DOT8(pk0, base,       sub0);
        DOT8(pk1, base + 256, sub1);
        DOT8(pk2, base + 512, sub2);
        DOT8(pk3, base + 768, sub3);
        acc0 += s0 * sub0;
        acc1 += s1 * sub1;
        acc2 += s2 * sub2;
        acc3 += s3 * sub3;
    }

    // ----- tail (1, 2, or 3 leftover groups) -----
    #define TAIL(g_expr, acc) do {                                             \
        const int g = (g_expr);                                                \
        const unsigned char* d = (const unsigned char*)(idx_base + g * 64) + boff; \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8);      \
        const float s = __half2float(scale_base[g]);                           \
        const int base = g * 256 + tid * 8;                                    \
        float sub; DOT8(pk, base, sub);                                        \
        (acc) += s * sub;                                                      \
    } while (0)

    if (tail >= 1) TAIL( quads << 2,      acc0);
    if (tail >= 2) TAIL((quads << 2) + 1, acc1);
    if (tail >= 3) TAIL((quads << 2) + 2, acc2);
    #undef TAIL
    #undef DOT8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
