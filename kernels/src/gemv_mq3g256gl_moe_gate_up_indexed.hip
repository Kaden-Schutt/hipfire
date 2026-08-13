// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for MQ3-GL — "global Lloyd": 3-bit codes
// against ONE 8-entry codebook shared by the whole tensor, plus a per-block
// fp16 scale. 3.0625 bpw. This is the 3-bit sibling of
// gemv_mq2g256gl_moe_gate_up_k8_indexed and the global-codebook sibling of
// gemv_mq3g256_lloyd_moe_gate_up_k8_indexed (112 B/group, per-block codebook).
//
// Why: post-FWHT blocks are Gaussian by CLT, so the optimal LEVEL SHAPE is
// identical in every block — a per-block fit re-derives the same shape ~4000×
// per tensor and differs only by scale. Dropping to a global codebook saves
// the 16 B/group of per-block fp16 levels that MQ3-Lloyd carries (112 → 96+2
// B/group, 3.5 → 3.0625 bpw). The fitted global codebook lands on the textbook
// Lloyd–Max Gaussian 3-bit levels (±0.2451, ±0.7560, ±1.3439, ±2.1520) —
// see GL_CB3 in crates/hipfire-quantize/src/main.rs.
//
// X must be FWHT-pre-rotated by the caller — identical convention to every
// other MQ* kernel; the rotation path is untouched by this format.
//
// Expert blob layout (SoA, two regions):
//   [0 .. M*gpr*96)                     : packed 3-bit indices, 96 B/group
//   [M*gpr*96 .. M*gpr*96 + M*gpr*2)    : fp16 per-block scales, 2 B/group
// Both regions are row-major in (row, group); gpr = K/256.
// NOTE vs the MQ2GL header's alignment claim: 96 B/group is 32 B aligned, not
// 64 B — the "every group base is 64 B aligned" argument does NOT carry over
// to the 3-bit format. The scale-region base M*gpr*96 is always even, so the
// __half* alignment does hold.
//
// What differs from the MQ2GL sibling (gemv_mq2g256gl_moe_gate_up_indexed.hip):
//   * codebook is 8 entries (cb0..cb7 scalar args, cb_lds[8]) not 4;
//   * index stride is 96 B/group not 64; boff = tid*3 not tid*2;
//   * the unpack is a 24-bit little-endian word sliced into 8×3 bits with a
//     cross-byte straddle (see below), not a 16-bit word sliced into 8×2 bits.
// What differs from the MQ3-Lloyd sibling
// (gemv_mq3g256_lloyd_moe_gate_up_indexed.hip):
//   * the codebook is loaded ONCE before the loop, so the main K4 loop carries
//     NO __syncthreads() and no per-group 16 B codebook fetch (the Lloyd
//     kernel re-loads a 32-slot LDS codebook and barriers every iteration);
//   * the per-block scale factors OUT of the 8-term dot product — one multiply
//     per group instead of folding a scale into each of the 8 lookups;
//   * indices/scales are SoA, so there is no 112 B interleaved group stride.
// Numerically the two are NOT bit-identical: the scale is applied once to the
// group sub-sum here versus per-term in the Lloyd kernel.
//
// The 3-bit unpack (the subtle part) — 96 B of indices per 256-group, split as
// 32 chunks × 3 bytes; thread `tid` owns chunk `tid`, i.e. weights
// [8*tid .. 8*tid+8) of the group, i.e. x_rot[g*256 + tid*8 + j] for j in 0..8.
// The encoder (quantize_mq3g256gl, main.rs) writes each chunk as
//     acc |= (code_j & 7) << (3*j);   bytes = acc & 0xFF, acc>>8, acc>>16
// so the kernel rebuilds the identical little-endian 24-bit word
//     pk = d[0] | (d[1] << 8) | (d[2] << 16)
// and slices it LSB-first in weight order: code_j = (pk >> (3*j)) & 7.
// Byte-wise, that means codes 2 and 5 STRADDLE a byte boundary:
//     q0 = b0 & 7                        q4 = (b1 >> 4) & 7
//     q1 = (b0 >> 3) & 7                 q5 = ((b1 >> 7) | (b2 << 1)) & 7
//     q2 = ((b0 >> 6) | (b1 << 2)) & 7   q6 = (b2 >> 2) & 7
//     q3 = (b1 >> 1) & 7                 q7 = (b2 >> 5) & 7
// Building the 24-bit word and shifting handles the straddle for free; do not
// "optimize" this into per-byte masks without reproducing the table above.
// Same cross-byte convention as uniform HFQ3-G256 / MQ3-Lloyd — only the
// codebook source and the region layout differ.
//
// Reconstruction: w = scale_block * cb[code]  (matches gl_encode_block, which
// picks the nearest global-codebook entry to v/scale, scale = fp16(RMS of the
// FWHT-rotated 256-block)).
//
// __launch_bounds__(32, 16) matches every other MQ3 kernel in the tree. The
// MQ2GL sibling asks for 20 and the compiler warns that it settled at 16
// anyway, so 16 is the honest ask here.
//
// Grid: [M, k_top, 1], block [32]. Output split: row<M/2 → y_gate, ≥M/2 → y_up.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256gl_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x_rot,                     // [K] (FWHT-rotated)
    float* __restrict__ y_gate,                          // [k_top × mi]
    float* __restrict__ y_up,                            // [k_top × mi]
    float cb0, float cb1, float cb2, float cb3,          // global codebook
    float cb4, float cb5, float cb6, float cb7,          // (8 entries, ascending)
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

    // Global codebook → LDS ONCE. Dynamic indexing of an 8-element register
    // array compiles to a v_cndmask chain; LDS gives a single ds_read_b32
    // shared by all 8 weights. Unlike the per-block MQ3-Lloyd kernel this
    // happens exactly once, so the main loop carries no barrier.
    __shared__ float cb_lds[8];
    if (tid == 0) {
        cb_lds[0] = cb0; cb_lds[1] = cb1; cb_lds[2] = cb2; cb_lds[3] = cb3;
        cb_lds[4] = cb4; cb_lds[5] = cb5; cb_lds[6] = cb6; cb_lds[7] = cb7;
    }
    __syncthreads();

    const char* idx_base   = A + (long long)row * groups_per_row * 96;
    const __half* scale_base = reinterpret_cast<const __half*>(
        A + (long long)M * groups_per_row * 96) + (long long)row * groups_per_row;

    const int boff = tid * 3;  // 3 bytes of indices per group per thread

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // 8 codes out of one 24-bit little-endian word, LSB-first in weight order.
    #define DOT8(pk, b, sub)                                       \
        (sub) = cb_lds[ (pk)        & 7u] * x_rot[(b)]             \
              + cb_lds[((pk) >>  3) & 7u] * x_rot[(b) + 1]         \
              + cb_lds[((pk) >>  6) & 7u] * x_rot[(b) + 2]         \
              + cb_lds[((pk) >>  9) & 7u] * x_rot[(b) + 3]         \
              + cb_lds[((pk) >> 12) & 7u] * x_rot[(b) + 4]         \
              + cb_lds[((pk) >> 15) & 7u] * x_rot[(b) + 5]         \
              + cb_lds[((pk) >> 18) & 7u] * x_rot[(b) + 6]         \
              + cb_lds[((pk) >> 21) & 7u] * x_rot[(b) + 7]

    // ----- main K4 loop — no barrier, no per-group codebook load -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const unsigned char* d0 = (const unsigned char*)(idx_base + (g    ) * 96) + boff;
        const unsigned char* d1 = (const unsigned char*)(idx_base + (g + 1) * 96) + boff;
        const unsigned char* d2 = (const unsigned char*)(idx_base + (g + 2) * 96) + boff;
        const unsigned char* d3 = (const unsigned char*)(idx_base + (g + 3) * 96) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

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
        const unsigned char* d = (const unsigned char*)(idx_base + g * 96) + boff; \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8)       \
                        | ((unsigned int)d[2] << 16);                          \
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
