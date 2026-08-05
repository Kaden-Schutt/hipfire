// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE down GEMV with scaled residual for MQ2-GL — "global
// Lloyd": 2-bit codes against ONE codebook shared by the whole tensor, plus a
// per-block fp16 scale. Down-projection sibling of
// gemv_mq2g256gl_moe_gate_up_k8_indexed, and the GL counterpart of
// gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed.
//
// What differs from the MQ2-Lloyd down kernel (72 B/group, per-block codebook):
//   1. CODEBOOK IS GLOBAL. It arrives as four scalar float kernel args
//      cb0..cb3 (canonically GL_CB2 = {-1.5104, -0.4528, 0.4528, 1.5104}) and
//      is staged into LDS ONCE before the loop. The Lloyd kernel reloads a
//      16-slot LDS codebook from the group headers on EVERY K4 iteration and
//      barriers each time; here the main loop carries no barrier at all.
//   2. LAYOUT IS SoA, NOT INTERLEAVED. Two regions, no 72 B group header:
//        [0 .. M*gpr*64)                  : packed 2-bit indices, 64 B/group
//        [M*gpr*64 .. M*gpr*64 + M*gpr*2) : fp16 per-block scales, 2 B/group
//      Both regions are row-major in (row, group). 64 B/group means every group
//      base is 64 B aligned (the Lloyd 72 B stride was only 8 B aligned), and
//      it is 11.1% less weight traffic on a BW-bound decode path.
//   3. THE SCALE FACTORS OUT of the 8-term dot product — one multiply per group
//      instead of folding into all four codebook entries. Reconstruction is
//      w = scale_block * cb[q], matching gl_encode_block (nearest neighbour of
//      v / fp16(rms of the FWHT-rotated 256-block) in the global codebook).
//
// Everything else is verbatim from the MQ2-Lloyd down sibling: 2-bit index
// packing (4 codes/byte, little-endian; thread tid owns bytes 2*tid, 2*tid+1 →
// weights 8*tid .. 8*tid+8), K4 accumulator partition with the three tail
// epilogues, the (acc0+acc1)+(acc2+acc3) fold, the 5-step wave32 __shfl_down
// reduce, and — critically — the epilogue.
//
// EPILOGUE IS ATOMIC-RESIDUAL SELF-COMBINING. Lane 0 of every (row, krank)
// block atomicAdds topk_weights[krank] * acc straight into x_residual[row].
// There is NO expanded per-expert buffer and NO separate down-combine kernel
// afterwards; this kernel IS the combine. One launch replaces k_top per-expert
// scaled-gemv-residual launches. (Consequence, same as the Lloyd sibling: the
// k_top grid-y blocks contend on one x_residual cell, so the FP add order is
// non-deterministic even at temp 0.)
//
// `rot_batch[krank * K .. krank * K + K]` must be FWHT-pre-rotated by the
// caller — each top-k expert has its own post-gate-up activation and each needs
// an independent rotation before this kernel sees it. The kernel never rotates.
//
// Grid: [M, k_top, 1], block [32]. K must be a multiple of 256.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const float* __restrict__ rot_batch,                 // [k_top × K] FWHT-rotated
    float*       __restrict__ x_residual,                // [M]
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
    const float* __restrict__ x = rot_batch + (long long)krank * K;

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

    #define DOT8(pk, b, sub)                                   \
        (sub) = cb_lds[ (pk)        & 3u] * x[(b)]             \
              + cb_lds[((pk) >>  2) & 3u] * x[(b) + 1]         \
              + cb_lds[((pk) >>  4) & 3u] * x[(b) + 2]         \
              + cb_lds[((pk) >>  6) & 3u] * x[(b) + 3]         \
              + cb_lds[((pk) >>  8) & 3u] * x[(b) + 4]         \
              + cb_lds[((pk) >> 10) & 3u] * x[(b) + 5]         \
              + cb_lds[((pk) >> 12) & 3u] * x[(b) + 6]         \
              + cb_lds[((pk) >> 14) & 3u] * x[(b) + 7]

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
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * acc);
    }
}
