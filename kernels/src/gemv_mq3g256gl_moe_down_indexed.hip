// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE down GEMV with scaled residual for MQ3-GL — "global
// Lloyd": 3-bit codes against ONE 8-entry codebook shared by the whole tensor,
// plus a per-block fp16 scale. 3.0625 bpw.
//
// This file is the intersection of two existing kernels:
//   * gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed  (qt=19/20 family)
//       — the EPILOGUE and routing contract come from here verbatim:
//         expert-ptrs dispatch, per-krank rotated activation, and the
//         atomicAdd(&x_residual[row], topk_weights[krank] * acc)
//         self-combining store. There is NO separate down-combine kernel.
//   * gemv_mq2g256gl_moe_gate_up_k8_indexed                   (qt=38)
//       — the GL LAYOUT and codebook plumbing come from here: SoA blob with
//         two regions, global codebook arriving as scalar kernel args and
//         staged into LDS exactly once, and the per-group scale factored out
//         of the 8-term dot product.
//
// What differs from the MQ3-Lloyd down sibling, and why:
//   1. NO __syncthreads() in the main loop. The per-block kernel reloads a
//      32-slot LDS codebook (4 groups × 8 fp16 entries read out of the weight
//      blob) on every K4 iteration and barriers each time; here the codebook
//      is 8 scalar kernel args staged into LDS once, before the loop.
//   2. 96 B/group of index traffic + 2 B/group of scale instead of the
//      per-block format's interleaved 112 B/group — 12.5% less weight traffic
//      on a BW-bound decode path.
//   3. The scale factors OUT of the 8-term dot: the codes are dotted against x
//      unscaled into a temporary, then multiplied by the block scale once per
//      group (`acc += s * sub`) instead of folding the scale into all 8
//      codebook lookups. This mirrors gemv_mq2g256gl exactly and matches the
//      encoder, which stores `code = argmin |v/scale - cb[k]|` — i.e.
//      reconstruction is `w = scale_block * cb[q]`.
//
// Expert blob layout (SoA, two regions), from `quantize_mq3g256gl`
// (crates/hipfire-quantize/src/main.rs):
//   [0 .. M*gpr*96)                     : packed 3-bit indices, 96 B/group
//   [M*gpr*96 .. M*gpr*96 + M*gpr*2)    : fp16 per-block scales, 2 B/group
// Both regions are row-major in (row, group); gpr = K/256.
// NOTE vs MQ2GL: 96 B/group is 32 B aligned, not 64 B — the MQ2GL header's
// "each group base is 64 B aligned" coalescing remark does NOT carry over.
//
// Index packing is bit-identical to MQ3-Lloyd (only the codebook source and
// the region layout differ): chunk c of 32 holds codes[8c .. 8c+8] as a
// little-endian 24-bit word, `acc |= code_j << (3*j)`. Thread `tid` owns
// chunk `tid` — 32 threads × 3 B = 96 B — so it reads bytes [3*tid, 3*tid+3)
// and slices q_j = (pk >> 3j) & 7 against x[g*256 + tid*8 + j].
//
// `rot_batch[krank * K .. krank * K + K]` must be FWHT-pre-rotated for the
// down projection (each top-k expert has its own post-gate-up activation;
// each needs an independent rotation before this kernel sees it). The kernel
// never rotates.
//
// Grid: [M, k_top, 1], block [32] (one wave32 → one output row).
// Launch bounds mirror the MQ3-Lloyd down sibling (32, 16), not the MQ2GL
// gate_up's (32, 20) — the MQ3 family carries the wider 8-entry codebook.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const float* __restrict__ rot_batch,                 // [k_top × K] FWHT-rotated
    float*       __restrict__ x_residual,                // [M]
    float cb0, float cb1, float cb2, float cb3,          // global codebook
    float cb4, float cb5, float cb6, float cb7,          // (8 entries, 3-bit)
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

    // Global codebook → LDS ONCE. Dynamic indexing of an 8-element register
    // array compiles to a v_cndmask chain; LDS gives a single ds_read_b32
    // shared by all 8 weights. Unlike the per-block kernel this happens
    // exactly once, so the main loop carries no barrier.
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

    #define DOT8(pk, b, sub)                                    \
        (sub) = cb_lds[ (pk)        & 7u] * x[(b)]              \
              + cb_lds[((pk) >>  3) & 7u] * x[(b) + 1]          \
              + cb_lds[((pk) >>  6) & 7u] * x[(b) + 2]          \
              + cb_lds[((pk) >>  9) & 7u] * x[(b) + 3]          \
              + cb_lds[((pk) >> 12) & 7u] * x[(b) + 4]          \
              + cb_lds[((pk) >> 15) & 7u] * x[(b) + 5]          \
              + cb_lds[((pk) >> 18) & 7u] * x[(b) + 6]          \
              + cb_lds[((pk) >> 21) & 7u] * x[(b) + 7]

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
    #define TAIL(g_expr, acc) do {                                                 \
        const int g = (g_expr);                                                    \
        const unsigned char* d = (const unsigned char*)(idx_base + g * 96) + boff; \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8)           \
                        | ((unsigned int)d[2] << 16);                              \
        const float s = __half2float(scale_base[g]);                               \
        const int base = g * 256 + tid * 8;                                        \
        float sub; DOT8(pk, base, sub);                                            \
        (acc) += s * sub;                                                          \
    } while (0)

    if (tail >= 1) TAIL( quads << 2,      acc0);
    if (tail >= 2) TAIL((quads << 2) + 1, acc1);
    if (tail >= 3) TAIL((quads << 2) + 2, acc2);
    #undef TAIL
    #undef DOT8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    // ATOMIC-RESIDUAL SELF-COMBINING — verbatim from the MQ3-Lloyd down
    // sibling. The k_top grid-y blocks contend on the same x_residual[row]
    // cell, so the FP add order across experts is non-deterministic. There is
    // deliberately NO expanded output buffer and NO separate combine kernel.
    if (tid == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * acc);
    }
}
