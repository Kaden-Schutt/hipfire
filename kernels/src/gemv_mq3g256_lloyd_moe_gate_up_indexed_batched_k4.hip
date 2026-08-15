// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// K4-unrolled BATCHED indexed MoE gate_up GEMV for MQ3-Lloyd (3-bit +
// per-block 8-entry fp16 codebook, 112 B/group). This is the multi-token
// sibling of `gemv_mq3g256_lloyd_moe_gate_up_k8_indexed`: MQ3-Lloyd had no
// batched variant at all, so batched prefill fell back off the MoE path for
// MQ3-Lloyd models.
//
// Composition (deliberate — two parents):
//   * Batching structure, grid, routing, X indexing, output layout and the
//     4-accumulator unroll come from
//     `gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4` VERBATIM.
//   * The weight decode (112 B/group, 8-entry fp16 codebook, 3-bit unpack,
//     LDS codebook staging) comes from
//     `gemv_mq3g256_lloyd_moe_gate_up_k8_indexed` VERBATIM.
//
// `_k4` = 4 independent accumulators / ILP unroll over 4 consecutive
// 256-groups. It is NOT k_top=4 — K_TOP is a runtime kernel argument
// (grid.y), exactly as in the MQ2 batched sibling. The trailing `k8` in the
// entry name is likewise legacy naming from the original k_top=8 shape, not a
// compile-time constant.
//
// Shapes (identical to the MQ2 batched sibling):
//   expert_ptrs : [n_exp]   device-pointer table, shared across tokens
//   topk_indices: [N × K_TOP]
//   x           : [N × K]                  (FWHT-rotated post-RMSNorm)
//   y_gate      : [N × K_TOP × MI]         (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Block = 32 threads (one wave32, one output row).
// X is per-TOKEN, not per-krank: the FWHT rotation is done once per token and
// re-used across that token's top-k experts (down is the per-krank one).
//
// Block layout (112 B / 256 weights) — vs MQ2-Lloyd's 72 B:
//   [0..16)   : 8 × fp16 codebook entries (sorted ascending)   [MQ2: 4 × fp16]
//   [16..112) : 96 bytes packed 3-bit indices (32 chunks × 3 bytes,
//               8 weights/chunk; per-thread 3-byte little-endian load,
//               (pk >> (3*i)) & 7u for i in 0..8).             [MQ2: 64 B, 2-bit]
//
// Two deliberate deviations from the MQ2 batched sibling, both inherited from
// the MQ3 decode and both required by it:
//   1. LDS codebook (32 slots = 4 groups × 8 entries) with a cooperative load
//      + `__syncthreads()` per K4 iteration, instead of MQ2's per-thread
//      `float cb[4]` register array. An 8-entry register array would be
//      dynamically indexed and would spill to scratch; every MQ3 kernel in the
//      tree stages the codebook through LDS or an 8-way ternary chain. All
//      barriers here are block-uniform (row/quads/tail derive from blockIdx.x
//      and K), so the early `row >= M` return cannot strand a barrier.
//   2. Tail groups use the MQ3 sibling's three `if (tail >= n)` expansions
//      folding into acc0/acc1/acc2, not MQ2 batched's single acc0 loop —
//      the MQ3 tail needs the `tid < 8` LDS reload + barrier per group, which
//      does not fit MQ2's barrier-free lambda. Numerically both land inside
//      the same FMA-order epsilon.
//
// `__launch_bounds__(32, 16)` matches every MQ3 kernel in the tree (MQ2 uses
// 20; MQ3's 8 codebook registers do not sustain it).
//
// Numerical contract:
// FMA reordering between this (4 acc) and the single-acc variants means the
// result is NOT bit-identical to a 1-acc kernel, but is within FMA-order
// epsilon. Group→accumulator mapping (g mod 4) and the fold parenthesization
// `(acc0 + acc1) + (acc2 + acc3)` match both parents exactly.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x_rot,                     // [N × K] FWHT-rotated
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x_rot + (long long)bid * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 112;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 3;  // 3 bytes of indices per group per thread
    const int xoff = tid * 8;

    // 32-slot LDS codebook: 4 groups × 8 entries. Group g's codebook
    // occupies cb_lds[g*8 .. g*8+8].
    __shared__ float cb_lds[32];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // ----- main K4 loop -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 112;
        const char* gp1 = gp0 + 112;
        const char* gp2 = gp1 + 112;
        const char* gp3 = gp2 + 112;

        // Cooperative load: 32 threads × 1 fp16 → 32 fp32 entries in LDS.
        // thread tid → group (tid >> 3), entry (tid & 7).
        {
            const __half* cb_h =
                (const __half*)(gp0 + (long long)(tid >> 3) * 112);
            cb_lds[tid] = __half2float(cb_h[tid & 7]);
        }
        __syncthreads();

        // Load 3 packed bytes (one per group, this thread's 3 bytes).
        const unsigned char* d0 = (const unsigned char*)(gp0 + 16) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 16) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 16) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 16) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

        int base = g * 256 + xoff;

        #define DOG3_LDS(pk, lds, b, a)                                   \
            (a) += cb_lds[(lds) + ( (pk)        & 7u)] * x_row[(b)]       \
                 + cb_lds[(lds) + (((pk) >>  3) & 7u)] * x_row[(b) + 1]   \
                 + cb_lds[(lds) + (((pk) >>  6) & 7u)] * x_row[(b) + 2]   \
                 + cb_lds[(lds) + (((pk) >>  9) & 7u)] * x_row[(b) + 3]   \
                 + cb_lds[(lds) + (((pk) >> 12) & 7u)] * x_row[(b) + 4]   \
                 + cb_lds[(lds) + (((pk) >> 15) & 7u)] * x_row[(b) + 5]   \
                 + cb_lds[(lds) + (((pk) >> 18) & 7u)] * x_row[(b) + 6]   \
                 + cb_lds[(lds) + (((pk) >> 21) & 7u)] * x_row[(b) + 7]

        DOG3_LDS(pk0,  0, base,        acc0);
        DOG3_LDS(pk1,  8, base + 256,  acc1);
        DOG3_LDS(pk2, 16, base + 512,  acc2);
        DOG3_LDS(pk3, 24, base + 768,  acc3);
        #undef DOG3_LDS
    }

    // ----- tail loop (1, 2, or 3 leftover groups) -----
    #define TAIL_LOAD_AND_DOT(g_expr, acc) do {                                 \
        const int g = (g_expr);                                                  \
        const char* gp = row_ptr + g * 112;                                      \
        if (tid < 8) {                                                           \
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);                \
        }                                                                        \
        __syncthreads();                                                         \
        const unsigned char* d = (const unsigned char*)(gp + 16) + boff;         \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8)         \
                        | ((unsigned int)d[2] << 16);                            \
        int base = g * 256 + xoff;                                               \
        (acc) += cb_lds[ (pk)        & 7u] * x_row[base]                         \
               + cb_lds[((pk) >>  3) & 7u] * x_row[base + 1]                     \
               + cb_lds[((pk) >>  6) & 7u] * x_row[base + 2]                     \
               + cb_lds[((pk) >>  9) & 7u] * x_row[base + 3]                     \
               + cb_lds[((pk) >> 12) & 7u] * x_row[base + 4]                     \
               + cb_lds[((pk) >> 15) & 7u] * x_row[base + 5]                     \
               + cb_lds[((pk) >> 18) & 7u] * x_row[base + 6]                     \
               + cb_lds[((pk) >> 21) & 7u] * x_row[base + 7];                    \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2,       acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1,  acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2,  acc2);
    #undef TAIL_LOAD_AND_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long stride = (long long)K_TOP * mi;
        if (row < mi) {
            y_gate[(long long)bid * stride + krank * mi + row] = acc;
        } else {
            y_up[(long long)bid * stride + krank * mi + (row - mi)] = acc;
        }
    }
}
