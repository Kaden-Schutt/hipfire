// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>

// Fused MoE down GEMV + K_TOP weighted-accumulate + residual add.
//
// Replaces THREE kernels of the atomic-free down path:
//   gemv_hfq4g256_moe_down_k8_indexed_batched_expanded  (down -> [N×K_TOP×M])
//   moe_down_combine_k8_batched                          (weighted sum over K_TOP)
//   the residual add                                     (x_residual += combined)
//
// Each block owns one (token, row). It loops all K_TOP experts INTERNALLY,
// computes each expert's down dot for `row`, weight-scales by topk_weights,
// register-accumulates across experts, and does a SINGLE race-free
//   x_residual[token][row] += sum_k weight[k] * down_k(row)
// Each output cell is owned by exactly one block, so no atomicAdd is needed
// AND no N×K_TOP×M expanded intermediate is materialized (the expanded path's
// 8× down-output DRAM write + re-read in combine is eliminated). This is the
// dominant intermediate-traffic waste in decode MoE FFN.
//
// Algorithm (WG-owns-row, register-accumulate over experts) learned from
// ZINC's dmmv_q4k_moe_fused_down_acc.comp (MIT, github.com/zolotukhin/zinc);
// reimplemented in HIP for the HFQ4-G256 (MQ4) layout. See NOTICE.
//
// NUMERICAL ORDER matches the (expanded + combine) path it replaces:
//   per expert : 4-accumulator pairwise (acc0..3) -> __shfl_down warp reduce
//   over experts: sequential  weighted_acc += weight[k] * per_expert_result  (k=0..K_TOP-1)
//   -> x_residual += weighted_acc
// so it reproduces that path's reduction ordering (byte-exactness target).
//
// Shapes:
//   expert_ptrs  : [n_exp]        device pointer table (per-expert down weight)
//   topk_indices : [N × K_TOP]
//   topk_weights : [N × K_TOP]
//   rot_batch    : [N × K_TOP × K] (silu*mul output, per (token,krank))
//   x_residual   : [N × M]        (in-place +=)
// MQ4 group = 136 bytes (4 scale + 4 zp + 128 packed 4-bit), 256 elems/group.
//
// Grid: (M, N). Block = 32 threads (one warp per output row).

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_down_k8_indexed_fused_acc(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ topk_weights,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ x_residual,                // [N × M] += result
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bid = blockIdx.y;   // token id
    const int tid = threadIdx.x;

    const int routing_base   = bid * K_TOP;
    const int groups_per_row = K / 256;
    const int boff           = tid * 4;
    const int quads          = groups_per_row >> 2;
    const int tail           = groups_per_row & 3;

    float weighted_acc = 0.0f;   // cross-expert running sum (valid on tid 0)

    for (int krank = 0; krank < K_TOP; krank++) {
        const int expert_id = topk_indices[routing_base + krank];
        const char* __restrict__ A =
            reinterpret_cast<const char*>(expert_ptrs[expert_id]);
        const float* __restrict__ x =
            rot_batch + ((long long)bid * K_TOP + krank) * K;
        const char* row_ptr = A + (long long)row * groups_per_row * 136;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
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

            int base = g * 256 + tid * 8;

            #define DOG(pk, sc, zp, b, a) \
                (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                     + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                     + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                     + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                     + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                     + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                     + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                     + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

            DOG(pk0, sc0, zp0, base, acc0);
            DOG(pk1, sc1, zp1, base + 256, acc1);
            DOG(pk2, sc2, zp2, base + 512, acc2);
            DOG(pk3, sc3, zp3, base + 768, acc3);
            #undef DOG
        }

        #define TAIL_DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        if (tail >= 1) {
            const int g = quads << 2;
            const char* gp = row_ptr + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            int base = g * 256 + tid * 8;
            TAIL_DOG(pk, sc, zp, base, acc0);
        }
        if (tail >= 2) {
            const int g = (quads << 2) + 1;
            const char* gp = row_ptr + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            int base = g * 256 + tid * 8;
            TAIL_DOG(pk, sc, zp, base, acc1);
        }
        if (tail >= 3) {
            const int g = (quads << 2) + 2;
            const char* gp = row_ptr + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            int base = g * 256 + tid * 8;
            TAIL_DOG(pk, sc, zp, base, acc2);
        }
        #undef TAIL_DOG

        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);

        if (tid == 0)
            weighted_acc += topk_weights[routing_base + krank] * acc;
    }

    if (tid == 0)
        x_residual[(long long)bid * M + row] += weighted_acc;
}
