// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched N-token indexed MoE gate_up GEMV for OQ4G256 — the batched sibling
// of gemv_oq4g256_moe_gate_up_k8_indexed (each token has its own x row and
// its own topk routing). OQ4 per-group block = 132 B [f32 scale | 128 signed
// nibbles], dequant w = sc * sext4(nib) (no zero-point). See the decode kernel
// and docs/plans/2026-06-30-minimax-moe-oq4-kernels.md for the format delta vs
// the HFQ4 (mq4) 136 B affine variant.
//
// Grid: (M, K_TOP, N). Block = 32 threads.

#define OQ4_BLK 132

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq4g256_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
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
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * OQ4_BLK;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG(pk, sc, b, a) \
        (a) += sc * ( (float)((int)((pk) << 28) >> 28) * x_row[(b)]     \
                    + (float)((int)((pk) << 24) >> 28) * x_row[(b) + 1] \
                    + (float)((int)((pk) << 20) >> 28) * x_row[(b) + 2] \
                    + (float)((int)((pk) << 16) >> 28) * x_row[(b) + 3] \
                    + (float)((int)((pk) << 12) >> 28) * x_row[(b) + 4] \
                    + (float)((int)((pk) <<  8) >> 28) * x_row[(b) + 5] \
                    + (float)((int)((pk) <<  4) >> 28) * x_row[(b) + 6] \
                    + (float)((int)(pk)         >> 28) * x_row[(b) + 7] )

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * OQ4_BLK;
        const char* gp1 = gp0 + OQ4_BLK;
        const char* gp2 = gp1 + OQ4_BLK;
        const char* gp3 = gp2 + OQ4_BLK;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 4 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 4 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 4 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 4 + boff);

        int base = g * 256 + tid * 8;

        DOG(pk0, sc0, base, acc0);
        DOG(pk1, sc1, base + 256, acc1);
        DOG(pk2, sc2, base + 512, acc2);
        DOG(pk3, sc3, base + 768, acc3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * OQ4_BLK;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        unsigned int pk = *(const unsigned int*)(gp + 4 + boff);
        int base = g * 256 + tid * 8;
        DOG(pk, sc, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * OQ4_BLK;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        unsigned int pk = *(const unsigned int*)(gp + 4 + boff);
        int base = g * 256 + tid * 8;
        DOG(pk, sc, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * OQ4_BLK;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        unsigned int pk = *(const unsigned int*)(gp + 4 + boff);
        int base = g * 256 + tid * 8;
        DOG(pk, sc, base, acc2);
    }
    #undef DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
