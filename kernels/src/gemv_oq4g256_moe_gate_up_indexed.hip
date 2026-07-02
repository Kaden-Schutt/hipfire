// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Index-aware fused MoE gate_up GEMV for OQ4G256 (Opus Quant W4 experts).
// Structurally identical to gemv_hfq4g256_moe_gate_up_k8_indexed; the ONLY
// difference is the per-group expert weight block:
//
//   HFQ4 (mq4): 136 B = [f32 scale | f32 zero-point | 128 unsigned nibbles],
//               dequant w = sc*nib + zp        (affine / asymmetric)
//   OQ4G256:    132 B = [f32 scale | 128 SIGNED nibbles],
//               dequant w = sc * sext4(nib)    (symmetric, no zero-point)
//
// This 132 B layout is the INTERLEAVED_BLOCK section the OQ4 arch-combined
// packing already emits; the minimax loader fuses each expert's interleaved
// section into the gate_up blob and the expert ptr table points at its base.
// At batch=1 decode this is W4A16 (unpack int4 × f32 activation), matching the
// dense gemv_oq4_grouped path — no int8 activation quant.
//
// Output layout matches the hfq4 variant: split into y_gate[K_TOP×MI] and
// y_up[K_TOP×MI], feeding the existing batched silu_mul_rotate.

#define OQ4_BLK 132

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq4g256_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
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
    const char* row_ptr = A + (long long)row * groups_per_row * OQ4_BLK;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // sc * sign-extend(nibble): ((int)nib << 28) >> 28 maps 0..15 → -8..7.
    #define DOG(pk, sc, b, a) \
        (a) += sc * ( (float)((int)((pk) << 28)        >> 28) * x[(b)]     \
                    + (float)((int)((pk) << 24)        >> 28) * x[(b) + 1] \
                    + (float)((int)((pk) << 20)        >> 28) * x[(b) + 2] \
                    + (float)((int)((pk) << 16)        >> 28) * x[(b) + 3] \
                    + (float)((int)((pk) << 12)        >> 28) * x[(b) + 4] \
                    + (float)((int)((pk) <<  8)        >> 28) * x[(b) + 5] \
                    + (float)((int)((pk) <<  4)        >> 28) * x[(b) + 6] \
                    + (float)((int)(pk)                >> 28) * x[(b) + 7] )

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
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
