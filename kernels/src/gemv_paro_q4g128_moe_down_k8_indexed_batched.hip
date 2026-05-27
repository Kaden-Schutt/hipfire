// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Atomic-free batched indexed MoE down GEMV for the HFQ4G128 (ParoQuant)
// layout — mechanical G128 port of
// `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`. Each (token, krank,
// row) workgroup writes a single output to its own slot in an expanded
// [N × K_TOP × M] buffer; `moe_down_combine_k8_batched` (dtype-independent)
// then folds K_TOP slots into the residual stream with topk_weights applied.
//
// Group layout (matches gemv_hfq4g128.hip): 72 B/group = 4 B scale
// + 4 B zero + 64 B packed nibbles (128 4-bit weights). 32 threads per
// block, 4 weights per thread per group, `byte_off = tid * 2`.
//
// Grid: (M, K_TOP, N). Block = 32 threads. No atomic ops — each output
// address is written by exactly one workgroup.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_paro_q4g128_moe_down_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 72;
    const int byte_off = tid * 2;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        const unsigned char* nb0 = (const unsigned char*)(gp0 + 8);
        const unsigned char* nb1 = (const unsigned char*)(gp1 + 8);
        const unsigned char* nb2 = (const unsigned char*)(gp2 + 8);
        const unsigned char* nb3 = (const unsigned char*)(gp3 + 8);

        unsigned char b00 = nb0[byte_off];
        unsigned char b01 = nb0[byte_off + 1];
        unsigned char b10 = nb1[byte_off];
        unsigned char b11 = nb1[byte_off + 1];
        unsigned char b20 = nb2[byte_off];
        unsigned char b21 = nb2[byte_off + 1];
        unsigned char b30 = nb3[byte_off];
        unsigned char b31 = nb3[byte_off + 1];

        int base = g * 128 + tid * 4;

        #define DOG(b0, b1, sc, zp, base_idx, a) \
            (a) += (sc * (float)((b0) & 0xFu)        + zp) * x[(base_idx)]     \
                 + (sc * (float)(((b0) >> 4) & 0xFu) + zp) * x[(base_idx) + 1] \
                 + (sc * (float)((b1) & 0xFu)        + zp) * x[(base_idx) + 2] \
                 + (sc * (float)(((b1) >> 4) & 0xFu) + zp) * x[(base_idx) + 3]

        DOG(b00, b01, sc0, zp0, base,       acc0);
        DOG(b10, b11, sc1, zp1, base + 128, acc1);
        DOG(b20, b21, sc2, zp2, base + 256, acc2);
        DOG(b30, b31, sc3, zp3, base + 384, acc3);
        #undef DOG
    }

    #define TAIL_DOG(b0, b1, sc, zp, base_idx, a) \
        (a) += (sc * (float)((b0) & 0xFu)        + zp) * x[(base_idx)]     \
             + (sc * (float)(((b0) >> 4) & 0xFu) + zp) * x[(base_idx) + 1] \
             + (sc * (float)((b1) & 0xFu)        + zp) * x[(base_idx) + 2] \
             + (sc * (float)(((b1) >> 4) & 0xFu) + zp) * x[(base_idx) + 3]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 72;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* nb = (const unsigned char*)(gp + 8);
        unsigned char b0 = nb[byte_off];
        unsigned char b1 = nb[byte_off + 1];
        int base = g * 128 + tid * 4;
        TAIL_DOG(b0, b1, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 72;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* nb = (const unsigned char*)(gp + 8);
        unsigned char b0 = nb[byte_off];
        unsigned char b1 = nb[byte_off + 1];
        int base = g * 128 + tid * 4;
        TAIL_DOG(b0, b1, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 72;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* nb = (const unsigned char*)(gp + 8);
        unsigned char b0 = nb[byte_off];
        unsigned char b1 = nb[byte_off + 1];
        int base = g * 128 + tid * 4;
        TAIL_DOG(b0, b1, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const long long out_offset =
            ((long long)bid * K_TOP + krank) * (long long)M + row;
        expert_outputs[out_offset] = acc;
    }
}
