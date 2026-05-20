// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched 2-way fused HFQ4-G256 GEMM for the Qwen3.5 FFN preamble.
// Batched counterpart of fused_gate_up_hfq4g256 — processes N batch elements
// per launch through both projections:
//
//   Y_gate[b] = W_gate · X[b]   for b in 0..N
//   Y_up[b]   = W_up   · X[b]
//
// X is the shared per-token FFN input [N × K]. Each block handles
// (1 output row of gate or up) × (up to BATCH_TILE tokens).
// Grid = (gate_m + up_m) × (ceil(N / BATCH_TILE)) × 1.
//
// Byte-exact with running fused_gate_up_hfq4g256 N times on the same x[b].

#define BATCH_TILE 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_gate_up_hfq4g256(
    const char*  __restrict__ A_gate,
    const char*  __restrict__ A_up,
    const float* __restrict__ X,       // [N × K]
    float*       __restrict__ Y_gate,  // [N × gate_m]
    float*       __restrict__ Y_up,    // [N × up_m]
    int gate_m, int up_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = N - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < gate_m) {
        A = A_gate; Y = Y_gate; row = gid;             out_stride = gate_m;
    } else {
        A = A_up;   Y = Y_up;   row = gid - gate_m;    out_stride = up_m;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0[BATCH_TILE], acc1[BATCH_TILE], acc2[BATCH_TILE], acc3[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
        acc2[b] = 0.0f;
        acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

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

        float w0_0 = sc0 * (float)((pk0       ) & 0xFu) + zp0;
        float w0_1 = sc0 * (float)((pk0 >>  4) & 0xFu) + zp0;
        float w0_2 = sc0 * (float)((pk0 >>  8) & 0xFu) + zp0;
        float w0_3 = sc0 * (float)((pk0 >> 12) & 0xFu) + zp0;
        float w0_4 = sc0 * (float)((pk0 >> 16) & 0xFu) + zp0;
        float w0_5 = sc0 * (float)((pk0 >> 20) & 0xFu) + zp0;
        float w0_6 = sc0 * (float)((pk0 >> 24) & 0xFu) + zp0;
        float w0_7 = sc0 * (float)((pk0 >> 28) & 0xFu) + zp0;

        float w1_0 = sc1 * (float)((pk1       ) & 0xFu) + zp1;
        float w1_1 = sc1 * (float)((pk1 >>  4) & 0xFu) + zp1;
        float w1_2 = sc1 * (float)((pk1 >>  8) & 0xFu) + zp1;
        float w1_3 = sc1 * (float)((pk1 >> 12) & 0xFu) + zp1;
        float w1_4 = sc1 * (float)((pk1 >> 16) & 0xFu) + zp1;
        float w1_5 = sc1 * (float)((pk1 >> 20) & 0xFu) + zp1;
        float w1_6 = sc1 * (float)((pk1 >> 24) & 0xFu) + zp1;
        float w1_7 = sc1 * (float)((pk1 >> 28) & 0xFu) + zp1;

        float w2_0 = sc2 * (float)((pk2       ) & 0xFu) + zp2;
        float w2_1 = sc2 * (float)((pk2 >>  4) & 0xFu) + zp2;
        float w2_2 = sc2 * (float)((pk2 >>  8) & 0xFu) + zp2;
        float w2_3 = sc2 * (float)((pk2 >> 12) & 0xFu) + zp2;
        float w2_4 = sc2 * (float)((pk2 >> 16) & 0xFu) + zp2;
        float w2_5 = sc2 * (float)((pk2 >> 20) & 0xFu) + zp2;
        float w2_6 = sc2 * (float)((pk2 >> 24) & 0xFu) + zp2;
        float w2_7 = sc2 * (float)((pk2 >> 28) & 0xFu) + zp2;

        float w3_0 = sc3 * (float)((pk3       ) & 0xFu) + zp3;
        float w3_1 = sc3 * (float)((pk3 >>  4) & 0xFu) + zp3;
        float w3_2 = sc3 * (float)((pk3 >>  8) & 0xFu) + zp3;
        float w3_3 = sc3 * (float)((pk3 >> 12) & 0xFu) + zp3;
        float w3_4 = sc3 * (float)((pk3 >> 16) & 0xFu) + zp3;
        float w3_5 = sc3 * (float)((pk3 >> 20) & 0xFu) + zp3;
        float w3_6 = sc3 * (float)((pk3 >> 24) & 0xFu) + zp3;
        float w3_7 = sc3 * (float)((pk3 >> 28) & 0xFu) + zp3;

        const int base = g * 256 + tid * 8;

        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const float* xb = X + (long long)(batch_start + b) * K;

            acc0[b] += w0_0 * xb[base    ] + w0_1 * xb[base + 1]
                     + w0_2 * xb[base + 2] + w0_3 * xb[base + 3]
                     + w0_4 * xb[base + 4] + w0_5 * xb[base + 5]
                     + w0_6 * xb[base + 6] + w0_7 * xb[base + 7];

            acc1[b] += w1_0 * xb[base + 256    ] + w1_1 * xb[base + 256 + 1]
                     + w1_2 * xb[base + 256 + 2] + w1_3 * xb[base + 256 + 3]
                     + w1_4 * xb[base + 256 + 4] + w1_5 * xb[base + 256 + 5]
                     + w1_6 * xb[base + 256 + 6] + w1_7 * xb[base + 256 + 7];

            acc2[b] += w2_0 * xb[base + 512    ] + w2_1 * xb[base + 512 + 1]
                     + w2_2 * xb[base + 512 + 2] + w2_3 * xb[base + 512 + 3]
                     + w2_4 * xb[base + 512 + 4] + w2_5 * xb[base + 512 + 5]
                     + w2_6 * xb[base + 512 + 6] + w2_7 * xb[base + 512 + 7];

            acc3[b] += w3_0 * xb[base + 768    ] + w3_1 * xb[base + 768 + 1]
                     + w3_2 * xb[base + 768 + 2] + w3_3 * xb[base + 768 + 3]
                     + w3_4 * xb[base + 768 + 4] + w3_5 * xb[base + 768 + 5]
                     + w3_6 * xb[base + 768 + 6] + w3_7 * xb[base + 768 + 7];
        }
    }

    #define TAIL_BODY(acc_arr)                                                  \
        {                                                                        \
            const char* gp = row_ptr + g * 136;                                  \
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));    \
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);             \
            float w0 = sc * (float)((pk       ) & 0xFu) + zp;                    \
            float w1 = sc * (float)((pk >>  4) & 0xFu) + zp;                    \
            float w2 = sc * (float)((pk >>  8) & 0xFu) + zp;                    \
            float w3 = sc * (float)((pk >> 12) & 0xFu) + zp;                    \
            float w4 = sc * (float)((pk >> 16) & 0xFu) + zp;                    \
            float w5 = sc * (float)((pk >> 20) & 0xFu) + zp;                    \
            float w6 = sc * (float)((pk >> 24) & 0xFu) + zp;                    \
            float w7 = sc * (float)((pk >> 28) & 0xFu) + zp;                    \
            const int base = g * 256 + tid * 8;                                  \
            _Pragma("unroll")                                                    \
            for (int b = 0; b < BATCH_TILE; b++) {                               \
                if (b >= local_bs) break;                                        \
                const float* xb = X + (long long)(batch_start + b) * K;          \
                (acc_arr)[b] += w0 * xb[base    ] + w1 * xb[base + 1]            \
                              + w2 * xb[base + 2] + w3 * xb[base + 3]            \
                              + w4 * xb[base + 4] + w5 * xb[base + 5]            \
                              + w6 * xb[base + 6] + w7 * xb[base + 7];           \
            }                                                                    \
        }

    if (tail >= 1) {
        const int g = quads << 2;
        TAIL_BODY(acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        TAIL_BODY(acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        TAIL_BODY(acc2);
    }
    #undef TAIL_BODY

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
