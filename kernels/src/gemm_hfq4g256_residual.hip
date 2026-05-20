// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched HFQ4-G256 GEMM with fused residual add:
//
//   for b in 0..N:
//     Y[b, row] += A[row] · X[b]         (residual accumulation)
//
// The single-token GEMV `gemv_hfq4g256_residual` runs this math one batch
// element at a time during decode. This batched variant processes N tokens
// per launch during prefill so each row's weights are loaded into the
// per-CU cache once and reused across all N elements.
//
// **Byte-exactness constraint:** MUST produce bitwise-identical output to
// calling gemv_hfq4g256_residual N times, because the MQ4 quality gate
// compares token IDs byte-for-byte. That requires keeping the same
// 4-accumulator interleave + pairwise final combine that gemv does:
//
//   acc0..acc3 = 0
//   for q in 0..quads:
//     acc0 += DOG(group[4q+0], x)
//     acc1 += DOG(group[4q+1], x)
//     acc2 += DOG(group[4q+2], x)
//     acc3 += DOG(group[4q+3], x)
//   // tail groups distributed to acc[g % 4]
//   result = (acc0 + acc1) + (acc2 + acc3)
//
// Per-batch-element storage: 4 accumulators × BATCH_TILE = 4×8 = 32 FP32s
// live in registers, plus scalars. At 32 threads/block that's manageable
// and leaves room for __launch_bounds__(32, 8) occupancy.
//
// Grid layout: (M rows) × (ceil(batch_size / BATCH_TILE) batch-tiles).
// Each block handles (1 row × up-to-BATCH_TILE batch elements).

#define BATCH_TILE 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_hfq4g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ X,   // [batch_size, K], row-major
    float* __restrict__ Y,         // [batch_size, M], row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    // Per-batch-element 4-accumulator interleave. Matches the GEMV's
    // reduction order exactly: group 4q+k feeds acc[k][b] for each b.
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

        // Pre-dequantize the 8 weights for each of the 4 groups. These
        // are invariant across the batch loop — the whole point of the
        // batching is to amortize these dequant ops across N tokens.
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

        // Batch loop. GEMV's DOG macro compiles to exactly this ordering
        // of 8 FMAs per group, so accumulating per-batch-element into
        // acc[k][b] with identical expression ordering yields bitwise
        // identical results vs running GEMV 8 times.
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const float* xb = X + (long long)(batch_start + b) * K;
            const float x0 = xb[base    ];
            const float x1 = xb[base + 1];
            const float x2 = xb[base + 2];
            const float x3 = xb[base + 3];
            const float x4 = xb[base + 4];
            const float x5 = xb[base + 5];
            const float x6 = xb[base + 6];
            const float x7 = xb[base + 7];

            acc0[b] += w0_0 * x0 + w0_1 * x1 + w0_2 * x2 + w0_3 * x3
                     + w0_4 * x4 + w0_5 * x5 + w0_6 * x6 + w0_7 * x7;

            const float x1_0 = xb[base + 256    ];
            const float x1_1 = xb[base + 256 + 1];
            const float x1_2 = xb[base + 256 + 2];
            const float x1_3 = xb[base + 256 + 3];
            const float x1_4 = xb[base + 256 + 4];
            const float x1_5 = xb[base + 256 + 5];
            const float x1_6 = xb[base + 256 + 6];
            const float x1_7 = xb[base + 256 + 7];

            acc1[b] += w1_0 * x1_0 + w1_1 * x1_1 + w1_2 * x1_2 + w1_3 * x1_3
                     + w1_4 * x1_4 + w1_5 * x1_5 + w1_6 * x1_6 + w1_7 * x1_7;

            const float x2_0 = xb[base + 512    ];
            const float x2_1 = xb[base + 512 + 1];
            const float x2_2 = xb[base + 512 + 2];
            const float x2_3 = xb[base + 512 + 3];
            const float x2_4 = xb[base + 512 + 4];
            const float x2_5 = xb[base + 512 + 5];
            const float x2_6 = xb[base + 512 + 6];
            const float x2_7 = xb[base + 512 + 7];

            acc2[b] += w2_0 * x2_0 + w2_1 * x2_1 + w2_2 * x2_2 + w2_3 * x2_3
                     + w2_4 * x2_4 + w2_5 * x2_5 + w2_6 * x2_6 + w2_7 * x2_7;

            const float x3_0 = xb[base + 768    ];
            const float x3_1 = xb[base + 768 + 1];
            const float x3_2 = xb[base + 768 + 2];
            const float x3_3 = xb[base + 768 + 3];
            const float x3_4 = xb[base + 768 + 4];
            const float x3_5 = xb[base + 768 + 5];
            const float x3_6 = xb[base + 768 + 6];
            const float x3_7 = xb[base + 768 + 7];

            acc3[b] += w3_0 * x3_0 + w3_1 * x3_1 + w3_2 * x3_2 + w3_3 * x3_3
                     + w3_4 * x3_4 + w3_5 * x3_5 + w3_6 * x3_6 + w3_7 * x3_7;
        }
    }

    // Tail groups (0–3 leftover groups after the quad loop). Each tail
    // group g lands in acc[g & 3][b] to preserve the GEMV's tail-group
    // distribution (see gemv_hfq4g256.hip for the bug story behind this).
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

    // Per-batch-element pairwise combine + warp reduction, matching GEMV.
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row] += val;
    }
}
