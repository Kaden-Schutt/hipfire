// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched HFQ6-G256 GEMM with fused residual add:
//
//   for b in 0..N:
//     Y[b, row] += A[row] . X[b]         (residual accumulation)
//
// 6-bit variant of gemm_hfq4g256_residual. HFQ6-G256 packs 256 weights
// per group in 200 bytes: [f32 scale][f32 zero][192B packed 6-bit data].
//
// 6-bit packing: 4 values per 3 bytes.
//   byte0 = q0 | (q1 << 6)
//   byte1 = (q1 >> 2) | (q2 << 4)
//   byte2 = (q2 >> 4) | (q3 << 2)
// Unpack:
//   q0 = b0 & 63
//   q1 = (b0 >> 6) | ((b1 & 0xF) << 2)
//   q2 = (b1 >> 4) | ((b2 & 3) << 4)
//   q3 = b2 >> 2
//
// Each thread handles 8 weights = 6 bytes (was 4 bytes for HFQ4).
// 32 threads × 8 weights = 256 weights per group.
//
// **Byte-exactness constraint:** MUST produce bitwise-identical output to
// calling the single-token GEMV N times, because the quality gate
// compares token IDs byte-for-byte. That requires keeping the same
// 4-accumulator interleave + pairwise final combine:
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
// Grid layout: (M rows) x (ceil(batch_size / BATCH_TILE) batch-tiles).
// Each block handles (1 row x up-to-BATCH_TILE batch elements).

#define BATCH_TILE 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_hfq6g256_residual(
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
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;  // byte offset into 6-bit data (8 values = 6 bytes)

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
        const char* gp0 = row_ptr + g * 200;
        const char* gp1 = gp0 + 200;
        const char* gp2 = gp1 + 200;
        const char* gp3 = gp2 + 200;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        // Read 6 bytes per group for this thread's 8 weights (6-bit packed)
        const unsigned char* dp0 = (const unsigned char*)(gp0 + 8 + boff);
        const unsigned char* dp1 = (const unsigned char*)(gp1 + 8 + boff);
        const unsigned char* dp2 = (const unsigned char*)(gp2 + 8 + boff);
        const unsigned char* dp3 = (const unsigned char*)(gp3 + 8 + boff);

        // Pre-dequantize the 8 weights for each of the 4 groups. These
        // are invariant across the batch loop -- the whole point of the
        // batching is to amortize these dequant ops across N tokens.

        // Group 0: unpack 6 bytes -> 8 x 6-bit values
        unsigned char b0_0 = dp0[0], b0_1 = dp0[1], b0_2 = dp0[2];
        unsigned char b0_3 = dp0[3], b0_4 = dp0[4], b0_5 = dp0[5];
        float w0_0 = sc0 * (float)(b0_0 & 63) + zp0;
        float w0_1 = sc0 * (float)((b0_0 >> 6) | ((b0_1 & 0xF) << 2)) + zp0;
        float w0_2 = sc0 * (float)((b0_1 >> 4) | ((b0_2 & 3) << 4)) + zp0;
        float w0_3 = sc0 * (float)(b0_2 >> 2) + zp0;
        float w0_4 = sc0 * (float)(b0_3 & 63) + zp0;
        float w0_5 = sc0 * (float)((b0_3 >> 6) | ((b0_4 & 0xF) << 2)) + zp0;
        float w0_6 = sc0 * (float)((b0_4 >> 4) | ((b0_5 & 3) << 4)) + zp0;
        float w0_7 = sc0 * (float)(b0_5 >> 2) + zp0;

        // Group 1
        unsigned char b1_0 = dp1[0], b1_1 = dp1[1], b1_2 = dp1[2];
        unsigned char b1_3 = dp1[3], b1_4 = dp1[4], b1_5 = dp1[5];
        float w1_0 = sc1 * (float)(b1_0 & 63) + zp1;
        float w1_1 = sc1 * (float)((b1_0 >> 6) | ((b1_1 & 0xF) << 2)) + zp1;
        float w1_2 = sc1 * (float)((b1_1 >> 4) | ((b1_2 & 3) << 4)) + zp1;
        float w1_3 = sc1 * (float)(b1_2 >> 2) + zp1;
        float w1_4 = sc1 * (float)(b1_3 & 63) + zp1;
        float w1_5 = sc1 * (float)((b1_3 >> 6) | ((b1_4 & 0xF) << 2)) + zp1;
        float w1_6 = sc1 * (float)((b1_4 >> 4) | ((b1_5 & 3) << 4)) + zp1;
        float w1_7 = sc1 * (float)(b1_5 >> 2) + zp1;

        // Group 2
        unsigned char b2_0 = dp2[0], b2_1 = dp2[1], b2_2 = dp2[2];
        unsigned char b2_3 = dp2[3], b2_4 = dp2[4], b2_5 = dp2[5];
        float w2_0 = sc2 * (float)(b2_0 & 63) + zp2;
        float w2_1 = sc2 * (float)((b2_0 >> 6) | ((b2_1 & 0xF) << 2)) + zp2;
        float w2_2 = sc2 * (float)((b2_1 >> 4) | ((b2_2 & 3) << 4)) + zp2;
        float w2_3 = sc2 * (float)(b2_2 >> 2) + zp2;
        float w2_4 = sc2 * (float)(b2_3 & 63) + zp2;
        float w2_5 = sc2 * (float)((b2_3 >> 6) | ((b2_4 & 0xF) << 2)) + zp2;
        float w2_6 = sc2 * (float)((b2_4 >> 4) | ((b2_5 & 3) << 4)) + zp2;
        float w2_7 = sc2 * (float)(b2_5 >> 2) + zp2;

        // Group 3
        unsigned char b3_0 = dp3[0], b3_1 = dp3[1], b3_2 = dp3[2];
        unsigned char b3_3 = dp3[3], b3_4 = dp3[4], b3_5 = dp3[5];
        float w3_0 = sc3 * (float)(b3_0 & 63) + zp3;
        float w3_1 = sc3 * (float)((b3_0 >> 6) | ((b3_1 & 0xF) << 2)) + zp3;
        float w3_2 = sc3 * (float)((b3_1 >> 4) | ((b3_2 & 3) << 4)) + zp3;
        float w3_3 = sc3 * (float)(b3_2 >> 2) + zp3;
        float w3_4 = sc3 * (float)(b3_3 & 63) + zp3;
        float w3_5 = sc3 * (float)((b3_3 >> 6) | ((b3_4 & 0xF) << 2)) + zp3;
        float w3_6 = sc3 * (float)((b3_4 >> 4) | ((b3_5 & 3) << 4)) + zp3;
        float w3_7 = sc3 * (float)(b3_5 >> 2) + zp3;

        const int base = g * 256 + tid * 8;

        // Batch loop. Identical accumulation ordering as the GEMV for
        // bitwise reproducibility.
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

    // Tail groups (0-3 leftover groups after the quad loop). Each tail
    // group g lands in acc[g & 3][b] to preserve the GEMV's tail-group
    // distribution.
    #define TAIL_BODY(acc_arr)                                                  \
        {                                                                        \
            const char* gp = row_ptr + g * 200;                                  \
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));    \
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
            const unsigned char* dp = (const unsigned char*)(gp + 8 + boff);     \
            unsigned char b0 = dp[0], b1 = dp[1], b2 = dp[2];                   \
            unsigned char b3 = dp[3], b4 = dp[4], b5 = dp[5];                   \
            float w0 = sc * (float)(b0 & 63) + zp;                              \
            float w1 = sc * (float)((b0 >> 6) | ((b1 & 0xF) << 2)) + zp;       \
            float w2 = sc * (float)((b1 >> 4) | ((b2 & 3) << 4)) + zp;         \
            float w3 = sc * (float)(b2 >> 2) + zp;                              \
            float w4 = sc * (float)(b3 & 63) + zp;                              \
            float w5 = sc * (float)((b3 >> 6) | ((b4 & 0xF) << 2)) + zp;       \
            float w6 = sc * (float)((b4 >> 4) | ((b5 & 3) << 4)) + zp;         \
            float w7 = sc * (float)(b5 >> 2) + zp;                              \
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
