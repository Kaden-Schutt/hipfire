// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched 4-way fused HFQ6-G256 GEMM for the Qwen3.5 DeltaNet LA preamble.
// Batched counterpart of fused_qkvza_hfq6g256 — processes N batch elements
// per launch through the same four projections:
//
//   Y_qkv[b]   = W_qkv   · X[b]      for b in 0..N
//   Y_z[b]     = W_z     · X[b]
//   Y_beta[b]  = W_beta  · X[b]
//   Y_alpha[b] = W_alpha · X[b]
//
// X is the shared per-token activation [N × K]. Each block handles
// (1 output row of one of the 4 projections) × (up to BATCH_TILE tokens).
// Grid = (total_m) × (ceil(N / BATCH_TILE)) × 1.
//
// HFQ6-G256 group layout: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes
// Per thread: 8 weights in 6 bytes (4 values per 3 bytes).
//
// Overwrite semantics (y[b][row] = acc), matching the GEMV reference.

#define BATCH_TILE 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_qkvza_hfq6g256(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ X,        // [N × K]
    float*       __restrict__ Y_qkv,    // [N × qkv_m]
    float*       __restrict__ Y_z,      // [N × z_m]
    float*       __restrict__ Y_beta,   // [N × beta_m]
    float*       __restrict__ Y_alpha,  // [N × alpha_m]
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = N - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    // Route this workgroup to its target matrix + output row + output stride.
    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < qkv_m) {
        A = A_qkv;   Y = Y_qkv;   row = gid;                              out_stride = qkv_m;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     Y = Y_z;     row = gid - qkv_m;                       out_stride = z_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  Y = Y_beta;  row = gid - (qkv_m + z_m);                out_stride = beta_m;
    } else {
        A = A_alpha; Y = Y_alpha; row = gid - (qkv_m + z_m + beta_m);       out_stride = alpha_m;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;

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

        const unsigned char* dp0 = (const unsigned char*)(gp0 + 8 + boff);
        const unsigned char* dp1 = (const unsigned char*)(gp1 + 8 + boff);
        const unsigned char* dp2 = (const unsigned char*)(gp2 + 8 + boff);
        const unsigned char* dp3 = (const unsigned char*)(gp3 + 8 + boff);

        unsigned char b0_0=dp0[0], b0_1=dp0[1], b0_2=dp0[2], b0_3=dp0[3], b0_4=dp0[4], b0_5=dp0[5];
        unsigned char b1_0=dp1[0], b1_1=dp1[1], b1_2=dp1[2], b1_3=dp1[3], b1_4=dp1[4], b1_5=dp1[5];
        unsigned char b2_0=dp2[0], b2_1=dp2[1], b2_2=dp2[2], b2_3=dp2[3], b2_4=dp2[4], b2_5=dp2[5];
        unsigned char b3_0=dp3[0], b3_1=dp3[1], b3_2=dp3[2], b3_3=dp3[3], b3_4=dp3[4], b3_5=dp3[5];

        float w0_0 = sc0*(float)(b0_0 & 63)+zp0;
        float w0_1 = sc0*(float)((b0_0>>6)|((b0_1&0xF)<<2))+zp0;
        float w0_2 = sc0*(float)((b0_1>>4)|((b0_2&3)<<4))+zp0;
        float w0_3 = sc0*(float)(b0_2>>2)+zp0;
        float w0_4 = sc0*(float)(b0_3 & 63)+zp0;
        float w0_5 = sc0*(float)((b0_3>>6)|((b0_4&0xF)<<2))+zp0;
        float w0_6 = sc0*(float)((b0_4>>4)|((b0_5&3)<<4))+zp0;
        float w0_7 = sc0*(float)(b0_5>>2)+zp0;

        float w1_0 = sc1*(float)(b1_0 & 63)+zp1;
        float w1_1 = sc1*(float)((b1_0>>6)|((b1_1&0xF)<<2))+zp1;
        float w1_2 = sc1*(float)((b1_1>>4)|((b1_2&3)<<4))+zp1;
        float w1_3 = sc1*(float)(b1_2>>2)+zp1;
        float w1_4 = sc1*(float)(b1_3 & 63)+zp1;
        float w1_5 = sc1*(float)((b1_3>>6)|((b1_4&0xF)<<2))+zp1;
        float w1_6 = sc1*(float)((b1_4>>4)|((b1_5&3)<<4))+zp1;
        float w1_7 = sc1*(float)(b1_5>>2)+zp1;

        float w2_0 = sc2*(float)(b2_0 & 63)+zp2;
        float w2_1 = sc2*(float)((b2_0>>6)|((b2_1&0xF)<<2))+zp2;
        float w2_2 = sc2*(float)((b2_1>>4)|((b2_2&3)<<4))+zp2;
        float w2_3 = sc2*(float)(b2_2>>2)+zp2;
        float w2_4 = sc2*(float)(b2_3 & 63)+zp2;
        float w2_5 = sc2*(float)((b2_3>>6)|((b2_4&0xF)<<2))+zp2;
        float w2_6 = sc2*(float)((b2_4>>4)|((b2_5&3)<<4))+zp2;
        float w2_7 = sc2*(float)(b2_5>>2)+zp2;

        float w3_0 = sc3*(float)(b3_0 & 63)+zp3;
        float w3_1 = sc3*(float)((b3_0>>6)|((b3_1&0xF)<<2))+zp3;
        float w3_2 = sc3*(float)((b3_1>>4)|((b3_2&3)<<4))+zp3;
        float w3_3 = sc3*(float)(b3_2>>2)+zp3;
        float w3_4 = sc3*(float)(b3_3 & 63)+zp3;
        float w3_5 = sc3*(float)((b3_3>>6)|((b3_4&0xF)<<2))+zp3;
        float w3_6 = sc3*(float)((b3_4>>4)|((b3_5&3)<<4))+zp3;
        float w3_7 = sc3*(float)(b3_5>>2)+zp3;

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

    // Tail groups — distributed into acc[g & 3][b] to preserve the GEMV
    // reference's tail-group layout (the 5302926 bug fix).
    #define TAIL_BODY(acc_arr)                                                  \
        {                                                                        \
            const char* gp = row_ptr + g * 200;                                  \
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));    \
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
            const unsigned char* dp = (const unsigned char*)(gp + 8 + boff);     \
            unsigned char b0=dp[0], b1=dp[1], b2=dp[2], b3=dp[3], b4=dp[4], b5=dp[5]; \
            float w0 = sc*(float)(b0 & 63)+zp;                                   \
            float w1 = sc*(float)((b0>>6)|((b1&0xF)<<2))+zp;                    \
            float w2 = sc*(float)((b1>>4)|((b2&3)<<4))+zp;                      \
            float w3 = sc*(float)(b2>>2)+zp;                                     \
            float w4 = sc*(float)(b3 & 63)+zp;                                   \
            float w5 = sc*(float)((b3>>6)|((b4&0xF)<<2))+zp;                    \
            float w6 = sc*(float)((b4>>4)|((b5&3)<<4))+zp;                      \
            float w7 = sc*(float)(b5>>2)+zp;                                     \
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
        // Overwrite semantics (matches fused_qkvza_hfq6g256).
        if (tid == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
