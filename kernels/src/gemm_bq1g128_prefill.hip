// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Tiled prefill GEMM for BQ1-G128 (PrismML Q1_0 binary), operating DIRECTLY
// on the packed 18-byte blocks. Binary sibling of gemm_tq2g128_prefill.hip --
// see that file for the three designs that lost to the scalar loop and why
// register blocking is what finally beats it.
//
// Layout difference: a 128-weight block carries 16 code bytes (8 weights per
// byte) rather than the ternary 32, so a BK=32 slice spans 4 code bytes per
// row instead of 8, and each thread decodes 8 weights per byte.
//
// Grid: [ceil(M/BM), ceil(N/BN)]. Block: 256 threads.
// Thread (tr, tc) = (tid / 16, tid % 16) owns rows tr*4.. and tokens tc*4...

#ifndef HIPFIRE_BQ1G128_PREFILL_KERNEL
#define HIPFIRE_BQ1G128_PREFILL_KERNEL gemm_bq1g128_prefill
#endif

#define BQ1P_BM 64
#define BQ1P_BN 64
#define BQ1P_BK 32
#define BQ1P_THREADS 256
// Stride 68: breaks the LDS bank pattern (68 % 32 == 4) and stays 4-float
// aligned so the inner loop can issue float4 loads.
#define BQ1P_LDS_STRIDE (BQ1P_BM + 4)

__launch_bounds__(BQ1P_THREADS, 2)
extern "C" __global__ void HIPFIRE_BQ1G128_PREFILL_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [N x K] row-major
    float* __restrict__ y,         // [N x M] row-major
    int M, int K, int N
) {
    const int row_base = blockIdx.x * BQ1P_BM;
    const int n_base = blockIdx.y * BQ1P_BN;
    const int tid = threadIdx.x;
    const int tr = tid / 16;
    const int tc = tid % 16;

    const int groups_per_row = K / 128;
    const long long row_bytes = (long long)groups_per_row * 18;

    __shared__ float w_lds[BQ1P_BK * BQ1P_LDS_STRIDE];
    __shared__ float x_lds[BQ1P_BK * BQ1P_LDS_STRIDE];

    float acc[4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BQ1P_BK) {
        const int g = k0 / 128;
        const int c = (k0 % 128) / 32;

        // ── stage weights: BM rows x 4 code bytes (= 32 weights) ──────────
        #pragma unroll
        for (int i = tid; i < BQ1P_BM * 4; i += BQ1P_THREADS) {
            const int rr = i >> 2;
            const int bb = i & 3;
            const int row = row_base + rr;
            float d = 0.0f;
            unsigned int code = 0u;
            if (row < M) {
                const char* gp = A + (long long)row * row_bytes + (long long)g * 18;
                d = (float)(*(const _Float16*)gp);
                code = *((const unsigned char*)(gp + 2 + c * 4 + bb));
            }
            const int kk = bb * 8;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                // bit set -> +d, clear -> -d. Rows past M carry d == 0, so
                // both branches contribute nothing.
                w_lds[(kk + e) * BQ1P_LDS_STRIDE + rr] =
                    ((code >> e) & 1u) ? d : -d;
            }
        }

        // ── stage x: BN tokens x BK floats, contiguous in k ───────────────
        #pragma unroll
        for (int i = tid; i < BQ1P_BN * BQ1P_BK; i += BQ1P_THREADS) {
            const int tt = i / BQ1P_BK;
            const int kk = i % BQ1P_BK;
            const int n = n_base + tt;
            x_lds[kk * BQ1P_LDS_STRIDE + tt] =
                (n < N) ? x[(long long)n * K + k0 + kk] : 0.0f;
        }
        __syncthreads();

        #pragma unroll 4
        for (int kk = 0; kk < BQ1P_BK; kk++) {
            const float4 wv = *(const float4*)&w_lds[kk * BQ1P_LDS_STRIDE + tr * 4];
            const float4 xv = *(const float4*)&x_lds[kk * BQ1P_LDS_STRIDE + tc * 4];
            const float wv0 = wv.x, wv1 = wv.y, wv2 = wv.z, wv3 = wv.w;
            const float xv0 = xv.x, xv1 = xv.y, xv2 = xv.z, xv3 = xv.w;
            acc[0][0] += wv0 * xv0; acc[0][1] += wv0 * xv1;
            acc[0][2] += wv0 * xv2; acc[0][3] += wv0 * xv3;
            acc[1][0] += wv1 * xv0; acc[1][1] += wv1 * xv1;
            acc[1][2] += wv1 * xv2; acc[1][3] += wv1 * xv3;
            acc[2][0] += wv2 * xv0; acc[2][1] += wv2 * xv1;
            acc[2][2] += wv2 * xv2; acc[2][3] += wv2 * xv3;
            acc[3][0] += wv3 * xv0; acc[3][1] += wv3 * xv1;
            acc[3][2] += wv3 * xv2; acc[3][3] += wv3 * xv3;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int a = 0; a < 4; a++) {
        const int row = row_base + tr * 4 + a;
        if (row >= M) continue;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            const int n = n_base + tc * 4 + b;
            if (n < N) y[(long long)n * M + row] = acc[a][b];
        }
    }
}
