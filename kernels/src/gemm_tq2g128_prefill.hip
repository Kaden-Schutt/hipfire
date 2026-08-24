// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Tiled prefill GEMM for TQ2-G128 (PrismML Q2_0 ternary), operating DIRECTLY
// on the packed 34-byte blocks: y[n][row] = A[row] . x[n].
//
// The scalar GEMV re-reads AND re-decodes every weight row once per token, so
// a prefill chunk of N tokens costs N weight passes. At the Bonsai FFN shape
// that saturates memory (~210 GB/s measured) and pins prefill to decode speed.
//
// Three designs were measured before this one; all lost to the scalar loop:
//   1. dequant -> F16 -> gemm_f16_x_f16_wmma. Writes 8x the packed bytes (a
//      22 MB ternary weight becomes 178 MB); the F16 GEMM alone was ~2x
//      slower than scalar at N=512 even discounting the 6.9 ms dequant.
//   2. Tile over N only (one wave per row). Each of the M row-blocks re-reads
//      the whole x tile; at TILE_N=32 that is 655 KB, far out of cache. 0.7x.
//   3. 2D tile with one accumulator per thread. LDS-bandwidth bound: 2 LDS
//      reads per FMA, and a single accumulator serialises the FMA chain. 0.6x.
//
// This version fixes (3) with REGISTER BLOCKING. Each thread owns a 4x4 output
// tile, so the 4 w values and 4 x values it loads from LDS feed 16 FMAs --
// 0.5 LDS reads per FMA instead of 2 -- and the 16 independent accumulators
// give the scheduler the ILP a single chain cannot.
//
// Block tile: BM x BN outputs over a BK slice of K, staged k-major in LDS so
// the staging reads from x are contiguous.
// Grid: [ceil(M/BM), ceil(N/BN)]. Block: 256 threads.
// Thread (tr, tc) = (tid / 16, tid % 16) owns rows tr*4.. and tokens tc*4...

#ifndef HIPFIRE_TQ2G128_PREFILL_KERNEL
#define HIPFIRE_TQ2G128_PREFILL_KERNEL gemm_tq2g128_prefill
#endif

#define TQ2P_BM 64
#define TQ2P_BN 64
#define TQ2P_BK 32
#define TQ2P_THREADS 256
// Stride 68, not 65: it must break the bank pattern (68 % 32 == 4, so
// consecutive k rows start 4 banks apart) AND stay 4-float aligned so the
// inner loop can issue float4 LDS loads instead of 4 scalars. 65 satisfies
// only the first.
#define TQ2P_LDS_STRIDE (TQ2P_BM + 4)

__launch_bounds__(TQ2P_THREADS, 2)
extern "C" __global__ void HIPFIRE_TQ2G128_PREFILL_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [N x K] row-major
    float* __restrict__ y,         // [N x M] row-major
    int M, int K, int N
) {
    const int row_base = blockIdx.x * TQ2P_BM;
    const int n_base = blockIdx.y * TQ2P_BN;
    const int tid = threadIdx.x;
    const int tr = tid / 16;   // 0..15 -> rows tr*4 .. tr*4+3
    const int tc = tid % 16;   // 0..15 -> tokens tc*4 .. tc*4+3

    const int groups_per_row = K / 128;
    const long long row_bytes = (long long)groups_per_row * 34;

    __shared__ float w_lds[TQ2P_BK * TQ2P_LDS_STRIDE];
    __shared__ float x_lds[TQ2P_BK * TQ2P_LDS_STRIDE];

    float acc[4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TQ2P_BK) {
        const int g = k0 / 128;          // which 128-weight group
        const int c = (k0 % 128) / 32;   // which 32-wide quarter of it

        // ── stage weights: BM rows x 8 code bytes (= 32 weights) ──────────
        #pragma unroll
        for (int i = tid; i < TQ2P_BM * 8; i += TQ2P_THREADS) {
            const int rr = i >> 3;       // 0..BM-1
            const int bb = i & 7;        // 0..7 code bytes
            const int row = row_base + rr;
            float d = 0.0f;
            unsigned int code = 0x55u;   // decodes to all-zero weights
            if (row < M) {
                const char* gp = A + (long long)row * row_bytes + (long long)g * 34;
                d = (float)(*(const _Float16*)gp);
                code = *((const unsigned char*)(gp + 2 + c * 8 + bb));
            }
            const int kk = bb * 4;
            w_lds[(kk + 0) * TQ2P_LDS_STRIDE + rr] = d * (float)((int)( code       & 0x3) - 1);
            w_lds[(kk + 1) * TQ2P_LDS_STRIDE + rr] = d * (float)((int)((code >> 2) & 0x3) - 1);
            w_lds[(kk + 2) * TQ2P_LDS_STRIDE + rr] = d * (float)((int)((code >> 4) & 0x3) - 1);
            w_lds[(kk + 3) * TQ2P_LDS_STRIDE + rr] = d * (float)((int)((code >> 6) & 0x3) - 1);
        }

        // ── stage x: BN tokens x BK floats, contiguous in k ───────────────
        #pragma unroll
        for (int i = tid; i < TQ2P_BN * TQ2P_BK; i += TQ2P_THREADS) {
            const int tt = i / TQ2P_BK;
            const int kk = i % TQ2P_BK;
            const int n = n_base + tt;
            x_lds[kk * TQ2P_LDS_STRIDE + tt] =
                (n < N) ? x[(long long)n * K + k0 + kk] : 0.0f;
        }
        __syncthreads();

        #pragma unroll 4
        for (int kk = 0; kk < TQ2P_BK; kk++) {
            const float4 wv = *(const float4*)&w_lds[kk * TQ2P_LDS_STRIDE + tr * 4];
            const float4 xv = *(const float4*)&x_lds[kk * TQ2P_LDS_STRIDE + tc * 4];
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
