// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Short-batch QKVZA verifier kernel derived from the certified gfx12 AR
// fused_qkvza_hfq4g256 schedule.
//
// The verifier only needs N=2..4. The ordinary WMMA path forms a 16-column
// tile and therefore executes mostly padding at these widths. This kernel
// assigns one wave32 to one output row, exactly like golden AR, and reuses
// every decoded HFQ4 weight dword across all live verifier tokens.
//
// Invariants retained from golden AR:
//   - one fused qkv/z/beta/alpha routing domain;
//   - fixed K=2048 and eight 256-element HFQ4 groups;
//   - temporal gfx12 raw-buffer weight loads from
//     gfx12_weight_cache_policy.inc;
//   - float4 activation loads, four independent accumulator chains, and the
//     pairwise final combine;
//   - the same wave32 shuffle reduction order.
//
// Grid: [qkv_m + z_m + beta_m + alpha_m, 1, 1]
// Block: [32, 1, 1]

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_qkvza_hfq4g256_golden_batch4_gfx12(
    const char* __restrict__ A_qkv,
    const char* __restrict__ A_z,
    const char* __restrict__ A_beta,
    const char* __restrict__ A_alpha,
    const float* __restrict__ X,
    float* __restrict__ Y_qkv,
    float* __restrict__ Y_z,
    float* __restrict__ Y_beta,
    float* __restrict__ Y_alpha,
    int qkv_m,
    int z_m,
    int beta_m,
    int alpha_m,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m || K != 2048 || N < 1 || N > 4) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < qkv_m) {
        A = A_qkv;
        Y = Y_qkv;
        row = gid;
        out_stride = qkv_m;
    } else if (gid < qkv_m + z_m) {
        A = A_z;
        Y = Y_z;
        row = gid - qkv_m;
        out_stride = z_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;
        Y = Y_beta;
        row = gid - (qkv_m + z_m);
        out_stride = beta_m;
    } else {
        A = A_alpha;
        Y = Y_alpha;
        row = gid - (qkv_m + z_m + beta_m);
        out_stride = alpha_m;
    }

    constexpr int groups_per_row = 8;
    constexpr int row_stride = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_stride;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif

    float acc[4][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };

    #define GOLDEN_DOG(pk, sc, zp, a, xv0, xv1) do { \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * (xv0).x \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (xv0).y \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (xv0).z \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (xv0).w \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (xv1).x \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (xv1).y \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (xv1).z \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (xv1).w; \
    } while (0)

    #pragma unroll
    for (int q = 0; q < 2; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (g + 0) * 136;
        const char* gp1 = row_ptr + (g + 1) * 136;
        const char* gp2 = row_ptr + (g + 2) * 136;
        const char* gp3 = row_ptr + (g + 3) * 136;

        if (q == 0) {
            __builtin_amdgcn_s_prefetch_data((const void*)(gp0 + 272), 0);
        }

        const float sc0 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136));
        const float zp0 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4));
        const float sc1 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136));
        const float zp1 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4));
        const float sc2 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136));
        const float zp2 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4));
        const float sc3 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136));
        const float zp3 = __builtin_bit_cast(
            float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4));

        const unsigned int pk0 =
            HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        const unsigned int pk1 =
            HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        const unsigned int pk2 =
            HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        const unsigned int pk3 =
            HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);

        #pragma unroll
        for (int n = 0; n < 4; n++) {
            if (n < N) {
                const float* x = X + (long long)n * 2048;
                const int base = g * 256 + tid * 8;
                const float4 x0a = *reinterpret_cast<const float4*>(x + base);
                const float4 x0b = *reinterpret_cast<const float4*>(x + base + 4);
                GOLDEN_DOG(pk0, sc0, zp0, acc[n][0], x0a, x0b);

                const float4 x1a = *reinterpret_cast<const float4*>(x + base + 256);
                const float4 x1b = *reinterpret_cast<const float4*>(x + base + 260);
                GOLDEN_DOG(pk1, sc1, zp1, acc[n][1], x1a, x1b);

                const float4 x2a = *reinterpret_cast<const float4*>(x + base + 512);
                const float4 x2b = *reinterpret_cast<const float4*>(x + base + 516);
                GOLDEN_DOG(pk2, sc2, zp2, acc[n][2], x2a, x2b);

                const float4 x3a = *reinterpret_cast<const float4*>(x + base + 768);
                const float4 x3b = *reinterpret_cast<const float4*>(x + base + 772);
                GOLDEN_DOG(pk3, sc3, zp3, acc[n][3], x3a, x3b);
            }
        }
    }
    #undef GOLDEN_DOG

    #pragma unroll
    for (int n = 0; n < 4; n++) {
        if (n < N) {
            float sum = (acc[n][0] + acc[n][1]) + (acc[n][2] + acc[n][3]);
            sum += __shfl_down(sum, 16);
            sum += __shfl_down(sum, 8);
            sum += __shfl_down(sum, 4);
            sum += __shfl_down(sum, 2);
            sum += __shfl_down(sum, 1);
            if (tid == 0) {
                Y[(long long)n * out_stride + row] = sum;
            }
        }
    }
}
