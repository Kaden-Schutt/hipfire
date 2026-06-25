// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ+ (W4A8) fused QKVZA DECODE (B=1): the four in-proj GEMVs (qkv/z/beta/alpha)
// in ONE launch, one wave32 per OUTPUT ROW, blockIdx.x demuxing which projection
// the row belongs to. Matches mq4's fused_qkvza_hfq4g256 decode shape exactly:
// one launch instead of four, and the FULL-PRECISION f32 activation is consumed
// directly (int8 weight dequantized inline) — so there is NO separate
// quantize_act_oq8 launch. See gemv_oq8_grouped.hip for the decode-numerics
// rationale (W4A16 decode is more accurate than W8A8 and launch-cheaper).
// gfx1103 wave32, ZERO LDS.
//
//   W_* : [m_*, K] int8 ++ [m_*, K/256] f32 scales (combined buffer each)
//   X   : [K] f32 (shared, rotated) ; Y_* : [m_*] f32
//   grid.x = qkv_m + z_m + beta_m + alpha_m, block = 32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_qkvza_oq8_gemv(
    const int8_t* __restrict__ W_qkv,
    const int8_t* __restrict__ W_z,
    const int8_t* __restrict__ W_beta,
    const int8_t* __restrict__ W_alpha,
    const float*  __restrict__ X,
    float*        __restrict__ Y_qkv,
    float*        __restrict__ Y_z,
    float*        __restrict__ Y_beta,
    float*        __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int group
) {
    // Demux blockIdx.x → (projection, row) over concatenated output rows.
    const int8_t* W;
    float* Y;
    int M, row;
    int gx = blockIdx.x;
    if (gx < qkv_m) {
        W = W_qkv; Y = Y_qkv; M = qkv_m; row = gx;
    } else if (gx < qkv_m + z_m) {
        W = W_z; Y = Y_z; M = z_m; row = gx - qkv_m;
    } else if (gx < qkv_m + z_m + beta_m) {
        W = W_beta; Y = Y_beta; M = beta_m; row = gx - qkv_m - z_m;
    } else if (gx < qkv_m + z_m + beta_m + alpha_m) {
        W = W_alpha; Y = Y_alpha; M = alpha_m; row = gx - qkv_m - z_m - beta_m;
    } else {
        return;
    }
    const int lane = threadIdx.x; // 0..31
    const int ng = K / group;
    const int8_t* w_row = W + (long long)row * K;
    const float* ws_row = (const float*)(W + (long long)M * K) + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * group + lane * 8; // 8 int8/lane (two int32)
        const int32_t w0 = *(const int32_t*)(w_row + boff);
        const int32_t w1 = *(const int32_t*)(w_row + boff + 4);
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int qw0 = (int)((w0 >> (8 * n)) & 0xff); qw0 = (qw0 << 24) >> 24;
            int qw1 = (int)((w1 >> (8 * n)) & 0xff); qw1 = (qw1 << 24) >> 24;
            gsum += (float)qw0 * X[boff + n];
            gsum += (float)qw1 * X[boff + 4 + n];
        }
        facc += gsum * ws_row[g];
    }
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] = facc;
    }
}
