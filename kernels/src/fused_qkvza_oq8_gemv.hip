// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ+ (W8A8) fused QKVZA DECODE (B=1): the four in-proj GEMVs (qkv/z/beta/alpha)
// in ONE launch, one wave32 per OUTPUT ROW, blockIdx.x demuxing which projection
// the row belongs to. No WMMA N-tile waste (unlike fused_qkvza_oq8_wmma, which is
// 16× wasteful at B=1), and one launch instead of four gemv_oq8_grouped launches —
// matches mq4's fused_qkv decode shape. Shared int8 activation (quantized once by
// the caller). gfx1103 wave32, ZERO LDS.
//
//   W_* : [m_*, K] int8 ++ [m_*, K/256] f32 scales (combined buffer each)
//   X   : [K] int8 (shared) ; Xs : [K/256] f32 (shared) ; Y_* : [m_*] f32
//   grid.x = qkv_m + z_m + beta_m + alpha_m, block = 32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_qkvza_oq8_gemv(
    const int8_t* __restrict__ W_qkv,
    const int8_t* __restrict__ W_z,
    const int8_t* __restrict__ W_beta,
    const int8_t* __restrict__ W_alpha,
    const int8_t* __restrict__ X,
    const float*  __restrict__ Xs,
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
        const int32_t x0 = *(const int32_t*)(X + boff);
        const int32_t x1 = *(const int32_t*)(X + boff + 4);
        int isum = 0;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int qw = (int)((w0 >> (8 * n)) & 0xff); qw = (qw << 24) >> 24;
            int qx = (int)((x0 >> (8 * n)) & 0xff); qx = (qx << 24) >> 24;
            isum += qw * qx;
            int qw1 = (int)((w1 >> (8 * n)) & 0xff); qw1 = (qw1 << 24) >> 24;
            int qx1 = (int)((x1 >> (8 * n)) & 0xff); qx1 = (qx1 << 24) >> 24;
            isum += qw1 * qx1;
        }
        for (int off = 16; off > 0; off >>= 1) {
            isum += __shfl_down(isum, off);
        }
        isum = __shfl(isum, 0);
        facc += (float)isum * ws_row[g] * Xs[g];
    }
    if (lane == 0) {
        Y[row] = facc;
    }
}
