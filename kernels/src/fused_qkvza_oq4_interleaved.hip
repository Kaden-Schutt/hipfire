// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ fused QKVZA DECODE (B=1), INTERLEAVED layout: 4 in-proj GEMVs in ONE
// launch, one wave32 per output row, blockIdx demux. Per-group record is
// contiguous [f32 scale][128 nibbles] (132 B) → one coalesced memory stream.
// W4A16 (f32 activation). gfx1103 wave32, ZERO LDS.
//
//   W_* : [m_*, ng*132] interleaved ; X : [K] f32 ; grid.x = qkv_m+z_m+beta_m+alpha_m.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_qkvza_oq4_interleaved(
    const uint8_t* __restrict__ W_qkv,
    const uint8_t* __restrict__ W_z,
    const uint8_t* __restrict__ W_beta,
    const uint8_t* __restrict__ W_alpha,
    const float*   __restrict__ X,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int group
) {
    const uint8_t* W;
    float* Y;
    int row;
    int gx = blockIdx.x;
    if (gx < qkv_m) { W = W_qkv; Y = Y_qkv; row = gx; }
    else if (gx < qkv_m + z_m) { W = W_z; Y = Y_z; row = gx - qkv_m; }
    else if (gx < qkv_m + z_m + beta_m) { W = W_beta; Y = Y_beta; row = gx - qkv_m - z_m; }
    else if (gx < qkv_m + z_m + beta_m + alpha_m) { W = W_alpha; Y = Y_alpha; row = gx - qkv_m - z_m - beta_m; }
    else { return; }

    const int lane = threadIdx.x;
    const int ng = K / group;
    const int gstride = 4 + group / 2;
    const uint8_t* row_base = W + (long long)row * ng * gstride;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const uint8_t* gp = row_base + (long long)g * gstride;
        const float scale = *(const float*)gp;
        const uint32_t wv = *(const uint32_t*)(gp + 4 + lane * 4);
        const int xoff = g * group + lane * 8;
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int qw = (int)((wv >> (4 * n)) & 0xf);
            qw = (qw << 28) >> 28;
            gsum += (float)qw * X[xoff + n];
        }
        facc += gsum * scale;
    }
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
