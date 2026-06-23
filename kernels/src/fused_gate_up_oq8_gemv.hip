// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ+ (W8A8) fused Gate+Up DECODE (B=1): gate_proj + up_proj GEMVs in ONE launch,
// one wave32 per output row, blockIdx.x demuxing gate vs up. No WMMA N-tile waste;
// one launch instead of two gemv_oq8_grouped. Shared int8 activation. wave32, ZERO LDS.
//
//   W_* : [m_*, K] int8 ++ [m_*, K/256] f32 scales ; X : [K] int8 ; Xs : [K/256] f32
//   grid.x = gate_m + up_m, block = 32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_gate_up_oq8_gemv(
    const int8_t* __restrict__ W_gate,
    const int8_t* __restrict__ W_up,
    const int8_t* __restrict__ X,
    const float*  __restrict__ Xs,
    float*        __restrict__ Y_gate,
    float*        __restrict__ Y_up,
    int gate_m, int up_m, int K, int group
) {
    const int8_t* W;
    float* Y;
    int M, row;
    int gx = blockIdx.x;
    if (gx < gate_m) {
        W = W_gate; Y = Y_gate; M = gate_m; row = gx;
    } else if (gx < gate_m + up_m) {
        W = W_up; Y = Y_up; M = up_m; row = gx - gate_m;
    } else {
        return;
    }
    const int lane = threadIdx.x;
    const int ng = K / group;
    const int8_t* w_row = W + (long long)row * K;
    const float* ws_row = (const float*)(W + (long long)M * K) + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * group + lane * 8;
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
