// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W8A8 DECODE GEMV (batch=1): y[m] = Σ_g sw[m,g]·sx[g]·Σ_{k∈g} qw·qx.
// One wave32 per output row, ZERO LDS, wave-shuffle reduction. The int8
// generalization of gemv_oq4_grouped — unlike the WMMA grouped GEMM
// (gemm_oq8_grouped_wmma), this wastes no 16×16 N-tile lanes at B=1, so the
// kernel is weight-bandwidth-bound instead of WMMA-occupancy-bound. group MUST
// be 256 (group/32 = 8 int8 per lane = two aligned int32 loads).
//
//   W  : [M, K] signed int8 weights              row stride = K
//   Ws : [M, K/256] f32 per-group weight scales
//   X  : [K] signed int8 activation (batch=1)
//   Xs : [K/256] f32 per-group activation scales
//   Y  : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq8_grouped(
    const int8_t*  __restrict__ W,
    const float*   __restrict__ Ws,
    const int8_t*  __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x; // 0..31
    const int ng = K / group;
    const int8_t* w_row = W + (long long)row * K; // K int8 bytes per row
    const float* ws_row = Ws + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * group + lane * 8; // 8 int8 = two int32 per lane
        const int32_t w0 = *(const int32_t*)(w_row + boff);
        const int32_t w1 = *(const int32_t*)(w_row + boff + 4);
        const int32_t x0 = *(const int32_t*)(X + boff);
        const int32_t x1 = *(const int32_t*)(X + boff + 4);
        int isum = 0;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int qw = (int)((w0 >> (8 * n)) & 0xff); qw = (qw << 24) >> 24; // sext int8
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
