// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ W4A16 DECODE GEMV with FUSED residual add (B=1): Y[m] += Σ_g sw·Σ qw·x.
// Identical to gemv_oq4_grouped but the epilogue does `Y[row] += facc` (read-
// modify-write; one wave owns the row, no race) instead of `Y[row] = facc`.
// This eliminates the separate add_inplace_f32 pass that the unfused wo/down
// path paid — mirroring mq4's gemv_hfq4g256_residual (the standout decode-gap
// item: ~261ms of add_inplace over 128 tokens on 0.8B). gfx1103 wave32, ZERO LDS.
//
//   W  : [M, K/2] packed signed int4 ++ [M, K/256] f32 scales (Ws view) ;
//   X  : [K] f32 (rotated) ; Y : [M] f32 — the residual stream, updated IN PLACE.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq4_grouped_residual(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int ng = K / group;
    const int gbytes = group / 2;
    const uint8_t* w_row = W + (long long)row * (K / 2);
    const float* ws_row = Ws + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * gbytes + lane * 4;
        const int xoff = g * group + lane * 8;
        uint32_t wv = *(const uint32_t*)(w_row + boff);
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int qw = (int)((wv >> (4 * n)) & 0xf);
            qw = (qw << 28) >> 28;
            gsum += (float)qw * X[xoff + n];
        }
        facc += gsum * ws_row[g];
    }
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] += facc; // fused residual add
    }
}
