// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W4A4 DECODE GEMV (batch=1): y[m] = Σ_g sw[m,g]·sx[g]·Σ_{k∈g} qw·qx.
// One wave32 per output row, ZERO LDS, wave-shuffle reduction. Unlike the WMMA
// grouped GEMM (gemm_oq4_grouped_wmma), this wastes no 16×16 N-tile lanes at
// B=1 — each lane reads exactly one uint32 (8 signed int4 nibbles) of weight and
// activation per group, so the kernel is weight-bandwidth-bound instead of
// WMMA-occupancy-bound. group MUST be 256 (gbytes/2 = 128 = 32 lanes × 4 B).
//
//   W  : [M, K/2] packed signed int4 (byte = k_even | k_odd<<4, two's-comp)
//   Ws : [M, K/256] f32 per-group weight scales
//   X  : [K/2] packed signed int4 activation (batch=1)
//   Xs : [K/256] f32 per-group activation scales
//   Y  : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq4_grouped(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x; // 0..31
    const int ng = K / group;
    const int gbytes = group / 2; // 128 for group=256
    const uint8_t* w_row = W + (long long)row * (K / 2);
    const float* ws_row = Ws + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * gbytes + lane * 4; // 4 bytes = 8 nibbles per lane
        uint32_t wv = *(const uint32_t*)(w_row + boff);
        uint32_t xv = *(const uint32_t*)(X + boff);
        int isum = 0;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int qw = (int)((wv >> (4 * n)) & 0xf);
            qw = (qw << 28) >> 28; // sign-extend low nibble
            int qx = (int)((xv >> (4 * n)) & 0xf);
            qx = (qx << 28) >> 28;
            isum += qw * qx;
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
