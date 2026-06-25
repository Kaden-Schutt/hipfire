// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant OQ4+ DECODE GEMV (batch=1): y[m] = Σ_g sw[m,g]·Σ_{k∈g} qw·x[k].
// One wave32 per output row, ZERO LDS, single wave-shuffle reduction.
//
// DECODE numerics note: at B=1 there is no WMMA, so we do NOT quantize the
// activation to int4 (that exists only to feed the iu4 WMMA tiles in the
// batched-prefill path). Instead we unpack the packed signed-int4 weight inline
// and multiply by the FULL-PRECISION f32 activation — i.e. W4A16 at decode,
// exactly like mq4's gemv_hfq4g256 and the OQ8+ decode GEMV. This (a) reads the
// 4-bit-RESIDENT weight (half the bytes of OQ8+'s int8-expanded weight → mq4
// decode bandwidth) while (b) keeping W4A16 decode quality (f32 acts), which
// sidesteps the int4-activation attractor that plain W4A4 decode hit. The AWQ
// math (W·s)·(x/s)=W·x still holds (awq divide folded into x_rot upstream, s in
// the weight). group MUST be 256 (gbytes/2 = 128 = 32 lanes × 4 B).
//
//   W  : [M, K/2] packed signed int4 (byte = k_even | k_odd<<4, two's-comp)
//   Ws : [M, K/256] f32 per-group weight scales
//   X  : [K] f32 activation (batch=1, rotated)
//   Y  : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq4_grouped(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const float*   __restrict__ X,
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
        const int xoff = g * group + lane * 8;  // 8 f32 activations per lane
        uint32_t wv = *(const uint32_t*)(w_row + boff);
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int qw = (int)((wv >> (4 * n)) & 0xf);
            qw = (qw << 28) >> 28; // sign-extend int4
            gsum += (float)qw * X[xoff + n];
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
