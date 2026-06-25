// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ fused Gate+Up DECODE (B=1), W4A16: gate_proj + up_proj GEMVs in ONE
// launch, one wave32 per output row, blockIdx.x demuxing gate vs up. Cuts the
// per-token dispatch count (2 gemv_oq4_grouped → 1) — part of matching mq4's
// fused decode (fused_gate_up_hfq4g256). 4-bit weight unpacked+sign-extended
// inline × f32 rotated activation. gfx1103 wave32, ZERO LDS.
//
//   W_* : [m_*, K/2] packed signed int4 ++ [m_*, K/256] f32 scales ; X : [K] f32
//   grid.x = gate_m + up_m, block = 32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_gate_up_oq4_gemv(
    const uint8_t* __restrict__ W_gate,
    const uint8_t* __restrict__ W_up,
    const float*   __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m, int K, int group
) {
    const uint8_t* W;
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
    const int gbytes = group / 2;
    const uint8_t* w_row = W + (long long)row * (K / 2);
    const float* ws_row = (const float*)(W + (long long)M * (K / 2)) + (long long)row * ng;

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
        Y[row] = facc;
    }
}
