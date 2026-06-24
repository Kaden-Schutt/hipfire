// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ fused Gate+Up DECODE (B=1), INTERLEAVED layout: gate+up GEMVs in ONE
// launch, blockIdx demux, contiguous [f32 scale][128 nibbles]/group. W4A16. wave32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_gate_up_oq4_interleaved(
    const uint8_t* __restrict__ W_gate,
    const uint8_t* __restrict__ W_up,
    const float*   __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m, int K, int group
) {
    const uint8_t* W;
    float* Y;
    int row;
    int gx = blockIdx.x;
    if (gx < gate_m) { W = W_gate; Y = Y_gate; row = gx; }
    else if (gx < gate_m + up_m) { W = W_up; Y = Y_up; row = gx - gate_m; }
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
