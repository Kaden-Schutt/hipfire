// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ W4A16 decode GEMV, INTERLEAVED layout + FUSED residual (Y += W·x).
// Per 256-group record is contiguous [f32 scale][128 nibbles] (132 B); one
// coalesced stream (vs the split layout's two). See gemv_oq4_interleaved. wave32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq4_interleaved_residual(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
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
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] += facc; // fused residual
    }
}
