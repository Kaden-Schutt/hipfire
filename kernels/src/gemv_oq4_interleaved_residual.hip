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
    // 4-group unroll (MLP / software-prefetch; mirrors mq4's gp0..gp3).
    #define OQ4IL_DOG(wv, xoff, sc) { \
        float gsum = 0.0f; \
        _Pragma("unroll") \
        for (int n = 0; n < 8; ++n) { int qw = (int)(((wv) >> (4 * n)) & 0xf); qw = (qw << 28) >> 28; gsum += (float)qw * X[(xoff) + n]; } \
        facc += gsum * (sc); }
    int g = 0;
    for (; g + 4 <= ng; g += 4) {
        const uint8_t* gp0 = row_base + (long long)(g + 0) * gstride;
        const uint8_t* gp1 = row_base + (long long)(g + 1) * gstride;
        const uint8_t* gp2 = row_base + (long long)(g + 2) * gstride;
        const uint8_t* gp3 = row_base + (long long)(g + 3) * gstride;
        const float s0 = *(const float*)gp0, s1 = *(const float*)gp1;
        const float s2 = *(const float*)gp2, s3 = *(const float*)gp3;
        const uint32_t w0 = *(const uint32_t*)(gp0 + 4 + lane * 4);
        const uint32_t w1 = *(const uint32_t*)(gp1 + 4 + lane * 4);
        const uint32_t w2 = *(const uint32_t*)(gp2 + 4 + lane * 4);
        const uint32_t w3 = *(const uint32_t*)(gp3 + 4 + lane * 4);
        OQ4IL_DOG(w0, (g + 0) * group + lane * 8, s0);
        OQ4IL_DOG(w1, (g + 1) * group + lane * 8, s1);
        OQ4IL_DOG(w2, (g + 2) * group + lane * 8, s2);
        OQ4IL_DOG(w3, (g + 3) * group + lane * 8, s3);
    }
    for (; g < ng; ++g) {
        const uint8_t* gp = row_base + (long long)g * gstride;
        const float scale = *(const float*)gp;
        const uint32_t wv = *(const uint32_t*)(gp + 4 + lane * 4);
        OQ4IL_DOG(wv, g * group + lane * 8, scale);
    }
    #undef OQ4IL_DOG
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] += facc; // fused residual
    }
}
