// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ W4A16 DECODE GEMV reading an INTERLEAVED group layout (vs the split
// [nibbles | scales] of gemv_oq4_grouped). Per 256-group per row the record is
// CONTIGUOUS: [f32 scale (4B)][128 nibble bytes] = 132 B/group. The scale and
// its nibbles are co-located → ONE memory stream per group (mq4-style), instead
// of the split layout's two streams (nibbles + a distant Ws region). The wall-
// clock ablation showed the OQ4+ decode GEMVs are 42% of decode at ~59% of peak
// bandwidth, so the access pattern matters. This kernel is the candidate decode
// kernel for an interleaved per-arch loader repack. gfx1103 wave32, ZERO LDS.
//
//   W : [M, ng*132] interleaved (per group: [f32 scale][128 nibbles]) ;
//   X : [K] f32 (rotated) ; Y : [M] f32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq4_interleaved(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int ng = K / group;
    const int gstride = 4 + group / 2;            // 132 bytes per group
    const uint8_t* row_base = W + (long long)row * ng * gstride;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const uint8_t* gp = row_base + (long long)g * gstride;
        const float scale = *(const float*)gp;     // contiguous with the nibbles
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
        Y[row] = facc;
    }
}
