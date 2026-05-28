// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 per-group O-LoRA batched GEMV — multi-row Q8_0 variant.
//
// Multi-row extension of wo_per_group_batched_q8_0.hip: one block
// computes R output rows, hoisting x loads across rows to improve
// L2 cache utilization. Grid = [ceil(M/R), B, G] (vs [M, B, G]).
//
// See gemv_hfq4g256_multirow.hip for the pattern that inspired this.
// On gfx1100 multi-row was neutral (already BW-saturated), but on
// gfx1151 (integrated GPU, shared memory) the L2 thrashing is more
// severe, so R=2 may help.

#include <hip/hip_runtime.h>

template<int R>
__device__ __forceinline__ void wo_per_group_batched_q8_0_multirow_body(
    const unsigned char* __restrict__ wo_a,
    const float* __restrict__ x_in,
    float* __restrict__ y_out,
    int G, int M, int K, int batch_size
) {
    const int r_base = blockIdx.x * R;
    const int b = blockIdx.y;
    const int g = blockIdx.z;
    if (r_base >= M || b >= batch_size || g >= G) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const size_t per_group_bytes = (size_t)M * blocks_per_row * 34;
    const size_t row_stride = (size_t)blocks_per_row * 34;

    const unsigned char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = r_base + r;
        int safe_row = (row < M) ? row : r_base;
        row_ptrs[r] = wo_a + (size_t)g * per_group_bytes + (size_t)safe_row * row_stride;
    }
    const float* x_row = x_in + ((size_t)b * G + g) * K;

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;

        float xv0 = x_row[(bi + 0) * 32 + tid];
        float xv1 = x_row[(bi + 1) * 32 + tid];
        float xv2 = x_row[(bi + 2) * 32 + tid];
        float xv3 = x_row[(bi + 3) * 32 + tid];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const unsigned char* b0 = row_ptrs[r] + (bi + 0) * 34;
            const unsigned char* b1 = row_ptrs[r] + (bi + 1) * 34;
            const unsigned char* b2 = row_ptrs[r] + (bi + 2) * 34;
            const unsigned char* b3 = row_ptrs[r] + (bi + 3) * 34;

            float d0 = (float)*((const _Float16*)b0);
            float d1 = (float)*((const _Float16*)b1);
            float d2 = (float)*((const _Float16*)b2);
            float d3 = (float)*((const _Float16*)b3);

            signed char q0 = (signed char)b0[2 + tid];
            signed char q1 = (signed char)b1[2 + tid];
            signed char q2 = (signed char)b2[2 + tid];
            signed char q3 = (signed char)b3[2 + tid];

            acc[r] += d0 * (float)q0 * xv0
                    + d1 * (float)q1 * xv1
                    + d2 * (float)q2 * xv2
                    + d3 * (float)q3 * xv3;
        }
    }

    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        float xv = x_row[bi * 32 + tid];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const unsigned char* bp = row_ptrs[r] + bi * 34;
            float d = (float)*((const _Float16*)bp);
            signed char qv = (signed char)bp[2 + tid];
            acc[r] += d * (float)qv * xv;
        }
    }

    for (int r = 0; r < R; r++) {
        float s = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = r_base + r;
        if (tid == 0 && row < M) {
            y_out[((size_t)b * G + g) * M + row] = s;
        }
    }
}

__launch_bounds__(32, 20)
extern "C" __global__ void wo_per_group_batched_q8_0_multirow_r2(
    const unsigned char* __restrict__ wo_a,
    const float* __restrict__ x_in,
    float* __restrict__ y_out,
    int G, int M, int K, int batch_size
) {
    wo_per_group_batched_q8_0_multirow_body<2>(wo_a, x_in, y_out, G, M, K, batch_size);
}

__launch_bounds__(32, 16)
extern "C" __global__ void wo_per_group_batched_q8_0_multirow_r4(
    const unsigned char* __restrict__ wo_a,
    const float* __restrict__ x_in,
    float* __restrict__ y_out,
    int G, int M, int K, int batch_size
) {
    wo_per_group_batched_q8_0_multirow_body<4>(wo_a, x_in, y_out, G, M, K, batch_size);
}
