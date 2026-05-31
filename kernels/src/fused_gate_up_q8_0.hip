// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt <kaden@hipfire.dev>
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused gate+up Q8_0 GEMV for SwiGLU FFN decode.
//
// Grid = [gate_m + up_m], block = 32 (one warp per output row).
// Each block routes to either the gate or up weight matrix. Both
// projections share the same input x — the second batch of blocks
// finds x still in L1 cache from the first batch.
//
// Q8_0 packing: 32 int8 weights per block, { scale(_Float16), 32 weights(int8) } = 34 B.
//
// Matches gemv_q8_0's 8-block unroll + launch_bounds for parity.
__launch_bounds__(32, 20)
extern "C" __global__ void fused_gate_up_q8_0(
    const unsigned char* __restrict__ a_gate,
    const unsigned char* __restrict__ a_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nblocks = K / 32;

    // Route to gate or up weight matrix
    const unsigned char* row_data;
    float* y_out;
    int out_row;
    if (row < gate_m) {
        row_data = a_gate + (size_t)row * nblocks * 34;
        y_out = y_gate;
        out_row = row;
    } else {
        row_data = a_up + (size_t)(row - gate_m) * nblocks * 34;
        y_out = y_up;
        out_row = row - gate_m;
    }

    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration
    const int outer_iters = nblocks / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks
    for (int bi = outer_iters * 8; bi < nblocks; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0) y_out[out_row] = sum;
}
