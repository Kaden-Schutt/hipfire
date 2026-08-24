// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Fused Gate+Up MQ3-Lloyd: two GEMVs in one launch.
// Per-row math identical to gemv_mq3g256_lloyd.hip (LDS codebook path).
// gfx1100/1101/1102 use the K4+LDS variant in
// fused_gate_up_mq3g256_lloyd.gfx1100.hip via the kernels.rs arch selector.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 16)
extern "C" __global__ void fused_gate_up_mq3g256_lloyd(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)local_row * groups_per_row * 112;
    const int boff = tid * 3;

    __shared__ float cbt[8];

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 112;

        if (tid < 8) {
            cbt[tid] = __half2float(((const __half*)gptr)[tid]);
        }
        __syncthreads();

        const unsigned char* data = (const unsigned char*)(gptr + 16);

        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        unsigned int q0 = pk & 7u;
        unsigned int q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u;
        unsigned int q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u;
        unsigned int q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u;
        unsigned int q7 = (pk >> 21) & 7u;

        int base_idx = g * 256 + tid * 8;

        acc += cbt[q0] * x[base_idx]
             + cbt[q1] * x[base_idx + 1]
             + cbt[q2] * x[base_idx + 2]
             + cbt[q3] * x[base_idx + 3]
             + cbt[q4] * x[base_idx + 4]
             + cbt[q5] * x[base_idx + 5]
             + cbt[q6] * x[base_idx + 6]
             + cbt[q7] * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[local_row] = acc;
}
