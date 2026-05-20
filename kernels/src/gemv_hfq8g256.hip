// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq8g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 264;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 264;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread = 8 bytes
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 8;

        acc += (scale * (float)data[byte_off]     + zero) * x[base_idx]
             + (scale * (float)data[byte_off + 1] + zero) * x[base_idx + 1]
             + (scale * (float)data[byte_off + 2] + zero) * x[base_idx + 2]
             + (scale * (float)data[byte_off + 3] + zero) * x[base_idx + 3]
             + (scale * (float)data[byte_off + 4] + zero) * x[base_idx + 4]
             + (scale * (float)data[byte_off + 5] + zero) * x[base_idx + 5]
             + (scale * (float)data[byte_off + 6] + zero) * x[base_idx + 6]
             + (scale * (float)data[byte_off + 7] + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
