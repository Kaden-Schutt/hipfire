// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq2g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 40;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 40;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));

        // 128 weights / 32 threads = 4 weights per thread = 1 byte
        int base_idx = g * 128 + tid * 4;
        unsigned char b = *((const unsigned char*)(gptr + 8 + tid));

        acc += (scale * (float)((b)      & 0x3) + zero) * x[base_idx]
             + (scale * (float)((b >> 2) & 0x3) + zero) * x[base_idx + 1]
             + (scale * (float)((b >> 4) & 0x3) + zero) * x[base_idx + 2]
             + (scale * (float)((b >> 6))       + zero) * x[base_idx + 3];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
