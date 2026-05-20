// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HFQ3-G128: 3-bit with 128-weight groups (finer granularity than G256).
// Block: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights (0.4375 B/w).
// 32 threads × 4 weights = 128. Each pair of threads shares 3 bytes (8 weights).
// Thread tid reads from byte offset (tid/2)*3, unpacks 4 of 8 weights.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq3g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 56;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 56;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 128 weights / 32 threads = 4 weights per thread
        // Packed as 16 chunks of 8 weights × 3 bits = 3 bytes each = 48 bytes
        // Thread pair (2*n, 2*n+1) shares chunk n's 3 bytes.
        // Even thread gets weights 0-3, odd thread gets weights 4-7.
        int chunk = tid / 2;
        int half = tid & 1; // 0=lower 4 weights, 1=upper 4 weights
        int byte_off = chunk * 3;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];

        // Unpack all 8 × 3-bit from 24 bits (same as G256 kernel)
        int q0 = b0 & 7;
        int q1 = (b0 >> 3) & 7;
        int q2 = ((b0 >> 6) | (b1 << 2)) & 7;
        int q3 = (b1 >> 1) & 7;
        int q4 = (b1 >> 4) & 7;
        int q5 = ((b1 >> 7) | (b2 << 1)) & 7;
        int q6 = (b2 >> 2) & 7;
        int q7 = (b2 >> 5) & 7;

        int base_idx = g * 128 + chunk * 8 + half * 4;

        if (half == 0) {
            acc += (scale * (float)q0 + zero) * x[base_idx]
                 + (scale * (float)q1 + zero) * x[base_idx + 1]
                 + (scale * (float)q2 + zero) * x[base_idx + 2]
                 + (scale * (float)q3 + zero) * x[base_idx + 3];
        } else {
            acc += (scale * (float)q4 + zero) * x[base_idx]
                 + (scale * (float)q5 + zero) * x[base_idx + 1]
                 + (scale * (float)q6 + zero) * x[base_idx + 2]
                 + (scale * (float)q7 + zero) * x[base_idx + 3];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
