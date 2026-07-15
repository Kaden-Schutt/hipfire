// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bq1g128(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 18;         // 18-byte binary blocks
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 18;
        float d = (float)(*(const _Float16*)gptr);      // FP16 scale at block start
        int base_idx = g * 128 + tid * 4;
        unsigned char b = *((const unsigned char*)(gptr + 2 + (tid >> 1)));
        int sh = (tid & 1) * 4;
        acc += d * (((b >> (sh + 0)) & 1) ? x[base_idx]     : -x[base_idx])
             + d * (((b >> (sh + 1)) & 1) ? x[base_idx + 1] : -x[base_idx + 1])
             + d * (((b >> (sh + 2)) & 1) ? x[base_idx + 2] : -x[base_idx + 2])
             + d * (((b >> (sh + 3)) & 1) ? x[base_idx + 3] : -x[base_idx + 3]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}
