// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// In-place inverse signed-FWHT-256 for query-tiled MQKV prefill output.
// Grid: [batch_size, n_heads]. Block: [32].
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ __launch_bounds__(32, 16) void mqkv_inverse_fwht256_batched(
    float* __restrict__ out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int batch_size
) {
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    if (row >= batch_size || h >= n_heads) return;

    const int lane = threadIdx.x & 31;
    const int d0 = lane * 8;
    float* dst = out + ((long long)row * n_heads + h) * 256;
    float v0 = dst[d0 + 0];
    float v1 = dst[d0 + 1];
    float v2 = dst[d0 + 2];
    float v3 = dst[d0 + 3];
    float v4 = dst[d0 + 4];
    float v5 = dst[d0 + 5];
    float v6 = dst[d0 + 6];
    float v7 = dst[d0 + 7];
    fwht_shfl_inverse_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, lane);
    dst[d0 + 0] = v0;
    dst[d0 + 1] = v1;
    dst[d0 + 2] = v2;
    dst[d0 + 3] = v3;
    dst[d0 + 4] = v4;
    dst[d0 + 5] = v5;
    dst[d0 + 6] = v6;
    dst[d0 + 7] = v7;
}
