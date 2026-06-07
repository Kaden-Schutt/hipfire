// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched four-way F16-weight x F32-input GEMV.
// X is [B, K]; outputs are [B, M*]. Grid: x=output row across all four
// projections, y=batch row. One wave computes one output row for one token.
__launch_bounds__(32)
extern "C" __global__ void fused_qkvza_f16_xf32_batched(
    const _Float16* __restrict__ Wqkv,
    const _Float16* __restrict__ Wz,
    const _Float16* __restrict__ Wbeta,
    const _Float16* __restrict__ Walpha,
    const float*    __restrict__ X,
    float*          __restrict__ Yqkv,
    float*          __restrict__ Yz,
    float*          __restrict__ Ybeta,
    float*          __restrict__ Yalpha,
    int Mqkv,
    int Mz,
    int Mbeta,
    int Malpha,
    int K,
    int B
) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int total = Mqkv + Mz + Mbeta + Malpha;
    if (row >= total || b >= B) return;

    const _Float16* W = Wqkv;
    float* Y = Yqkv;
    int local = row;
    int M = Mqkv;
    if (local >= Mqkv) {
        local -= Mqkv;
        W = Wz;
        Y = Yz;
        M = Mz;
        if (local >= Mz) {
            local -= Mz;
            W = Wbeta;
            Y = Ybeta;
            M = Mbeta;
            if (local >= Mbeta) {
                local -= Mbeta;
                W = Walpha;
                Y = Yalpha;
                M = Malpha;
            }
        }
    }
    if (local >= M) return;

    const _Float16* w_row = W + (size_t)local * K;
    const float* x_row = X + (size_t)b * K;
    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x_row[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) Y[(size_t)b * M + local] = sum;
}
