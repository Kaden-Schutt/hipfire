// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Four independent F16-weight x F32-input GEMVs in one launch.
// Used by the Qwen3.5 FP32-DeltaNet fallback for BF16/F16 baseline models.
__launch_bounds__(32)
extern "C" __global__ void fused_qkvza_f16_xf32(
    const _Float16* __restrict__ Wqkv,
    const _Float16* __restrict__ Wz,
    const _Float16* __restrict__ Wbeta,
    const _Float16* __restrict__ Walpha,
    const float*    __restrict__ x,
    float*          __restrict__ yqkv,
    float*          __restrict__ yz,
    float*          __restrict__ ybeta,
    float*          __restrict__ yalpha,
    int Mqkv,
    int Mz,
    int Mbeta,
    int Malpha,
    int K
) {
    int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int total = Mqkv + Mz + Mbeta + Malpha;
    if (row >= total) return;

    const _Float16* W = Wqkv;
    float* y = yqkv;
    int local = row;
    int M = Mqkv;
    if (local >= Mqkv) {
        local -= Mqkv;
        W = Wz;
        y = yz;
        M = Mz;
        if (local >= Mz) {
            local -= Mz;
            W = Wbeta;
            y = ybeta;
            M = Mbeta;
            if (local >= Mbeta) {
                local -= Mbeta;
                W = Walpha;
                y = yalpha;
                M = Malpha;
            }
        }
    }
    if (local >= M) return;

    const _Float16* w_row = W + (size_t)local * K;
    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[local] = sum;
}
