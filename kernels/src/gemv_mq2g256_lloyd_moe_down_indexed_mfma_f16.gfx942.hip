// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float vfloat4 __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_mfma_f16_gfx942(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const half_t* __restrict__ rot_batch,
    float* __restrict__ x_residual,
    int M,
    int K
) {
    __shared__ half_t lds_x[256];
    const int lane = threadIdx.x;
    const int n_lane = lane & 15;
    const int strip = lane >> 4;
    const int row_start = blockIdx.x * 16;
    if (row_start >= M) return;
    const int krank = blockIdx.y;
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const half_t* __restrict__ x = rot_batch + (long long)krank * K;

    const int my_row = row_start + n_lane;
    const int safe_row = my_row < M ? my_row : M - 1;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)safe_row * groups_per_row * 72;
    vfloat4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; ++g) {
        *reinterpret_cast<half4*>(lds_x + lane * 4) =
            *reinterpret_cast<const half4*>(x + g * 256 + lane * 4);
        __syncthreads();

        const char* gp = row_ptr + g * 72;
        const half_t* cb = reinterpret_cast<const half_t*>(gp);
        #pragma unroll
        for (int ki = 0; ki < 256; ki += 16) {
            const unsigned char packed = *(reinterpret_cast<const unsigned char*>(
                gp + 8 + (ki + strip * 4) / 4));
            half4 a;
            a[0] = cb[(packed >> 0) & 3u];
            a[1] = cb[(packed >> 2) & 3u];
            a[2] = cb[(packed >> 4) & 3u];
            a[3] = cb[(packed >> 6) & 3u];
            const half4 b = *reinterpret_cast<const half4*>(lds_x + ki + strip * 4);
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
        }
    }

    if (n_lane == 0) {
        const float scale = topk_weights[krank];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row_start + strip * 4 + r;
            if (row < M) atomicAdd(&x_residual[row], scale * acc[r]);
        }
    }
}
