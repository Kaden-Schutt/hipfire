// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Down-projection sibling of the gfx942 F32 MFMA-as-GEMV ceiling probe.
// Arithmetic inputs stay F32; only the reduction/launch structure changes.

typedef float vfloat4 __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_mfma_gfx942(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ x_residual,
    int M,
    int K
) {
    const int lane = threadIdx.x;
    const int n_lane = lane & 15;
    const int strip = lane >> 4;
    const int row_start = blockIdx.x * 16;
    if (row_start >= M) return;
    const int krank = blockIdx.y;
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;

    const int my_row = row_start + n_lane;
    const int safe_row = my_row < M ? my_row : M - 1;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)safe_row * groups_per_row * 72;
    vfloat4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + g * 72;
        const __half* cbh = reinterpret_cast<const __half*>(gp);
        float cb0 = strip == 0 ? __half2float(cbh[0]) : 0.0f;
        float cb1 = strip == 0 ? __half2float(cbh[1]) : 0.0f;
        float cb2 = strip == 0 ? __half2float(cbh[2]) : 0.0f;
        float cb3 = strip == 0 ? __half2float(cbh[3]) : 0.0f;
        cb0 = __shfl(cb0, n_lane, 64);
        cb1 = __shfl(cb1, n_lane, 64);
        cb2 = __shfl(cb2, n_lane, 64);
        cb3 = __shfl(cb3, n_lane, 64);

        #pragma unroll 4
        for (int ki = 0; ki < 256; ki += 4) {
            unsigned int packed = strip == 0
                ? *(reinterpret_cast<const unsigned char*>(gp + 8 + ki / 4))
                : 0u;
            packed = __shfl(packed, n_lane, 64);
            const unsigned int q = (packed >> (strip * 2)) & 3u;
            const float a = q == 0 ? cb0 : (q == 1 ? cb1 : (q == 2 ? cb2 : cb3));

            float b = n_lane == 0 ? x[g * 256 + ki + strip] : 0.0f;
            b = __shfl(b, strip * 16, 64);
            acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
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
