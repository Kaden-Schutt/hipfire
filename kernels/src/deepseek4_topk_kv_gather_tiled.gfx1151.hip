// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 tiled transpose for the ratio=4 DeepSeek V4 top-k gather.
//
// The incumbent assigns one 512-thread block to each selected row. Its reads
// are contiguous, but each wave stores with a 512-float stride. This variant
// stages a 32x32 tile in LDS so both the row-major cache reads and the
// dimension-major output writes are contiguous. The padded LDS row removes
// transpose bank conflicts.
extern "C" __global__ __launch_bounds__(256, 4)
void deepseek4_topk_kv_gather_tiled_f32_buf(
    const float* __restrict__ kv_cache,         // [N_compressed, head_dim]
    const int*   __restrict__ topk_idx,         // [K]
    float*       __restrict__ out,              // [head_dim, out_stride]
    const int*   __restrict__ K_buf,            // [1]
    const int*   __restrict__ N_compressed_buf, // [1]
    int head_dim,
    int out_stride,
    int col_offset,
    float scale
) {
    constexpr int TILE = 32;
    __shared__ float tile[TILE][TILE + 1];
    __shared__ int selected[TILE];

    const int tid = threadIdx.x;
    const int k_base = blockIdx.x * TILE;
    const int d_base = blockIdx.y * TILE;
    const int K = K_buf[0];
    const int N_compressed = N_compressed_buf[0];

    if (tid < TILE) {
        const int k = k_base + tid;
        selected[tid] = k < K ? topk_idx[k] : -1;
    }
    __syncthreads();

    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        const int linear = tid + phase * blockDim.x;
        const int local_k = linear >> 5;
        const int local_d = linear & 31;
        const int k = k_base + local_k;
        const int d = d_base + local_d;
        const int idx = selected[local_k];
        float val = 0.0f;
        if (k < K && d < head_dim && idx >= 0 && idx < N_compressed) {
            val = kv_cache[idx * head_dim + d] * scale;
        }
        tile[local_k][local_d] = val;
    }
    __syncthreads();

    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        const int linear = tid + phase * blockDim.x;
        const int local_d = linear >> 5;
        const int local_k = linear & 31;
        const int k = k_base + local_k;
        const int d = d_base + local_d;
        if (k < K && d < head_dim) {
            out[d * out_stride + col_offset + k] = tile[local_k][local_d];
        }
    }
}
