// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ const float *
compressor_shard_row(const float *cache0, const float *cache1,
                     const float *cache2, const float *cache3, int global_row,
                     int row_elems, int world, int block_rows) {
  const int global_block = global_row / block_rows;
  const int owner = global_block % world;
  const int local_row = (global_block / world) * block_rows +
                        global_row % block_rows;
  const float *base = owner == 0   ? cache0
                      : owner == 1 ? cache1
                      : owner == 2 ? cache2
                                   : cache3;
  return base + (size_t)local_row * row_elems;
}

extern "C" __global__ __launch_bounds__(256, 4)
void deepseek4_topk_kv_gather_batched_tiled_sharded_gfx1201(
    const float *__restrict__ cache0, const float *__restrict__ cache1,
    const float *__restrict__ cache2, const float *__restrict__ cache3,
    const int *__restrict__ topk_idx, float *__restrict__ out, int K,
    int head_dim, int N_compressed, int out_stride, int col_offset,
    float scale, int batch_size, int world, int block_rows) {
  constexpr int TILE = 32;
  __shared__ float tile[TILE][TILE + 1];
  __shared__ int selected[TILE];

  const int tid = threadIdx.x;
  const int k_base = blockIdx.x * TILE;
  const int d_base = blockIdx.y * TILE;
  const int b = blockIdx.z;
  if (b >= batch_size)
    return;

  if (tid < TILE) {
    const int k = k_base + tid;
    selected[tid] = k < K ? topk_idx[(size_t)b * K + k] : -1;
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
    float value = 0.0f;
    if (k < K && d < head_dim && idx >= 0 && idx < N_compressed) {
      const float *row = compressor_shard_row(
          cache0, cache1, cache2, cache3, idx, head_dim, world, block_rows);
      value = row[d] * scale;
    }
    tile[local_k][local_d] = value;
  }
  __syncthreads();

#pragma unroll
  for (int phase = 0; phase < 4; ++phase) {
    const int linear = tid + phase * blockDim.x;
    const int local_d = linear >> 5;
    const int local_k = linear & 31;
    const int k = k_base + local_k;
    const int d = d_base + local_d;
    if (k < K && d < head_dim)
      out[((size_t)b * head_dim + d) * out_stride + col_offset + k] =
          tile[local_k][local_d];
  }
}
