// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 batched top-K KV gather with an LDS transpose.
//
// The portable route assigns one 512-thread block to every (batch, selected
// row) pair. Reads are contiguous, but its stores are separated by out_stride
// floats. This kernel stages a 32x32 [selected, dimension] tile so both the
// cache reads and the dimension-major output writes are contiguous. The +1
// LDS column removes transpose bank conflicts.
extern "C" __global__ __launch_bounds__(256, 4)
void deepseek4_topk_kv_gather_batched_tiled_gfx1201(
    const float *__restrict__ kv_cache, // [N_compressed, head_dim]
    const int *__restrict__ topk_idx,   // [batch_size, K]
    float *__restrict__ out,            // [batch_size, head_dim, out_stride]
    int K, int head_dim, int N_compressed, int out_stride, int col_offset,
    float scale, int batch_size) {
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
    float val = 0.0f;
    if (k < K && d < head_dim && idx >= 0 && idx < N_compressed)
      val = kv_cache[(size_t)idx * head_dim + d] * scale;
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
      out[((size_t)b * head_dim + d) * out_stride + col_offset + k] =
          tile[local_k][local_d];
    }
  }
}
