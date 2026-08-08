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

extern "C" __global__ void deepseek4_topk_kv_gather_f32_buf_sharded_gfx1201(
    const float *__restrict__ cache0, const float *__restrict__ cache1,
    const float *__restrict__ cache2, const float *__restrict__ cache3,
    const int *__restrict__ topk_idx, float *__restrict__ out,
    const int *__restrict__ K_buf, const int *__restrict__ N_compressed_buf,
    int head_dim, int out_stride, int col_offset, float scale, int world,
    int block_rows) {
  const int K = K_buf[0];
  const int N_compressed = N_compressed_buf[0];
  const int k = blockIdx.x;
  const int d = threadIdx.x;
  if (k >= K || d >= head_dim)
    return;
  const int idx = topk_idx[k];
  float value = 0.0f;
  if (idx >= 0 && idx < N_compressed) {
    const float *row = compressor_shard_row(
        cache0, cache1, cache2, cache3, idx, head_dim, world, block_rows);
    value = row[d] * scale;
  }
  out[d * out_stride + col_offset + k] = value;
}
