// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define MAX_HEADS 64

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

extern "C" __global__ __launch_bounds__(MAX_HEADS, 4)
void indexer_relu_score_f32_buf_sharded_gfx1201(
    const float *__restrict__ q, const float *__restrict__ cache0,
    const float *__restrict__ cache1, const float *__restrict__ cache2,
    const float *__restrict__ cache3, const float *__restrict__ weights,
    float *__restrict__ scores, const int *__restrict__ N_buf, int H, int D,
    int world, int block_rows) {
  const int N = N_buf[0];
  const int n = blockIdx.x;
  const int tid = threadIdx.x;
  if (n >= N) {
    if (tid == 0)
      scores[n] = -1e30f;
    return;
  }
  if (tid >= H)
    return;

  const float *qh = q + (size_t)tid * D;
  const float *kt = compressor_shard_row(cache0, cache1, cache2, cache3, n,
                                         D, world, block_rows);
  float dot = 0.0f;
  for (int d = 0; d < D; ++d)
    dot += qh[d] * kt[d];
  const float v = (dot > 0.0f ? dot : 0.0f) * weights[tid];

  __shared__ float reduce[MAX_HEADS];
  reduce[tid] = v;
  __syncthreads();
  for (int s = MAX_HEADS / 2; s > 0; s >>= 1) {
    if (tid < s)
      reduce[tid] += reduce[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    scores[n] = reduce[0];
}
