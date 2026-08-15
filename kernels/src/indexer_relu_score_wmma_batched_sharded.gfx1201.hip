// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define WMMA_TILE 16
#define HEADS_PER_WARP 16
#define WARPS_PER_BLOCK 4
#define MAX_HEADS (HEADS_PER_WARP * WARPS_PER_BLOCK)

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

// Arithmetic and global score ordering are identical to the incumbent gfx12
// WMMA kernel. Only the K-row address translation changes.
__launch_bounds__(128, 2) extern "C" __global__ void
indexer_relu_score_wmma_batched_sharded_gfx1201(
    const float *__restrict__ q, const float *__restrict__ cache0,
    const float *__restrict__ cache1, const float *__restrict__ cache2,
    const float *__restrict__ cache3, const float *__restrict__ weights,
    const int *__restrict__ n_per_batch, float *__restrict__ scores, int H,
    int D, int N_max, int batch_size, int world, int block_rows) {
  const int b = blockIdx.x;
  const int n_start = blockIdx.y * WMMA_TILE;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int k_half = lane >> 4;
  if (b >= batch_size || n_start >= N_max)
    return;

  const int n_valid_b = n_per_batch[b];
  if (n_valid_b <= 0 || n_start >= n_valid_b) {
    if (tid < WMMA_TILE) {
      const int slot = n_start + tid;
      if (slot < N_max)
        scores[(size_t)b * N_max + slot] = -1e30f;
    }
    return;
  }

  const int h_start = warp_id * HEADS_PER_WARP;
  const int my_h = h_start + lane16;
  const int my_n = n_start + lane16;
  const int safe_h = (my_h < H) ? my_h : (H - 1);
  const int safe_n = (my_n < n_valid_b) ? my_n : (n_valid_b - 1);
  const float *qh = q + ((size_t)b * H + safe_h) * D;
  const float *kn = compressor_shard_row(cache0, cache1, cache2, cache3,
                                         safe_n, D, world, block_rows);

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < D; k += WMMA_TILE) {
    half8_t q_frag;
    half8_t k_frag;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q_frag[i] = (_Float16)qh[k + k_half * 8 + i];
      k_frag[i] = (_Float16)kn[k + k_half * 8 + i];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_frag, k_frag,
                                                            acc);
  }

  __shared__ float lds[MAX_HEADS][WMMA_TILE];
  const int slot = n_start + lane16;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int head = h_start + 8 * k_half + j;
    float value = 0.0f;
    if (head < H && slot < n_valid_b) {
      const float qk = acc[j];
      value = (qk > 0.0f ? qk : 0.0f) * weights[(size_t)b * H + head];
    }
    lds[head][lane16] = value;
  }
  __syncthreads();

  if (tid < WMMA_TILE) {
    const int out_slot = n_start + tid;
    if (out_slot < n_valid_b) {
      float sum = 0.0f;
#pragma unroll
      for (int h = 0; h < MAX_HEADS; ++h)
        if (h < H)
          sum += lds[h][tid];
      scores[(size_t)b * N_max + out_slot] = sum;
    } else if (out_slot < N_max) {
      scores[(size_t)b * N_max + out_slot] = -1e30f;
    }
  }
}
