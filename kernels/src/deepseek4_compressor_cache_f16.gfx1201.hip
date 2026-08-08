// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define MAX_HEADS 64

extern "C" __global__ __launch_bounds__(MAX_HEADS, 4)
void indexer_relu_score_f16_buf_gfx1201(
    const float *__restrict__ q, const _Float16 *__restrict__ cache,
    const float *__restrict__ weights, float *__restrict__ scores,
    const int *__restrict__ N_buf, int H, int D) {
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
  const _Float16 *kt = cache + (size_t)n * D;
  float dot = 0.0f;
  for (int d = 0; d < D; ++d)
    dot += qh[d] * (float)kt[d];
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

extern "C" __global__ __launch_bounds__(MAX_HEADS, 4)
void indexer_relu_score_batched_f16_gfx1201(
    const float *__restrict__ q, const _Float16 *__restrict__ cache,
    const float *__restrict__ weights,
    const int *__restrict__ n_per_batch, float *__restrict__ scores, int H,
    int D, int N_max, int batch_size) {
  const int n = blockIdx.x;
  const int b = blockIdx.y;
  if (n >= N_max || b >= batch_size)
    return;
  const int tid = threadIdx.x;
  if (tid >= H)
    return;
  if (n >= n_per_batch[b]) {
    if (tid == 0)
      scores[(size_t)b * N_max + n] = -1e30f;
    return;
  }
  const float *qh = q + ((size_t)b * H + tid) * D;
  const _Float16 *kt = cache + (size_t)n * D;
  float dot = 0.0f;
  for (int d = 0; d < D; ++d)
    dot += qh[d] * (float)kt[d];
  const float v =
      (dot > 0.0f ? dot : 0.0f) * weights[(size_t)b * H + tid];
  __shared__ float reduce[MAX_HEADS];
  reduce[tid] = v;
  __syncthreads();
  for (int s = MAX_HEADS / 2; s > 0; s >>= 1) {
    if (tid < s)
      reduce[tid] += reduce[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    scores[(size_t)b * N_max + n] = reduce[0];
}

// WMMA fragment width differs by generation, so the A/B staging and the D
// row mapping below are both generation-specific.
//
//   gfx12  A/B are 8-wide per lane; the wave splits each 16-element K chunk
//          across lane halves (k_half selects which half). D row is
//          8 * k_half + j.
//   gfx11  A/B are 16-wide per lane; every lane carries the whole 16-element
//          K chunk and lanes 16..31 duplicate lanes 0..15. D row is
//          2 * j + k_half.
//
// Both keep the same D column (lane16), the same accumulator type, and F32
// accumulation. See attention_dflash_wmma.hip for the established gfx11
// spelling and its D-layout comment.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define HF_WMMA_FRAG_WIDTH 8
typedef _Float16 __attribute__((ext_vector_type(8))) half_frag_t;
#else
#define HF_WMMA_FRAG_WIDTH 16
typedef _Float16 __attribute__((ext_vector_type(16))) half_frag_t;
#endif
typedef float __attribute__((ext_vector_type(8))) float8_t;
#define WMMA_TILE 16
#define HEADS_PER_WARP 16
#define WARPS_PER_BLOCK 4

__launch_bounds__(128, 2) extern "C" __global__ void
indexer_relu_score_wmma_batched_f16_gfx1201(
    const float *__restrict__ q, const _Float16 *__restrict__ cache,
    const float *__restrict__ weights,
    const int *__restrict__ n_per_batch, float *__restrict__ scores, int H,
    int D, int N_max, int batch_size) {
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
  const int safe_h = my_h < H ? my_h : H - 1;
  const int safe_n = my_n < n_valid_b ? my_n : n_valid_b - 1;
  const float *qh = q + ((size_t)b * H + safe_h) * D;
  const _Float16 *kn = cache + (size_t)safe_n * D;
  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < D; k += WMMA_TILE) {
    half_frag_t q_frag;
    half_frag_t k_frag;
#if defined(__gfx1200__) || defined(__gfx1201__)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q_frag[i] = (_Float16)qh[k + k_half * 8 + i];
      k_frag[i] = (_Float16)kn[k + k_half * 8 + i];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_frag, k_frag,
                                                            acc);
#else
#pragma unroll
    for (int i = 0; i < WMMA_TILE; ++i) {
      q_frag[i] = (_Float16)qh[k + i];
      k_frag[i] = (_Float16)kn[k + i];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag, k_frag, acc);
#endif
  }
  __shared__ float lds[MAX_HEADS][WMMA_TILE];
  const int slot = n_start + lane16;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    const int head = h_start + 8 * k_half + j;
#else
    // gfx11 wave32 D layout: element j of lane t is row 2*j + (t >> 4),
    // column t & 15 — so the head stride across j is 2, not 1, and k_half
    // selects the odd row rather than the upper eight heads.
    const int head = h_start + 2 * j + k_half;
#endif
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

extern "C" __global__ void deepseek4_topk_kv_gather_f16_buf_gfx1201(
    const _Float16 *__restrict__ cache, const int *__restrict__ topk_idx,
    float *__restrict__ out, const int *__restrict__ K_buf,
    const int *__restrict__ N_compressed_buf, int head_dim, int out_stride,
    int col_offset, float scale) {
  const int K = K_buf[0];
  const int N = N_compressed_buf[0];
  const int k = blockIdx.x;
  const int d = threadIdx.x;
  if (k >= K || d >= head_dim)
    return;
  const int idx = topk_idx[k];
  float value = 0.0f;
  if (idx >= 0 && idx < N)
    value = (float)cache[(size_t)idx * head_dim + d] * scale;
  out[d * out_stride + col_offset + k] = value;
}

extern "C" __global__ void
deepseek4_topk_kv_gather_identity_f16_buf_gfx1201(
    const _Float16 *__restrict__ cache, float *__restrict__ out,
    const int *__restrict__ K_buf, int head_dim, int out_stride) {
  const int K = K_buf[0];
  const int k = blockIdx.x;
  const int d = threadIdx.x;
  if (k < K && d < head_dim)
    out[d * out_stride + k] =
        (float)cache[(size_t)k * head_dim + d];
}

extern "C" __global__ __launch_bounds__(256, 4)
void deepseek4_topk_kv_gather_batched_tiled_f16_gfx1201(
    const _Float16 *__restrict__ cache, const int *__restrict__ topk_idx,
    float *__restrict__ out, int K, int head_dim, int N_compressed,
    int out_stride, int col_offset, float scale, int batch_size) {
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
    if (k < K && d < head_dim && idx >= 0 && idx < N_compressed)
      value = (float)cache[(size_t)idx * head_dim + d] * scale;
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

extern "C" __global__ void
deepseek4_topk_kv_gather_identity_batched_f16_gfx1201(
    const _Float16 *__restrict__ cache, float *__restrict__ out, int K,
    int head_dim, int out_stride, int batch_size) {
  const int k = blockIdx.x;
  const int b = blockIdx.y;
  const int d = threadIdx.x;
  if (k < K && b < batch_size && d < head_dim)
    out[((size_t)b * head_dim + d) * out_stride + k] =
        (float)cache[(size_t)k * head_dim + d];
}
