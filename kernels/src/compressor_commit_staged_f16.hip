// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define BLOCK_HIDDEN 256

// Address-only twin of compressor_softmax_pool_f32_buf: slot_buf is the
// commit gate, while arithmetic lands in one F32 staging row.
extern "C" __global__ void compressor_softmax_pool_f32_staged_buf(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state, float *__restrict__ staged,
    const int *__restrict__ slot_buf, int T, int head_dim) {
  if (slot_buf[0] < 0)
    return;
  const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
  if (d >= head_dim)
    return;
  float max_s = score_state[d];
  for (int t = 1; t < T; ++t) {
    const float s = score_state[t * head_dim + d];
    if (s > max_s)
      max_s = s;
  }
  float sum_exp = 0.0f;
  float weighted = 0.0f;
  for (int t = 0; t < T; ++t) {
    const float e = __expf(score_state[t * head_dim + d] - max_s);
    sum_exp += e;
    weighted += e * kv_state[t * head_dim + d];
  }
  staged[d] = weighted / sum_exp;
}

// Address-only twin of rmsnorm_f32_at_slot_buf.
extern "C" __global__ void rmsnorm_f32_staged_buf(
    float *__restrict__ staged, const float *__restrict__ weight,
    const int *__restrict__ slot_buf, int n, float eps) {
  extern __shared__ float sdata[];
  if (slot_buf[0] < 0)
    return;
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = staged[i];
    sum_sq += v * v;
  }
  sdata[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  const float rms = rsqrtf(sdata[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    staged[i] = staged[i] * weight[i] * rms;
}

// Address-only twin of rope_tail_yarn_interleaved_at_slot_buf_f32.
extern "C" __global__ void
rope_tail_yarn_interleaved_staged_buf_f32(
    float *__restrict__ staged, const int *__restrict__ pos_buf,
    const int *__restrict__ slot_buf, int head_dim, int n_rot,
    float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float corr_low, float corr_high) {
  if (slot_buf[0] < 0)
    return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = n_rot / 2;
  if (i >= half)
    return;
  const int pos = pos_buf[0];
  const int idx_pair = i * 2;
  if (ext_factor == 0.0f) {
    const float freq =
        1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    const float angle = (float)pos * freq;
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);
    const int d_even = head_dim - n_rot + 2 * i;
    const int d_odd = d_even + 1;
    const float x0 = staged[d_even];
    const float x1 = staged[d_odd];
    staged[d_even] = x0 * cos_a - x1 * sin_a;
    staged[d_odd] = x0 * sin_a + x1 * cos_a;
    return;
  }
  const float freq =
      1.0f / powf(freq_base, (float)idx_pair / (float)n_rot);
  const float theta_extrap = (float)pos * freq;
  const float theta_interp = freq_scale * theta_extrap;
  float theta = theta_interp;
  float mscale = attn_factor;
  const float y =
      ((float)i - corr_low) / fmaxf(0.001f, corr_high - corr_low);
  const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
  const float ramp_mix = ramp * ext_factor;
  theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
  mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
  const float cos_a = cosf(theta) * mscale;
  const float sin_a = sinf(theta) * mscale;
  const int d_even = head_dim - n_rot + idx_pair;
  const int d_odd = d_even + 1;
  const float x0 = staged[d_even];
  const float x1 = staged[d_odd];
  staged[d_even] = x0 * cos_a - x1 * sin_a;
  staged[d_odd] = x0 * sin_a + x1 * cos_a;
}

extern "C" __global__ void cast_f32_to_f16_at_slot_buf(
    const float *__restrict__ staged, _Float16 *__restrict__ cache,
    const int *__restrict__ slot_buf, int n) {
  const int slot = slot_buf[0];
  if (slot < 0)
    return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    cache[(size_t)slot * n + i] = (_Float16)staged[i];
}
