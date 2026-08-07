// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12 M=128 sister of the established M=64 grouped MQ2-Lloyd kernel.
// Eight row warps share one 16-slot X tile, halving X staging traffic without
// changing any row's decode or WMMA accumulation order.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_mem_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define M_TILE 128
#define N_TILE 16
#define M_PER_WARP 16
#define K_GROUP 256

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2_gfx12(
    const unsigned long long *__restrict__ expert_weight_ptrs,
    const int *__restrict__ expert_tile_ids,
    const int *__restrict__ sorted_slot_index,
    const _Float16 *__restrict__ X_src, float *__restrict__ Y_grouped, int M,
    int K, int x_row_div, int m_total) {
  const int block_m = blockIdx.x * M_TILE;
  const int tile_y = blockIdx.y;
  const int slot_start = tile_y * N_TILE;
  if (block_m >= M || slot_start >= m_total)
    return;

  const int expert_id = expert_tile_ids[tile_y];
  if (expert_id < 0)
    return;
  const char *__restrict__ A =
      reinterpret_cast<const char *>(expert_weight_ptrs[expert_id]);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int k_half = lane >> 4;
  const int warp_m_start = block_m + warp_id * M_PER_WARP;
  const int my_row = warp_m_start + lane16;
  const int safe_row = my_row < M ? my_row : M - 1;

  __shared__ int slot_x_row[N_TILE];
  if (tid < N_TILE) {
    const int slot_idx = slot_start + tid;
    int x_row = -1;
    if (slot_idx < m_total) {
      const int flat = sorted_slot_index[slot_idx];
      if (flat >= 0)
        x_row = x_row_div > 1 ? flat / x_row_div : flat;
    }
    slot_x_row[tid] = x_row;
  }

  __shared__ _Float16 lds_x[N_TILE * K_GROUP];
  const int groups_per_row = K / K_GROUP;
  const long long row_stride = (long long)groups_per_row * 72;
  const char *row_base = A + (long long)safe_row * row_stride;
  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const half16_mem_t zero16 = {0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0};

  __syncthreads();
  for (int g = 0; g < groups_per_row; ++g) {
    // 256 threads stage 16 slots x 256 values: one contiguous half16/thread.
    const int slot_in_block = tid >> 4;
    const int k_off = (tid & 15) * 16;
    const int x_row = slot_x_row[slot_in_block];
    half16_mem_t value;
    if (x_row >= 0) {
      const long long x_off =
          (long long)x_row * K + g * K_GROUP + k_off;
      value = *reinterpret_cast<const half16_mem_t *>(X_src + x_off);
    } else {
      value = zero16;
    }
    _Float16 *dst = lds_x + slot_in_block * K_GROUP + k_off;
    *reinterpret_cast<half16_mem_t *>(dst) = value;
    __syncthreads();

    const char *gp = row_base + (long long)g * 72;
    const __half *cb_h = reinterpret_cast<const __half *>(gp);
    const _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
    const _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
    const _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
    const _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
    const _Float16 d01 = cb1 - cb0;
    const _Float16 d02 = cb2 - cb0;
    const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;

#define DQ8(reg, i, pk, sh)                                                   \
  do {                                                                         \
    const _Float16 qlo = (_Float16)(int)(((pk) >> (sh)) & 1u);                \
    const _Float16 qhi = (_Float16)(int)(((pk) >> ((sh) + 1)) & 1u);          \
    (reg)[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor;             \
  } while (0)

#pragma unroll
    for (int kt = 0; kt < 16; ++kt) {
      const int k_tile = kt * 16;
      const unsigned int packed =
          *reinterpret_cast<const unsigned int *>(gp + 8 + k_tile / 4);
      const int bit = k_half * 16;
      half8_t a;
      DQ8(a, 0, packed, bit + 0);
      DQ8(a, 1, packed, bit + 2);
      DQ8(a, 2, packed, bit + 4);
      DQ8(a, 3, packed, bit + 6);
      DQ8(a, 4, packed, bit + 8);
      DQ8(a, 5, packed, bit + 10);
      DQ8(a, 6, packed, bit + 12);
      DQ8(a, 7, packed, bit + 14);
      const _Float16 *lds_slot = lds_x + lane16 * K_GROUP;
      const half8_t b = *reinterpret_cast<const half8_t *>(
          lds_slot + k_tile + k_half * 8);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
#undef DQ8
    __syncthreads();
  }

  const int out_col = slot_start + lane16;
  if (out_col < m_total) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int out_row = warp_m_start + 8 * k_half + j;
      if (out_row < M)
        Y_grouped[(long long)out_col * M + out_row] = acc[j];
    }
  }
}
