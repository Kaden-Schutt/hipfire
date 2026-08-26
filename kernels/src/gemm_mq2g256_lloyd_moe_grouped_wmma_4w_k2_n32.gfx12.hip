// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12-native N=32 tile-pair sibling of
// gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_gfx12.
//
// Two adjacent 16-slot expert tiles share one block. When their expert ids
// match (common after the padded expert sort), each MQ2-Lloyd A fragment is
// decoded once and consumed by two RDNA4 WMMAs. At an expert boundary the two
// fragments are decoded independently, preserving the incumbent work.
//
// RDNA4 contract: half8 operands, K split by k_grp=(lane>>4), and
// acc[j] = C[8*k_grp+j][lane&15].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_mem_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define M_TILE 64
#define N_TILE 16
#define M_PER_WARP 16
#define K_GROUP 256
#define SLOTS_PER_WARP 4

__launch_bounds__(128, 2) extern "C" __global__ void
gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32_gfx12(
    const unsigned long long *__restrict__ expert_weight_ptrs,
    const int *__restrict__ expert_tile_ids,
    const int *__restrict__ sorted_slot_index,
    const _Float16 *__restrict__ X_src, float *__restrict__ Y_grouped, int M,
    int K, int x_row_div, int m_total) {
  const int block_m = blockIdx.x * M_TILE;
  const int n_tiles = (m_total + N_TILE - 1) / N_TILE;
  const int tile0 = blockIdx.y * 2;
  const int tile1 = tile0 + 1;
  if (block_m >= M || tile0 >= n_tiles)
    return;

  const int e0 = expert_tile_ids[tile0];
  const int e1 = tile1 < n_tiles ? expert_tile_ids[tile1] : -1;
  if (e0 < 0 && e1 < 0)
    return;
  const bool same = e0 >= 0 && e0 == e1;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int lane_row = lane & 15;
  const int k_grp = lane >> 4;
  const int warp_m_start = block_m + warp_id * M_PER_WARP;
  const int my_row = warp_m_start + lane_row;
  const int safe_row = my_row < M ? my_row : M - 1;

  __shared__ int slot_x_row[2 * N_TILE];
  if (tid < 2 * N_TILE) {
    const int slot_idx = tile0 * N_TILE + tid;
    int x_row = -1;
    if (slot_idx < m_total) {
      const int flat = sorted_slot_index[slot_idx];
      if (flat >= 0)
        x_row = x_row_div > 1 ? flat / x_row_div : flat;
    }
    slot_x_row[tid] = x_row;
  }

  __shared__ _Float16 lds_x[2 * N_TILE * K_GROUP];
  const int groups_per_row = K / K_GROUP;
  const long long row_stride = (long long)groups_per_row * 72;
  const char *A0 = e0 >= 0
                       ? reinterpret_cast<const char *>(expert_weight_ptrs[e0])
                       : nullptr;
  const char *A1 = e1 >= 0
                       ? reinterpret_cast<const char *>(expert_weight_ptrs[e1])
                       : nullptr;
  const char *row0 = A0 ? A0 + (long long)safe_row * row_stride : nullptr;
  const char *row1 = A1 ? A1 + (long long)safe_row * row_stride : nullptr;

  float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const half16_mem_t zero16 = {0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0};

  __syncthreads();
  for (int g = 0; g < groups_per_row; ++g) {
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int slot = sub * N_TILE + warp_id * SLOTS_PER_WARP + (lane >> 3);
      const int k_off = (lane & 7) * 32;
      const int x_row = slot_x_row[slot];
      half16_mem_t va, vb;
      if (x_row >= 0) {
        const long long x_off =
            (long long)x_row * K + g * K_GROUP + k_off;
        va = *reinterpret_cast<const half16_mem_t *>(X_src + x_off);
        vb = *reinterpret_cast<const half16_mem_t *>(X_src + x_off + 16);
      } else {
        va = zero16;
        vb = zero16;
      }
      _Float16 *dst = lds_x + slot * K_GROUP + k_off;
      *reinterpret_cast<half16_mem_t *>(dst) = va;
      *reinterpret_cast<half16_mem_t *>(dst + 16) = vb;
    }
    __syncthreads();

#define DQ8(reg, i, pk, sh)                                                   \
  do {                                                                         \
    const _Float16 qlo = (_Float16)(int)(((pk) >> (sh)) & 1u);                \
    const _Float16 qhi = (_Float16)(int)(((pk) >> ((sh) + 1)) & 1u);          \
    (reg)[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor;             \
  } while (0)

#define LOAD_COEFFS(gp_)                                                       \
  const __half *cb_h = reinterpret_cast<const __half *>(gp_);                 \
  const _Float16 cb0 = (_Float16)__half2float(cb_h[0]);                       \
  const _Float16 cb1 = (_Float16)__half2float(cb_h[1]);                       \
  const _Float16 cb2 = (_Float16)__half2float(cb_h[2]);                       \
  const _Float16 cb3 = (_Float16)__half2float(cb_h[3]);                       \
  const _Float16 d01 = cb1 - cb0;                                             \
  const _Float16 d02 = cb2 - cb0;                                             \
  const _Float16 d_xor = cb3 - cb2 - cb1 + cb0

    if (same) {
      const char *gp = row0 + (long long)g * 72;
      LOAD_COEFFS(gp);
      const _Float16 *lds0 = lds_x + lane_row * K_GROUP;
      const _Float16 *lds1 = lds_x + (N_TILE + lane_row) * K_GROUP;
#pragma unroll
      for (int kt = 0; kt < 16; ++kt) {
        const int k_tile = kt * 16;
        const unsigned int packed =
            *reinterpret_cast<const unsigned int *>(gp + 8 + k_tile / 4);
        const int bit = k_grp * 16;
        half8_t a;
        DQ8(a, 0, packed, bit + 0);
        DQ8(a, 1, packed, bit + 2);
        DQ8(a, 2, packed, bit + 4);
        DQ8(a, 3, packed, bit + 6);
        DQ8(a, 4, packed, bit + 8);
        DQ8(a, 5, packed, bit + 10);
        DQ8(a, 6, packed, bit + 12);
        DQ8(a, 7, packed, bit + 14);
        const half8_t b0 = *reinterpret_cast<const half8_t *>(
            lds0 + k_tile + k_grp * 8);
        const half8_t b1 = *reinterpret_cast<const half8_t *>(
            lds1 + k_tile + k_grp * 8);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b0, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b1, acc1);
      }
    } else {
      if (e0 >= 0) {
        const char *gp = row0 + (long long)g * 72;
        LOAD_COEFFS(gp);
        const _Float16 *lds0 = lds_x + lane_row * K_GROUP;
#pragma unroll
        for (int kt = 0; kt < 16; ++kt) {
          const int k_tile = kt * 16;
          const unsigned int packed =
              *reinterpret_cast<const unsigned int *>(gp + 8 + k_tile / 4);
          const int bit = k_grp * 16;
          half8_t a;
          DQ8(a, 0, packed, bit + 0);
          DQ8(a, 1, packed, bit + 2);
          DQ8(a, 2, packed, bit + 4);
          DQ8(a, 3, packed, bit + 6);
          DQ8(a, 4, packed, bit + 8);
          DQ8(a, 5, packed, bit + 10);
          DQ8(a, 6, packed, bit + 12);
          DQ8(a, 7, packed, bit + 14);
          const half8_t b0 = *reinterpret_cast<const half8_t *>(
              lds0 + k_tile + k_grp * 8);
          acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b0, acc0);
        }
      }
      if (e1 >= 0) {
        const char *gp = row1 + (long long)g * 72;
        LOAD_COEFFS(gp);
        const _Float16 *lds1 = lds_x + (N_TILE + lane_row) * K_GROUP;
#pragma unroll
        for (int kt = 0; kt < 16; ++kt) {
          const int k_tile = kt * 16;
          const unsigned int packed =
              *reinterpret_cast<const unsigned int *>(gp + 8 + k_tile / 4);
          const int bit = k_grp * 16;
          half8_t a;
          DQ8(a, 0, packed, bit + 0);
          DQ8(a, 1, packed, bit + 2);
          DQ8(a, 2, packed, bit + 4);
          DQ8(a, 3, packed, bit + 6);
          DQ8(a, 4, packed, bit + 8);
          DQ8(a, 5, packed, bit + 10);
          DQ8(a, 6, packed, bit + 12);
          DQ8(a, 7, packed, bit + 14);
          const half8_t b1 = *reinterpret_cast<const half8_t *>(
              lds1 + k_tile + k_grp * 8);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b1, acc1);
        }
      }
    }

#undef LOAD_COEFFS
#undef DQ8
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int out_row = warp_m_start + 8 * k_grp + j;
    if (out_row < M) {
      const int col0 = tile0 * N_TILE + lane_row;
      if (e0 >= 0 && col0 < m_total)
        Y_grouped[(long long)col0 * M + out_row] = acc0[j];
      const int col1 = tile1 * N_TILE + lane_row;
      if (e1 >= 0 && col1 < m_total)
        Y_grouped[(long long)col1 * M + out_row] = acc1[j];
    }
  }
}
