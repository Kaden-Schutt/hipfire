// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1151 DeepSeek V4 indexer score with cooperative K staging.
//
// The portable WMMA kernel assigns the four wave32 warps in a block to four
// disjoint 16-head tiles. Those warps all consume the same 16-slot x 128-wide
// K tile, but each warp independently loads and converts it from F32 to F16.
// This variant stages that complete tile in LDS once per block. Q conversion,
// WMMA issue order, accumulator layout, ReLU, weighting, and the fixed-order
// head reduction are unchanged.
//
// Grid:  [batch_size, ceil(N_max / 16), 1]
// Block: [128, 1, 1]
// LDS:   16 x 128 x 2 + 64 x 16 x 4 = 8192 B

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define WMMA_TILE 16
#define HEADS_PER_WARP 16
#define WARPS_PER_BLOCK 4
#define MAX_HEADS (HEADS_PER_WARP * WARPS_PER_BLOCK)

__launch_bounds__(128, 2) extern "C" __global__ void
indexer_relu_score_wmma_batched_klds_gfx1151(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ weights,
    const int* __restrict__ n_per_batch,
    float* __restrict__ scores,
    int H,
    int D,
    int N_max,
    int batch_size) {
    const int b = blockIdx.x;
    const int n_block = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = tid & 15;

    if (b >= batch_size) return;
    const int n_start = n_block * WMMA_TILE;
    if (n_start >= N_max) return;

    const int n_valid_b = n_per_batch[b];
    if (n_start >= n_valid_b) {
        if (tid < WMMA_TILE) {
            const int slot = n_start + tid;
            if (slot < N_max) scores[(size_t)b * N_max + slot] = -1e30f;
        }
        return;
    }

    const int h_start = warp_id * HEADS_PER_WARP;
    const int my_h = h_start + lane16;
    const int my_n = n_start + lane16;
    const int safe_h = (my_h < H) ? my_h : (H - 1);
    const float* qh = q + ((size_t)b * H + safe_h) * D;

    // All four warps consume the same K tile. Each of the 128 threads stages
    // one D column for every slot, reducing four identical F32 reads and
    // conversions to one. The complete tile is staged before WMMA so there is
    // no barrier in the eight-step K loop.
    __shared__ _Float16 k_lds[WMMA_TILE][128];
    #pragma unroll
    for (int slot_in_tile = 0; slot_in_tile < WMMA_TILE; ++slot_in_tile) {
        const int slot = n_start + slot_in_tile;
        const int safe_n = (slot < n_valid_b) ? slot : (n_valid_b - 1);
        k_lds[slot_in_tile][tid] = (_Float16)k_cache[(size_t)safe_n * D + tid];
    }
    __syncthreads();

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < D; k += WMMA_TILE) {
        half16_t q_frag;
        half16_t k_frag;
        #pragma unroll
        for (int i = 0; i < WMMA_TILE; ++i) {
            q_frag[i] = (_Float16)qh[k + i];
            k_frag[i] = k_lds[lane16][k + i];
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag, k_frag, acc);
    }

    __shared__ float partials[MAX_HEADS][WMMA_TILE];
    if (lane < WMMA_TILE) {
        #pragma unroll
        for (int hh = 0; hh < HEADS_PER_WARP; ++hh) {
            partials[h_start + hh][lane] = 0.0f;
        }
    }
    __syncthreads();

    const int slot_col = my_n;
    if (slot_col < n_valid_b) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int tile_row = 2 * j + (tid >> 4);
            const int head = h_start + tile_row;
            if (head < H) {
                const float qk = acc[j];
                const float w = weights[(size_t)b * H + head];
                partials[head][lane16] = (qk > 0.0f ? qk : 0.0f) * w;
            }
        }
    }
    __syncthreads();

    if (tid < WMMA_TILE) {
        const int slot = n_start + tid;
        if (slot < n_valid_b) {
            float sum = 0.0f;
            #pragma unroll
            for (int h = 0; h < MAX_HEADS; ++h) {
                if (h < H) sum += partials[h][tid];
            }
            scores[(size_t)b * N_max + slot] = sum;
        } else if (slot < N_max) {
            scores[(size_t)b * N_max + slot] = -1e30f;
        }
    }
}
