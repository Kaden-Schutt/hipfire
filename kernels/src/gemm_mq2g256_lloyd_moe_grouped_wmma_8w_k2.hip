// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// 8-warp MoE-grouped MQ2-Lloyd WMMA GEMM for gfx1151 (RDNA3.5).
// Multi-warp port of `gemm_mq2g256_lloyd_moe_grouped_wmma_k2`.
// Mirrors the lever pedapudi proved in llama.cpp #21284 (4-warp MMQ
// tile pattern) and that hipfire's own `gemm_q8_0_wmma_4w` already
// validated at 2.0-3.3× microbench at the equivalent dense shapes.
//
// Per-block geometry:
//   - 256 threads (8 warps × 32 lanes)
//   - Output tile: 128 M-rows × 16 slot-cols = 2048 outputs
//   - Warp_id w owns M-rows [w*16, w*16+16)
//   - The 16 slot-cols are SHARED across all 8 warps via LDS-staged X
//
// Expert-spanning constraint:
//   The slot dim must remain at 16 (= one expert tile) because
//   `expert_id = expert_tile_ids[tile_y]` and one block can only
//   handle one expert's weights. The M-row dim widens to 128 via 8
//   cooperating warps; each warp's weight rows are independent, but
//   the X-slot data is block-shared. This is the lever — 8× less
//   B-fragment memory traffic per FLOP.
//
// LDS:
//   - 8 KB for X stage (16 slots × 256 fp16 per K-group)
//   - 64 B for slot→x_row index cache
//   - Total ≪ 64 KB per-block cap
//
// Wave count vs single-warp baseline:
//   Same total waves (block count / 8 × waves/block × 8 = unchanged).
//   The win is not from fewer waves; it is from each B-fragment load
//   being amortized across 8 warps instead of 1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         128
#define N_TILE         16   // pinned at 16 (one expert tile)
#define WARPS          8
#define M_PER_WARP     16   // M_TILE / WARPS
#define K_GROUP        256  // MQ2-Lloyd group size
#define SLOTS_PER_WARP 2    // N_TILE / WARPS

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / N_TILE]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int block_m   = blockIdx.x * M_TILE;
    const int tile_y    = blockIdx.y;
    const int slot_start = tile_y * N_TILE;

    if (block_m >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3
    const int lane    = tid & 31;   // 0..31
    const int lane16  = lane & 15;  // 0..15 (WMMA fragment lane)

    // Each warp owns 16 M-rows.
    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row       = warp_m_start + lane16;
    const int safe_row     = (my_row < M) ? my_row : (M - 1);

    // Cache the 16 slot→x_row mappings in LDS so every warp shares
    // the lookup without 8× redundant global loads.
    __shared__ int slot_x_row[N_TILE];
    if (tid < N_TILE) {
        const int slot_idx = slot_start + tid;
        int x_row = -1;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) {
                x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
            }
        }
        slot_x_row[tid] = x_row;
    }

    // LDS X stage: [16 slots × 256 K] = 8 KB per K-group.
    __shared__ _Float16 lds_x[N_TILE * K_GROUP];

    const int groups_per_row    = K / K_GROUP;
    const long long row_stride  = (long long)groups_per_row * 72;
    const char* row_base        = A + (long long)safe_row * row_stride;

    // One accumulator per warp: 16-row × 16-col output sub-tile.
    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const half16_t zero_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    __syncthreads();  // make slot_x_row visible to all warps

    for (int g = 0; g < groups_per_row; g++) {
        // ── Cooperative X load: 256 threads → 16 slots × 256 K.
        // Each warp handles 2 contiguous slots:
        //   slot_in_block = warp_id * 2 + (lane / 16)
        // Within the 16-lane group for one slot, lane%16 ∈ {0..15}
        // covers k_off ∈ {0, 16, 32, ..., 240}. Each thread loads
        // one half16_t for 32 B total — coalesced.
        {
            const int slot_in_block = warp_id * SLOTS_PER_WARP + (lane >> 4);
            const int k_off         = (lane & 15) * 16;
            const int x_row         = slot_x_row[slot_in_block];
            half16_t va;
            if (x_row >= 0) {
                const long long x_off = (long long)x_row * K + g * K_GROUP + k_off;
                va = *(const half16_t*)(X_src + x_off);
            } else {
                va = zero_b;
            }
            _Float16* dst = lds_x + slot_in_block * K_GROUP + k_off;
            *(half16_t*)(dst) = va;
        }
        __syncthreads();

        // ── Per-warp: decode 16 rows of weights for this K-group.
        const char* gp = row_base + g * 72;
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        const _Float16 cb0   = (_Float16)__half2float(cb_h[0]);
        const _Float16 cb1   = (_Float16)__half2float(cb_h[1]);
        const _Float16 cb2   = (_Float16)__half2float(cb_h[2]);
        const _Float16 cb3   = (_Float16)__half2float(cb_h[3]);
        // Bilinear-interp decomposition (same as single-warp variant).
        // Avoids the ternary cb[q] form the compiler refuses to vectorize.
        const _Float16 d01   = cb1 - cb0;
        const _Float16 d02   = cb2 - cb0;
        const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Per-warp index loads: 1 uint32 per K-tile (16 weights × 2 bits).
            const unsigned int pa = *(const unsigned int*)(gp + 8 + k_off_a / 4);
            const unsigned int pb = *(const unsigned int*)(gp + 8 + k_off_b / 4);

            // B fragments from LDS — shared across all 4 warps.
            // Each lane L reads slot (L & 15) for K-elements [k_off, k_off+16).
            const _Float16* lds_slot = lds_x + lane16 * K_GROUP;
            const half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
            const half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);

            // Decode A fragment (this warp's 16 weight rows for this K-tile).
            half16_t a_a;
            #define DQ(reg, i, pk, sh) do { \
                _Float16 qlo = (_Float16)(int)(((pk) >> (sh))     & 0x1u); \
                _Float16 qhi = (_Float16)(int)(((pk) >> ((sh)+1)) & 0x1u); \
                reg[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor; \
            } while (0)

            DQ(a_a, 0,  pa,  0); DQ(a_a, 1,  pa,  2); DQ(a_a, 2,  pa,  4); DQ(a_a, 3,  pa,  6);
            DQ(a_a, 4,  pa,  8); DQ(a_a, 5,  pa, 10); DQ(a_a, 6,  pa, 12); DQ(a_a, 7,  pa, 14);
            DQ(a_a, 8,  pa, 16); DQ(a_a, 9,  pa, 18); DQ(a_a, 10, pa, 20); DQ(a_a, 11, pa, 22);
            DQ(a_a, 12, pa, 24); DQ(a_a, 13, pa, 26); DQ(a_a, 14, pa, 28); DQ(a_a, 15, pa, 30);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pb,  0); DQ(a_a, 1,  pb,  2); DQ(a_a, 2,  pb,  4); DQ(a_a, 3,  pb,  6);
            DQ(a_a, 4,  pb,  8); DQ(a_a, 5,  pb, 10); DQ(a_a, 6,  pb, 12); DQ(a_a, 7,  pb, 14);
            DQ(a_a, 8,  pb, 16); DQ(a_a, 9,  pb, 18); DQ(a_a, 10, pb, 20); DQ(a_a, 11, pb, 22);
            DQ(a_a, 12, pb, 24); DQ(a_a, 13, pb, 26); DQ(a_a, 14, pb, 28); DQ(a_a, 15, pb, 30);

            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
        __syncthreads();  // release LDS before next K-group's load
    }

    // ── Output writes.
    // Per RDNA3 wave32 WMMA: acc[j] on lane L holds C[2*j + (L>>4)][L & 15].
    // Each warp owns M-rows [warp_m_start, warp_m_start + 16).
    const int out_col = slot_start + lane16;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = warp_m_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
