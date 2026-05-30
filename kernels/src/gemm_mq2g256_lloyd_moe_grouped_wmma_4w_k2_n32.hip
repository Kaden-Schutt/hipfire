// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// N_TILE=32 (tile-pair) variant of gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2
// for gfx1151 (RDNA3.5). Each block processes TWO consecutive 16-wide slot
// tiles. The MQ2-Lloyd dequant is ~250:1 vs the WMMA it feeds, and the
// baseline kernel rebuilds the per-warp A-fragment for every single WMMA
// (N_TILE=16). Here, when both sub-tiles map to the SAME expert (the common
// case after the expert-sorted scatter), the A-fragment is decoded ONCE and
// fed to TWO WMMAs (one per sub-tile) — halving per-token dequant ALU. At an
// expert boundary (e0 != e1) the two sub-tiles decode independently, so there
// is no regression versus two N=16 blocks.
//
// Scatter is unchanged (BLOCK_M=16); only the dispatch halves grid.y to
// (m_total/16 + 1)/2. Y_grouped output stays slot-indexed → unscatter unchanged.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         64
#define N_TILE         16   // per sub-tile (WMMA N)
#define WARPS          4
#define M_PER_WARP     16
#define K_GROUP        256
#define SLOTS_PER_WARP 4

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int block_m = blockIdx.x * M_TILE;
    const int n_tiles = (m_total + 15) / 16;
    const int tile0   = blockIdx.y * 2;
    const int tile1   = tile0 + 1;
    if (block_m >= M || tile0 >= n_tiles) return;

    const int e0 = expert_tile_ids[tile0];
    const int e1 = (tile1 < n_tiles) ? expert_tile_ids[tile1] : -1;
    if (e0 < 0 && e1 < 0) return;
    const bool same = (e0 >= 0) && (e0 == e1);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3
    const int lane    = tid & 31;
    const int lane16  = lane & 15;  // WMMA fragment lane / weight row within warp

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row       = warp_m_start + lane16;
    const int safe_row     = (my_row < M) ? my_row : (M - 1);

    // 32 slot→x_row mappings: sub-tile 0 = slots [tile0*16, +16),
    // sub-tile 1 = slots [tile0*16+16, +32). sorted_slot_index is contiguous.
    __shared__ int slot_x_row[2 * N_TILE];
    if (tid < 2 * N_TILE) {
        const int slot_idx = tile0 * N_TILE + tid;
        int x_row = -1;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
        slot_x_row[tid] = x_row;
    }

    // LDS X stage: [32 slots × 256 K] = 16 KB per K-group.
    __shared__ _Float16 lds_x[2 * N_TILE * K_GROUP];

    const int groups_per_row   = K / K_GROUP;
    const long long row_stride  = (long long)groups_per_row * 72;
    const char* A0 = (e0 >= 0) ? reinterpret_cast<const char*>(expert_weight_ptrs[e0]) : nullptr;
    const char* A1 = (e1 >= 0) ? reinterpret_cast<const char*>(expert_weight_ptrs[e1]) : nullptr;
    const char* rb0 = A0 ? (A0 + (long long)safe_row * row_stride) : nullptr;
    const char* rb1 = A1 ? (A1 + (long long)safe_row * row_stride) : nullptr;

    float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const half16_t zero_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    __syncthreads();  // slot_x_row visible

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative X load: 128 threads → 32 slots × 256 K (two 16-slot passes).
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const int slot_in_block = sub * N_TILE + warp_id * SLOTS_PER_WARP + (lane >> 3);
            const int k_off         = (lane & 7) * 32;
            const int x_row         = slot_x_row[slot_in_block];
            half16_t va, vb;
            if (x_row >= 0) {
                const long long x_off = (long long)x_row * K + (long long)g * K_GROUP + k_off;
                va = *(const half16_t*)(X_src + x_off);
                vb = *(const half16_t*)(X_src + x_off + 16);
            } else {
                va = zero_b; vb = zero_b;
            }
            _Float16* dst = lds_x + slot_in_block * K_GROUP + k_off;
            *(half16_t*)(dst)      = va;
            *(half16_t*)(dst + 16) = vb;
        }
        __syncthreads();

        // cb0..cb3 codebook → bilinear coeffs (decode the 4-entry MQ2-Lloyd
        // codebook exactly: value = cb0 + qlo*d01 + qhi*d02 + qlo*qhi*d_xor).
        #define LOAD_CB(rb) \
            const char* gp = (rb) + (long long)g * 72; \
            const __half* cb_h = reinterpret_cast<const __half*>(gp); \
            const _Float16 cb0 = (_Float16)__half2float(cb_h[0]); \
            const _Float16 cb1 = (_Float16)__half2float(cb_h[1]); \
            const _Float16 cb2 = (_Float16)__half2float(cb_h[2]); \
            const _Float16 cb3 = (_Float16)__half2float(cb_h[3]); \
            const _Float16 d01 = cb1 - cb0; \
            const _Float16 d02 = cb2 - cb0; \
            const _Float16 d_xor = cb3 - cb2 - cb1 + cb0

        #define DQ(reg, i, pk, sh) do { \
            _Float16 qlo = (_Float16)(int)(((pk) >> (sh))     & 0x1u); \
            _Float16 qhi = (_Float16)(int)(((pk) >> ((sh)+1)) & 0x1u); \
            reg[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor; \
        } while (0)

        #define DECODE_AFRAG(a, k_off_q) do { \
            const unsigned int pk = *(const unsigned int*)(gp + 8 + (k_off_q) / 4); \
            DQ(a,0,pk,0);  DQ(a,1,pk,2);  DQ(a,2,pk,4);  DQ(a,3,pk,6); \
            DQ(a,4,pk,8);  DQ(a,5,pk,10); DQ(a,6,pk,12); DQ(a,7,pk,14); \
            DQ(a,8,pk,16); DQ(a,9,pk,18); DQ(a,10,pk,20); DQ(a,11,pk,22); \
            DQ(a,12,pk,24);DQ(a,13,pk,26); DQ(a,14,pk,28); DQ(a,15,pk,30); \
        } while (0)

        if (same) {
            LOAD_CB(rb0);
            const _Float16* lds0 = lds_x + lane16 * K_GROUP;
            const _Float16* lds1 = lds_x + (N_TILE + lane16) * K_GROUP;
            #pragma unroll
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                half16_t a_a;
                DECODE_AFRAG(a_a, k_off);
                const half16_t b0 = *(const half16_t*)(lds0 + k_off);
                const half16_t b1 = *(const half16_t*)(lds1 + k_off);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b0, acc0);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b1, acc1);
            }
        } else {
            if (e0 >= 0) {
                LOAD_CB(rb0);
                const _Float16* lds0 = lds_x + lane16 * K_GROUP;
                #pragma unroll
                for (int kt = 0; kt < 16; kt++) {
                    const int k_off = kt * 16;
                    half16_t a_a;
                    DECODE_AFRAG(a_a, k_off);
                    const half16_t b0 = *(const half16_t*)(lds0 + k_off);
                    acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b0, acc0);
                }
            }
            if (e1 >= 0) {
                LOAD_CB(rb1);
                const _Float16* lds1 = lds_x + (N_TILE + lane16) * K_GROUP;
                #pragma unroll
                for (int kt = 0; kt < 16; kt++) {
                    const int k_off = kt * 16;
                    half16_t a_a;
                    DECODE_AFRAG(a_a, k_off);
                    const half16_t b1 = *(const half16_t*)(lds1 + k_off);
                    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b1, acc1);
                }
            }
        }
        #undef LOAD_CB
        #undef DQ
        #undef DECODE_AFRAG
        __syncthreads();  // release LDS before next K-group
    }

    // Output writes — same fragment→(row,col) mapping as the baseline, for
    // both sub-tiles. acc[j] on lane L holds C[2*j + (L>>4)][L & 15].
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = warp_m_start + 2 * j + (lane >> 4);
        if (out_row < M) {
            const int oc0 = tile0 * N_TILE + lane16;
            if (e0 >= 0 && oc0 < m_total) Y_grouped[(long long)oc0 * M + out_row] = acc0[j];
            const int oc1 = tile1 * N_TILE + lane16;
            if (e1 >= 0 && oc1 < m_total) Y_grouped[(long long)oc1 * M + out_row] = acc1[j];
        }
    }
}
