// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// cndmask-dequant variant of gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2.
// Identical geometry/LDS/occupancy; the ONLY change is the per-weight MQ2
// dequant. The baseline converts each unpacked bit int→f32→f16 (disasm:
// ~96 v_cvt_f32_ubyte0 + ~96 v_cvt_f16_f32 per 2 WMMA) then multiplies.
// Here the bilinear addends are chosen with f16 SELECTS (cndmask):
//   v = cb0 + (blo?d01:0) + (bhi?d02:0) + ((blo&bhi)?d_xor:0)
// which is bit-exact (blo,bhi ∈ {0,1} ⇒ select == multiply) but drops the
// int→f16 converts and shortens the dequant dependency chain — the right
// lever once profiling showed the kernel is latency/dependency-bound, not
// VALU-throughput-bound. No extra LDS/VGPR, so occupancy is preserved.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         64
#define N_TILE         16   // pinned at 16 (one expert tile)
#define WARPS          4
#define M_PER_WARP     16   // M_TILE / WARPS
#define K_GROUP        256  // MQ2-Lloyd group size
#define SLOTS_PER_WARP 4    // N_TILE / WARPS

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(
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

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row       = warp_m_start + lane16;
    const int safe_row     = (my_row < M) ? my_row : (M - 1);

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

    __shared__ _Float16 lds_x[N_TILE * K_GROUP];

    const int groups_per_row    = K / K_GROUP;
    const long long row_stride  = (long long)groups_per_row * 72;
    const char* row_base        = A + (long long)safe_row * row_stride;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const half16_t zero_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    __syncthreads();

    for (int g = 0; g < groups_per_row; g++) {
        {
            const int slot_in_block = warp_id * SLOTS_PER_WARP + (lane >> 3);
            const int k_off         = (lane & 7) * 32;
            const int x_row         = slot_x_row[slot_in_block];
            half16_t va, vb;
            if (x_row >= 0) {
                const long long x_off = (long long)x_row * K + g * K_GROUP + k_off;
                va = *(const half16_t*)(X_src + x_off);
                vb = *(const half16_t*)(X_src + x_off + 16);
            } else {
                va = zero_b;
                vb = zero_b;
            }
            _Float16* dst = lds_x + slot_in_block * K_GROUP + k_off;
            *(half16_t*)(dst)      = va;
            *(half16_t*)(dst + 16) = vb;
        }
        __syncthreads();

        const char* gp = row_base + g * 72;
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        const _Float16 cb0   = (_Float16)__half2float(cb_h[0]);
        const _Float16 cb1   = (_Float16)__half2float(cb_h[1]);
        const _Float16 cb2   = (_Float16)__half2float(cb_h[2]);
        const _Float16 cb3   = (_Float16)__half2float(cb_h[3]);
        const _Float16 d01   = cb1 - cb0;
        const _Float16 d02   = cb2 - cb0;
        const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;
        const _Float16 zero_h = (_Float16)0.0f;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            const unsigned int pa = *(const unsigned int*)(gp + 8 + k_off_a / 4);
            const unsigned int pb = *(const unsigned int*)(gp + 8 + k_off_b / 4);

            const _Float16* lds_slot = lds_x + lane16 * K_GROUP;
            const half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
            const half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);

            // cndmask dequant: bit-exact bilinear via f16 selects, no int→f16.
            half16_t a_a;
            #define DQ(reg, i, pk, sh) do { \
                const unsigned int blo = ((pk) >> (sh))     & 0x1u; \
                const unsigned int bhi = ((pk) >> ((sh)+1)) & 0x1u; \
                const _Float16 a0 = blo        ? d01   : zero_h; \
                const _Float16 a1 = bhi        ? d02   : zero_h; \
                const _Float16 a2 = (blo & bhi)? d_xor : zero_h; \
                reg[i] = cb0 + a0 + a1 + a2; \
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
        __syncthreads();
    }

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
