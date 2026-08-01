// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
// MQ2-Lloyd grouped-GEMM channel probe for exact gfx1030 via sdot4.
//
// Twin of the gfx1151 i8-WMMA kernel. It is not production-dispatched and does
// not admit A3B MQ2; its direct test method uses the same host contract:
//   - MQ2-Lloyd weights (72 B/group: 4×f16 codebook + 64 B 2-bit indices)
//   - X pre-quantized to block_q8_1_mmq via ensure_q8_1_mmq_x
//   - Grid [ceil(M/16), ceil(m_total/16)], Block [32], LDS 0
//   - Y_grouped layout [m_total × M] f32
//
// Thread map (weight-reuse oriented):
//   row_local = tid & 15          → one of the 16 output rows
//   slot_half = tid >> 4          → 0 covers slots 0..7, 1 covers 8..15
//   acc[s] = Y[slot_start + slot_half*8 + s][row_start + row_local]
//
// B (activation) traffic: lanes 0..15 load one slot's Q8_1 sub-block once
// per (kb, sb). Row threads pull the four slots they own via __shfl from
// source lane (slot_half*8 + s). Streaming MQ2 indices plus bounded inner
// unrolling retains 16-slot weight reuse without the original VGPR spills.
//
// Math: per group, quantize the 4-entry f16 Lloyd codebook to int8 + scale
// once; decode 2-bit indices via byte-extract LUT; sdot4 vs Q8_1 qs;
// `acc += sc * d_x * sumi` (no zp — MQ2-Lloyd is codebook-only).
#if defined(__gfx1030__)
#define HIPFIRE_GFX10_MQ2L_MMQ 1
#else
#define HIPFIRE_GFX10_MQ2L_MMQ 0
#endif
#define QK8_1 32
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");
#if !HIPFIRE_GFX10_MQ2L_MMQ
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}
#else
__device__ __forceinline__ int mq2l_lut4(unsigned int pk, int shift, unsigned int cbp) {
    const unsigned int q = (pk >> shift) & 0xFFu;
    const unsigned int sel =  (q & 0x03u)
                           | ((q & 0x0Cu) << 6)
                           | ((q & 0x30u) << 12)
                           | ((q & 0xC0u) << 18);
    return (int)__builtin_amdgcn_perm(cbp, cbp, sel);
}
__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030(
    const unsigned long long*  __restrict__ expert_weight_ptrs,
    const int*                 __restrict__ expert_tile_ids,
    const int*                 __restrict__ sorted_slot_index,
    const block_q8_1_mmq*      __restrict__ X_q8_src,
    float*                     __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (row_start >= M || slot_start >= m_total) return;
    // Host expert_tile_ids remains one entry per 16-slot routing tile.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    const int row_local = tid & 15;
    const int slot_half = tid >> 4; // 0 → slots 0..7, 1 → slots 8..15
    const int out_row = row_start + row_local;
    const int safe_row = (out_row < M) ? out_row : (M - 1);

    // Only lanes 0..15 own B-cache entries for the sixteen physical slots.
    // All row lanes consume them through wave shuffles below.
    int my_x_row = -1;
    if (tid < 16) {
        const int slot_idx = slot_start + tid;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) {
                my_x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
            }
        }
    }

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        // Quantize this row's 4-entry Lloyd codebook to int8 + scale ONCE.
        const __half* cbh = reinterpret_cast<const __half*>(gp);
        float cb0 = __half2float(cbh[0]);
        float cb1 = __half2float(cbh[1]);
        float cb2 = __half2float(cbh[2]);
        float cb3 = __half2float(cbh[3]);
        float amax = fmaxf(fmaxf(fabsf(cb0), fabsf(cb1)), fmaxf(fabsf(cb2), fabsf(cb3)));
        float sc = amax * (1.0f / 127.0f);
        float inv_sc = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q0 = (int)rintf(cb0 * inv_sc);
        int q1 = (int)rintf(cb1 * inv_sc);
        int q2 = (int)rintf(cb2 * inv_sc);
        int q3 = (int)rintf(cb3 * inv_sc);
        unsigned int cbp = ((unsigned int)(q0 & 0xFF))
                         | ((unsigned int)(q1 & 0xFF) << 8)
                         | ((unsigned int)(q2 & 0xFF) << 16)
                         | ((unsigned int)(q3 & 0xFF) << 24);
        // Stream the two packed index words needed by each sub-block.
        const unsigned int* pkp = reinterpret_cast<const unsigned int*>(gp + 8);
        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                // ── B cache: each lane loads ONE slot's Q8_1 sub-block ──
                float my_dx = 0.f;
                int my_b0=0, my_b1=0, my_b2=0, my_b3=0;
                int my_b4=0, my_b5=0, my_b6=0, my_b7=0;
                if (my_x_row >= 0) {
                    const block_q8_1_mmq* xb =
                        X_q8_src + (long long)kb_q8 * x_src_rows + my_x_row;
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    my_dx = dm_f.x;
                    const int* qs_i = reinterpret_cast<const int*>(xb->qs + sb * 32);
                    my_b0 = qs_i[0]; my_b1 = qs_i[1]; my_b2 = qs_i[2]; my_b3 = qs_i[3];
                    my_b4 = qs_i[4]; my_b5 = qs_i[5]; my_b6 = qs_i[6]; my_b7 = qs_i[7];
                }
                // Broadcast as int bits so __shfl works (float via bitcast).
                int my_dx_bits = __float_as_int(my_dx);

                // Weight side: decode 32 int8 products from 2 × u32 (ONCE per row).
                const int pk_base = kb_in_group * 8 + sb * 2;
                const unsigned int pk0 = pkp[pk_base + 0];
                const unsigned int pk1 = pkp[pk_base + 1];
                const int a0 = mq2l_lut4(pk0, 0,  cbp);
                const int a1 = mq2l_lut4(pk0, 8,  cbp);
                const int a2 = mq2l_lut4(pk0, 16, cbp);
                const int a3 = mq2l_lut4(pk0, 24, cbp);
                const int a4 = mq2l_lut4(pk1, 0,  cbp);
                const int a5 = mq2l_lut4(pk1, 8,  cbp);
                const int a6 = mq2l_lut4(pk1, 16, cbp);
                const int a7 = mq2l_lut4(pk1, 24, cbp);

                // Dot the weight pack against eight activation slots.
                #pragma unroll 4
                for (int s = 0; s < 8; s++) {
                    const int src = slot_half * 8 + s; // 0..15
                    float d_x = __int_as_float(__shfl(my_dx_bits, src));
                    int b0 = __shfl(my_b0, src);
                    int b1 = __shfl(my_b1, src);
                    int b2 = __shfl(my_b2, src);
                    int b3 = __shfl(my_b3, src);
                    int b4 = __shfl(my_b4, src);
                    int b5 = __shfl(my_b5, src);
                    int b6 = __shfl(my_b6, src);
                    int b7 = __shfl(my_b7, src);
                    int sumi = 0;
                    sumi = HIPFIRE_SDOT4_CALL(a0, b0, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a1, b1, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a2, b2, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a3, b3, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a4, b4, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a5, b5, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a6, b6, sumi, false);
                    sumi = HIPFIRE_SDOT4_CALL(a7, b7, sumi, false);
                    acc[s] += sc * d_x * (float)sumi;
                }
            }
        }
    }
    if (out_row < M) {
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const int out_col = slot_start + slot_half * 8 + s;
            if (out_col < m_total) {
                Y_grouped[(long long)out_col * M + out_row] = acc[s];
            }
        }
    }
}
#endif
