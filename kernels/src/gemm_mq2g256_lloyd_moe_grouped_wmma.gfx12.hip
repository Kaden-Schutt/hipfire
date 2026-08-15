// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) MQ2-Lloyd grouped-WMMA GEMM for MoE prefill.
// gfx12 sister of `gemm_mq2g256_lloyd_moe_grouped_wmma_k2` (gfx11). Until
// now the MQ2-Lloyd grouped family was gfx11-only (_k2 / _4w_k2 / _8w_k2
// variants), so MQ2-Lloyd batched prefill could not run on gfx1201; this
// file closes that hole. Same 9-arg kernarg contract, same scatter
// pipeline (moe_scatter_fused_k8), same unscatter
// (moe_gate_up_unscatter_k8 / moe_down_combine_grouped_k8), same grid.
//
// DISTINCT from `gemm_hfq4g256_moe_grouped_wmma_gfx12` (that kernel is
// affine `sc*q+zp` on 4-bit packs, 136 B/group). This kernel is MQ2-Lloyd:
// 4-entry fp16 codebook + 2-bit direct lookup, 72 B/group.
//
// MQ2-Lloyd-G256 storage layout (72 B/group) — IDENTICAL to the gfx11
// sibling, byte for byte:
//   bytes [0..8)   : 4 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [8..72)  : 64 B packed 2-bit indices — 256 values × 2 bits, LSB-first
//                    (4 indices per byte). Each K-tile (16 weights) uses
//                    exactly 4 B (16 × 2 bits = 32 bits) at 8 + kt*4.
//                    Each k_grp half owns 8 values = the low/high 16 bits of
//                    that dword.
//   16 K-tiles × 4 B/tile = 64 B ≡ bytes [8..72). ✓
//
// gfx11 -> gfx12 port deltas (the four standard rules, mirroring
// `gemm_mq3g256_lloyd_moe_grouped_wmma.gfx12` — the closest reference,
// which differs from this file only in codebook size and index width):
//   1. half16_t -> half8_t operands (8 fp16/lane instead of 16).
//   2. K split across wave halves: k_grp = tid >> 4 selects which half of
//      each 16-wide K-tile this lane owns. The gfx11 lane consumed all 32
//      index bits of a tile (shifts 0,2,…,30); the gfx12 lane consumes its
//      own 16 bits (pre-shift by k_grp*16, then shifts 0,2,…,14). The B
//      operand follows with `xg + kt*16 + k_grp*8`.
//   3. `_gfx12` suffix on the kernel symbol AND on the WMMA builtin
//      (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12).
//   4. 8-row-block accumulator: acc[j] = C[8*k_grp + j][m_lane], vs
//      gfx11's interleaved acc[j] = C[2*j + (tid>>4)][tid & 15].
// Instruction count is unchanged — 16 WMMA per group either way (gfx11:
// 8 iterations × 2; gfx12: 4 iterations × 4) — but each gfx12 lane now
// decodes only 8 weights per WMMA instead of 16, so the K4 unroll is what
// keeps enough independent decode chains in flight to hide the codebook
// FMA latency. This mirrors the MQ3 gfx12 reference (K4) rather than the
// gfx11 `_k2` sibling (K2).
//
// Index-load safety: the dword read at `gp + 8 + kt*4` spans bytes
// [8+kt*4, 12+kt*4). Worst case kt=15 → [68, 72), the last 4 B of the
// group — never one byte past it (contrast the HFQ6 gfx12 kernel, which
// must assemble its trailing 3 B by hand to avoid a VM fault). The address
// is 4 B aligned: the group stride 72 and the row stride
// groups_per_row*72 are both multiples of 8, so `gp + 8` is 8 B aligned.
//
// Dequant: 2D bilinear decomposition of the 4-entry codebook (verbatim
// from the gfx11 sibling, so a weight decodes to the same fp16 bits):
//   q = qhi*2 + qlo, qhi/qlo ∈ {0,1}
//   cb[q] = cb0 + qlo*(cb1-cb0) + qhi*(cb2-cb0) + qhi*qlo*(cb3-cb2-cb1+cb0)
// 3 FMA per weight, fully vectorizable. Critical: avoids the ternary-chain
// cb[q] formulation, which the compiler refuses to vectorize → no WMMA
// emitted (verified via objdump 2026-05-22).
//
// Per-lane codebook: each m_lane uses its OWN weight row's codebook
// (safe_row = row_start + m_lane), so the __shared__ LDS broadcast used by
// some gfx11 variants is WRONG for the grouped MoE contract — 16 different
// A-rows per tile means 16 different codebooks. Registers it is.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    // Sentinel early-return: dispatcher may launch up to m_total_max/16
    // tiles to skip the m_total dtoh sync; scatter pre-fills the upper-
    // bound range of expert_tile_ids with -1.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X-row gather (per m_lane; both k_grp lanes for the same m_lane read
    // the same source row). -1 padding lanes contribute a zero B row.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to the last valid M-row (the
    // masked output write below discards the result anyway).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;   // 72 B/group

    // This lane's 16-bit slice of every K-tile's 32-bit index dword.
    const int kshift = k_grp * 16;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        // Per-lane codebook: 4 fp16 entries from this lane's weight row.
        // Each m_lane has a DIFFERENT safe_row → different codebook.
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
        _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
        _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
        _Float16 cb3 = (_Float16)__half2float(cb_h[3]);

        // 2D bilinear coefficients (exact for any 4-entry table).
        const _Float16 d01   = cb1 - cb0;                 // qlo bit
        const _Float16 d02   = cb2 - cb0;                 // qhi bit
        const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;     // qhi & qlo

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // K4 unroll: 4 K-tiles per inner body. Each lane reads the K-tile's
        // whole 4 B index dword (aligned, in-bounds) and keeps its own
        // 16-bit half via `>> kshift`; the residual high bits for k_grp==0
        // are never referenced because every shift below is ≤ 14 and masked
        // with 0x1u.
        for (int kt = 0; kt < 16; kt += 4) {
            unsigned int pa = (*(const unsigned int*)(gp + 8 + (kt + 0) * 4)) >> kshift;
            unsigned int pb = (*(const unsigned int*)(gp + 8 + (kt + 1) * 4)) >> kshift;
            unsigned int pc = (*(const unsigned int*)(gp + 8 + (kt + 2) * 4)) >> kshift;
            unsigned int pd = (*(const unsigned int*)(gp + 8 + (kt + 3) * 4)) >> kshift;

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
                b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
                b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
                b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);
            } else { b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b; }

            // 2D bilinear decode per element: qlo=(idx>>0)&1, qhi=(idx>>1)&1.
            // `(_Float16)(int)` is an element-wise convert; compiler vectorizes.
            #define DQ2(d, pk, sh) do { \
                _Float16 qlo = (_Float16)(int)(((pk) >> (sh))     & 0x1u); \
                _Float16 qhi = (_Float16)(int)(((pk) >> ((sh)+1)) & 0x1u); \
                (d) = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor; \
            } while (0)

            // Each half8_t covers 8 weights (one k_grp's half of a K-tile).
            #define DQ2_8(d, pk) \
                DQ2((d)[0], (pk),  0); DQ2((d)[1], (pk),  2); \
                DQ2((d)[2], (pk),  4); DQ2((d)[3], (pk),  6); \
                DQ2((d)[4], (pk),  8); DQ2((d)[5], (pk), 10); \
                DQ2((d)[6], (pk), 12); DQ2((d)[7], (pk), 14)

            half8_t a_reg;
            DQ2_8(a_reg, pa); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ2_8(a_reg, pb); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ2_8(a_reg, pc); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ2_8(a_reg, pd); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
            #undef DQ2_8
            #undef DQ2
        }
    }

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Direct write (=) — Y_grouped is fresh per call.
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
