// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 grouped-WMMA-GEMM for MoE prefill, gfx1151 (Strix Halo).
//
// The Path 2 (grouped) sister of the Path 1 indexed E8 GEMV. Same SGLang-style
// scatter pipeline + wave32 WMMA 16×16×16 F16 layout as
// `gemm_hfq4g256_moe_grouped_wmma_k2`, but the A operand is dequantized from the
// mfp4-E8 lattice format instead of HFQ4 nibbles. This is the ONLY E8 path that
// AMORTIZES expert-weight reads: each expert weight tile is loaded once per
// 16-row × 16-slot output tile and reused across all 16 slots, vs the indexed
// GEMV which re-reads the weight per (row, token). On the 256 GB/s gfx1151 APU
// the routed-expert GEMVs are memory-bound, so this is the real prefill /
// batched-verify lever.
//
// E8 weight layout (bit-identical decode to gemv_mfp4g32_e8.gfx1151.hip):
//   per row: 16-byte header (f16 row_scale at byte 0) + (K/32) × 17-byte blocks.
//   group  : 256 weights = 8 blocks × 17 B = 136 B (group stride).
//   block  : 1 byte E4M3 block_scale + 4 × 32-bit E8 codewords (8 weights each).
//   value  : row_scale × cvt_e4m3(block_scale) × 0.88 × e8_decode(codeword)[i].
//
// X is the FWHT-rotated activations (x_rot_batch / rot_batch), auto-converted
// f32→f16 by the wrapper via ensure_fp16_x (cached) and read here as f16 for
// the WMMA B operand — same convention as the HFQ4/HFQ6 grouped sisters.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f; // defensive: NaN code -> max finite (never emitted)
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}
#endif

__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mfp4g32_e8_moe_grouped_wmma(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K] f16
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather: each lane (tid & 15) gets its own routed slot; -1 → zero row.
    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB rows to the last valid one; masked store discards.
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const char* row_ptr = A + (long long)safe_row * (16 + (long long)row_block_bytes);
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)row_ptr) ;
    const char* blocks_ptr = row_ptr + 16;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_ptr = blocks_ptr + (long long)g * 136;
        const _Float16* xg = (x_base != nullptr) ? (x_base + (long long)g * 256) : nullptr;

        // 16 WMMA tiles of 16 K-values each = 256 K per group.
        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;
            const int block_idx = k_off >> 5;          // k_off / 32, 0..7
            const char* block_ptr = group_ptr + (long long)block_idx * 17;
            const float ssc = row_scale_f
                * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * 0.88f;
            const int cw_base = (k_off & 31) >> 3;      // 0 or 2 (two 16-tiles per block)
            const unsigned int cw0 = *(const unsigned int*)(block_ptr + 1 + cw_base * 4);
            const unsigned int cw1 = *(const unsigned int*)(block_ptr + 1 + (cw_base + 1) * 4);

            float p0[8], p1[8];
            e8_decode_index(cw0, p0);
            e8_decode_index(cw1, p1);

            half16_t a_a;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_a[i]     = (_Float16)(p0[i] * ssc);
                a_a[8 + i] = (_Float16)(p1[i] * ssc);
            }

            half16_t b = (xg != nullptr) ? *(const half16_t*)(xg + k_off) : zero_b;

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = slot lane (m_total-dim).
    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
