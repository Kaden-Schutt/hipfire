// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx12/RDNA4 sister of attention_dflash_wmma_m32.hip (M=32, 2-wave).
//
// Same algorithm (the M=16 FlashAttention-2 kernel duplicated per-wave on
// the M-axis: wave 0 owns rows 0..15, wave 1 owns rows 16..31). Only the
// WMMA fragment layout changes for gfx12 — identically to
// attention_dflash_wmma.gfx12.hip: the K=16 contraction is split across the
// two wave-halves (half_id owns K-elements [half_id*8 .. +8)), so A/B
// operands are `half8_t`, loads gain a `half_id*8` offset, the accumulator
// reads out as D[my_row_base + 8*half_id + j, half] (contiguous-per-half)
// instead of gfx11's D[my_row_base + 2*j + half_id, half], and the intrinsic
// is `_w32_gfx12`. The per-wave `my_row_base` offset is preserved verbatim.
//
// See attention_dflash_wmma_m32.hip for the full algorithm commentary.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

extern "C" __launch_bounds__(64, 2)
__global__ void attention_dflash_wmma_m32_f32_gfx12(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    const int head = blockIdx.x;
    const int qt = blockIdx.y;
    const int q_start = qt * 32;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    const int wave = tid >> 5;             // 0 or 1
    const int lane = tid & 31;             // 0..31 within wave
    const int half = lane & 15;            // 0..15, lane-within-half
    const int half_id = lane >> 4;         // 0 (lanes 0..15) or 1 (lanes 16..31)
    const int my_row_base = wave * 16;     // wave 0 → 0, wave 1 → 16

    // ─── LDS layout (identical to gfx11 sibling) ─────────────────────
    extern __shared__ float sdata[];
    float* Q_lds = sdata;
    float* V_lds = Q_lds + 32 * head_dim;
    float* O_lds = V_lds + 16 * head_dim;
    float* S_lds = O_lds + 32 * head_dim;
    float* m_lds = S_lds + 32 * 16;
    float* l_lds = m_lds + 32;
    float* alpha_lds = l_lds + 32;

    // ─── Init Q_lds, O_lds, m_lds, l_lds ─────────────────────────────
    const int cols_per_half = head_dim / 2;
    const int my_col_base = half_id * cols_per_half;
    for (int row = 0; row < 16; ++row) {
        int gq = q_start + my_row_base + row;
        const float* q_row = (gq < B)
            ? (q + gq * q_stride + head * head_dim)
            : nullptr;
        #pragma unroll
        for (int d = 0; d < cols_per_half; ++d) {
            int idx = (my_row_base + row) * head_dim + my_col_base + d;
            Q_lds[idx] = q_row ? q_row[my_col_base + d] : 0.0f;
            O_lds[idx] = 0.0f;
        }
    }
    if (lane < 16) {
        m_lds[my_row_base + lane] = -INFINITY;
        l_lds[my_row_base + lane] = 0.0f;
    }
    __syncthreads();

    // ─── Loop over K-tiles ────────────────────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += 16) {
        int kt_size = L - kt_start;
        if (kt_size > 16) kt_size = 16;

        const int my_k = kt_start + half;
        const bool k_valid = my_k < L;
        const float* my_k_row = k_valid
            ? (k + my_k * kv_stride + kv_head * head_dim)
            : nullptr;

        // ┌─── Phase A: S_tile = Q_tile @ K_tile^T  via WMMA ───────────┐
        //   gfx12: each half_id owns K-elements [half_id*8 .. +8).
        // └─────────────────────────────────────────────────────────────┘
        float8_t s_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            int dk = d0 + half_id * 8;
            half8_t a_reg;
            const float* q_row = Q_lds + (my_row_base + half) * head_dim + dk;
            #pragma unroll
            for (int j = 0; j < 8; ++j) a_reg[j] = (_Float16)q_row[j];

            half8_t b_reg;
            if (k_valid) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) b_reg[j] = (_Float16)my_k_row[dk + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) b_reg[j] = (_Float16)0.0f;
            }
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, s_acc);
        }

        // Write S to LDS (apply scale + mask invalid K columns).
        //   gfx12: s_acc[j] at lane t → S[my_row_base + 8*half_id + j, half]
        const int s_col = half;
        const bool col_valid = (kt_start + s_col) < L;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int s_row = my_row_base + 8 * half_id + j;
            float s_val = s_acc[j] * scale;
            if (!col_valid) s_val = -INFINITY;
            S_lds[s_row * 16 + s_col] = s_val;
        }

        // Stage V into LDS in [k, d] layout (wave 0 only; shared).
        if (wave == 0) {
            const int v_writer_row = half;
            const float* my_v_row = (kt_start + v_writer_row) < L
                ? (v + (kt_start + v_writer_row) * kv_stride + kv_head * head_dim)
                : nullptr;
            for (int d = my_col_base; d < my_col_base + cols_per_half; ++d) {
                float vv = (my_v_row && v_writer_row < kt_size)
                    ? my_v_row[d]
                    : 0.0f;
                V_lds[v_writer_row * head_dim + d] = vv;
            }
        }
        __syncthreads();

        // ┌─── Phase B: per-row online softmax update ──────────────────┐
        if (lane < 16) {
            int row = my_row_base + lane;
            float* s_row = S_lds + row * 16;
            float tm = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 16; ++j) tm = fmaxf(tm, s_row[j]);

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);
            alpha_lds[row] = alpha;

            float ts = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float e = expf(s_row[j] - m_new);
                s_row[j] = e;
                ts += e;
            }
            l_lds[row] = alpha * l_lds[row] + ts;
            m_lds[row] = m_new;
        }
        __syncthreads();

        // ┌─── Phase C: O_run = alpha * O_run + softmax(S) @ V  via WMMA ─┐
        //   gfx12: contraction over the 16 key positions; each half_id
        //   owns key-positions [half_id*8 .. +8).
        // └───────────────────────────────────────────────────────────────┘
        for (int i = lane; i < 16 * head_dim; i += 32) {
            int row = my_row_base + (i / head_dim);
            int col = i % head_dim;
            O_lds[row * head_dim + col] *= alpha_lds[row];
        }
        __syncthreads();

        half8_t a_reg_sm;
        {
            const float* sm_row = S_lds + (my_row_base + half) * 16 + half_id * 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) a_reg_sm[j] = (_Float16)sm_row[j];
        }

        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            half8_t b_reg;
            int my_d = d0 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                b_reg[j] = (_Float16)V_lds[(half_id * 8 + j) * head_dim + my_d];
            }

            float8_t o_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            o_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg_sm, b_reg, o_acc);

            // gfx12: o_acc[j] at lane t → O[my_row_base + 8*half_id + j, d0 + half]
            const int o_col = d0 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 8 * half_id + j;
                O_lds[o_row * head_dim + o_col] += o_acc[j];
            }
        }
        __syncthreads();
    }

    // ─── Finalize: out = O_run / l_run ────────────────────────────────
    for (int i = lane; i < 16 * head_dim; i += 32) {
        int row = my_row_base + (i / head_dim);
        int col = i % head_dim;
        int gq = q_start + row;
        if (gq < B) {
            float l = l_lds[row];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[gq * q_stride + head * head_dim + col] = O_lds[row * head_dim + col] * inv_l;
        }
    }
}
