// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// FlashAttention WMMA (RDNA3 wave32), M=32 query tile, **N=128 K-tile**,
// K/V f16 in DRAM, V_lds and S_lds in f16. Q and output stay f32.
//
// Compared to attention_dflash_wmma_n64_f16kv.hip:
//   - K-tile width 64 → 128 (s_acc grows from [4] to [8]).
//   - V_lds: 64×head_dim f32 → 128×head_dim f16. Same bytes (32 KB) but
//     twice the K-row capacity, which is what enables N=128.
//   - S_lds: 32×64 f32 (8 KB) → 32×128 f16 (8 KB). Same bytes, twice
//     the K-row range; allows phase A to fit S for the whole N=128
//     tile.
//   - Outer-loop iterations: L/64=305 → L/128=152 at vision shape.
//     Halves __syncthreads, softmax setup, and per-tile alpha-scaling
//     overhead.
//
// LDS budget at head_dim=128:
//   V_lds[128*128] f16 = 32 KB
//   O_lds[32*128]  f32 = 16 KB
//   S_lds[32*128]  f16 =  8 KB
//   scalars (m + l + alpha, 32 each f32) = 384 B
//   Total ≈ 56.4 KB ≤ 64 KB cap.
//
// VGPR budget per lane (rough):
//   Q_frags[8]  half16 = 64
//   s_acc[8]    float8 = 64
//   scratch (a_reg, b_reg, o_acc)     ≈ 32
//   ≈ 160 / lane, under gfx1151 budget.
//
// LDS bank-conflict note for f16 S_lds: each S row is 128 f16 = 256 B
// = 64 dwords. A row stride of 64 dwords means lane-stride-zero on
// dword bank addressing — the same all-bank-zero pattern the f32
// N=64 path had. The hardware coalesces half-word reads from
// adjacent lanes within a single dword, so the practical conflict
// rate stays bounded.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __launch_bounds__(64, 2)
__global__ void attention_dflash_wmma_n128_f16kv_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
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

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int my_row_base = wave * 16;

    constexpr int d_chunks = 8;       // head_dim / 16
    constexpr int n_chunks = 8;       // K-tile / 16 = 128 / 16
    constexpr int n_tile   = 128;
    if (head_dim != 128) return;

    // ─── LDS layout (mixed f16 / f32) ────────────────────────────────
    // Declared as a float* base; we cast to _Float16* for V_lds and S_lds.
    // V_lds and S_lds occupy half their nominal f32 slot count.
    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;                        // 128 * 128 _Float16 = 16K floats worth
    float*    O_lds = (float*)(V_lds + n_tile * head_dim);     //  32 * 128 floats   =  4K floats
    _Float16* S_lds = (_Float16*)(O_lds + 32 * head_dim);      //  32 * 128 _Float16 =  2K floats worth
    float*    m_lds = (float*)(S_lds + 32 * n_tile);
    float*    l_lds = m_lds + 32;
    float*    alpha_lds = l_lds + 32;

    // ─── Load Q into per-lane registers ──────────────────────────────
    half16_t Q_frags[8];
    {
        int gq = q_start + my_row_base + half;
        const float* q_row = (gq < B)
            ? (q + gq * q_stride + head * head_dim)
            : nullptr;
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            if (q_row) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags[dc][j] = (_Float16)q_row[d0 + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags[dc][j] = (_Float16)0.0f;
            }
        }
    }

    // ─── Init O_lds, m_lds, l_lds ────────────────────────────────────
    const int cols_per_half = head_dim / 2;
    const int my_col_base = half_id * cols_per_half;
    for (int row = 0; row < 16; ++row) {
        #pragma unroll
        for (int d = 0; d < cols_per_half; ++d) {
            int idx = (my_row_base + row) * head_dim + my_col_base + d;
            O_lds[idx] = 0.0f;
        }
    }
    if (lane < 16) {
        m_lds[my_row_base + lane] = -INFINITY;
        l_lds[my_row_base + lane] = 0.0f;
    }
    __syncthreads();

    // ─── Loop over K-tiles of width 128 ──────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += n_tile) {

        // ── Phase A: S[16, 128] = Q[16, D] @ K[128, D]^T via WMMA ────
        // 8 K-chunks per outer tile; s_acc[c] holds 16-cols-wide WMMA
        // accumulator (frag_c) for c=0..7.
        float8_t s_acc[8];
        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            s_acc[c] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        #pragma unroll
        for (int kc = 0; kc < n_chunks; ++kc) {
            int my_k = kt_start + kc * 16 + half;
            bool k_valid = my_k < L;
            const _Float16* my_k_row = k_valid
                ? (k + my_k * kv_stride + kv_head * head_dim)
                : nullptr;

            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16;
                half16_t a_reg = Q_frags[dc];

                half16_t b_reg;
                if (k_valid) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) b_reg[j] = my_k_row[d0 + j];
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
                }
                s_acc[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, s_acc[kc]);
            }
        }

        // Write S to LDS (f16), apply scale + mask invalid K columns.
        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            int s_col = c * 16 + half;
            bool col_valid = (kt_start + s_col) < L;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int s_row = my_row_base + 2 * j + half_id;
                float s_val_f32 = s_acc[c][j] * scale;
                if (!col_valid) s_val_f32 = -INFINITY;
                S_lds[s_row * n_tile + s_col] = (_Float16)s_val_f32;
            }
        }

        // Stage V into V_lds (f16) in [k, d] layout. 128 K-rows × hd.
        // Each wave handles 64 K-rows; within a wave, 32 lanes load
        // adjacent dwords across one row at a time. Reading f16 from
        // DRAM, writing f16 to LDS.
        for (int kr_local = 0; kr_local < 64; ++kr_local) {
            int kr = wave * 64 + kr_local;
            int gk = kt_start + kr;
            bool valid = gk < L;
            const _Float16* v_row = valid
                ? (v + gk * kv_stride + kv_head * head_dim)
                : nullptr;
            for (int d_base = 0; d_base < head_dim; d_base += 32) {
                int d = d_base + lane;
                _Float16 vv = v_row ? v_row[d] : (_Float16)0.0f;
                V_lds[kr * head_dim + d] = vv;
            }
        }
        __syncthreads();

        // ── Phase B: per-row online softmax update ───────────────────
        // Each wave handles 16 rows. Lanes 0..15 active; 16..31 idle.
        if (lane < 16) {
            int row = my_row_base + lane;
            _Float16* s_row = S_lds + row * n_tile;
            float tm = -INFINITY;
            #pragma unroll
            for (int j = 0; j < n_tile; ++j) tm = fmaxf(tm, (float)s_row[j]);

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);
            alpha_lds[row] = alpha;

            float ts = 0.0f;
            #pragma unroll
            for (int j = 0; j < n_tile; ++j) {
                float e = expf((float)s_row[j] - m_new);
                s_row[j] = (_Float16)e;
                ts += e;
            }
            l_lds[row] = alpha * l_lds[row] + ts;
            m_lds[row] = m_new;
        }
        __syncthreads();

        // ── Phase C: O = alpha * O + softmax(S) @ V (fused) ──────────
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            float8_t o_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                int s_col_base = c * 16;

                half16_t a_reg_sm;
                {
                    const _Float16* sm_row = S_lds + (my_row_base + half) * n_tile + s_col_base;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) a_reg_sm[j] = sm_row[j];
                }

                half16_t b_reg;
                int my_d = d0 + half;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_reg[j] = V_lds[(s_col_base + j) * head_dim + my_d];
                }

                o_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg_sm, b_reg, o_acc);
            }

            int o_col = d0 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                float alpha = alpha_lds[o_row];
                int o_idx = o_row * head_dim + o_col;
                O_lds[o_idx] = alpha * O_lds[o_idx] + o_acc[j];
            }
        }
        __syncthreads();
    }

    // ─── Finalize: out = O_run / l_run ───────────────────────────────
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
