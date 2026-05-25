// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// FlashAttention WMMA (RDNA3 wave32), M=32 query tile, N=64 K-tile,
// with **K and V already stored as f16 in DRAM** (Q and output remain
// f32). Same algorithmic structure as `attention_dflash_wmma_n64.hip`
// — see that file's header for the full rationale on M=32 / N=64 /
// Q-in-registers / fused alpha-scale.
//
// Why f16 K/V: per the rocprof analysis in
// `docs/plans/dots-ocr.perf-investigation.md` (§1 and §8.3) the
// vision-encoder attention is DRAM-bandwidth-bound (L2 hit < 1 %).
// K and V are read from DRAM ~610 times each (once per query block,
// near-zero L2 reuse). Storing them as f16 halves their DRAM byte
// footprint and halves the attention kernel's DRAM traffic.
//
// Cost: one f32 → f16 cast of K and V before the first attention call
// (single-pass over the QKV-projection outputs). At the vision shape
// (B=L=19520, hd=128, 12 heads) the cast is ~120 MB of work vs ~73 GB
// of K+V DRAM traffic per attention call — amortises to noise.
//
// Differences vs `attention_dflash_wmma_n64.hip`:
//   - `k` and `v` parameters are `const _Float16*` instead of `const
//     float*`.
//   - Phase A K loads drop the f32→f16 cast (b_reg[j] = k_row[d0 + j]
//     directly).
//   - V-stage casts f16→f32 when writing to V_lds (V_lds stays f32 so
//     phase C is identical).
//
// Everything else (LDS layout, Q frag loads, softmax, phase C) is
// byte-identical.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __launch_bounds__(64, 2)
__global__ void attention_dflash_wmma_n64_f16kv_f32(
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

    constexpr int d_chunks = 8;
    if (head_dim != 128) return;

    extern __shared__ float sdata[];
    float* V_lds = sdata;
    float* O_lds = V_lds + 64 * head_dim;
    float* S_lds = O_lds + 32 * head_dim;
    float* m_lds = S_lds + 32 * 64;
    float* l_lds = m_lds + 32;
    float* alpha_lds = l_lds + 32;

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

    // ─── Loop over K-tiles of width 64 ───────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += 64) {

        // Phase A: S[16, 64] = Q[16, D] @ K[64, D]^T via WMMA
        float8_t s_acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            s_acc[c] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        #pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
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

        // Write S to LDS, apply scale + mask invalid K columns.
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int s_col = c * 16 + half;
            bool col_valid = (kt_start + s_col) < L;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int s_row = my_row_base + 2 * j + half_id;
                float s_val = s_acc[c][j] * scale;
                if (!col_valid) s_val = -INFINITY;
                S_lds[s_row * 64 + s_col] = s_val;
            }
        }

        // Stage V into V_lds in [k, d] f32 layout. K-rows split between
        // waves (32 each); within a wave, lanes 0..31 take adjacent
        // dwords across one row at a time. V is read as f16 from DRAM
        // and cast to f32 on the way to LDS.
        for (int kr_local = 0; kr_local < 32; ++kr_local) {
            int kr = wave * 32 + kr_local;
            int gk = kt_start + kr;
            bool valid = gk < L;
            const _Float16* v_row = valid
                ? (v + gk * kv_stride + kv_head * head_dim)
                : nullptr;
            for (int d_base = 0; d_base < head_dim; d_base += 32) {
                int d = d_base + lane;
                float vv = v_row ? (float)v_row[d] : 0.0f;
                V_lds[kr * head_dim + d] = vv;
            }
        }
        __syncthreads();

        // Phase B: per-row online softmax update
        if (lane < 16) {
            int row = my_row_base + lane;
            float* s_row = S_lds + row * 64;
            float tm = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 64; ++j) tm = fmaxf(tm, s_row[j]);

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);
            alpha_lds[row] = alpha;

            float ts = 0.0f;
            #pragma unroll
            for (int j = 0; j < 64; ++j) {
                float e = expf(s_row[j] - m_new);
                s_row[j] = e;
                ts += e;
            }
            l_lds[row] = alpha * l_lds[row] + ts;
            m_lds[row] = m_new;
        }
        __syncthreads();

        // Phase C: O = alpha * O + softmax(S) @ V (fused alpha-scale)
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            float8_t o_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int s_col_base = c * 16;

                half16_t a_reg_sm;
                {
                    const float* sm_row = S_lds + (my_row_base + half) * 64 + s_col_base;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) a_reg_sm[j] = (_Float16)sm_row[j];
                }

                half16_t b_reg;
                int my_d = d0 + half;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_reg[j] = (_Float16)V_lds[(s_col_base + j) * head_dim + my_d];
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
