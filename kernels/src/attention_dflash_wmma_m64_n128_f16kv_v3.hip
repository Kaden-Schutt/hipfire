// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// v3 of attention_dflash_wmma_m64_n128_f16kv — keeps everything from
// v2 (M=64, N=128, K/V f16 DRAM, V_lds + S_lds f16, O register-resident,
// padded S_lds, cooperative softmax) and adds:
//
//   **Hoist S_lds reads out of the phase-C inner d-chunk loop.**
//
// v2's phase C re-reads `a_reg_sm` (the softmax(S) row chunk for K
// cols c*16..c*16+15) eight times per outer K-tile — once per d-chunk
// — even though it only depends on `c`. v3 reorders phase C to outer
// `c`, inner `dc` so S_lds is read once per c. Phase C S_lds reads
// drop from 8 dc × 8 c × 16 = 1024 / lane / outer iter to 8 c × 16 =
// 128 / lane / outer iter. 8× reduction.
//
// The cost: `o_acc` for each d-chunk must live across all 8 c
// iterations (so we accumulate SV across K-chunks before alpha-fold).
// Holding 8 × float8_t = 64 VGPRs / lane. Total VGPR per lane ≈ 220
// (Q_frags 64 + O_frags 64 + o_acc_local 64 + scratch ~30) — right
// at the 256 cap, same as v2.
//
// We also shift the alpha-fold from end-of-phase-C to start-of-phase-C
// so we can accumulate SV directly into the running output without an
// intermediate copy:
//     O_frags = alpha * O_frags + sum_c (softmax(S)_c @ V_c)
//   becomes:
//     O_frags *= alpha;  // start
//     for each c: o_acc_per_dc[dc] += softmax(S)_c @ V_c[dc]
//     for each dc: O_frags[dc] += o_acc_per_dc[dc]
//
// This matches the llama.cpp `fattn-wmma-f16.cu` pattern of holding
// KQ_b once and reusing across d-chunks.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

constexpr int S_LDS_STRIDE = 130;  // 128 + 2 padding to break 16-way bank conflict

extern "C" __launch_bounds__(128, 1)
__global__ void attention_dflash_wmma_m64_n128_f16kv_v3_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    const int head = blockIdx.x;
    const int qt = blockIdx.y;
    const int q_start = qt * 64;
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
    constexpr int n_chunks = 8;
    constexpr int n_tile   = 128;
    constexpr int m_tile   = 64;
    if (head_dim != 128) return;

    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds + n_tile * head_dim);
    float*    m_lds = (float*)(S_lds + m_tile * S_LDS_STRIDE);
    float*    l_lds = m_lds + m_tile;
    float*    alpha_lds = l_lds + m_tile;

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

    // ─── Init O_frags, m_lds, l_lds ──────────────────────────────────
    float8_t O_frags[8];
    #pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        O_frags[dc] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (lane < 16) {
        m_lds[my_row_base + lane] = -INFINITY;
        l_lds[my_row_base + lane] = 0.0f;
    }
    __syncthreads();

    // ─── Loop over K-tiles of width 128 ──────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += n_tile) {

        // Phase A — unchanged from v2.
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

        // Write S to LDS (f16 padded stride).
        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            int s_col = c * 16 + half;
            bool col_valid = (kt_start + s_col) < L;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int s_row = my_row_base + 2 * j + half_id;
                float s_val_f32 = s_acc[c][j] * scale;
                if (!col_valid) s_val_f32 = -INFINITY;
                S_lds[s_row * S_LDS_STRIDE + s_col] = (_Float16)s_val_f32;
            }
        }

        // Stage V into V_lds (f16). 128 K-rows × head_dim.
        for (int kr_local = 0; kr_local < 32; ++kr_local) {
            int kr = wave * 32 + kr_local;
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

        // Phase B — cooperative wave-32 softmax (unchanged from v2).
        #pragma unroll 1
        for (int r_local = 0; r_local < 16; ++r_local) {
            int row = my_row_base + r_local;
            _Float16* s_row = S_lds + row * S_LDS_STRIDE;

            float local_max = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = lane + k * 32;
                local_max = fmaxf(local_max, (float)s_row[idx]);
            }
            local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 4));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 8));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 16));
            float tm = local_max;

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);

            float local_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = lane + k * 32;
                float e = expf((float)s_row[idx] - m_new);
                s_row[idx] = (_Float16)e;
                local_sum += e;
            }
            local_sum += __shfl_xor(local_sum, 1);
            local_sum += __shfl_xor(local_sum, 2);
            local_sum += __shfl_xor(local_sum, 4);
            local_sum += __shfl_xor(local_sum, 8);
            local_sum += __shfl_xor(local_sum, 16);

            if (lane == 0) {
                l_lds[row] = alpha * l_lds[row] + local_sum;
                m_lds[row] = m_new;
                alpha_lds[row] = alpha;
            }
        }
        __syncthreads();

        // ── Phase C v3: alpha-fold at start, then outer c, inner dc ──

        // Step 1: alpha-fold O_frags in place. Each lane reads alpha
        // for its 8 rows (in WMMA frag_c lane layout) from LDS.
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                float alpha = alpha_lds[o_row];
                O_frags[dc][j] *= alpha;
            }
        }

        // Step 2: SV accumulation. For each c, load softmax(S) row
        // chunk ONCE and reuse across all 8 d-chunks.
        // o_acc_per_dc[dc] accumulates SV contributions across c.
        float8_t o_acc_per_dc[8];
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            o_acc_per_dc[dc] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            int s_col_base = c * 16;

            // Load softmax(S)[my_row+half, c*16..c*16+15] ONCE.
            half16_t a_reg_sm;
            {
                const _Float16* sm_row = S_lds + (my_row_base + half) * S_LDS_STRIDE + s_col_base;
                #pragma unroll
                for (int j = 0; j < 16; ++j) a_reg_sm[j] = sm_row[j];
            }

            // 8 SV WMMAs against this a_reg_sm, one per d-chunk.
            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16;

                half16_t b_reg;
                int my_d = d0 + half;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_reg[j] = V_lds[(s_col_base + j) * head_dim + my_d];
                }

                o_acc_per_dc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_reg_sm, b_reg, o_acc_per_dc[dc]);
            }
        }

        // Step 3: accumulate o_acc_per_dc into O_frags.
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_frags[dc][j] += o_acc_per_dc[dc][j];
            }
        }

        __syncthreads();
    }

    // ─── Finalize ────────────────────────────────────────────────────
    #pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        int o_col = dc * 16 + half;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int o_row = my_row_base + 2 * j + half_id;
            int gq = q_start + o_row;
            if (gq < B) {
                float l = l_lds[o_row];
                float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                out[gq * q_stride + head * head_dim + o_col] = O_frags[dc][j] * inv_l;
            }
        }
    }
}
