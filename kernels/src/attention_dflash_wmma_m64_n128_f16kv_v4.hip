// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// v4 of attention_dflash_wmma_m64_n128_f16kv — V_lds transpose from v3.
//
// Changes from v3:
// - V_lds transposed from [n_tile][head_dim] to [head_dim][V_T_STRIDE]
// - Phase C reads become contiguous (16 f16 → ds_read_b128 vectorizable)
// - Phase C reads are bank-conflict-free (stride 130 dwords = 65 f16 words,
//   (d*65 + (s_col+j)/2) % 32 varies by lane — no 16-way conflict)
// - V staging writes become scattered across head_dim rows
// - V_T_STRIDE = n_tile + 2 = 130 (same as S_LDS_STRIDE for symmetry)
//
// LDS budget: V_lds_T[128][130] = 33280 bytes + S_lds[64][130] = 16640 bytes +
// m/l/alpha = 768 bytes = 50688 bytes = 49.5 KB → 1 WG/CU (under 64 KB cap)

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

constexpr int S_LDS_STRIDE = 130;  // 128 + 2 padding to break 16-way bank conflict
constexpr int V_T_STRIDE = 130;    // n_tile(128) + 2 padding for bank-conflict-free reads

extern "C" __launch_bounds__(128, 1)
__global__ void attention_dflash_wmma_m64_n128_f16kv_v4_f32(
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
    // V_lds transposed: [head_dim][V_T_STRIDE] instead of [n_tile][head_dim]
    _Float16* V_lds_T = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds_T + head_dim * V_T_STRIDE);
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

        // Phase A — unchanged from v3.
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

            #pragma unroll 4
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

        // Stage V into V_lds_T (transposed layout).
        // V_lds_T[d * V_T_STRIDE + kr] where d ∈ [0, head_dim), kr ∈ [0, n_tile)
        // Write is scattered across head_dim rows, but Phase C reads become
        // contiguous (16 consecutive f16) and bank-conflict-free.
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
                V_lds_T[d * V_T_STRIDE + kr] = vv;  // transposed write
            }
        }
        __syncthreads();

        // Phase B — cooperative wave-32 softmax (unchanged from v3).
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

        // ── Phase C v4: V_lds transposed for vectorized reads ────────

        // Alpha-fold O_frags in place.
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                float alpha = alpha_lds[o_row];
                O_frags[dc][j] *= alpha;
            }
        }

        // SV accumulation with transposed V reads.
        // V_lds_T[my_d * V_T_STRIDE + (s_col_base + j)] — 16 consecutive f16,
        // compiler can vectorize to ds_read_b128 or ds_read_b256.
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
            #pragma unroll 4
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16;
                int my_d = d0 + half;

                half16_t b_reg;
                // Transposed read: 16 consecutive f16 values
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_reg[j] = V_lds_T[my_d * V_T_STRIDE + s_col_base + j];
                }

                O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_reg_sm, b_reg, O_frags[dc]);
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