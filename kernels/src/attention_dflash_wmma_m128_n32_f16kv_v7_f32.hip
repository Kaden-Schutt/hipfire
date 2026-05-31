// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// v7: M=128 two-pass sub-tiling (§14.4C).
//
// Each block processes 128 query rows as two sequential M=64 sub-tiles
// within the same K-tile loop. K is loaded once per K-tile and shared
// between both sub-tiles (b_reg reused). V is loaded per sub-tile.
//
// Block count halves vs v5 (ceil(B/128) vs ceil(B/64)).
// K traffic: halved. V traffic: unchanged. Net DRAM: ~25% reduction.
//
// LDS: 26.4 KB — V_lds(8KB) + S_lds(16.6KB) + m/l/alpha × 2(1.5KB)
// → 2 WG/CU maintained (26.4 × 2 = 52.8 KB < 64 KB).

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

constexpr int S_LDS_STRIDE = 130;
constexpr int V_tile = 32;
constexpr int SUB_M = 64;

extern "C" __global__ void __launch_bounds__(128, 2)
attention_dflash_wmma_m128_n32_f16kv_v7_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    const int head = blockIdx.x;
    const int qt128 = blockIdx.y;
    const int q128_start = qt128 * 128;
    const int tid = threadIdx.x;

    if (head >= n_heads || q128_start >= B) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int hl = lane & 15;       // half-lane (0..15)
    const int hid = lane >> 4;      // half-id (0 or 1)
    const int my_row_base = wave * 16;

    constexpr int d_chunks = 8;
    constexpr int n_chunks = 8;
    constexpr int n_tile   = 128;

    if (head_dim != 128) return;

    // LDS layout: V_lds + S_lds + (m, l, alpha) × 2 sub-tiles
    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds + V_tile * head_dim);       // 32×128 f16 = 8 KB
    float*    m0_lds = (float*)(S_lds + SUB_M * S_LDS_STRIDE);     // 64×130 f16 = 16.64 KB
    float*    l0_lds = m0_lds + SUB_M;                              // 64×4 = 256 B
    float*    a0_lds = l0_lds + SUB_M;                              // 256 B
    float*    m1_lds = a0_lds + SUB_M;                              // 256 B
    float*    l1_lds = m1_lds + SUB_M;                              // 256 B
    float*    a1_lds = l1_lds + SUB_M;                              // 256 B
    // Total LDS: 8192 + 16640 + 64*4*6 = 25600 bytes = 25.0 KB → 2 WG/CU

    // Load Q for BOTH sub-tiles into registers
    half16_t Q_frags_0[8];
    half16_t Q_frags_1[8];
    {
        int gq0 = q128_start + my_row_base + hl;
        int gq1 = q128_start + SUB_M + my_row_base + hl;
        const float* q_row_0 = (gq0 < B) ? (q + gq0 * q_stride + head * head_dim) : nullptr;
        const float* q_row_1 = (gq1 < B) ? (q + gq1 * q_stride + head * head_dim) : nullptr;
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            if (q_row_0) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags_0[dc][j] = (_Float16)q_row_0[d0 + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags_0[dc][j] = (_Float16)0.0f;
            }
            if (q_row_1) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags_1[dc][j] = (_Float16)q_row_1[d0 + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags_1[dc][j] = (_Float16)0.0f;
            }
        }
    }

    // O_frags for both sub-tiles
    float8_t O_frags_0[8];
    float8_t O_frags_1[8];
    #pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        O_frags_0[dc] = (float8_t){0,0,0,0,0,0,0,0};
        O_frags_1[dc] = (float8_t){0,0,0,0,0,0,0,0};
    }

    // Init m/l for both sub-tiles
    if (lane < 16) {
        m0_lds[my_row_base + lane] = -INFINITY;
        l0_lds[my_row_base + lane] = 0.0f;
        m1_lds[my_row_base + lane] = -INFINITY;
        l1_lds[my_row_base + lane] = 0.0f;
    }
    __syncthreads();

    // === Main K-tile loop ===
    for (int kt_start = 0; kt_start < L; kt_start += n_tile) {

        // ---- Phase A: shared K load, QK for both sub-tiles ----
        float8_t s_acc_0[8];
        float8_t s_acc_1[8];
        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            s_acc_0[c] = (float8_t){0,0,0,0,0,0,0,0};
            s_acc_1[c] = (float8_t){0,0,0,0,0,0,0,0};
        }

        #pragma unroll
        for (int kc = 0; kc < n_chunks; ++kc) {
            int my_k = kt_start + kc * 16 + hl;
            bool k_valid = my_k < L;
            const _Float16* my_k_row = k_valid
                ? (k + my_k * kv_stride + kv_head * head_dim) : nullptr;

            #pragma unroll 4
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16;
                half16_t b_reg;
                if (k_valid) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) b_reg[j] = my_k_row[d0 + j];
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
                }
                s_acc_0[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    Q_frags_0[dc], b_reg, s_acc_0[kc]);
                s_acc_1[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    Q_frags_1[dc], b_reg, s_acc_1[kc]);
            }
        }

        // ---- Process each sub-tile (Phase B + C) ----
        for (int subtile = 0; subtile < 2; ++subtile) {
            float8_t* s_acc   = (subtile == 0) ? s_acc_0 : s_acc_1;
            float8_t* O_frags = (subtile == 0) ? O_frags_0 : O_frags_1;
            float* m_lds = (subtile == 0) ? m0_lds : m1_lds;
            float* l_lds = (subtile == 0) ? l0_lds : l1_lds;
            float* alpha_lds_ptr = (subtile == 0) ? a0_lds : a1_lds;

            // Write S to LDS
            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                int s_col = c * 16 + hl;
                bool col_valid = (kt_start + s_col) < L;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int s_row = my_row_base + 2 * j + hid;
                    float sv = s_acc[c][j] * scale;
                    if (!col_valid) sv = -INFINITY;
                    S_lds[s_row * S_LDS_STRIDE + s_col] = (_Float16)sv;
                }
            }

            // Phase B: softmax
            #pragma unroll 1
            for (int r_local = 0; r_local < 16; ++r_local) {
                int row = my_row_base + r_local;
                _Float16* s_row = S_lds + row * S_LDS_STRIDE;

                float local_max = -INFINITY;
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    local_max = fmaxf(local_max, (float)s_row[lane + kk * 32]);
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
                for (int kk = 0; kk < 4; ++kk) {
                    int idx = lane + kk * 32;
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
                    alpha_lds_ptr[row] = alpha;
                }
            }
            __syncthreads();

            // Alpha-fold O_frags
            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int o_row = my_row_base + 2 * j + hid;
                    O_frags[dc][j] *= alpha_lds_ptr[o_row];
                }
            }

            // Phase C: V staging + SV (4 v_chunks)
            #pragma unroll
            for (int v_chunk = 0; v_chunk < 4; ++v_chunk) {
                int v_start = v_chunk * V_tile;

                // Stage V into V_lds (same as v5)
                for (int kr_local = 0; kr_local < 8; ++kr_local) {
                    int kr = wave * 8 + kr_local;
                    int gk = kt_start + v_start + kr;
                    bool valid = gk < L;
                    const _Float16* v_row = valid
                        ? (v + gk * kv_stride + kv_head * head_dim) : nullptr;
                    for (int d_base = 0; d_base < head_dim; d_base += 32) {
                        int d = d_base + lane;
                        _Float16 vv = v_row ? v_row[d] : (_Float16)0.0f;
                        V_lds[kr * head_dim + d] = vv;
                    }
                }
                __syncthreads();

                // SV accumulation (same as v5)
                #pragma unroll
                for (int c = 0; c < n_chunks; ++c) {
                    int s_col_base = c * 16;
                    if (s_col_base >= v_start && s_col_base < v_start + V_tile) {
                        int s_col_local = s_col_base - v_start;
                        half16_t a_reg_sm;
                        {
                            const _Float16* sm_row = S_lds + (my_row_base + hl) * S_LDS_STRIDE + s_col_base;
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) a_reg_sm[j] = sm_row[j];
                        }
                        #pragma unroll 4
                        for (int dc = 0; dc < d_chunks; ++dc) {
                            int my_d = dc * 16 + hl;
                            half16_t b_reg;
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                b_reg[j] = V_lds[(s_col_local + j) * head_dim + my_d];
                            }
                            O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                a_reg_sm, b_reg, O_frags[dc]);
                        }
                    }
                }
                __syncthreads();
            } // end v_chunk
        } // end subtile
    } // end kt_start

    // Finalize: write O for both sub-tiles
    #pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        int o_col = dc * 16 + hl;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int o_row = my_row_base + 2 * j + hid;
            // Sub-tile 0
            int gq0 = q128_start + o_row;
            if (gq0 < B) {
                float l0 = l0_lds[o_row];
                float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
                out[gq0 * q_stride + head * head_dim + o_col] = O_frags_0[dc][j] * inv_l0;
            }
            // Sub-tile 1
            int gq1 = q128_start + SUB_M + o_row;
            if (gq1 < B) {
                float l1 = l1_lds[o_row];
                float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
                out[gq1 * q_stride + head * head_dim + o_col] = O_frags_1[dc][j] * inv_l1;
            }
        }
    }
}
