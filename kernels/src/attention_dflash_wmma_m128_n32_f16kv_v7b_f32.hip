// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// v7b: M=128 two-pass sub-tiling, SEQUENTIAL (no K-sharing).
//
// Simpler than v7: each sub-tile is processed independently through all
// K-tiles before starting the next. Only loads Q_frags for one sub-tile
// at a time → same VGPR pressure as v5. No K-sharing.
//
// The only benefit over v5 is halved block count → potentially better
// L2 cache behavior (K+V data from sub-tile 0 may persist in L2 for
// sub-tile 1).
//
// LDS: 25.6 KB (same as v5). 2 WG/CU.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

constexpr int S_LDS_STRIDE = 130;
constexpr int V_tile = 32;
constexpr int SUB_M = 64;

extern "C" __global__ void __launch_bounds__(128, 2)
attention_dflash_wmma_m128_n32_f16kv_v7b_f32(
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
    const int hl = lane & 15;
    const int hid = lane >> 4;
    const int my_row_base = wave * 16;

    constexpr int d_chunks = 8;
    constexpr int n_chunks = 8;
    constexpr int n_tile   = 128;

    if (head_dim != 128) return;

    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds + V_tile * head_dim);
    float*    m_lds = (float*)(S_lds + SUB_M * S_LDS_STRIDE);
    float*    l_lds = m_lds + SUB_M;
    float*    alpha_lds = l_lds + SUB_M;

    // Process two sub-tiles sequentially (same VGPR footprint as v5)
    for (int subtile = 0; subtile < 2; ++subtile) {
        int q_start = q128_start + subtile * SUB_M;

        // Load Q for this sub-tile
        half16_t Q_frags[8];
        {
            int gq = q_start + my_row_base + hl;
            const float* q_row = (gq < B)
                ? (q + gq * q_stride + head * head_dim) : nullptr;
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

        float8_t O_frags[8];
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            O_frags[dc] = (float8_t){0,0,0,0,0,0,0,0};
        }
        if (lane < 16) {
            m_lds[my_row_base + lane] = -INFINITY;
            l_lds[my_row_base + lane] = 0.0f;
        }
        __syncthreads();

        // K-tile loop (identical to v5)
        for (int kt_start = 0; kt_start < L; kt_start += n_tile) {
            float8_t s_acc[8];
            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                s_acc[c] = (float8_t){0,0,0,0,0,0,0,0};
            }

            // Phase A: QK
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
                    s_acc[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        Q_frags[dc], b_reg, s_acc[kc]);
                }
            }

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
                for (int kk = 0; kk < 4; ++kk)
                    local_max = fmaxf(local_max, (float)s_row[lane + kk * 32]);
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
                    alpha_lds[row] = alpha;
                }
            }
            __syncthreads();

            // Alpha-fold O_frags
            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int o_row = my_row_base + 2 * j + hid;
                    O_frags[dc][j] *= alpha_lds[o_row];
                }
            }

            // Phase C: V staging + SV
            #pragma unroll
            for (int v_chunk = 0; v_chunk < 4; ++v_chunk) {
                int v_start = v_chunk * V_tile;
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
                            for (int j = 0; j < 16; ++j)
                                b_reg[j] = V_lds[(s_col_local + j) * head_dim + my_d];
                            O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                a_reg_sm, b_reg, O_frags[dc]);
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Finalize this sub-tile
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int o_col = dc * 16 + hl;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + hid;
                int gq = q_start + o_row;
                if (gq < B) {
                    float l = l_lds[o_row];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    out[gq * q_stride + head * head_dim + o_col] = O_frags[dc][j] * inv_l;
                }
            }
        }
    }
}
