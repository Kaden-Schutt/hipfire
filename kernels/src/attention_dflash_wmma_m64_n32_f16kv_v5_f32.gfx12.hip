// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12/RDNA4 sister of attention_dflash_wmma_m64_n32_f16kv_v5_f32.hip.
// The algorithm is the same v5 dots.ocr winner, but gfx12 WMMA uses half8_t
// operands and `_w32_gfx12`; each lane group owns one half of the K tile.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

constexpr int S_LDS_STRIDE = 130;
constexpr int V_tile = 32;

extern "C" __global__ void __launch_bounds__(128, 2)
attention_dflash_wmma_m64_n32_f16kv_v5_f32_gfx12(
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
    if (head_dim != 128) return;

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
    constexpr int n_tile = 128;
    constexpr int m_tile = 64;

    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds + V_tile * head_dim);
    float* m_lds = (float*)(S_lds + m_tile * S_LDS_STRIDE);
    float* l_lds = m_lds + m_tile;
    float* alpha_lds = l_lds + m_tile;

    half8_t Q_frags[8];
    {
        int gq = q_start + my_row_base + half;
        const float* q_row = (gq < B) ? (q + gq * q_stride + head * head_dim) : nullptr;
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16 + half_id * 8;
            if (q_row) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) Q_frags[dc][j] = (_Float16)q_row[d0 + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) Q_frags[dc][j] = (_Float16)0.0f;
            }
        }
    }

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

    for (int kt_start = 0; kt_start < L; kt_start += n_tile) {
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
                int d0 = dc * 16 + half_id * 8;
                half8_t a_reg = Q_frags[dc];

                half8_t b_reg;
                if (k_valid) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) b_reg[j] = my_k_row[d0 + j];
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) b_reg[j] = (_Float16)0.0f;
                }
                s_acc[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, s_acc[kc]);
            }
        }

        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            int s_col = c * 16 + half;
            bool col_valid = (kt_start + s_col) < L;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int s_row = my_row_base + 8 * half_id + j;
                float s_val_f32 = s_acc[c][j] * scale;
                if (!col_valid) s_val_f32 = -INFINITY;
                S_lds[s_row * S_LDS_STRIDE + s_col] = (_Float16)s_val_f32;
            }
        }

        #pragma unroll 1
        for (int r_local = 0; r_local < 16; ++r_local) {
            int row = my_row_base + r_local;
            _Float16* s_row = S_lds + row * S_LDS_STRIDE;

            float local_max = -INFINITY;
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                int idx = lane + kk * 32;
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

        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 8 * half_id + j;
                O_frags[dc][j] *= alpha_lds[o_row];
            }
        }

        #pragma unroll
        for (int v_chunk = 0; v_chunk < 4; ++v_chunk) {
            int v_start = v_chunk * V_tile;

            for (int kr_local = 0; kr_local < 8; ++kr_local) {
                int kr = wave * 8 + kr_local;
                int gk = kt_start + v_start + kr;
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

            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                int s_col_base = c * 16;

                if (s_col_base >= v_start && s_col_base < v_start + V_tile) {
                    int s_col_local = s_col_base - v_start;

                    half8_t a_reg_sm;
                    {
                        const _Float16* sm_row =
                            S_lds + (my_row_base + half) * S_LDS_STRIDE + s_col_base + half_id * 8;
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) a_reg_sm[j] = sm_row[j];
                    }

                    #pragma unroll 4
                    for (int dc = 0; dc < d_chunks; ++dc) {
                        int d0 = dc * 16;

                        half8_t b_reg;
                        int my_d = d0 + half;
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            b_reg[j] = V_lds[(s_col_local + half_id * 8 + j) * head_dim + my_d];
                        }

                        O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                            a_reg_sm, b_reg, O_frags[dc]);
                    }
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        int o_col = dc * 16 + half;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int o_row = my_row_base + 8 * half_id + j;
            int gq = q_start + o_row;
            if (gq < B) {
                float l_sum = l_lds[o_row];
                float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                out[gq * q_stride + head * head_dim + o_col] = O_frags[dc][j] * inv_l;
            }
        }
    }
}
