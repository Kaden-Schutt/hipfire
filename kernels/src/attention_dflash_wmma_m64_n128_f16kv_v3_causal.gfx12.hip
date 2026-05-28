// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire - see LICENSE and NOTICE in the project root.

// gfx12/RDNA4 sibling of attention_dflash_wmma_m64_n128_f16kv_v3_causal.
//
// This is not a macro-only port of the gfx11 source. RDNA4's wave32 WMMA
// fp16 operands carry 8 fp16 values per lane instead of 16, and the C fragment
// maps contiguous rows per lane-group:
//
//   acc[j] = C[8 * (lane >> 4) + j][lane & 15]
//
// The algorithmic shape is unchanged: M=64, N=128, f16 K/V, padded S_lds,
// cooperative softmax, causal tile skip, and phase-C S_lds hoisting.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

constexpr int S_LDS_STRIDE_CAUSAL_GFX12 = 130;

extern "C" __launch_bounds__(128, 1)
__global__ void attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale,
    int is_causal
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
    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int my_row_base = wave * 16;

    constexpr int d_chunks = 8;
    constexpr int n_chunks = 8;
    constexpr int n_tile = 128;
    constexpr int m_tile = 64;
    if (head_dim != 128) return;

    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds + n_tile * head_dim);
    float* m_lds = (float*)(S_lds + m_tile * S_LDS_STRIDE_CAUSAL_GFX12);
    float* l_lds = m_lds + m_tile;
    float* alpha_lds = l_lds + m_tile;

    half8_t Q_frags[8];
    {
        int gq = q_start + my_row_base + m_lane;
        const float* q_row = (gq < B)
            ? (q + gq * q_stride + head * head_dim)
            : nullptr;
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16 + k_grp * 8;
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
        if (is_causal && kt_start >= q_start + m_tile) continue;

        float8_t s_acc[8];
        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            s_acc[c] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        #pragma unroll
        for (int kc = 0; kc < n_chunks; ++kc) {
            int my_k = kt_start + kc * 16 + m_lane;
            bool k_valid = my_k < L;
            const _Float16* my_k_row = k_valid
                ? (k + my_k * kv_stride + kv_head * head_dim)
                : nullptr;

            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16 + k_grp * 8;
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
            int s_col = c * 16 + m_lane;
            bool col_valid = (kt_start + s_col) < L;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int s_row = my_row_base + 8 * k_grp + j;
                float s_val_f32 = s_acc[c][j] * scale;
                if (!col_valid) s_val_f32 = -INFINITY;
                if (is_causal && (kt_start + s_col > q_start + s_row)) s_val_f32 = -INFINITY;
                S_lds[s_row * S_LDS_STRIDE_CAUSAL_GFX12 + s_col] = (_Float16)s_val_f32;
            }
        }

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

        #pragma unroll 1
        for (int r_local = 0; r_local < 16; ++r_local) {
            int row = my_row_base + r_local;
            _Float16* s_row = S_lds + row * S_LDS_STRIDE_CAUSAL_GFX12;

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
                int o_row = my_row_base + 8 * k_grp + j;
                float alpha = alpha_lds[o_row];
                O_frags[dc][j] *= alpha;
            }
        }

        float8_t o_acc_per_dc[8];
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            o_acc_per_dc[dc] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        #pragma unroll
        for (int c = 0; c < n_chunks; ++c) {
            int s_col_base = c * 16;

            half8_t a_reg_sm;
            {
                const _Float16* sm_row =
                    S_lds + (my_row_base + m_lane) * S_LDS_STRIDE_CAUSAL_GFX12 + s_col_base;
                #pragma unroll
                for (int j = 0; j < 8; ++j) a_reg_sm[j] = sm_row[k_grp * 8 + j];
            }

            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16;

                half8_t b_reg;
                int my_d = d0 + m_lane;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    b_reg[j] = V_lds[(s_col_base + k_grp * 8 + j) * head_dim + my_d];
                }

                o_acc_per_dc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_reg_sm, b_reg, o_acc_per_dc[dc]);
            }
        }

        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_frags[dc][j] += o_acc_per_dc[dc][j];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        int o_col = dc * 16 + m_lane;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int o_row = my_row_base + 8 * k_grp + j;
            int gq = q_start + o_row;
            if (gq < B) {
                float l = l_lds[o_row];
                float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                out[gq * q_stride + head * head_dim + o_col] = O_frags[dc][j] * inv_l;
            }
        }
    }
}
