// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Variant of `attention_dflash_wmma_m64_n128_f16kv.hip` with two
// changes:
//
//   1. **S_lds row stride padded 128 → 130 f16 elements** to break a
//      16-way LDS bank conflict in phase C. With unpadded stride of
//      128 f16 = 256 B = 64 dwords, every lane within a wave hits the
//      *same* bank on each S_lds read of `a_reg_sm[j] = sm_row[j]`
//      (lane stride 64 dwords mod 32 = 0). Padding the row to 130 f16
//      = 260 B gives 65 dwords stride; lane stride mod 32 = 1, so the
//      16 active lanes land in 16 different banks → no conflict.
//      Costs 0.25 KB extra LDS at M=64 (64 rows × 2 extra f16 × 2 B).
//
//   2. **Cooperative wave-32 softmax** — Phase B now uses all 32 lanes
//      per row instead of one lane per row across 16 lanes. Each lane
//      processes 4 values (128 / 32), max + exp + sum are reduced via
//      `__shfl_xor` butterfly. Rows are processed sequentially within
//      a wave (16 rows × ~30 cycles each instead of 16 rows
//      in parallel × ~1700 cycles each on a single lane).
//
// Everything else (M=64, N=128, K/V f16 DRAM, V_lds f16, O in registers,
// 4-wave block, dispatch math) is identical to the v1 kernel.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// Padded S_lds row stride. Must be > n_tile (128) AND give a lane
// bank-stride ≠ 0 mod 32. 130 f16 = 65 dwords, stride mod 32 = 1.
constexpr int S_LDS_STRIDE = 130;

extern "C" __launch_bounds__(128, 1)
__global__ void attention_dflash_wmma_m64_n128_f16kv_v2_f32(
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

    const int wave = tid >> 5;             // 0..3
    const int lane = tid & 31;             // 0..31 within wave
    const int half = lane & 15;            // 0..15
    const int half_id = lane >> 4;
    const int my_row_base = wave * 16;

    constexpr int d_chunks = 8;
    constexpr int n_chunks = 8;
    constexpr int n_tile   = 128;
    constexpr int m_tile   = 64;
    if (head_dim != 128) return;

    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;                                  // 128 × 128 f16
    _Float16* S_lds = (_Float16*)(V_lds + n_tile * head_dim);            // 64 × S_LDS_STRIDE f16
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

        // Phase A: S[16, 128] = Q[16, D] @ K[128, D]^T via WMMA
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

        // Write S to LDS using **padded stride** S_LDS_STRIDE. Apply
        // scale + mask invalid K columns.
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

        // ── Phase B: cooperative wave-32 softmax ─────────────────────
        // Each wave handles its 16 rows sequentially; all 32 lanes
        // cooperate on each row. 128 vals / 32 lanes = 4 vals/lane.
        // Butterfly reduce via __shfl_xor over [1, 2, 4, 8, 16].
        #pragma unroll 1
        for (int r_local = 0; r_local < 16; ++r_local) {
            int row = my_row_base + r_local;
            _Float16* s_row = S_lds + row * S_LDS_STRIDE;

            // Per-lane local max.
            float local_max = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = lane + k * 32;
                local_max = fmaxf(local_max, (float)s_row[idx]);
            }
            // Wave butterfly max.
            local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 4));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 8));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 16));
            float tm = local_max;

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);

            // Per-lane local sum-of-exp (also writes exp back to LDS).
            float local_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = lane + k * 32;
                float e = expf((float)s_row[idx] - m_new);
                s_row[idx] = (_Float16)e;
                local_sum += e;
            }
            // Wave butterfly sum.
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

        // Phase C: O = alpha * O + softmax(S) @ V (fused, O in regs).
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            float8_t o_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                int s_col_base = c * 16;

                half16_t a_reg_sm;
                {
                    const _Float16* sm_row = S_lds + (my_row_base + half) * S_LDS_STRIDE + s_col_base;
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

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                float alpha = alpha_lds[o_row];
                O_frags[dc][j] = alpha * O_frags[dc][j] + o_acc[j];
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
