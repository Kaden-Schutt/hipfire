// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// FlashAttention WMMA (RDNA3 wave32), **M=64 query tile**, N=128 K-tile,
// K/V f16 in DRAM, V_lds and S_lds in f16, **O register-resident** (no
// O_lds — frees the LDS budget the doubled query rows would otherwise
// have eaten).
//
// Compared to attention_dflash_wmma_n128_f16kv.hip:
//   - M widens 32 → 64 queries per block. B/M halves from 610 to 305
//     query blocks per head, which halves K and V DRAM traffic per
//     attention call (each (K, V) is read once per query block —
//     L2 hit < 1 % means no cross-block reuse).
//   - Block size 64 → 128 threads (2 waves → 4 waves). Each wave still
//     owns 16 query rows; the WMMA shape is unchanged.
//   - O moves from O_lds[M × head_dim] f32 to per-lane O_frags[d_chunks]
//     float8_t register arrays. 8 float8_t × 8 d-chunks = 64 VGPRs/lane
//     of persistent running output, in the WMMA frag_c lane layout
//     (lane t holds 8 floats at rows {2*j + (t>>4)} for j=0..7).
//
// LDS budget at head_dim=128:
//   V_lds[128*128] f16 = 32 KB
//   S_lds[64*128]  f16 = 16 KB
//   scalars (m + l + alpha, 64 each f32) = 0.8 KB
//   Total ≈ 48.8 KB ≤ 64 KB cap.
//
// VGPR budget per lane (rough):
//   Q_frags[8]   half16 = 64
//   s_acc[8]     float8 = 64
//   O_frags[8]   float8 = 64
//   scratch (a_reg, b_reg, o_acc) ≈ 30
//   ≈ 220 / lane. RDNA3 wave32 budget allows this with 1 block per CU
//   (4 waves), which is fine on a DRAM-bound workload.
//
// Why this should win: per the rocprof analysis the workload is DRAM
// K+V bound, and halving the query-block count halves K and V DRAM
// traffic per attention call (~73 GB → ~36.5 GB at f16). Theoretical
// floor on Strix Halo LPDDR5X drops from 635 ms to 317 ms.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __launch_bounds__(128, 1)
__global__ void attention_dflash_wmma_m64_n128_f16kv_f32(
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

    const int wave = tid >> 5;             // 0..3 — wave index in 4-wave block
    const int lane = tid & 31;             // 0..31 within wave
    const int half = lane & 15;            // 0..15
    const int half_id = lane >> 4;         // 0 or 1
    const int my_row_base = wave * 16;     // 0, 16, 32, 48

    constexpr int d_chunks = 8;
    constexpr int n_chunks = 8;            // K-tile 128 / 16
    constexpr int n_tile   = 128;
    constexpr int m_tile   = 64;
    if (head_dim != 128) return;

    // ─── LDS layout (no O_lds — O lives in registers) ────────────────
    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;                          // 128 * 128 f16 = 16K f32 slots
    _Float16* S_lds = (_Float16*)(V_lds + n_tile * head_dim);    //  64 * 128 f16 =  4K f32 slots
    float*    m_lds = (float*)(S_lds + m_tile * n_tile);
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
    // O_frags[dc][j] is in WMMA frag_c layout: lane t holds the value at
    // row `my_row_base + 2*j + (t>>4)`, col `dc*16 + (t&15)`.
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

        // Stage V into V_lds (f16). 128 K-rows × head_dim. Each of the
        // 4 waves handles 32 contiguous K-rows. Within a wave, 32 lanes
        // load adjacent dwords across one row at a time.
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

        // Phase B: per-row online softmax update.
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

        // Phase C: O = alpha * O + softmax(S) @ V, with O in registers.
        // Per d-chunk, accumulate o_acc over the 8 K-chunks, then
        // alpha-fold into O_frags[dc] in place.
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

            // Alpha-fold into O_frags[dc] in place.
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                float alpha = alpha_lds[o_row];
                O_frags[dc][j] = alpha * O_frags[dc][j] + o_acc[j];
            }
        }
        __syncthreads();
    }

    // ─── Finalize: out = O_frags / l_run ─────────────────────────────
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
