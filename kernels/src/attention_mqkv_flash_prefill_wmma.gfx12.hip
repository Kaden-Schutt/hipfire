// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Query-tiled MQKV flash prefill attention for gfx12, head_dim=256.
// Compile with MQ_BITS=3 or MQ_BITS=4. K and V use the existing full-precision
// cnorm + packed Lloyd-centroid record formats; no cache-format change.
//
// One wave handles 16 query rows and 16 key rows per tile. K/V dequantization
// is amortized across the query tile, matching the Q8 WMMA prefill shape.
// The output remains in the V cache's FWHT-256 space; a following linear
// inverse kernel restores normal-space output.
//
// Grid: [ceil(batch_size/16), n_heads]. Block: [32].
#include <hip/hip_runtime.h>
#include "turbo_common.h"

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define M_TILE 16
#define N_TILE 16
#define S_STRIDE 18
#define V_STRIDE 18
#define HEAD_DIM 256
#define D_CHUNKS 16

#if MQ_BITS == 3
#define MQ_ATTN_FN attention_mq3kv_flash_prefill_wmma_gfx12
#define K_BYTES_PER_HEAD 100
#define V_BYTES_PER_HEAD 100
#elif MQ_BITS == 4
#define MQ_ATTN_FN attention_mq4kv_flash_prefill_wmma_gfx12
#define K_BYTES_PER_HEAD 132
#define V_BYTES_PER_HEAD 132
#else
#error "MQ_BITS must be 3 or 4"
#endif

extern "C" __global__ __launch_bounds__(32) void MQ_ATTN_FN(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int n_kv_heads,
    int batch_size,
    float scale_attn
) {
    const int q_tile = blockIdx.x;
    const int h = blockIdx.y;
    if (h >= n_heads) return;

    const int q_base = q_tile * M_TILE;
    if (q_base >= batch_size) return;

    const int lane = threadIdx.x & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int rows = min(M_TILE, batch_size - q_base);
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_stride = n_heads * HEAD_DIM;
    const int k_row_stride = n_kv_heads * K_BYTES_PER_HEAD;
    const int v_row_stride = n_kv_heads * V_BYTES_PER_HEAD;
    const int k_head_off = kv_h * K_BYTES_PER_HEAD;
    const int v_head_off = kv_h * V_BYTES_PER_HEAD;

    extern __shared__ char smem_raw[];
    _Float16* Q_lds = (_Float16*)smem_raw;
    _Float16* V_lds_T = Q_lds + M_TILE * HEAD_DIM;
    _Float16* S_lds = V_lds_T + HEAD_DIM * V_STRIDE;
    float* m_lds = (float*)(S_lds + M_TILE * S_STRIDE);
    float* l_lds = m_lds + M_TILE;
    float* a_lds = l_lds + M_TILE;

    // Rotate each query row in f32, then stage the WMMA operand in f16.
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
#if MQ_BITS == 3
        const int d0 = lane * 8;
        const float* qr = q + (long long)(q_base + min(r, rows - 1)) * q_stride + h * HEAD_DIM;
        float v0 = (r < rows) ? qr[d0 + 0] : 0.0f;
        float v1 = (r < rows) ? qr[d0 + 1] : 0.0f;
        float v2 = (r < rows) ? qr[d0 + 2] : 0.0f;
        float v3 = (r < rows) ? qr[d0 + 3] : 0.0f;
        float v4 = (r < rows) ? qr[d0 + 4] : 0.0f;
        float v5 = (r < rows) ? qr[d0 + 5] : 0.0f;
        float v6 = (r < rows) ? qr[d0 + 6] : 0.0f;
        float v7 = (r < rows) ? qr[d0 + 7] : 0.0f;
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, lane);
        Q_lds[r * HEAD_DIM + d0 + 0] = (_Float16)v0;
        Q_lds[r * HEAD_DIM + d0 + 1] = (_Float16)v1;
        Q_lds[r * HEAD_DIM + d0 + 2] = (_Float16)v2;
        Q_lds[r * HEAD_DIM + d0 + 3] = (_Float16)v3;
        Q_lds[r * HEAD_DIM + d0 + 4] = (_Float16)v4;
        Q_lds[r * HEAD_DIM + d0 + 5] = (_Float16)v5;
        Q_lds[r * HEAD_DIM + d0 + 6] = (_Float16)v6;
        Q_lds[r * HEAD_DIM + d0 + 7] = (_Float16)v7;
#else
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const int d0 = kh * 128 + lane * 4;
            const float* qr = q + (long long)(q_base + min(r, rows - 1)) * q_stride + h * HEAD_DIM;
            float v0 = (r < rows) ? qr[d0 + 0] : 0.0f;
            float v1 = (r < rows) ? qr[d0 + 1] : 0.0f;
            float v2 = (r < rows) ? qr[d0 + 2] : 0.0f;
            float v3 = (r < rows) ? qr[d0 + 3] : 0.0f;
            fwht_shfl_forward(v0, v1, v2, v3, signs1, signs2, lane);
            Q_lds[r * HEAD_DIM + d0 + 0] = (_Float16)v0;
            Q_lds[r * HEAD_DIM + d0 + 1] = (_Float16)v1;
            Q_lds[r * HEAD_DIM + d0 + 2] = (_Float16)v2;
            Q_lds[r * HEAD_DIM + d0 + 3] = (_Float16)v3;
        }
#endif
    }
    if (lane < M_TILE) {
        m_lds[lane] = -INFINITY;
        l_lds[lane] = 0.0f;
        a_lds[lane] = 1.0f;
    }
    __syncthreads();

    float8_t O_frags[D_CHUNKS];
#pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
        O_frags[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    int seq_len = 0;
    for (int r = 0; r < rows; ++r)
        seq_len = max(seq_len, positions[q_base + r] + 1);

    for (int kt = 0; kt < seq_len; kt += N_TILE) {
        const int t_len = min(N_TILE, seq_len - kt);
        const int my_k = kt + half;
        const bool k_ok = half < t_len;
        const unsigned char* kb = k_ok
            ? k_cache + (size_t)my_k * k_row_stride + k_head_off
            : nullptr;
        const float k_cnorm = k_ok ? *(const float*)kb : 0.0f;

        float8_t s_acc = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dc = 0; dc < D_CHUNKS; ++dc) {
            const int d0 = dc * 16 + half_id * 8;
            half8_t a_reg;
            half8_t b_reg;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                a_reg[j] = Q_lds[half * HEAD_DIM + d0 + j];
            if (k_ok) {
#if MQ_BITS == 3
                const unsigned char* base = kb + 4 + (d0 >> 3) * 3;
                const unsigned int packed = (unsigned int)base[0]
                    | ((unsigned int)base[1] << 8)
                    | ((unsigned int)base[2] << 16);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    b_reg[j] = (_Float16)(k_cnorm * TURBO_C3_256[(packed >> (j * 3)) & 7]);
#else
                const int kh = d0 >> 7;
                const int within = d0 & 127;
                const unsigned int packed = *(const unsigned int*)(
                    kb + 4 + kh * 64 + (within >> 2) * 2);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    b_reg[j] = (_Float16)(k_cnorm * TURBO_C4[(packed >> (j * 4)) & 15]);
#endif
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) b_reg[j] = (_Float16)0.0f;
            }
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a_reg, b_reg, s_acc);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = 8 * half_id + j;
            const int c = half;
            const float sv = s_acc[j] * scale_attn;
            const bool valid =
                (r < rows) && (c < t_len) && ((kt + c) <= positions[q_base + r]);
            S_lds[r * S_STRIDE + c] = valid ? (_Float16)sv : (_Float16)(-INFINITY);
        }
        __syncthreads();

        if (lane < rows) {
            _Float16* srow = S_lds + lane * S_STRIDE;
            float tm = -INFINITY;
#pragma unroll
            for (int c = 0; c < N_TILE; ++c)
                tm = fmaxf(tm, (float)srow[c]);
            const float m_old = m_lds[lane];
            const float m_new = fmaxf(m_old, tm);
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < N_TILE; ++c) {
                const float e = (m_new == -INFINITY)
                    ? 0.0f
                    : __expf((float)srow[c] - m_new);
                srow[c] = (_Float16)e;
                sum += e;
            }
            m_lds[lane] = m_new;
            l_lds[lane] = l_lds[lane] * alpha + sum;
            a_lds[lane] = alpha;
        } else if (lane < M_TILE) {
#pragma unroll
            for (int c = 0; c < N_TILE; ++c)
                S_lds[lane * S_STRIDE + c] = (_Float16)0.0f;
            a_lds[lane] = 1.0f;
        }
        __syncthreads();

        const int d_lo = lane * 8;
#pragma unroll
        for (int kr = 0; kr < N_TILE; ++kr) {
            if (kr < t_len) {
                const unsigned char* vb =
                    v_cache + (size_t)(kt + kr) * v_row_stride + v_head_off;
                const float v_cnorm = *(const float*)vb;
#if MQ_BITS == 3
                const unsigned char* base = vb + 4 + lane * 3;
                const unsigned int packed = (unsigned int)base[0]
                    | ((unsigned int)base[1] << 8)
                    | ((unsigned int)base[2] << 16);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] =
                        (_Float16)(v_cnorm * TURBO_C3_256[(packed >> (e * 3)) & 7]);
#else
                const unsigned int packed = *(const unsigned int*)(vb + 4 + lane * 4);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] =
                        (_Float16)(v_cnorm * TURBO_C4_256[(packed >> (e * 4)) & 15]);
#endif
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        half8_t p_reg;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            p_reg[j] = S_lds[half * S_STRIDE + half_id * 8 + j];
#pragma unroll
        for (int dc = 0; dc < D_CHUNKS; ++dc)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                O_frags[dc][j] *= a_lds[8 * half_id + j];

        half8_t v_reg;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v_reg[j] = V_lds_T[half * V_STRIDE + half_id * 8 + j];
#pragma unroll
        for (int dc = 0; dc < D_CHUNKS; ++dc) {
            half8_t v_next;
            if (dc + 1 < D_CHUNKS) {
                const int d_next = (dc + 1) * 16 + half;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    v_next[j] = V_lds_T[d_next * V_STRIDE + half_id * 8 + j];
            }
            O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                p_reg, v_reg, O_frags[dc]);
            if (dc + 1 < D_CHUNKS) v_reg = v_next;
        }
        __syncthreads();
    }

#pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) {
        const int d = dc * 16 + half;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = 8 * half_id + j;
            if (r >= rows) continue;
            const float denom = l_lds[r];
            const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
            out[(long long)(q_base + r) * q_stride + h * HEAD_DIM + d] =
                O_frags[dc][j] * inv;
        }
    }
}
