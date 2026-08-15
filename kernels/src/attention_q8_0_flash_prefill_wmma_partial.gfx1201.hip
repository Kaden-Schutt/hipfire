// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.
//
// Benchmark-only exact LongSpec partition primitive for Muse Glimmer on gfx1201.
//
// Fork of attention_q8_0_flash_prefill_wmma.gfx12.hip. Preserves HD128 gfx12
// WMMA arithmetic, Q/K/V f16 staging, M_TILE=16, N_TILE=16, causal masks,
// online m/l update, and fixed tuning (SPLIT_Q=0, PREFETCH_V=1, FIXED_HEAD_DIM).
//
// Only the epilogue differs: instead of writing normalized O, each query/head
// record is written as F32 partials[B,H,2+HD] = [m, l, O_unnormalized[HD]].
// Prefix/suffix partials over disjoint KV ranges merge via
// attention_partition_merge_f32 with stable LSE.
//
// Fixed verifier shape: B=16, H=32, KVH=2, HD=128.
// Grid:  [ceil(batch_size / M_TILE), n_heads]  (= [1,32] at fixed B)
// Block: 32 threads (one wave32).

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define M_TILE 16
#define N_TILE 16
#define S_STRIDE 18
#define V_STRIDE 18
#define MAX_D_CHUNKS 16
#ifndef FIXED_HEAD_DIM
#define FIXED_HEAD_DIM 0
#endif
#ifndef SPLIT_Q
#define SPLIT_Q 0
#endif
#ifndef PREFETCH_V
#define PREFETCH_V 1
#endif

extern "C" __global__ __launch_bounds__(32) void attention_q8_0_flash_prefill_wmma_partial_gfx1201(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn
) {
#if FIXED_HEAD_DIM
    if (head_dim != FIXED_HEAD_DIM) return;
#endif
    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    if (h >= n_heads) return;

    const int q_base = q_tile * M_TILE;
    if (q_base >= batch_size) return;

    const int lane    = threadIdx.x & 31;
    const int half    = lane & 15;
    const int half_id = lane >> 4;

    const int bph          = head_dim / 32;
    const int total_blocks = n_kv_heads * bph;
    const int row_stride   = total_blocks * 34;
    const int kv_group     = n_heads / n_kv_heads;
    const int kv_h         = h / kv_group;
    const int kv_blk       = kv_h * bph;
    const int q_stride     = n_heads * head_dim;
#if FIXED_HEAD_DIM
    const int d_chunks = FIXED_HEAD_DIM / 16;
#else
    const int d_chunks = head_dim / 16;
#endif

    extern __shared__ char smem_raw[];
    _Float16* Q_lds = (_Float16*)smem_raw;
#if SPLIT_Q
    _Float16* Q_lo_lds = Q_lds + M_TILE * head_dim;
    _Float16* V_lds_T = Q_lo_lds + M_TILE * head_dim;
#else
    _Float16* V_lds_T = Q_lds + M_TILE * head_dim;
#endif
    _Float16* S_lds = V_lds_T + head_dim * V_STRIDE;
    float*    m_lds = (float*)(S_lds + M_TILE * S_STRIDE);
    float*    l_lds = m_lds + M_TILE;
    float*    a_lds = l_lds + M_TILE;

    const int rows = min(M_TILE, batch_size - q_base);

    for (int i = lane; i < M_TILE * head_dim; i += 32) {
        const int r = i / head_dim;
        const int d = i % head_dim;
        const float qv = (r < rows)
            ? q[(long long)(q_base + r) * q_stride + h * head_dim + d]
            : 0.0f;
        const _Float16 qhi = (_Float16)qv;
        Q_lds[i] = qhi;
#if SPLIT_Q
        Q_lo_lds[i] = (_Float16)(qv - (float)qhi);
#endif
    }
    if (lane < M_TILE) {
        m_lds[lane] = -INFINITY;
        l_lds[lane] = 0.0f;
        a_lds[lane] = 1.0f;
    }
    __syncthreads();

    // gfx12 C mapping: row=8*half_id+j, col=half.
    float8_t O_frags[MAX_D_CHUNKS];
#pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc)
        O_frags[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    int seq_len = 0;
    for (int r = 0; r < rows; ++r)
        seq_len = max(seq_len, positions[q_base + r] + 1);

    for (int kt = 0; kt < seq_len; kt += N_TILE) {
        const int t_len = min(N_TILE, seq_len - kt);

        // S = Q K^T. On gfx12 half_id selects one 8-wide slice of K.
        float8_t s_acc = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        {
            const int my_k = kt + half;
            const bool k_ok = half < t_len;
            const unsigned char* kb =
                k_ok ? (k_cache + (size_t)my_k * row_stride + kv_blk * 34) : nullptr;

#pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                const int d0 = dc * 16 + half_id * 8;
                half8_t a_reg;
                half8_t b_reg;
                for (int j = 0; j < 8; ++j)
                    a_reg[j] = Q_lds[half * head_dim + d0 + j];
#if SPLIT_Q
                half8_t a_lo;
                for (int j = 0; j < 8; ++j)
                    a_lo[j] = Q_lo_lds[half * head_dim + d0 + j];
#endif
                if (k_ok) {
                    const unsigned char* blk = kb + (d0 >> 5) * 34;
                    const float ks = (float)*((const _Float16*)blk);
                    const int off = 2 + (d0 & 31);
                    for (int j = 0; j < 8; ++j)
                        b_reg[j] =
                            (_Float16)(ks * (float)((signed char)blk[off + j]));
                } else {
                    for (int j = 0; j < 8; ++j)
                        b_reg[j] = (_Float16)0.0f;
                }
                s_acc =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_reg, b_reg, s_acc);
#if SPLIT_Q
                s_acc =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_lo, b_reg, s_acc);
#endif
            }
        }

        for (int j = 0; j < 8; ++j) {
            const int r = 8 * half_id + j;
            const int c = half;
            const float sv = s_acc[j] * scale_attn;
            const bool valid =
                (r < rows) && (c < t_len) && ((kt + c) <= positions[q_base + r]);
            S_lds[r * S_STRIDE + c] =
                valid ? (_Float16)sv : (_Float16)(-INFINITY);
        }
        __syncthreads();

        if (lane < rows) {
            _Float16* srow = S_lds + lane * S_STRIDE;
            float tm = -INFINITY;
            for (int c = 0; c < N_TILE; ++c)
                tm = fmaxf(tm, (float)srow[c]);
            const float m_old = m_lds[lane];
            const float m_new = fmaxf(m_old, tm);
            const float alpha =
                (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            float sum = 0.0f;
            for (int c = 0; c < N_TILE; ++c) {
                const float e =
                    (m_new == -INFINITY) ? 0.0f : __expf((float)srow[c] - m_new);
                srow[c] = (_Float16)e;
                sum += e;
            }
            m_lds[lane] = m_new;
            l_lds[lane] = l_lds[lane] * alpha + sum;
            a_lds[lane] = alpha;
        } else if (lane < M_TILE) {
            for (int c = 0; c < N_TILE; ++c)
                S_lds[lane * S_STRIDE + c] = (_Float16)0.0f;
            a_lds[lane] = 1.0f;
        }
        __syncthreads();

        const int dpl = head_dim / 32;
        const int d_lo = lane * dpl;
        const int v_blk = d_lo >> 5;
        const int v_off = 2 + (d_lo & 31);
        for (int kr = 0; kr < N_TILE; ++kr) {
            if (kr < t_len) {
                const unsigned char* vb =
                    v_cache + (size_t)(kt + kr) * row_stride
                    + kv_blk * 34 + v_blk * 34;
                const float vs = (float)*((const _Float16*)vb);
                for (int e = 0; e < dpl; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] =
                        (_Float16)(vs * (float)((signed char)vb[v_off + e]));
            } else {
                for (int e = 0; e < dpl; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        half8_t p_reg;
        for (int j = 0; j < 8; ++j)
            p_reg[j] = S_lds[half * S_STRIDE + half_id * 8 + j];
#pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc)
            for (int j = 0; j < 8; ++j)
                O_frags[dc][j] *= a_lds[8 * half_id + j];

#if PREFETCH_V
        // Double-buffer V fragments so the next LDS read can overlap the
        // current WMMA instead of serializing each read at dscnt=0.
        half8_t v_reg;
        for (int j = 0; j < 8; ++j)
            v_reg[j] = V_lds_T[half * V_STRIDE + half_id * 8 + j];
#pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            half8_t v_next;
            if (dc + 1 < d_chunks) {
                const int d_next = (dc + 1) * 16 + half;
                for (int j = 0; j < 8; ++j)
                    v_next[j] =
                        V_lds_T[d_next * V_STRIDE + half_id * 8 + j];
            }
            O_frags[dc] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    p_reg, v_reg, O_frags[dc]);
            if (dc + 1 < d_chunks)
                v_reg = v_next;
        }
#else
#pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            const int d = dc * 16 + half;
            half8_t v_reg;
            for (int j = 0; j < 8; ++j)
                v_reg[j] = V_lds_T[d * V_STRIDE + half_id * 8 + j];
            O_frags[dc] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    p_reg, v_reg, O_frags[dc]);
        }
#endif
        __syncthreads();
    }

    // Retained online-softmax state + unnormalized O (no 1/l).
    // Record layout: [m, l, O_unnormalized[head_dim]]  stride = 2 + head_dim.
    const int p_stride = 2 + head_dim;
    if (half == 0) {
        for (int j = 0; j < 8; ++j) {
            const int r = 8 * half_id + j;
            if (r >= rows)
                continue;
            float* rec = partials
                + ((long long)(q_base + r) * n_heads + h) * p_stride;
            rec[0] = m_lds[r];
            rec[1] = l_lds[r];
        }
    }
#pragma unroll
    for (int dc = 0; dc < d_chunks; ++dc) {
        const int d = dc * 16 + half;
        for (int j = 0; j < 8; ++j) {
            const int r = 8 * half_id + j;
            if (r >= rows)
                continue;
            float* rec = partials
                + ((long long)(q_base + r) * n_heads + h) * p_stride;
            rec[2 + d] = O_frags[dc][j];
        }
    }
}
