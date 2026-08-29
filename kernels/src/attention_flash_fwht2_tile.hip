// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile for fwht2: K at FWHT-rotated 2-bit (V2 split-norm), V at Q8_0 or Lloyd2/3/4.
// 2-pass over 128-element halves. V2 header = two IEEE fp16 cnorms (half0, half1).
//
// K cache: 4-byte header + half-major 2-bit payload (1 byte per 4 packed values).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht2_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int v_mode          // 8 = legacy Q8_0 V; 2/3/4 = two-half lloyd FWHT-rotated centroid V
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int n_halves = head_dim / 128;

    const int k_bytes_per_head = 4 + head_dim / 4;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + h * head_dim;
    float mq[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float a = q_head[d0 + 0];
        float b = q_head[d0 + 1];
        float c = q_head[d0 + 2];
        float d = q_head[d0 + 3];
        fwht_shfl_forward(a, b, c, d, signs1, signs2, tid);
        mq[half * 4 + 0] = a;
        mq[half * 4 + 1] = b;
        mq[half * 4 + 2] = c;
        mq[half * 4 + 3] = d;
    }

    int t_local = 0;
    for (; t_local + 3 < tile_len; t_local += 4) {
        const unsigned char* kb0 = k_cache + (size_t)(tile_start + t_local + 0) * k_bytes_per_pos + k_head_off;
        const unsigned char* kb1 = k_cache + (size_t)(tile_start + t_local + 1) * k_bytes_per_pos + k_head_off;
        const unsigned char* kb2 = k_cache + (size_t)(tile_start + t_local + 2) * k_bytes_per_pos + k_head_off;
        const unsigned char* kb3 = k_cache + (size_t)(tile_start + t_local + 3) * k_bytes_per_pos + k_head_off;
        float cn0_0 = (float)((const _Float16*)kb0)[0], cn0_1 = (float)((const _Float16*)kb0)[1];
        float cn1_0 = (float)((const _Float16*)kb1)[0], cn1_1 = (float)((const _Float16*)kb1)[1];
        float cn2_0 = (float)((const _Float16*)kb2)[0], cn2_1 = (float)((const _Float16*)kb2)[1];
        float cn3_0 = (float)((const _Float16*)kb3)[0], cn3_1 = (float)((const _Float16*)kb3)[1];
        float partial0 = 0.0f, partial1 = 0.0f, partial2 = 0.0f, partial3 = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const float cn0 = (half == 0) ? cn0_0 : cn0_1;
            const float cn1 = (half == 0) ? cn1_0 : cn1_1;
            const float cn2 = (half == 0) ? cn2_0 : cn2_1;
            const float cn3 = (half == 0) ? cn3_0 : cn3_1;
            unsigned char pk0 = kb0[4 + half * 32 + tid];
            unsigned char pk1 = kb1[4 + half * 32 + tid];
            unsigned char pk2 = kb2[4 + half * 32 + tid];
            unsigned char pk3 = kb3[4 + half * 32 + tid];
            #define DOT2(P, CN, PK) do { \
                (P) += mq[half * 4 + 0] * ((CN) * TURBO_C2[(PK) & 3]) \
                     + mq[half * 4 + 1] * ((CN) * TURBO_C2[((PK) >> 2) & 3]) \
                     + mq[half * 4 + 2] * ((CN) * TURBO_C2[((PK) >> 4) & 3]) \
                     + mq[half * 4 + 3] * ((CN) * TURBO_C2[((PK) >> 6) & 3]); \
            } while (0)
            DOT2(partial0, cn0, pk0);
            DOT2(partial1, cn1, pk1);
            DOT2(partial2, cn2, pk2);
            DOT2(partial3, cn3, pk3);
            #undef DOT2
        }
        for (int off = 16; off > 0; off >>= 1) {
            partial0 += __shfl_xor(partial0, off);
            partial1 += __shfl_xor(partial1, off);
            partial2 += __shfl_xor(partial2, off);
            partial3 += __shfl_xor(partial3, off);
        }
        if (tid == 0) {
            scores[t_local + 0] = partial0 * scale_attn;
            scores[t_local + 1] = partial1 * scale_attn;
            scores[t_local + 2] = partial2 * scale_attn;
            scores[t_local + 3] = partial3 * scale_attn;
        }
    }
    for (; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        const float cn0 = (float)((const _Float16*)kb)[0];
        const float cn1 = (float)((const _Float16*)kb)[1];
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const float cnorm = (half == 0) ? cn0 : cn1;
            unsigned char packed = kb[4 + half * 32 + tid];
            partial += mq[half * 4 + 0] * (cnorm * TURBO_C2[(packed)      & 0x3])
                     + mq[half * 4 + 1] * (cnorm * TURBO_C2[(packed >> 2) & 0x3])
                     + mq[half * 4 + 2] * (cnorm * TURBO_C2[(packed >> 4) & 0x3])
                     + mq[half * 4 + 3] * (cnorm * TURBO_C2[(packed >> 6) & 0x3]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    if (v_mode == 8) {
        // ── Legacy Q8_0 V (normal space, per-half layout) — unchanged ──
        float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
            for (int half = 0; half < n_halves; half++) {
                const int d_base = half * 128 + tid * 4;
                const int v_bi       = d_base / 32;
                const int v_bj_base  = d_base % 32;
                const int v_blk_off  = (v_kv_head_blk + v_bi) * 34;
                const unsigned char* vbh = vb + v_blk_off;
                float vs = (float)*((const _Float16*)vbh);
                out_vec[half * 4 + 0] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 0]));
                out_vec[half * 4 + 1] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 1]));
                out_vec[half * 4 + 2] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 2]));
                out_vec[half * 4 + 3] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 3]));
            }
        }
        for (int half = 0; half < n_halves; half++) {
            const int d_base = half * 128 + tid * 4;
            p[2 + d_base + 0] = out_vec[half * 4 + 0];
            p[2 + d_base + 1] = out_vec[half * 4 + 1];
            p[2 + d_base + 2] = out_vec[half * 4 + 2];
            p[2 + d_base + 3] = out_vec[half * 4 + 3];
        }
    } else {
        // ── Two-half lloyd2/3/4 V (V2): independent fp16 cnorm per 128-wide half.
        // Partials laid out at half*128+tid*4+i for two independent fwht_shfl_inverse.
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_bytes_per_head = 4 + (head_dim * v_mode) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            const float cn_h0 = (float)((const _Float16*)vb)[0];
            const float cn_h1 = (float)((const _Float16*)vb)[1];
            const unsigned char* base = vb + 4 + tid * v_mode;   // v_mode bytes/thread
            unsigned int packed = (unsigned int)base[0];
            if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
            if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
            if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const float cnorm = (half == 0) ? cn_h0 : cn_h1;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int idx = half * 4 + i;
                    const int shift = idx * v_mode;
                    float c;
                    if (v_mode == 2)      c = TURBO_C2[(packed >> shift) & 3];
                    else if (v_mode == 3) c = TURBO_C3[(packed >> shift) & 7];
                    else                  c = TURBO_C4[(packed >> shift) & 15];
                    ov[idx] += w * (cnorm * c);
                }
            }
        }
        // V accumulated in ROTATED space per half. Inverse FWHT deferred to
        // the lloyd reduce (two independent fwht_shfl_inverse, one per half).
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int d_base = half * 128 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                p[2 + d_base + i] = ov[half * 4 + i];
        }
    }
}
