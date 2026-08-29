// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile — fwht3 KV V2: K at two independent FWHT-128 3-bit
// halves; V at Q8_0 OR two-half lloyd2/3/4 FWHT-rotated centroid, gated by
// v_mode.
//
// V2 header: two IEEE fp16 cnorms (bytes 0..1 half0, 2..3 half1).
// Per-lane 24-bit K payload: four half-0 indices then four half-1 indices.
// Lloyd rotated-V partials at half*128 + tid*4 + i.
//
// Tuned for head_dim=256 (Qwen 3.5): 32 threads × 4 values × 2 halves.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1]. LDS: tile_size * 4.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht3_tile(
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

    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q (4 values per half) + two independent signed-FWHT-128 forwards.
    const float* q_head = q + h * head_dim;
    float mq[8];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
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

    // Phase A: Q·K (3-bit K, V2). Keep four token dots independent so global-load
    // latency and the five shuffle-reduction steps can overlap, matching the
    // scheduling shape of the tuned Q8 flash tile.
    // Pack order: four half-0 3-bit indices (bits 0..11), then four half-1 (12..23).
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
        const unsigned char* b0 = kb0 + 4 + tid * 3;
        const unsigned char* b1 = kb1 + 4 + tid * 3;
        const unsigned char* b2 = kb2 + 4 + tid * 3;
        const unsigned char* b3 = kb3 + 4 + tid * 3;
        unsigned int pk0 = (unsigned int)b0[0] | ((unsigned int)b0[1] << 8) | ((unsigned int)b0[2] << 16);
        unsigned int pk1 = (unsigned int)b1[0] | ((unsigned int)b1[1] << 8) | ((unsigned int)b1[2] << 16);
        unsigned int pk2 = (unsigned int)b2[0] | ((unsigned int)b2[1] << 8) | ((unsigned int)b2[2] << 16);
        unsigned int pk3 = (unsigned int)b3[0] | ((unsigned int)b3[1] << 8) | ((unsigned int)b3[2] << 16);
        #define DOT3(CN0, CN1, PK) ( \
            mq[0] * ((CN0) * TURBO_C3[(PK)        & 7]) \
          + mq[1] * ((CN0) * TURBO_C3[((PK) >> 3)  & 7]) \
          + mq[2] * ((CN0) * TURBO_C3[((PK) >> 6)  & 7]) \
          + mq[3] * ((CN0) * TURBO_C3[((PK) >> 9)  & 7]) \
          + mq[4] * ((CN1) * TURBO_C3[((PK) >> 12) & 7]) \
          + mq[5] * ((CN1) * TURBO_C3[((PK) >> 15) & 7]) \
          + mq[6] * ((CN1) * TURBO_C3[((PK) >> 18) & 7]) \
          + mq[7] * ((CN1) * TURBO_C3[((PK) >> 21) & 7]))
        float partial0 = DOT3(cn0_0, cn0_1, pk0);
        float partial1 = DOT3(cn1_0, cn1_1, pk1);
        float partial2 = DOT3(cn2_0, cn2_1, pk2);
        float partial3 = DOT3(cn3_0, cn3_1, pk3);
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
        #undef DOT3
    }
    for (; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        const float cn0 = (float)((const _Float16*)kb)[0];
        const float cn1 = (float)((const _Float16*)kb)[1];
        const unsigned char* base = kb + 4 + tid * 3;
        unsigned int packed = (unsigned int)base[0]
                            | ((unsigned int)base[1] << 8)
                            | ((unsigned int)base[2] << 16);
        float partial = mq[0] * (cn0 * TURBO_C3[ packed        & 7])
                      + mq[1] * (cn0 * TURBO_C3[(packed >> 3)  & 7])
                      + mq[2] * (cn0 * TURBO_C3[(packed >> 6)  & 7])
                      + mq[3] * (cn0 * TURBO_C3[(packed >> 9)  & 7])
                      + mq[4] * (cn1 * TURBO_C3[(packed >> 12) & 7])
                      + mq[5] * (cn1 * TURBO_C3[(packed >> 15) & 7])
                      + mq[6] * (cn1 * TURBO_C3[(packed >> 18) & 7])
                      + mq[7] * (cn1 * TURBO_C3[(packed >> 21) & 7]);
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // Phase B: tile max
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // Phase C: exp + sum
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

    // Phase D: V — Q8 unchanged (tid*8 contiguous); Lloyd two-half layout.
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    if (v_mode == 8) {
        // Legacy Q8_0 V (normal space) — unchanged behavior.
        const int d0 = tid * 8;
        float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_blocks_per_head = head_dim / 32;
        const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
        const int v_kv_head_blk     = kv_h * v_blocks_per_head;
        const int v_row_stride      = v_total_blocks * 34;
        int tv = 0;
        for (; tv + 3 < tile_len; tv += 4) {
            float w0 = scores[tv + 0], w1 = scores[tv + 1];
            float w2 = scores[tv + 2], w3 = scores[tv + 3];
            const unsigned char* vb0 = v_cache + (size_t)(tile_start + tv + 0) * v_row_stride;
            const unsigned char* vb1 = v_cache + (size_t)(tile_start + tv + 1) * v_row_stride;
            const unsigned char* vb2 = v_cache + (size_t)(tile_start + tv + 2) * v_row_stride;
            const unsigned char* vb3 = v_cache + (size_t)(tile_start + tv + 3) * v_row_stride;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d % 32;
                const int v_blk_off = (v_kv_head_blk + bi) * 34;
                const unsigned char* v0 = vb0 + v_blk_off;
                const unsigned char* v1 = vb1 + v_blk_off;
                const unsigned char* v2 = vb2 + v_blk_off;
                const unsigned char* v3 = vb3 + v_blk_off;
                float vs0 = (float)*((const _Float16*)v0);
                float vs1 = (float)*((const _Float16*)v1);
                float vs2 = (float)*((const _Float16*)v2);
                float vs3 = (float)*((const _Float16*)v3);
                out_vec[i] += w0 * (vs0 * (float)((signed char)v0[2 + bj]));
                out_vec[i] += w1 * (vs1 * (float)((signed char)v1[2 + bj]));
                out_vec[i] += w2 * (vs2 * (float)((signed char)v2[2 + bj]));
                out_vec[i] += w3 * (vs3 * (float)((signed char)v3[2 + bj]));
            }
        }
        for (; tv < tile_len; tv++) {
            float w = scores[tv];
            const unsigned char* vb = v_cache + (size_t)(tile_start + tv) * v_row_stride;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d % 32;
                const unsigned char* vbh = vb + (v_kv_head_blk + bi) * 34;
                float vs = (float)*((const _Float16*)vbh);
                out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            p[2 + d0 + i] = out_vec[i];
        }
    } else if (v_mode == 3) {
        // Dedicated TURBO_C3 four-token-unrolled Lloyd3 V (no dynamic bit-width branch).
        // Pack: four half-0 indices then four half-1 into 24-bit per-lane payload.
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_bytes_per_head = 4 + (head_dim * 3) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        int tv = 0;
        for (; tv + 3 < tile_len; tv += 4) {
            float w0 = scores[tv + 0], w1 = scores[tv + 1];
            float w2 = scores[tv + 2], w3 = scores[tv + 3];
            const unsigned char* vb0 = v_cache + (size_t)(tile_start + tv + 0) * v_bytes_per_pos + v_head_off;
            const unsigned char* vb1 = v_cache + (size_t)(tile_start + tv + 1) * v_bytes_per_pos + v_head_off;
            const unsigned char* vb2 = v_cache + (size_t)(tile_start + tv + 2) * v_bytes_per_pos + v_head_off;
            const unsigned char* vb3 = v_cache + (size_t)(tile_start + tv + 3) * v_bytes_per_pos + v_head_off;
            float c00 = (float)((const _Float16*)vb0)[0], c01 = (float)((const _Float16*)vb0)[1];
            float c10 = (float)((const _Float16*)vb1)[0], c11 = (float)((const _Float16*)vb1)[1];
            float c20 = (float)((const _Float16*)vb2)[0], c21 = (float)((const _Float16*)vb2)[1];
            float c30 = (float)((const _Float16*)vb3)[0], c31 = (float)((const _Float16*)vb3)[1];
            const unsigned char* base0 = vb0 + 4 + tid * 3;
            const unsigned char* base1 = vb1 + 4 + tid * 3;
            const unsigned char* base2 = vb2 + 4 + tid * 3;
            const unsigned char* base3 = vb3 + 4 + tid * 3;
            unsigned int pk0 = (unsigned int)base0[0] | ((unsigned int)base0[1] << 8) | ((unsigned int)base0[2] << 16);
            unsigned int pk1 = (unsigned int)base1[0] | ((unsigned int)base1[1] << 8) | ((unsigned int)base1[2] << 16);
            unsigned int pk2 = (unsigned int)base2[0] | ((unsigned int)base2[1] << 8) | ((unsigned int)base2[2] << 16);
            unsigned int pk3 = (unsigned int)base3[0] | ((unsigned int)base3[1] << 8) | ((unsigned int)base3[2] << 16);
            #define ACC3(W, CN0, CN1, PK) do { \
                ov[0] += (W) * ((CN0) * TURBO_C3[(PK)        & 7]); \
                ov[1] += (W) * ((CN0) * TURBO_C3[((PK) >> 3)  & 7]); \
                ov[2] += (W) * ((CN0) * TURBO_C3[((PK) >> 6)  & 7]); \
                ov[3] += (W) * ((CN0) * TURBO_C3[((PK) >> 9)  & 7]); \
                ov[4] += (W) * ((CN1) * TURBO_C3[((PK) >> 12) & 7]); \
                ov[5] += (W) * ((CN1) * TURBO_C3[((PK) >> 15) & 7]); \
                ov[6] += (W) * ((CN1) * TURBO_C3[((PK) >> 18) & 7]); \
                ov[7] += (W) * ((CN1) * TURBO_C3[((PK) >> 21) & 7]); \
            } while (0)
            ACC3(w0, c00, c01, pk0);
            ACC3(w1, c10, c11, pk1);
            ACC3(w2, c20, c21, pk2);
            ACC3(w3, c30, c31, pk3);
            #undef ACC3
        }
        for (; tv < tile_len; tv++) {
            float w = scores[tv];
            const unsigned char* vb = v_cache + (size_t)(tile_start + tv) * v_bytes_per_pos + v_head_off;
            const float cn0 = (float)((const _Float16*)vb)[0];
            const float cn1 = (float)((const _Float16*)vb)[1];
            const unsigned char* base = vb + 4 + tid * 3;
            unsigned int packed = (unsigned int)base[0]
                                | ((unsigned int)base[1] << 8)
                                | ((unsigned int)base[2] << 16);
            ov[0] += w * (cn0 * TURBO_C3[ packed        & 7]);
            ov[1] += w * (cn0 * TURBO_C3[(packed >> 3)  & 7]);
            ov[2] += w * (cn0 * TURBO_C3[(packed >> 6)  & 7]);
            ov[3] += w * (cn0 * TURBO_C3[(packed >> 9)  & 7]);
            ov[4] += w * (cn1 * TURBO_C3[(packed >> 12) & 7]);
            ov[5] += w * (cn1 * TURBO_C3[(packed >> 15) & 7]);
            ov[6] += w * (cn1 * TURBO_C3[(packed >> 18) & 7]);
            ov[7] += w * (cn1 * TURBO_C3[(packed >> 21) & 7]);
        }
        // V accumulated in ROTATED space per half. Inverse deferred to lloyd reduce.
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int d_base = half * 128 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                p[2 + d_base + i] = ov[half * 4 + i];
        }
    } else {
        // lloyd2/4: two-half V2 format; one-token-at-a-time is fine for this screen.
        // Bit order: four half-0 indices first, then four half-1 (shift = idx * v_mode).
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
            const unsigned char* base = vb + 4 + tid * v_mode;
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
                    float c = (v_mode == 2)
                        ? TURBO_C2[(packed >> shift) & 3]
                        : TURBO_C4[(packed >> shift) & 15];
                    ov[idx] += w * (cnorm * c);
                }
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int d_base = half * 128 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                p[2 + d_base + i] = ov[half * 4 + i];
        }
    }
}
