// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile kernel — fwht4 KV: K at signed-FWHT-rotated 4-bit, V at Q8_0.
//
// Structurally identical to attention_flash_asym4_tile except for the
// Q-rotation primitive — Givens (per-quad cos/sin) replaced with signed-FWHT
// (128-wide butterfly via ds_swizzle_b32). The K cache byte layout is
// byte-identical to asym4 (4 B cnorm + nibble-packed indices), so this
// kernel reads it the same way.
//
// Q is forward-rotated by the same signed-FWHT used at K-write time so the
// dot product in the rotated 4-bit space equals the dot in original space
// (Hadamard is unitary modulo scale; cnorm absorbs the L2 difference).
//
// Handles head_dim ∈ {128, 256}. For hd=128: one 128-element FWHT.
// For hd=256: two independent 128-element FWHTs (same signs reused — per-half
// independence is marginal, matches K-write side).
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (WAVE32).
// LDS: tile_size floats for scores.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht4_tile(
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
    int v_mode          // 8 = legacy Q8_0 V; 2/3/4 = two-half lloyd FWHT-128 centroid V
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
    const int n_halves = head_dim / 128;  // 1 for hd=128, 2 for hd=256

    // ── K geometry (fwht4 layout — byte-identical to asym4) ──────────
    const int k_bytes_per_head = 4 + head_dim / 2;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    // ── V geometry (Q8_0 layout) ────────────────────────────────────
    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // ── Load + FWHT-rotate Q per half ────────────────────────────────
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

    // ═══ Phase A: Q·K dot products (K in rotated 4-bit space) ═══
    int t_local = 0;
    for (; t_local + 3 < tile_len; t_local += 4) {
        const unsigned char* kb0 = k_cache + (size_t)(tile_start + t_local + 0) * k_bytes_per_pos + k_head_off;
        const unsigned char* kb1 = k_cache + (size_t)(tile_start + t_local + 1) * k_bytes_per_pos + k_head_off;
        const unsigned char* kb2 = k_cache + (size_t)(tile_start + t_local + 2) * k_bytes_per_pos + k_head_off;
        const unsigned char* kb3 = k_cache + (size_t)(tile_start + t_local + 3) * k_bytes_per_pos + k_head_off;
        float cn0 = *(const float*)kb0, cn1 = *(const float*)kb1;
        float cn2 = *(const float*)kb2, cn3 = *(const float*)kb3;
        float partial0 = 0.0f, partial1 = 0.0f, partial2 = 0.0f, partial3 = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            unsigned short pk0 = *(const unsigned short*)(kb0 + 4 + half * 64 + tid * 2);
            unsigned short pk1 = *(const unsigned short*)(kb1 + 4 + half * 64 + tid * 2);
            unsigned short pk2 = *(const unsigned short*)(kb2 + 4 + half * 64 + tid * 2);
            unsigned short pk3 = *(const unsigned short*)(kb3 + 4 + half * 64 + tid * 2);
            #define DOT4(P, CN, PK) do { \
                (P) += mq[half * 4 + 0] * ((CN) * TURBO_C4[(PK) & 15]) \
                     + mq[half * 4 + 1] * ((CN) * TURBO_C4[((PK) >> 4) & 15]) \
                     + mq[half * 4 + 2] * ((CN) * TURBO_C4[((PK) >> 8) & 15]) \
                     + mq[half * 4 + 3] * ((CN) * TURBO_C4[((PK) >> 12) & 15]); \
            } while (0)
            DOT4(partial0, cn0, pk0);
            DOT4(partial1, cn1, pk1);
            DOT4(partial2, cn2, pk2);
            DOT4(partial3, cn3, pk3);
            #undef DOT4
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
        float cnorm = *(const float*)kb;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            unsigned short pk = *(const unsigned short*)(kb + 4 + half * 64 + tid * 2);
            partial += mq[half * 4 + 0] * (cnorm * TURBO_C4[(pk)       & 0xF])
                     + mq[half * 4 + 1] * (cnorm * TURBO_C4[(pk >> 4)  & 0xF])
                     + mq[half * 4 + 2] * (cnorm * TURBO_C4[(pk >> 8)  & 0xF])
                     + mq[half * 4 + 3] * (cnorm * TURBO_C4[(pk >> 12) & 0xF]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // ═══ Phase B: tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
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

    // ═══ Phase D: V-weighted accumulation ═══
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
        // ── Two-half lloyd2/3/4 V: per-half FWHT-128 centroid (V2 layout).
        // Header: two fp16 cnorms. Payload: half0 four indices then half1 four
        // in the existing v_mode bytes/lane. Partials at half*128+tid*4+i.
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_bytes_per_head = 4 + (head_dim * v_mode) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            float cn0 = (float)((const _Float16*)vb)[0];
            float cn1 = (float)((const _Float16*)vb)[1];
            const unsigned char* base = vb + 4 + tid * v_mode;   // v_mode bytes/thread
            unsigned int packed = (unsigned int)base[0];
            if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
            if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
            if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
            for (int half = 0; half < n_halves; half++) {
                float cnorm = (half == 0) ? cn0 : cn1;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int idx_i = half * 4 + i;
                    float c;
                    if (v_mode == 2)      c = TURBO_C2[(packed >> (idx_i * 2)) & 3];
                    else if (v_mode == 3) c = TURBO_C3[(packed >> (idx_i * 3)) & 7];
                    else                  c = TURBO_C4[(packed >> (idx_i * 4)) & 15];
                    ov[half * 4 + i] += w * (cnorm * c);
                }
            }
        }
        // V accumulated in ROTATED space per half. Inverse FWHT deferred to
        // lloyd reduce (two independent fwht_shfl_inverse, one per half).
        for (int half = 0; half < n_halves; half++) {
            const int d_base = half * 128 + tid * 4;
            p[2 + d_base + 0] = ov[half * 4 + 0];
            p[2 + d_base + 1] = ov[half * 4 + 1];
            p[2 + d_base + 2] = ov[half * 4 + 2];
            p[2 + d_base + 3] = ov[half * 4 + 3];
        }
    }
}
