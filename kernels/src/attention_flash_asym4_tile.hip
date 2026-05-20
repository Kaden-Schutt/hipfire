// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile kernel — asym4 KV: K at givens4 (4-bit rotated), V at Q8_0.
//
// Handles head_dim ∈ {128, 256}. For hd=128: one pass. For hd=256: two
// passes over the head, each covering 128 values with its own Givens
// rotation slot. Q is loaded in both halves, rotated per half, and the
// QK dot product sums contributions from both halves before wave-reduce.
//
// V is stored in normal Q8_0 space → V accumulation is in normal space
// (per-thread owns 4 dims for hd=128, or 8 dims for hd=256). The Q8_0
// flash reduce consumes the partials as-is — no inverse rotation needed.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (WAVE32).
// LDS: tile_size floats for scores.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym4_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles
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

    // ── K geometry (givens4 layout) ─────────────────────────────────
    const int k_bytes_per_head = 4 + head_dim / 2;        // 68 (hd128) / 132 (hd256)
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    // ── V geometry (Q8_0 layout) ────────────────────────────────────
    const int v_blocks_per_head = head_dim / 32;          // 4 (hd128) / 8 (hd256)
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // ── Load + rotate Q for each half (2 sets of 4 values for hd=256) ──
    const float* q_head = q + h * head_dim;
    float mq[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float a = q_head[d0 + 0];
        float b = q_head[d0 + 1];
        float c = q_head[d0 + 2];
        float d = q_head[d0 + 3];
        const int block_tid = half * 32 + tid;
        givens_forward(a, b, c, d, cos_theta, sin_theta, block_tid);
        mq[half * 4 + 0] = a;
        mq[half * 4 + 1] = b;
        mq[half * 4 + 2] = c;
        mq[half * 4 + 3] = d;
    }

    // ═══ Phase A: Q·K dot products (K in rotated 4-bit space) ═══
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        float cnorm = *(const float*)kb;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            // Nibble offset: 4 (cnorm) + half * 64 + tid * 2.
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

    // ═══ Phase D: V-weighted accumulation (NORMAL space Q8_0) ═══
    // For hd=128: each thread owns 4 dims (d_base = tid*4).
    // For hd=256: each thread owns 8 dims (d_base = tid*8) — 2 Q8 blocks.
    // We use a loop over "dim pairs of 4" that iterates n_halves times.
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

    // ═══ Write tile partials (output already in NORMAL space) ═══
    // Stride by max_tiles (matches Q8_0 reduce layout).
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        p[2 + d_base + 0] = out_vec[half * 4 + 0];
        p[2 + d_base + 1] = out_vec[half * 4 + 1];
        p[2 + d_base + 2] = out_vec[half * 4 + 2];
        p[2 + d_base + 3] = out_vec[half * 4 + 3];
    }
}
