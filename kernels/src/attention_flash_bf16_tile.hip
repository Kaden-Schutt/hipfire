// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Flash attention tile kernel for FLAT BF16 KV, with a sliding-window mask.
// Decode / single-position sibling of `attention_flash_q8_0_tile.hip`.
//
// This is a direct port of that kernel with the Q8_0 block addressing removed:
// where Q8 reads a 34-byte block (fp16 scale + 32 int8) and multiplies by the
// scale, this reads 2-byte bf16 values straight out of a flat
// `[pos][kv_head][d]` array and widens them by a 16-bit shift.
//
// WHAT IS DELIBERATELY IDENTICAL to the Q8 kernel, and must stay so:
//   * the per-thread `d` mapping in Phase A (`k_bi = half*4 + tid/8`,
//     `k_off = (tid%8)*4`) and in Phase D (`d_base = half*128 + tid*4`);
//   * the order of the FMA accumulation and the descending XOR reduction;
//   * the partials layout `[n_heads][max_tiles][2 + head_dim]` and the
//     `max_tiles` (NOT n_tiles) row stride.
// The first two keep a bf16-vs-q8 comparison a measurement of the STORAGE
// tier rather than of a different summation order. The third is what lets
// this share `attention_flash_q8_0_reduce` unmodified — that kernel only ever
// touches f32 partials, so it is KV-dtype-agnostic and there is no bf16
// reduce kernel.
//
// The gfx1151 DPP reduction variant from the Q8 kernel is NOT ported. It is an
// opt-in micro-optimisation behind HIPFIRE_GFX1151_ATTENTION_TILE_DPP and is
// wired only to the Q8 module name; adding it here would be an unmeasured
// second code path on a tier whose whole purpose is to be a reference.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (one WAVE32).
// LDS: tile_size floats for scores + head_dim floats for Q.
#include <hip/hip_runtime.h>

// Widen one bf16 (as raw bits) to f32. BF16 is exactly the top 16 bits of an
// f32, so this is a shift — no table, no rounding, exact.
static __device__ __forceinline__ float hipfire_bf16_to_f32(unsigned short b) {
    return __builtin_bit_cast(float, ((unsigned int)b) << 16);
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_bf16_tile(
    const float* __restrict__ q,          // [n_heads, head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,         // [n_heads, max_tiles, 2 + head_dim]
    const int* __restrict__ pos_buf,      // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int window                            // sliding-window span; <= 0 = full causal
) {
    const int seq_len = pos_buf[0] + 1;
    // Sliding-window lower bound: keys at t < win_lo are outside the window
    // [seq_len-window, seq_len) and masked. window <= 0 = full causal.
    const int win_lo = (window > 0) ? (seq_len - window) : 0;
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

    // Flat bf16: one element per (pos, kv_head, d), stride kv_dim per position.
    const int kv_dim = n_kv_heads * head_dim;
    const unsigned short* k16 = (const unsigned short*)k_cache;
    const unsigned short* v16 = (const unsigned short*)v_cache;
    const int head_base = kv_h * head_dim;

    // Number of 128-element halves spanning head_dim. One pass per half in
    // Phase A (Q·K) and Phase D (V) so a 32-thread wave covers the full
    // head_dim regardless of size.
    const int n_halves = (head_dim + 127) / 128;

    // LDS layout: [tile_size scores][head_dim Q values]
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* q_lds  = sdata + tile_size;

    // Load Q into LDS. 32 threads × 4 values × n_halves = head_dim values.
    const float* q_head = q + h * head_dim;
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        q_lds[d_base + 0] = q_head[d_base + 0];
        q_lds[d_base + 1] = q_head[d_base + 1];
        q_lds[d_base + 2] = q_head[d_base + 2];
        q_lds[d_base + 3] = q_head[d_base + 3];
    }
    __syncthreads();

    // ═══ Phase A: compute tile_len dot products (wave-cooperative) ═══
    // Thread assignment mirrors the Q8 kernel exactly: within a 128-element
    // half, lane `tid` owns the 4 dims at `(tid/8)*32 + (tid%8)*4`. In Q8
    // terms that is block `bi = tid/8`, byte offset `(tid%8)*4` inside it;
    // here it is just a flat dim offset, but the mapping is preserved so the
    // two kernels sum in the same order.
    const int k_off = (tid % 8) * 4;

    int valid_start = win_lo - tile_start;
    if (valid_start < 0) valid_start = 0;
    if (valid_start > tile_len) valid_start = tile_len;
    for (int t_local = 0; t_local < valid_start; t_local++) {
        if (tid == 0) scores[t_local] = -INFINITY;
    }

    int t_local = valid_start;
    for (; t_local + 3 < tile_len; t_local += 4) {
        const int t0 = tile_start + t_local + 0;
        const int t1 = tile_start + t_local + 1;
        const int t2 = tile_start + t_local + 2;
        const int t3 = tile_start + t_local + 3;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int d_start = (half * 4 + tid / 8) * 32 + k_off;
            const int elem = head_base + d_start;
            const unsigned short* kb0 = k16 + (size_t)t0 * kv_dim + elem;
            const unsigned short* kb1 = k16 + (size_t)t1 * kv_dim + elem;
            const unsigned short* kb2 = k16 + (size_t)t2 * kv_dim + elem;
            const unsigned short* kb3 = k16 + (size_t)t3 * kv_dim + elem;
            const float* qb_thread = q_lds + d_start;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                partial0 += qb_thread[i] * hipfire_bf16_to_f32(kb0[i]);
                partial1 += qb_thread[i] * hipfire_bf16_to_f32(kb1[i]);
                partial2 += qb_thread[i] * hipfire_bf16_to_f32(kb2[i]);
                partial3 += qb_thread[i] * hipfire_bf16_to_f32(kb3[i]);
            }
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
        const int t = tile_start + t_local;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int d_start = (half * 4 + tid / 8) * 32 + k_off;
            const unsigned short* kb = k16 + (size_t)t * kv_dim + head_base + d_start;
            const float* qb_thread = q_lds + d_start;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                partial += qb_thread[i] * hipfire_bf16_to_f32(kb[i]);
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // ═══ Phase B: find tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        const float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    const float tile_sum = local_sum;
    __syncthreads();

    // ═══ Write tile partials ═══
    // Stride by max_tiles (derived from max_seq), NOT the dynamic n_tiles, so
    // the layout matches attention_flash_q8_0_reduce's per-head row stride.
    // Getting this wrong makes every head except h=0 read stale/uninitialised
    // partials whenever seq_len < max_seq — see the note on the Q8 kernel.
    const int max_tiles = (max_seq + tile_size - 1) / tile_size;
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    // ═══ Phase D: V-weighted accumulation ═══
    // Loops over halves so all head_dim output dims are written, not just the
    // first 128 — the reduce reads the full head_dim row, and a short write
    // leaves it folding whatever the previous caller left in the shared
    // partials scratch.
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        const int elem = head_base + d_base;
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        for (int tl = 0; tl < tile_len; tl++) {
            const float w = scores[tl];
            const unsigned short* vb =
                v16 + (size_t)(tile_start + tl) * kv_dim + elem;
            out0 += w * hipfire_bf16_to_f32(vb[0]);
            out1 += w * hipfire_bf16_to_f32(vb[1]);
            out2 += w * hipfire_bf16_to_f32(vb[2]);
            out3 += w * hipfire_bf16_to_f32(vb[3]);
        }
        p[2 + d_base + 0] = out0;
        p[2 + d_base + 1] = out1;
        p[2 + d_base + 2] = out2;
        p[2 + d_base + 3] = out3;
    }
}
