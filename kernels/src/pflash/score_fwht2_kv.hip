// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// PFlash per-block scoring on fwht2 K cache.
//
// Sibling of `pflash_score_fwht3_kv_blocks` / `_fwht4_kv_blocks`. fwht2
// is the smallest K storage in the FWHT family — 2-bit indices into the
// FWHT-128-tuned TURBO_C2 LUT, 4 centroids only, but at head_dim=256
// the head_dim averaging in the cosine score makes this aggressive
// quantization potentially viable for scoring even though it's too
// noisy for inference attention.
//
// Per "try-all-anyway" — likely the worst NIAH gate of the three; ship
// for ablation completeness, not as the default.
//
// As with fwht3/fwht4, cosine is rotation-invariant — no inverse FWHT
// in the scoring path.
//
// fwht2 layout (matches `kv_cache_write_asym_k_fwht2` and
// `attention_flash_fwht2_tile`):
//   per position, per head:
//     bytes [0..4]:                     f32 cnorm
//     bytes [4..4+head_dim/4]:          packed 2-bit codes (4/byte)
//   per head total bytes:  4 + head_dim / 4
//                          = 4 + 64 = 68 bytes for head_dim=256
//
// head_dim=256 uses two FWHT-128 halves (same shape as fwht4). Within
// a half, 32 thread groups each own 4 dims, packed into ONE byte
// (4 codes × 2 bits each = 8 bits = 1 byte).
//
//   code[0..3] = (byte >> (i*2)) & 0x3
//
// Byte offset within the head: `4 + half*32 + tg*1`. The 32-byte stride
// between halves matches 32 threads × 1 byte/thread = 32 bytes/half.

#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void pflash_score_fwht2_kv_blocks(
    const unsigned char* __restrict__ k_cache,
    float* __restrict__ scores,
    int n_pos,
    int n_kv_heads,
    int head_dim,
    int block_size,
    int n_blocks,
    int last_pos
) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    int block_start = block_idx * block_size;
    int block_end = block_start + block_size;
    if (block_end > n_pos) block_end = n_pos;
    int block_len = block_end - block_start;
    if (block_len <= 0) {
        if (threadIdx.x == 0) scores[block_idx] = 0.0f;
        return;
    }

    int kv_dim = n_kv_heads * head_dim;
    int bytes_per_head = 4 + head_dim / 4;
    int bytes_per_pos = n_kv_heads * bytes_per_head;

    __shared__ float s_dot[256];
    __shared__ float s_nb[256];
    __shared__ float s_nl[256];

    int tid = threadIdx.x;

    float dot = 0.0f;
    float nb = 0.0f;
    float nl = 0.0f;

    for (int d = tid; d < kv_dim; d += 256) {
        int h = d / head_dim;
        int dim_in_head = d - h * head_dim;
        int half = dim_in_head / 128;
        int d_in_half = dim_in_head - half * 128;
        int thread_group = d_in_half / 4;
        int local_idx = d_in_half - thread_group * 4;
        int bit_off = local_idx * 2;

        // last-pos K[d]
        const unsigned char* last_head =
            k_cache + (size_t)last_pos * bytes_per_pos + h * bytes_per_head;
        float last_cnorm = *(const float*)last_head;
        unsigned int last_byte =
            (unsigned int)last_head[4 + half * 32 + thread_group];
        int last_code = (last_byte >> bit_off) & 0x3;
        float last_v = TURBO_C2[last_code] * last_cnorm;

        // Block-mean K[d]
        float sum = 0.0f;
        for (int pos = block_start; pos < block_end; ++pos) {
            const unsigned char* head_base =
                k_cache + (size_t)pos * bytes_per_pos + h * bytes_per_head;
            float cnorm = *(const float*)head_base;
            unsigned int byte =
                (unsigned int)head_base[4 + half * 32 + thread_group];
            int code = (byte >> bit_off) & 0x3;
            sum += TURBO_C2[code] * cnorm;
        }
        float mean = sum / (float)block_len;

        dot += mean * last_v;
        nb += mean * mean;
        nl += last_v * last_v;
    }

    s_dot[tid] = dot;
    s_nb[tid] = nb;
    s_nl[tid] = nl;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
            s_nb[tid] += s_nb[tid + stride];
            s_nl[tid] += s_nl[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = sqrtf(s_nb[0]) * sqrtf(s_nl[0]);
        if (denom < 1e-12f) denom = 1e-12f;
        scores[block_idx] = s_dot[0] / denom;
    }
}
