// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// PFlash per-block scoring on fwht4 K cache.
//
// Sibling of `pflash_score_fwht3_kv_blocks`: same grid + reduction
// shape, different dequant. fwht4 stores 4-bit indices into the
// FWHT-128-tuned TURBO_C4 LUT, with a per-head f32 `cnorm` followed by
// `head_dim/2` packed nibbles.
//
// Why fwht4 here at all: the user asked to try the full fwht{4,3,2}
// family. fwht4 keeps more precision per element (16 centroids vs 8
// for fwht3 and 4 for fwht2) at the cost of larger K storage (132 B/head
// vs 100 B fwht3 / 68 B fwht2 at head_dim=256). For the cosine-scorer
// the precision usually doesn't matter much (the head_dim averaging
// washes out quantization noise), so fwht3 is expected to win the
// throughput-vs-quality tradeoff. fwht4 is shipped as a research
// variant — useful if fwht3 ever shows a NIAH regression.
//
// As with fwht3, cosine is rotation-invariant under FWHT, so the
// scorer doesn't undo the FWHT — same algebra:
//
//   cos_sim(F(a), F(b)) = cos_sim(a, b)
//   mean(F(K_pos))      = F(mean(K_pos))
//
// fwht4 layout (matches `kv_cache_write_asym_k_fwht4` and
// `attention_flash_fwht4_tile`):
//   per position, per head:
//     bytes [0..4]:                     f32 cnorm
//     bytes [4..4+head_dim/2]:          packed 4-bit codes
//   per head total bytes:  4 + head_dim / 2
//                          = 4 + 128 = 132 bytes for head_dim=256
//   per position bytes:    n_kv_heads * bytes_per_head
//
// head_dim=256 uses two independent FWHT-128 halves (the write kernel
// runs `n_halves = head_dim / 128 = 2`). Within a half, thread `tg`
// (0..31) owns dims `tg*4..tg*4+3` and stores them in 2 consecutive
// bytes interpreted as one little-endian u16:
//
//   packed = b0 | (b1<<8)            // 16 bits = 4 codes × 4 bits
//   code[0..3] = (packed >> (i*4)) & 0xF
//
// Byte offset within the head: `4 + half*64 + tg*2`. The 64-byte stride
// between halves matches 32 threads × 2 bytes/thread = 64 bytes/half.

#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void pflash_score_fwht4_kv_blocks(
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
    int bytes_per_head = 4 + head_dim / 2;
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
        // Two halves of FWHT-128 per head. Within a half, 32 thread
        // groups each own 4 contiguous dims.
        int half = dim_in_head / 128;
        int d_in_half = dim_in_head - half * 128;
        int thread_group = d_in_half / 4;
        int local_idx = d_in_half - thread_group * 4;
        int bit_off = local_idx * 4;

        // last-pos K[d]
        const unsigned char* last_head =
            k_cache + (size_t)last_pos * bytes_per_pos + h * bytes_per_head;
        float last_cnorm = *(const float*)last_head;
        const unsigned char* last_base =
            last_head + 4 + half * 64 + thread_group * 2;
        unsigned int last_packed =
              (unsigned int)last_base[0]
            | ((unsigned int)last_base[1] << 8);
        int last_code = (last_packed >> bit_off) & 0xF;
        float last_v = TURBO_C4[last_code] * last_cnorm;

        // Block-mean K[d]
        float sum = 0.0f;
        for (int pos = block_start; pos < block_end; ++pos) {
            const unsigned char* head_base =
                k_cache + (size_t)pos * bytes_per_pos + h * bytes_per_head;
            float cnorm = *(const float*)head_base;
            const unsigned char* base =
                head_base + 4 + half * 64 + thread_group * 2;
            unsigned int packed =
                  (unsigned int)base[0]
                | ((unsigned int)base[1] << 8);
            int code = (packed >> bit_off) & 0xF;
            sum += TURBO_C4[code] * cnorm;
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
