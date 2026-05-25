// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// PFlash per-block scoring on fwht3 K cache.
//
// Drop-in replacement for `pflash_score_q8_kv_blocks` when the drafter
// runs with fwht3 KV (the LDS-cliff-free path for ctx > 15K). Same
// `[block_idx]` grid, same `[256]` thread block, same shared-mem
// reduction shape, same cosine output. The only thing that changes is
// the K dequant: fwht3 stores 3-bit indices into TURBO_C3_256 plus a
// per-head f32 `cnorm` (the original head's L2 norm pre-FWHT).
//
// Why no inverse FWHT here: the score is `cos_sim(K_block_mean, K_last)`,
// and cosine is rotation-invariant under orthonormal transforms. The
// signed-FWHT-256 the K-write kernel applies is orthonormal, so
//
//   cos_sim(F(a), F(b)) = cos_sim(a, b)
//
// meaning we can compute the cosine entirely in FWHT-rotated space.
// Similarly, the per-position mean commutes with FWHT linearity:
//
//   mean( F(K_pos) ) = F( mean(K_pos) )
//
// so block-mean in rotated space equals F(block-mean in original space),
// and its cosine with rotated last-K equals the original-space cosine.
// This is why a "scorer in rotated space" gives identical scores to a
// (much more expensive) scorer that inverse-FWHTs every K before
// reducing.
//
// fwht3 layout (matches `kv_cache_write_asym_k_fwht3`):
//   per position, per head:
//     bytes [0..4]:    f32 cnorm (the original head's L2 norm)
//     bytes [4..4+(head_dim*3)/8]: packed 3-bit codes
//   per head total bytes:  4 + (head_dim * 3) / 8
//                          = 4 + 96 = 100 bytes for head_dim=256
//   per position bytes:    n_kv_heads * bytes_per_head
//
// Per-thread 3-bit packing (matches the write kernel + the flash tile
// dequant): thread `tg` (0..31 for head_dim=256) owns dims
// `tg*8 .. tg*8+7`; their 8 codes pack into 3 consecutive bytes at
// `cnorm[4] + tg*3..tg*3+3`, low-bit-first:
//
//   code[0] = (packed >>  0) & 7
//   code[1] = (packed >>  3) & 7
//   ...
//   code[7] = (packed >> 21) & 7
//
// where `packed = b0 | (b1<<8) | (b2<<16)`.

#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void pflash_score_fwht3_kv_blocks(
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
    int bytes_per_head = 4 + (head_dim * 3) / 8;
    int bytes_per_pos = n_kv_heads * bytes_per_head;

    __shared__ float s_dot[256];
    __shared__ float s_nb[256];
    __shared__ float s_nl[256];

    int tid = threadIdx.x;

    float dot = 0.0f;
    float nb = 0.0f;
    float nl = 0.0f;

    // Stride over the full kv_dim. Each iteration handles one dim `d` and
    // recomputes both the last-pos K[d] and the block-mean K[d] for that
    // dim. The block-mean read is the inner loop — re-reading `last K[d]`
    // for every (block, dim) iter on the GPU is cheaper than the alt
    // (separate pre-dequant pass + global memory roundtrip).
    for (int d = tid; d < kv_dim; d += 256) {
        int h = d / head_dim;
        int dim_in_head = d - h * head_dim;
        // The fwht3 write kernel uses 32 threads × 8 dims per thread, so
        // `thread_group = dim_in_head / 8` selects which 3-byte packed
        // group within the head holds this dim, and `local_idx` is its
        // index inside the packed 24-bit word (0..7).
        int thread_group = dim_in_head / 8;
        int local_idx = dim_in_head - thread_group * 8;
        int bit_off = local_idx * 3;

        // last-pos K[d]: load cnorm + the relevant 3 bytes, extract the
        // 3-bit code, dequant.
        const unsigned char* last_head =
            k_cache + (size_t)last_pos * bytes_per_pos + h * bytes_per_head;
        float last_cnorm = *(const float*)last_head;
        const unsigned char* last_base = last_head + 4 + thread_group * 3;
        unsigned int last_packed =
              (unsigned int)last_base[0]
            | ((unsigned int)last_base[1] << 8)
            | ((unsigned int)last_base[2] << 16);
        int last_code = (last_packed >> bit_off) & 7;
        float last_v = TURBO_C3_256[last_code] * last_cnorm;

        // Block-mean K[d]: sum the dequanted value over all positions in
        // this block, divide by block_len.
        float sum = 0.0f;
        for (int pos = block_start; pos < block_end; ++pos) {
            const unsigned char* head_base =
                k_cache + (size_t)pos * bytes_per_pos + h * bytes_per_head;
            float cnorm = *(const float*)head_base;
            const unsigned char* base = head_base + 4 + thread_group * 3;
            unsigned int packed =
                  (unsigned int)base[0]
                | ((unsigned int)base[1] << 8)
                | ((unsigned int)base[2] << 16);
            int code = (packed >> bit_off) & 7;
            sum += TURBO_C3_256[code] * cnorm;
        }
        float mean = sum / (float)block_len;

        dot += mean * last_v;
        nb += mean * mean;
        nl += last_v * last_v;
    }

    // 256-way shared-mem reduction — identical shape to pflash_score_q8_kv.
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
