// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// K-only write for fwht3, head_dim=512 (Gemma 4 full-attention layers).
//
// Single-pass signed-FWHT-512 rotation + 3-bit quantize (Lloyd-Max N(0, 1/512)).
// V is written separately via kv_cache_write_q8_0.
//
// Unlike asym3_hd512 (which splits dims into two 256-wide Givens chunks per lane),
// FWHT-512 is a single global transform over all 512 dims.  Each thread owns
// 16 *consecutive* dims (d0 = tid*16 .. tid*16+15).  The 9-pass butterfly
// (4 register-local + 5 ds_swizzle) covers the full 512 with no cross-warp
// communication — 32 lanes x 16 elements = 512.
//
// Output layout per head: [4B cnorm][32 threads x 6 bytes packed 3-bit] = 196 B/head.
// Packing is contiguous-per-lane: lane tid writes 6 bytes at offset 4 + tid*6.
// First 3 bytes = idx[0..7] packed 3-bit LSB-first; next 3 bytes = idx[8..15].
// NOTE: This layout differs from asym3_hd512 (which uses tid*3 / k_chunk+tid*3
// split across the two 256-dim halves); the read kernel must match this exactly.
//
// Grid: [n_kv_heads]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_fwht3_hd512(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads, int head_dim
) {
    if (head_dim != 512) return;

    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = pos_buf[0];

    const int bytes_per_head = 4 + (head_dim * 3) / 8;   // 196 for hd=512
    const int bytes_per_pos  = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* src = k_src + h * head_dim;
    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    // Load 512 floats into LDS.
    for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
    __syncthreads();

    // Parallel L2 norm.
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
    __syncthreads();

    // Each thread owns 16 consecutive dims: d0 = tid*16 .. tid*16+15.
    const int d0 = tid * 16;
    float v0  = x[d0 +  0], v1  = x[d0 +  1];
    float v2  = x[d0 +  2], v3  = x[d0 +  3];
    float v4  = x[d0 +  4], v5  = x[d0 +  5];
    float v6  = x[d0 +  6], v7  = x[d0 +  7];
    float v8  = x[d0 +  8], v9  = x[d0 +  9];
    float v10 = x[d0 + 10], v11 = x[d0 + 11];
    float v12 = x[d0 + 12], v13 = x[d0 + 13];
    float v14 = x[d0 + 14], v15 = x[d0 + 15];

    // Single-pass signed-FWHT-512 (9 passes = 4 register-local + 5 swizzle).
    fwht_shfl_forward_512(v0, v1, v2, v3, v4, v5, v6, v7,
                          v8, v9, v10, v11, v12, v13, v14, v15,
                          signs1, signs2, tid);

    int idx[16];
    idx[0]  = turbo_quantize_3bit_512(v0);
    idx[1]  = turbo_quantize_3bit_512(v1);
    idx[2]  = turbo_quantize_3bit_512(v2);
    idx[3]  = turbo_quantize_3bit_512(v3);
    idx[4]  = turbo_quantize_3bit_512(v4);
    idx[5]  = turbo_quantize_3bit_512(v5);
    idx[6]  = turbo_quantize_3bit_512(v6);
    idx[7]  = turbo_quantize_3bit_512(v7);
    idx[8]  = turbo_quantize_3bit_512(v8);
    idx[9]  = turbo_quantize_3bit_512(v9);
    idx[10] = turbo_quantize_3bit_512(v10);
    idx[11] = turbo_quantize_3bit_512(v11);
    idx[12] = turbo_quantize_3bit_512(v12);
    idx[13] = turbo_quantize_3bit_512(v13);
    idx[14] = turbo_quantize_3bit_512(v14);
    idx[15] = turbo_quantize_3bit_512(v15);

    // Centroid-norm reduce for cnorm.
    float local_rsq = 0.0f;
    for (int i = 0; i < 16; i++) local_rsq += TURBO_C3_512[idx[i]] * TURBO_C3_512[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    // Pack 16 x 3-bit = 48 bits = 6 bytes per lane, contiguous at 4 + tid*6.
    // packed0 = idx[0..7]  (24 bits); packed1 = idx[8..15] (24 bits).
    unsigned int packed0 = 0;
    for (int i = 0; i < 8; i++) packed0 |= ((unsigned int)(idx[i]     & 7)) << (i * 3);
    unsigned int packed1 = 0;
    for (int i = 0; i < 8; i++) packed1 |= ((unsigned int)(idx[8 + i] & 7)) << (i * 3);

    unsigned char* base = out + 4 + tid * 6;
    base[0] = (unsigned char)( packed0        & 0xFF);
    base[1] = (unsigned char)((packed0 >>  8) & 0xFF);
    base[2] = (unsigned char)((packed0 >> 16) & 0xFF);
    base[3] = (unsigned char)( packed1        & 0xFF);
    base[4] = (unsigned char)((packed1 >>  8) & 0xFF);
    base[5] = (unsigned char)((packed1 >> 16) & 0xFF);
}
