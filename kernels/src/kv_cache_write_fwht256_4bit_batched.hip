// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched 256-wide V write at 4-bit: signed-FWHT-256 + 4-bit quantize.
// Processes N positions in one launch. Identical rotation layout to
// kv_cache_write_asym_k_fwht3_batched; only quant fn / LUT / bytes_per_head / packing differ.
//
// Byte layout: [4B cnorm][32 threads × 4 bytes packed 4-bit indices] = 132 B/head at head_dim=256.
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1].
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_fwht256_4bit_batched(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= n_kv_heads || bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int pos = positions[bid];

    const int kv_dim = n_kv_heads * head_dim;
    const int bytes_per_head = 4 + (head_dim * 4) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* head_src = k_src + bid * kv_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += 32) x[d] = head_src[d];
    __syncthreads();

    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
    __syncthreads();

    const int d0 = tid * 8;
    float v0 = x[d0 + 0], v1 = x[d0 + 1];
    float v2 = x[d0 + 2], v3 = x[d0 + 3];
    float v4 = x[d0 + 4], v5 = x[d0 + 5];
    float v6 = x[d0 + 6], v7 = x[d0 + 7];

    fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

    int idx[8];
    idx[0] = turbo_quantize_4bit_256(v0);
    idx[1] = turbo_quantize_4bit_256(v1);
    idx[2] = turbo_quantize_4bit_256(v2);
    idx[3] = turbo_quantize_4bit_256(v3);
    idx[4] = turbo_quantize_4bit_256(v4);
    idx[5] = turbo_quantize_4bit_256(v5);
    idx[6] = turbo_quantize_4bit_256(v6);
    idx[7] = turbo_quantize_4bit_256(v7);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) local_rsq += TURBO_C4_256[idx[i]] * TURBO_C4_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;
    if (tid == 0) *(float*)out = cnorm;

    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 15)) << (i * 4);
    unsigned char* base = out + 4 + tid * 4;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
    base[3] = (unsigned char)((packed >> 24) & 0xFF);
}
