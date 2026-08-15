// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched K-only write for asym3, head_dim=512 (Gemma 4 full-attention layers).
//
// N-token counterpart to `kv_cache_write_asym_k_givens3_hd512`. Each
// workgroup handles (head, batch_idx); positions are read from a device
// array (one i32 per token).
//
// Layout per pos (per head): [4B cnorm][32 threads × 6 bytes packed indices]
// = 4 + 192 = 196 B/head. Same as the single-token hd=512 kernel; the only
// changes are the outer batch dim and the per-batch input/output base
// pointer offsets.
//
// Grid: [n_kv_heads, batch_size]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

__device__ __forceinline__ void givens_rot_inplace_hd512_batched(
    float& a, float& b,
    float c, float s)
{
    float a2 = a * c - b * s;
    float b2 = a * s + b * c;
    a = a2; b = b2;
}

extern "C" __global__ void kv_cache_write_asym_k_givens3_hd512_batched(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,
    const int* __restrict__ positions,    // [batch_size] i32
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads, int head_dim, int batch_size
) {
    if (head_dim != 512) return;

    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int b = blockIdx.y;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = positions[b];

    const int bytes_per_head = 4 + (head_dim * 3) / 8;        // 196 for hd=512
    const int bytes_per_pos  = n_kv_heads * bytes_per_head;
    const int k_chunk_bytes  = (256 * 3) / 8;                 // 96

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    // Per-batch input base: row b in the [batch × n_kv_heads × head_dim] src.
    const float* src = k_src + ((long long)b * n_kv_heads + h) * head_dim;
    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
    __syncthreads();

    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
    __syncthreads();

    const int d0_lo = tid * 8;
    const int d0_hi = d0_lo + 256;

    float v0  = x[d0_lo + 0], v1  = x[d0_lo + 1];
    float v2  = x[d0_lo + 2], v3  = x[d0_lo + 3];
    float v4  = x[d0_lo + 4], v5  = x[d0_lo + 5];
    float v6  = x[d0_lo + 6], v7  = x[d0_lo + 7];
    float v8  = x[d0_hi + 0], v9  = x[d0_hi + 1];
    float v10 = x[d0_hi + 2], v11 = x[d0_hi + 3];
    float v12 = x[d0_hi + 4], v13 = x[d0_hi + 5];
    float v14 = x[d0_hi + 6], v15 = x[d0_hi + 7];

    const int b0_lo = tid * 4;
    const int b0_hi = b0_lo + 128;
    givens_rot_inplace_hd512_batched(v0,  v1,  cos_theta[b0_lo + 0], sin_theta[b0_lo + 0]);
    givens_rot_inplace_hd512_batched(v2,  v3,  cos_theta[b0_lo + 1], sin_theta[b0_lo + 1]);
    givens_rot_inplace_hd512_batched(v4,  v5,  cos_theta[b0_lo + 2], sin_theta[b0_lo + 2]);
    givens_rot_inplace_hd512_batched(v6,  v7,  cos_theta[b0_lo + 3], sin_theta[b0_lo + 3]);
    givens_rot_inplace_hd512_batched(v8,  v9,  cos_theta[b0_hi + 0], sin_theta[b0_hi + 0]);
    givens_rot_inplace_hd512_batched(v10, v11, cos_theta[b0_hi + 1], sin_theta[b0_hi + 1]);
    givens_rot_inplace_hd512_batched(v12, v13, cos_theta[b0_hi + 2], sin_theta[b0_hi + 2]);
    givens_rot_inplace_hd512_batched(v14, v15, cos_theta[b0_hi + 3], sin_theta[b0_hi + 3]);

    int idx[16];
    idx[0]  = turbo_quantize_3bit_256(v0);
    idx[1]  = turbo_quantize_3bit_256(v1);
    idx[2]  = turbo_quantize_3bit_256(v2);
    idx[3]  = turbo_quantize_3bit_256(v3);
    idx[4]  = turbo_quantize_3bit_256(v4);
    idx[5]  = turbo_quantize_3bit_256(v5);
    idx[6]  = turbo_quantize_3bit_256(v6);
    idx[7]  = turbo_quantize_3bit_256(v7);
    idx[8]  = turbo_quantize_3bit_256(v8);
    idx[9]  = turbo_quantize_3bit_256(v9);
    idx[10] = turbo_quantize_3bit_256(v10);
    idx[11] = turbo_quantize_3bit_256(v11);
    idx[12] = turbo_quantize_3bit_256(v12);
    idx[13] = turbo_quantize_3bit_256(v13);
    idx[14] = turbo_quantize_3bit_256(v14);
    idx[15] = turbo_quantize_3bit_256(v15);

    float local_rsq = 0.0f;
    for (int i = 0; i < 16; i++) local_rsq += TURBO_C3_256[idx[i]] * TURBO_C3_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    unsigned int packed_lo = 0;
    for (int i = 0; i < 8; i++) packed_lo |= ((unsigned int)(idx[i] & 7)) << (i * 3);
    unsigned int packed_hi = 0;
    for (int i = 0; i < 8; i++) packed_hi |= ((unsigned int)(idx[8 + i] & 7)) << (i * 3);

    unsigned char* base_lo = out + 4 + tid * 3;
    unsigned char* base_hi = out + 4 + k_chunk_bytes + tid * 3;
    base_lo[0] = (unsigned char)( packed_lo        & 0xFF);
    base_lo[1] = (unsigned char)((packed_lo >> 8)  & 0xFF);
    base_lo[2] = (unsigned char)((packed_lo >> 16) & 0xFF);
    base_hi[0] = (unsigned char)( packed_hi        & 0xFF);
    base_hi[1] = (unsigned char)((packed_hi >> 8)  & 0xFF);
    base_hi[2] = (unsigned char)((packed_hi >> 16) & 0xFF);
}
