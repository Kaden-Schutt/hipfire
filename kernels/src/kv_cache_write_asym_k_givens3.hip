// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// K-only write for asym3: Givens rotation + 3-bit quantize (Lloyd-Max N(0, 1/256)).
// V is written separately via kv_cache_write_q8_0.
//
// Per RotorQuant: 3-bit gives the best quality/compression tradeoff at
// rotated K — 8 centroids cover the N(0, 1/d) distribution well without
// wasting codebook on empty tails (as 4-bit does). Paper reports 3-bit
// K planar + FP16 V ≈ 0% PPL loss.
//
// Head layout (head_dim=256): 32 threads × 8 values = 256 dims covered
// in a single pass. 8 × 3-bit = 24 bits = 3 bytes per thread.
//   [4B cnorm][32 threads × 3 bytes packed indices] = 100 B/head.
//
// Givens: each thread handles 4 2D blocks (indices tid*4 .. tid*4+3).
// cos_theta/sin_theta size = head_dim/2 = 128 entries.
//
// Grid: [n_kv_heads]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

__device__ __forceinline__ void givens_rot_inplace(
    float& a, float& b,
    float c, float s)
{
    float a2 = a * c - b * s;
    float b2 = a * s + b * c;
    a = a2; b = b2;
}

extern "C" __global__ void kv_cache_write_asym_k_givens3(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = pos_buf[0];

    // 3-bit packed: 4B cnorm + 32 threads × 3 bytes
    const int bytes_per_head = 4 + (head_dim * 3) / 8;   // 100 for hd=256
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* src = k_src + h * head_dim;
    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
    __syncthreads();

    // Parallel L2 norm
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
    __syncthreads();

    // Each thread handles 8 consecutive dims = 4 Givens blocks.
    const int d0 = tid * 8;
    float v0 = x[d0 + 0], v1 = x[d0 + 1];
    float v2 = x[d0 + 2], v3 = x[d0 + 3];
    float v4 = x[d0 + 4], v5 = x[d0 + 5];
    float v6 = x[d0 + 6], v7 = x[d0 + 7];

    const int b0 = tid * 4;
    givens_rot_inplace(v0, v1, cos_theta[b0 + 0], sin_theta[b0 + 0]);
    givens_rot_inplace(v2, v3, cos_theta[b0 + 1], sin_theta[b0 + 1]);
    givens_rot_inplace(v4, v5, cos_theta[b0 + 2], sin_theta[b0 + 2]);
    givens_rot_inplace(v6, v7, cos_theta[b0 + 3], sin_theta[b0 + 3]);

    // Quantize 8 values → 3-bit indices. Lloyd-Max centroids for N(0, 1/256).
    int idx[8];
    idx[0] = turbo_quantize_3bit_256(v0);
    idx[1] = turbo_quantize_3bit_256(v1);
    idx[2] = turbo_quantize_3bit_256(v2);
    idx[3] = turbo_quantize_3bit_256(v3);
    idx[4] = turbo_quantize_3bit_256(v4);
    idx[5] = turbo_quantize_3bit_256(v5);
    idx[6] = turbo_quantize_3bit_256(v6);
    idx[7] = turbo_quantize_3bit_256(v7);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) local_rsq += TURBO_C3_256[idx[i]] * TURBO_C3_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    // Pack 8 × 3-bit = 24 bits = 3 bytes per thread.
    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 7)) << (i * 3);
    unsigned char* base = out + 4 + tid * 3;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
}
