// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// 256-wide V write at 2-bit: signed-FWHT-256 rotation + 2-bit quantize (Lloyd-Max N(0, 1/256)).
// Identical rotation layout to kv_cache_write_asym_k_fwht3 (8 dims/thread, tid*8); only the
// quant fn (turbo_quantize_2bit_256), centroid LUT (TURBO_C2_256), bytes_per_head, and packing
// width differ. This matches the 256-wide fwht3 attention kernel's rotation so V can be read back
// under fwht3 K.
//
// Byte layout: [4B cnorm][32 threads × 2 bytes packed 2-bit indices] = 68 B/head at head_dim=256.
//
// Grid: [n_kv_heads]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_fwht256_2bit(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = pos_buf[0];

    const int bytes_per_head = 4 + (head_dim * 2) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* src = k_src + h * head_dim;
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

    // Each thread handles 8 consecutive dims = tid*8 .. tid*8+7.
    const int d0 = tid * 8;
    float v0 = x[d0 + 0], v1 = x[d0 + 1];
    float v2 = x[d0 + 2], v3 = x[d0 + 3];
    float v4 = x[d0 + 4], v5 = x[d0 + 5];
    float v6 = x[d0 + 6], v7 = x[d0 + 7];

    // Single-pass signed-FWHT over the full 256 dims.
    fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

    int idx[8];
    idx[0] = turbo_quantize_2bit_256(v0);
    idx[1] = turbo_quantize_2bit_256(v1);
    idx[2] = turbo_quantize_2bit_256(v2);
    idx[3] = turbo_quantize_2bit_256(v3);
    idx[4] = turbo_quantize_2bit_256(v4);
    idx[5] = turbo_quantize_2bit_256(v5);
    idx[6] = turbo_quantize_2bit_256(v6);
    idx[7] = turbo_quantize_2bit_256(v7);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) local_rsq += TURBO_C2_256[idx[i]] * TURBO_C2_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 3)) << (i * 2);
    unsigned char* base = out + 4 + tid * 2;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8) & 0xFF);
}
