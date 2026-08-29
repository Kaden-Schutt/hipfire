// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched V2 Lloyd2 V write: two independent 128-wide halves (signed-FWHT-128
// + 2-bit Lloyd-Max N(0, 1/128)). Processes N positions in one launch.
// Identical format to kv_cache_write_fwht256_2bit: two fp16 cnorms, half0
// indices then half1 indices per lane.
//
// Byte layout: [2×fp16 cnorm][32 threads × 2 bytes packed 2-bit indices] =
// 68 B/head at head_dim=256.
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1].
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_fwht256_2bit_batched(
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
    const int bytes_per_head = 4 + (head_dim * 2) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* head_src = k_src + bid * kv_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += 32) x[d] = head_src[d];
    __syncthreads();

    int idx[8];
    float cnorm_h[2];

    for (int half = 0; half < 2; half++) {
        const int base = half * 128;
        const int d0 = base + tid * 4;

        float lsq = 0.0f;
        float v0 = x[d0 + 0], v1 = x[d0 + 1], v2 = x[d0 + 2], v3 = x[d0 + 3];
        lsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

        float fa = v0 * inv_norm, fb = v1 * inv_norm;
        float fc = v2 * inv_norm, fd = v3 * inv_norm;
        fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

        int i0 = half * 4 + 0;
        int i1 = half * 4 + 1;
        int i2 = half * 4 + 2;
        int i3 = half * 4 + 3;
        idx[i0] = turbo_quantize_2bit(fa);
        idx[i1] = turbo_quantize_2bit(fb);
        idx[i2] = turbo_quantize_2bit(fc);
        idx[i3] = turbo_quantize_2bit(fd);

        float local_rsq =
            TURBO_C2[idx[i0]] * TURBO_C2[idx[i0]] +
            TURBO_C2[idx[i1]] * TURBO_C2[idx[i1]] +
            TURBO_C2[idx[i2]] * TURBO_C2[idx[i2]] +
            TURBO_C2[idx[i3]] * TURBO_C2[idx[i3]];
        scratch[tid] = local_rsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        float recon_norm = sqrtf(scratch[0]);
        cnorm_h[half] = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    }

    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;
    if (tid == 0) {
        ((__half*)out)[0] = __float2half(cnorm_h[0]);
        ((__half*)out)[1] = __float2half(cnorm_h[1]);
    }

    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 3)) << (i * 2);
    unsigned char* base = out + 4 + tid * 2;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8) & 0xFF);
}
