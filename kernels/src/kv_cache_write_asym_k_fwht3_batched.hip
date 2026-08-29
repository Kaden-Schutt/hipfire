// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched K-only write for fwht3 V2: two independent signed-FWHT-128 halves +
// 3-bit quantize (TURBO_C3). Processes N positions in one launch.
// V via kv_cache_write_q8_0_batched. hd=256 only.
//
// Byte-identical to sequential fwht3:
//   [2×fp16 cnorm][32 × 3B packs with half0 idxs then half1 idxs].
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1].
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_asym_k_fwht3_batched(
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
    const int bytes_per_head = 4 + (head_dim * 3) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int n_halves = head_dim / 128;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* head_src = k_src + bid * kv_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += 32) x[d] = head_src[d];
    __syncthreads();

    float cnorm_h[2] = {0.0f, 0.0f};
    int idx[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];

        float lsq = fa * fa + fb * fb + fc * fc + fd * fd;
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float orig_half_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_half_norm > 1e-10f) ? 1.0f / orig_half_norm : 0.0f;

        fa *= inv_norm; fb *= inv_norm; fc *= inv_norm; fd *= inv_norm;
        fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

        int i0 = half * 4 + 0, i1 = half * 4 + 1, i2 = half * 4 + 2, i3 = half * 4 + 3;
        idx[i0] = turbo_quantize_3bit(fa);
        idx[i1] = turbo_quantize_3bit(fb);
        idx[i2] = turbo_quantize_3bit(fc);
        idx[i3] = turbo_quantize_3bit(fd);

        float local_rsq =
            TURBO_C3[idx[i0]] * TURBO_C3[idx[i0]] +
            TURBO_C3[idx[i1]] * TURBO_C3[idx[i1]] +
            TURBO_C3[idx[i2]] * TURBO_C3[idx[i2]] +
            TURBO_C3[idx[i3]] * TURBO_C3[idx[i3]];
        scratch[tid] = local_rsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float recon_norm = sqrtf(scratch[0]);
        cnorm_h[half] = (recon_norm > 1e-10f) ? orig_half_norm / recon_norm : orig_half_norm;
    }

    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;
    if (tid == 0) {
        ((__half*)out)[0] = __float2half(cnorm_h[0]);
        ((__half*)out)[1] = __float2half(cnorm_h[1]);
    }

    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 7)) << (i * 3);
    unsigned char* base = out + 4 + tid * 3;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
}
