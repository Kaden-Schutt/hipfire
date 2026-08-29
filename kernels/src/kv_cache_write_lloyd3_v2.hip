// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// V2 Lloyd3 V write: two independent 128-wide halves (signed-FWHT-128 + 3-bit
// Lloyd-Max N(0, 1/128) via TURBO_C3). Each half is L2-normalized, FWHT'd with
// shared signs1/signs2[0..127], quantized with turbo_quantize_3bit, and assigned
// cnorm_h = orig_half_norm / reconstructed_centroid_half_norm.
// Header is two IEEE fp16 cnorms (half0 at bytes 0..1, half1 at 2..3).
// hd=256 only.
//
// Byte layout: [2×fp16 cnorm][32 threads × 3 bytes packed 3-bit indices] =
// 100 B/head at head_dim=256. Per-lane 24-bit payload packs half0's four
// indices first (bits 0..11), then half1's four indices (bits 12..23).
// Thread owns dims half*128+tid*4+i.
//
// Dedicated Lloyd3 V path so FWHT3 K can stay on the shared V1 K writer while
// Lloyd3 readers consume this V2 record format.
//
// Grid: [n_kv_heads]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_lloyd3_v2(
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

    const int bytes_per_head = 4 + (head_dim * 3) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int n_halves = head_dim / 128;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* src = k_src + h * head_dim;
    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
    __syncthreads();

    float cnorm_h[2] = {0.0f, 0.0f};
    int idx[8];

    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float fa = x[d0];
        float fb = x[d0 + 1];
        float fc = x[d0 + 2];
        float fd = x[d0 + 3];

        float lsq = fa * fa + fb * fb + fc * fc + fd * fd;
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float orig_half_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_half_norm > 1e-10f) ? 1.0f / orig_half_norm : 0.0f;

        fa *= inv_norm; fb *= inv_norm; fc *= inv_norm; fd *= inv_norm;
        fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

        int i0 = half * 4 + 0;
        int i1 = half * 4 + 1;
        int i2 = half * 4 + 2;
        int i3 = half * 4 + 3;
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

    if (tid == 0) {
        ((__half*)out)[0] = __float2half(cnorm_h[0]);
        ((__half*)out)[1] = __float2half(cnorm_h[1]);
    }

    // half0 indices in bits 0..11, half1 indices in bits 12..23.
    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 7)) << (i * 3);
    unsigned char* base = out + 4 + tid * 3;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
}
