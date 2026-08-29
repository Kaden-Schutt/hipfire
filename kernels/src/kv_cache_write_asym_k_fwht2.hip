// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// K-only write for fwht2: signed-FWHT-128 rotation + 2-bit quantize.
// V is written separately via kv_cache_write_q8_0.
//
// Structurally identical to kv_cache_write_asym_k_givens2 except for the
// rotation primitive — Givens (per-quad cos/sin) replaced with signed-FWHT
// (128-wide butterfly via ds_swizzle_b32).
//
// Handles head_dim ∈ {128, 256}. hd=128 → 36 B/head. hd=256 → 68 B/head.
// Storage: [4B cnorm][head_dim/4 packed bytes, 4 values per byte]
//   (byte-identical to asym2 — scoring kernels are rotation-agnostic.)
//
// Grid: [n_kv_heads]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_asym_k_fwht2(
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

    const int bytes_per_head = 4 + head_dim / 4;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int n_halves = head_dim / 128;

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

    float local_rsq = 0.0f;
    int idx[8];

    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float fa = x[d0];
        float fb = x[d0 + 1];
        float fc = x[d0 + 2];
        float fd = x[d0 + 3];
        // signs1/signs2 are shared across halves — matches fwht4 pattern.
        fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

        int i0 = half * 4 + 0;
        int i1 = half * 4 + 1;
        int i2 = half * 4 + 2;
        int i3 = half * 4 + 3;
        idx[i0] = turbo_quantize_2bit(fa); local_rsq += TURBO_C2[idx[i0]] * TURBO_C2[idx[i0]];
        idx[i1] = turbo_quantize_2bit(fb); local_rsq += TURBO_C2[idx[i1]] * TURBO_C2[idx[i1]];
        idx[i2] = turbo_quantize_2bit(fc); local_rsq += TURBO_C2[idx[i2]] * TURBO_C2[idx[i2]];
        idx[i3] = turbo_quantize_2bit(fd); local_rsq += TURBO_C2[idx[i3]] * TURBO_C2[idx[i3]];
    }

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    // Pack: 4 values × 2 bits = 1 byte per thread per half.
    for (int half = 0; half < n_halves; half++) {
        int base = half * 4;
        int write_off = 4 + half * 32 + tid;
        out[write_off] = (unsigned char)(
            (idx[base + 0] & 0x3) |
            ((idx[base + 1] & 0x3) << 2) |
            ((idx[base + 2] & 0x3) << 4) |
            ((idx[base + 3] & 0x3) << 6)
        );
    }
}
