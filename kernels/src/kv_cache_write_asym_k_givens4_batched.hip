// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched K-only write for asym4: Givens rotation + 4-bit quantize.
// Processes N positions in one launch. V is written separately via
// kv_cache_write_q8_0_batched. Handles head_dim ∈ {128, 256} via 2-pass loop.
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1].
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __global__ void kv_cache_write_asym_k_givens4_batched(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,           // [batch_size × kv_dim]
    const int* __restrict__ positions,         // [batch_size] absolute positions
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
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
    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int n_halves = head_dim / 128;

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

    float local_rsq = 0.0f;
    int idx[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
        const int block_tid = half * 32 + tid;
        givens_forward(fa, fb, fc, fd, cos_theta, sin_theta, block_tid);

        int i0 = half * 4 + 0, i1 = half * 4 + 1, i2 = half * 4 + 2, i3 = half * 4 + 3;
        idx[i0] = turbo_quantize_4bit(fa); local_rsq += TURBO_C4[idx[i0]] * TURBO_C4[idx[i0]];
        idx[i1] = turbo_quantize_4bit(fb); local_rsq += TURBO_C4[idx[i1]] * TURBO_C4[idx[i1]];
        idx[i2] = turbo_quantize_4bit(fc); local_rsq += TURBO_C4[idx[i2]] * TURBO_C4[idx[i2]];
        idx[i3] = turbo_quantize_4bit(fd); local_rsq += TURBO_C4[idx[i3]] * TURBO_C4[idx[i3]];
    }

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;
    if (tid == 0) *(float*)out = cnorm;
    for (int half = 0; half < n_halves; half++) {
        int base = half * 4;
        int write_off = 4 + half * 64 + tid * 2;
        out[write_off]     = (unsigned char)((idx[base + 1] << 4) | (idx[base + 0] & 0xF));
        out[write_off + 1] = (unsigned char)((idx[base + 3] << 4) | (idx[base + 2] & 0xF));
    }
}
