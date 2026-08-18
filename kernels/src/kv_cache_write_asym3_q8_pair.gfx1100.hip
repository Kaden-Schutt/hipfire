// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

__device__ __forceinline__ void givens_rot_inplace_asym3_pair(
    float& a, float& b,
    float c, float s)
{
    float a2 = a * c - b * s;
    float b2 = a * s + b * c;
    a = a2; b = b2;
}

// Exact heterogeneous-grid sibling of kv_cache_write_asym_k_givens3 plus
// kv_cache_write_q8_0. The first n_kv_heads workgroups write rotated asym3 K;
// the remaining n_kv_heads * (head_dim / 32) workgroups write Q8_0 V. Keeping
// the original workgroup geometries preserves both quantizers' arithmetic and
// only removes the launch boundary between the independent cache writes.
extern "C" __global__ void kv_cache_write_asym3_q8_pair_gfx1100(
    unsigned char* __restrict__ k_dst,
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads,
    int head_dim
) {
    const int combined_gid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    if (combined_gid < n_kv_heads) {
        const int h = combined_gid;
        const int nthreads = blockDim.x;
        const int bytes_per_head = 4 + (head_dim * 3) / 8;
        const int bytes_per_pos = n_kv_heads * bytes_per_head;

        extern __shared__ float smem[];
        float* x = smem;
        float* scratch = smem + head_dim;

        const float* src = k_src + h * head_dim;
        unsigned char* out =
            k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

        for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
        __syncthreads();

        float lsq = 0.0f;
        for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

        for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
        __syncthreads();

        const int d0 = tid * 8;
        float v0 = x[d0 + 0], v1 = x[d0 + 1];
        float v2 = x[d0 + 2], v3 = x[d0 + 3];
        float v4 = x[d0 + 4], v5 = x[d0 + 5];
        float v6 = x[d0 + 6], v7 = x[d0 + 7];

        const int b0 = tid * 4;
        givens_rot_inplace_asym3_pair(
            v0, v1, cos_theta[b0 + 0], sin_theta[b0 + 0]);
        givens_rot_inplace_asym3_pair(
            v2, v3, cos_theta[b0 + 1], sin_theta[b0 + 1]);
        givens_rot_inplace_asym3_pair(
            v4, v5, cos_theta[b0 + 2], sin_theta[b0 + 2]);
        givens_rot_inplace_asym3_pair(
            v6, v7, cos_theta[b0 + 3], sin_theta[b0 + 3]);

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
        for (int i = 0; i < 8; i++)
            local_rsq += TURBO_C3_256[idx[i]] * TURBO_C3_256[idx[i]];

        scratch[tid] = local_rsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        float recon_norm = sqrtf(scratch[0]);
        float cnorm =
            (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
        if (tid == 0) *(float*)out = cnorm;

        unsigned int packed = 0;
        for (int i = 0; i < 8; i++)
            packed |= ((unsigned int)(idx[i] & 7)) << (i * 3);
        unsigned char* base = out + 4 + tid * 3;
        base[0] = (unsigned char)(packed & 0xFF);
        base[1] = (unsigned char)((packed >> 8) & 0xFF);
        base[2] = (unsigned char)((packed >> 16) & 0xFF);
        return;
    }

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    const int gid = combined_gid - n_kv_heads;
    if (gid >= total_blocks) return;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int elem_offset = head_idx * head_dim + block_idx * 32 + tid;

    float val = v_src[elem_offset];
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    unsigned char* out =
        v_dst + (size_t)pos * total_blocks * 34 + gid * 34;
    if (tid == 0) *((_Float16*)out) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}
