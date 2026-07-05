// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS batched F32 GQA-4 causal attention.
// Same ABI/math as attention_f32_batched_gqa4 in
// kernels/src/attention_f32_batched.hip: grid
// [n_kv_heads * (kv_group/4), batch_size]; each block computes FOUR query heads
// that share one KV head, loading each K/V row once. The generic kernel stages
// scores[4 * max_ctx_len] in LDS (unbounded). This variant keeps four
// register-resident online-softmax states (m, l, acc) per wave32, zero
// __shared__. One wave32 per (head-quad, row); lane l owns dims { l, l+32, ... }.
// head_dim % 32 == 0.

#include <hip/hip_runtime.h>

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#define HIPFIRE_ATTN_MAX_DL 16

#if HIPFIRE_GFX1103
__device__ __forceinline__ float hipfire_wave_allsum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, 32);
    return v;
}
#endif

extern "C" __launch_bounds__(32, 1)
__global__ void attention_f32_batched_gqa4_gfx1103(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_ctx_len,
    float scale_attn
) {
#if HIPFIRE_GFX1103
    const int group4 = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const int kv_group = n_heads / n_kv_heads;
    const int groups_per_kv = kv_group / 4;
    const int kv_h = group4 / groups_per_kv;
    const int group_in_kv = group4 - kv_h * groups_per_kv;
    const int head_base = kv_h * kv_group + group_in_kv * 4;
    if (kv_h >= n_kv_heads || head_base + 3 >= n_heads) return;

    const int seq_len = positions[b] + 1;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;
    const int kv_off = kv_h * head_dim;

    // Each lane owns dims { lane, lane+32, ... }; 4 query heads share the load.
    float q0[HIPFIRE_ATTN_MAX_DL], q1[HIPFIRE_ATTN_MAX_DL];
    float q2[HIPFIRE_ATTN_MAX_DL], q3[HIPFIRE_ATTN_MAX_DL];
    float a0[HIPFIRE_ATTN_MAX_DL], a1[HIPFIRE_ATTN_MAX_DL];
    float a2[HIPFIRE_ATTN_MAX_DL], a3[HIPFIRE_ATTN_MAX_DL];
    const float* qh0 = q + (size_t)b * q_stride + (head_base + 0) * head_dim;
    const float* qh1 = q + (size_t)b * q_stride + (head_base + 1) * head_dim;
    const float* qh2 = q + (size_t)b * q_stride + (head_base + 2) * head_dim;
    const float* qh3 = q + (size_t)b * q_stride + (head_base + 3) * head_dim;
    int ndl = 0;
    for (int d = lane; d < head_dim; d += 32) {
        q0[ndl] = qh0[d]; q1[ndl] = qh1[d]; q2[ndl] = qh2[d]; q3[ndl] = qh3[d];
        a0[ndl] = 0.0f; a1[ndl] = 0.0f; a2[ndl] = 0.0f; a3[ndl] = 0.0f;
        ndl++;
    }

    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f, m3 = -1e30f;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_cache + (size_t)t * kv_stride + kv_off;
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32) {
            const float kk = k_t[d];
            p0 += q0[di] * kk; p1 += q1[di] * kk; p2 += q2[di] * kk; p3 += q3[di] * kk;
            di++;
        }
        const float s0 = hipfire_wave_allsum32(p0) * scale_attn;
        const float s1 = hipfire_wave_allsum32(p1) * scale_attn;
        const float s2 = hipfire_wave_allsum32(p2) * scale_attn;
        const float s3 = hipfire_wave_allsum32(p3) * scale_attn;

        const float nm0 = fmaxf(m0, s0), nm1 = fmaxf(m1, s1);
        const float nm2 = fmaxf(m2, s2), nm3 = fmaxf(m3, s3);
        const float al0 = __expf(m0 - nm0), al1 = __expf(m1 - nm1);
        const float al2 = __expf(m2 - nm2), al3 = __expf(m3 - nm3);
        const float e0 = __expf(s0 - nm0), e1 = __expf(s1 - nm1);
        const float e2 = __expf(s2 - nm2), e3 = __expf(s3 - nm3);
        l0 = l0 * al0 + e0; l1 = l1 * al1 + e1; l2 = l2 * al2 + e2; l3 = l3 * al3 + e3;

        const float* v_t = v_cache + (size_t)t * kv_stride + kv_off;
        di = 0;
        for (int d = lane; d < head_dim; d += 32) {
            const float vv = v_t[d];
            a0[di] = a0[di] * al0 + e0 * vv;
            a1[di] = a1[di] * al1 + e1 * vv;
            a2[di] = a2[di] * al2 + e2 * vv;
            a3[di] = a3[di] * al3 + e3 * vv;
            di++;
        }
        m0 = nm0; m1 = nm1; m2 = nm2; m3 = nm3;
    }

    float* o0 = out + (size_t)b * q_stride + (head_base + 0) * head_dim;
    float* o1 = out + (size_t)b * q_stride + (head_base + 1) * head_dim;
    float* o2 = out + (size_t)b * q_stride + (head_base + 2) * head_dim;
    float* o3 = out + (size_t)b * q_stride + (head_base + 3) * head_dim;
    const float i0 = 1.0f / l0, i1 = 1.0f / l1, i2 = 1.0f / l2, i3 = 1.0f / l3;
    int di = 0;
    for (int d = lane; d < head_dim; d += 32) {
        o0[d] = a0[di] * i0; o1[d] = a1[di] * i1; o2[d] = a2[di] * i2; o3[d] = a3[di] * i3;
        di++;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = q[0] + k_cache[0] + v_cache[0] + (float)positions[0] + scale_attn
            + (float)(n_kv_heads + head_dim + max_ctx_len);
#endif
}
