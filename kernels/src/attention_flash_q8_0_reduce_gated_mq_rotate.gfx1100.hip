// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL
#define HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100
#endif

// Qwen3.5/3.6 decode epilogue sibling. It preserves the portable reducer's
// operation order, applies the existing sigmoid gate, and emits the exact MQ
// rotation expected by the following WO projection.
//
// Keep this gfx1100-only body in its own translation unit: compiling it beside
// the portable reducer perturbs LLVM codegen for gfx12 even though dispatch is
// architecture-gated.
extern "C" __launch_bounds__(256, 4)
__global__ void HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int head_dim,
    const int* __restrict__ pos_buf,
    int tile_size,
    int max_tiles
) {
    const int h = blockIdx.x;
    if (h >= n_heads || head_dim != 256) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = min((seq_len + tile_size - 1) / tile_size, max_tiles);

    extern __shared__ float corr_lut[];
    float* rotate_values = corr_lut + max_tiles;
    __shared__ float global_max_shared;
    __shared__ float global_sum_shared;

    if (tid == 0) {
        float global_max = -1e30f;
        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            if (p[1] > 0.0f)
                global_max = fmaxf(global_max, p[0]);
        }
        global_max_shared = global_max;
    }
    __syncthreads();

    for (int tile = tid; tile < n_tiles; tile += blockDim.x) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        corr_lut[tile] = p[1] > 0.0f ? expf(p[0] - global_max_shared) : 0.0f;
    }
    __syncthreads();

    if (tid == 0) {
        float global_sum = 0.0f;
        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            global_sum += p[1] * corr_lut[tile];
        }
        global_sum_shared = global_sum;
    }
    __syncthreads();

    const float inv_sum = global_sum_shared > 0.0f ? 1.0f / global_sum_shared : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            acc += p[2 + d] * corr_lut[tile];
        }
        const int index = h * head_dim + d;
        const float value = acc * inv_sum;
        const float gate_scale = 1.0f / (1.0f + expf(-gate[index]));
        rotate_values[d] = value * gate_scale;
    }
    __syncthreads();

    // Match mq_rotate_x exactly. One attention head is one 256-value MQ group,
    // and the first wave owns the same eight contiguous values per lane.
    if (tid >= 32) return;
    const int lane = tid;
    const int d0 = lane * 8;
    float v0 = rotate_values[d0]     * signs1[d0];
    float v1 = rotate_values[d0 + 1] * signs1[d0 + 1];
    float v2 = rotate_values[d0 + 2] * signs1[d0 + 2];
    float v3 = rotate_values[d0 + 3] * signs1[d0 + 3];
    float v4 = rotate_values[d0 + 4] * signs1[d0 + 4];
    float v5 = rotate_values[d0 + 5] * signs1[d0 + 5];
    float v6 = rotate_values[d0 + 6] * signs1[d0 + 6];
    float v7 = rotate_values[d0 + 7] * signs1[d0 + 7];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while (0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float scale = 0.0625f;
    float* dst = out + h * head_dim + d0;
    dst[0] = v0 * scale * signs2[d0];
    dst[1] = v1 * scale * signs2[d0 + 1];
    dst[2] = v2 * scale * signs2[d0 + 2];
    dst[3] = v3 * scale * signs2[d0 + 3];
    dst[4] = v4 * scale * signs2[d0 + 4];
    dst[5] = v5 * scale * signs2[d0 + 5];
    dst[6] = v6 * scale * signs2[d0 + 6];
    dst[7] = v7 * scale * signs2[d0 + 7];
}
