// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// DeepSeek V4-clamped fused SwiGLU + FWHT rotation: rotate(silu(clamp(gate)) * clamp(up)).
//
// Replaces the DeepSeek V4 decode sequence:
//   deepseek4_silu_mul_clamp_f32(gate, up, gate, swiglu_limit)  // ~9 µs launch + 8 KB write
//   rotate_x_mq(gate, silu_rot, im)                       // ~9 µs launch + 8 KB read
// with a single kernel. Eliminates the intermediate `gate` buffer
// write/read round-trip (im=2048 → ~16 KB BW saved per layer per token).
//
// DeepSeek V4 asymmetric SwiGLU clamp (matches upstream inference/model.py:Expert.forward):
//   if swiglu_limit > 0:
//     up   = clamp(up,   -limit, +limit)   (symmetric)
//     gate = clamp(gate,         +limit)   (one-sided, cap from above only)
//   out  = silu(gate) * up                 (silu applied AFTER clamp)
//
// Structure mirrors fused_silu_mul_mq_rotate exactly — only Phase 1
// differs (added two-sided clamp on up, one-sided clamp on gate, then
// SwiGLU). Same parallelism: per 256-element group, 32-thread wave per
// group.

extern "C" __launch_bounds__(32, 16)
__global__ void deepseek4_fused_silu_mul_clamp_mq_rotate(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float swiglu_limit
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    // Batch row: grid.y is the batch dim. For decode grid=[groups_total, 1, 1]
    // and blockIdx.y = 0 (byte-identical to pre-batching).
    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
          float* x_rot_b = x_rot + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: read gate + up, apply DeepSeek V4 clamp, compute silu(gate)*up,
    // apply signs1. Asymmetric clamp matches deepseek4_silu_mul_clamp_f32.
    #define CLAMP_GU(g, u) do { \
        if ((g) > swiglu_limit) (g) = swiglu_limit; \
        if ((u) >  swiglu_limit) (u) =  swiglu_limit; \
        if ((u) < -swiglu_limit) (u) = -swiglu_limit; \
    } while(0)
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))

    float g0 = gate_b[base];     float u0 = up_b[base];     CLAMP_GU(g0, u0);
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1]; CLAMP_GU(g1, u1);
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2]; CLAMP_GU(g2, u2);
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3]; CLAMP_GU(g3, u3);
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4]; CLAMP_GU(g4, u4);
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5]; CLAMP_GU(g5, u5);
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6]; CLAMP_GU(g6, u6);
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7]; CLAMP_GU(g7, u7);

    float v0 = SILU_MUL(g0, u0) * signs1[d0];
    float v1 = SILU_MUL(g1, u1) * signs1[d0 + 1];
    float v2 = SILU_MUL(g2, u2) * signs1[d0 + 2];
    float v3 = SILU_MUL(g3, u3) * signs1[d0 + 3];
    float v4 = SILU_MUL(g4, u4) * signs1[d0 + 4];
    float v5 = SILU_MUL(g5, u5) * signs1[d0 + 5];
    float v6 = SILU_MUL(g6, u6) * signs1[d0 + 6];
    float v7 = SILU_MUL(g7, u7) * signs1[d0 + 7];
    #undef SILU_MUL
    #undef CLAMP_GU

    // Local butterfly: strides 1, 2, 4 — identical to mq_rotate_x.
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

    // Wave butterfly via ds_swizzle — same as mq_rotate_x.
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

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

    const float s = 0.0625f;
    x_rot_b[base]     = v0 * s * signs2[d0];
    x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
}
