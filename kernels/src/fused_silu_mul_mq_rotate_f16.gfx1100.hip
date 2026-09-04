// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused SwiGLU + FWHT rotation with direct FP16 store (gfx1100).
//
// S4-f16-residual-inputs: FFN down producer for the frozen F16 sidecar
// consumed by gemm_mq4g256v2_residual_wmma_f16. Replaces, per FFN (both LA
// and FA layers):
//   fused_silu_mul_mq_rotate_batched  (grid [(K/256)*N], block 32)
//   convert_f32_to_f16                (GEMM prologue)
// with ONE launch. Bit-exact contract: every F16 element must equal the old
// pipeline's F32 store reloaded and cast by convert_f32_to_f16. Phase 1 is
// fused_silu_mul_mq_rotate's non-gfx1030 form verbatim (silu(z) =
// z/(1+exp(-z)), register-only, one element at a time), the butterfly is
// mq_rotate_x verbatim, and the final store uses convert's `(_Float16)`
// cast on the identical F32 value. No FP32 reassociation.
//
// Parallelism is per 256-element group, same as the source kernels — each
// workgroup owns its group; each element's silu is computed exactly once.
//
// Grid: [groups_per_row, N, 1] (groups_per_row = K/256).
// Block: [32, 1, 1]. No LDS (register-only).
extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate_f16_batched_gfx1100(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ out,
    int K)
{
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
    _Float16* out_b = out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: read gate + up, compute silu(gate)*up, apply signs1.
    // silu(z) = z / (1 + exp(-z)). Verbatim from fused_silu_mul_mq_rotate.
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = SILU_MUL(g0, u0) * signs1[d0];
    float v1 = SILU_MUL(g1, u1) * signs1[d0 + 1];
    float v2 = SILU_MUL(g2, u2) * signs1[d0 + 2];
    float v3 = SILU_MUL(g3, u3) * signs1[d0 + 3];
    float v4 = SILU_MUL(g4, u4) * signs1[d0 + 4];
    float v5 = SILU_MUL(g5, u5) * signs1[d0 + 5];
    float v6 = SILU_MUL(g6, u6) * signs1[d0 + 6];
    float v7 = SILU_MUL(g7, u7) * signs1[d0 + 7];
    #undef SILU_MUL

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

    // convert_f32_to_f16's cast on the identical F32 value.
    const float s = 0.0625f;
    out_b[base]     = (_Float16)(v0 * s * signs2[d0]);
    out_b[base + 1] = (_Float16)(v1 * s * signs2[d0 + 1]);
    out_b[base + 2] = (_Float16)(v2 * s * signs2[d0 + 2]);
    out_b[base + 3] = (_Float16)(v3 * s * signs2[d0 + 3]);
    out_b[base + 4] = (_Float16)(v4 * s * signs2[d0 + 4]);
    out_b[base + 5] = (_Float16)(v5 * s * signs2[d0 + 5]);
    out_b[base + 6] = (_Float16)(v6 * s * signs2[d0 + 6]);
    out_b[base + 7] = (_Float16)(v7 * s * signs2[d0 + 7]);
}

// AWQ-aware sibling: divides silu(gate)*up by awq_scale (1D, length K,
// unrotated basis) AFTER the silu*up reduction but BEFORE the signs1 gather
// and FWHT — verbatim from fused_silu_mul_mq_rotate_awq. Completes
// `(W·s)·(silu(g)*u/s) = W·silu(g)*u`. Dispatched only when the consuming
// w_down carries an awq_scale.
extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate_awq_f16_batched_gfx1100(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ out,
    int K)
{
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
    _Float16* out_b = out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: verbatim from fused_silu_mul_mq_rotate_awq.
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = (SILU_MUL(g0, u0) / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (SILU_MUL(g1, u1) / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (SILU_MUL(g2, u2) / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (SILU_MUL(g3, u3) / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (SILU_MUL(g4, u4) / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (SILU_MUL(g5, u5) / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (SILU_MUL(g6, u6) / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (SILU_MUL(g7, u7) / awq_scale[base + 7]) * signs1[d0 + 7];
    #undef SILU_MUL

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
    out_b[base]     = (_Float16)(v0 * s * signs2[d0]);
    out_b[base + 1] = (_Float16)(v1 * s * signs2[d0 + 1]);
    out_b[base + 2] = (_Float16)(v2 * s * signs2[d0 + 2]);
    out_b[base + 3] = (_Float16)(v3 * s * signs2[d0 + 3]);
    out_b[base + 4] = (_Float16)(v4 * s * signs2[d0 + 4]);
    out_b[base + 5] = (_Float16)(v5 * s * signs2[d0 + 5]);
    out_b[base + 6] = (_Float16)(v6 * s * signs2[d0 + 6]);
    out_b[base + 7] = (_Float16)(v7 * s * signs2[d0 + 7]);
}
