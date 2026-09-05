// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused sigmoid(gate)*attn + FWHT rotation with direct FP16 store (gfx1100).
//
// S4-f16-residual-inputs: FA post-attention producer for the frozen F16
// sidecar consumed by gemm_mq4g256v2_residual_wmma_f16. Replaces, per FA
// layer:
//   sigmoid_mul_f32       (in-place out[i] *= sigmoid(gate[i]), numel kernel)
//   rotate_x_mq_batched   (grid [(K/256)*N], block 32)
//   convert_f32_to_f16    (GEMM prologue)
// with ONE launch. Bit-exact contract: every F16 element must equal the old
// pipeline's F32 store reloaded and cast by convert_f32_to_f16. The F32
// store/load round trip is exact, so computing the identical F32 value
// in-register (same sigmoid formula as sigmoid_mul_f32, same signs1 gather
// and butterfly as mq_rotate_x) and casting with the same `(_Float16)` cast
// is bit-identical. No FP32 reassociation.
//
// NOTE: unlike the old path, this kernel does NOT mutate the attn input
// (the old in-place sigmoid write is skipped). The S4 state contract keeps
// F32 inputs live; nothing downstream reads the sigmoided F32 values — the
// residual GEMM consumes only the sidecar.
//
// Parallelism is per 256-element group, same as mq_rotate_x — each
// workgroup owns its group, reads attn+gate once, and applies the per-group
// butterfly. Each element's sigmoid is computed exactly once.
//
// Grid: [groups_per_row, N, 1] (groups_per_row = K/256).
// Block: [32, 1, 1]. No LDS (register-only, like mq_rotate_x).
extern "C" __launch_bounds__(32, 16)
__global__ void sigmoid_mul_mq_rotate_f16_batched_gfx1100(
    const float* __restrict__ attn,
    const float* __restrict__ gate,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ out,
    int K)
{
    const int groups_total = K / 256;
    const int group = blockIdx.x;
    if (group >= groups_total) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* attn_b = attn + batch_off;
    const float* gate_b = gate + batch_off;
    _Float16* out_b = out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: exact sigmoid_mul_f32 element order (`s = 1/(1+exp(-g))`,
    // `m = a*s`), then the mq_rotate_x signs1 gather.
    float a0 = attn_b[base];     float g0 = gate_b[base];
    float a1 = attn_b[base + 1]; float g1 = gate_b[base + 1];
    float a2 = attn_b[base + 2]; float g2 = gate_b[base + 2];
    float a3 = attn_b[base + 3]; float g3 = gate_b[base + 3];
    float a4 = attn_b[base + 4]; float g4 = gate_b[base + 4];
    float a5 = attn_b[base + 5]; float g5 = gate_b[base + 5];
    float a6 = attn_b[base + 6]; float g6 = gate_b[base + 6];
    float a7 = attn_b[base + 7]; float g7 = gate_b[base + 7];

    float s0 = 1.0f / (1.0f + expf(-g0));
    float s1 = 1.0f / (1.0f + expf(-g1));
    float s2 = 1.0f / (1.0f + expf(-g2));
    float s3 = 1.0f / (1.0f + expf(-g3));
    float s4 = 1.0f / (1.0f + expf(-g4));
    float s5 = 1.0f / (1.0f + expf(-g5));
    float s6 = 1.0f / (1.0f + expf(-g6));
    float s7 = 1.0f / (1.0f + expf(-g7));

    float v0 = (a0 * s0) * signs1[d0];
    float v1 = (a1 * s1) * signs1[d0 + 1];
    float v2 = (a2 * s2) * signs1[d0 + 2];
    float v3 = (a3 * s3) * signs1[d0 + 3];
    float v4 = (a4 * s4) * signs1[d0 + 4];
    float v5 = (a5 * s5) * signs1[d0 + 5];
    float v6 = (a6 * s6) * signs1[d0 + 6];
    float v7 = (a7 * s7) * signs1[d0 + 7];

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
    const float sc = 0.0625f;
    out_b[base]     = (_Float16)(v0 * sc * signs2[d0]);
    out_b[base + 1] = (_Float16)(v1 * sc * signs2[d0 + 1]);
    out_b[base + 2] = (_Float16)(v2 * sc * signs2[d0 + 2]);
    out_b[base + 3] = (_Float16)(v3 * sc * signs2[d0 + 3]);
    out_b[base + 4] = (_Float16)(v4 * sc * signs2[d0 + 4]);
    out_b[base + 5] = (_Float16)(v5 * sc * signs2[d0 + 5]);
    out_b[base + 6] = (_Float16)(v6 * sc * signs2[d0 + 6]);
    out_b[base + 7] = (_Float16)(v7 * sc * signs2[d0 + 7]);
}

// AWQ-aware sibling: divides the sigmoided product by awq_scale (1D,
// length K, unrotated basis) BEFORE the signs1 gather — the exact mirror of
// rotate_x_mq_awq's `(x/scale)*signs1`. Dispatched only when the consuming
// wo carries an awq_scale.
extern "C" __launch_bounds__(32, 16)
__global__ void sigmoid_mul_mq_rotate_awq_f16_batched_gfx1100(
    const float* __restrict__ attn,
    const float* __restrict__ gate,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ out,
    int K)
{
    const int groups_total = K / 256;
    const int group = blockIdx.x;
    if (group >= groups_total) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* attn_b = attn + batch_off;
    const float* gate_b = gate + batch_off;
    _Float16* out_b = out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float a0 = attn_b[base];     float g0 = gate_b[base];
    float a1 = attn_b[base + 1]; float g1 = gate_b[base + 1];
    float a2 = attn_b[base + 2]; float g2 = gate_b[base + 2];
    float a3 = attn_b[base + 3]; float g3 = gate_b[base + 3];
    float a4 = attn_b[base + 4]; float g4 = gate_b[base + 4];
    float a5 = attn_b[base + 5]; float g5 = gate_b[base + 5];
    float a6 = attn_b[base + 6]; float g6 = gate_b[base + 6];
    float a7 = attn_b[base + 7]; float g7 = gate_b[base + 7];

    float s0 = 1.0f / (1.0f + expf(-g0));
    float s1 = 1.0f / (1.0f + expf(-g1));
    float s2 = 1.0f / (1.0f + expf(-g2));
    float s3 = 1.0f / (1.0f + expf(-g3));
    float s4 = 1.0f / (1.0f + expf(-g4));
    float s5 = 1.0f / (1.0f + expf(-g5));
    float s6 = 1.0f / (1.0f + expf(-g6));
    float s7 = 1.0f / (1.0f + expf(-g7));

    // ((a*s)/scale)*signs1: matches rotate_x_mq_awq reading the stored
    // sigmoided product and computing (x/scale)*signs1.
    float v0 = ((a0 * s0) / awq_scale[base])     * signs1[d0];
    float v1 = ((a1 * s1) / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = ((a2 * s2) / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = ((a3 * s3) / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = ((a4 * s4) / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = ((a5 * s5) / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = ((a6 * s6) / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = ((a7 * s7) / awq_scale[base + 7]) * signs1[d0 + 7];

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

    const float sc = 0.0625f;
    out_b[base]     = (_Float16)(v0 * sc * signs2[d0]);
    out_b[base + 1] = (_Float16)(v1 * sc * signs2[d0 + 1]);
    out_b[base + 2] = (_Float16)(v2 * sc * signs2[d0 + 2]);
    out_b[base + 3] = (_Float16)(v3 * sc * signs2[d0 + 3]);
    out_b[base + 4] = (_Float16)(v4 * sc * signs2[d0 + 4]);
    out_b[base + 5] = (_Float16)(v5 * sc * signs2[d0 + 5]);
    out_b[base + 6] = (_Float16)(v6 * sc * signs2[d0 + 6]);
    out_b[base + 7] = (_Float16)(v7 * sc * signs2[d0 + 7]);
}
