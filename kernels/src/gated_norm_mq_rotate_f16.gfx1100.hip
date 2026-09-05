// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched gated RMSNorm + FWHT rotation with direct FP16 store (gfx1100).
//
// S4-f16-residual-inputs: LA post-GDN producer for the frozen F16 sidecar
// consumed by gemm_mq4g256v2_residual_wmma_f16. Replaces, per LA layer:
//   gated_norm_f32_batched  (grid [n_heads, N], block 32)
//   rotate_x_mq_batched     (grid [(K/256)*N], block 32)
//   convert_f32_to_f16      (GEMM prologue)
// with ONE launch. Bit-exact contract: every F16 element must equal the old
// pipeline's F32 store reloaded and cast by convert_f32_to_f16
// (`out[i] = (_Float16)in[i]`). The F32 store/load round trip is exact, so
// computing the identical F32 value in-register (same expression order as
// the two source kernels) and casting with the same `(_Float16)` cast is
// bit-identical. No FP32 reassociation.
//
// Phase A replicates gated_norm_f32's per-lane accumulation and XOR
// reduction exactly (one wave32 per head, head_dim == 128 required, two
// waves per 256-group — the batched form of the decode
// gated_norm_mq_rotate_gfx1100 kernel). Phase B is mq_rotate_x's original
// lane ownership and butterfly verbatim, with the final F32 store replaced
// by the convert kernel's cast.
//
// Grid: [groups_per_row, N, 1] (groups_per_row = K/256 = n_heads/2).
// Block: [64, 1, 1]. LDS: 256 floats (one group).
extern "C" __launch_bounds__(64, 8)
__global__ void gated_norm_mq_rotate_f16_batched_gfx1100(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ out,
    int n_heads,
    int head_dim,
    float eps,
    int K)
{
    if (head_dim != 128) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    // Two heads per 256-group at head_dim == 128.
    if (wave >= 2) return;
    const int group = blockIdx.x;
    const int batch = blockIdx.y;
    const int groups_per_row = K / 256;
    if (group >= groups_per_row) return;
    const int head = group * 2 + wave;
    if (head >= n_heads) return;

    const long long batch_off = (long long)batch * K;
    const float* xh = x + batch_off + (long long)head * head_dim;
    const float* zh = z + batch_off + (long long)head * head_dim;

    __shared__ float normalized[256];
    float* oh = normalized + wave * head_dim;

    // Phase A: match gated_norm_f32's per-lane accumulation and XOR
    // reduction exactly.
    float sq_sum = 0.0f;
    for (int i = lane; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    const float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = lane; i < head_dim; i += 32) {
        const float normed = xh[i] * inv_rms * weight[i];
        const float z_val = zh[i];
        const float silu_z = z_val / (1.0f + expf(-z_val));
        oh[i] = normed * silu_z;
    }
    __syncthreads();

    // Phase B: mq_rotate_x's original lane ownership and butterfly.
    // Each 256-value MQ group spans exactly two normalized heads.
    if (wave != 0) return;
    const int d0 = lane * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(normalized + d0);
    const float4 x1 = *reinterpret_cast<const float4*>(normalized + d0 + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float v0 = x0.x * s10.x;
    float v1 = x0.y * s10.y;
    float v2 = x0.z * s10.z;
    float v3 = x0.w * s10.w;
    float v4 = x1.x * s11.x;
    float v5 = x1.y * s11.y;
    float v6 = x1.z * s11.z;
    float v7 = x1.w * s11.w;

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

    // convert_f32_to_f16's cast, applied to the identical F32 value the old
    // pipeline stored and reloaded.
    const float scale = 0.0625f;
    _Float16* out_b = out + batch_off + (long long)group * 256;
    out_b[d0]     = (_Float16)(v0 * scale * s20.x);
    out_b[d0 + 1] = (_Float16)(v1 * scale * s20.y);
    out_b[d0 + 2] = (_Float16)(v2 * scale * s20.z);
    out_b[d0 + 3] = (_Float16)(v3 * scale * s20.w);
    out_b[d0 + 4] = (_Float16)(v4 * scale * s21.x);
    out_b[d0 + 5] = (_Float16)(v5 * scale * s21.y);
    out_b[d0 + 6] = (_Float16)(v6 * scale * s21.z);
    out_b[d0 + 7] = (_Float16)(v7 * scale * s21.w);
}

// AWQ-aware sibling: divides the gated-norm output by awq_scale (1D,
// length K, unrotated basis) BEFORE the signs1 gather and FWHT — the exact
// mirror of rotate_x_mq_awq's `(x/scale)*signs1`, completing
// `(W·s)·(x/s) = W·x`. Dispatched only when the consuming wo carries an
// awq_scale; byte-identical to the plain symbol when scales are absent
// (which is the only state that file format carries pre-AWQ).
extern "C" __launch_bounds__(64, 8)
__global__ void gated_norm_mq_rotate_awq_f16_batched_gfx1100(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ out,
    int n_heads,
    int head_dim,
    float eps,
    int K)
{
    if (head_dim != 128) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    if (wave >= 2) return;
    const int group = blockIdx.x;
    const int batch = blockIdx.y;
    const int groups_per_row = K / 256;
    if (group >= groups_per_row) return;
    const int head = group * 2 + wave;
    if (head >= n_heads) return;

    const long long batch_off = (long long)batch * K;
    const float* xh = x + batch_off + (long long)head * head_dim;
    const float* zh = z + batch_off + (long long)head * head_dim;
    const long long head_off = (long long)head * head_dim;

    __shared__ float normalized[256];
    float* oh = normalized + wave * head_dim;

    float sq_sum = 0.0f;
    for (int i = lane; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    const float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = lane; i < head_dim; i += 32) {
        const float normed = xh[i] * inv_rms * weight[i];
        const float z_val = zh[i];
        const float silu_z = z_val / (1.0f + expf(-z_val));
        // (normed*silu)/scale: matches rotate_x_mq_awq reading the stored
        // gated-norm output and computing (x/scale)*signs1.
        oh[i] = normed * silu_z / awq_scale[head_off + i];
    }
    __syncthreads();

    if (wave != 0) return;
    const int d0 = lane * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(normalized + d0);
    const float4 x1 = *reinterpret_cast<const float4*>(normalized + d0 + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float v0 = x0.x * s10.x;
    float v1 = x0.y * s10.y;
    float v2 = x0.z * s10.z;
    float v3 = x0.w * s10.w;
    float v4 = x1.x * s11.x;
    float v5 = x1.y * s11.y;
    float v6 = x1.z * s11.z;
    float v7 = x1.w * s11.w;

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
    _Float16* out_b = out + batch_off + (long long)group * 256;
    out_b[d0]     = (_Float16)(v0 * scale * s20.x);
    out_b[d0 + 1] = (_Float16)(v1 * scale * s20.y);
    out_b[d0 + 2] = (_Float16)(v2 * scale * s20.z);
    out_b[d0 + 3] = (_Float16)(v3 * scale * s20.w);
    out_b[d0 + 4] = (_Float16)(v4 * scale * s21.x);
    out_b[d0 + 5] = (_Float16)(v5 * scale * s21.y);
    out_b[d0 + 6] = (_Float16)(v6 * scale * s21.z);
    out_b[d0 + 7] = (_Float16)(v7 * scale * s21.w);
}
