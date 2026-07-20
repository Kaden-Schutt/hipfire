// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define LFM2_CONV_KMAX 8

// Exact LFM2 single-token conv epilogue followed by the MQ4-G256 activation
// rotation expected by out_proj. One wave owns one 256-channel MQ group; each
// lane computes the same eight channel-local conv results as
// conv1d_gated_decode_f32, then applies mq_rotate_x's butterfly unchanged.
extern "C" __launch_bounds__(32, 16)
__global__ void conv1d_gated_decode_mq_rotate_f32(
    const float* __restrict__ bcx,
    float* __restrict__ state,
    const float* __restrict__ weight,
    float* __restrict__ out_rot,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int B, int C, int K)
{
    if (B != 1 || (C & 255) != 0 || K < 1 || K > LFM2_CONV_KMAX) return;
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    const int groups = C / 256;
    if (group >= groups || lane >= 32) return;

    const int d0 = lane * 8;
    const int base = group * 256 + d0;
    const int hist = K - 1;
    float value[8];

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = base + j;
        const float b_gate = bcx[c];
        const float c_gate = bcx[C + c];
        const float x_in = bcx[2 * C + c];
        const float bx = b_gate * x_in;
        const long long sbase = (long long)c * hist;

        float win[LFM2_CONV_KMAX];
        for (int t = 0; t < hist; ++t) win[t] = state[sbase + t];
        win[hist] = bx;
        float acc = 0.0f;
        for (int t = 0; t < K; ++t) acc += win[t] * weight[c * K + t];
        value[j] = c_gate * acc;

        for (int t = 0; t < hist - 1; ++t) state[sbase + t] = state[sbase + t + 1];
        if (hist > 0) state[sbase + hist - 1] = bx;
    }

    float v0 = value[0] * signs1[d0];
    float v1 = value[1] * signs1[d0 + 1];
    float v2 = value[2] * signs1[d0 + 2];
    float v3 = value[3] * signs1[d0 + 3];
    float v4 = value[4] * signs1[d0 + 4];
    float v5 = value[5] * signs1[d0 + 5];
    float v6 = value[6] * signs1[d0 + 6];
    float v7 = value[7] * signs1[d0 + 7];

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
    out_rot[base] = v0 * scale * signs2[d0];
    out_rot[base + 1] = v1 * scale * signs2[d0 + 1];
    out_rot[base + 2] = v2 * scale * signs2[d0 + 2];
    out_rot[base + 3] = v3 * scale * signs2[d0 + 3];
    out_rot[base + 4] = v4 * scale * signs2[d0 + 4];
    out_rot[base + 5] = v5 * scale * signs2[d0 + 5];
    out_rot[base + 6] = v6 * scale * signs2[d0 + 6];
    out_rot[base + 7] = v7 * scale * signs2[d0 + 7];
}
