// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Consumer-fold lever (HIPFIRE_QKVZA_FUSEDNORM=1): fused_gate_up_hfq4g256
// with RMSNorm + AWQ divide + FWHT rotation inline. Bit-exact vs the direct
// producer pair - same contract as fused_qkvza_hfq4g256_fusednorm.gfx1100.hip
// (reduction-tree slot mapping, elementwise order, per-group DOG into
// acc[g&3], tail semantics preserved by the g&3 mapping).

__launch_bounds__(32, 12)
extern "C" __global__ void fused_gate_up_hfq4g256_fusednorm(
    const char*  __restrict__ A_gate,
    const char*  __restrict__ A_up,
    const float* __restrict__ x_raw,
    const float* __restrict__ gamma,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float*       __restrict__ y_gate,
    float*       __restrict__ y_up,
    int gate_m, int up_m,
    int K, float eps
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; row = gid;
    } else {
        A = A_up;   y = y_up;   row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
    const int d0 = tid * 8;

    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    float p4 = 0.f, p5 = 0.f, p6 = 0.f, p7 = 0.f;
    for (int g = 0; g < groups_per_row; g++) {
        const float4 xa = *reinterpret_cast<const float4*>(x_raw + g * 256 + d0);
        const float4 xb = *reinterpret_cast<const float4*>(x_raw + g * 256 + d0 + 4);
        p0 += xa.x * xa.x; p1 += xa.y * xa.y; p2 += xa.z * xa.z; p3 += xa.w * xa.w;
        p4 += xb.x * xb.x; p5 += xb.y * xb.y; p6 += xb.z * xb.z; p7 += xb.w * xb.w;
    }
#pragma unroll
    for (int s = 128; s >= 8; s >>= 1) {
        const int dl = s >> 3;
        const float q0 = __shfl(p0, tid + dl);
        const float q1 = __shfl(p1, tid + dl);
        const float q2 = __shfl(p2, tid + dl);
        const float q3 = __shfl(p3, tid + dl);
        const float q4 = __shfl(p4, tid + dl);
        const float q5 = __shfl(p5, tid + dl);
        const float q6 = __shfl(p6, tid + dl);
        const float q7 = __shfl(p7, tid + dl);
        if (tid * 8 < (unsigned)s) { p0 += q0; p1 += q1; p2 += q2; p3 += q3; p4 += q4; p5 += q5; p6 += q6; p7 += q7; }
    }
    if (tid == 0) { p0 += p4; p1 += p5; p2 += p6; p3 += p7; }
    if (tid == 0) { p0 += p2; p1 += p3; }
    if (tid == 0) { p0 += p1; }
    const float sum_sq = __shfl(p0, 0);
    const float rms = rsqrtf(sum_sq / (float)K + eps);

    // Debug tap (probe): negative eps -> emit rms from row-0 blocks.
    if (eps < 0.0f) {
        if (row == 0 && tid == 0) {
            y[0] = rms;
        }
        return;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#define HBFLY(v, pat, str) do { \
    float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
    if ((tid) & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
} while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const int base = g * 256 + d0;
        const float4 xa = *reinterpret_cast<const float4*>(x_raw + base);
        const float4 xb = *reinterpret_cast<const float4*>(x_raw + base + 4);
        const float4 wa = *reinterpret_cast<const float4*>(gamma + base);
        const float4 wb = *reinterpret_cast<const float4*>(gamma + base + 4);
        const float4 aa = *reinterpret_cast<const float4*>(awq_scale + base);
        const float4 ab = *reinterpret_cast<const float4*>(awq_scale + base + 4);

        float v0 = xa.x * wa.x * rms / aa.x * s10.x;
        float v1 = xa.y * wa.y * rms / aa.y * s10.y;
        float v2 = xa.z * wa.z * rms / aa.z * s10.z;
        float v3 = xa.w * wa.w * rms / aa.w * s10.w;
        float v4 = xb.x * wb.x * rms / ab.x * s11.x;
        float v5 = xb.y * wb.y * rms / ab.y * s11.y;
        float v6 = xb.z * wb.z * rms / ab.z * s11.z;
        float v7 = xb.w * wb.w * rms / ab.w * s11.w;

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

        HBFLY(v0, 0x041F, 1); HBFLY(v1, 0x041F, 1); HBFLY(v2, 0x041F, 1); HBFLY(v3, 0x041F, 1);
        HBFLY(v4, 0x041F, 1); HBFLY(v5, 0x041F, 1); HBFLY(v6, 0x041F, 1); HBFLY(v7, 0x041F, 1);
        HBFLY(v0, 0x081F, 2); HBFLY(v1, 0x081F, 2); HBFLY(v2, 0x081F, 2); HBFLY(v3, 0x081F, 2);
        HBFLY(v4, 0x081F, 2); HBFLY(v5, 0x081F, 2); HBFLY(v6, 0x081F, 2); HBFLY(v7, 0x081F, 2);
        HBFLY(v0, 0x101F, 4); HBFLY(v1, 0x101F, 4); HBFLY(v2, 0x101F, 4); HBFLY(v3, 0x101F, 4);
        HBFLY(v4, 0x101F, 4); HBFLY(v5, 0x101F, 4); HBFLY(v6, 0x101F, 4); HBFLY(v7, 0x101F, 4);
        HBFLY(v0, 0x201F, 8); HBFLY(v1, 0x201F, 8); HBFLY(v2, 0x201F, 8); HBFLY(v3, 0x201F, 8);
        HBFLY(v4, 0x201F, 8); HBFLY(v5, 0x201F, 8); HBFLY(v6, 0x201F, 8); HBFLY(v7, 0x201F, 8);
        HBFLY(v0, 0x401F, 16); HBFLY(v1, 0x401F, 16); HBFLY(v2, 0x401F, 16); HBFLY(v3, 0x401F, 16);
        HBFLY(v4, 0x401F, 16); HBFLY(v5, 0x401F, 16); HBFLY(v6, 0x401F, 16); HBFLY(v7, 0x401F, 16);

        const float xr0 = v0 * 0.0625f * s20.x;
        const float xr1 = v1 * 0.0625f * s20.y;
        const float xr2 = v2 * 0.0625f * s20.z;
        const float xr3 = v3 * 0.0625f * s20.w;
        const float xr4 = v4 * 0.0625f * s21.x;
        const float xr5 = v5 * 0.0625f * s21.y;
        const float xr6 = v6 * 0.0625f * s21.z;
        const float xr7 = v7 * 0.0625f * s21.w;

        const char* gp = row_ptr + g * 136;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        float& acc = (g & 3) == 0 ? acc0 : (g & 3) == 1 ? acc1 : (g & 3) == 2 ? acc2 : acc3;
        acc += (sc * (float)((pk) & 0xFu)         + zp) * xr0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xr1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xr2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xr3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xr4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xr5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xr6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xr7;
    }
#undef HBFLY

    float acc = (acc0 + acc1) + (acc2 + acc3);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        y[row] = acc;
    }
}
