// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// FWHT-128 activation rotation for MQ4G128 dispatch path.
// Mirrors mq_rotate_x in gemv_mq4g256.hip with N=128 instead of 256.
// Used by Gpu::rotate_x_mq_128 in crates/rdna-compute/src/dispatch.rs.
// Weight encoding (CPU side) MUST use cpu_fwht_128 with the SAME sign seeds (43, 1043).

#include <hip/hip_runtime.h>

// MagnumQuant GEMV: FWHT-rotated HFQ4-G128.
// x_rot is PRE-ROTATED by the caller (one FWHT pass on x, not per-row).
// The inner loop is identical to standard HFQ4-G128 — zero rotation overhead.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g128(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,   // PRE-ROTATED x (FWHT already applied)
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;  // 4 + 4 + 64 bytes per group (scale + zero + 64 nibble bytes)
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_idx = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        acc += (scale * (float)(b0 & 0xF) + zero) * x_rot[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x_rot[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x_rot[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x_rot[base_idx + 3];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// Rotate x in-place: applies FWHT to each 128-element group.
// Called ONCE before the GEMV, not per-row.
// 32 threads per group, 4 elements per thread, ds_swizzle butterfly.
//
// Grid: [groups_total, batch, 1]. For decode the caller passes batch=1 and
// blockIdx.y is always 0 (byte-identical to the pre-batching version).
// For prefill, batch=N; each block processes one group × one batch row.
// x_in / x_out are laid out as [batch × K] row-major.
extern "C" __launch_bounds__(32, 16)
__global__ void mq_rotate_x_128(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 128;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* x_in_b  = x_in  + batch_off;
          float* x_out_b = x_out + batch_off;

    int d0 = tid * 4;
    int base = group * 128 + d0;

    float v0 = x_in_b[base]     * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];

    // Local butterfly: strides 1, 2
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle
    // (covers element strides 4, 8, 16, 32, 64 — completes the FWHT-128)
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY4(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str)

    HBFLY4(0x041F, 1);
    HBFLY4(0x081F, 2);
    HBFLY4(0x101F, 4);
    HBFLY4(0x201F, 8);
    HBFLY4(0x401F, 16);
    #undef HBFLY4
    #undef HBFLY

    const float s = 0.0883883476f;  // 1/sqrt(128)
    x_out_b[base]     = v0 * s * signs2[d0];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
}
