// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Native wave64 lowering of the MQ G256 activation rotation for CDNA3.
//
// One wave owns one 256-element group. Each lane holds four adjacent values;
// the two register-local FWHT stages cover strides 1 and 2, and six wave64
// XOR exchanges cover strides 4 through 128. This removes the partially
// active wave64 produced by launching the wave32-shaped kernel with 32 threads.
extern "C" __global__ __launch_bounds__(64, 8)
void mq_rotate_x_gfx942(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int lane = threadIdx.x;
    const int groups_total = K / 256;
    const int linear_group = blockIdx.x;
    const int batch = linear_group / groups_total;
    const int group = linear_group - batch * groups_total;
    const long long batch_off = (long long)batch * K;
    const int d0 = lane * 4;
    const int base = group * 256 + d0;
    const float* x_in_b = x_in + batch_off;
    float* x_out_b = x_out + batch_off;

    float v0 = x_in_b[base] * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];

    float t = v0;
    v0 = v0 + v1;
    v1 = t - v1;
    t = v2;
    v2 = v2 + v3;
    v3 = t - v3;

    t = v0;
    v0 = v0 + v2;
    v2 = t - v2;
    t = v1;
    v1 = v1 + v3;
    v3 = t - v3;

#define HBFLY(v, stride) do { \
    const float partner = __shfl_xor((v), (stride), 64); \
    (v) = (lane & (stride)) ? partner - (v) : (v) + partner; \
} while (0)
#define HBFLY4(stride) \
    HBFLY(v0, stride); HBFLY(v1, stride); HBFLY(v2, stride); HBFLY(v3, stride)

    HBFLY4(1);
    HBFLY4(2);
    HBFLY4(4);
    HBFLY4(8);
    HBFLY4(16);
    HBFLY4(32);

#undef HBFLY4
#undef HBFLY

    const float scale = 0.0625f;
    x_out_b[base] = v0 * scale * signs2[d0];
    x_out_b[base + 1] = v1 * scale * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * scale * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * scale * signs2[d0 + 3];
}
