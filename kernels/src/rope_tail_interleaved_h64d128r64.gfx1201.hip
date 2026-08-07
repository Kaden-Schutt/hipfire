// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 DeepSeek V4 indexer-Q tail RoPE.

#include <hip/hip_runtime.h>

// DeepSeek V4 Flash 0731 indexer shape: H=64, D=128, rotary tail=64.
//
// The portable kernel assigns one 32-lane wave to the 32 rotary pairs and
// loops over all 64 heads.  On gfx1201 that leaves the rest of the device idle
// for a latency-visible bookkeeping launch.  This kernel keeps that exact
// arithmetic loop, but stripes its head iterations across independent waves.
// Keeping the loop body identical avoids the 1-ULP drift produced when LLVM
// constant-folds or reschedules a fully specialized per-head body.
extern "C" __global__ void rope_tail_interleaved_h64d128r64_gfx1201(
    float* __restrict__ q,
    const int* __restrict__ pos_buf,
    int head_dim,
    int n_rot,
    float freq_base
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = n_rot / 2;
    if (i >= half) return;

    const int pos = *pos_buf;
    const float freq =
        1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    const float angle = (float)pos * freq;
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);

    const int d_even = head_dim - n_rot + 2 * i;
    const int d_odd = d_even + 1;
    for (int head = blockIdx.y; head < 64; head += gridDim.y) {
        const int base = head * head_dim;
        const float x0 = q[base + d_even];
        const float x1 = q[base + d_odd];
        q[base + d_even] = x0 * cos_a - x1 * sin_a;
        q[base + d_odd] = x0 * sin_a + x1 * cos_a;
    }
}
