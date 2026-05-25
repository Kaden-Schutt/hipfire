// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused `y[i] += c * x[i]` — replaces (scale_f32 + add_inplace_f32) pairs
// in hot paths like the MoE FFN accumulator (shared + routed experts).
//
// Two variants:
//   _cpu_scalar: c is passed as a kernarg f32 (CPU-supplied).
//   _gpu_scalar: c is read from the first element of a GPU buffer,
//                which is what you want when c was computed on-device
//                (e.g. sigmoid of a small gate projection) so you can
//                skip the otherwise-synchronous D2H download.

extern "C" __global__ void scaled_add_inplace_cpu_scalar_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    float c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += c * x[i];
}

extern "C" __global__ void scaled_add_inplace_gpu_scalar_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ c_buf,
    int n
) {
    // All threads read the same scalar — coalesced broadcast from DRAM is
    // fine for a one-element read; the hardware will cache it.
    const float c = c_buf[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += c * x[i];
}
