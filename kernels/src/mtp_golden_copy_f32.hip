// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Replay-recordable D2D copy used only to persist outputs from the canonical
// single-token scratch buffers. A raw HIP memcpy is not part of Redline's
// retained launch tape, so this tiny kernel keeps hidden/GDN tape writes
// inside the same ordered PM4 sequence.
extern "C" __global__ void mtp_golden_copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}
