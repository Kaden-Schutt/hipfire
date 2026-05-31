// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Ring write of a single per-position projection into a per-layer state
// ring buffer. HIP-graphs-safe twin of the
// `memcpy_dtod_auto(state.sub_offset(slot*proj_dim, ...), buf, ...)`
// pattern in compressor_forward_impl. The ring slot index changes every
// position (overlap=true: slot ∈ [ratio, 2*ratio), overlap=false: slot ∈
// [0, ratio)) so the offset cannot be baked at capture time. Reading
// `ring_slot_buf[0]` keeps the captured graph correct on replay.
//
// -1 sentinel is accepted (host can use it to skip), though the ring
// write fires at EVERY position in the current compressor path.

#include <hip/hip_runtime.h>

extern "C" __global__ void state_ring_write_f32_buf(
    const float* __restrict__ src,            // [proj_dim]
    float*       __restrict__ state,          // [state_rows, proj_dim]
    const int*   __restrict__ ring_slot_buf,  // [1]
    int proj_dim
) {
    const int slot = ring_slot_buf[0];
    if (slot < 0) return;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= proj_dim) return;
    state[(size_t)slot * proj_dim + d] = src[d];
}
