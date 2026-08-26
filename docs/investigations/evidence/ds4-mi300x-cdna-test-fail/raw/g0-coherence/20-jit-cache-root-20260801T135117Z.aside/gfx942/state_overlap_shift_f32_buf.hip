// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// On commit, shift the second half of a state ring down to the first
// half: state[:ratio*proj_dim] = state[ratio*proj_dim:2*ratio*proj_dim].
// HIP-graphs-safe twin of the post-commit `memcpy_dtod_auto` shift in
// compressor_forward_impl. Reads commit slot from `commit_slot_buf[0]`;
// -1 sentinel → no-op so the captured graph can include this shift at
// every replay and only fire on real commit positions.
//
// Safe in-place: every thread reads from the upper half (offset
// ratio*proj_dim + idx) and writes to the lower half (offset idx), and
// idx < ratio*proj_dim — no overlap between read and write halves.

#include <hip/hip_runtime.h>

extern "C" __global__ void state_overlap_shift_f32_buf(
    float*       __restrict__ state,            // [2*ratio, proj_dim]
    const int*   __restrict__ commit_slot_buf,  // [1] — -1 = no-op
    int ratio,
    int proj_dim
) {
    const int commit = commit_slot_buf[0];
    if (commit < 0) return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ratio * proj_dim;
    if (idx >= total) return;
    state[idx] = state[total + idx];
}
