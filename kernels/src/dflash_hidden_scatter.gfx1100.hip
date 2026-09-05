// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact gfx1100 DFlash hidden-ring scatters (S2 launch fusion).
//
// These replace the per-row `memcpy_dtod_at` storms in
// `HiddenStateRingBuffer::commit_staging_to_ring` (5 extracts x up to 2
// wrap segments) and `scatter_hidden_block_to_interleaved`
// ((n_rows - r_skip) x 5 row copies) with one kernel launch each. Pure F32
// copies: every destination element is written exactly once by exactly one
// thread, so the result is bit-identical to the sequential loops regardless
// of execution order. No floating-point arithmetic touches the data, hence
// no reassociation concern.
//
// Both kernels are head-dependent (kernargs bake the current head /
// start_slot), so they must only ever run OUTSIDE any hipGraph capture —
// the Rust launchers refuse capture mode and fall back to the loop.

// commit5: staging[ext][r, :] -> dst[ext][(head + r) % max_pos, :]
// for ext in 0..5, r in 0..n. Matches the loop's head->end + 0->tail split
// copy exactly: (head + r) % max_pos walks head..max_pos-1, 0...
extern "C" __launch_bounds__(256, 1)
__global__ void dflash_hidden_commit5_gfx1100(
    const float* __restrict__ s0,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    const float* __restrict__ s3,
    const float* __restrict__ s4,
    float* __restrict__ d0,
    float* __restrict__ d1,
    float* __restrict__ d2,
    float* __restrict__ d3,
    float* __restrict__ d4,
    int head,
    int n,
    int hidden,
    int max_pos)
{
    const unsigned long long uh = (unsigned long long)hidden;
    const unsigned long long un = (unsigned long long)n;
    const unsigned long long total = 5ULL * un * uh;
    const unsigned long long tid =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (tid >= total) {
        return;
    }
    const unsigned long long c = tid % uh;
    const unsigned long long tmp = tid / uh;
    const unsigned long long r = tmp % un;
    const unsigned long long ext = tmp / un;
    const unsigned long long dst_row =
        ((unsigned long long)head + r) % (unsigned long long)max_pos;
    const unsigned long long src_idx = r * uh + c;
    const unsigned long long dst_idx = dst_row * uh + c;

    // One (src, dst) pair per extract; ext < 5 is enforced by the grid.
    if (ext == 0) {
        d0[dst_idx] = s0[src_idx];
    } else if (ext == 1) {
        d1[dst_idx] = s1[src_idx];
    } else if (ext == 2) {
        d2[dst_idx] = s2[src_idx];
    } else if (ext == 3) {
        d3[dst_idx] = s3[src_idx];
    } else {
        d4[dst_idx] = s4[src_idx];
    }
}

// scatter5: ring[ext][slot, :] -> dst[dst_row, ext, :] for the retained
// rows of the latest block. Logical row r (r_skip <= r < r_skip + rows)
// lives in ring slot (start_slot + (r - r_skip)) % max_pos and lands at
// dst row dst_row_offset + r (absolute) or
// (dst_row_offset + r) % dst_modulus (windowed ring). dst_modulus ==
// 0xFFFFFFFFFFFFFFFF selects the absolute path, mirroring the loop's
// `dst_modulus == usize::MAX` branch. dst is [row, 5, hidden] row-major.
extern "C" __launch_bounds__(256, 1)
__global__ void dflash_hidden_scatter5_gfx1100(
    const float* __restrict__ s0,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    const float* __restrict__ s3,
    const float* __restrict__ s4,
    float* __restrict__ dst,
    unsigned long long dst_row_offset,
    unsigned long long dst_modulus,
    int start_slot,
    int rows,
    int r_skip,
    int hidden,
    int max_pos)
{
    const unsigned long long uh = (unsigned long long)hidden;
    const unsigned long long total = (unsigned long long)rows * 5ULL * uh;
    const unsigned long long tid =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (tid >= total) {
        return;
    }
    const unsigned long long c = tid % uh;
    const unsigned long long tmp = tid / uh;
    const unsigned long long ext = tmp % 5ULL;
    const unsigned long long rr = tmp / 5ULL;
    const unsigned long long r = rr + (unsigned long long)r_skip;
    const unsigned long long slot =
        ((unsigned long long)start_slot + rr) % (unsigned long long)max_pos;
    unsigned long long dst_row;
    if (dst_modulus == 0xFFFFFFFFFFFFFFFFULL) {
        dst_row = dst_row_offset + r;
    } else {
        dst_row = (dst_row_offset + r) % dst_modulus;
    }
    const unsigned long long src_idx = slot * uh + c;
    const unsigned long long dst_idx = dst_row * (5ULL * uh) + ext * uh + c;

    if (ext == 0) {
        dst[dst_idx] = s0[src_idx];
    } else if (ext == 1) {
        dst[dst_idx] = s1[src_idx];
    } else if (ext == 2) {
        dst[dst_idx] = s2[src_idx];
    } else if (ext == 3) {
        dst[dst_idx] = s3[src_idx];
    } else {
        dst[dst_idx] = s4[src_idx];
    }
}
