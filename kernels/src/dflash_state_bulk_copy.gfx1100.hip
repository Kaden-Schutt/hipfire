// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 (launch-fusion): descriptor-driven DeltaNet snapshot bulk copy (gfx1100).
//!
//! Replaces the 384-per-cycle `hipMemcpyDtoD` storm in
//! `DeltaNetSnapshot::{save_from, restore_to}` (48 LA layers x S/scale/conv/EF)
//! with two fixed-grid launches per decode cycle: one consuming the persistent
//! forward (live -> backup) descriptor table for save, one consuming the
//! reverse (backup -> live) table for restore.
//!
//! Each work item copies `cnt` bytes from `src + off` to `dst + off`. Tables
//! are built once at snapshot allocation with 64-KiB-aligned chunk offsets, so
//! every vector lane is 16 B aligned. The kernel is a pure byte copy: no
//! atomics, no cross-item communication, disjoint ranges — bit-exact and
//! deterministic by construction, safe for hipGraph capture and Redline tape
//! replay. One block per work item, 256 threads, 16 B `float4` vector loop
//! with a scalar byte tail for non-multiple-of-16 counts.

#include <hip/hip_runtime.h>

struct DflashStateCopyDesc {
    unsigned long long src;
    unsigned long long dst;
    unsigned long long off;
    unsigned long long cnt;
};

extern "C" __global__ void dflash_state_bulk_copy_gfx1100(
    const struct DflashStateCopyDesc* __restrict__ desc,
    unsigned int n_items
) {
    unsigned int b = blockIdx.x;
    if (b >= n_items) return;
    struct DflashStateCopyDesc d = desc[b];
    if (d.cnt == 0) return;
    const unsigned char* __restrict__ s =
        (const unsigned char*)d.src + d.off;
    unsigned char* __restrict__ t =
        (unsigned char*)d.dst + d.off;
    unsigned int tid = threadIdx.x;
    unsigned int nt = blockDim.x;
    // 16 B vector body. Pointers are 16 B aligned (hipMalloc base alignment
    // plus 64-KiB-aligned chunk offsets) so float4 traffic is aligned.
    unsigned long long vec_n = d.cnt >> 4;
    for (unsigned long long i = tid; i < vec_n; i += nt) {
        ((float4*)t)[i] = ((const float4*)s)[i];
    }
    // Scalar tail for counts that are not a multiple of 16.
    unsigned long long done = vec_n << 4;
    for (unsigned long long i = done + tid; i < d.cnt; i += nt) {
        t[i] = s[i];
    }
}
