// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Software-prefetch: reads the first cache line of each active expert's
// weight slab into L2, so a downstream GEMM accessing those slabs incurs
// L2 hits instead of HBM misses.
//
// expert_ptrs: [n_exp] u64 device pointer table (one per expert)
// expert_ids : [n_ids] i32 tile-ID or topk-index list; negative entries skipped
// n_ids      : number of entries in expert_ids
// prefetch_bytes: bytes per slab to touch (multiple of cache-line = 128)
//
// Each thread iterates a strided range of expert_ids, loads the first
// `prefetch_bytes` of the corresponding weight slab, and discards the
// result. The load fills L1+L2 on RDNA3.5.
extern "C" __global__ void prefetch_expert_slabs(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ expert_ids,
    int n_ids,
    int prefetch_bytes
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    for (int i = tid; i < n_ids; i += bdim) {
        int e = expert_ids[i];
        if (e >= 0) {
            const volatile unsigned char* slab =
                reinterpret_cast<const volatile unsigned char*>(expert_ptrs[e]);
            // Touch one cache line (128 B) at the start of the slab.
            // A single 4-byte load brings in the full cache line.
            (void)slab[0];
            (void)slab[64];
        }
    }
}
