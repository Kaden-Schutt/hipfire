// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// KVarN write-side gather (Phase D1 component #2): transpose a contiguous run of
// token-major K rows into the channel-major `[head_dim × GROUP]` tiles that
// `kvarn_quantize_tile` consumes. K arrives from the forward as
// `[n_tokens × kv_dim]` (kv_dim = n_kv_heads*head_dim, token-major); a KVarN
// record quantizes one kv-head's keys over a 128-token block, oriented
// `[channel(row) × token(col)]`. This kernel does only the gather/transpose —
// the variance-norm + 4-bit pack stays in the shared `kvarn_quantize_tile`.
//
//   T[((b*n_kv_heads + h)*head_dim + ch)*GROUP + tok]
//       = K[(b*GROUP + tok)*kv_dim + h*head_dim + ch]
//
// for b in [0, n_blocks), h in [0, n_kv_heads), ch in [0, head_dim), tok in
// [0, GROUP). The source must hold exactly n_blocks*GROUP contiguous tokens
// (caller slices complete blocks). Grid:[n_blocks*n_kv_heads] (one block per
// output tile), block:[256].

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(256)
extern "C" __global__ void kvarn_gather_k_tiles(
    const float* __restrict__ K,   // [n_blocks*GROUP, kv_dim] token-major f32
    float*       __restrict__ T,   // [n_blocks*n_kv_heads, head_dim*GROUP] f32
    int n_blocks, int n_kv_heads, int head_dim, int group
) {
    const int tile = blockIdx.x;             // (b*n_kv_heads + h)
    if (tile >= n_blocks * n_kv_heads) return;
    const int b = tile / n_kv_heads;
    const int h = tile - b * n_kv_heads;
    const int kv_dim = n_kv_heads * head_dim;
    const int tile_elems = head_dim * group;
    float* tdst = T + (long long)tile * tile_elems;
    const float* ksrc = K + (long long)b * group * kv_dim + h * head_dim;

    for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
        const int ch = i / group;            // row = channel
        const int tok = i - ch * group;      // col = token within block
        tdst[i] = ksrc[(long long)tok * kv_dim + ch];
    }
}
