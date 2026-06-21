// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// KVarN read-side build (Phase D1 component #3): materialize a token-major f16
// shadow K cache `[n_tokens × kv_dim]` from the block-tiled KVarN records (full
// 128-token blocks) plus the f32 recent-window staging ring (the trailing
// partial block). This is the dequant+scatter the chosen v1 read path needs:
// the channel-major `[head_dim × GROUP]` records are transposed back to
// token-major so the f16-K / Q8-V flash kernel can read K like any other cache.
//
//   gt = global token index in [0, n_full_blocks*GROUP + tail_len)
//   for gt in a full block b = gt/GROUP, tok = gt%GROUP, head h, channel ch:
//     rec = REC + (b*n_kv_heads + h)*rec_bytes
//     deq = (q[ch,tok]*scale_abs[ch] + zp_abs[ch]) * s_col[tok]
//     OUT[gt*kv_dim + h*head_dim + ch] = (half)deq
//   for gt in the tail (t = gt - n_full_blocks*GROUP):
//     OUT[gt*kv_dim + j] = (half)WINDOW[t*kv_dim + j]      (j over kv_dim)
//
// Record layout matches kvarn_quantize_tile / pack_kvarn_tile. Zero LDS.
// Grid:[n_full_blocks*GROUP + tail_len] (one block per output token), block:[256].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__launch_bounds__(256)
extern "C" __global__ void kvarn_build_kcache(
    const uint8_t* __restrict__ REC,     // block-tiled KVarN records
    const float*   __restrict__ WINDOW,  // [GROUP, kv_dim] f32 staging ring
    __half*        __restrict__ OUT,     // [n_tokens, kv_dim] f16 token-major
    int n_full_blocks, int tail_len,
    int n_kv_heads, int head_dim, int group, int rec_bytes
) {
    const int gt = blockIdx.x;
    const int n_full = n_full_blocks * group;
    if (gt >= n_full + tail_len) return;
    const int kv_dim = n_kv_heads * head_dim;
    const int tile_elems = head_dim * group;
    const int qbytes = (tile_elems + 1) / 2;
    const int off_scale = qbytes;
    const int off_zp = off_scale + head_dim * 2;
    const int off_scol = off_zp + head_dim * 2;

    if (gt < n_full) {
        const int b = gt / group;
        const int tok = gt - b * group;
        for (int j = threadIdx.x; j < kv_dim; j += blockDim.x) {
            const int h = j / head_dim;
            const int ch = j - h * head_dim;
            const uint8_t* rec = REC + (long long)(b * n_kv_heads + h) * rec_bytes;
            const __half* scale_abs = (const __half*)(rec + off_scale);
            const __half* zp_abs = (const __half*)(rec + off_zp);
            const __half* s_col = (const __half*)(rec + off_scol);
            const int i = ch * group + tok;
            const uint8_t byte = rec[i >> 1];
            const float q = (float)((i & 1) == 0 ? (byte & 0xf) : (byte >> 4));
            const float sa = __half2float(scale_abs[ch]);
            const float za = __half2float(zp_abs[ch]);
            const float sc = __half2float(s_col[tok]);
            OUT[(long long)gt * kv_dim + j] = __float2half((q * sa + za) * sc);
        }
    } else {
        const int t = gt - n_full;
        for (int j = threadIdx.x; j < kv_dim; j += blockDim.x) {
            OUT[(long long)gt * kv_dim + j] = __float2half(WINDOW[(long long)t * kv_dim + j]);
        }
    }
}
