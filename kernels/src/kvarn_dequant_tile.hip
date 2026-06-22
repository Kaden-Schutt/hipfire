// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// KVarN tile dequantizer (Phase D1 read-side): unpack an on-device KVarN record
// → f16 tile, the input the reused asym4/q8 flash-attention kernels consume.
// deq[r,c] = (q*scale_abs[r] + zp_abs[r]) * s_col[c]. One block per tile, no
// Sinkhorn (pure read) — fully parallel, zero LDS.
//
// Record layout (== kvarn_quantize_tile / pack_kvarn_tile_bits):
//   [ (r*c)/(8/bits) ]  q codes, 8/bits per byte LSB-first (bits=4 = 2 nibbles/byte)
//   [ r_dim*2  ]   scale_abs[r] fp16
//   [ r_dim*2  ]   zp_abs[r]    fp16
//   [ c_dim*2  ]   s_col[c]     fp16
//
// Grid:[n_tiles], block:[256]. OUT is [n_tiles, r_dim*c_dim] __half row-major.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__launch_bounds__(256)
extern "C" __global__ void kvarn_dequant_tile(
    const uint8_t* __restrict__ REC,   // [n_tiles, record_bytes]
    __half*        __restrict__ OUT,    // [n_tiles, r_dim*c_dim]
    int n_tiles, int r_dim, int c_dim, int record_bytes,
    int bits // bits per code; codes/byte = 8/bits. 4 = legacy nibble layout, 2 = 2-bit.
) {
    const int t = blockIdx.x;
    if (t >= n_tiles) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const uint8_t* rec = REC + (long long)t * record_bytes;
    __half* out = OUT + (long long)t * r_dim * c_dim;

    const int n = r_dim * c_dim;
    const int cpb = 8 / bits;              // codes per byte
    const int mask = (1 << bits) - 1;
    const int qbytes = (n + cpb - 1) / cpb;
    const int off_scale = qbytes;
    const int off_zp = off_scale + r_dim * 2;
    const int off_scol = off_zp + r_dim * 2;
    const __half* scale_abs = (const __half*)(rec + off_scale);
    const __half* zp_abs = (const __half*)(rec + off_zp);
    const __half* s_col = (const __half*)(rec + off_scol);

    for (int i = tid; i < n; i += nth) {
        const int r = i / c_dim;
        const int c = i - r * c_dim;
        const uint8_t byte = rec[i / cpb];
        const float q = (float)((byte >> ((i % cpb) * bits)) & mask);
        const float sa = __half2float(scale_abs[r]);
        const float za = __half2float(zp_abs[r]);
        const float sc = __half2float(s_col[c]);
        out[i] = __float2half((q * sa + za) * sc);
    }
}
