// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MagnumQuant 2-bit GEMV (Lloyd codebook) with fused residual add.
// y[row] += A[row] · x.
//
// Same body as gemv_mq2g256_lloyd.hip (4-way ternary lookup; baseline for
// non-RDNA3 archs). Only the final write differs (+= instead of =). Stands in
// exactly the same relation to gemv_mq2g256_lloyd.hip that
// gemv_mq3g256_lloyd_residual.hip does to gemv_mq3g256_lloyd.hip, so the
// residual result is bit-identical to `gemv_mq2g256_lloyd` followed by
// `add_inplace_f32` (same accumulation order, same single accumulator).
//
// Differences vs the MQ3-Lloyd residual sibling — these are the ONLY things
// that change between the two formats:
//   * group stride 112 B -> 72 B
//   * codebook 8 × fp16 (16 B) -> 4 × fp16 (8 B), so the packed-index region
//     starts at gptr + 8, not gptr + 16
//   * index width 3 bit -> 2 bit: 64 B of indices per 256-group instead of
//     96 B, so each thread owns 2 bytes (tid*2) instead of 3 (tid*3), and the
//     unpack is a clean 4-per-byte with NO cross-byte straddle — the 24-bit
//     `pk` assembly the MQ3 kernel needs is unnecessary here; the two bytes
//     are decoded independently.
//   * CB_LOOKUP is a 4-way ternary chain instead of an 8-way one.
// Everything else — grid [M,1,1] / block [32,1,1], the row byte stride with
// its (long long) cast, x indexing (g*256 + tid*8, ascending), the single-
// accumulator flat group loop, the 5-step wave32 __shfl_down reduction, and
// the tid==0 write guard — is identical to both siblings.
//
// launch_bounds(32, 20) matches gemv_mq2g256_lloyd.hip; the whole MQ2-Lloyd
// family targets 20 waves/EU, where MQ3-Lloyd targets 16. Occupancy is a perf
// knob only — the math is unaffected. RDNA3 caps at 16 waves/SIMD, so hipcc
// emits a benign "failed to meet occupancy target" -Wpass-failed warning,
// exactly as the existing MQ2-Lloyd kernels already do.
//
// X must be FWHT-256 pre-rotated by the caller (MQ-family contract: rotate X
// once per token, then re-use). This kernel never rotates.
//
// Block layout (72 B / 256 weights, bandwidth-identical to uniform MQ2):
//   bytes [0..8)   : 4 × fp16 codebook entries (sorted ascending at quant time)
//   bytes [8..72)  : 64 B packed 2-bit indices, 4 per byte, little-endian
//                    within the byte — byte i holds weights 4i..4i+4 at bit
//                    offsets 0,2,4,6. (Encoder: hipfire-quantize/src/main.rs,
//                    `byte_val |= (indices[4*i + j] & 0x3) << (j*2)`.)
// Thread tid therefore owns bytes 2*tid and 2*tid+1 = weights 8*tid..8*tid+8,
// which is exactly x[base_idx .. base_idx+8] with base_idx = g*256 + tid*8.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;

        // Load 4 fp16 codebook entries → 4 floats (kept in registers).
        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb0 = __half2float(cb_h[0]);
        float cb1 = __half2float(cb_h[1]);
        float cb2 = __half2float(cb_h[2]);
        float cb3 = __half2float(cb_h[3]);

        const unsigned char* data = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 2;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];

        // Branchless 2-bit codebook lookup via inline switch on q ∈ {0,1,2,3}.
        // Compiler emits a register-resident jump table; no global memory hit.
        #define CB_LOOKUP(q) ((q) == 0 ? cb0 : ((q) == 1 ? cb1 : ((q) == 2 ? cb2 : cb3)))

        acc += CB_LOOKUP(b0 & 3)        * x[base_idx]
             + CB_LOOKUP((b0 >> 2) & 3) * x[base_idx + 1]
             + CB_LOOKUP((b0 >> 4) & 3) * x[base_idx + 2]
             + CB_LOOKUP(b0 >> 6)       * x[base_idx + 3]
             + CB_LOOKUP(b1 & 3)        * x[base_idx + 4]
             + CB_LOOKUP((b1 >> 2) & 3) * x[base_idx + 5]
             + CB_LOOKUP((b1 >> 4) & 3) * x[base_idx + 6]
             + CB_LOOKUP(b1 >> 6)       * x[base_idx + 7];

        #undef CB_LOOKUP
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
