// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Dequantize an MFP4G32-Lloyd matrix [M × K] into FP16 [M × K] row-major.
//
// MFP4G32-Lloyd byte layout (== MFP4G32 rows + ONE per-TENSOR 16-entry fp16
// codebook prepended ONCE before row 0):
//
//   bytes [0 .. 32)                : 16 × fp16 codebook entries (the per-tensor
//                                    Lloyd-Max centroids, sorted ascending at
//                                    quant time). Replaces the fixed E2M1 grid.
//   bytes [32 .. 32 + M*row_bytes) : M rows, each row byte-identical to MFP4G32:
//       per-row 16-B header (row_scale_a:f16 @0, row_scale_b:f16 @2,
//                            block_count:u16 @4, flags:u8 @6=0x05, ...)
//       then (K/32) blocks × 17 B (UE8M0 block_e:u8 + 16 B packed nibbles).
//
// Per element reconstruction (the ONLY difference from MFP4G32 is cb[] vs E2M1):
//   value = row_scale_a * 2^(block_e - 127) * codebook[nibble]
//
// The weights are stored FWHT-rotated (rotation flag 0x05); this kernel emits
// the rotated, codebook-reconstructed FP16 weight. Used as the CDNA3 dequant-
// to-f16 utility (round-trip gate + any rocBLAS/MFMA consumer). The FWHT-on-x
// is handled by the existing mfp4 rotation path; this kernel does NOT rotate.
//
// Launch: grid = (M, K/256, 1), block = (32, 1, 1). Each of the 32 threads
// dequants 8 elements (one block of an 8-block, 256-element "group").
//   block_in_iter = tid >> 2   (0..7)  which block of 8 this thread serves
//   nib_off       = (tid & 3) * 4      byte offset into the block's 16-B nibbles
//   K_in_group    = tid * 8            first element index within the group
extern "C" __launch_bounds__(32, 16)
__global__ void dequantize_mfp4g32_lloyd_to_f16(
    const char*  __restrict__ A,      // [32 codebook][M × row_bytes]
    _Float16*    __restrict__ W_f16,  // [M × K] row-major
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // ── Per-tensor codebook (32-B prefix), loaded into LDS once per block. ──
    // 16 fp16 entries. Use built-in `_Float16` to avoid <hip/hip_fp16.h>
    // (its CUDA-shim complex_builtins has a max/fmax decl bug in ROCm 7.2.2).
    __shared__ _Float16 cb[16];
    if (tid < 16) {
        const unsigned short* cb_bits = (const unsigned short*)(A);
        cb[tid] = __builtin_bit_cast(_Float16, cb_bits[tid]);
    }
    __syncthreads();

    const int groups_per_row = K / 256;            // 8 HFP4 blocks each
    const int row_block_bytes = (K / 32) * 17;     // = groups_per_row * 136
    // +32: skip the per-tensor codebook prefix before row 0.
    const long long row_off =
        32LL + (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                // first block of this row

    // FP16 row scale lives in the first 2 bytes of the per-row header.
    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    // This thread's block within the 8-block group + its 4-byte nibble window.
    const int block_in_iter = tid >> 2;            // 0..7
    const int nib_off = (tid & 3) * 4;             // 0/4/8/12

    // 136-B group stride; this thread's block at +17*block_in_iter.
    const char* gp = row_ptr + (long long)group * 136 + block_in_iter * 17;
    const unsigned char block_e = *(const unsigned char*)(gp);

    // UE8M0 dequant: 2^(block_e - 127) is the FP32 with biased exp = block_e,
    // mantissa 0 — pure bit construction (portable, no device ldexpf).
    const float sc =
        row_scale_f * __builtin_bit_cast(float, ((unsigned int)block_e) << 23);

    const unsigned int pk = *(const unsigned int*)(gp + 1 + nib_off);

    // First output element index for this thread within the full row:
    //   base = group*256 + block_in_iter*32 + (tid&3)*8 == group*256 + tid*8
    _Float16* out = W_f16 + (long long)row * K + (long long)group * 256 + tid * 8;

    out[0] = (_Float16)(sc * (float)cb[(pk      ) & 0xFu]);
    out[1] = (_Float16)(sc * (float)cb[(pk >>  4) & 0xFu]);
    out[2] = (_Float16)(sc * (float)cb[(pk >>  8) & 0xFu]);
    out[3] = (_Float16)(sc * (float)cb[(pk >> 12) & 0xFu]);
    out[4] = (_Float16)(sc * (float)cb[(pk >> 16) & 0xFu]);
    out[5] = (_Float16)(sc * (float)cb[(pk >> 20) & 0xFu]);
    out[6] = (_Float16)(sc * (float)cb[(pk >> 24) & 0xFu]);
    out[7] = (_Float16)(sc * (float)cb[(pk >> 28) & 0xFu]);
}
