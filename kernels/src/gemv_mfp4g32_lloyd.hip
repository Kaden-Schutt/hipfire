// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// MFP4G32-Lloyd GEMV — mfp4 (E2M1 + UE8M0 g32 + FP16 row scale + offline FWHT)
// with the fixed E2M1 grid REPLACED by a per-TENSOR 16-entry fp16 codebook.
//
// Byte layout:
//   bytes [0..32)                  : 16 × fp16 per-tensor codebook (sorted asc)
//   bytes [32..32 + M*row_bytes)   : M rows, each byte-identical to mfp4:
//       16-B row header (row_scale_a:f16 @0, ..., flags:u8 @6 = 0x05)
//       then (K/32) blocks × 17 B (UE8M0 block_e:u8 + 16 B nibbles).
// Per element:  value = row_scale_a * 2^(block_e - 127) * codebook[nibble]
//
// x is the FWHT-rotated activation (the runtime rotates x via the existing
// mfp4 path; the weights are stored pre-rotated). This kernel is the
// "prerotated" GEMV — it does NO rotation, exactly like gemv_hfp4g32.
//
// Structure ports gemv_hfp4g32.hip 1:1 (same 136-B group stride, same
// 4-accumulator interleave + tail-by-g%4 + (acc0+acc1)+(acc2+acc3) finalize),
// so the FP add ordering matches the mfp4 baseline. The ONLY change is the
// LDS LUT is the per-tensor codebook (read from the 32-B prefix) instead of
// the hard-coded E2M1 lattice, and the row base offset is +32.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mfp4g32_lloyd(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // 16-entry per-tensor codebook in LDS (32 B), loaded once by lanes 0..15
    // from the tensor prefix at A[0..32). Built-in `_Float16` avoids
    // <hip/hip_fp16.h> (ROCm 7.2.2 complex_builtins max/fmax decl bug).
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const unsigned short* cb_bits = (const unsigned short*)(A);
        lut[tid] = __builtin_bit_cast(_Float16, cb_bits[tid]);
    }
    __syncthreads();

    const int groups_per_row = K / 256;            // each group = 8 mfp4 blocks
    const int row_block_bytes = (K / 32) * 17;     // = groups_per_row * 136
    // +32: per-tensor codebook prefix precedes row 0.
    const long long row_off =
        32LL + (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                // first block of this row

    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    const int block_in_iter = tid >> 2;            // 0..7
    const int nib_off = (tid & 3) * 4;             // 0/4/8/12
    const int boff = nib_off;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    const int thread_block_off = block_in_iter * 17;

    #define MFPL_DOG(pk, sc, b, a) \
        (a) += (sc * (float)lut[(pk)        & 0xFu]) * x[(b)    ] \
             + (sc * (float)lut[((pk) >>  4) & 0xFu]) * x[(b) + 1] \
             + (sc * (float)lut[((pk) >>  8) & 0xFu]) * x[(b) + 2] \
             + (sc * (float)lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + (sc * (float)lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + (sc * (float)lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + (sc * (float)lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + (sc * (float)lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned char e0 = *(const unsigned char*)(gp0);
        const unsigned char e1 = *(const unsigned char*)(gp1);
        const unsigned char e2 = *(const unsigned char*)(gp2);
        const unsigned char e3 = *(const unsigned char*)(gp3);

        const float sc0 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e0) << 23);
        const float sc1 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e1) << 23);
        const float sc2 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e2) << 23);
        const float sc3 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e3) << 23);

        const unsigned int pk0 = *(const unsigned int*)(gp0 + 1 + boff);
        const unsigned int pk1 = *(const unsigned int*)(gp1 + 1 + boff);
        const unsigned int pk2 = *(const unsigned int*)(gp2 + 1 + boff);
        const unsigned int pk3 = *(const unsigned int*)(gp3 + 1 + boff);

        const int base = g * 256 + tid * 8;        // (tid>>2)*32 + (tid&3)*8 == tid*8

        MFPL_DOG(pk0, sc0, base,         acc0);
        MFPL_DOG(pk1, sc1, base + 256,   acc1);
        MFPL_DOG(pk2, sc2, base + 512,   acc2);
        MFPL_DOG(pk3, sc3, base + 768,   acc3);
    }
    #undef MFPL_DOG

    #define MFPL_TAIL_DOG(pk, sc, b, a) \
        (a) += (sc * (float)lut[(pk)        & 0xFu]) * x[(b)    ] \
             + (sc * (float)lut[((pk) >>  4) & 0xFu]) * x[(b) + 1] \
             + (sc * (float)lut[((pk) >>  8) & 0xFu]) * x[(b) + 2] \
             + (sc * (float)lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + (sc * (float)lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + (sc * (float)lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + (sc * (float)lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + (sc * (float)lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e) << 23);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8;
        MFPL_TAIL_DOG(pk, sc, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e) << 23);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8;
        MFPL_TAIL_DOG(pk, sc, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e) << 23);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8;
        MFPL_TAIL_DOG(pk, sc, base, acc2);
    }
    #undef MFPL_TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
