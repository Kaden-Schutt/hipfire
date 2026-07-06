// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Software-pipelined variant of gemv_hfq5g256_residual_wave64. Identical
// math; the only structural change is across-quad weight prefetch — quad
// q+1's packed bytes + scales/zeros are issued *before* quad q's compute
// chain runs, so the L2 fills overlap with the FMA chain instead of
// stalling the load unit each iteration.
//
// Mirror of gemv_hfq4g256_residual_wave64_prefetch.hip's structure with
// HFQ5's 5-bit unpack and wider per-thread footprint:
//   HFQ4: 4 bytes/thread/group (1 dword)
//   HFQ5: 5 bytes/thread/group (1.25 dwords; loaded as 1 u32 + 1 byte,
//                               or 5 individual byte loads)
//
// Layout / topology unchanged from the non-prefetched wave64 sibling:
// block=[64,1,1], 2 rows/WG (warp_id=0,1), 4-quad interleave with 4
// accumulators. Bit-exact with the non-prefetched base modulo IEEE FMA
// reordering tolerance.
//
// Group layout: 8 B header (f32 scale + f32 min) + 160 B payload
// (32 chunks × 5 bytes, 8 codes per chunk) = 168 B/group. Per-thread
// footprint is 8 weights = 5 bytes (8 weights per 5 bytes packing).

extern "C" __global__ void gemv_hfq5g256_residual_wave64_prefetch(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 168;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int byte_off = lane * 5;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // 5-bit unpack from 5 bytes into 8 weights, dequant + FMA against x[].
    // Same arithmetic as gemv_hfq5g256_residual_wave64.hip's UNPACK_AND_DOG:
    //   q0 =  b0 & 0x1f
    //   q1 = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3)
    //   q2 =  (b1 >> 2) & 0x1f
    //   q3 = ((b1 >> 7) & 0x01) | ((b2 & 0x0f) << 1)
    //   q4 = ((b2 >> 4) & 0x0f) | ((b3 & 0x01) << 4)
    //   q5 =  (b3 >> 1) & 0x1f
    //   q6 = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2)
    //   q7 =  (b4 >> 3) & 0x1f
    #define UNPACK_AND_DOG(b0, b1, b2, b3, b4, sc, zp, base, a) do {              \
        int _q0 =  (b0) & 0x1f;                                                   \
        int _q1 = (((b0) >> 5) & 0x07) | (((b1) & 0x03) << 3);                    \
        int _q2 =  ((b1) >> 2) & 0x1f;                                           \
        int _q3 = (((b1) >> 7) & 0x01) | (((b2) & 0x0f) << 1);                    \
        int _q4 = (((b2) >> 4) & 0x0f) | (((b3) & 0x01) << 4);                    \
        int _q5 =  ((b3) >> 1) & 0x1f;                                           \
        int _q6 = (((b3) >> 6) & 0x03) | (((b4) & 0x07) << 2);                    \
        int _q7 =  ((b4) >> 3) & 0x1f;                                           \
        (a) += (sc * (float)_q0 + zp) * x[(base)]                                \
             + (sc * (float)_q1 + zp) * x[(base) + 1]                            \
             + (sc * (float)_q2 + zp) * x[(base) + 2]                            \
             + (sc * (float)_q3 + zp) * x[(base) + 3]                            \
             + (sc * (float)_q4 + zp) * x[(base) + 4]                            \
             + (sc * (float)_q5 + zp) * x[(base) + 5]                            \
             + (sc * (float)_q6 + zp) * x[(base) + 6]                            \
             + (sc * (float)_q7 + zp) * x[(base) + 7];                           \
    } while (0)

    // Load all 4 groups of quad q_idx into the staged registers.
    // Each group: 4 bytes scale + 4 bytes zero + 5 bytes packed at byte_off.
    #define LOAD_QUAD(q_idx,                                                       \
                      sc0v, zp0v, sc1v, zp1v, sc2v, zp2v, sc3v, zp3v,              \
                      b00, b01, b02, b03, b04,                                     \
                      b10, b11, b12, b13, b14,                                     \
                      b20, b21, b22, b23, b24,                                     \
                      b30, b31, b32, b33, b34)                                     \
        do {                                                                       \
            const int g_ = (q_idx) << 2;                                           \
            const char* gp0_ = row_ptr + g_ * 168;                                 \
            const char* gp1_ = gp0_ + 168;                                         \
            const char* gp2_ = gp1_ + 168;                                         \
            const char* gp3_ = gp2_ + 168;                                         \
            (sc0v) = __builtin_bit_cast(float, *(const unsigned int*)(gp0_));      \
            (zp0v) = __builtin_bit_cast(float, *(const unsigned int*)(gp0_ + 4));  \
            (sc1v) = __builtin_bit_cast(float, *(const unsigned int*)(gp1_));      \
            (zp1v) = __builtin_bit_cast(float, *(const unsigned int*)(gp1_ + 4));  \
            (sc2v) = __builtin_bit_cast(float, *(const unsigned int*)(gp2_));      \
            (zp2v) = __builtin_bit_cast(float, *(const unsigned int*)(gp2_ + 4));  \
            (sc3v) = __builtin_bit_cast(float, *(const unsigned int*)(gp3_));      \
            (zp3v) = __builtin_bit_cast(float, *(const unsigned int*)(gp3_ + 4));  \
            const unsigned char* d0_ = (const unsigned char*)(gp0_ + 8);            \
            const unsigned char* d1_ = (const unsigned char*)(gp1_ + 8);            \
            const unsigned char* d2_ = (const unsigned char*)(gp2_ + 8);            \
            const unsigned char* d3_ = (const unsigned char*)(gp3_ + 8);            \
            (b00) = d0_[byte_off];     (b01) = d0_[byte_off + 1];                  \
            (b02) = d0_[byte_off + 2]; (b03) = d0_[byte_off + 3];                  \
            (b04) = d0_[byte_off + 4];                                             \
            (b10) = d1_[byte_off];     (b11) = d1_[byte_off + 1];                  \
            (b12) = d1_[byte_off + 2]; (b13) = d1_[byte_off + 3];                  \
            (b14) = d1_[byte_off + 4];                                             \
            (b20) = d2_[byte_off];     (b21) = d2_[byte_off + 1];                  \
            (b22) = d2_[byte_off + 2]; (b23) = d2_[byte_off + 3];                  \
            (b24) = d2_[byte_off + 4];                                             \
            (b30) = d3_[byte_off];     (b31) = d3_[byte_off + 1];                  \
            (b32) = d3_[byte_off + 2]; (b33) = d3_[byte_off + 3];                  \
            (b34) = d3_[byte_off + 4];                                             \
        } while (0)

    if (quads > 0) {
        // Prologue: load quad 0 into the staged registers.
        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        unsigned char b00, b01, b02, b03, b04;
        unsigned char b10, b11, b12, b13, b14;
        unsigned char b20, b21, b22, b23, b24;
        unsigned char b30, b31, b32, b33, b34;
        LOAD_QUAD(0, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
                  b00, b01, b02, b03, b04,
                  b10, b11, b12, b13, b14,
                  b20, b21, b22, b23, b24,
                  b30, b31, b32, b33, b34);

        // Steady state: compute(q) then prefetch(q+1) into next-iter regs.
        for (int q = 0; q < quads - 1; q++) {
            float n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3;
            unsigned char n_b00, n_b01, n_b02, n_b03, n_b04;
            unsigned char n_b10, n_b11, n_b12, n_b13, n_b14;
            unsigned char n_b20, n_b21, n_b22, n_b23, n_b24;
            unsigned char n_b30, n_b31, n_b32, n_b33, n_b34;
            LOAD_QUAD(q + 1,
                      n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3,
                      n_b00, n_b01, n_b02, n_b03, n_b04,
                      n_b10, n_b11, n_b12, n_b13, n_b14,
                      n_b20, n_b21, n_b22, n_b23, n_b24,
                      n_b30, n_b31, n_b32, n_b33, n_b34);

            int base = (q << 2) * 256 + lane * 8;
            UNPACK_AND_DOG(b00, b01, b02, b03, b04, sc0, zp0, base,         acc0);
            UNPACK_AND_DOG(b10, b11, b12, b13, b14, sc1, zp1, base + 256,   acc1);
            UNPACK_AND_DOG(b20, b21, b22, b23, b24, sc2, zp2, base + 512,   acc2);
            UNPACK_AND_DOG(b30, b31, b32, b33, b34, sc3, zp3, base + 768,   acc3);

            sc0 = n_sc0; zp0 = n_zp0; sc1 = n_sc1; zp1 = n_zp1;
            sc2 = n_sc2; zp2 = n_zp2; sc3 = n_sc3; zp3 = n_zp3;
            b00 = n_b00; b01 = n_b01; b02 = n_b02; b03 = n_b03; b04 = n_b04;
            b10 = n_b10; b11 = n_b11; b12 = n_b12; b13 = n_b13; b14 = n_b14;
            b20 = n_b20; b21 = n_b21; b22 = n_b22; b23 = n_b23; b24 = n_b24;
            b30 = n_b30; b31 = n_b31; b32 = n_b32; b33 = n_b33; b34 = n_b34;
        }

        // Epilogue: compute the last quad whose data was prefetched.
        {
            int base = ((quads - 1) << 2) * 256 + lane * 8;
            UNPACK_AND_DOG(b00, b01, b02, b03, b04, sc0, zp0, base,         acc0);
            UNPACK_AND_DOG(b10, b11, b12, b13, b14, sc1, zp1, base + 256,   acc1);
            UNPACK_AND_DOG(b20, b21, b22, b23, b24, sc2, zp2, base + 512,   acc2);
            UNPACK_AND_DOG(b30, b31, b32, b33, b34, sc3, zp3, base + 768,   acc3);
        }
    }
    #undef LOAD_QUAD

    // Tail (1-3 trailing groups, no prefetch — too few iterations to amortize).
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 168;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        unsigned char b0 = d[byte_off],     b1 = d[byte_off + 1];
        unsigned char b2 = d[byte_off + 2], b3 = d[byte_off + 3];
        unsigned char b4 = d[byte_off + 4];
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(b0, b1, b2, b3, b4, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 168;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        unsigned char b0 = d[byte_off],     b1 = d[byte_off + 1];
        unsigned char b2 = d[byte_off + 2], b3 = d[byte_off + 3];
        unsigned char b4 = d[byte_off + 4];
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(b0, b1, b2, b3, b4, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 168;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        unsigned char b0 = d[byte_off],     b1 = d[byte_off + 1];
        unsigned char b2 = d[byte_off + 2], b3 = d[byte_off + 3];
        unsigned char b4 = d[byte_off + 4];
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(b0, b1, b2, b3, b4, sc, zp, base, acc2);
    }
    #undef UNPACK_AND_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] += acc;
}
