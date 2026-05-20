// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Software-pipelined variant of gemv_hfq6g256_residual_wave64. Identical
// math; the only structural change is across-quad weight prefetch — quad
// q+1's packed bytes + scales/zeros are issued *before* quad q's compute
// chain runs, so the L2 fills overlap with the FMA chain instead of
// stalling the load unit each iteration.
//
// Mirror of gemv_hfq4g256_residual_wave64_prefetch.hip's structure with
// HFQ6's 6-bit unpack and wider per-thread footprint:
//   HFQ4: 4 bytes/thread/group (1 dword)
//   HFQ6: 6 bytes/thread/group (1.5 dwords; loaded as 2 separate u32s
//                               with a partial-byte tail)
//
// Layout / topology unchanged from the non-prefetched wave64 sibling:
// block=[64,1,1], 2 rows/WG (warp_id=0,1), 4-quad interleave with 4
// accumulators. Bit-exact with the non-prefetched base modulo IEEE FMA
// reordering tolerance.
//
// VGPR cost: HFQ4 sibling is 29 VGPR base, ~41-44 VGPR with prefetch.
// HFQ6's wider unpack + slightly larger packed footprint should land
// ~46-50 VGPR — still 5 waves/SIMD on gfx906 (granule cap 64-VGPR for
// 5 waves). If occupancy regresses below 5, dial back to a 2-quad
// pipeline depth (split into pairs).

extern "C" __global__ void gemv_hfq6g256_residual_wave64_prefetch(
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
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int byte_off = lane * 6;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // 6-bit unpack from 6 bytes into 8 weights, dequant + FMA against x[].
    // Same arithmetic as gemv_hfq6g256_residual_wave64.hip's UNPACK_AND_DOG.
    #define UNPACK_AND_DOG(b0, b1, b2, b3, b4, b5, sc, zp, base, a) do {           \
        int _q0 = (b0) & 63;                                                        \
        int _q1 = ((b0) >> 6) | (((b1) & 0xF) << 2);                                \
        int _q2 = ((b1) >> 4) | (((b2) & 3)  << 4);                                 \
        int _q3 = (b2) >> 2;                                                        \
        int _q4 = (b3) & 63;                                                        \
        int _q5 = ((b3) >> 6) | (((b4) & 0xF) << 2);                                \
        int _q6 = ((b4) >> 4) | (((b5) & 3)  << 4);                                 \
        int _q7 = (b5) >> 2;                                                        \
        (a) += (sc * (float)_q0 + zp) * x[(base)]                                   \
             + (sc * (float)_q1 + zp) * x[(base) + 1]                               \
             + (sc * (float)_q2 + zp) * x[(base) + 2]                               \
             + (sc * (float)_q3 + zp) * x[(base) + 3]                               \
             + (sc * (float)_q4 + zp) * x[(base) + 4]                               \
             + (sc * (float)_q5 + zp) * x[(base) + 5]                               \
             + (sc * (float)_q6 + zp) * x[(base) + 6]                               \
             + (sc * (float)_q7 + zp) * x[(base) + 7];                              \
    } while (0)

    // Load all 4 groups of quad q_idx into the staged registers.
    // Each group: 4 bytes scale + 4 bytes zero + 6 bytes packed at byte_off.
    //
    // The 6 packed bytes per group could be loaded as a single u32 + u16
    // when (byte_off & 3) == 2, or as 6 individual byte loads. We use a
    // u32 + u16 load approach: u32 covers bytes [0..3], u16 covers
    // [4..5]. byte_off = lane * 6 ∈ {0, 6, 12, 18, 24, 30, 36, 42, 48, 54,
    // 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144,
    // 150, 156, 162, 168, 174, 180, 186}. (byte_off & 3) cycles {0,2,0,2,...},
    // so half the lanes naturally u32-align and the other half are
    // 2-byte aligned. Both are dword-fast on gfx906.
    //
    // We just use 6 byte-loads per group via uchar* — the compiler can
    // coalesce these into the natural alignment. This matches the
    // non-prefetched sibling's pattern; benchmarking will tell us if
    // moving to u32+u16 explicit loads buys anything.
    #define LOAD_QUAD(q_idx,                                                       \
                      sc0v, zp0v, sc1v, zp1v, sc2v, zp2v, sc3v, zp3v,              \
                      b00, b01, b02, b03, b04, b05,                                \
                      b10, b11, b12, b13, b14, b15,                                \
                      b20, b21, b22, b23, b24, b25,                                \
                      b30, b31, b32, b33, b34, b35)                                \
        do {                                                                       \
            const int g_ = (q_idx) << 2;                                           \
            const char* gp0_ = row_ptr + g_ * 200;                                 \
            const char* gp1_ = gp0_ + 200;                                         \
            const char* gp2_ = gp1_ + 200;                                         \
            const char* gp3_ = gp2_ + 200;                                         \
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
            (b04) = d0_[byte_off + 4]; (b05) = d0_[byte_off + 5];                  \
            (b10) = d1_[byte_off];     (b11) = d1_[byte_off + 1];                  \
            (b12) = d1_[byte_off + 2]; (b13) = d1_[byte_off + 3];                  \
            (b14) = d1_[byte_off + 4]; (b15) = d1_[byte_off + 5];                  \
            (b20) = d2_[byte_off];     (b21) = d2_[byte_off + 1];                  \
            (b22) = d2_[byte_off + 2]; (b23) = d2_[byte_off + 3];                  \
            (b24) = d2_[byte_off + 4]; (b25) = d2_[byte_off + 5];                  \
            (b30) = d3_[byte_off];     (b31) = d3_[byte_off + 1];                  \
            (b32) = d3_[byte_off + 2]; (b33) = d3_[byte_off + 3];                  \
            (b34) = d3_[byte_off + 4]; (b35) = d3_[byte_off + 5];                  \
        } while (0)

    if (quads > 0) {
        // Prologue: load quad 0 into the staged registers.
        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        unsigned char b00, b01, b02, b03, b04, b05;
        unsigned char b10, b11, b12, b13, b14, b15;
        unsigned char b20, b21, b22, b23, b24, b25;
        unsigned char b30, b31, b32, b33, b34, b35;
        LOAD_QUAD(0, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
                  b00, b01, b02, b03, b04, b05,
                  b10, b11, b12, b13, b14, b15,
                  b20, b21, b22, b23, b24, b25,
                  b30, b31, b32, b33, b34, b35);

        // Steady state: compute(q) then prefetch(q+1) into next-iter regs.
        for (int q = 0; q < quads - 1; q++) {
            float n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3;
            unsigned char n_b00, n_b01, n_b02, n_b03, n_b04, n_b05;
            unsigned char n_b10, n_b11, n_b12, n_b13, n_b14, n_b15;
            unsigned char n_b20, n_b21, n_b22, n_b23, n_b24, n_b25;
            unsigned char n_b30, n_b31, n_b32, n_b33, n_b34, n_b35;
            LOAD_QUAD(q + 1,
                      n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3,
                      n_b00, n_b01, n_b02, n_b03, n_b04, n_b05,
                      n_b10, n_b11, n_b12, n_b13, n_b14, n_b15,
                      n_b20, n_b21, n_b22, n_b23, n_b24, n_b25,
                      n_b30, n_b31, n_b32, n_b33, n_b34, n_b35);

            int base = (q << 2) * 256 + lane * 8;
            UNPACK_AND_DOG(b00, b01, b02, b03, b04, b05, sc0, zp0, base,         acc0);
            UNPACK_AND_DOG(b10, b11, b12, b13, b14, b15, sc1, zp1, base + 256,   acc1);
            UNPACK_AND_DOG(b20, b21, b22, b23, b24, b25, sc2, zp2, base + 512,   acc2);
            UNPACK_AND_DOG(b30, b31, b32, b33, b34, b35, sc3, zp3, base + 768,   acc3);

            sc0 = n_sc0; zp0 = n_zp0; sc1 = n_sc1; zp1 = n_zp1;
            sc2 = n_sc2; zp2 = n_zp2; sc3 = n_sc3; zp3 = n_zp3;
            b00 = n_b00; b01 = n_b01; b02 = n_b02; b03 = n_b03; b04 = n_b04; b05 = n_b05;
            b10 = n_b10; b11 = n_b11; b12 = n_b12; b13 = n_b13; b14 = n_b14; b15 = n_b15;
            b20 = n_b20; b21 = n_b21; b22 = n_b22; b23 = n_b23; b24 = n_b24; b25 = n_b25;
            b30 = n_b30; b31 = n_b31; b32 = n_b32; b33 = n_b33; b34 = n_b34; b35 = n_b35;
        }

        // Epilogue: compute the last quad whose data was prefetched.
        {
            int base = ((quads - 1) << 2) * 256 + lane * 8;
            UNPACK_AND_DOG(b00, b01, b02, b03, b04, b05, sc0, zp0, base,         acc0);
            UNPACK_AND_DOG(b10, b11, b12, b13, b14, b15, sc1, zp1, base + 256,   acc1);
            UNPACK_AND_DOG(b20, b21, b22, b23, b24, b25, sc2, zp2, base + 512,   acc2);
            UNPACK_AND_DOG(b30, b31, b32, b33, b34, b35, sc3, zp3, base + 768,   acc3);
        }
    }
    #undef LOAD_QUAD

    // Tail (1-3 trailing groups, no prefetch — too few iterations to amortize).
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 200;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        unsigned char b0 = d[byte_off],     b1 = d[byte_off + 1];
        unsigned char b2 = d[byte_off + 2], b3 = d[byte_off + 3];
        unsigned char b4 = d[byte_off + 4], b5 = d[byte_off + 5];
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(b0, b1, b2, b3, b4, b5, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 200;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        unsigned char b0 = d[byte_off],     b1 = d[byte_off + 1];
        unsigned char b2 = d[byte_off + 2], b3 = d[byte_off + 3];
        unsigned char b4 = d[byte_off + 4], b5 = d[byte_off + 5];
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(b0, b1, b2, b3, b4, b5, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 200;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        unsigned char b0 = d[byte_off],     b1 = d[byte_off + 1];
        unsigned char b2 = d[byte_off + 2], b3 = d[byte_off + 3];
        unsigned char b4 = d[byte_off + 4], b5 = d[byte_off + 5];
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(b0, b1, b2, b3, b4, b5, sc, zp, base, acc2);
    }
    #undef UNPACK_AND_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] += acc;
}
