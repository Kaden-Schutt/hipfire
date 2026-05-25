// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Wave64-native counterpart to gemv_hfq6g256_residual for gfx906 / CDNA1+ /
// CDNA3 (anywhere `has_wave64_native` returns true). The wave32 base kernel
// runs at half throughput on a wave64-native arch because half the wave's
// lanes mask out per pyramid `__shfl_down`.
//
// block=[64,1,1] packs two rows per block (one per warp); grid halves from
// M to (M + 1) / 2. Each warp's 32-lane reduction is byte-exact with the
// wave32 base kernel's pyramid (lane 0 per warp writes output; the
// `__shfl_down` chain stays within the 32-lane warp).
//
// Mirror of `gemv_hfq4g256_residual_wave64.hip`'s structure with HFQ6's
// 6-bit unpack from `gemv_hfq6g256_residual.hip`. Per-thread footprint
// is 8 weights = 6 bytes (4 weights per 3 bytes packing), wider than
// HFQ4's 4 bytes per thread; expect ~10–15 % higher VGPR usage. If
// occupancy regresses, dial back to `__launch_bounds__(64, N)` with
// smaller N, or split the 4-quad accumulator into 2.
//
// Identical math to gemv_hfq6g256_residual: per-group dequant
// `acc += (scale * q + zero) * x[k]` for k ∈ [0..7] per thread per group,
// plus 4-accumulator interleave (q ∈ {0,1,2,3} pyramid) when
// groups_per_row >= 4. Output line uses `+=` to fuse the residual add.

extern "C" __global__ void gemv_hfq6g256_residual_wave64(
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

    // 6-bit unpack: 8 weights = 48 bits = 6 bytes per thread per group.
    //   q0 = b0 & 63
    //   q1 = (b0 >> 6) | ((b1 & 0xF) << 2)
    //   q2 = (b1 >> 4) | ((b2 & 3)  << 4)
    //   q3 = b2 >> 2
    //   q4..q7: same pattern on bytes 3..5
    #define UNPACK_AND_DOG(gp, sc, zp, b, a) do {                                      \
        const unsigned char* _data = (const unsigned char*)(gp + 8);                   \
        unsigned char _b0 = _data[byte_off];                                            \
        unsigned char _b1 = _data[byte_off + 1];                                        \
        unsigned char _b2 = _data[byte_off + 2];                                        \
        unsigned char _b3 = _data[byte_off + 3];                                        \
        unsigned char _b4 = _data[byte_off + 4];                                        \
        unsigned char _b5 = _data[byte_off + 5];                                        \
        int _q0 = _b0 & 63;                                                             \
        int _q1 = (_b0 >> 6) | ((_b1 & 0xF) << 2);                                      \
        int _q2 = (_b1 >> 4) | ((_b2 & 3)  << 4);                                       \
        int _q3 = _b2 >> 2;                                                             \
        int _q4 = _b3 & 63;                                                             \
        int _q5 = (_b3 >> 6) | ((_b4 & 0xF) << 2);                                      \
        int _q6 = (_b4 >> 4) | ((_b5 & 3)  << 4);                                       \
        int _q7 = _b5 >> 2;                                                             \
        (a) += (sc * (float)_q0 + zp) * x[(b)]                                          \
             + (sc * (float)_q1 + zp) * x[(b) + 1]                                      \
             + (sc * (float)_q2 + zp) * x[(b) + 2]                                      \
             + (sc * (float)_q3 + zp) * x[(b) + 3]                                      \
             + (sc * (float)_q4 + zp) * x[(b) + 4]                                      \
             + (sc * (float)_q5 + zp) * x[(b) + 5]                                      \
             + (sc * (float)_q6 + zp) * x[(b) + 6]                                      \
             + (sc * (float)_q7 + zp) * x[(b) + 7];                                     \
    } while(0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const char* gp1 = gp0 + 200;
        const char* gp2 = gp1 + 200;
        const char* gp3 = gp2 + 200;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        int base = g * 256 + lane * 8;

        UNPACK_AND_DOG(gp0, sc0, zp0, base,       acc0);
        UNPACK_AND_DOG(gp1, sc1, zp1, base + 256, acc1);
        UNPACK_AND_DOG(gp2, sc2, zp2, base + 512, acc2);
        UNPACK_AND_DOG(gp3, sc3, zp3, base + 768, acc3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 200;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(gp, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 200;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(gp, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 200;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        int base = g * 256 + lane * 8;
        UNPACK_AND_DOG(gp, sc, zp, base, acc2);
    }
    #undef UNPACK_AND_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] += acc;   // each warp's lane 0 writes its row
}
