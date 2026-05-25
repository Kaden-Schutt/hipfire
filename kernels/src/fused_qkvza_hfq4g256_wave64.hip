// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Wave64-native variant of fused_qkvza_hfq4g256 for CDNA3 (MI300X / gfx94x).
//
// The base kernel uses block=[32,1,1] = half a wave64, wasting 32 lane slots
// per workgroup on wave64 hardware. This variant uses block=[64,1,1] = one
// full wave64 per block, handling 2 rows (2 gids) per block — one per warp
// of 32 lanes. Each warp runs the same 4-accumulator inner loop over its row
// as the base kernel, so output is byte-exact vs running `fused_qkvza_hfq4g256`
// (same accumulation order, same pyramid reduction, same tail handling).
//
// Win comes from halving the workgroup count: on CDNA3 each [32,1,1] block
// was still consuming a whole 64-lane wave slot with upper lanes idle, so
// halving the block count halves the wave pressure on the scheduler while
// producing identical math.
//
// Dispatch grid: ((total_m + 1) / 2) × [64,1,1]. If total_m is odd the
// trailing warp early-returns on bounds check.

extern "C" __global__ void fused_qkvza_hfq4g256_wave64(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ x,
    float*       __restrict__ y_qkv,
    float*       __restrict__ y_z,
    float*       __restrict__ y_beta,
    float*       __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;       // 0 or 1
    const int lane = tid & 31;          // 0..31 within warp

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    // Same routing as the base kernel — each warp selects its own matrix.
    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv;   y = y_qkv;   row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     y = y_z;     row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  y = y_beta;  row = gid - (qkv_m + z_m);
    } else {
        A = A_alpha; y = y_alpha; row = gid - (qkv_m + z_m + beta_m);
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = lane * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + lane * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base,       acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + lane * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + lane * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + lane * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    // 32-lane intra-warp pyramid. On wave64, shuffles span the whole wave but
    // each warp's lane 0 reads its own warp's partials since offsets stay <32.
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] = acc;
}
