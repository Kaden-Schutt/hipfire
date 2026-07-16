// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

using hipfire_hfq4_u32x4 = unsigned int __attribute__((ext_vector_type(4)));

// Fixed-shape gfx1151 MQ4/HFQ4 LM head over an exact AoSoA4 shadow:
//
//   headers: [M][8 groups][scale,zp]             (64 bytes/row)
//   payload: [M][2 quads][32 lanes][4 groups]   (1024 bytes/row)
//
// Each lane obtains pk0..pk3 with one aligned B128 load.  The arithmetic,
// four-accumulator ownership, combine, and wave reduction are identical to
// gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_lm_head_aosoa4_scalar_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    (void)K; // Host selector proves K=2048 before choosing this symbol.

    const long long header_plane_bytes = ((long long)M * 64 + 511) & ~511ll;
    const char* row_headers = A + (long long)row * 64;
    const unsigned int row_payload_offset =
        (unsigned int)(header_plane_bytes + (long long)row * 1024);
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    #pragma clang loop unroll(disable)
    for (int q = 0; q < 2; q++) {
        const unsigned int* h =
            reinterpret_cast<const unsigned int*>(row_headers + q * 32);
        // Keep the incumbent's proven uniform scalar-load form. A single
        // address-space(4) u32x8 load was rejected after ISA inspection showed
        // LLVM incorrectly reusing h[0] for every scale/zero operand.
        const float sc0 = __builtin_bit_cast(float, h[0]);
        const float zp0 = __builtin_bit_cast(float, h[1]);
        const float sc1 = __builtin_bit_cast(float, h[2]);
        const float zp1 = __builtin_bit_cast(float, h[3]);
        const float sc2 = __builtin_bit_cast(float, h[4]);
        const float zp2 = __builtin_bit_cast(float, h[5]);
        const float sc3 = __builtin_bit_cast(float, h[6]);
        const float zp3 = __builtin_bit_cast(float, h[7]);

        const hipfire_hfq4_u32x4 packed =
            __builtin_amdgcn_raw_buffer_load_b128(
                weight_rsrc,
                row_payload_offset + (unsigned int)q * 512u + (unsigned int)tid * 16u,
                0,
                HIPFIRE_WEIGHT_CPOL_AUX);
        const unsigned int pk0 = packed.x;
        const unsigned int pk1 = packed.y;
        const unsigned int pk2 = packed.z;
        const unsigned int pk3 = packed.w;
        const int base = q * 1024 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * x[(b)] \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
