// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 (RDNA3) HFQ4-G256 GEMV: 4x group unroll, packed uint32 loads.
// launch_bounds(32, 16) — gfx1100 max is 16 waves/SIMD (NOT 20 like gfx1010).
// 1536 VGPRs / 16 = 96 VGPRs budget at full occupancy.

#ifndef HIPFIRE_HFQ4G256_KERNEL
#define HIPFIRE_HFQ4G256_KERNEL gemv_hfq4g256
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_HFQ4G256_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_HFQ4G256_K2048)
    // The Qwen3.5-35B LM head is always K=2048. Baking its eight groups into
    // the code removes the dynamic divide, loop trip count, and all tail
    // control while retaining the exact four-accumulator FMA order.
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if defined(HIPFIRE_HFQ4G256_K2048)
    #pragma unroll
#endif
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136));
        float zp0 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4));
        float sc1 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136));
        float zp1 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4));
        float sc2 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136));
        float zp2 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4));
        float sc3 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136));
        float zp3 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4));

        unsigned int pk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        unsigned int pk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
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

#if !defined(HIPFIRE_HFQ4G256_K2048)
    // Tail handling: groups_per_row may not be a multiple of 4. Each tail
    // group must accumulate into its OWN accumulator (acc[g % 4]) to
    // preserve the 4-way interleaving that the combine step assumes.
    // Dumping every tail group into acc0 is a bug — it makes acc0 carry
    // N+tail groups while acc1..acc3 only carry N, producing asymmetric
    // FP accumulation that drifts noticeably over many GEMV calls.
    // Measured: breaks MQ4 4B and 0.8B coherence on long generations.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp, g * 136));
        float zp = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 4, g * 136 + 4));
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 8 + boff, g * 136 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp, g * 136));
        float zp = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 4, g * 136 + 4));
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 8 + boff, g * 136 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp, g * 136));
        float zp = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 4, g * 136 + 4));
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 8 + boff, g * 136 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG
#endif

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
