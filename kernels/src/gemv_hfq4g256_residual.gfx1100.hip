// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 (RDNA3) HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
// Same 4x group unroll and packed uint32 loads as gemv_hfq4g256.gfx1100.hip —
// only the final write differs (+= instead of =). Used for wo and w_down in
// the Qwen3.5 forward path to eliminate the add_inplace_f32 follow-up launch.

#ifndef HIPFIRE_RESIDUAL_KERNEL
#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_RESIDUAL_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_RDNA3_RESIDUAL_K2048)
    // Exact A3B decode contract; host routing enforces K==2048.
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
#if defined(HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC)
    const unsigned int row_offset =
        (unsigned int)row * (unsigned int)groups_per_row * 136u;
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#else
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif
#endif

#if defined(HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC)
    #define RESIDUAL_WEIGHT_LOAD(ptr, offset) \
        HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(row_rsrc, ptr, offset, row_offset)
#else
    #define RESIDUAL_WEIGHT_LOAD(ptr, offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, ptr, offset)
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
    #define DOG_X8(pk, sc, zp, a, xa, xb) \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * (xa).x \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (xa).y \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (xa).z \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (xa).w \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (xb).x \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (xb).y \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (xb).z \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (xb).w
#endif

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base = g * 256 + tid * 8;
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        // RDNA3 oracle: issue the full activation quad before the independent
        // header/packed-weight stream, then give LLVM a load-free VALU burst.
        // This preserves one output row per wave (unlike the gfx12 dual-row
        // kernel) so gfx1100 keeps enough blocks to occupy all 96 CUs.
        const float4 x0a = *reinterpret_cast<const float4*>(x + base);
        const float4 x0b = *reinterpret_cast<const float4*>(x + base + 4);
        const float4 x1a = *reinterpret_cast<const float4*>(x + base + 256);
        const float4 x1b = *reinterpret_cast<const float4*>(x + base + 260);
        const float4 x2a = *reinterpret_cast<const float4*>(x + base + 512);
        const float4 x2b = *reinterpret_cast<const float4*>(x + base + 516);
        const float4 x3a = *reinterpret_cast<const float4*>(x + base + 768);
        const float4 x3b = *reinterpret_cast<const float4*>(x + base + 772);
#endif
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
#if defined(HIPFIRE_RDNA3_RESIDUAL_BUFFER_CONSUME)
        // Raw-buffer lowering otherwise keeps all four weight triples live
        // beside 32 activation values and crosses gfx1100's VGPR occupancy
        // boundary. Consume one group at a time while retaining the exact
        // independent accumulator and FMA order.
        #define LOAD_DOG(gp, offset, accumulator, xa, xb) do { \
            const float sc = __builtin_bit_cast(float, \
                RESIDUAL_WEIGHT_LOAD(gp, offset)); \
            const float zp = __builtin_bit_cast(float, \
                RESIDUAL_WEIGHT_LOAD((gp) + 4, (offset) + 4)); \
            const unsigned int pk = RESIDUAL_WEIGHT_LOAD( \
                (gp) + 8 + boff, (offset) + 8 + boff); \
            DOG_X8(pk, sc, zp, accumulator, xa, xb); \
        } while (0)
        LOAD_DOG(gp0, (g + 0) * 136, acc0, x0a, x0b);
        LOAD_DOG(gp1, (g + 1) * 136, acc1, x1a, x1b);
        LOAD_DOG(gp2, (g + 2) * 136, acc2, x2a, x2b);
        LOAD_DOG(gp3, (g + 3) * 136, acc3, x3a, x3b);
        #undef LOAD_DOG
#else
        float sc0 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp0, (g + 0) * 136));
        float zp0 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp0 + 4, (g + 0) * 136 + 4));
        float sc1 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp1, (g + 1) * 136));
        float zp1 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp1 + 4, (g + 1) * 136 + 4));
        float sc2 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp2, (g + 2) * 136));
        float zp2 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp2 + 4, (g + 2) * 136 + 4));
        float sc3 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp3, (g + 3) * 136));
        float zp3 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp3 + 4, (g + 3) * 136 + 4));

        unsigned int pk0 = RESIDUAL_WEIGHT_LOAD(gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        unsigned int pk1 = RESIDUAL_WEIGHT_LOAD(gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = RESIDUAL_WEIGHT_LOAD(gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = RESIDUAL_WEIGHT_LOAD(gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);

        DOG_X8(pk0, sc0, zp0, acc0, x0a, x0b);
        DOG_X8(pk1, sc1, zp1, acc1, x1a, x1b);
        DOG_X8(pk2, sc2, zp2, acc2, x2a, x2b);
        DOG_X8(pk3, sc3, zp3, acc3, x3a, x3b);
#endif
#else
        float sc0 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp0, (g + 0) * 136));
        float zp0 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp0 + 4, (g + 0) * 136 + 4));
        float sc1 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp1, (g + 1) * 136));
        float zp1 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp1 + 4, (g + 1) * 136 + 4));
        float sc2 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp2, (g + 2) * 136));
        float zp2 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp2 + 4, (g + 2) * 136 + 4));
        float sc3 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp3, (g + 3) * 136));
        float zp3 = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp3 + 4, (g + 3) * 136 + 4));

        unsigned int pk0 = RESIDUAL_WEIGHT_LOAD(gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        unsigned int pk1 = RESIDUAL_WEIGHT_LOAD(gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = RESIDUAL_WEIGHT_LOAD(gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = RESIDUAL_WEIGHT_LOAD(gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);

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
#endif
    }

    // Tail groups must accumulate into their own accumulator (acc[g % 4]) —
    // see gemv_hfq4g256.gfx1100.hip for the full bug explanation. Breaks
    // MQ4 4B / 0.8B quality when groups_per_row isn't a multiple of 4.
#if !HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]
#endif

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
#endif
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp, g * 136));
        float zp = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp + 4, g * 136 + 4));
        unsigned int pk = RESIDUAL_WEIGHT_LOAD(gp + 8 + boff, g * 136 + 8 + boff);
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        DOG_X8(pk, sc, zp, acc0, xa, xb);
#else
        TAIL_DOG(pk, sc, zp, base, acc0);
#endif
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
#endif
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp, g * 136));
        float zp = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp + 4, g * 136 + 4));
        unsigned int pk = RESIDUAL_WEIGHT_LOAD(gp + 8 + boff, g * 136 + 8 + boff);
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        DOG_X8(pk, sc, zp, acc1, xa, xb);
#else
        TAIL_DOG(pk, sc, zp, base, acc1);
#endif
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
#endif
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp, g * 136));
        float zp = __builtin_bit_cast(float, RESIDUAL_WEIGHT_LOAD(gp + 4, g * 136 + 4));
        unsigned int pk = RESIDUAL_WEIGHT_LOAD(gp + 8 + boff, g * 136 + 8 + boff);
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
        DOG_X8(pk, sc, zp, acc2, xa, xb);
#else
        TAIL_DOG(pk, sc, zp, base, acc2);
#endif
    }
#if HIPFIRE_RDNA3_RESIDUAL_STAGE_X32
    #undef DOG_X8
#else
    #undef TAIL_DOG
#endif
    #undef RESIDUAL_WEIGHT_LOAD

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
