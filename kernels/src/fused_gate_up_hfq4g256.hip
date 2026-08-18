// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_FUSED_GATE_UP_KERNEL
#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_hfq4g256
#endif

#ifndef HIPFIRE_HFQ4_DOT_REFORM
#define HIPFIRE_HFQ4_DOT_REFORM 0
#endif

#ifndef HIPFIRE_HFQ4_QUAD_PREFETCH
#define HIPFIRE_HFQ4_QUAD_PREFETCH 0
#endif

#ifndef HIPFIRE_HFQ4_SETPRIO
#define HIPFIRE_HFQ4_SETPRIO 0
#endif

#ifndef HIPFIRE_HFQ4_LANE0_HEADERS
#define HIPFIRE_HFQ4_LANE0_HEADERS 0
#endif

#ifndef HIPFIRE_HFQ4_STAGE_X32
#define HIPFIRE_HFQ4_STAGE_X32 0
#endif

// Fused Gate+Up HFQ4-G256: two GEMVs in one launch (saves 1 launch per
// FFN layer vs separate gate/up calls). Grid = [gate_m + up_m] — each
// block processes one row from either A_gate or A_up based on its index.
//
// Per-row math is identical to gemv_hfq4g256.hip: 4-accumulator
// group-interleaved summation with pairwise combine for cross-arch
// byte-exact output against the MQ4 quality-gate baselines. See
// gemv_hfq4g256.hip and gemv_hfq4g256.gfx1100.hip for the full writeup
// on the FP-associativity drift that motivates this structure.
//
// Launch bounds are (32, 16) to match the RDNA3 variant's register
// budget and keep codegen similar across archs.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_FUSED_GATE_UP_KERNEL(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

#if defined(HIPFIRE_FUSED_GATE_UP_K1024)
    const int groups_per_row = 4;
#elif defined(HIPFIRE_FUSED_GATE_UP_K5120)
    const int groups_per_row = 20;
#elif defined(HIPFIRE_GLIMMER_GATE_UP_K6656)
    // Muse Glimmer AR decode fused gate+up path: exact host-enforced shape is
    // gate_m=up_m=19968, K=6656 (26 groups, quads=6, tail=2). Arithmetic
    // (accumulator ownership, FMA order, tail placement, reduction tree)
    // stays identical to the generic template.
    const int groups_per_row = 26;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)local_row * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if HIPFIRE_HFQ4_QUAD_PREFETCH
    #define LOAD_QUAD(q_idx, sc0v, zp0v, sc1v, zp1v, sc2v, zp2v, sc3v, zp3v, pk0v, pk1v, pk2v, pk3v) do { \
        const int g_ = (q_idx) << 2; \
        const char* gp0_ = row_ptr + g_ * 136; \
        const char* gp1_ = gp0_ + 136; \
        const char* gp2_ = gp1_ + 136; \
        const char* gp3_ = gp2_ + 136; \
        (sc0v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0_, (g_ + 0) * 136)); \
        (zp0v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0_ + 4, (g_ + 0) * 136 + 4)); \
        (sc1v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1_, (g_ + 1) * 136)); \
        (zp1v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1_ + 4, (g_ + 1) * 136 + 4)); \
        (sc2v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2_, (g_ + 2) * 136)); \
        (zp2v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2_ + 4, (g_ + 2) * 136 + 4)); \
        (sc3v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3_, (g_ + 3) * 136)); \
        (zp3v) = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3_ + 4, (g_ + 3) * 136 + 4)); \
        (pk0v) = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0_ + 8 + boff, (g_ + 0) * 136 + 8 + boff); \
        (pk1v) = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1_ + 8 + boff, (g_ + 1) * 136 + 8 + boff); \
        (pk2v) = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2_ + 8 + boff, (g_ + 2) * 136 + 8 + boff); \
        (pk3v) = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3_ + 8 + boff, (g_ + 3) * 136 + 8 + boff); \
    } while (0)
    #if HIPFIRE_HFQ4_DOT_REFORM
    #define PREFETCH_DOG(pk, sc, zp, b, a) do { \
        const float4 xa = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 xb = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float q01 = (float)((pk) & 0xFu) * xa.x \
                        + (float)(((pk) >> 4) & 0xFu) * xa.y; \
        const float q23 = (float)(((pk) >> 8) & 0xFu) * xa.z \
                        + (float)(((pk) >> 12) & 0xFu) * xa.w; \
        const float q45 = (float)(((pk) >> 16) & 0xFu) * xb.x \
                        + (float)(((pk) >> 20) & 0xFu) * xb.y; \
        const float q67 = (float)(((pk) >> 24) & 0xFu) * xb.z \
                        + (float)(((pk) >> 28) & 0xFu) * xb.w; \
        const float qdot = (q01 + q23) + (q45 + q67); \
        const float xsum = ((xa.x + xa.y) + (xa.z + xa.w)) \
                         + ((xb.x + xb.y) + (xb.z + xb.w)); \
        (a) += (sc) * qdot + (zp) * xsum; \
    } while (0)
    #else
    #define PREFETCH_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)] \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]
    #endif

    if (quads > 0) {
        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        unsigned int pk0, pk1, pk2, pk3;
        LOAD_QUAD(0, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
                     pk0, pk1, pk2, pk3);

        for (int q = 0; q < quads - 1; q++) {
            float n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3;
            unsigned int n_pk0, n_pk1, n_pk2, n_pk3;
            LOAD_QUAD(q + 1,
                      n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3,
                      n_pk0, n_pk1, n_pk2, n_pk3);

            const int base = (q << 2) * 256 + tid * 8;
            PREFETCH_DOG(pk0, sc0, zp0, base, acc0);
            PREFETCH_DOG(pk1, sc1, zp1, base + 256, acc1);
            PREFETCH_DOG(pk2, sc2, zp2, base + 512, acc2);
            PREFETCH_DOG(pk3, sc3, zp3, base + 768, acc3);

            sc0 = n_sc0; zp0 = n_zp0; sc1 = n_sc1; zp1 = n_zp1;
            sc2 = n_sc2; zp2 = n_zp2; sc3 = n_sc3; zp3 = n_zp3;
            pk0 = n_pk0; pk1 = n_pk1; pk2 = n_pk2; pk3 = n_pk3;
        }

        const int base = ((quads - 1) << 2) * 256 + tid * 8;
        PREFETCH_DOG(pk0, sc0, zp0, base, acc0);
        PREFETCH_DOG(pk1, sc1, zp1, base + 256, acc1);
        PREFETCH_DOG(pk2, sc2, zp2, base + 512, acc2);
        PREFETCH_DOG(pk3, sc3, zp3, base + 768, acc3);
    }
    #undef PREFETCH_DOG
    #undef LOAD_QUAD
#else
    for (int q = 0; q < quads; q++) {
#if HIPFIRE_HFQ4_SETPRIO
        asm volatile("s_setprio 1" ::: "memory");
#endif
        const int g = q << 2;
        const int base = g * 256 + tid * 8;
#if HIPFIRE_HFQ4_STAGE_X32
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

#if HIPFIRE_HFQ4_LANE0_HEADERS
        unsigned int sc0_bits = 0, zp0_bits = 0, sc1_bits = 0, zp1_bits = 0;
        unsigned int sc2_bits = 0, zp2_bits = 0, sc3_bits = 0, zp3_bits = 0;
        if (tid == 0) {
            sc0_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136);
            zp0_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4);
            sc1_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136);
            zp1_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4);
            sc2_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136);
            zp2_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4);
            sc3_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136);
            zp3_bits = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4);
        }
        const float sc0 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)sc0_bits, 0));
        const float zp0 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)zp0_bits, 0));
        const float sc1 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)sc1_bits, 0));
        const float zp1 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)zp1_bits, 0));
        const float sc2 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)sc2_bits, 0));
        const float zp2 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)zp2_bits, 0));
        const float sc3 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)sc3_bits, 0));
        const float zp3 = __builtin_bit_cast(float,
            (unsigned int)__builtin_amdgcn_readlane((int)zp3_bits, 0));
#else
        float sc0 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136));
        float zp0 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4));
        float sc1 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136));
        float zp1 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4));
        float sc2 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136));
        float zp2 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4));
        float sc3 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136));
        float zp3 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4));
#endif

        unsigned int pk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        unsigned int pk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);
#if HIPFIRE_HFQ4_SETPRIO
        asm volatile("s_setprio 0" ::: "memory");
#endif

        #if HIPFIRE_HFQ4_STAGE_X32
        #define DOG_X8(pk, sc, zp, a, xa, xb) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * (xa).x \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (xa).y \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (xa).z \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (xa).w \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (xb).x \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (xb).y \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (xb).z \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (xb).w
        DOG_X8(pk0, sc0, zp0, acc0, x0a, x0b);
        DOG_X8(pk1, sc1, zp1, acc1, x1a, x1b);
        DOG_X8(pk2, sc2, zp2, acc2, x2a, x2b);
        DOG_X8(pk3, sc3, zp3, acc3, x3a, x3b);
        #undef DOG_X8
        #elif HIPFIRE_HFQ4_DOT_REFORM
        #define DOG(pk, sc, zp, b, a) do { \
            const float4 xa = *reinterpret_cast<const float4*>(x + (b)); \
            const float4 xb = *reinterpret_cast<const float4*>(x + (b) + 4); \
            const float q01 = (float)((pk) & 0xFu) * xa.x \
                            + (float)(((pk) >> 4) & 0xFu) * xa.y; \
            const float q23 = (float)(((pk) >> 8) & 0xFu) * xa.z \
                            + (float)(((pk) >> 12) & 0xFu) * xa.w; \
            const float q45 = (float)(((pk) >> 16) & 0xFu) * xb.x \
                            + (float)(((pk) >> 20) & 0xFu) * xb.y; \
            const float q67 = (float)(((pk) >> 24) & 0xFu) * xb.z \
                            + (float)(((pk) >> 28) & 0xFu) * xb.w; \
            const float qdot = (q01 + q23) + (q45 + q67); \
            const float xsum = ((xa.x + xa.y) + (xa.z + xa.w)) \
                             + ((xb.x + xb.y) + (xb.z + xb.w)); \
            (a) += (sc) * qdot + (zp) * xsum; \
        } while (0)
        #else
        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]
        #endif

        #if !HIPFIRE_HFQ4_STAGE_X32
        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
        #endif
    }
#endif

    // Tail groups must accumulate into acc[g % 4] — see
    // gemv_hfq4g256.gfx1100.hip for the full bug explanation.
    #if HIPFIRE_HFQ4_DOT_REFORM
    #define TAIL_DOG(pk, sc, zp, b, a) do { \
        const float4 xa = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 xb = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float q01 = (float)((pk) & 0xFu) * xa.x \
                        + (float)(((pk) >> 4) & 0xFu) * xa.y; \
        const float q23 = (float)(((pk) >> 8) & 0xFu) * xa.z \
                        + (float)(((pk) >> 12) & 0xFu) * xa.w; \
        const float q45 = (float)(((pk) >> 16) & 0xFu) * xb.x \
                        + (float)(((pk) >> 20) & 0xFu) * xb.y; \
        const float q67 = (float)(((pk) >> 24) & 0xFu) * xb.z \
                        + (float)(((pk) >> 28) & 0xFu) * xb.w; \
        const float qdot = (q01 + q23) + (q45 + q67); \
        const float xsum = ((xa.x + xa.y) + (xa.z + xa.w)) \
                         + ((xb.x + xb.y) + (xb.z + xb.w)); \
        (a) += (sc) * qdot + (zp) * xsum; \
    } while (0)
    #else
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

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[local_row] = acc;
}
