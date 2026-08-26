// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_HFQ4G256_KERNEL
#define HIPFIRE_HFQ4G256_KERNEL gemv_hfq4g256
#endif

#ifndef HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#endif

// HFQ4-G256 GEMV: default (arch-agnostic) kernel.
//
// Ported from gemv_hfq4g256.gfx1100.hip in order to produce BYTE-EXACT
// results with the RDNA3 variant. Before this port, the default kernel
// used a single-accumulator sequential summation:
//     acc = (((0 + g0) + g1) + g2) + ...
// while the gfx1100 variant uses a 4-accumulator interleaved pattern:
//     acc0,1,2,3 = running sums over groups {g mod 4 == 0,1,2,3}
//     acc = (acc0 + acc1) + (acc2 + acc3)
//
// FP addition is not associative, so the two orderings produce different
// LSB rounding. The MQ4 quality gate generates its baseline on gfx1100,
// which meant any arch falling through to this default kernel (gfx1010,
// gfx1013 / BC-250 Cyan Skillfish, etc.) silently drifted from the
// baseline starting as early as token 10 on Qwen3.5-0.8B — classic FP
// accumulation divergence, not memory or APU semantics.
//
// The 5302926 "fix(gfx1100): tail group accumulator bug" commit
// established the invariant that tail groups must accumulate into
// acc[g % 4] rather than acc0. That fix was only applied to the gfx1100
// variant; this commit propagates the full 4-accumulator structure and
// the tail-distribution invariant to the default path so every arch
// hitting it produces byte-exact output against the committed baselines.
//
// Launch bounds set to (32, 16) to match the gfx1100 variant's register
// budget. On gfx1010 this is 1024 / 16 = 64 VGPRs/wave (down from 51 at
// 20 waves/SIMD), which is a minor occupancy reduction but necessary to
// keep the compiler from spilling the extra acc1..acc3 state and
// potentially picking a different FMA/MADD sequence.

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
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const unsigned int row_weight_offset =
        (unsigned int)((long long)row * groups_per_row * 136);
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    #if defined(HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER)
        #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #else
        #define LOAD_WEIGHT_HEADER(ptr, offset) \
            HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, (offset))
    #endif
    #define LOAD_WEIGHT_PACKED(ptr, offset) \
        HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, (offset))
#else
    #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #define LOAD_WEIGHT_PACKED(ptr, offset) (*(const unsigned int*)(ptr))
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA0 = LOAD_WEIGHT_HEADER(gp0, goff);
        const unsigned int hB0 = LOAD_WEIGHT_HEADER(gp0 + 4, goff + 4u);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = LOAD_WEIGHT_HEADER(gp1, goff + 136u);
        const unsigned int hB1 = LOAD_WEIGHT_HEADER(gp1 + 4, goff + 140u);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = LOAD_WEIGHT_HEADER(gp2, goff + 272u);
        const unsigned int hB2 = LOAD_WEIGHT_HEADER(gp2 + 4, goff + 276u);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = LOAD_WEIGHT_HEADER(gp3, goff + 408u);
        const unsigned int hB3 = LOAD_WEIGHT_HEADER(gp3 + 4, goff + 412u);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        unsigned int pk0 = LOAD_WEIGHT_PACKED(gp0 + 8 + boff, goff + 8u + boff);
        unsigned int pk1 = LOAD_WEIGHT_PACKED(gp1 + 8 + boff, goff + 144u + boff);
        unsigned int pk2 = LOAD_WEIGHT_PACKED(gp2 + 8 + boff, goff + 280u + boff);
        unsigned int pk3 = LOAD_WEIGHT_PACKED(gp3 + 8 + boff, goff + 416u + boff);

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

    // Tail handling: groups_per_row may not be a multiple of 4. Each tail
    // group must accumulate into its OWN accumulator (acc[g % 4]) to
    // preserve the 4-way interleaving that the combine step assumes.
    // See gemv_hfq4g256.gfx1100.hip for the full bug writeup.
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
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = LOAD_WEIGHT_PACKED(gp + 8 + boff, goff + 8u + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = LOAD_WEIGHT_PACKED(gp + 8 + boff, goff + 8u + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 136u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = LOAD_WEIGHT_PACKED(gp + 8 + boff, goff + 8u + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG
    #undef LOAD_WEIGHT_PACKED
    #undef LOAD_WEIGHT_HEADER

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// ─────────────────────────────────────────────────────────────────────────
// x-batched variant: one weight row loaded ONCE, dotted against B separate
// activation vectors.
//
// Why: the multi-slot decode path used to run one GEMV launch per slot, so
// the 270 MB lm_head weight matrix (M=248320, K=2048, HFQ4-G256) was read B
// times per decode step. Every slot wants the same weights against a
// different x, so the reads collapse to one pass. This is the last
// per-slot-serialised op in the multi-slot forward; everything else is
// already a single launch across all rows.
//
// Bit-exactness is the binding constraint, not speed: the multi-slot path is
// gated by a golden test demanding 0.000x divergence from the reference's
// per-token GEMV. So each acc[b] here accumulates the SAME terms in the SAME
// order as the scalar kernel above -- the only change is which x vector is
// indexed. The weight decode (sc/zp/pk) is hoisted out of the b loop, which
// is a pure CSE of identical values, not a reassociation.
//
// x is [B x K] row-major, y is [B x M] row-major, so y[b*M + row] lands
// exactly where a caller holding a [n_slots x vocab] logits block wants it.
//
// Compiled only into its own module (guarded below), so every existing
// variant of this file is unaffected -- the kernel resource gate fingerprints
// those, and this must not perturb their VGPR/occupancy numbers.
#if defined(HIPFIRE_HFQ4G256_XBATCH)

#ifndef HIPFIRE_HFQ4G256_XBATCH_MAX
#define HIPFIRE_HFQ4G256_XBATCH_MAX 4
#endif

// Occupancy 8, not the scalar kernel's 16: B accumulator sets multiply the
// live float state by B. Asking for 16 waves/SIMD at B=4 makes the compiler
// spill, and a spilled accumulator is exactly what the scalar kernel's
// comment warns changes the FMA sequence.
__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_HFQ4G256_XBATCH_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_HFQ4G256_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0[HIPFIRE_HFQ4G256_XBATCH_MAX];
    float acc1[HIPFIRE_HFQ4G256_XBATCH_MAX];
    float acc2[HIPFIRE_HFQ4G256_XBATCH_MAX];
    float acc3[HIPFIRE_HFQ4G256_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_HFQ4G256_XBATCH_MAX; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        // Plain pointer loads: the buffer-load / cache-policy paths the
        // scalar kernel selects between are tuned for one x vector per pass,
        // and this kernel's whole point is that the weight pass is amortised
        // over B. Kept simple and identical across arches on purpose.
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

        const int base = g * 256 + tid * 8;

        #define XDOG(pk, sc, zp, bs, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * xb[(bs)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xb[(bs) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xb[(bs) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xb[(bs) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xb[(bs) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xb[(bs) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xb[(bs) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xb[(bs) + 7]

        #pragma unroll
        for (int b = 0; b < HIPFIRE_HFQ4G256_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            XDOG(pk0, sc0, zp0, base,       acc0[b]);
            XDOG(pk1, sc1, zp1, base + 256, acc1[b]);
            XDOG(pk2, sc2, zp2, base + 512, acc2[b]);
            XDOG(pk3, sc3, zp3, base + 768, acc3[b]);
        }
        #undef XDOG
    }

    // Tail groups accumulate into acc[g % 4], matching the scalar kernel's
    // 4-way interleaving that the combine below assumes.
    #define XTAIL(pk, sc, zp, bs, a) \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * xb[(bs)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xb[(bs) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xb[(bs) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xb[(bs) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xb[(bs) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xb[(bs) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xb[(bs) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xb[(bs) + 7]

    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_HFQ4G256_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            if (t == 0)      { XTAIL(pk, sc, zp, base, acc0[b]); }
            else if (t == 1) { XTAIL(pk, sc, zp, base, acc1[b]); }
            else             { XTAIL(pk, sc, zp, base, acc2[b]); }
        }
    }
    #undef XTAIL

    #pragma unroll
    for (int b = 0; b < HIPFIRE_HFQ4G256_XBATCH_MAX; b++) {
        if (b >= B) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) y[(long long)b * M + row] = acc;
    }
}

#endif  // HIPFIRE_HFQ4G256_XBATCH
