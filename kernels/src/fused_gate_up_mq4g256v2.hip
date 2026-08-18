// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_FUSED_GATE_UP_KERNEL
#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_mq4g256v2
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

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned int hA_sc0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136);
        const unsigned int hB_sc0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4);
        const unsigned int hs_sc0 = (tid < 16) ? hA_sc0 : hB_sc0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 >> 16)));
        const unsigned int hA_sc1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136);
        const unsigned int hB_sc1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4);
        const unsigned int hs_sc1 = (tid < 16) ? hA_sc1 : hB_sc1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 >> 16)));
        const unsigned int hA_sc2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136);
        const unsigned int hB_sc2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4);
        const unsigned int hs_sc2 = (tid < 16) ? hA_sc2 : hB_sc2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 >> 16)));
        const unsigned int hA_sc3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136);
        const unsigned int hB_sc3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4);
        const unsigned int hs_sc3 = (tid < 16) ? hA_sc3 : hB_sc3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 >> 16)));
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

    // Tail groups must accumulate into acc[g % 4] — see
    // gemv_hfq4g256.gfx1100.hip for the full bug explanation.
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
        const unsigned int hA_sc = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp, g * 136);
        const unsigned int hB_sc = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 4, g * 136 + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 8 + boff, g * 136 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        const unsigned int hA_sc = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp, g * 136);
        const unsigned int hB_sc = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 4, g * 136 + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 8 + boff, g * 136 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int hA_sc = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp, g * 136);
        const unsigned int hB_sc = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp + 4, g * 136 + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
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
