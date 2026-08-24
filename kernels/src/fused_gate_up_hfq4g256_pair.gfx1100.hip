// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_FUSED_GATE_UP_PAIR_KERNEL
#define HIPFIRE_FUSED_GATE_UP_PAIR_KERNEL fused_gate_up_hfq4g256_pair_gfx1100
#endif

#ifndef HIPFIRE_GATE_UP_PAIR2
#define HIPFIRE_GATE_UP_PAIR2 0
#endif

// gfx1100 dense gate+up pairing experiment. One wave computes matching rows
// from both matrices, so each eight-value activation slice is loaded once and
// consumed by both projections. Weight traffic and each output's four-chain
// accumulation order remain identical to fused_gate_up_hfq4g256.hip.
//
// The host admits this kernel only when gate_m == up_m. Grid=[gate_m], block=32.
// Default OFF: Qwen3.6-27B MQ4 on a W7900 (ROCm 7.14, HIP/q8, graph off,
// 8-token warmup + 2 s DPM settle + 64 timed tokens) measured 35.6/35.4
// tok/s for the base kernel and 35.3/35.3 for this arm in A/B/B/A order
// (-0.56% median throughput; +0.43% median p50 latency). The emitted object
// is spill-free (44 VGPR / 23 SGPR), and a 65-token greedy trace was exact.
// Keep this as a negative oracle: activation reuse is already cache-absorbed.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_FUSED_GATE_UP_PAIR_KERNEL(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int row = blockIdx.x;
    if (row >= gate_m || row >= up_m) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* gate_ptr = A_gate + (long long)row * row_stride;
    const char* up_ptr = A_up + (long long)row * row_stride;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto gate_rsrc = HIPFIRE_WEIGHT_RSRC(gate_ptr);
    const auto up_rsrc = HIPFIRE_WEIGHT_RSRC(up_ptr);
#endif

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG_X8(sc, zp, pk, a, xa, xb) \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (xa).x \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (xa).y \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (xa).z \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (xa).w \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (xb).x \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (xb).y \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (xb).z \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (xb).w

    // Limit the live range to one HFQ group. This keeps both four-chain
    // accumulators resident without retaining a full four-group x tile.
    #define PROCESS_GROUP(group_index, chain) do { \
        const int group = (group_index); \
        const int base = group * 256 + tid * 8; \
        const float4 xa = *reinterpret_cast<const float4*>(x + base); \
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4); \
        const char* gg = gate_ptr + group * 136; \
        float gsc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, gg, group * 136)); \
        float gzp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, gg + 4, group * 136 + 4)); \
        unsigned int gpk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            gate_rsrc, gg + 8 + boff, group * 136 + 8 + boff); \
        DOG_X8(gsc, gzp, gpk, gacc##chain, xa, xb); \
        const char* uu = up_ptr + group * 136; \
        float usc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, uu, group * 136)); \
        float uzp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, uu + 4, group * 136 + 4)); \
        unsigned int upk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            up_rsrc, uu + 8 + boff, group * 136 + 8 + boff); \
        DOG_X8(usc, uzp, upk, uacc##chain, xa, xb); \
        asm volatile("" \
            : "+v"(gacc0), "+v"(gacc1), "+v"(gacc2), "+v"(gacc3), \
              "+v"(uacc0), "+v"(uacc1), "+v"(uacc2), "+v"(uacc3) \
            : : "memory"); \
    } while (0)

#if HIPFIRE_GATE_UP_PAIR2
    // Retain two groups of x and both weight streams at once. Pair1's single
    // group live range cut VMEM concurrency in half; this restores a middle
    // schedule without approaching the 96-VGPR gfx1100 occupancy ceiling.
    #define PROCESS_GROUP_PAIR(group0, chain0, group1, chain1) do { \
        const int ga = (group0); \
        const int gb = (group1); \
        const int ba = ga * 256 + tid * 8; \
        const int bb = gb * 256 + tid * 8; \
        const float4 xa0 = *reinterpret_cast<const float4*>(x + ba); \
        const float4 xb0 = *reinterpret_cast<const float4*>(x + ba + 4); \
        const float4 xa1 = *reinterpret_cast<const float4*>(x + bb); \
        const float4 xb1 = *reinterpret_cast<const float4*>(x + bb + 4); \
        const char* gg0 = gate_ptr + ga * 136; \
        const char* gg1 = gate_ptr + gb * 136; \
        const float gsc0 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, gg0, ga * 136)); \
        const float gzp0 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, gg0 + 4, ga * 136 + 4)); \
        const unsigned int gpk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            gate_rsrc, gg0 + 8 + boff, ga * 136 + 8 + boff); \
        const float gsc1 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, gg1, gb * 136)); \
        const float gzp1 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, gg1 + 4, gb * 136 + 4)); \
        const unsigned int gpk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            gate_rsrc, gg1 + 8 + boff, gb * 136 + 8 + boff); \
        const char* uu0 = up_ptr + ga * 136; \
        const char* uu1 = up_ptr + gb * 136; \
        const float usc0 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, uu0, ga * 136)); \
        const float uzp0 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, uu0 + 4, ga * 136 + 4)); \
        const unsigned int upk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            up_rsrc, uu0 + 8 + boff, ga * 136 + 8 + boff); \
        const float usc1 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, uu1, gb * 136)); \
        const float uzp1 = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, uu1 + 4, gb * 136 + 4)); \
        const unsigned int upk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            up_rsrc, uu1 + 8 + boff, gb * 136 + 8 + boff); \
        DOG_X8(gsc0, gzp0, gpk0, gacc##chain0, xa0, xb0); \
        DOG_X8(usc0, uzp0, upk0, uacc##chain0, xa0, xb0); \
        DOG_X8(gsc1, gzp1, gpk1, gacc##chain1, xa1, xb1); \
        DOG_X8(usc1, uzp1, upk1, uacc##chain1, xa1, xb1); \
        asm volatile("" \
            : "+v"(gacc0), "+v"(gacc1), "+v"(gacc2), "+v"(gacc3), \
              "+v"(uacc0), "+v"(uacc1), "+v"(uacc2), "+v"(uacc3) \
            : : "memory"); \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_GROUP_PAIR(g, 0, g + 1, 1);
        PROCESS_GROUP_PAIR(g + 2, 2, g + 3, 3);
    }
    #undef PROCESS_GROUP_PAIR
#else
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_GROUP(g, 0);
        PROCESS_GROUP(g + 1, 1);
        PROCESS_GROUP(g + 2, 2);
        PROCESS_GROUP(g + 3, 3);
    }
#endif
    if (tail >= 1) {
        PROCESS_GROUP(quads << 2, 0);
    }
    if (tail >= 2) {
        PROCESS_GROUP((quads << 2) + 1, 1);
    }
    if (tail >= 3) {
        PROCESS_GROUP((quads << 2) + 2, 2);
    }
    #undef PROCESS_GROUP
    #undef DOG_X8

    float gate = (gacc0 + gacc1) + (gacc2 + gacc3);
    float up = (uacc0 + uacc1) + (uacc2 + uacc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down(gate, offset);
        up += __shfl_down(up, offset);
    }

    if (tid == 0) {
        y_gate[row] = gate;
        y_up[row] = up;
    }
}
