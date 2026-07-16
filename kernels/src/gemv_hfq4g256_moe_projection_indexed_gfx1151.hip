// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

using hipfire_projection_f32x4 = float __attribute__((ext_vector_type(4)));

#ifndef HIPFIRE_MOE_PROJECTION_KERNEL
#error "HIPFIRE_MOE_PROJECTION_KERNEL must name the gate-only or up-only entry point"
#endif

#ifndef HIPFIRE_MOE_PAIRED_WAVES
#define HIPFIRE_MOE_PAIRED_WAVES 0
#endif

#if !HIPFIRE_MOE_PAIRED_WAVES
#ifndef HIPFIRE_MOE_PROJECTION
#error "HIPFIRE_MOE_PROJECTION must be 0 (gate) or 1 (up)"
#endif

#if HIPFIRE_MOE_PROJECTION != 0 && HIPFIRE_MOE_PROJECTION != 1
#error "HIPFIRE_MOE_PROJECTION must be 0 (gate) or 1 (up)"
#endif
#endif

#if HIPFIRE_MOE_PAIRED_WAVES
#define HIPFIRE_MOE_PROJECTION_BLOCK 64
#define HIPFIRE_MOE_PROJECTION_MIN_BLOCKS 8
#define HIPFIRE_MOE_PROJECTION_OUTPUT_ARGS \
    float* __restrict__ y_gate, \
    float* __restrict__ y_up,
#else
#define HIPFIRE_MOE_PROJECTION_BLOCK 32
#define HIPFIRE_MOE_PROJECTION_MIN_BLOCKS 16
#define HIPFIRE_MOE_PROJECTION_OUTPUT_ARGS float* __restrict__ y,
#endif

// gfx1151 structural decode kernel for the fixed MQ4R A3B K=2048 route.
//
// The generic kernel owns gate and up together. That shape serializes the two
// independent expert-weight streams and keeps eight accumulator chains live in
// one wave. This entry point owns exactly one projection. Redline records a
// gate launch and an up launch as independent producers, places them on
// retained PM4 queues, and joins them only at fused_silu_mul_mq_rotate.
//
// Arithmetic within one output is deliberately identical to the generic
// kernel: four accumulator chains consume groups 0/4, 1/5, 2/6, and 3/7;
// those chains are combined as (a0+a1)+(a2+a3), then reduced wave32 in the
// same order. Only the unrelated projection has left the live range.
__launch_bounds__(HIPFIRE_MOE_PROJECTION_BLOCK, HIPFIRE_MOE_PROJECTION_MIN_BLOCKS)
extern "C" __global__ void HIPFIRE_MOE_PROJECTION_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    HIPFIRE_MOE_PROJECTION_OUTPUT_ARGS
    int M,
    int K
) {
    // Host dispatch admits this kernel only for K=2048. Keep K in the ABI so
    // capture/replay validation can retain the same shape contract as the
    // generic kernel without placing it in the address calculation.
    (void)K;
    const int mi = M >> 1;
    const int row = blockIdx.x;
    const int krank = blockIdx.y;
#if HIPFIRE_MOE_PAIRED_WAVES
    const int projection = threadIdx.x >> 5;
    const int tid = threadIdx.x & 31;
    float* __restrict__ out = projection == 0 ? y_gate : y_up;
#else
    constexpr int projection = HIPFIRE_MOE_PROJECTION;
    const int tid = threadIdx.x;
    float* __restrict__ out = y;
#endif
    if (row >= mi) return;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    constexpr unsigned int groups_per_row = 8;
    constexpr unsigned int group_bytes = 136;
    constexpr unsigned int row_bytes = groups_per_row * group_bytes;
    const unsigned int projection_row =
        (unsigned int)row + (unsigned int)projection * (unsigned int)mi;
    const unsigned int row_offset = projection_row * row_bytes;
    const int boff = tid * 4;

    // Radiowave lowering: one expert-wide SRD, temporal raw-buffer weight
    // requests, and scalar row/group offsets. The activation is shared and hot;
    // leave its naturally aligned float4 stream to LLVM.
    const auto expert_rsrc = HIPFIRE_WEIGHT_RSRC(A);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)          + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)

    // Two groups per live range. The generic fused kernel needs a four-group
    // clause to keep both projection streams moving; the split design gets its
    // outer MLP from concurrent producers, so each wave can spend materially
    // fewer VGPRs on transient activation and weight state.
    #define PROCESS_PAIR(g, a0, a1) do { \
        const int base = (g) * 256 + tid * 8; \
        const hipfire_projection_f32x4 x0a = \
            *reinterpret_cast<const hipfire_projection_f32x4*>(x + base); \
        const hipfire_projection_f32x4 x0b = \
            *reinterpret_cast<const hipfire_projection_f32x4*>(x + base + 4); \
        const hipfire_projection_f32x4 x1a = \
            *reinterpret_cast<const hipfire_projection_f32x4*>(x + base + 256); \
        const hipfire_projection_f32x4 x1b = \
            *reinterpret_cast<const hipfire_projection_f32x4*>(x + base + 260); \
        asm volatile("" : : \
            "v"(x0a), "v"(x0b), "v"(x1a), "v"(x1b) : "memory"); \
        const unsigned int group_offset = (unsigned int)(g) * group_bytes; \
        const unsigned int scalar_offset = row_offset + group_offset; \
        const unsigned int sc0_bits = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, 0, scalar_offset); \
        const unsigned int zp0_bits = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, 4, scalar_offset); \
        const unsigned int pk0 = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, (unsigned int)boff + 8, scalar_offset); \
        const unsigned int sc1_bits = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, 136, scalar_offset); \
        const unsigned int zp1_bits = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, 140, scalar_offset); \
        const unsigned int pk1 = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, (unsigned int)boff + 144, scalar_offset); \
        const float sc0 = __builtin_bit_cast(float, sc0_bits); \
        const float zp0 = __builtin_bit_cast(float, zp0_bits); \
        const float sc1 = __builtin_bit_cast(float, sc1_bits); \
        const float zp1 = __builtin_bit_cast(float, zp1_bits); \
        DOG_X8(sc0, zp0, pk0, a0, \
               x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w); \
        DOG_X8(sc1, zp1, pk1, a1, \
               x1a.x, x1a.y, x1a.z, x1a.w, x1b.x, x1b.y, x1b.z, x1b.w); \
        asm volatile("" : "+v"(a0), "+v"(a1) : : "memory"); \
    } while (0)

    PROCESS_PAIR(0, acc0, acc1);
    PROCESS_PAIR(2, acc2, acc3);
    PROCESS_PAIR(4, acc0, acc1);
    PROCESS_PAIR(6, acc2, acc3);

    #undef PROCESS_PAIR
    #undef DOG_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        out[krank * mi + row] = acc;
    }
}

#undef HIPFIRE_MOE_PROJECTION_OUTPUT_ARGS
#undef HIPFIRE_MOE_PROJECTION_MIN_BLOCKS
#undef HIPFIRE_MOE_PROJECTION_BLOCK
