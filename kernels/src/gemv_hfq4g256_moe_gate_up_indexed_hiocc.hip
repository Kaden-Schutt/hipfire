// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// High-occupancy experiment variant of `gemv_hfq4g256_moe_gate_up_k8_indexed`
// (opt-in HIPFIRE_MOE_GATE_UP_HIOCC=1, default OFF). Identical routing/output
// contract as the base kernel — same grid (M blocks, `row >= mi` guard), same
// [32,1,1] block, same y_gate[K_TOP×MI] / y_up[K_TOP×MI] layout. Feeds the
// existing batched silu_mul_rotate. Coherence-equivalent (NOT byte-exact): the
// accumulator reassociation reorders the float adds.
//
// Intent: REDUCE VGPR to raise occupancy (the opposite direction of the
// row-tile / P=8 / dp4a levers, which ADDED VGPR and regressed by lowering
// occupancy). Two VGPR-reduction techniques vs the base:
//   1. Accumulator tree 4-way (gacc0..3/uacc0..3) → 2-way (gacc0/1, uacc0/1).
//   2. Wave-uniform group scale/zp forced to SGPR via __builtin_amdgcn_readfirstlane
//      (the base loads them as vector global_loads even though the address is
//      wave-uniform).
//
// MEASURED RESULT (gfx1201 / R9700, ROCm 7.2.2, qwen3.6-35b-a3b.mq4r q8):
// this does NOT raise occupancy and does NOT improve decode. The base kernel
// already sits at 96 VGPR = floor(1536/96) = 16 waves/SIMD = the hardware MAX
// VGPR-occupancy on RDNA4. Both techniques land VGPR at 95 (SGPR 21→33), still
// 16 waves/SIMD — there is no VGPR headroom to unlock. Achieved occupancy
// (~71% at profile_standard) is dispatch/latency-limited, not VGPR-limited, and
// MemUnitBusy is ~75% (the memory pipe is NOT half-idle). Kept as an opt-in,
// falsified-in-place experiment record; see the PR / kernel-oracle notes.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed_hiocc(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int mi = M >> 1;
    const int row = blockIdx.x;
    if (row >= mi) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    // Read expert_id from device and index into the pointer table.
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;

    float gacc0 = 0.0f, gacc1 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8(b) \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]

    // sc/zp are wave-uniform (address depends only on row/expert, not tid):
    // force them into SGPR so they stop consuming VGPR.
    #define DOG_X8(gp, a) do { \
        float sc = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(*(const unsigned int*)(gp))); \
        float zp = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(*(const unsigned int*)((gp) + 4))); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        {
            LOAD_X8(base0);
            DOG_X8(gate_ptr + g * 136, gacc0);
            DOG_X8(up_ptr + g * 136, uacc0);
        }
        {
            LOAD_X8(base0 + 256);
            DOG_X8(gate_ptr + (g + 1) * 136, gacc1);
            DOG_X8(up_ptr + (g + 1) * 136, uacc1);
        }
        {
            LOAD_X8(base0 + 512);
            DOG_X8(gate_ptr + (g + 2) * 136, gacc0);
            DOG_X8(up_ptr + (g + 2) * 136, uacc0);
        }
        {
            LOAD_X8(base0 + 768);
            DOG_X8(gate_ptr + (g + 3) * 136, gacc1);
            DOG_X8(up_ptr + (g + 3) * 136, uacc1);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(gate_ptr + g * 136, gacc0);
        DOG_X8(up_ptr + g * 136, uacc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(gate_ptr + g * 136, gacc1);
        DOG_X8(up_ptr + g * 136, uacc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(gate_ptr + g * 136, gacc0);
        DOG_X8(up_ptr + g * 136, uacc0);
    }
    #undef DOG_X8
    #undef LOAD_X8

    float gacc = gacc0 + gacc1;
    float uacc = uacc0 + uacc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        gacc += __shfl_down(gacc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        uacc += __shfl_down(uacc, offset);

    if (tid == 0) {
        y_gate[krank * mi + row] = gacc;
        y_up[krank * mi + row] = uacc;
    }
}
