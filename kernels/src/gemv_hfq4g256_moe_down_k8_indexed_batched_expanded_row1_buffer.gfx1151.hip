// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 decode specialization for the K=512 routed-down projection.
//
// One wave owns one output row instead of four.  The four-row source raises
// temporal-buffer codegen from 79 to 92 VGPR, which erases the cache-lowering
// win on Strix Halo.  This shape keeps two independent K=512 accumulators and
// exact pairwise/reduction order while exposing four independently schedulable
// waves.  Routing-table, expert-pointer, and weight reads all stay in VMEM so
// a hash-bound Radiowave manifest can certify the lean Redline cache boundary.

__launch_bounds__(32, 16)
extern "C" __global__ void
gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;
    const int routing_index = bid * K_TOP + krank;

    const auto indices_rsrc = HIPFIRE_WEIGHT_RSRC(
        reinterpret_cast<const char*>(topk_indices));
    const unsigned int expert_id = HIPFIRE_WEIGHT_LOAD_U32(
        indices_rsrc, (unsigned int)routing_index * 4u);
    const auto pointers_rsrc = HIPFIRE_WEIGHT_RSRC(
        reinterpret_cast<const char*>(expert_ptrs));
    const unsigned int expert_lo = HIPFIRE_WEIGHT_LOAD_U32(
        pointers_rsrc, expert_id * 8u);
    const unsigned int expert_hi = HIPFIRE_WEIGHT_LOAD_U32(
        pointers_rsrc, expert_id * 8u + 4u);
    const unsigned long long expert_bits =
        (unsigned long long)expert_lo | ((unsigned long long)expert_hi << 32);
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_bits);
    const float* __restrict__ x = rot_batch + (long long)routing_index * 512;

    const char* row_ptr = A + (long long)row * 272;
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
    const int boff = tid * 4;

    const float4 x0a = *reinterpret_cast<const float4*>(x + tid * 8);
    const float4 x0b = *reinterpret_cast<const float4*>(x + tid * 8 + 4);
    const float4 x1a = *reinterpret_cast<const float4*>(x + 256 + tid * 8);
    const float4 x1b = *reinterpret_cast<const float4*>(x + 260 + tid * 8);

    #define DOG_GROUP(offset, accumulator, xa, xb) do { \
        const float sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_U32(row_rsrc, (offset))); \
        const float zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_U32(row_rsrc, (offset) + 4u)); \
        const unsigned int pk = HIPFIRE_WEIGHT_LOAD_U32( \
            row_rsrc, (offset) + 8u + (unsigned int)boff); \
        (accumulator) += (sc * (float)(pk & 0xFu) + zp) * (xa).x \
            + (sc * (float)((pk >> 4) & 0xFu) + zp) * (xa).y \
            + (sc * (float)((pk >> 8) & 0xFu) + zp) * (xa).z \
            + (sc * (float)((pk >> 12) & 0xFu) + zp) * (xa).w \
            + (sc * (float)((pk >> 16) & 0xFu) + zp) * (xb).x \
            + (sc * (float)((pk >> 20) & 0xFu) + zp) * (xb).y \
            + (sc * (float)((pk >> 24) & 0xFu) + zp) * (xb).z \
            + (sc * (float)((pk >> 28) & 0xFu) + zp) * (xb).w; \
    } while (0)

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    DOG_GROUP(0u, acc0, x0a, x0b);
    DOG_GROUP(136u, acc1, x1a, x1b);
    #undef DOG_GROUP

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const long long out_base = (long long)routing_index * M;
        expert_outputs[out_base + row] = acc;
    }
}
