// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_MOE_DOWN_ROW2_KERNEL
#define HIPFIRE_MOE_DOWN_ROW2_KERNEL \
    gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151
#endif

#ifndef HIPFIRE_MOE_DOWN_ROW2_CLUSTERED
#define HIPFIRE_MOE_DOWN_ROW2_CLUSTERED 0
#endif

// gfx1151 K=512 routed-down compromise: two rows per wave.  Row1 proved that
// the full routing/weight stream can be VMEM-only at 29 VGPR but did not keep
// enough independent requests in flight; the four-row source emitted 92 VGPR.
// This variant targets the intervening occupancy/MLP point while retaining the
// exact two-slot accumulation and wave32 reduction order for each output row.

__launch_bounds__(32, 16)
extern "C" __global__ void
HIPFIRE_MOE_DOWN_ROW2_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M, int K, int K_TOP
) {
    const int row0 = blockIdx.x * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
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

    const char* row_ptr0 = A + (long long)row0 * 272;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * 272;
    const auto row_rsrc0 = HIPFIRE_WEIGHT_RSRC(row_ptr0);
    const auto row_rsrc1 = HIPFIRE_WEIGHT_RSRC(row_ptr1);
    const int boff = tid * 4;

    const float4 x0a = *reinterpret_cast<const float4*>(x + tid * 8);
    const float4 x0b = *reinterpret_cast<const float4*>(x + tid * 8 + 4);
    const float4 x1a = *reinterpret_cast<const float4*>(x + 256 + tid * 8);
    const float4 x1b = *reinterpret_cast<const float4*>(x + 260 + tid * 8);

    #define DOG_VALUES(sc, zp, pk, accumulator, xa, xb) do { \
        (accumulator) += (sc * (float)(pk & 0xFu) + zp) * (xa).x \
            + (sc * (float)((pk >> 4) & 0xFu) + zp) * (xa).y \
            + (sc * (float)((pk >> 8) & 0xFu) + zp) * (xa).z \
            + (sc * (float)((pk >> 12) & 0xFu) + zp) * (xa).w \
            + (sc * (float)((pk >> 16) & 0xFu) + zp) * (xb).x \
            + (sc * (float)((pk >> 20) & 0xFu) + zp) * (xb).y \
            + (sc * (float)((pk >> 24) & 0xFu) + zp) * (xb).z \
            + (sc * (float)((pk >> 28) & 0xFu) + zp) * (xb).w; \
    } while (0)

    float acc0 = 0.0f, acc1 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f;
#if HIPFIRE_MOE_DOWN_ROW2_CLUSTERED
    // Keep the 12 weight VMEM operations ahead of the dependent VALU. The
    // default immediate-consume source emitted 90 waits / 72 delay_alu at only
    // 34 VGPR; this trades some of that unused register headroom for request
    // overlap while preserving each row's accumulator and FMA order.
    #define LOAD_GROUP(resource, offset, sc, zp, pk) do { \
        (sc) = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_U32(resource, (offset))); \
        (zp) = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_U32(resource, (offset) + 4u)); \
        (pk) = HIPFIRE_WEIGHT_LOAD_U32( \
            resource, (offset) + 8u + (unsigned int)boff); \
    } while (0)
    float sc00, zp00, sc01, zp01, sc10, zp10, sc11, zp11;
    unsigned int pk00, pk01, pk10, pk11;
    LOAD_GROUP(row_rsrc0, 0u, sc00, zp00, pk00);
    LOAD_GROUP(row_rsrc0, 136u, sc01, zp01, pk01);
    LOAD_GROUP(row_rsrc1, 0u, sc10, zp10, pk10);
    LOAD_GROUP(row_rsrc1, 136u, sc11, zp11, pk11);
    #undef LOAD_GROUP
    DOG_VALUES(sc00, zp00, pk00, acc0, x0a, x0b);
    DOG_VALUES(sc10, zp10, pk10, bcc0, x0a, x0b);
    DOG_VALUES(sc01, zp01, pk01, acc1, x1a, x1b);
    DOG_VALUES(sc11, zp11, pk11, bcc1, x1a, x1b);
#else
    #define LOAD_DOG(resource, offset, accumulator, xa, xb) do { \
        const float sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_U32(resource, (offset))); \
        const float zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_U32(resource, (offset) + 4u)); \
        const unsigned int pk = HIPFIRE_WEIGHT_LOAD_U32( \
            resource, (offset) + 8u + (unsigned int)boff); \
        DOG_VALUES(sc, zp, pk, accumulator, xa, xb); \
    } while (0)
    LOAD_DOG(row_rsrc0, 0u, acc0, x0a, x0b);
    LOAD_DOG(row_rsrc1, 0u, bcc0, x0a, x0b);
    LOAD_DOG(row_rsrc0, 136u, acc1, x1a, x1b);
    LOAD_DOG(row_rsrc1, 136u, bcc1, x1a, x1b);
    #undef LOAD_DOG
#endif
    #undef DOG_VALUES

    float acc = acc0 + acc1;
    float bcc = bcc0 + bcc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        const long long out_base = (long long)routing_index * M;
        expert_outputs[out_base + row0] = acc;
        if (have_row1) expert_outputs[out_base + row1] = bcc;
    }
}
