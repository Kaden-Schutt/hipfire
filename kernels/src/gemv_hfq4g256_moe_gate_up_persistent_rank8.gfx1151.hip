// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

using hipfire_gate_persistent_u32x2 =
    unsigned int __attribute__((ext_vector_type(2)));

// gfx1151 fixed-K=2048, fixed-rank=8 gate/up producer.
//
// One 256-thread workgroup contains one wave per selected expert rank. The
// shared activation is staged once, every wave resolves its route once, and
// the workgroup then walks a persistent row stripe. This replaces the base
// shape's repeated activation and routing loads for every (row, rank) pair
// while keeping gate and up fused inside a wave and preserving their four
// accumulator chains exactly.
__launch_bounds__(256, 4)
extern "C" __global__ void
gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M,
    int K
) {
    (void)K;
    const int block_tid = threadIdx.x;
    const int krank = block_tid >> 5;
    const int tid = block_tid & 31;
    const int mi = M >> 1;

    // Transposed [group][element][lane] layout avoids the lane*8 LDS bank
    // pattern. All eight waves cooperate on the one immutable activation.
    __shared__ float x_lds[2048];
    #pragma unroll
    for (int d = block_tid; d < 2048; d += 256) {
        const int group = d >> 8;
        const int j = (d >> 5) & 7;
        const int lane = d & 31;
        x_lds[d] = x[group * 256 + lane * 8 + j];
    }
    __syncthreads();

    // Rank is fixed for the lifetime of this persistent wave, so routing and
    // the expert SRD are paid once rather than once for every output row.
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const auto expert_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    constexpr unsigned int row_bytes = 8u * 136u;
    const int boff = tid * 4;

    #define DOG_LDS(sc, zp, pk, group, acc) do { \
        (acc) += ((sc) * (float)((pk) & 0xFu)          + (zp)) * x_lds[(group) * 256 + 0 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * x_lds[(group) * 256 + 1 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * x_lds[(group) * 256 + 2 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * x_lds[(group) * 256 + 3 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * x_lds[(group) * 256 + 4 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * x_lds[(group) * 256 + 5 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * x_lds[(group) * 256 + 6 * 32 + tid] \
               + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * x_lds[(group) * 256 + 7 * 32 + tid]; \
    } while (0)

    #define PROCESS_GROUP(group, gacc, uacc) do { \
        const unsigned int group_offset = (unsigned int)(group) * 136u; \
        const unsigned int gate_offset = gate_row_offset + group_offset; \
        const unsigned int up_offset = up_row_offset + group_offset; \
        const char* gate_group = A + gate_offset; \
        const char* up_group = A + up_offset; \
        const hipfire_gate_persistent_u32x2 gate_h = *( \
            const hipfire_gate_persistent_u32x2 __attribute__((address_space(4)))*)gate_group; \
        const unsigned int gate_pk = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, (unsigned int)boff + 8u, gate_offset); \
        DOG_LDS(__builtin_bit_cast(float, gate_h.x), \
            __builtin_bit_cast(float, gate_h.y), gate_pk, group, gacc); \
        asm volatile("" ::: "memory"); \
        const hipfire_gate_persistent_u32x2 up_h = *( \
            const hipfire_gate_persistent_u32x2 __attribute__((address_space(4)))*)up_group; \
        const unsigned int up_pk = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
            expert_rsrc, (unsigned int)boff + 8u, up_offset); \
        DOG_LDS(__builtin_bit_cast(float, up_h.x), \
            __builtin_bit_cast(float, up_h.y), up_pk, group, uacc); \
        asm volatile("" ::: "memory"); \
    } while (0)

    for (int row = (int)blockIdx.x; row < mi; row += (int)gridDim.x) {
        const unsigned int gate_row_offset = (unsigned int)row * row_bytes;
        const unsigned int up_row_offset = (unsigned int)(row + mi) * row_bytes;
        float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
        float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;

        PROCESS_GROUP(0, gacc0, uacc0);
        PROCESS_GROUP(1, gacc1, uacc1);
        PROCESS_GROUP(2, gacc2, uacc2);
        PROCESS_GROUP(3, gacc3, uacc3);
        PROCESS_GROUP(4, gacc0, uacc0);
        PROCESS_GROUP(5, gacc1, uacc1);
        PROCESS_GROUP(6, gacc2, uacc2);
        PROCESS_GROUP(7, gacc3, uacc3);

        float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
        float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
        for (int offset = 16; offset > 0; offset >>= 1) {
            gacc += __shfl_down(gacc, offset);
            uacc += __shfl_down(uacc, offset);
        }
        if (tid == 0) {
            y_gate[krank * mi + row] = gacc;
            y_up[krank * mi + row] = uacc;
        }
    }

    #undef PROCESS_GROUP
    #undef DOG_LDS
}
