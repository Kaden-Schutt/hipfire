// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 Radiowave K=512 routed-down experiment. One wave owns eight
// consecutive output rows and reuses the same 16 activation values per lane
// across all eight independent weight streams. This halves both activation
// requests and workgroup count relative to the four-row production kernel.

__launch_bounds__(32, 8)
extern "C" __global__ void
gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float*       __restrict__ expert_outputs,
    int M, int K, int K_TOP
) {
    (void)K;
    const int row0 = blockIdx.x << 3;
    if (row0 >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)routing_base + krank) * 512;

    constexpr long long row_stride = 2LL * 136;
    const char* row_ptr0 = A + (long long)(row0 + 0) * row_stride;
    const char* row_ptr1 = A + (long long)(row0 + 1) * row_stride;
    const char* row_ptr2 = A + (long long)(row0 + 2) * row_stride;
    const char* row_ptr3 = A + (long long)(row0 + 3) * row_stride;
    const char* row_ptr4 = A + (long long)(row0 + 4) * row_stride;
    const char* row_ptr5 = A + (long long)(row0 + 5) * row_stride;
    const char* row_ptr6 = A + (long long)(row0 + 6) * row_stride;
    const char* row_ptr7 = A + (long long)(row0 + 7) * row_stride;
    const int boff = tid * 4;

    const int base = tid * 8;
    const float4 xv0a = *reinterpret_cast<const float4*>(x + base);
    const float4 xv0b = *reinterpret_cast<const float4*>(x + base + 4);
    const float4 xv1a = *reinterpret_cast<const float4*>(x + base + 256);
    const float4 xv1b = *reinterpret_cast<const float4*>(x + base + 260);

    float a00 = 0.0f, a01 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f;
    float a40 = 0.0f, a41 = 0.0f;
    float a50 = 0.0f, a51 = 0.0f;
    float a60 = 0.0f, a61 = 0.0f;
    float a70 = 0.0f, a71 = 0.0f;

    #define DOG_ONE(gp, acc, xa, xb) do { \
        const float sc = __builtin_bit_cast(float, \
            *(const unsigned int*)(gp)); \
        const float zp = __builtin_bit_cast(float, \
            *(const unsigned int*)((gp) + 4)); \
        const unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (acc) += (sc * (float)(pk & 0xFu)          + zp) * (xa).x \
               + (sc * (float)((pk >> 4) & 0xFu)  + zp) * (xa).y \
               + (sc * (float)((pk >> 8) & 0xFu)  + zp) * (xa).z \
               + (sc * (float)((pk >> 12) & 0xFu) + zp) * (xa).w \
               + (sc * (float)((pk >> 16) & 0xFu) + zp) * (xb).x \
               + (sc * (float)((pk >> 20) & 0xFu) + zp) * (xb).y \
               + (sc * (float)((pk >> 24) & 0xFu) + zp) * (xb).z \
               + (sc * (float)((pk >> 28) & 0xFu) + zp) * (xb).w; \
    } while (0)

    #define DOG_ROW(ptr, acc0, acc1) do { \
        DOG_ONE(ptr, acc0, xv0a, xv0b); \
        DOG_ONE((ptr) + 136, acc1, xv1a, xv1b); \
    } while (0)

    DOG_ROW(row_ptr0, a00, a01);
    DOG_ROW(row_ptr1, a10, a11);
    DOG_ROW(row_ptr2, a20, a21);
    DOG_ROW(row_ptr3, a30, a31);
    DOG_ROW(row_ptr4, a40, a41);
    DOG_ROW(row_ptr5, a50, a51);
    DOG_ROW(row_ptr6, a60, a61);
    DOG_ROW(row_ptr7, a70, a71);
    #undef DOG_ROW
    #undef DOG_ONE

    float s0 = a00 + a01;
    float s1 = a10 + a11;
    float s2 = a20 + a21;
    float s3 = a30 + a31;
    float s4 = a40 + a41;
    float s5 = a50 + a51;
    float s6 = a60 + a61;
    float s7 = a70 + a71;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down(s0, offset);
        s1 += __shfl_down(s1, offset);
        s2 += __shfl_down(s2, offset);
        s3 += __shfl_down(s3, offset);
        s4 += __shfl_down(s4, offset);
        s5 += __shfl_down(s5, offset);
        s6 += __shfl_down(s6, offset);
        s7 += __shfl_down(s7, offset);
    }

    if (tid == 0) {
        const long long out =
            ((long long)routing_base + krank) * (long long)M + row0;
        expert_outputs[out + 0] = s0;
        expert_outputs[out + 1] = s1;
        expert_outputs[out + 2] = s2;
        expert_outputs[out + 3] = s3;
        expert_outputs[out + 4] = s4;
        expert_outputs[out + 5] = s5;
        expert_outputs[out + 6] = s6;
        expert_outputs[out + 7] = s7;
    }
}
