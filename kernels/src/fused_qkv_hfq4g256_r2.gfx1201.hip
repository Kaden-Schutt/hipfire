// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fixed-shape gfx1201 FullAttention QKV projection for dense Qwen3.5-27B.
// One wave owns two adjacent output rows. The rows keep independent weight
// streams and accumulator chains while sharing each K=5120 activation load.

__launch_bounds__(32, 16)
extern "C" __global__ void fused_qkv_hfq4g256_r2_k5120_global_gfx1201(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const float* __restrict__ x,
    float*       __restrict__ y_q,
    float*       __restrict__ y_k,
    float*       __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K
) {
    (void)K;
    const int tid = threadIdx.x;
    const int gid0 = blockIdx.x << 1;
    const int gid1 = gid0 + 1;
    const int total_m = q_m + k_m + v_m;
    if (gid0 >= total_m) return;
    const bool have_row1 = gid1 < total_m;

    const char* A0;
    const char* A1;
    float* y0;
    float* y1;
    int row0;
    int row1;

    if (gid0 < q_m) {
        A0 = A_q; y0 = y_q; row0 = gid0;
    } else if (gid0 < q_m + k_m) {
        A0 = A_k; y0 = y_k; row0 = gid0 - q_m;
    } else {
        A0 = A_v; y0 = y_v; row0 = gid0 - q_m - k_m;
    }

    if (gid1 < q_m) {
        A1 = A_q; y1 = y_q; row1 = gid1;
    } else if (gid1 < q_m + k_m) {
        A1 = A_k; y1 = y_k; row1 = gid1 - q_m;
    } else {
        A1 = A_v; y1 = y_v; row1 = gid1 - q_m - k_m;
    }

    constexpr int groups_per_row = 20;
    constexpr int quads = 5;
    constexpr long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A0 + (long long)row0 * row_stride;
    const char* row_ptr1 = have_row1
        ? A1 + (long long)row1 * row_stride
        : row_ptr0;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc0 = HIPFIRE_WEIGHT_RSRC(row_ptr0);
    const auto row_rsrc1 = HIPFIRE_WEIGHT_RSRC(row_ptr1);
#endif

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

    #define LOAD_X8_INTO(base, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (base)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (base) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y; \
        const float pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y; \
        const float pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    #define DOG_X8(rsrc, gp, off, acc, pfx) do { \
        const float sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, gp, off)); \
        const float zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, (gp) + 4, (off) + 4)); \
        const unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            rsrc, (gp) + 8 + boff, (off) + 8 + boff); \
        (acc) += (sc * (float)((pk) & 0xFu)         + zp) * pfx##0 \
               + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * pfx##1 \
               + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * pfx##2 \
               + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * pfx##3 \
               + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * pfx##4 \
               + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * pfx##5 \
               + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * pfx##6 \
               + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * pfx##7; \
    } while (0)

    // Keep the dual-row body compact; full unrolling pushes this fixed-K
    // schedule into scratch and makes it ineligible for retained PM4 replay.
    #pragma clang loop unroll(disable)
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int xbase = g * 256 + tid * 8;
        LOAD_X8_INTO(xbase, xa);
        LOAD_X8_INTO(xbase + 256, xb);
        LOAD_X8_INTO(xbase + 512, xc);
        LOAD_X8_INTO(xbase + 768, xd);

        const int off0 = (g + 0) * 136;
        const int off1 = (g + 1) * 136;
        const int off2 = (g + 2) * 136;
        const int off3 = (g + 3) * 136;
        DOG_X8(row_rsrc0, row_ptr0 + off0, off0, a0, xa);
        DOG_X8(row_rsrc1, row_ptr1 + off0, off0, b0, xa);
        DOG_X8(row_rsrc0, row_ptr0 + off1, off1, a1, xb);
        DOG_X8(row_rsrc1, row_ptr1 + off1, off1, b1, xb);
        DOG_X8(row_rsrc0, row_ptr0 + off2, off2, a2, xc);
        DOG_X8(row_rsrc1, row_ptr1 + off2, off2, b2, xc);
        DOG_X8(row_rsrc0, row_ptr0 + off3, off3, a3, xd);
        DOG_X8(row_rsrc1, row_ptr1 + off3, off3, b3, xd);
    }
    #undef DOG_X8
    #undef LOAD_X8_INTO

    float sum0 = (a0 + a1) + (a2 + a3);
    float sum1 = (b0 + b1) + (b2 + b3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (tid == 0) {
        y0[row0] = sum0;
        if (have_row1) y1[row1] = sum1;
    }
}
