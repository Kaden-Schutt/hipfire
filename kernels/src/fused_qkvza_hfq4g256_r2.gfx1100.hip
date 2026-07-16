// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 Radiowave experiment for the fixed K=2048 DeltaNet QKVZA shape.
// One wave owns two adjacent output rows and reuses the same 32 activation
// values across both independent weight streams. Each row retains the shipping
// four-chain dot-product and wave32 reduction order.

__launch_bounds__(32, 16)
extern "C" __global__ void fused_qkvza_hfq4g256_k2048_r2(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ x,
    float*       __restrict__ y_qkv,
    float*       __restrict__ y_z,
    float*       __restrict__ y_beta,
    float*       __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    (void)K;
    const int tid = threadIdx.x;
    const int gid0 = blockIdx.x << 1;
    const int gid1 = gid0 + 1;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid0 >= total_m) return;
    const bool have_row1 = gid1 < total_m;

    const char* A0;
    const char* A1;
    float* y0;
    float* y1;
    int row0;
    int row1;

    if (gid0 < qkv_m) {
        A0 = A_qkv; y0 = y_qkv; row0 = gid0;
    } else if (gid0 < qkv_m + z_m) {
        A0 = A_z; y0 = y_z; row0 = gid0 - qkv_m;
    } else if (gid0 < qkv_m + z_m + beta_m) {
        A0 = A_beta; y0 = y_beta; row0 = gid0 - qkv_m - z_m;
    } else {
        A0 = A_alpha; y0 = y_alpha; row0 = gid0 - qkv_m - z_m - beta_m;
    }

    if (gid1 < qkv_m) {
        A1 = A_qkv; y1 = y_qkv; row1 = gid1;
    } else if (gid1 < qkv_m + z_m) {
        A1 = A_z; y1 = y_z; row1 = gid1 - qkv_m;
    } else if (gid1 < qkv_m + z_m + beta_m) {
        A1 = A_beta; y1 = y_beta; row1 = gid1 - qkv_m - z_m;
    } else {
        A1 = A_alpha; y1 = y_alpha; row1 = gid1 - qkv_m - z_m - beta_m;
    }

    // Eight 136-byte HFQ4 groups per K=2048 row.
    constexpr long long row_stride = 8LL * 136;
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

    // Keep the two quads in a compact loop. Fully unrolling this body doubles
    // the already dual-row instruction footprint and loses to instruction
    // fetch pressure on gfx1100.
    #pragma clang loop unroll(disable)
    for (int q = 0; q < 2; q++) {
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
