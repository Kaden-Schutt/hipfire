// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

// gfx1151 Radiowave experiment for the fixed K=2048 DeltaNet QKVZA shape.
// A physical wave64 computes two adjacent output rows.  The low half loads
// each 32-float activation stripe once and V_PERMLANE64 transfers the values
// to the matching lanes in the high half.  Each half retains its own weight
// stream, four accumulators, and wave32 reduction tree.

__device__ __forceinline__ float gfx1151_qkvza_other_half(float v) {
    return __uint_as_float(__builtin_amdgcn_permlane64(__float_as_uint(v)));
}

__launch_bounds__(64, 8)
extern "C" __global__ void fused_qkvza_hfq4g256_wave64_share_x_gfx1151(
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
    const int physical_lane = threadIdx.x;
    const int half = physical_lane >> 5;
    const int lane = physical_lane & 31;
    const int gid = (blockIdx.x << 1) + half;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv; y = y_qkv; row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z; y = y_z; row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta; y = y_beta; row = gid - qkv_m - z_m;
    } else {
        A = A_alpha; y = y_alpha; row = gid - qkv_m - z_m - beta_m;
    }

    constexpr long long row_stride = 8LL * 136;
    const char* row_ptr = A + (long long)row * row_stride;
    const int boff = lane * 4;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #define DOG_SHARED_X(g, acc) do { \
        float4 low_x0 = {0.0f, 0.0f, 0.0f, 0.0f}; \
        float4 low_x1 = {0.0f, 0.0f, 0.0f, 0.0f}; \
        if (half == 0) { \
            const int xb = (g) * 256 + lane * 8; \
            low_x0 = *reinterpret_cast<const float4*>(x + xb); \
            low_x1 = *reinterpret_cast<const float4*>(x + xb + 4); \
        } \
        const float ox0 = gfx1151_qkvza_other_half(low_x0.x); \
        const float ox1 = gfx1151_qkvza_other_half(low_x0.y); \
        const float ox2 = gfx1151_qkvza_other_half(low_x0.z); \
        const float ox3 = gfx1151_qkvza_other_half(low_x0.w); \
        const float ox4 = gfx1151_qkvza_other_half(low_x1.x); \
        const float ox5 = gfx1151_qkvza_other_half(low_x1.y); \
        const float ox6 = gfx1151_qkvza_other_half(low_x1.z); \
        const float ox7 = gfx1151_qkvza_other_half(low_x1.w); \
        const float xv0 = half == 0 ? low_x0.x : ox0; \
        const float xv1 = half == 0 ? low_x0.y : ox1; \
        const float xv2 = half == 0 ? low_x0.z : ox2; \
        const float xv3 = half == 0 ? low_x0.w : ox3; \
        const float xv4 = half == 0 ? low_x1.x : ox4; \
        const float xv5 = half == 0 ? low_x1.y : ox5; \
        const float xv6 = half == 0 ? low_x1.z : ox6; \
        const float xv7 = half == 0 ? low_x1.w : ox7; \
        const char* gp = row_ptr + (g) * 136; \
        const float sc = __builtin_bit_cast(float, \
            *(const unsigned int*)(gp)); \
        const float zp = __builtin_bit_cast(float, \
            *(const unsigned int*)(gp + 4)); \
        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff); \
        (acc) += (sc * (float)(pk & 0xFu)          + zp) * xv0 \
               + (sc * (float)((pk >> 4) & 0xFu)  + zp) * xv1 \
               + (sc * (float)((pk >> 8) & 0xFu)  + zp) * xv2 \
               + (sc * (float)((pk >> 12) & 0xFu) + zp) * xv3 \
               + (sc * (float)((pk >> 16) & 0xFu) + zp) * xv4 \
               + (sc * (float)((pk >> 20) & 0xFu) + zp) * xv5 \
               + (sc * (float)((pk >> 24) & 0xFu) + zp) * xv6 \
               + (sc * (float)((pk >> 28) & 0xFu) + zp) * xv7; \
    } while (0)

    // Keep the fixed eight groups in the shipping four-chain ownership order.
    #pragma clang loop unroll(disable)
    for (int q = 0; q < 2; q++) {
        const int g = q << 2;
        DOG_SHARED_X(g + 0, acc0);
        DOG_SHARED_X(g + 1, acc1);
        DOG_SHARED_X(g + 2, acc2);
        DOG_SHARED_X(g + 3, acc3);
    }
    #undef DOG_SHARED_X

    float sum = (acc0 + acc1) + (acc2 + acc3);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);

    if (lane == 0) y[row] = sum;
}
