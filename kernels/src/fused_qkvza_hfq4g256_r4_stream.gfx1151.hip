// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

using hipfire_qkvza_r4_u32x2 =
    unsigned int __attribute__((ext_vector_type(2)));
using hipfire_qkvza_r4_f32x4 = float __attribute__((ext_vector_type(4)));

// gfx1151 fixed-K large-QKVZA row tile.
//
// One wave owns four adjacent rows.  The shipping one-row kernel reloads the
// same 2048-float activation for every output row; the earlier R2 experiment
// reduced those loads but kept a complete 32-value quad live beside two rows'
// accumulators and reached the 96-VGPR edge.  This schedule traverses K one
// 256-element group at a time: eight activation values stay live while four
// independent weight rows consume them.  Four accumulator chains per row and
// their final reduction order remain identical to the shipping kernel.
//
// The host admits this kernel only when all four matrix row counts are
// divisible by four, so a tile never crosses a matrix boundary and no partial
// row path is required.
__launch_bounds__(32, 16)
extern "C" __global__ void fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151(
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
    const int gid = blockIdx.x << 2;
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
    const unsigned int row_off0 = (unsigned int)((long long)(row + 0) * row_stride);
    const unsigned int row_off1 = (unsigned int)((long long)(row + 1) * row_stride);
    const unsigned int row_off2 = (unsigned int)((long long)(row + 2) * row_stride);
    const unsigned int row_off3 = (unsigned int)((long long)(row + 3) * row_stride);
    const char* row_ptr0 = A + row_off0;
    const char* row_ptr1 = A + row_off1;
    const char* row_ptr2 = A + row_off2;
    const char* row_ptr3 = A + row_off3;
    // One matrix resource plus four wave-uniform scalar offsets replaces four
    // complete descriptors.  This keeps row selection on SALU and trims the
    // long-lived address state that pushed the first R4 form into scratch.
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(x), 0, 0xffffffffu, 0x31004000u);
    const int boff = tid * 4;

    float r0a0 = 0.0f, r0a1 = 0.0f, r0a2 = 0.0f, r0a3 = 0.0f;
    float r1a0 = 0.0f, r1a1 = 0.0f, r1a2 = 0.0f, r1a3 = 0.0f;
    float r2a0 = 0.0f, r2a1 = 0.0f, r2a2 = 0.0f, r2a3 = 0.0f;
    float r3a0 = 0.0f, r3a1 = 0.0f, r3a2 = 0.0f, r3a3 = 0.0f;

#define HIPFIRE_QKVZA_R4_DOG(pk, sc, zp, acc, xa, xb) do { \
        (acc) += ((sc) * (float)((pk) & 0xFu)          + (zp)) * (xa).x \
               + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (xa).y \
               + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (xa).z \
               + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (xa).w \
               + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (xb).x \
               + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (xb).y \
               + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (xb).z \
               + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (xb).w; \
    } while (0)

#define HIPFIRE_QKVZA_R4_GROUP(group, acc0, acc1, acc2, acc3) do { \
        const int g_ = (group); \
        const int xbyte_ = (g_ * 256 + tid * 8) * 4; \
        const hipfire_qkvza_r4_f32x4 xa_ = __builtin_bit_cast( \
            hipfire_qkvza_r4_f32x4, __builtin_amdgcn_raw_buffer_load_b128( \
                x_rsrc, xbyte_, 0, 0)); \
        const hipfire_qkvza_r4_f32x4 xb_ = __builtin_bit_cast( \
            hipfire_qkvza_r4_f32x4, __builtin_amdgcn_raw_buffer_load_b128( \
                x_rsrc, xbyte_ + 16, 0, 0)); \
        const int goff_ = g_ * 136; \
        { \
            const char* gp0_ = row_ptr0 + goff_; \
            const hipfire_qkvza_r4_u32x2 h0_ = *( \
                const hipfire_qkvza_r4_u32x2 __attribute__((address_space(4)))*)gp0_; \
            const unsigned int pk0_ = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
                weight_rsrc, goff_ + 8 + boff, row_off0); \
            HIPFIRE_QKVZA_R4_DOG(pk0_, __builtin_bit_cast(float, h0_.x), \
                __builtin_bit_cast(float, h0_.y), acc0, xa_, xb_); \
        } \
        asm volatile("" ::: "memory"); \
        { \
            const char* gp1_ = row_ptr1 + goff_; \
            const hipfire_qkvza_r4_u32x2 h1_ = *( \
                const hipfire_qkvza_r4_u32x2 __attribute__((address_space(4)))*)gp1_; \
            const unsigned int pk1_ = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
                weight_rsrc, goff_ + 8 + boff, row_off1); \
            HIPFIRE_QKVZA_R4_DOG(pk1_, __builtin_bit_cast(float, h1_.x), \
                __builtin_bit_cast(float, h1_.y), acc1, xa_, xb_); \
        } \
        asm volatile("" ::: "memory"); \
        { \
            const char* gp2_ = row_ptr2 + goff_; \
            const hipfire_qkvza_r4_u32x2 h2_ = *( \
                const hipfire_qkvza_r4_u32x2 __attribute__((address_space(4)))*)gp2_; \
            const unsigned int pk2_ = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
                weight_rsrc, goff_ + 8 + boff, row_off2); \
            HIPFIRE_QKVZA_R4_DOG(pk2_, __builtin_bit_cast(float, h2_.x), \
                __builtin_bit_cast(float, h2_.y), acc2, xa_, xb_); \
        } \
        asm volatile("" ::: "memory"); \
        { \
            const char* gp3_ = row_ptr3 + goff_; \
            const hipfire_qkvza_r4_u32x2 h3_ = *( \
                const hipfire_qkvza_r4_u32x2 __attribute__((address_space(4)))*)gp3_; \
            const unsigned int pk3_ = HIPFIRE_WEIGHT_LOAD_SPLIT_U32( \
                weight_rsrc, goff_ + 8 + boff, row_off3); \
            HIPFIRE_QKVZA_R4_DOG(pk3_, __builtin_bit_cast(float, h3_.x), \
                __builtin_bit_cast(float, h3_.y), acc3, xa_, xb_); \
        } \
        asm volatile("" ::: "memory"); \
    } while (0)

    // Keep the two four-group quads rolled for instruction-cache locality;
    // each group is consumed before the next group makes another x stripe live.
    #pragma clang loop unroll(disable)
    for (int q = 0; q < 2; q++) {
        const int g = q << 2;
        HIPFIRE_QKVZA_R4_GROUP(g + 0, r0a0, r1a0, r2a0, r3a0);
        HIPFIRE_QKVZA_R4_GROUP(g + 1, r0a1, r1a1, r2a1, r3a1);
        HIPFIRE_QKVZA_R4_GROUP(g + 2, r0a2, r1a2, r2a2, r3a2);
        HIPFIRE_QKVZA_R4_GROUP(g + 3, r0a3, r1a3, r2a3, r3a3);
    }
    #undef HIPFIRE_QKVZA_R4_GROUP
    #undef HIPFIRE_QKVZA_R4_DOG

    float sum0 = (r0a0 + r0a1) + (r0a2 + r0a3);
    float sum1 = (r1a0 + r1a1) + (r1a2 + r1a3);
    float sum2 = (r2a0 + r2a1) + (r2a2 + r2a3);
    float sum3 = (r3a0 + r3a1) + (r3a2 + r3a3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
    }
    if (tid == 0) {
        y[row + 0] = sum0;
        y[row + 1] = sum1;
        y[row + 2] = sum2;
        y[row + 3] = sum3;
    }
}
