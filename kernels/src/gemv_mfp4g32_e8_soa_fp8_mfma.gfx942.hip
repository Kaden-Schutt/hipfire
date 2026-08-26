// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — experimental MFP4-E8 SoA decode GEMV via gfx942 FP8 MFMA.

#include <hip/hip_runtime.h>

typedef float vfloat4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_fp8_mfma_gfx942(
    unsigned char b
) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(
        float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index_fp8_mfma_gfx942(
    unsigned int idx,
    float p[8]
) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; ++i) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sl += e[i];
    }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; ++i) {
        const float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

__device__ __forceinline__ unsigned long long pack_fp8x8_gfx942(
    const float v[8]
) {
    unsigned int lo = 0u;
    unsigned int hi = 0u;
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(v[4], v[5], hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(v[6], v[7], hi, true);
    return (unsigned long long)lo | ((unsigned long long)hi << 32);
}

// One wave computes 16 output rows. gfx942 has no scalar FP8 dot4, so the
// batch-1 vector is broadcast across all 16 MFMA columns. Only column zero is
// retained. This intentionally trades surplus matrix FLOPs for 16-row weight
// decode/issue parallelism; the matrix cores are otherwise idle during GEMV.
//
// The activation conversion is performed once per 32-element K block by four
// lanes, then broadcast as packed FP8 fragments. It is deliberately inline in
// this first micro-screen so the result includes the full arithmetic/quality
// cost without adding a producer-side scratch contract.
extern "C" __global__ __launch_bounds__(64)
void gemv_mfp4g32_e8_soa_fp8_mfma_gfx942(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int lane = threadIdx.x;
    const int row_lane = lane & 15;
    const int k_strip = lane >> 4;
    const int row_start = (int)blockIdx.x * 16;
    const int my_row = row_start + row_lane;
    const int safe_row = my_row < M ? my_row : M - 1;
    if (row_start >= M) return;

    const int n_blocks = K / 32;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total =
        16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)safe_row * row_total;
    const float row_scale = (float)__builtin_bit_cast(
        _Float16, *(const unsigned short*)hdr);
    const char* scale_base = hdr + 16;
    const char* cw_base = scale_base + scale_padded;

    vfloat4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = 0; block < n_blocks; ++block) {
        const unsigned int code = *(const unsigned int*)(
            cw_base + (long long)block * 16 + k_strip * 4);
        float w[8];
        e8_decode_index_fp8_mfma_gfx942(code, w);
        const unsigned long long a_frag = pack_fp8x8_gfx942(w);

        unsigned int x_lo = 0u;
        unsigned int x_hi = 0u;
        if (row_lane == 0) {
            const int base = block * 32 + k_strip * 8;
            float xv[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) xv[i] = x[base + i];
            const unsigned long long packed = pack_fp8x8_gfx942(xv);
            x_lo = (unsigned int)packed;
            x_hi = (unsigned int)(packed >> 32);
        }
        const int x_source_lane = k_strip * 16;
        x_lo = __shfl(x_lo, x_source_lane, 64);
        x_hi = __shfl(x_hi, x_source_lane, 64);
        const unsigned long long b_frag =
            (unsigned long long)x_lo | ((unsigned long long)x_hi << 32);

        const vfloat4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
        const vfloat4 partial = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            (long)a_frag, (long)b_frag, zero, 0, 0, 0);

        const float own_scale = row_scale
            * cvt_e4m3_scale_to_f32_fp8_mfma_gfx942(
                *(const unsigned char*)(scale_base + block))
            * 0.88f;
        float output_scales[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            output_scales[r] = __shfl(own_scale, k_strip * 4 + r, 64);
        }
        if (row_lane == 0) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r] += partial[r] * output_scales[r];
            }
        }
    }

    if (row_lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int output_row = row_start + k_strip * 4 + r;
            if (output_row < M) y[output_row] = acc[r];
        }
    }
}
