// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — wide-query dense MFP4-E8-SoA WMMA GEMM for gfx1151 prefill.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#ifndef E8_PREFILL_WIDE_TILES
#define E8_PREFILL_WIDE_TILES 8
#endif

#ifndef E8_PREFILL_WIDE_KERNEL_NAME
#define E8_PREFILL_WIDE_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b8_gfx1151
#endif

__device__ __forceinline__ float e8_wide_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned int mant = (unsigned int)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 =
        __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ void e8_wide_decode(
    unsigned int idx,
    float p[8]
) {
    const int coset = (int)(idx >> 31);
    unsigned int e[8];
    unsigned int sl = 0;
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (4 * i)) & 0xFu;
        sl += e[i];
    }
    const unsigned int p7 = ((idx >> 28) & 0x7u) << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

// One wave retains 8 or 16 independent 16-token WMMA accumulators. The
// decoded A fragment remains in registers and is consumed by every tile,
// matching the one-wave query-reuse topology used by Qwen's query16 kernel.
// Grid: [ceil(M/16), ceil(B/(16*tiles))]. Block: [32]. No LDS.
__launch_bounds__(32)
extern "C" __global__ void E8_PREFILL_WIDE_KERNEL_NAME(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int B
) {
    const int tid = threadIdx.x;
    const int lane = tid & 15;
    const int half = tid >> 4;
    const int row_start = blockIdx.x * 16;
    const int batch_start =
        blockIdx.y * (16 * E8_PREFILL_WIDE_TILES);
    if (row_start >= M || batch_start >= B) return;

    const int safe_row =
        (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int n_blocks = K / 32;
    const int scale_padded = (n_blocks + 15) & ~15;
    const long long row_bytes =
        16LL + (long long)scale_padded + (long long)n_blocks * 16;
    const char* row_ptr = A + (long long)safe_row * row_bytes;
    const float row_scale =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)row_ptr);
    const unsigned char* scale_base =
        reinterpret_cast<const unsigned char*>(row_ptr + 16);
    const char* cw_base = row_ptr + 16 + scale_padded;

    float8_t acc[E8_PREFILL_WIDE_TILES];
    #pragma unroll
    for (int tile = 0; tile < E8_PREFILL_WIDE_TILES; tile++) {
        acc[tile] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
    }
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int k_off = 0; k_off < K; k_off += 16) {
        const int block = k_off >> 5;
        const int codeword = (k_off & 31) >> 3;
        const float scale =
            row_scale * e8_wide_scale_to_f32(scale_base[block]) * 0.88f;
        const unsigned int cw0 = *reinterpret_cast<const unsigned int*>(
            cw_base + (long long)block * 16 + codeword * 4);
        const unsigned int cw1 = *reinterpret_cast<const unsigned int*>(
            cw_base + (long long)block * 16 + (codeword + 1) * 4);
        float p0[8], p1[8];
        e8_wide_decode(cw0, p0);
        e8_wide_decode(cw1, p1);

        half16_t a;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a[i] = (_Float16)(p0[i] * scale);
            a[8 + i] = (_Float16)(p1[i] * scale);
        }

        #pragma unroll
        for (int tile = 0; tile < E8_PREFILL_WIDE_TILES; tile++) {
            const int batch_row = batch_start + tile * 16 + lane;
            const _Float16* x_row = batch_row < B
                ? X + (long long)batch_row * K
                : nullptr;
            const half16_t b = x_row != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row + k_off)
                : zero_b;
            acc[tile] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[tile]);
        }
    }

    #pragma unroll
    for (int tile = 0; tile < E8_PREFILL_WIDE_TILES; tile++) {
        const int out_col = batch_start + tile * 16 + lane;
        if (out_col < B) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 2 * j + half;
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[tile][j];
                }
            }
        }
    }
}
