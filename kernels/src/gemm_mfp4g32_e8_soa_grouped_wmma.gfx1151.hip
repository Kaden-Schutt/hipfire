// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — grouped mfp4-E8-SoA WMMA GEMM for gfx1151 prefill.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#ifndef E8_GROUPED_PREFILL_BATCH_TILES
#define E8_GROUPED_PREFILL_BATCH_TILES 1
#endif

#ifndef E8_GROUPED_PREFILL_KERNEL_NAME
#define E8_GROUPED_PREFILL_KERNEL_NAME \
    gemm_mfp4g32_e8_soa_grouped_wmma_gfx1151
#endif

__device__ __forceinline__ float e8_soa_grouped_scale_to_f32(
    unsigned char b
) {
    const int exp = (b >> 3) & 0xF;
    const unsigned int mant = (unsigned int)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 =
        __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ void e8_soa_grouped_decode(
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

// A is E8-SoA [G,M,K], X is f16 [B,G,K], Y is f32 [B,G,M].
// grid.z selects a group so each weight row remains contiguous while X/Y
// retain their interleaved batch-major group layout.
__launch_bounds__(32, 2)
extern "C" __global__ void E8_GROUPED_PREFILL_KERNEL_NAME(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int G,
    int M,
    int K,
    int B
) {
    const int tid = threadIdx.x;
    const int lane = tid & 15;
    const int group = blockIdx.z;
    const int row_start = blockIdx.x * 16;
    const int batch_start =
        blockIdx.y * (16 * E8_GROUPED_PREFILL_BATCH_TILES);
    if (group >= G || row_start >= M || batch_start >= B) return;

    const int safe_row =
        (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int batch_row0 = batch_start + lane;
    const _Float16* x_row0 =
        batch_row0 < B
            ? X + ((long long)batch_row0 * G + group) * K
            : nullptr;
#if E8_GROUPED_PREFILL_BATCH_TILES >= 2
    const int batch_row1 = batch_row0 + 16;
    const _Float16* x_row1 =
        batch_row1 < B
            ? X + ((long long)batch_row1 * G + group) * K
            : nullptr;
#endif
#if E8_GROUPED_PREFILL_BATCH_TILES >= 4
    const int batch_row2 = batch_row0 + 32;
    const int batch_row3 = batch_row0 + 48;
    const _Float16* x_row2 =
        batch_row2 < B
            ? X + ((long long)batch_row2 * G + group) * K
            : nullptr;
    const _Float16* x_row3 =
        batch_row3 < B
            ? X + ((long long)batch_row3 * G + group) * K
            : nullptr;
#endif

    const int n_blocks = K / 32;
    const int scale_padded = (n_blocks + 15) & ~15;
    const long long row_bytes =
        16LL + (long long)scale_padded + (long long)n_blocks * 16;
    const long long weight_row = (long long)group * M + safe_row;
    const char* row_ptr = A + weight_row * row_bytes;
    const float row_scale =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)row_ptr);
    const unsigned char* scale_base =
        reinterpret_cast<const unsigned char*>(row_ptr + 16);
    const char* cw_base = row_ptr + 16 + scale_padded;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
#if E8_GROUPED_PREFILL_BATCH_TILES >= 2
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
#if E8_GROUPED_PREFILL_BATCH_TILES >= 4
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int k_off = 0; k_off < K; k_off += 16) {
        const int block = k_off >> 5;
        const int codeword = (k_off & 31) >> 3;
        const float scale =
            row_scale *
            e8_soa_grouped_scale_to_f32(scale_base[block]) *
            0.88f;
        const unsigned int cw0 = *reinterpret_cast<const unsigned int*>(
            cw_base + (long long)block * 16 + codeword * 4);
        const unsigned int cw1 = *reinterpret_cast<const unsigned int*>(
            cw_base + (long long)block * 16 + (codeword + 1) * 4);
        float p0[8], p1[8];
        e8_soa_grouped_decode(cw0, p0);
        e8_soa_grouped_decode(cw1, p1);

        half16_t a;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a[i] = (_Float16)(p0[i] * scale);
            a[8 + i] = (_Float16)(p1[i] * scale);
        }
        const half16_t b0 =
            x_row0 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row0 + k_off)
                : zero_b;
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc0);
#if E8_GROUPED_PREFILL_BATCH_TILES >= 2
        const half16_t b1 =
            x_row1 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row1 + k_off)
                : zero_b;
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc1);
#endif
#if E8_GROUPED_PREFILL_BATCH_TILES >= 4
        const half16_t b2 =
            x_row2 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row2 + k_off)
                : zero_b;
        const half16_t b3 =
            x_row3 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row3 + k_off)
                : zero_b;
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc2);
        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc3);
#endif
    }

    const int out_col0 = batch_start + lane;
    if (out_col0 < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                const long long out_idx =
                    ((long long)out_col0 * G + group) * M + out_row;
                Y[out_idx] = acc0[j];
            }
        }
    }
#if E8_GROUPED_PREFILL_BATCH_TILES >= 2
    const int out_col1 = out_col0 + 16;
    if (out_col1 < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                const long long out_idx =
                    ((long long)out_col1 * G + group) * M + out_row;
                Y[out_idx] = acc1[j];
            }
        }
    }
#endif
#if E8_GROUPED_PREFILL_BATCH_TILES >= 4
    const int out_col2 = out_col0 + 32;
    const int out_col3 = out_col0 + 48;
    if (out_col2 < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                const long long out_idx =
                    ((long long)out_col2 * G + group) * M + out_row;
                Y[out_idx] = acc2[j];
            }
        }
    }
    if (out_col3 < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                const long long out_idx =
                    ((long long)out_col3 * G + group) * M + out_row;
                Y[out_idx] = acc3[j];
            }
        }
    }
#endif
}
