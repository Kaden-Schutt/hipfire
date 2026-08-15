// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — grouped mfp4-E8-SoA WMMA GEMM for gfx12 prefill.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 wave32 WMMA consumes eight FP16 values per lane. The two 16-lane
// groups own the lower and upper K8 halves of each K16 tile respectively.
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ float e8_soa_grouped_gfx12_scale_to_f32(
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

__device__ __forceinline__ void e8_soa_grouped_gfx12_decode(
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
// grid.z selects a group while preserving the batch-major interleaved X/Y
// layout used by DeepSeek4 O-LoRA.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mfp4g32_e8_soa_grouped_wmma_gfx1201(
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
    const int k_group = tid >> 4;
    const int group = blockIdx.z;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    if (group >= G || row_start >= M || batch_start >= B) return;

    const int safe_row =
        (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int batch_row = batch_start + lane;
    const _Float16* x_row =
        batch_row < B
            ? X + ((long long)batch_row * G + group) * K
            : nullptr;

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

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k_off = 0; k_off < K; k_off += 16) {
        const int block = k_off >> 5;
        const int codeword_base = (k_off & 31) >> 3;
        const int codeword = codeword_base + k_group;
        const float scale =
            row_scale *
            e8_soa_grouped_gfx12_scale_to_f32(scale_base[block]) *
            0.88f;
        const unsigned int cw = *reinterpret_cast<const unsigned int*>(
            cw_base + (long long)block * 16 + codeword * 4);
        float p[8];
        e8_soa_grouped_gfx12_decode(cw, p);

        half8_t a;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a[i] = (_Float16)(p[i] * scale);
        }
        const half8_t b =
            x_row != nullptr
                ? *reinterpret_cast<const half8_t*>(
                    x_row + k_off + k_group * 8)
                : zero_b;
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }

    const int out_row_base = row_start + 8 * k_group;
    if (batch_row < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = out_row_base + j;
            if (out_row < M) {
                const long long out_idx =
                    ((long long)batch_row * G + group) * M + out_row;
                Y[out_idx] = acc[j];
            }
        }
    }
}
