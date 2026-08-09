// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — cooperative dense MFP4-E8-SoA WMMA GEMM for gfx1151 prefill.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Four waves consume one decoded 16-row x 128-K weight slab. Each wave keeps
// the production B4 schedule (64 prompt rows), so a workgroup covers 256 rows
// while decoding the weight slab once instead of four times. The 128-K slab
// keeps the two workgroup barriers outside the WMMA K=16 inner loop.
constexpr int E8_COOP_WAVES = 4;
constexpr int E8_COOP_K = 128;
constexpr int E8_COOP_BATCH_PER_WAVE = 64;
constexpr int E8_COOP_BATCH = E8_COOP_WAVES * E8_COOP_BATCH_PER_WAVE;

__device__ __forceinline__ float e8_coop_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned int mant = (unsigned int)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 =
        __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ void e8_coop_decode_store(
    unsigned int idx,
    float scale,
    _Float16* __restrict__ dst
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
        const float p = coset ? c + 0.5f : c;
        dst[i] = (_Float16)(p * scale);
    }
}

// A is E8-SoA [M,K], X is f16 [B,K], Y is f32 [B,M].
// Grid: [ceil(M/16), ceil(B/256)]. Block: [128]. Static LDS: 4 KiB.
__launch_bounds__(128, 2)
extern "C" __global__ void gemm_mfp4g32_e8_soa_wmma_coop4_gfx1151(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int B
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int wave_tid = tid & 31;
    const int lane = wave_tid & 15;
    const int half = wave_tid >> 4;
    const int row_start = blockIdx.x * 16;
    const int batch_start =
        blockIdx.y * E8_COOP_BATCH + wave * E8_COOP_BATCH_PER_WAVE;
    if (row_start >= M) return;

    __shared__ _Float16 a_tile[16 * E8_COOP_K];

    const int n_blocks = K / 32;
    const int scale_padded = (n_blocks + 15) & ~15;
    const long long row_bytes =
        16LL + (long long)scale_padded + (long long)n_blocks * 16;

    const int batch_row0 = batch_start + lane;
    const int batch_row1 = batch_row0 + 16;
    const int batch_row2 = batch_row0 + 32;
    const int batch_row3 = batch_row0 + 48;
    const _Float16* x_row0 =
        batch_row0 < B ? X + (long long)batch_row0 * K : nullptr;
    const _Float16* x_row1 =
        batch_row1 < B ? X + (long long)batch_row1 * K : nullptr;
    const _Float16* x_row2 =
        batch_row2 < B ? X + (long long)batch_row2 * K : nullptr;
    const _Float16* x_row3 =
        batch_row3 < B ? X + (long long)batch_row3 * K : nullptr;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int k_base = 0; k_base < K; k_base += E8_COOP_K) {
        // There are 16 rows x 16 codewords per 128-K slab. The 128
        // workitems decode two unique codewords apiece into a canonical LDS
        // tile; the duplicate half-wave and four-wave decodes disappear.
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int code_id = tid + pass * 128;
            const int row = code_id >> 4;
            const int code_in_slab = code_id & 15;
            const int safe_row =
                (row_start + row < M) ? (row_start + row) : (M - 1);
            const char* row_ptr = A + (long long)safe_row * row_bytes;
            const float row_scale = (float)__builtin_bit_cast(
                _Float16, *(const unsigned short*)row_ptr);
            const unsigned char* scale_base =
                reinterpret_cast<const unsigned char*>(row_ptr + 16);
            const char* cw_base = row_ptr + 16 + scale_padded;
            const int global_codeword = (k_base >> 3) + code_in_slab;
            const int block = global_codeword >> 2;
            const int codeword = global_codeword & 3;
            const float scale =
                row_scale * e8_coop_scale_to_f32(scale_base[block]) * 0.88f;
            const unsigned int cw = *reinterpret_cast<const unsigned int*>(
                cw_base + (long long)block * 16 + codeword * 4);
            e8_coop_decode_store(
                cw,
                scale,
                a_tile + row * E8_COOP_K + code_in_slab * 8);
        }
        __syncthreads();

        #pragma unroll
        for (int k_local = 0; k_local < E8_COOP_K; k_local += 16) {
            const int k_off = k_base + k_local;
            const half16_t a = *reinterpret_cast<const half16_t*>(
                a_tile + lane * E8_COOP_K + k_local);
            const half16_t b0 = x_row0 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row0 + k_off)
                : zero_b;
            const half16_t b1 = x_row1 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row1 + k_off)
                : zero_b;
            const half16_t b2 = x_row2 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row2 + k_off)
                : zero_b;
            const half16_t b3 = x_row3 != nullptr
                ? *reinterpret_cast<const half16_t*>(x_row3 + k_off)
                : zero_b;
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc1);
            acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc2);
            acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc3);
        }
        __syncthreads();
    }

    const int out_col0 = batch_start + lane;
    const int out_col1 = out_col0 + 16;
    const int out_col2 = out_col0 + 32;
    const int out_col3 = out_col0 + 48;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 2 * j + half;
        if (out_row < M) {
            if (out_col0 < B) Y[(long long)out_col0 * M + out_row] = acc0[j];
            if (out_col1 < B) Y[(long long)out_col1 * M + out_row] = acc1[j];
            if (out_col2 < B) Y[(long long)out_col2 * M + out_row] = acc2[j];
            if (out_col3 < B) Y[(long long)out_col3 * M + out_row] = acc3[j];
        }
    }
}
