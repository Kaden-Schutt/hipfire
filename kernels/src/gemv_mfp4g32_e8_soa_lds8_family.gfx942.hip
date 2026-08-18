// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — native gfx942 qt35 LDS-X projection family.

#include <hip/hip_runtime.h>

// CDNA3-only mechanism candidate. Eight native wave64 workers compute eight
// output rows while loading each 256-value X tile once into LDS. No RDNA
// source, include, launch geometry, or half-wave reduction is inherited.
namespace {

__device__ __forceinline__ float e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(
        float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ unsigned e8_element(
    unsigned codeword,
    int element
) {
    if (element < 7) return (codeword >> (4 * element)) & 0xFu;
    unsigned sum_low = 0u;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        sum_low += (codeword >> (4 * i)) & 0xFu;
    }
    const unsigned high = (codeword >> 28) & 0x7u;
    const unsigned even_part = high << 1;
    return even_part | ((sum_low + even_part) & 1u);
}

__device__ __forceinline__ float dot_e8_half(
    unsigned codeword,
    int half,
    float4 xv
) {
    const int first = half << 2;
    const float coset = (codeword >> 31) ? 0.5f : 0.0f;
    const float p0 = (float)((int)e8_element(codeword, first + 0) - 7) + coset;
    const float p1 = (float)((int)e8_element(codeword, first + 1) - 7) + coset;
    const float p2 = (float)((int)e8_element(codeword, first + 2) - 7) + coset;
    const float p3 = (float)((int)e8_element(codeword, first + 3) - 7) + coset;
    float sum = p0 * xv.x;
    sum += p1 * xv.y;
    sum += p2 * xv.z;
    sum += p3 * xv.w;
    return sum;
}

template <int kCols, int kScaleBytes, int kRowBytes>
__device__ __forceinline__ void project_row_tile(
    const char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* x_tile
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int row = (int)blockIdx.x * 8 + wave;
    const char* row_base = weights + (long long)row * kRowBytes;
    const char* scale_base = row_base + 16;
    const char* codeword_base = scale_base + kScaleBytes;

    float row_scale = 0.0f;
    if (lane == 0) {
        row_scale = (float)__builtin_bit_cast(
            _Float16, *(const unsigned short*)row_base);
    }
    row_scale = __shfl(row_scale, 0, 64);

    const int block_in_group = lane >> 3;
    const int codeword_slot = (lane >> 1) & 3;
    const int half = lane & 1;
    const int block_leader = lane & ~7;
    const int pair_leader = lane & ~1;
    constexpr int kQuantGroups = kCols / 256;

    float acc = 0.0f;
    #pragma unroll
    for (int quant_group = 0; quant_group < kQuantGroups; ++quant_group) {
        if (tid < 256) x_tile[tid] = x[quant_group * 256 + tid];
        __syncthreads();

        const int block = quant_group * 8 + block_in_group;
        float block_scale = 0.0f;
        if ((lane & 7) == 0) {
            block_scale = row_scale
                * e4m3_scale_to_f32(
                    *(const unsigned char*)(scale_base + block))
                * 0.88f;
        }
        block_scale = __shfl(block_scale, block_leader, 64);

        unsigned codeword = 0u;
        if ((lane & 1) == 0) {
            codeword = *(const unsigned int*)(
                codeword_base + (long long)block * 16 + codeword_slot * 4);
        }
        codeword = __shfl(codeword, pair_leader, 64);
        const float4 xv = *(const float4*)(x_tile + lane * 4);
        acc += block_scale * dot_e8_half(codeword, half, xv);
        __syncthreads();
    }

    for (int offset = 32; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 64);
    }
    if (lane == 0) y[row] = acc;
}

} // namespace

#define DEFINE_LDS8_KERNEL(symbol, cols, scales, rowbytes)                    \
extern "C" __global__ __launch_bounds__(512, 1)                              \
void symbol(                                                                  \
    const char* __restrict__ weights,                                         \
    const float* __restrict__ x,                                              \
    float* __restrict__ y                                                     \
) {                                                                           \
    __shared__ __align__(16) float x_tile[256];                               \
    project_row_tile<cols, scales, rowbytes>(weights, x, y, x_tile);          \
}

DEFINE_LDS8_KERNEL(gemv_mfp4g32_e8_soa_lds8_k1024_gfx942_candidate,
                   1024, 32, 560)
DEFINE_LDS8_KERNEL(gemv_mfp4g32_e8_soa_lds8_k2048_gfx942_candidate,
                   2048, 64, 1104)
DEFINE_LDS8_KERNEL(gemv_mfp4g32_e8_soa_lds8_k4096_gfx942_candidate,
                   4096, 128, 2192)
DEFINE_LDS8_KERNEL(gemv_mfp4g32_e8_soa_lds8_k8192_gfx942_candidate,
                   8192, 256, 4368)

// O-LoRA variant: blockIdx.y selects one of eight group-local weight/X/Y
// regions. Each block still shares only the X belonging to its own group.
extern "C" __global__ __launch_bounds__(512, 1)
void gemv_mfp4g32_e8_soa_grouped_lds8_gfx942_candidate(
    const char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    constexpr int kRows = 1024;
    constexpr int kCols = 4096;
    constexpr int kRowBytes = 2192;
    const int group = (int)blockIdx.y;
    const char* group_weights = weights + (long long)group * kRows * kRowBytes;
    const float* group_x = x + (long long)group * kCols;
    float* group_y = y + (long long)group * kRows;
    __shared__ __align__(16) float x_tile[256];
    project_row_tile<kCols, 128, kRowBytes>(
        group_weights, group_x, group_y, x_tile);
}

#undef DEFINE_LDS8_KERNEL
