// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — native gfx942 wave64x4 grouped MFP4-E8 SoA projection candidate.

#include <hip/hip_runtime.h>

// This source is deliberately exact-shape and CDNA-owned.  It does not share
// an implementation, launch geometry, or wave decomposition with an RDNA
// kernel.  Four native wave64s compute four independent rows per workgroup.
//
// Production DeepSeek4 O-LoRA shape:
//   A[8, 1024, 4096] @ x[8, 4096] -> y[8, 1024]
//
// qt35 row layout at K=4096:
//   16-byte header | 128 E4M3 scales | 128 x 16-byte E8 codeword blocks

namespace {

constexpr int kGroups = 8;
constexpr int kRows = 1024;
constexpr int kCols = 4096;
constexpr int kBlocks = kCols / 32;
constexpr int kScaleBytes = 128;
constexpr int kRowBytes = 16 + kScaleBytes + kBlocks * 16;

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

} // namespace

extern "C" __global__ __launch_bounds__(256, 2)
void gemv_mfp4g32_e8_soa_grouped_wave64x4_gfx942_candidate(
    const char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int group = (int)blockIdx.y;
    const int row = (int)blockIdx.x * 4 + wave;

    // The host launch is exact: grid=(256, 8), block=(256, 1, 1).
    // Keeping these guards documents the contract without introducing a
    // partial-wave return before the wave collectives.
    if (group >= kGroups || row >= kRows) return;

    const long long global_row = (long long)group * kRows + row;
    const char* row_base = weights + global_row * kRowBytes;
    const char* scale_base = row_base + 16;
    const char* codeword_base = scale_base + kScaleBytes;
    const float* x_group = x + (long long)group * kCols;

    float row_scale = 0.0f;
    if (lane == 0) {
        row_scale = (float)__builtin_bit_cast(
            _Float16, *(const unsigned short*)row_base);
    }
    row_scale = __shfl(row_scale, 0, 64);

    // A 256-value qt35 group contains eight 32-value blocks.  Eight lanes
    // form a block subgroup: four lane-pairs consume the four codewords, and
    // each lane consumes one contiguous float4.  Pair broadcasts keep the
    // full-wave mapping at the original 128 codeword bytes per row/group.
    const int block_in_group = lane >> 3;
    const int codeword_slot = (lane >> 1) & 3;
    const int half = lane & 1;
    const int block_leader = lane & ~7;
    const int pair_leader = lane & ~1;

    float acc = 0.0f;
    #pragma unroll
    for (int quant_group = 0; quant_group < 16; ++quant_group) {
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

        const int x_offset = quant_group * 256 + lane * 4;
        const float4 xv = *(const float4*)(x_group + x_offset);
        acc += block_scale * dot_e8_half(codeword, half, xv);
    }

    // Native wave64 reduction: every lane contributes to exactly one row.
    for (int offset = 32; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 64);
    }
    if (lane == 0) y[global_row] = acc;
}
