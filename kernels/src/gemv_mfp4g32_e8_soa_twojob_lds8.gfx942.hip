// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — native gfx942 two-job qt35 projection scheduler with LDS X reuse.

#include <hip/hip_runtime.h>

// Exact DeepSeek V4 shared-expert gate/up shape.  This is a CDNA3-native
// candidate: eight full wave64 workers compute eight rows while sharing each
// 256-value input tile once through LDS.  It does not include, call, or mirror
// an RDNA kernel.
namespace {

constexpr int kJobs = 2;
constexpr int kRows = 2048;
constexpr int kCols = 4096;
constexpr int kQuantGroups = kCols / 256;
constexpr int kBlocks = kCols / 32;
constexpr int kScaleBytes = 128;
constexpr int kRowBytes = 16 + kScaleBytes + kBlocks * 16;
constexpr int kWavesPerBlock = 8;
constexpr int kRowTiles = kRows / kWavesPerBlock;

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

extern "C" __global__ __launch_bounds__(512, 1)
void gemv_mfp4g32_e8_soa_twojob_lds8_gfx942_candidate(
    const char* __restrict__ weights0,
    const char* __restrict__ weights1,
    const float* __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int job = (int)blockIdx.x / kRowTiles;
    const int row_tile = (int)blockIdx.x - job * kRowTiles;
    const int row = row_tile * kWavesPerBlock + wave;

    // The exact launch is grid=(512,1,1), block=(512,1,1).  Therefore every
    // block has one job and eight valid rows; no lane can return before the
    // cooperative barriers.
    const char* weights = job == 0 ? weights0 : weights1;
    float* y = job == 0 ? y0 : y1;
    const char* row_base = weights + (long long)row * kRowBytes;
    const char* scale_base = row_base + 16;
    const char* codeword_base = scale_base + kScaleBytes;

    __shared__ __align__(16) float x_tile[256];

    float row_scale = 0.0f;
    if (lane == 0) {
        row_scale = (float)__builtin_bit_cast(
            _Float16, *(const unsigned short*)row_base);
    }
    row_scale = __shfl(row_scale, 0, 64);

    // Eight lanes cover one 32-value qt35 block.  Each lane-pair shares one
    // codeword; the two lanes consume its low/high float4.  All eight waves
    // reuse the same LDS X tile, so a block issues 256 global X loads rather
    // than 8 * 256.
    const int block_in_group = lane >> 3;
    const int codeword_slot = (lane >> 1) & 3;
    const int half = lane & 1;
    const int block_leader = lane & ~7;
    const int pair_leader = lane & ~1;

    float acc = 0.0f;
    #pragma unroll
    for (int quant_group = 0; quant_group < kQuantGroups; ++quant_group) {
        if (tid < 256) {
            x_tile[tid] = x[quant_group * 256 + tid];
        }
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

        // Prevent the first half of the workgroup from overwriting x_tile
        // before every wave has consumed the current quant group.
        __syncthreads();
    }

    for (int offset = 32; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 64);
    }
    if (lane == 0) y[row] = acc;
}
