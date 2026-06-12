// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx1151 HFQ4-G256 x signed-Q4 activation IU4-WMMA prototype.
//
// This validates the affine correction path needed before MQ4/HFQ4 can try
// native IU4 in production. It is intentionally not routed into model code.
//
// Weight block: existing HFQ4-G256 layout [f32 scale][f32 zero][128B qs].
// Activation scratch:
//   X_qs  [N, K/2] packed signed 4-bit two's-complement values
//   X_d   [N, K/32] symmetric Q4 scale
//   X_sum [N, K/32] sum(qx_s4) for the same 32-K subblock
//
// Reinterpret weight nibbles as signed by qs = qu - 8:
//   w = scale * qu + zero = scale * qs + (zero + 8 * scale)
//   x ~= dx * qx
//   dot(w, x) ~= scale * dx * dot(qs, qx)
//              + (zero + 8 * scale) * dx * sum(qx)

#if defined(__gfx1151__)

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

static __device__ __forceinline__ uint32_t shift_u4_to_s4(uint32_t packed) {
    uint32_t out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t q = (packed >> (4 * i)) & 0x0fu;
        out |= ((q + 8u) & 0x0fu) << (4 * i);
    }
    return out;
}

static __device__ __forceinline__ int32x2_t load_s4x16_from_hfq4(const uint8_t* p) {
    const uint32_t lo = shift_u4_to_s4(*(const uint32_t*)(p + 0));
    const uint32_t hi = shift_u4_to_s4(*(const uint32_t*)(p + 4));
    return (int32x2_t){(int32_t)lo, (int32_t)hi};
}

static __device__ __forceinline__ int32x2_t load_s4x16(const uint8_t* p) {
    const uint32_t lo = *(const uint32_t*)(p + 0);
    const uint32_t hi = *(const uint32_t*)(p + 4);
    return (int32x2_t){(int32_t)lo, (int32_t)hi};
}

extern "C" __launch_bounds__(32, 4)
__global__ void gemm_hfq4g256_s4_mmq_gfx1151(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ X_qs,
    const float* __restrict__ X_d,
    const int32_t* __restrict__ X_sum,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int lane = tid & 15;
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    if (row_start >= M || col_start >= N) return;

    const int col = col_start + lane;
    const int safe_col = (col < N) ? col : (N - 1);
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const int x_stride = K / 2;
    const int x_meta_stride = K / 32;
    const uint8_t* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; ++g) {
        const uint8_t* gp = row_base + g * 136;
        const float sc_self = __builtin_bit_cast(float, *(const uint32_t*)(gp + 0));
        const float zp_self = __builtin_bit_cast(float, *(const uint32_t*)(gp + 4));
        const float bias_self = zp_self + 8.0f * sc_self;

        float sc_row[8];
        float bias_row[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            bias_row[j] = __shfl(bias_self, src_lane);
        }

        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int k_off = sb * 32;
            const int meta_idx = g * 8 + sb;
            const float dx = (col < N) ? X_d[(long long)safe_col * x_meta_stride + meta_idx] : 0.0f;
            const float sum_qx = (col < N) ? (float)X_sum[(long long)safe_col * x_meta_stride + meta_idx] : 0.0f;

            const int32x2_t a0 = load_s4x16_from_hfq4(gp + 8 + (k_off + 0) / 2);
            const int32x2_t a1 = load_s4x16_from_hfq4(gp + 8 + (k_off + 16) / 2);
            const uint8_t* xp = X_qs + (long long)safe_col * x_stride + k_off / 2;
            const int32x2_t b0 = (col < N) ? load_s4x16(xp + 0) : (int32x2_t){0, 0};
            const int32x2_t b1 = (col < N) ? load_s4x16(xp + 8) : (int32x2_t){0, 0};

            cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a0, true, b0, cacc, true);
            cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a1, true, b1, cacc, true);

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] += sc_row[j] * dx * (float)cacc[j] + bias_row[j] * dx * sum_qx;
            }
        }
    }

    if (col < N) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_hfq4g256_s4_mmq_gfx1151(
    const uint8_t*, const uint8_t*, const float*, const int32_t*, float*, int, int, int
) {}

#endif
