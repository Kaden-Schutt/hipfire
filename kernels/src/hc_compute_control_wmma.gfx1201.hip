// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — native gfx1201 WMMA Hyper-Connections control projection.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#ifndef HC_WMMA_BATCH_TILES
#define HC_WMMA_BATCH_TILES 1
#endif

#ifndef HC_WMMA_KERNEL_NAME
#define HC_WMMA_KERNEL_NAME hc_compute_control_wmma_b1_gfx1201
#endif

// W is [M,K] F16, X is [N,K] F32, and Y is [N,M] F32. Each decoded X value
// is lowered to F16 at the gfx12 WMMA boundary. The token-specific RMS factor
// and control-row base are fused into the final store.
extern "C" __global__ __launch_bounds__(32)
void HC_WMMA_KERNEL_NAME(
    const _Float16* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ inv_rms,
    const _Float16* __restrict__ base,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * (16 * HC_WMMA_BATCH_TILES);
    const int tid = threadIdx.x;
    const int lane_m = tid & 15;
    const int k_group = tid >> 4;
    if (row_start >= M || col_start >= N) return;

    const int weight_row = row_start + lane_m;
    const bool weight_valid = weight_row < M;
    float8_t acc[HC_WMMA_BATCH_TILES];
    #pragma unroll
    for (int tile = 0; tile < HC_WMMA_BATCH_TILES; ++tile)
        acc[tile] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};

    for (int k0 = 0; k0 < K; k0 += 16) {
        half8_t a;
        if (weight_valid) {
            const _Float16* src = W + (size_t)weight_row * K + k0 + k_group * 8;
            a = *reinterpret_cast<const half8_t*>(src);
        } else {
            a = (half8_t){0, 0, 0, 0, 0, 0, 0, 0};
        }

        #pragma unroll
        for (int tile = 0; tile < HC_WMMA_BATCH_TILES; ++tile) {
            const int token = col_start + tile * 16 + lane_m;
            half8_t b;
            if (token < N) {
                const float* src = X + (size_t)token * K + k0 + k_group * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) b[j] = (_Float16)src[j];
            } else {
                b = (half8_t){0, 0, 0, 0, 0, 0, 0, 0};
            }
            acc[tile] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a, b, acc[tile]);
        }
    }

    #pragma unroll
    for (int tile = 0; tile < HC_WMMA_BATCH_TILES; ++tile) {
        const int token = col_start + tile * 16 + lane_m;
        if (token < N) {
            const float scale = inv_rms[token];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = row_start + 8 * k_group + j;
                if (row < M) {
                    Y[(size_t)token * M + row] =
                        acc[tile][j] * scale + (float)base[row];
                }
            }
        }
    }
}
