// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W4A4 core: grouped signed-INT4 × signed-INT4 GEMM with per-group
// scale rescale folded into the epilogue. gfx1103 (RDNA3) wave32, register-tiled,
// ZERO LDS. Productionizes the host-tiled E5 recipe into one kernel.
//
//   W : [M, K/2] packed signed int4  (byte = k_even | k_odd<<4, two's-comp)
//   Ws: [M, K/G] f32  per-group weight scales (G = group size)
//   X : [B, K/2] packed signed int4
//   Xs: [B, K/G] f32  per-group activation scales
//   Y : [B, M] f32
//
// Y[b,m] = Σ_g scale_w[m,g]·scale_x[b,g] · Σ_{k∈g} q_w[m,k]·q_x[b,k]
// The inner int4·int4 dot runs on v_wmma_i32_16x16x16_iu4 (exact i32); each
// 256-group's i32 partial is rescaled to f32 and accumulated. Requires
// K % GROUP == 0 and GROUP % 16 == 0.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

static __device__ __forceinline__ int32x2_t load_u4x16(const uint8_t* p) {
    return (int32x2_t){(int32_t)*(const uint32_t*)(p + 0),
                       (int32_t)*(const uint32_t*)(p + 4)};
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_oq4_grouped_wmma(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group
) {
    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    if (row_start >= M || batch_start >= B) return;

    const int a_row = row_start + lane;
    const int x_row = batch_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int safe_x = (x_row < B) ? x_row : (B - 1);
    const int stride = K / 2;          // bytes per packed row
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;

    const uint8_t* w_row = W + (long long)safe_a * stride;
    const uint8_t* x_prow = X + (long long)safe_x * stride;

    float8_t facc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < n_groups; ++g) {
        int32x8_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
        const int g_byte = g * (group / 2);
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            const int32x2_t a = load_u4x16(w_row + g_byte + kt * 8);
            const int32x2_t b = load_u4x16(x_prow + g_byte + kt * 8);
            iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, iacc, false);
        }
        // Per-group rescale: activation scale is per (b-col); weight scale is
        // per (m-row), and each lane owns 8 different m-rows (acc[j]).
        const int b_col = batch_start + lane;
        const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + (tid >> 4);
            const float sw = (m_row < M) ? Ws[(long long)m_row * n_groups + g] : 0.f;
            facc[j] += (float)iacc[j] * sw * sx;
        }
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = facc[j];
            }
        }
    }
}
