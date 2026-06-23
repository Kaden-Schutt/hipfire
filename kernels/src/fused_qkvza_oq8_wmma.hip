// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ+ (W8A8 / int8-expanded) fused QKVZA: four grouped-iu8 GEMMs (in_proj_qkv,
// in_proj_z, in_proj_beta, in_proj_alpha for the DeltaNet linear-attention block)
// in ONE launch over a shared int8 activation. The int8 generalization of
// fused_qkvza_oq4_wmma — inner body identical to gemm_oq8_grouped_wmma, blockIdx.x
// tile-range demuxes which projection each block computes. ONE launch + ONE
// activation quantization replaces the per-projection gemv loop (the decode
// launch-count win on gfx1103, which is per-token launch/sync bound). wave32,
// register-tiled, ZERO LDS.
//
//   W_* : [m_*, K] int8 ++ [m_*, K/G] f32 scales (combined buffer each)
//   Xq  : [B, K] int8 (shared) ; Xs : [B, K/G] f32 (shared)
//   Y_* : [B, m_*] f32
//
// grid.x = Σ ceil(m_*/16), grid.y = ceil(B/16), block=32.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void fused_qkvza_oq8_wmma(
    const int8_t* __restrict__ W_qkv,
    const int8_t* __restrict__ W_z,
    const int8_t* __restrict__ W_beta,
    const int8_t* __restrict__ W_alpha,
    const int8_t* __restrict__ Xq,
    const float*  __restrict__ Xs,
    float*        __restrict__ Y_qkv,
    float*        __restrict__ Y_z,
    float*        __restrict__ Y_beta,
    float*        __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int B, int group
) {
    const int t_qkv = (qkv_m + 15) / 16;
    const int t_z   = (z_m + 15) / 16;
    const int t_beta = (beta_m + 15) / 16;
    const int8_t* W;
    float* Y;
    int M, row_start;
    int gx = blockIdx.x;
    if (gx < t_qkv) {
        W = W_qkv; Y = Y_qkv; M = qkv_m; row_start = gx * 16;
    } else if (gx < t_qkv + t_z) {
        W = W_z; Y = Y_z; M = z_m; row_start = (gx - t_qkv) * 16;
    } else if (gx < t_qkv + t_z + t_beta) {
        W = W_beta; Y = Y_beta; M = beta_m; row_start = (gx - t_qkv - t_z) * 16;
    } else {
        W = W_alpha; Y = Y_alpha; M = alpha_m;
        row_start = (gx - t_qkv - t_z - t_beta) * 16;
    }
    const int batch_start = blockIdx.y * 16;
    if (row_start >= M || batch_start >= B) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    const int a_row = row_start + lane;
    const int x_row = batch_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int safe_x = (x_row < B) ? x_row : (B - 1);
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;

    const float* Ws = (const float*)(W + (long long)M * K);
    const int8_t* w_row = W + (long long)safe_a * K;
    const int8_t* x_prow = Xq + (long long)safe_x * K;

    float8_t facc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < n_groups; ++g) {
        int32x8_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
        const int g_off = g * group;
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            const int32x4_t a = *(const int32x4_t*)(w_row + g_off + kt * 16);
            const int32x4_t b = *(const int32x4_t*)(x_prow + g_off + kt * 16);
            iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, iacc, false);
        }
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
