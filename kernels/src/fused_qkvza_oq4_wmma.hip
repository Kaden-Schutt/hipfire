// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W4A4 fused QKVZA: four grouped-iu4 GEMMs (in_proj_qkv, in_proj_z,
// in_proj_beta, in_proj_alpha for the DeltaNet linear-attention block) in ONE
// launch over a shared int4 activation. gfx1103 wave32, register-tiled, ZERO
// LDS. Inner body identical to gemm_oq4_grouped_wmma; blockIdx.x tile-range
// demuxes which of the 4 projections each block computes, so the shared Xq/Xs is
// read once and only one launch + one activation quantization is needed.
//
//   W_* : [m_*, K/2] nibbles ++ [m_*, K/G] f32 scales (combined buffer each)
//   Xq  : [B, K/2] packed signed int4 (shared) ; Xs : [B, K/G] f32 (shared)
//   Y_* : [B, m_*] f32
//
// grid.x = Σ ceil(m_*/16)  (whole tiles per projection — no straddle),
// grid.y = ceil(B/16), block=32.

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
extern "C" __global__ void fused_qkvza_oq4_wmma(
    const uint8_t* __restrict__ W_qkv,
    const uint8_t* __restrict__ W_z,
    const uint8_t* __restrict__ W_beta,
    const uint8_t* __restrict__ W_alpha,
    const uint8_t* __restrict__ Xq,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int B, int group
) {
    const int t_qkv = (qkv_m + 15) / 16;
    const int t_z   = (z_m + 15) / 16;
    const int t_beta = (beta_m + 15) / 16;
    // Demux by cumulative tile ranges.
    const uint8_t* W;
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
    const int stride = K / 2;
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;

    const float* Ws = (const float*)(W + (long long)M * stride);
    const uint8_t* w_row = W + (long long)safe_a * stride;
    const uint8_t* x_prow = Xq + (long long)safe_x * stride;

    float8_t facc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < n_groups; ++g) {
        int32x8_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
        const int g_byte = g * (group / 2);
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            const int32x2_t a = load_u4x16(w_row + g_byte + kt * 8);
            const int32x2_t b = load_u4x16(x_prow + g_byte + kt * 8);
            iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, iacc, false);
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
