// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus-Quant W4A8, register-tiled: 4-bit weight × 8-bit activation. gfx1103
// (RDNA3) wave32, ZERO LDS. The weight is stored packed signed-int4 (¼ the bf16
// footprint, ½ of oq8), unpacked to int8 in-register, and multiplied by the
// dynamic-int8 activation on v_wmma_i32_16x16x16_iu8 — so the compute matches
// the tiled oq8 kernel (int8 WMMA) while halving weight-memory traffic again.
// Keeps int8 activation precision (unlike W4A4), which matters for diffusion.
//
//   W : [M, K/2] packed signed int4  (byte = k_even | k_odd<<4, two's-comp)
//   Ws: [M, K/G] f32  per-group weight scales
//   X : [B, K]   signed int8          Xs: [B, K/G] f32  per-group act scales
//   Y : [B, M] f32
//   Y[b,m] = Σ_g sw[m,g]·sx[b,g] · Σ_{k∈g} qw[m,k]·qx[b,k]
// Requires K % group == 0 and group % 16 == 0. MB×NB 16×16 output subtiles per
// wave (independent int32 group accumulators → ILP, fragment reuse).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

// Unpack 16 packed signed-int4 (8 bytes) into 16 signed int8 packed as int32x4,
// preserving K order (byte b -> low nibble = even k, high nibble = odd k). Each
// nibble is sign-extended: v = (nib ^ 8) - 8.
static __device__ __forceinline__ int32x4_t unpack_i4x16_to_i8(const uint8_t* p) {
    const uint32_t lo = *(const uint32_t*)(p + 0); // bytes 0..3 -> int8 0..7
    const uint32_t hi = *(const uint32_t*)(p + 4); // bytes 4..7 -> int8 8..15
    int32x4_t out;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t src = (w < 2) ? lo : hi;
        const int byte0 = (w & 1) ? 2 : 0;         // two source bytes -> one int32 (4 int8)
        const uint32_t b0 = (src >> (byte0 * 8)) & 0xff;
        const uint32_t b1 = (src >> ((byte0 + 1) * 8)) & 0xff;
        const int s0 = (int)((b0 & 0xf) ^ 0x8) - 8; // even k
        const int s1 = (int)((b0 >> 4) ^ 0x8) - 8;  // odd  k
        const int s2 = (int)((b1 & 0xf) ^ 0x8) - 8;
        const int s3 = (int)((b1 >> 4) ^ 0x8) - 8;
        out[w] = (s0 & 0xff) | ((s1 & 0xff) << 8) | ((s2 & 0xff) << 16) | ((s3 & 0xff) << 24);
    }
    return out;
}

#define DEFINE_TILED_OQ4A8_GEMM(NAME, TILE_MB, TILE_NB, MINWAVES)               \
__launch_bounds__(32, MINWAVES)                                                \
extern "C" __global__ void NAME(                                               \
    const uint8_t* __restrict__ W, const float* __restrict__ Ws,               \
    const int8_t*  __restrict__ X, const float* __restrict__ Xs,               \
    float* __restrict__ Y,                                                      \
    int M, int K, int B, int group                                             \
) {                                                                            \
    const int tid  = threadIdx.x;                                             \
    const int lane = tid & 15;                                                \
    const int row_start   = blockIdx.x * (16 * TILE_MB);                      \
    const int batch_start = blockIdx.y * (16 * TILE_NB);                      \
    if (row_start >= M || batch_start >= B) return;                          \
    const int wstride = K / 2;                                                \
    const int n_groups = K / group;                                          \
    const int tiles_per_group = group / 16;                                  \
    const uint8_t* w_row[TILE_MB];                                            \
    const int8_t*  x_row[TILE_NB];                                            \
    _Pragma("unroll")                                                         \
    for (int mi = 0; mi < TILE_MB; ++mi) {                                    \
        int m = row_start + mi * 16 + lane;                                   \
        w_row[mi] = W + (long long)((m < M) ? m : (M - 1)) * wstride;         \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                    \
        int b = batch_start + ni * 16 + lane;                                 \
        x_row[ni] = X + (long long)((b < B) ? b : (B - 1)) * K;               \
    }                                                                         \
    float8_t facc[TILE_MB][TILE_NB];                                         \
    _Pragma("unroll")                                                         \
    for (int mi = 0; mi < TILE_MB; ++mi)                                      \
        _Pragma("unroll")                                                     \
        for (int ni = 0; ni < TILE_NB; ++ni)                                  \
            facc[mi][ni] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};                 \
    for (int g = 0; g < n_groups; ++g) {                                      \
        int32x8_t iacc[TILE_MB][TILE_NB];                                    \
        _Pragma("unroll")                                                     \
        for (int mi = 0; mi < TILE_MB; ++mi)                                  \
            _Pragma("unroll")                                                 \
            for (int ni = 0; ni < TILE_NB; ++ni)                              \
                iacc[mi][ni] = {0,0,0,0,0,0,0,0};                             \
        const int g_wbyte = g * (group / 2);                                 \
        const int g_off = g * group;                                         \
        for (int kt = 0; kt < tiles_per_group; ++kt) {                       \
            int32x4_t a_frag[TILE_MB];                                       \
            int32x4_t x_frag[TILE_NB];                                       \
            _Pragma("unroll")                                                 \
            for (int mi = 0; mi < TILE_MB; ++mi)                              \
                a_frag[mi] = unpack_i4x16_to_i8(w_row[mi] + g_wbyte + kt * 8);\
            _Pragma("unroll")                                                 \
            for (int ni = 0; ni < TILE_NB; ++ni)                              \
                x_frag[ni] = *(const int32x4_t*)(x_row[ni] + g_off + kt * 16);\
            _Pragma("unroll")                                                 \
            for (int mi = 0; mi < TILE_MB; ++mi)                              \
                _Pragma("unroll")                                             \
                for (int ni = 0; ni < TILE_NB; ++ni)                          \
                    iacc[mi][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(\
                        true, a_frag[mi], true, x_frag[ni], iacc[mi][ni], false); \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int ni = 0; ni < TILE_NB; ++ni) {                                \
            const int b_col = batch_start + ni * 16 + lane;                   \
            const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f; \
            _Pragma("unroll")                                                 \
            for (int mi = 0; mi < TILE_MB; ++mi) {                            \
                _Pragma("unroll")                                             \
                for (int j = 0; j < 8; ++j) {                                 \
                    const int m_row = row_start + mi * 16 + 2 * j + (tid >> 4);\
                    const float sw = (m_row < M) ? Ws[(long long)m_row * n_groups + g] : 0.f; \
                    facc[mi][ni][j] += (float)iacc[mi][ni][j] * sw * sx;      \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                    \
        const int out_col = batch_start + ni * 16 + lane;                     \
        if (out_col >= B) continue;                                           \
        _Pragma("unroll")                                                     \
        for (int mi = 0; mi < TILE_MB; ++mi) {                                \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 8; ++j) {                                     \
                const int out_row = row_start + mi * 16 + 2 * j + (tid >> 4); \
                if (out_row < M)                                             \
                    Y[(long long)out_col * M + out_row] = facc[mi][ni][j];    \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_TILED_OQ4A8_GEMM(gemm_oq4a8_tiled_wmma_2x2, 2, 2, 4)
DEFINE_TILED_OQ4A8_GEMM(gemm_oq4a8_tiled_wmma_4x2, 4, 2, 2)
DEFINE_TILED_OQ4A8_GEMM(gemm_oq4a8_tiled_wmma_2x4, 2, 4, 2)
