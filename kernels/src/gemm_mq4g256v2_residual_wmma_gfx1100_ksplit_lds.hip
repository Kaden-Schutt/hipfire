// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 split-K LDS residual for MQ4G256V2 (qt=44), DFlash verify tier.
// Arithmetic / header / interleaved-C authority:
//   gemm_mq4g256v2_residual_wmma.hip  (base gfx11 half16 K2 kernel)
//
// One workgroup owns a single 16x16 output tile
// (row_start = blockIdx.x*16, batch_start = blockIdx.y*16). KW waves own
// disjoint K-ranges of that SAME tile: wave w owns groups
// [w*G/KW, (w+1)*G/KW) where G = K/256. Each wave runs the base kernel's
// exact per-group loop (dual headers, DQ macro, 16 WMMA per group) on its
// K-range into its private float8 acc, then the accs reduce through LDS in
// fixed wave order and wave 0 applies the single Y += once.
//
// Grid:  [ceil(M/16), ceil(N/16), 1]
// Block: [32*KW, 1, 1]. Static LDS KW*1 KiB (red[KW][8][32] floats);
// dynamic shared_mem=0. Instantiated: ks2/ks4/ks8.
// Production policy (exact gfx1100, non-replay/non-capture, N<=16):
// small-batch residual shapes (DFlash verify). Launcher requires
// K % 256 == 0, G % KW == 0, G >= KW (else fall back to a smaller KW or base).
// Output: Y += sum with unique owner; gfx11 interleaved C mapping.
// No atomics, no global partial buffer, no early return before the barrier
// (M/N tails masked inside, as mw_lds does). Reduction order w=0..KW-1 is
// fixed, hence deterministic; fp32 association differs from the base kernel
// so bit-exactness is NOT claimed (parity gate: relL2 <= 1e-5).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define GEN_RESID_KSPLIT_LDS(KW)                                                              \
    extern "C" __launch_bounds__(32 * (KW), 1) __global__ void                                \
        gemm_mq4g256v2_residual_wmma_gfx1100_ks##KW##_lds(                                    \
            const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
            int M, int K, int N) {                                                            \
        const int tid = threadIdx.x;                                                          \
        const int lane = tid & 31;                                                            \
        const int wave_id = tid >> 5;                                                         \
        const int ml = lane & 15;                                                             \
        const int row_start = blockIdx.x * 16;                                                \
        const int batch_start = blockIdx.y * 16;                                              \
        /* Same tile for every wave; tails duplicate row M-1 / batch 0 (discarded). */        \
        const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);               \
        const int out_col = batch_start + ml;                                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                   \
        const int groups_per_row = K / 256;                                                   \
        const int groups_per_wave = groups_per_row / (KW);                                    \
        const int g_begin = wave_id * groups_per_wave;                                        \
        const int g_end = g_begin + groups_per_wave;                                          \
        const char* row_base = A + (long long)safe_row * groups_per_row * 136;                \
        const _Float16* x_base = X + (long long)safe_batch * K;                                \
                                                                                              \
        /* Base-kernel register footprint: one float8 acc + one half16 a/b per lane. */       \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                       \
                                                                                              \
        for (int g = g_begin; g < g_end; g++) {                                               \
            const char* gp = row_base + g * 136;                                              \
            const unsigned int hA = *(const unsigned int*)(gp);                               \
            const unsigned int hB = *(const unsigned int*)(gp + 4);                            \
            const _Float16 sc0 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));      \
            const _Float16 zp0 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));          \
            const _Float16 sc1 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));      \
            const _Float16 zp1 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));          \
            const _Float16* xg = x_base + g * 256;                                            \
                                                                                              \
            _Pragma("unroll")                                                                 \
            for (int kt = 0; kt < 16; kt++) {                                                 \
                const int k_off = kt * 16;                                                    \
                const _Float16 sc_h = (kt < 8) ? sc0 : sc1;                                   \
                const _Float16 zp_h = (kt < 8) ? zp0 : zp1;                                   \
                                                                                              \
                unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);                \
                unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);            \
                                                                                              \
                half16_t a_reg;                                                               \
                a_reg[0] = sc_h * (_Float16)(float)((pk0 >> 0) & 0xFu) + zp_h;                \
                a_reg[1] = sc_h * (_Float16)(float)((pk0 >> 4) & 0xFu) + zp_h;                \
                a_reg[2] = sc_h * (_Float16)(float)((pk0 >> 8) & 0xFu) + zp_h;                \
                a_reg[3] = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;               \
                a_reg[4] = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;               \
                a_reg[5] = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;               \
                a_reg[6] = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;               \
                a_reg[7] = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;               \
                a_reg[8] = sc_h * (_Float16)(float)((pk1 >> 0) & 0xFu) + zp_h;                \
                a_reg[9] = sc_h * (_Float16)(float)((pk1 >> 4) & 0xFu) + zp_h;                \
                a_reg[10] = sc_h * (_Float16)(float)((pk1 >> 8) & 0xFu) + zp_h;               \
                a_reg[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;              \
                a_reg[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;              \
                a_reg[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;              \
                a_reg[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;              \
                a_reg[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;              \
                                                                                              \
                half16_t b_reg = *(const half16_t*)(xg + k_off);                              \
                                                                                              \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);          \
            }                                                                                 \
        }                                                                                     \
                                                                                              \
        /* KW waves x 8 acc lanes x 32 lanes: KW KiB. Lane-consecutive: bank-clean. */        \
        __shared__ float red[KW][8][32];                                                      \
        _Pragma("unroll")                                                                     \
        for (int j = 0; j < 8; j++)                                                           \
            red[wave_id][j][lane] = acc[j];                                                   \
        /* All threads reach the barrier every launch — no early returns. */                  \
        __syncthreads();                                                                      \
                                                                                              \
        /* Wave 0 sums in FIXED order w=0..KW-1 (deterministic), single Y += owner. */        \
        if (wave_id == 0) {                                                                   \
            float8_t sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                   \
            for (int w = 0; w < (KW); w++) {                                                  \
                _Pragma("unroll")                                                             \
                for (int j = 0; j < 8; j++)                                                   \
                    sum[j] += red[w][j][lane];                                                \
            }                                                                                 \
            /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. */                  \
            if (out_col < N) {                                                                \
                _Pragma("unroll")                                                             \
                for (int j = 0; j < 8; j++) {                                                 \
                    const int out_row = row_start + 2 * j + (lane >> 4);                       \
                    if (out_row < M)                                                          \
                        Y[(long long)out_col * M + out_row] += sum[j];                        \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    }

GEN_RESID_KSPLIT_LDS(2)
GEN_RESID_KSPLIT_LDS(4)
GEN_RESID_KSPLIT_LDS(8)
