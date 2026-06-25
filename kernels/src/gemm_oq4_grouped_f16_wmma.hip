// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ batched PREFILL core: W4A16 grouped GEMM. 4-bit-resident signed-int4
// weights are dequantized to f16 inline and multiplied by FULL-PRECISION f16
// activations via f16×f16 WMMA — the batched analog of the W4A16 decode GEMV
// (gemv_oq4_grouped). This replaces the W4A4 gemm_oq4_grouped_wmma (int4×int4)
// for OQ4+ prefill: no int4 ACTIVATION quant, so the batched-vs-per-token
// rotation delta is NOT amplified into incoherence (the W4A4 batched divergence
// root cause). Same numerics as mq4's gemm_*_hfq4g256_wmma, symmetric (scale
// only, no zero-point). gfx1100+ (RDNA3 wave32 WMMA), register-tiled, ZERO LDS.
//
//   W  : [M, K/2] packed signed int4 ++ [M, K/G] f32 scales (one combined buffer;
//        the f32 per-group scales live right after the M*(K/2) nibble region)
//   X  : [B, K]   f16 activation (full precision)
//   Y  : [B, M]   f32
//
// Y[b,m] = Σ_g sw[m,g] · Σ_{k∈g} dequant(qw[m,k]) · x[b,k]
// The f32 per-group scale is folded into the f16 weight at dequant time (per the
// loaded A-row), so no epilogue rescale is needed — identical to mq4.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_oq4_grouped_f16_wmma(
    const uint8_t*  __restrict__ W,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y,
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
    const int stride = K / 2;            // bytes per packed weight row
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;

    // Scales live right after the M*(K/2) nibble region in the combined buffer.
    const float*    Ws     = (const float*)(W + (long long)M * stride);
    const uint8_t*  w_row  = W + (long long)safe_a * stride;
    const float*    ws_row = Ws + (long long)safe_a * n_groups;
    const _Float16* x_prow = X + (long long)safe_x * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // Dequant 16 signed-int4 nibbles (two uint32) → half16, scaled by sw.
    // areg is a half16_t vector lvalue (subscript-assigned directly).
    #define DEQ16(areg, pk0, pk1, sw) do { \
        _Float16 _s = (sw); \
        (areg)[0]=_s*(_Float16)(float)((((int)((pk0)      &0xFu))<<28)>>28); \
        (areg)[1]=_s*(_Float16)(float)((((int)(((pk0)>>4) &0xFu))<<28)>>28); \
        (areg)[2]=_s*(_Float16)(float)((((int)(((pk0)>>8) &0xFu))<<28)>>28); \
        (areg)[3]=_s*(_Float16)(float)((((int)(((pk0)>>12)&0xFu))<<28)>>28); \
        (areg)[4]=_s*(_Float16)(float)((((int)(((pk0)>>16)&0xFu))<<28)>>28); \
        (areg)[5]=_s*(_Float16)(float)((((int)(((pk0)>>20)&0xFu))<<28)>>28); \
        (areg)[6]=_s*(_Float16)(float)((((int)(((pk0)>>24)&0xFu))<<28)>>28); \
        (areg)[7]=_s*(_Float16)(float)((((int)(((pk0)>>28)&0xFu))<<28)>>28); \
        (areg)[8]=_s*(_Float16)(float)((((int)((pk1)      &0xFu))<<28)>>28); \
        (areg)[9]=_s*(_Float16)(float)((((int)(((pk1)>>4) &0xFu))<<28)>>28); \
        (areg)[10]=_s*(_Float16)(float)((((int)(((pk1)>>8) &0xFu))<<28)>>28); \
        (areg)[11]=_s*(_Float16)(float)((((int)(((pk1)>>12)&0xFu))<<28)>>28); \
        (areg)[12]=_s*(_Float16)(float)((((int)(((pk1)>>16)&0xFu))<<28)>>28); \
        (areg)[13]=_s*(_Float16)(float)((((int)(((pk1)>>20)&0xFu))<<28)>>28); \
        (areg)[14]=_s*(_Float16)(float)((((int)(((pk1)>>24)&0xFu))<<28)>>28); \
        (areg)[15]=_s*(_Float16)(float)((((int)(((pk1)>>28)&0xFu))<<28)>>28); \
    } while (0)

    for (int g = 0; g < n_groups; ++g) {
        const _Float16 sw = (_Float16)ws_row[g];     // this A-row's group scale
        const int g_byte = g * (group / 2);
        const _Float16* x_g = x_prow + g * group;
        // 2× K-tile unroll with software pipelining: load tile B's weight+act
        // while tile A's WMMA is in flight (mirrors mq4 gemm_qkv_hfq4g256_wmma K2).
        for (int kt = 0; kt < tiles_per_group; kt += 2) {
            const uint32_t a0 = *(const uint32_t*)(w_row + g_byte + kt * 8);
            const uint32_t a1 = *(const uint32_t*)(w_row + g_byte + kt * 8 + 4);
            const uint32_t b0 = *(const uint32_t*)(w_row + g_byte + (kt + 1) * 8);
            const uint32_t b1 = *(const uint32_t*)(w_row + g_byte + (kt + 1) * 8 + 4);
            half16_t xa = *(const half16_t*)(x_g + kt * 16);
            half16_t xb = *(const half16_t*)(x_g + (kt + 1) * 16);
            half16_t a_reg;
            DEQ16(a_reg, a0, a1, sw);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, xa, acc);
            DEQ16(a_reg, b0, b1, sw);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, xb, acc);
        }
    }
    #undef DEQ16

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
