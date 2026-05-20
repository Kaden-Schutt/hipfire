// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ═══ FP32→FP16 conversion kernel ═══
// One-shot bulk conversion of activation matrix [N × K] from FP32 to FP16.
// Runs once before the GEMM batch, amortized across M rows.
// Bandwidth: read N*K*4 + write N*K*2 = 6 bytes/elem. Trivial vs GEMM cost.
extern "C" __launch_bounds__(256)
__global__ void convert_f32_to_f16(
    const float* __restrict__ in,
    _Float16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = (_Float16)in[i];
}

// ═══ FP16-input batched HFQ4-G256 GEMM with fused residual add ═══
//
// Same math as gemm_hfq4g256_residual, but:
//   - X is pre-converted to FP16 (half the load bandwidth)
//   - Weights dequant to FP16
//   - Inner loop uses __hfma2 (v_pk_fma_f16, 2 FP16 FMAs/cycle)
//   - Cross-group accumulation in FP32 (precision)
//   - Final warp reduction in FP32
//
// The X bandwidth reduction is the primary win: each batch element loads
// K × 2 bytes instead of K × 4 bytes per row iteration.
//
// Grid/block layout identical to FP32 variant: M × ceil(N/BATCH_TILE), 32 threads.

#define BATCH_TILE 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_hfq4g256_residual_fp16(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,  // [batch_size, K], row-major (FP16, pre-converted)
    float* __restrict__ Y,           // [batch_size, M], row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;  // byte offset into nibble data (8 nibbles = 4 bytes)

    // FP32 accumulators — cross-group precision.
    float acc0[BATCH_TILE], acc1[BATCH_TILE], acc2[BATCH_TILE], acc3[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    // Dequant helper: 4 bytes (8 nibbles) → 4 × __half2 (8 FP16 weights)
    #define DEQUANT_H2(pk, sc_h, zp_h, h01, h23, h45, h67)           \
        do {                                                           \
            __half w0 = __float2half((float)((pk      ) & 0xFu));      \
            __half w1 = __float2half((float)((pk >>  4) & 0xFu));      \
            __half w2 = __float2half((float)((pk >>  8) & 0xFu));      \
            __half w3 = __float2half((float)((pk >> 12) & 0xFu));      \
            __half w4 = __float2half((float)((pk >> 16) & 0xFu));      \
            __half w5 = __float2half((float)((pk >> 20) & 0xFu));      \
            __half w6 = __float2half((float)((pk >> 24) & 0xFu));      \
            __half w7 = __float2half((float)((pk >> 28) & 0xFu));      \
            /* w = scale * q + zero in FP16 */                         \
            __half2 sc2 = __half2half2(sc_h);                          \
            __half2 zp2 = __half2half2(zp_h);                          \
            h01 = __hfma2(__halves2half2(w0, w1), sc2, zp2);           \
            h23 = __hfma2(__halves2half2(w2, w3), sc2, zp2);           \
            h45 = __hfma2(__halves2half2(w4, w5), sc2, zp2);           \
            h67 = __hfma2(__halves2half2(w6, w7), sc2, zp2);           \
        } while(0)

    // Load 8 FP16 X values as 4 × __half2 (direct load, no conversion)
    #define LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67)              \
        do {                                                           \
            const __half2* xp = (const __half2*)(xb + (base));         \
            xh01 = xp[0]; xh23 = xp[1]; xh45 = xp[2]; xh67 = xp[3]; \
        } while(0)

    // Dot product: 4 × __hfma2 into __half2, then extract to FP32 and add to acc
    #define DOT_H2_ACC(wh01, wh23, wh45, wh67, xh01, xh23, xh45, xh67, acc) \
        do {                                                                   \
            __half2 s = __hmul2(wh01, xh01);                                   \
            s = __hfma2(wh23, xh23, s);                                        \
            s = __hfma2(wh45, xh45, s);                                        \
            s = __hfma2(wh67, xh67, s);                                        \
            (acc) += __low2float(s) + __high2float(s);                         \
        } while(0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        // Load scale/zero as FP16
        __half sc0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0)));
        __half zp0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4)));
        __half sc1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1)));
        __half zp1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4)));
        __half sc2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2)));
        __half zp2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4)));
        __half sc3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3)));
        __half zp3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4)));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        // Dequant 4 groups × 8 weights → 4 groups × 4 __half2
        __half2 w0_01, w0_23, w0_45, w0_67;
        __half2 w1_01, w1_23, w1_45, w1_67;
        __half2 w2_01, w2_23, w2_45, w2_67;
        __half2 w3_01, w3_23, w3_45, w3_67;
        DEQUANT_H2(pk0, sc0_h, zp0_h, w0_01, w0_23, w0_45, w0_67);
        DEQUANT_H2(pk1, sc1_h, zp1_h, w1_01, w1_23, w1_45, w1_67);
        DEQUANT_H2(pk2, sc2_h, zp2_h, w2_01, w2_23, w2_45, w2_67);
        DEQUANT_H2(pk3, sc3_h, zp3_h, w3_01, w3_23, w3_45, w3_67);

        const int base = g * 256 + tid * 8;

        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const _Float16* xb = X + (long long)(batch_start + b) * K;

            __half2 xh01, xh23, xh45, xh67;

            LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67);
            DOT_H2_ACC(w0_01, w0_23, w0_45, w0_67, xh01, xh23, xh45, xh67, acc0[b]);

            LOAD_X_H2(xb, base + 256, xh01, xh23, xh45, xh67);
            DOT_H2_ACC(w1_01, w1_23, w1_45, w1_67, xh01, xh23, xh45, xh67, acc1[b]);

            LOAD_X_H2(xb, base + 512, xh01, xh23, xh45, xh67);
            DOT_H2_ACC(w2_01, w2_23, w2_45, w2_67, xh01, xh23, xh45, xh67, acc2[b]);

            LOAD_X_H2(xb, base + 768, xh01, xh23, xh45, xh67);
            DOT_H2_ACC(w3_01, w3_23, w3_45, w3_67, xh01, xh23, xh45, xh67, acc3[b]);
        }
    }

    // Tail groups
    #define TAIL_FP16(acc_arr, g_idx)                                            \
        {                                                                         \
            const int g = g_idx;                                                  \
            const char* gp = row_ptr + g * 136;                                   \
            __half sc_h = __float2half(__builtin_bit_cast(float,                   \
                           *(const unsigned int*)(gp)));                           \
            __half zp_h = __float2half(__builtin_bit_cast(float,                   \
                           *(const unsigned int*)(gp + 4)));                       \
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);              \
            __half2 wh01, wh23, wh45, wh67;                                       \
            DEQUANT_H2(pk, sc_h, zp_h, wh01, wh23, wh45, wh67);                  \
            const int base = g * 256 + tid * 8;                                   \
            _Pragma("unroll")                                                     \
            for (int b = 0; b < BATCH_TILE; b++) {                                \
                if (b >= local_bs) break;                                         \
                const _Float16* xb = X + (long long)(batch_start + b) * K;           \
                __half2 xh01, xh23, xh45, xh67;                                  \
                LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67);                    \
                DOT_H2_ACC(wh01, wh23, wh45, wh67,                               \
                           xh01, xh23, xh45, xh67, (acc_arr)[b]);                \
            }                                                                     \
        }

    if (tail >= 1) { TAIL_FP16(acc0, quads << 2); }
    if (tail >= 2) { TAIL_FP16(acc1, (quads << 2) + 1); }
    if (tail >= 3) { TAIL_FP16(acc2, (quads << 2) + 2); }

    #undef DEQUANT_H2
    #undef LOAD_X_H2
    #undef DOT_H2_ACC
    #undef TAIL_FP16

    // Pairwise combine + warp reduction (FP32, same as FP32 variant)
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row] += val;
    }
}
