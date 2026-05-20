// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// v_dot2_f32_f16-accelerated batched 4-way fused HFQ6-G256 GEMM for RDNA2.
// Available on gfx1011, gfx1012, gfx1030, gfx1031, gfx1032 (and gfx1100+).
// Not available on gfx1010 (Navi 10) or gfx1013 (Van Gogh/BC-250 APU).
//
// The `amd_mixed_dot(a, b, c, false)` HIP wrapper emits one `v_dot2_f32_f16`
// instruction computing: a.x*b.x + a.y*b.y + c with FP32 accumulation. This
// replaces the FP16-packed pattern's `__hmul2 + __hfma2×3 + __low2float/__high2float
// + add` (≈6 ops) with 4 dot instructions for the same 8 FMAs worth of work —
// and keeps the accumulator in FP32 throughout, eliminating the periodic
// cast-to-FP32.
//
// HFQ6-G256 group layout: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes
// Per thread: 8 weights in 6 bytes (4 values per 3 bytes).
//
// Overwrite semantics (Y = acc) — same as the scalar/fp16/wmma fused variants.

#define BATCH_TILE 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_qkvza_hfq6g256_dot2(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,        // [N × K], FP16 pre-converted
    float*         __restrict__ Y_qkv,     // [N × qkv_m], FP32
    float*         __restrict__ Y_z,       // [N × z_m], FP32
    float*         __restrict__ Y_beta,    // [N × beta_m], FP32
    float*         __restrict__ Y_alpha,   // [N × alpha_m], FP32
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = N - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    // Route this workgroup to its target matrix + output row + output stride.
    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < qkv_m) {
        A = A_qkv;   Y = Y_qkv;   row = gid;                               out_stride = qkv_m;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     Y = Y_z;     row = gid - qkv_m;                        out_stride = z_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  Y = Y_beta;  row = gid - (qkv_m + z_m);                 out_stride = beta_m;
    } else {
        A = A_alpha; Y = Y_alpha; row = gid - (qkv_m + z_m + beta_m);        out_stride = alpha_m;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;  // byte offset into 6-bit data (8 values = 6 bytes)

    // 4-accumulator interleave (matches scalar/FP16 reference for byte-compare).
    float acc0[BATCH_TILE], acc1[BATCH_TILE], acc2[BATCH_TILE], acc3[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    // Dequant helper: 6 bytes (8 × 6-bit values) → 4 × __half2 (8 FP16 weights).
    #define DEQUANT_H2_6BIT(dp, sc_h, zp_h, h01, h23, h45, h67)            \
        do {                                                                \
            unsigned char b0 = (dp)[0], b1 = (dp)[1], b2 = (dp)[2];        \
            unsigned char b3 = (dp)[3], b4 = (dp)[4], b5 = (dp)[5];        \
            __half w0 = __float2half((float)(b0 & 63));                     \
            __half w1 = __float2half((float)((b0 >> 6) | ((b1 & 0xF) << 2))); \
            __half w2 = __float2half((float)((b1 >> 4) | ((b2 & 3) << 4))); \
            __half w3 = __float2half((float)(b2 >> 2));                     \
            __half w4 = __float2half((float)(b3 & 63));                     \
            __half w5 = __float2half((float)((b3 >> 6) | ((b4 & 0xF) << 2))); \
            __half w6 = __float2half((float)((b4 >> 4) | ((b5 & 3) << 4))); \
            __half w7 = __float2half((float)(b5 >> 2));                     \
            __half2 sc2 = __half2half2(sc_h);                               \
            __half2 zp2 = __half2half2(zp_h);                               \
            h01 = __hfma2(__halves2half2(w0, w1), sc2, zp2);                \
            h23 = __hfma2(__halves2half2(w2, w3), sc2, zp2);                \
            h45 = __hfma2(__halves2half2(w4, w5), sc2, zp2);                \
            h67 = __hfma2(__halves2half2(w6, w7), sc2, zp2);                \
        } while(0)

    #define LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67)              \
        do {                                                           \
            const __half2* xp = (const __half2*)(xb + (base));         \
            xh01 = xp[0]; xh23 = xp[1]; xh45 = xp[2]; xh67 = xp[3];   \
        } while(0)

    // Dot-product: 4 × v_dot2_f32_f16 via amd_mixed_dot (HIP wrapper around
    // __ockl_fdot2). Each call does a.x*b.x + a.y*b.y + c with FP32 accum.
    #define DOT2_ACC(wh01, wh23, wh45, wh67, xh01, xh23, xh45, xh67, acc) \
        do {                                                                 \
            (acc) = amd_mixed_dot((wh01), (xh01), (acc), false);             \
            (acc) = amd_mixed_dot((wh23), (xh23), (acc), false);             \
            (acc) = amd_mixed_dot((wh45), (xh45), (acc), false);             \
            (acc) = amd_mixed_dot((wh67), (xh67), (acc), false);             \
        } while(0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const char* gp1 = gp0 + 200;
        const char* gp2 = gp1 + 200;
        const char* gp3 = gp2 + 200;

        __half sc0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0)));
        __half zp0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4)));
        __half sc1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1)));
        __half zp1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4)));
        __half sc2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2)));
        __half zp2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4)));
        __half sc3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3)));
        __half zp3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4)));

        const unsigned char* dp0 = (const unsigned char*)(gp0 + 8 + boff);
        const unsigned char* dp1 = (const unsigned char*)(gp1 + 8 + boff);
        const unsigned char* dp2 = (const unsigned char*)(gp2 + 8 + boff);
        const unsigned char* dp3 = (const unsigned char*)(gp3 + 8 + boff);

        __half2 w0_01, w0_23, w0_45, w0_67;
        __half2 w1_01, w1_23, w1_45, w1_67;
        __half2 w2_01, w2_23, w2_45, w2_67;
        __half2 w3_01, w3_23, w3_45, w3_67;
        DEQUANT_H2_6BIT(dp0, sc0_h, zp0_h, w0_01, w0_23, w0_45, w0_67);
        DEQUANT_H2_6BIT(dp1, sc1_h, zp1_h, w1_01, w1_23, w1_45, w1_67);
        DEQUANT_H2_6BIT(dp2, sc2_h, zp2_h, w2_01, w2_23, w2_45, w2_67);
        DEQUANT_H2_6BIT(dp3, sc3_h, zp3_h, w3_01, w3_23, w3_45, w3_67);

        const int base = g * 256 + tid * 8;

        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const _Float16* xb = X + (long long)(batch_start + b) * K;

            __half2 xh01, xh23, xh45, xh67;

            LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67);
            DOT2_ACC(w0_01, w0_23, w0_45, w0_67, xh01, xh23, xh45, xh67, acc0[b]);

            LOAD_X_H2(xb, base + 256, xh01, xh23, xh45, xh67);
            DOT2_ACC(w1_01, w1_23, w1_45, w1_67, xh01, xh23, xh45, xh67, acc1[b]);

            LOAD_X_H2(xb, base + 512, xh01, xh23, xh45, xh67);
            DOT2_ACC(w2_01, w2_23, w2_45, w2_67, xh01, xh23, xh45, xh67, acc2[b]);

            LOAD_X_H2(xb, base + 768, xh01, xh23, xh45, xh67);
            DOT2_ACC(w3_01, w3_23, w3_45, w3_67, xh01, xh23, xh45, xh67, acc3[b]);
        }
    }

    // Tail groups — distribute to acc[g & 3] to match scalar accumulation order.
    #define TAIL_DOT2(acc_arr, g_idx)                                            \
        {                                                                         \
            const int g = g_idx;                                                  \
            const char* gp = row_ptr + g * 200;                                   \
            __half sc_h = __float2half(__builtin_bit_cast(float,                   \
                           *(const unsigned int*)(gp)));                           \
            __half zp_h = __float2half(__builtin_bit_cast(float,                   \
                           *(const unsigned int*)(gp + 4)));                       \
            const unsigned char* dp = (const unsigned char*)(gp + 8 + boff);      \
            __half2 wh01, wh23, wh45, wh67;                                       \
            DEQUANT_H2_6BIT(dp, sc_h, zp_h, wh01, wh23, wh45, wh67);             \
            const int base = g * 256 + tid * 8;                                   \
            _Pragma("unroll")                                                     \
            for (int b = 0; b < BATCH_TILE; b++) {                                \
                if (b >= local_bs) break;                                         \
                const _Float16* xb = X + (long long)(batch_start + b) * K;           \
                __half2 xh01, xh23, xh45, xh67;                                  \
                LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67);                    \
                DOT2_ACC(wh01, wh23, wh45, wh67,                                 \
                         xh01, xh23, xh45, xh67, (acc_arr)[b]);                  \
            }                                                                     \
        }

    if (tail >= 1) { TAIL_DOT2(acc0, quads << 2); }
    if (tail >= 2) { TAIL_DOT2(acc1, (quads << 2) + 1); }
    if (tail >= 3) { TAIL_DOT2(acc2, (quads << 2) + 2); }

    #undef DEQUANT_H2_6BIT
    #undef LOAD_X_H2
    #undef DOT2_ACC
    #undef TAIL_DOT2

    // Pairwise combine + warp reduction — overwrite Y (fused qkvza semantics, not +=).
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
