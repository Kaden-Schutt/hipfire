// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ fused QKVZA DECODE (B=1), W4A8 dp4a — gfx1103 / RDNA3 (and gfx1100/1/2).
// The four in-proj GEMVs in ONE launch, one wave32 per OUTPUT ROW, blockIdx.x
// demuxing the projection. Uses RDNA3's v_dot4_i32_iu8 (`__builtin_amdgcn_sudot4`,
// dot8-insts) — 4 signed-int8 MACs/instruction. The 4-bit symmetric weight is
// unpacked+sign-extended to int8 and dotted against the int8-quantized activation
// (quantize_act_oq8, per-256-group symmetric); the per-group int32 dot is rescaled
// by sc_w·sc_x (no zero-point → no correction term). gfx1103 wave32, ZERO LDS.
//
// gfx1151 (RDNA3.5) has its own variant in kernels/src/gfx1151/ (it also carries
// the legacy dot1-insts v_dot4_i32_i8, so it is kept separate for independent
// tuning rather than sharing this gfx1103 source).
//
//   W_* : [m_*, K/2] packed signed int4 ++ [m_*, K/256] f32 scales (combined buf)
//   Xq  : [K] int8 (shared, quantized) ; Xs : [K/256] f32 act scales
//   grid.x = qkv_m + z_m + beta_m + alpha_m, block = 32.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void fused_qkvza_oq4_dp4a(
    const uint8_t* __restrict__ W_qkv,
    const uint8_t* __restrict__ W_z,
    const uint8_t* __restrict__ W_beta,
    const uint8_t* __restrict__ W_alpha,
    const int8_t*  __restrict__ Xq,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int group
) {
    const uint8_t* W;
    float* Y;
    int M, row;
    int gx = blockIdx.x;
    if (gx < qkv_m) {
        W = W_qkv; Y = Y_qkv; M = qkv_m; row = gx;
    } else if (gx < qkv_m + z_m) {
        W = W_z; Y = Y_z; M = z_m; row = gx - qkv_m;
    } else if (gx < qkv_m + z_m + beta_m) {
        W = W_beta; Y = Y_beta; M = beta_m; row = gx - qkv_m - z_m;
    } else if (gx < qkv_m + z_m + beta_m + alpha_m) {
        W = W_alpha; Y = Y_alpha; M = alpha_m; row = gx - qkv_m - z_m - beta_m;
    } else {
        return;
    }
    const int lane = threadIdx.x;
    const int ng = K / group;
    const int gbytes = group / 2;
    const uint8_t* w_row = W + (long long)row * (K / 2);
    const float* ws_row = (const float*)(W + (long long)M * (K / 2)) + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * gbytes + lane * 4;  // 8 nibbles
        const int xoff = g * group + lane * 8;    // 8 int8 activations
        const uint32_t wv = *(const uint32_t*)(w_row + boff);
        #define SX(s) ((int)(((int)(((wv) >> (s)) & 0xFu) << 28) >> 28) & 0xFF)
        const int int_a = SX(0) | (SX(4) << 8) | (SX(8) << 16) | (SX(12) << 24);
        const int int_b = SX(16) | (SX(20) << 8) | (SX(24) << 16) | (SX(28) << 24);
        #undef SX
        const int xq_a = *(const int*)(Xq + xoff);
        const int xq_b = *(const int*)(Xq + xoff + 4);
        int isum = __builtin_amdgcn_sudot4(true, int_a, true, xq_a, 0, false);
        isum = __builtin_amdgcn_sudot4(true, int_b, true, xq_b, isum, false);
        facc += (float)isum * ws_row[g] * Xs[g];
    }
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] = facc;
    }
}
