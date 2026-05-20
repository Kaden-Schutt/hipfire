// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Dequant HFQ4-G256 packed weights to row-major FP16.
// Input:  A_packed [M × groups_per_row × 136 bytes] (4-bit packed)
// Output: W_f16 [M × K] FP16 (row-major, WMMA-ready)
//
// Grid: [M, groups_per_row]. Block: [32].
// Each block dequants one 256-element group for one row.
// Thread t dequants 8 elements (tid*8..tid*8+7).

extern "C" __launch_bounds__(32, 16)
__global__ void dequant_hfq4g256_to_f16(
    const char* __restrict__ A_packed,
    _Float16* __restrict__ W_f16,
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;

    const char* gp = A_packed + (long long)row * groups_per_row * 136 + group * 136;
    float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));
    float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
    _Float16 sc_h = (_Float16)scale;
    _Float16 zp_h = (_Float16)zero;

    const int boff = tid * 4;  // 8 nibbles = 4 bytes
    unsigned int pk = *(const unsigned int*)(gp + 8 + boff);

    _Float16* out = W_f16 + (long long)row * K + group * 256 + tid * 8;
    out[0] = sc_h * (_Float16)(float)((pk      ) & 0xFu) + zp_h;
    out[1] = sc_h * (_Float16)(float)((pk >>  4) & 0xFu) + zp_h;
    out[2] = sc_h * (_Float16)(float)((pk >>  8) & 0xFu) + zp_h;
    out[3] = sc_h * (_Float16)(float)((pk >> 12) & 0xFu) + zp_h;
    out[4] = sc_h * (_Float16)(float)((pk >> 16) & 0xFu) + zp_h;
    out[5] = sc_h * (_Float16)(float)((pk >> 20) & 0xFu) + zp_h;
    out[6] = sc_h * (_Float16)(float)((pk >> 24) & 0xFu) + zp_h;
    out[7] = sc_h * (_Float16)(float)((pk >> 28) & 0xFu) + zp_h;
}
