// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Routed-expert parent-checkpoint weight decode: packed E2M1 (I8) +
// per-row UE8M0 group-32 scales → BF16. Hot path — called per-expert into
// a reusable scratch buffer during routing.
//
//   W : I8    [M, K/2]   two E2M1 nibbles/byte, low nibble = even k
//   S : UE8M0 [M, K/32]  per-row, per-32-K-group
//   O : BF16  [M, K]     row-major, K = logical width (multiple of 32)
//
//   out[m][k] = bf16( e2m1_lut[nibble(W[m][k/2], k&1)]
//                     * ue8m0(S[m][k/32]) )
//
// Geometry:
//   block = [256, 1, 1]           __launch_bounds__(256, 4)
//   grid  = [M, ceil((K/32)/256)] one thread owns one 32-element K-group
// Each lane loads the group's 16 packed bytes as one uint4, multiplies by
// the single UE8M0 scale, and writes 16 bf16 pairs (32 values).
//
// UE8M0: same policy as the dense FP8 sibling — exp2f for b==0 (bit form
// wrong), explicit NaN for b==0xFF, bit form for 1..=254.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// OCP float4_e2m1fn exact LUT. Index = 4-bit code (sign@3, exp@2:1 bias 1,
// mant@0). No inf / NaN encodings.
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float ue8m0_to_f32_dq(unsigned char b) {
    if (b == 0xFFu) {
        return __builtin_nanf("");
    }
    if (b == 0u) {
        return exp2f(-127.0f);
    }
    return __builtin_bit_cast(float, (unsigned int)b << 23);
}

extern "C" __global__ __launch_bounds__(256, 4)
void dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(
    const unsigned char* __restrict__ w,   // packed E2M1 [M, K/2]
    const unsigned char* __restrict__ s,   // UE8M0 [M, K/32]
    __hip_bfloat16* __restrict__ out,      // BF16 [M, K]
    int M,
    int K                                 // logical K, multiple of 32
) {
    const int m = (int)blockIdx.x;
    if (m >= M) {
        return;
    }

    const int n_groups = K >> 5;                    // K / 32
    const int g = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (g >= n_groups) {
        return;
    }

    const float scale = ue8m0_to_f32_dq(
        s[(long long)m * (long long)n_groups + (long long)g]);

    // 16 packed bytes = 32 E2M1 codes. uint4 is one 16B load.
    const unsigned int* wp = reinterpret_cast<const unsigned int*>(
        w + (long long)m * (long long)(K >> 1) + (long long)g * 16LL);
    const uint4 packed = *reinterpret_cast<const uint4*>(wp);

    __hip_bfloat16* op =
        out + (long long)m * (long long)K + (long long)g * 32LL;

    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
        unsigned int word = words[wi];
        #pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            const unsigned char byte = (unsigned char)(word & 0xFFu);
            word >>= 8;
            const float lo = E2M1_LUT[byte & 0x0Fu] * scale;
            const float hi = E2M1_LUT[byte >> 4] * scale;
            const int o = wi * 8 + bi * 2;
            op[o]     = __hip_bfloat16(lo);         // even k = low nibble
            op[o + 1] = __hip_bfloat16(hi);         // odd  k = high nibble
        }
    }
}
