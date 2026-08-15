// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Dense parent-checkpoint weight decode: F8_E4M3 + UE8M0 128×128 block
// scales → BF16. Model-lifetime staging op (not a hot-path decode).
//
//   W : E4M3  [M, K]              row-major
//   S : UE8M0 [ceil(M/128), ceil(K/128)] row-major
//   O : BF16  [M, K]              row-major
//
//   out[m][k] = bf16( e4m3(W[m][k]) * ue8m0(S[m/128][k/128]) )
//
// Geometry:
//   block = [256, 1, 1]
//   grid  = [M, 1, 1]             one workgroup per output row
// Each lane walks K with stride 256, loading uchar4 (4 E4M3 codes) when
// aligned and finishing any K%4 tail scalar. Ragged trailing 128×128 scale
// blocks are handled by integer division (m/128, k/128) against
// sK = ceil(K/128); no floor truncation of the last partial block.
//
// UE8M0 decode: b==0xFF → NaN (must propagate; bit-shift form yields Inf).
// b==0 → 2^-127 via exp2f (bit form 0<<23 is +0.0, wrong).
// b in 1..=254 → bit form (b<<23) is the exact power-of-two.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__device__ __forceinline__ float ue8m0_to_f32_dq(unsigned char b) {
    // NaN scale is checkpoint corruption — caller finite-checks must see it.
    if (b == 0xFFu) {
        return __builtin_nanf("");
    }
    // b==0 cannot use the <<23 bit form (that yields +0.0, not 2^-127).
    if (b == 0u) {
        return exp2f(-127.0f);
    }
    return __builtin_bit_cast(float, (unsigned int)b << 23);
}

// OCP float8_e4m3fn (finite-only): sign@7, exp@6:3 bias 7, mant@2:0.
// exp==0 → subnormal 2^-6 * (mant/8); exp==0xF && mant==0x7 → NaN.
__device__ __forceinline__ float e4m3_to_f32_dq(unsigned char b) {
    const unsigned sign = (b >> 7) & 1u;
    const int exp = (int)((b >> 3) & 0xFu);
    const unsigned mant = (unsigned)(b & 0x7u);
    float mag;
    if (exp == 0) {
        // 2^-6 * (mant/8) = mant * 2^-9
        mag = (float)mant * 0.001953125f;
    } else if (exp == 0xF && mant == 0x7u) {
        // Only NaN encoding (0x7F / 0xFF). Preserve sign bit in the payload.
        const unsigned bits = 0x7FC00000u | (sign << 31);
        return __builtin_bit_cast(float, bits);
    } else {
        // 2^(exp-7) * (1 + mant/8); f32 exp field = (exp-7)+127 = exp+120
        const float pow2 = __builtin_bit_cast(
            float, ((unsigned int)(exp + 120)) << 23);
        mag = pow2 * (1.0f + (float)mant * 0.125f);
    }
    return sign ? -mag : mag;
}

extern "C" __global__ __launch_bounds__(256)
void dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(
    const unsigned char* __restrict__ w,   // E4M3 [M, K]
    const unsigned char* __restrict__ s,   // UE8M0 [ceil(M/128), ceil(K/128)]
    __hip_bfloat16* __restrict__ out,      // BF16 [M, K]
    int M,
    int K
) {
    const int m = (int)blockIdx.x;
    if (m >= M) {
        return;
    }

    const int sK = (K + 127) / 128;                 // ceil(K/128)
    const int scale_row = m >> 7;                   // m / 128
    const unsigned char* row_w = w + (long long)m * (long long)K;
    __hip_bfloat16* row_out = out + (long long)m * (long long)K;
    const unsigned char* scale_base = s + (long long)scale_row * (long long)sK;

    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;

    // Vector path: 4 E4M3 codes per load. Cover full uchar4-aligned prefix.
    const int n_vec = K >> 2;                       // floor(K/4)
    for (int v = tid; v < n_vec; v += nthreads) {
        const int k0 = v << 2;
        // All 4 k's share the same 128-wide scale column when k0%128 <= 124;
        // still safe to recompute per element (branchless, cheap).
        const uchar4 codes = *reinterpret_cast<const uchar4*>(row_w + k0);
        const unsigned char bytes[4] = {codes.x, codes.y, codes.z, codes.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = k0 + i;
            const float scale = ue8m0_to_f32_dq(scale_base[k >> 7]);
            const float val = e4m3_to_f32_dq(bytes[i]) * scale;
            row_out[k] = __hip_bfloat16(val);
        }
    }

    // Scalar tail for K not divisible by 4 (ragged K, not just ragged scale).
    for (int k = (n_vec << 2) + tid; k < K; k += nthreads) {
        const float scale = ue8m0_to_f32_dq(scale_base[k >> 7]);
        const float val = e4m3_to_f32_dq(row_w[k]) * scale;
        row_out[k] = __hip_bfloat16(val);
    }
}
