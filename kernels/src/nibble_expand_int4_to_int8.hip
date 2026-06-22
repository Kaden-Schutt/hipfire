// SPDX-License-Identifier: Apache-2.0
// hipfire — reference kernel layer: packed int4 → int8 weight expand (W4A8).
//
// The W4A8 production target stores weights as 4-bit codes (memory/bandwidth win) but
// rides the iu8 WMMA core, so the codes are EXPANDED to int8 on load. This is the
// boring reference expand: WP[M, K/2] (two 4-bit two's-complement codes per byte,
// low nibble = even k, high nibble = odd k) → W8[M, K] int8 in [-8, 7].
//
// Expand: sign-extend the 4-bit code to int8 via (int8)(n<<4)>>4. The per-group
// weight scale is NOT applied here — it stays out of the WMMA and is folded into the
// int32→f32 dequant (dequant_i32_rowcol), same as W8A8. One thread per output element.
//
// Grid: [ceil(M*K / 256)]. Block: [256].

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(256)
extern "C" __global__ void nibble_expand_int4_to_int8(
    const uint8_t* __restrict__ WP, // [M, K/2] packed 4-bit codes
    signed char* __restrict__ W8,   // [M, K] int8 (sign-extended, [-8,7])
    int m, int k
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)m * k;
    if (i >= n) return;
    // i indexes [m, k] row-major. Two codes per byte: byte = WP[i/2], even k = low.
    const uint8_t byte = WP[i >> 1];
    const uint8_t code = ((i & 1) == 0) ? (byte & 0xf) : (byte >> 4);
    // Sign-extend 4-bit two's complement to int8: shift into the high nibble, then
    // arithmetic right shift back.
    W8[i] = (signed char)((signed char)(code << 4) >> 4);
}
