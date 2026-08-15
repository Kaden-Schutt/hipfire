// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#ifndef HIPFIRE_GFX1151_XOR_SHUFFLE_H
#define HIPFIRE_GFX1151_XOR_SHUFFLE_H

// gfx1151 lane exchange table, bit-equivalent to
// __shfl_xor(bits, stride, 32). Keep this header behind an exact-architecture
// dispatch: the selected instructions are gfx11-specific.
__device__ __forceinline__ int hipfire_xor_shuffle_b32_gfx1151(
    int bits,
    int stride
) {
    switch (stride) {
        case 1:
            return __builtin_amdgcn_mov_dpp(bits, 0xb1, 0xf, 0xf, false);
        case 2:
            return __builtin_amdgcn_mov_dpp(bits, 0x4e, 0xf, 0xf, false);
        case 4:
            return __builtin_amdgcn_ds_swizzle(bits, 0x101f);
        case 8:
            return __builtin_amdgcn_ds_swizzle(bits, 0x201f);
        case 16:
            return __builtin_amdgcn_permlanex16(
                bits, bits, 0x76543210, 0xfedcba98, false, false);
        default:
            return __shfl_xor(bits, stride, 32);
    }
}

#endif
