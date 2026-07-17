// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact Qwen3.5-27B dense decode gate/up producer for gfx1201.
//
// The generic fused entry point assigns gate and up rows to independent
// waves. This variant gives one wave the matching gate/up row pair. The two
// dot products share each activation load while retaining independent weight
// streams and the base kernel's four accumulator chains. Grid=[gate_m], so it
// also halves the number of scheduled workgroups without reducing weight
// parallelism.
//
// Host admission is deliberately exact: K=5120 and gate_m=up_m=17408.
// Twenty HFQ4-G256 groups means five complete quads and no tail machinery.
// Set HIPFIRE_GFX1201_HFQ4_27B_GATE_UP_PAIR=0 for the one-row control.

#ifndef HIPFIRE_GATE_UP_PAIR_KERNEL
#define HIPFIRE_GATE_UP_PAIR_KERNEL fused_gate_up_hfq4g256_pair_k5120_global_gfx1201
#endif

#ifndef HIPFIRE_GATE_UP_PAIR_SKEW
#define HIPFIRE_GATE_UP_PAIR_SKEW 0
#endif

#ifndef HIPFIRE_GATE_UP_PAIR_WINDOW
#define HIPFIRE_GATE_UP_PAIR_WINDOW 4
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_GATE_UP_PAIR_KERNEL(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
#if HIPFIRE_GATE_UP_PAIR_SKEW
    , int up_skew
#endif
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= gate_m || row >= up_m) return;
    (void)K;

#if HIPFIRE_GATE_UP_PAIR_SKEW
    // Pair a gate row with a different up row so the two simultaneous packed
    // weight streams do not share identical low address bits. This is a pure
    // execution permutation: every gate/up output row still has one writer.
    int up_row_index = row + up_skew;
    if (up_row_index >= up_m) up_row_index -= up_m;
#else
    const int up_row_index = row;
#endif

    constexpr int groups_per_row = 20;
    constexpr int group_bytes = 136;
    constexpr int row_bytes = groups_per_row * group_bytes;
    const char* gate_row = A_gate + (long long)row * row_bytes;
    const char* up_row = A_up + (long long)up_row_index * row_bytes;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto gate_rsrc = HIPFIRE_WEIGHT_RSRC(gate_row);
    const auto up_rsrc = HIPFIRE_WEIGHT_RSRC(up_row);
#endif

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;

    #define PAIRED_DOG(group, gacc, uacc) do { \
        const int _g = (group); \
        const int _off = _g * group_bytes; \
        const char* _gg = gate_row + _off; \
        const char* _ug = up_row + _off; \
        const float _gsc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, _gg, _off)); \
        const float _gzp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(gate_rsrc, _gg + 4, _off + 4)); \
        const unsigned int _gpk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            gate_rsrc, _gg + 8 + boff, _off + 8 + boff); \
        const float _usc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, _ug, _off)); \
        const float _uzp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(up_rsrc, _ug + 4, _off + 4)); \
        const unsigned int _upk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            up_rsrc, _ug + 8 + boff, _off + 8 + boff); \
        const int _base = _g * 256 + tid * 8; \
        const float4 _xa = *reinterpret_cast<const float4*>(x + _base); \
        const float4 _xb = *reinterpret_cast<const float4*>(x + _base + 4); \
        (gacc) += (_gsc * (float)(_gpk & 0xFu) + _gzp) * _xa.x \
                + (_gsc * (float)((_gpk >> 4) & 0xFu) + _gzp) * _xa.y \
                + (_gsc * (float)((_gpk >> 8) & 0xFu) + _gzp) * _xa.z \
                + (_gsc * (float)((_gpk >> 12) & 0xFu) + _gzp) * _xa.w \
                + (_gsc * (float)((_gpk >> 16) & 0xFu) + _gzp) * _xb.x \
                + (_gsc * (float)((_gpk >> 20) & 0xFu) + _gzp) * _xb.y \
                + (_gsc * (float)((_gpk >> 24) & 0xFu) + _gzp) * _xb.z \
                + (_gsc * (float)((_gpk >> 28) & 0xFu) + _gzp) * _xb.w; \
        (uacc) += (_usc * (float)(_upk & 0xFu) + _uzp) * _xa.x \
                + (_usc * (float)((_upk >> 4) & 0xFu) + _uzp) * _xa.y \
                + (_usc * (float)((_upk >> 8) & 0xFu) + _uzp) * _xa.z \
                + (_usc * (float)((_upk >> 12) & 0xFu) + _uzp) * _xa.w \
                + (_usc * (float)((_upk >> 16) & 0xFu) + _uzp) * _xb.x \
                + (_usc * (float)((_upk >> 20) & 0xFu) + _uzp) * _xb.y \
                + (_usc * (float)((_upk >> 24) & 0xFu) + _uzp) * _xb.z \
                + (_usc * (float)((_upk >> 28) & 0xFu) + _uzp) * _xb.w; \
    } while (0)

    // Full unroll holds all five quads live and spills one VGPR to private
    // memory on gfx1201. Preserve the compact loop just like the admitted
    // dual-row residual kernel: it stays below Redline's zero-scratch limit.
    #pragma unroll 1
    for (int q = 0; q < 5; q++) {
        const int g = q << 2;
        PAIRED_DOG(g + 0, gacc0, uacc0);
#if HIPFIRE_GATE_UP_PAIR_WINDOW == 1
        asm volatile("" ::: "memory");
#endif
        PAIRED_DOG(g + 1, gacc1, uacc1);
#if HIPFIRE_GATE_UP_PAIR_WINDOW <= 2
        asm volatile("" ::: "memory");
#endif
        PAIRED_DOG(g + 2, gacc2, uacc2);
#if HIPFIRE_GATE_UP_PAIR_WINDOW == 1
        asm volatile("" ::: "memory");
#endif
        PAIRED_DOG(g + 3, gacc3, uacc3);
#if HIPFIRE_GATE_UP_PAIR_WINDOW <= 2
        asm volatile("" ::: "memory");
#endif
    }
    #undef PAIRED_DOG

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        gacc += __shfl_down(gacc, offset);
        uacc += __shfl_down(uacc, offset);
    }
    if (tid == 0) {
        y_gate[row] = gacc;
        y_up[up_row_index] = uacc;
    }
}
