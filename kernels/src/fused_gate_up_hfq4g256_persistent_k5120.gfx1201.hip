// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact Qwen3.5-27B dense-decode gate/up producer for gfx1201.
//
// Eight wave32 consumers share one K=5120 activation staged in LDS, then walk
// persistent output-row stripes.  The admitted paired-row kernel reloads the
// same activation once for every output row (normally as cache hits); this
// shape replaces those repeated vector-memory instructions with LDS reads
// while preserving the paired gate/up weight streams and arithmetic order.
//
// Grid size is selected by the host experiment.  No persistent residency or
// CU identity is assumed: (block, wave) workers cover rows by a grid-wide
// stride, so every positive grid size writes every output row exactly once.

extern "C" __launch_bounds__(256, 1)
__global__ void fused_gate_up_hfq4g256_persistent_k5120_global_gfx1201(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    (void)K;
    const int block_tid = threadIdx.x;
    const int wave = block_tid >> 5;
    const int tid = block_tid & 31;

    // Transposed [group][element-within-lane][lane] storage makes each
    // consumer wave's eight scalar LDS reads bank-contiguous.
    __shared__ float x_lds[5120];
    #pragma unroll
    for (int d = block_tid; d < 5120; d += 256) {
        const int group = d >> 8;
        const int j = (d >> 5) & 7;
        const int lane = d & 31;
        x_lds[d] = x[group * 256 + lane * 8 + j];
    }
    __syncthreads();

    constexpr unsigned int groups_per_row = 20;
    constexpr unsigned int group_bytes = 136;
    constexpr unsigned int row_bytes = groups_per_row * group_bytes;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto gate_rsrc = HIPFIRE_WEIGHT_RSRC(A_gate);
    const auto up_rsrc = HIPFIRE_WEIGHT_RSRC(A_up);
#endif
    const unsigned int boff = (unsigned int)tid * 4u;
    const int worker = (int)blockIdx.x * 8 + wave;
    const int worker_count = (int)gridDim.x * 8;

    #define PERSISTENT_PAIRED_DOG(group, gacc, uacc) do { \
        const unsigned int _g = (unsigned int)(group); \
        const unsigned int _group_offset = _g * group_bytes; \
        const unsigned int _gate_offset = gate_row_offset + _group_offset; \
        const unsigned int _up_offset = up_row_offset + _group_offset; \
        const char* _gg = gate_row + _group_offset; \
        const char* _ug = up_row + _group_offset; \
        const float _gsc = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32( \
            gate_rsrc, _gg, 0u, _gate_offset)); \
        const float _gzp = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32( \
            gate_rsrc, _gg + 4, 4u, _gate_offset)); \
        const unsigned int _gpk = HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32( \
            gate_rsrc, _gg + 8 + boff, boff + 8u, _gate_offset); \
        const float _usc = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32( \
            up_rsrc, _ug, 0u, _up_offset)); \
        const float _uzp = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32( \
            up_rsrc, _ug + 4, 4u, _up_offset)); \
        const unsigned int _upk = HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32( \
            up_rsrc, _ug + 8 + boff, boff + 8u, _up_offset); \
        const float _x0 = x_lds[_g * 256u + 0u * 32u + (unsigned int)tid]; \
        const float _x1 = x_lds[_g * 256u + 1u * 32u + (unsigned int)tid]; \
        const float _x2 = x_lds[_g * 256u + 2u * 32u + (unsigned int)tid]; \
        const float _x3 = x_lds[_g * 256u + 3u * 32u + (unsigned int)tid]; \
        const float _x4 = x_lds[_g * 256u + 4u * 32u + (unsigned int)tid]; \
        const float _x5 = x_lds[_g * 256u + 5u * 32u + (unsigned int)tid]; \
        const float _x6 = x_lds[_g * 256u + 6u * 32u + (unsigned int)tid]; \
        const float _x7 = x_lds[_g * 256u + 7u * 32u + (unsigned int)tid]; \
        (gacc) += (_gsc * (float)(_gpk & 0xFu) + _gzp) * _x0 \
                + (_gsc * (float)((_gpk >> 4) & 0xFu) + _gzp) * _x1 \
                + (_gsc * (float)((_gpk >> 8) & 0xFu) + _gzp) * _x2 \
                + (_gsc * (float)((_gpk >> 12) & 0xFu) + _gzp) * _x3 \
                + (_gsc * (float)((_gpk >> 16) & 0xFu) + _gzp) * _x4 \
                + (_gsc * (float)((_gpk >> 20) & 0xFu) + _gzp) * _x5 \
                + (_gsc * (float)((_gpk >> 24) & 0xFu) + _gzp) * _x6 \
                + (_gsc * (float)((_gpk >> 28) & 0xFu) + _gzp) * _x7; \
        (uacc) += (_usc * (float)(_upk & 0xFu) + _uzp) * _x0 \
                + (_usc * (float)((_upk >> 4) & 0xFu) + _uzp) * _x1 \
                + (_usc * (float)((_upk >> 8) & 0xFu) + _uzp) * _x2 \
                + (_usc * (float)((_upk >> 12) & 0xFu) + _uzp) * _x3 \
                + (_usc * (float)((_upk >> 16) & 0xFu) + _uzp) * _x4 \
                + (_usc * (float)((_upk >> 20) & 0xFu) + _uzp) * _x5 \
                + (_usc * (float)((_upk >> 24) & 0xFu) + _uzp) * _x6 \
                + (_usc * (float)((_upk >> 28) & 0xFu) + _uzp) * _x7; \
    } while (0)

    for (int row = worker; row < gate_m && row < up_m; row += worker_count) {
        const unsigned int gate_row_offset = (unsigned int)row * row_bytes;
        const unsigned int up_row_offset = (unsigned int)row * row_bytes;
        const char* gate_row = A_gate + gate_row_offset;
        const char* up_row = A_up + up_row_offset;
        float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
        float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;

        #pragma unroll 1
        for (int q = 0; q < 5; q++) {
            const int g = q << 2;
            PERSISTENT_PAIRED_DOG(g + 0, gacc0, uacc0);
            PERSISTENT_PAIRED_DOG(g + 1, gacc1, uacc1);
            PERSISTENT_PAIRED_DOG(g + 2, gacc2, uacc2);
            PERSISTENT_PAIRED_DOG(g + 3, gacc3, uacc3);
        }

        float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
        float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
        for (int offset = 16; offset > 0; offset >>= 1) {
            gacc += __shfl_down(gacc, offset);
            uacc += __shfl_down(uacc, offset);
        }
        if (tid == 0) {
            y_gate[row] = gacc;
            y_up[row] = uacc;
        }
    }

    #undef PERSISTENT_PAIRED_DOG
}
