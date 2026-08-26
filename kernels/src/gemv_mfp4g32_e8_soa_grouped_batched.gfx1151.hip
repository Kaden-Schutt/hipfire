// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — batched grouped mfp4-E8 SoA decode-GEMV for small-batch
// (speculative verify) forwards on gfx1151.
//
//   A[G, M, K] @ X[B, G, K] -> Y[B, G, M]
//
// One wave owns one (group, row) pair, reads that weight row EXACTLY ONCE,
// and applies it to `E8_GROUPED_BATCHED_B` activation vectors. Layout, scale
// conversion and lattice decode are copied verbatim from
// `gemv_mfp4g32_e8_soa_grouped`, so B=1 is bit-identical to the decode path.
//
// Why this exists. `wo_per_group_batched_e8_fallback` — the batched O-LoRA A
// projection — routes gfx1151 E8-SoA through
// `gemm_mfp4g32_e8_soa_grouped_wmma`, a grouped block-diagonal WMMA launch
// that tiles the TOKEN axis at 16. Below B=16 it computes a full 16-token
// tile regardless of how many tokens are real. Profiled on the ds4 gfx1151
// route at B=1 (100 iterations, prefix prefill differenced out), that single
// substitution is +10.38 ms/token — 4.23 ms on the decode grouped GEMV
// against 14.61 ms on the grouped WMMA — and is the largest remaining term in
// the 22.4 ms gap between a B=1 batched forward and an AR decode step.
//
// This is the same fix already applied to the plain dense tier in
// `gemv_mfp4g32_e8_soa_batched.gfx1151.hip`, in the grouped tier. The
// accumulation order differs between the two: the plain decode GEMV strides
// four accumulators across four groups, the grouped decode GEMV strides TWO
// across a pair. Bit-exactness is against whichever decode kernel the path
// actually replaces, so this file mirrors the pair stride.

#include <hip/hip_runtime.h>

#ifndef E8_GROUPED_BATCHED_B
#define E8_GROUPED_BATCHED_B 6
#endif
#ifndef E8_GROUPED_BATCHED_KERNEL
#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_gfx1151
#endif

__device__ __forceinline__ float cvt_e4m3_scale_grouped_batched(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index_grouped_batched(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sum_low = 0u;
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sum_low += e[i];
    }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int e7_even = e7_high << 1;
    e[7] = e7_even | ((sum_low + e7_even) & 1u);
    for (int i = 0; i < 8; i++) {
        const float coordinate = (float)((int)e[i] - 7);
        p[i] = coset ? coordinate + 0.5f : coordinate;
    }
}

// A: E8-SoA [G, M, K].  X: f32 [B, G, K].  Y: f32 [B, G, M].
__launch_bounds__(32)
extern "C" __global__ void E8_GROUPED_BATCHED_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int G,
    int M,
    int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M || group >= G) return;
    const int tid = threadIdx.x;

    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const long long global_row = (long long)group * M + row;
    const char* hdr = A + global_row * row_total;

    const float row_scale =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const char* scale_base = hdr + 16;
    const char* code_base = hdr + 16 + scale_padded;
    const int block_in_group = tid >> 2;
    const int code_slot = tid & 3;

    // X and Y strides between tokens. Each token carries the full G groups,
    // so token b's slice for THIS group starts at b*G*K + group*K.
    const long long x_token_stride = (long long)G * K;
    const long long y_token_stride = (long long)G * M;
    const float* x_base = X + (long long)group * K;

    // Two accumulators PER TOKEN, strided across a group pair, mirroring
    // `gemv_mfp4g32_e8_soa_grouped` operation-for-operation: group g -> acc0,
    // g+1 -> acc1, odd trailing group -> acc0, final combine acc0 + acc1.
    // Float addition does not reassociate, so a single sequential accumulator
    // would be numerically equivalent but NOT bit-identical — and this kernel
    // feeds speculative verify, whose logits decide token acceptance.
    //
    // B is a compile-time constant so these stay in registers; a runtime bound
    // would force a dynamically indexed array into scratch.
    float acc0[E8_GROUPED_BATCHED_B], acc1[E8_GROUPED_BATCHED_B];
#pragma unroll
    for (int b = 0; b < E8_GROUPED_BATCHED_B; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
    }

    const int pairs = groups_per_row >> 1;
    for (int pair = 0; pair < pairs; pair++) {
        const int group0 = pair << 1;
        const int block0 = group0 * 8 + block_in_group;
        const int block1 = block0 + 8;

        const float scale0 = row_scale
            * cvt_e4m3_scale_grouped_batched(*(const unsigned char*)(scale_base + block0))
            * 0.88f;
        const float scale1 = row_scale
            * cvt_e4m3_scale_grouped_batched(*(const unsigned char*)(scale_base + block1))
            * 0.88f;

        const unsigned int code0 =
            *(const unsigned int*)(code_base + (long long)block0 * 16 + code_slot * 4);
        const unsigned int code1 =
            *(const unsigned int*)(code_base + (long long)block1 * 16 + code_slot * 4);

        const int base0 = group0 * 256 + tid * 8;
        const int base1 = base0 + 256;

        // ONE reused decode buffer, not two. Splitting costs 8 VGPRs and was
        // measured perf-null on the plain decode kernel (LLVM scalar-replaces
        // the array, so the apparent write-after-read hazard never existed).
        // Numerics are unchanged: each group's accumulate consumes `points`
        // immediately after its own decode.
        float points[8];

        e8_decode_index_grouped_batched(code0, points);
#pragma unroll
        for (int b = 0; b < E8_GROUPED_BATCHED_B; b++) {
            const float* xb = x_base + (long long)b * x_token_stride;
            acc0[b] +=
                scale0 * points[0] * xb[base0]
                + scale0 * points[1] * xb[base0 + 1]
                + scale0 * points[2] * xb[base0 + 2]
                + scale0 * points[3] * xb[base0 + 3]
                + scale0 * points[4] * xb[base0 + 4]
                + scale0 * points[5] * xb[base0 + 5]
                + scale0 * points[6] * xb[base0 + 6]
                + scale0 * points[7] * xb[base0 + 7];
        }

        e8_decode_index_grouped_batched(code1, points);
#pragma unroll
        for (int b = 0; b < E8_GROUPED_BATCHED_B; b++) {
            const float* xb = x_base + (long long)b * x_token_stride;
            acc1[b] +=
                scale1 * points[0] * xb[base1]
                + scale1 * points[1] * xb[base1 + 1]
                + scale1 * points[2] * xb[base1 + 2]
                + scale1 * points[3] * xb[base1 + 3]
                + scale1 * points[4] * xb[base1 + 4]
                + scale1 * points[5] * xb[base1 + 5]
                + scale1 * points[6] * xb[base1 + 6]
                + scale1 * points[7] * xb[base1 + 7];
        }
    }

    // Odd trailing group, into acc0 (matches the decode kernel).
    if (groups_per_row & 1) {
        const int group0 = pairs << 1;
        const int block = group0 * 8 + block_in_group;
        const float scale = row_scale
            * cvt_e4m3_scale_grouped_batched(*(const unsigned char*)(scale_base + block))
            * 0.88f;
        const unsigned int code =
            *(const unsigned int*)(code_base + (long long)block * 16 + code_slot * 4);
        const int base = group0 * 256 + tid * 8;
        float points[8];
        e8_decode_index_grouped_batched(code, points);
#pragma unroll
        for (int b = 0; b < E8_GROUPED_BATCHED_B; b++) {
            const float* xb = x_base + (long long)b * x_token_stride;
            acc0[b] +=
                scale * points[0] * xb[base]
                + scale * points[1] * xb[base + 1]
                + scale * points[2] * xb[base + 2]
                + scale * points[3] * xb[base + 3]
                + scale * points[4] * xb[base + 4]
                + scale * points[5] * xb[base + 5]
                + scale * points[6] * xb[base + 6]
                + scale * points[7] * xb[base + 7];
        }
    }

#pragma unroll
    for (int b = 0; b < E8_GROUPED_BATCHED_B; b++) {
        float acc = acc0[b] + acc1[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down(acc, offset);
        }
        if (tid == 0) {
            Y[(long long)b * y_token_stride + global_row] = acc;
        }
    }
}
