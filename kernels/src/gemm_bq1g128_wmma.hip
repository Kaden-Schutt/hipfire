// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated prefill GEMM for BQ1G128 (PrismML Q1_0 binary) weights
// + F32 activations -> F32 output. gfx1100+ (RDNA3/3.5/4 wave32 WMMA).
//
// Layout:
//   A : BQ1G128-packed [M, K/128 x 18] bytes  ([f16 d][16 B codes] per group)
//   X : [B, K] F32 row-major (staged to F16 in LDS)
//   Y : [B, M] F32 row-major
//
// Grid:  [ceil(M / (16*WAVES)), ceil(B / (16*BT))]. Block: [32*WAVES].
//
// WHY LDS. The first version copied gemm_hfq4g256_wmma.hip, which reads X
// straight from global with one wave per 16-row tile. That is right for
// decode-shaped B, but at prefill B it loses: every 16-row block re-reads the
// whole x tile, so per-row cost grows with M -- measured 0.54 us/row at
// M=5120 against 1.16 us/row at M=17408, where mq4's fused prefill kernel
// holds 0.39 us/row. The working set was BT*16 tokens x K = 1.25 MB per
// workgroup against mq4's 160 KB, so it thrashed cache.
//
// The fix is to stop making weight reuse and activation reuse compete. BT
// batch-tiles per wave buy weight reuse; WAVES row-tiles per BLOCK, sharing
// one LDS-staged x tile, buy activation reuse. x is now read from global once
// per (16*WAVES) rows instead of once per 16.
//
// Layout difference from the ternary sibling: 8 codes per byte, so ONE K=16
// WMMA sub-tile is 2 code bytes, and the decode is bit i of a 16-bit word
// selecting +-d.
//
// gfx12 vs gfx11 fragment path (same kernel symbol):
//   gfx12  A/X are half8; k_grp = tid >> 4 selects K half of the 16-wide tile
//          (load X at kt*16 + k_grp*8; decode the matching 8 bits of pk);
//          WMMA is _w32_gfx12; acc row is row_start + 8*k_grp + j.
//   gfx11  A/X are half16; every lane carries the full K=16 tile; WMMA is
//          unsuffixed _w32; acc row is row_start + 2*j + (tid >> 4).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(8)))  half8_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// Batch-tiles per wave (weight reuse): a 16-wide batch tile makes each weight
// row get re-read ceil(B/16) times.
#ifndef HIPFIRE_LOWBIT_WMMA_BT
#define HIPFIRE_LOWBIT_WMMA_BT 4
#endif
// Row-tiles per block (activation reuse), one per wave.
#ifndef HIPFIRE_LOWBIT_WMMA_WAVES
#define HIPFIRE_LOWBIT_WMMA_WAVES 16
#endif

#define LB_BT     HIPFIRE_LOWBIT_WMMA_BT
#define LB_WAVES  HIPFIRE_LOWBIT_WMMA_WAVES
#define LB_TOKENS (16 * LB_BT)
// 144 halfs = 288 B per staged token row. Must be a multiple of 16 halfs so
// the half16_t / half8_t LDS reads stay 32 B / 16 B aligned, and 288/4 = 72
// dwords with 72 % 32 = 8, so consecutive token rows land 8 banks apart
// instead of colliding (a bare 128 would be 0 mod 32 -- every lane on one
// bank).
#define LB_XSTRIDE 144

__launch_bounds__(32 * LB_WAVES, 2)
extern "C" __global__ void gemm_bq1g128_wmma(
    const char*     __restrict__ A,    // BQ1G128 packed
    const float*    __restrict__ X,    // [B, K] F32
    float*          __restrict__ Y,    // [B, M] F32
    int M, int K, int B
) {
    const int wave = threadIdx.x >> 5;
    const int tid  = threadIdx.x & 31;
    const int lane = tid & 15;
#if defined(__gfx1200__) || defined(__gfx1201__)
    const int k_grp = tid >> 4;
#endif

    const int row_start   = blockIdx.x * (16 * LB_WAVES) + wave * 16;
    const int batch_start = blockIdx.y * LB_TOKENS;
    if (batch_start >= B) return;

    const int groups_per_row = K / 128;
    const long long row_stride = (long long)groups_per_row * 18;

    // Rows past M are clamped; their outputs are discarded at the store, so
    // they cost work but never corrupt. The wave must NOT early-return: it
    // still has to reach __syncthreads() for the cooperative staging.
    const int my_m   = row_start + lane;
    const int safe_m = (my_m < M) ? my_m : (M - 1);
    const char* a_row = A + (long long)safe_m * row_stride;

    __shared__ _Float16 x_lds[LB_TOKENS * LB_XSTRIDE];

    float8_t acc[LB_BT];
    #pragma unroll
    for (int bt = 0; bt < LB_BT; bt++) {
        acc[bt] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative stage of this group's x slice: LB_TOKENS x 128, F32 ->
        // F16. Every wave in the block consumes it, which is the point.
        __syncthreads();
        for (int i = threadIdx.x; i < LB_TOKENS * 128; i += 32 * LB_WAVES) {
            const int t  = i >> 7;       // token within the tile
            const int kk = i & 127;      // element within the group
            const int n  = batch_start + t;
            x_lds[t * LB_XSTRIDE + kk] =
                (n < B) ? (_Float16)X[(long long)n * K + g * 128 + kk] : (_Float16)0.0f;
        }
        __syncthreads();

        const char* gp = a_row + g * 18;
        const _Float16 d = *(const _Float16*)gp;
        const _Float16 nd = -d;
        const unsigned char* qs = (const unsigned char*)(gp + 2);

        #pragma unroll 1
        for (int kt = 0; kt < 8; kt++) {
            const unsigned char* q = qs + kt * 2;
            const unsigned int pk = (unsigned int)q[0] | ((unsigned int)q[1] << 8);

#if defined(__gfx1200__) || defined(__gfx1201__)
            half8_t a_frag;
            #define DQ8(i) a_frag[i] = ((pk >> (k_grp * 8 + (i))) & 1u) ? d : nd
            DQ8(0); DQ8(1); DQ8(2); DQ8(3);
            DQ8(4); DQ8(5); DQ8(6); DQ8(7);
            #undef DQ8

            #pragma unroll
            for (int bt = 0; bt < LB_BT; bt++) {
                if (batch_start + bt * 16 >= B) break;
                const half8_t x_frag = *(const half8_t*)
                    &x_lds[(bt * 16 + lane) * LB_XSTRIDE + kt * 16 + k_grp * 8];
                acc[bt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_frag, x_frag, acc[bt]);
            }
#else
            half16_t a_frag;
            #define DQ(i) a_frag[i] = ((pk >> (i)) & 1u) ? d : nd
            DQ(0);  DQ(1);  DQ(2);  DQ(3);
            DQ(4);  DQ(5);  DQ(6);  DQ(7);
            DQ(8);  DQ(9);  DQ(10); DQ(11);
            DQ(12); DQ(13); DQ(14); DQ(15);
            #undef DQ

            #pragma unroll
            for (int bt = 0; bt < LB_BT; bt++) {
                if (batch_start + bt * 16 >= B) break;
                const half16_t x_frag = *(const half16_t*)
                    &x_lds[(bt * 16 + lane) * LB_XSTRIDE + kt * 16];
                acc[bt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_frag, x_frag, acc[bt]);
            }
#endif
        }
    }

#if defined(__gfx1200__) || defined(__gfx1201__)
    // gfx12: row = 8*k_grp + j, col (= batch) = tid & 15.
    #pragma unroll
    for (int bt = 0; bt < LB_BT; bt++) {
        const int out_col = batch_start + bt * 16 + lane;
        if (out_col >= B) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[bt][j];
            }
        }
    }
#else
    // gfx11: row = 2*j + (tid>>4), col (= batch) = tid & 15.
    #pragma unroll
    for (int bt = 0; bt < LB_BT; bt++) {
        const int out_col = batch_start + bt * 16 + lane;
        if (out_col >= B) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[bt][j];
            }
        }
    }
#endif
}
