// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated prefill GEMM for TQ2G128 (PrismML Q2_0 ternary) weights
// + F32 activations -> F32 output. gfx1100+ / gfx1200+ (RDNA3/3.5/4 wave32 WMMA).
//
// Layout:
//   A : TQ2G128-packed [M, K/128 x 34] bytes  ([f16 d][32 B codes] per group)
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
// The ternary layout falls out unusually cleanly: codes pack LSB-first, 4 per
// byte, so 4 consecutive code bytes hold exactly 16 codes -- one whole K=16
// WMMA sub-tile per u32, with code i at bit 2*i. (The bytes are read
// individually and recombined because a group is 34 B, so the code array sits
// at offset 2 mod 4 and a direct u32 load would be misaligned.)
//
// gfx12/RDNA4 WMMA differs from gfx11: half8 operands, K split by
// k_grp = tid >> 4 (each lane owns 8 of 16 K elements), builtin
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12, and contiguous-per-half
// C rows: acc[j] -> row_start + 8*k_grp + j. gfx11 keeps half16, the
// unsuffixed builtin, and interleaved rows row_start + 2*j + (tid >> 4).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
typedef _Float16 __attribute__((ext_vector_type(8)))  half8_t;
#else
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
#endif
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
// the vector LDS reads stay 32 B aligned, and 288/4 = 72 dwords with
// 72 % 32 = 8, so consecutive token rows land 8 banks apart instead of
// colliding (a bare 128 would be 0 mod 32 -- every lane on one bank).
#define LB_XSTRIDE 144

__launch_bounds__(32 * LB_WAVES, 2)
extern "C" __global__ void gemm_tq2g128_wmma(
    const char*     __restrict__ A,    // TQ2G128 packed
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
    const long long row_stride = (long long)groups_per_row * 34;

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

        const char* gp = a_row + g * 34;
        const _Float16 d = *(const _Float16*)gp;
        const unsigned char* qs = (const unsigned char*)(gp + 2);

        #pragma unroll 1
        for (int kt = 0; kt < 8; kt++) {
            const unsigned char* q = qs + kt * 4;
            const unsigned int pk = (unsigned int)q[0]
                                  | ((unsigned int)q[1] << 8)
                                  | ((unsigned int)q[2] << 16)
                                  | ((unsigned int)q[3] << 24);

#if defined(__gfx1200__) || defined(__gfx1201__)
            // gfx12: each lane-group owns 8 of the 16 ternary codes in pk.
            const unsigned int pk_half = pk >> (k_grp * 16);
            half8_t a_frag;
            #define DQ(i) a_frag[i] = d * (_Float16)((int)((pk_half >> (2 * (i))) & 0x3u) - 1)
            DQ(0);  DQ(1);  DQ(2);  DQ(3);
            DQ(4);  DQ(5);  DQ(6);  DQ(7);
            #undef DQ

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
            #define DQ(i) a_frag[i] = d * (_Float16)((int)((pk >> (2 * (i))) & 0x3u) - 1)
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
