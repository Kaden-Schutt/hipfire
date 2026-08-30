// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 multi-wave same-row LDS gate+up for MQ4G256V2 (qt=44).
// Arithmetic / header / interleaved-C authority:
//   gemm_gate_up_mq4g256v2_wmma.hip  (base gfx11 half16 K2 kernel)
// Stage/compute structure authority:
//   gemm_mq4g256v2_residual_wmma_gfx1100_mw_lds.hip
//
// One workgroup owns a single 16-row global tile spanning gate then up
// (total_m = gate_m + up_m). NW waves own distinct adjacent 16-column N
// tiles. For each 256-weight group the block cooperatively stages the
// dequantized 16x256 half tile into static LDS (tile-major [kt][row][ki],
// 8 KiB), routing each safe global row to A_gate or A_up, then every wave
// reuses that A fragment against its private X columns.
//
// Staging: 128 logical jobs in a strided loop (job=tid; job<128; job+=32*NW).
// job row=job/8, segment=job%8. Each job owns 32 adjacent weights (two K-tiles)
// via four aligned u32 payload loads and one dual-half V2 header (gp+0 for
// segment<4, gp+4 for segment>=4). Exact authority dequant:
//   sc_h * (_Float16)(float)(nibble) + zp_h
// All 32*NW threads execute both __syncthreads per group; no early returns.
// M tails duplicate row total_m-1 into invalid LDS rows (discarded by
// out_row<total_m). Partial N lanes and whole tail waves read X batch 0 and
// discard via out_col<N.
//
// Grid:  [ceil(total_m/16), ceil(N/(16*NW)), 1]
// Block: [32*NW, 1, 1]. Static LDS 9216 B; dynamic shared_mem=0.
// Instantiated: mw4/mw8 (blocks 128/256, N tiles 64/128).
// Production exact gfx1100 N>=384 outside capture/replay: tiles64=ceil(N/64);
// even → MW8 (same padded extent as MW4, more reuse), odd → MW4 (avoids
// MW8's extra 64 padded columns). Below 384 retains BT6. Output: overwrite
// (=) to separate Y_gate / Y_up with correct strides; gfx11 interleaved C.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_GATE_UP_MW_LDS(NW)                                                                 \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                                 \
        gemm_gate_up_mq4g256v2_wmma_gfx1100_mw##NW##_lds(                                      \
            const char* __restrict__ A_gate,                                                   \
            const char* __restrict__ A_up,                                                     \
            const _Float16* __restrict__ X,                                                    \
            float* __restrict__ Y_gate,                                                        \
            float* __restrict__ Y_up,                                                          \
            int gate_m,                                                                        \
            int up_m,                                                                          \
            int K,                                                                             \
            int N) {                                                                           \
        const int tid = threadIdx.x;                                                           \
        const int lane = tid & 31;                                                             \
        const int wave_id = tid >> 5;                                                          \
        const int ml = lane & 15;                                                              \
        const int total_m = gate_m + up_m;                                                     \
        const int row_start = blockIdx.x * 16;                                                 \
        const int n_block_start = blockIdx.y * (16 * (NW));                                    \
        const int out_col = n_block_start + wave_id * 16 + ml;                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                    \
        const int groups_per_row = K / 256;                                                    \
                                                                                               \
        /* Tile-major [kt][row][ki], ki padded to 18: 16*16*18 half = 9216 B.                  \
           36 B row stride, gcd(9,32)=1 — bank-conflict-free compute reads                     \
           (tools/kernels/gfx1100/mb_lds_pad18 screen, 2026-08-30: 1.52x probe). */            \
        __shared__ __align__(32) _Float16 w_lds[16][16][18];                                    \
                                                                                               \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                        \
                                                                                               \
        for (int g = 0; g < groups_per_row; g++) {                                             \
            /* 128 logical staging jobs: 16 rows × 8 segments. */                              \
            for (int job = tid; job < 128; job += 32 * (NW)) {                                 \
                const int row = job / 8;                                                       \
                const int segment = job % 8;                                                   \
                const int safe_row =                                                           \
                    (row_start + row < total_m) ? (row_start + row) : (total_m - 1);            \
                const char* A;                                                                 \
                int local_row;                                                                 \
                if (safe_row < gate_m) {                                                       \
                    A = A_gate;                                                                \
                    local_row = safe_row;                                                      \
                } else {                                                                       \
                    A = A_up;                                                                  \
                    local_row = safe_row - gate_m;                                             \
                }                                                                              \
                const char* gp =                                                               \
                    A + (long long)local_row * groups_per_row * 136 + (long long)g * 136;       \
                /* One relevant dual-half V2 header for this segment's K half. */              \
                const unsigned int h =                                                         \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                      \
                const _Float16 sc_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));    \
                const _Float16 zp_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));        \
                /* Four u32 payload loads (16 B); group stride 136 alternates 16B align. */    \
                const unsigned int pk0 = *(const unsigned int*)(gp + 8 + segment * 16 + 0);    \
                const unsigned int pk1 = *(const unsigned int*)(gp + 8 + segment * 16 + 4);    \
                const unsigned int pk2 = *(const unsigned int*)(gp + 8 + segment * 16 + 8);    \
                const unsigned int pk3 = *(const unsigned int*)(gp + 8 + segment * 16 + 12);   \
                const int kt0 = 2 * segment;                                                   \
                const int kt1 = kt0 + 1;                                                       \
                half16_t frag;                                                                 \
                frag[0] = sc_h * (_Float16)(float)((pk0 >> 0) & 0xFu) + zp_h;                  \
                frag[1] = sc_h * (_Float16)(float)((pk0 >> 4) & 0xFu) + zp_h;                  \
                frag[2] = sc_h * (_Float16)(float)((pk0 >> 8) & 0xFu) + zp_h;                  \
                frag[3] = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;                 \
                frag[4] = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;                 \
                frag[5] = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;                 \
                frag[6] = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;                 \
                frag[7] = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;                 \
                frag[8] = sc_h * (_Float16)(float)((pk1 >> 0) & 0xFu) + zp_h;                  \
                frag[9] = sc_h * (_Float16)(float)((pk1 >> 4) & 0xFu) + zp_h;                  \
                frag[10] = sc_h * (_Float16)(float)((pk1 >> 8) & 0xFu) + zp_h;                 \
                frag[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;                \
                frag[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;                \
                frag[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;                \
                frag[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;                \
                frag[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;                \
                *(half16_t*)(&w_lds[kt0][row][0]) = frag;                                      \
                frag[0] = sc_h * (_Float16)(float)((pk2 >> 0) & 0xFu) + zp_h;                  \
                frag[1] = sc_h * (_Float16)(float)((pk2 >> 4) & 0xFu) + zp_h;                  \
                frag[2] = sc_h * (_Float16)(float)((pk2 >> 8) & 0xFu) + zp_h;                  \
                frag[3] = sc_h * (_Float16)(float)((pk2 >> 12) & 0xFu) + zp_h;                 \
                frag[4] = sc_h * (_Float16)(float)((pk2 >> 16) & 0xFu) + zp_h;                 \
                frag[5] = sc_h * (_Float16)(float)((pk2 >> 20) & 0xFu) + zp_h;                 \
                frag[6] = sc_h * (_Float16)(float)((pk2 >> 24) & 0xFu) + zp_h;                 \
                frag[7] = sc_h * (_Float16)(float)((pk2 >> 28) & 0xFu) + zp_h;                 \
                frag[8] = sc_h * (_Float16)(float)((pk3 >> 0) & 0xFu) + zp_h;                  \
                frag[9] = sc_h * (_Float16)(float)((pk3 >> 4) & 0xFu) + zp_h;                  \
                frag[10] = sc_h * (_Float16)(float)((pk3 >> 8) & 0xFu) + zp_h;                 \
                frag[11] = sc_h * (_Float16)(float)((pk3 >> 12) & 0xFu) + zp_h;                \
                frag[12] = sc_h * (_Float16)(float)((pk3 >> 16) & 0xFu) + zp_h;                \
                frag[13] = sc_h * (_Float16)(float)((pk3 >> 20) & 0xFu) + zp_h;                \
                frag[14] = sc_h * (_Float16)(float)((pk3 >> 24) & 0xFu) + zp_h;                \
                frag[15] = sc_h * (_Float16)(float)((pk3 >> 28) & 0xFu) + zp_h;                \
                *(half16_t*)(&w_lds[kt1][row][0]) = frag;                                      \
            }                                                                                  \
            /* All threads reach both barriers every group — no early returns. */              \
            __syncthreads();                                                                   \
                                                                                               \
            /* K2 ascending: kt pairs 0,1; 2,3; ... 14,15. One float8 acc / wave. */           \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;           \
            _Pragma("unroll")                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                               \
                half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]);                        \
                half16_t b_reg = *(const half16_t*)(xg + kt * 16);                             \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
                a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]);                             \
                b_reg = *(const half16_t*)(xg + (kt + 1) * 16);                                \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
            }                                                                                  \
                                                                                               \
            __syncthreads();                                                                   \
        }                                                                                      \
                                                                                               \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. Overwrite (=). */        \
        if (out_col < N) {                                                                     \
            _Pragma("unroll")                                                                  \
            for (int j = 0; j < 8; j++) {                                                      \
                const int out_row = row_start + 2 * j + (lane >> 4);                            \
                if (out_row < total_m) {                                                       \
                    if (out_row < gate_m) {                                                    \
                        Y_gate[(long long)out_col * gate_m + out_row] = acc[j];                \
                    } else {                                                                   \
                        Y_up[(long long)out_col * up_m + (out_row - gate_m)] = acc[j];         \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
    }

GEN_GATE_UP_MW_LDS(4)
GEN_GATE_UP_MW_LDS(8)
