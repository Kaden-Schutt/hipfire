// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 multi-wave same-row LDS qkvza for MQ4G256V2 (qt=44).
// Arithmetic / header / interleaved-C authority:
//   gemm_qkvza_mq4g256v2_wmma.hip  (base gfx11 half16 K2 kernel)
// Stage/compute structure authority:
//   gemm_mq4g256v2_residual_wmma_gfx1100_mw_lds.hip
//   gemm_gate_up_mq4g256v2_wmma_gfx1100_mw_lds.hip
//
// One workgroup owns a single 16-row global tile spanning qkv then z then
// beta then alpha (total_m = qkv_m+z_m+beta_m+alpha_m). NW waves own distinct
// adjacent 16-column N tiles. For each 256-weight group the block cooperatively
// stages the dequantized 16x256 half tile into static LDS (tile-major
// [kt][row][ki], 8 KiB), routing each safe global row to A_qkv/A_z/A_beta/
// A_alpha, then every wave reuses that A fragment against its private X columns.
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
// Block: [32*NW, 1, 1]. Static LDS 8192 B; dynamic shared_mem=0.
// Instantiated: mw4/mw8/mw12/mw16 (blocks 128/256/384/512, N tiles 64/128/192/256).
// Production exact gfx1100 N>=192 outside capture/replay: env HIPFIRE_MQ4V2_QKVZA_MW_LDS
// default-off selects 0/off/4/8/12/16. Overwrite (=) to four separate
// Y_qkv/Y_z/Y_beta/Y_alpha with correct strides; gfx11 interleaved C.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_QKVZA_MW_LDS(NW)                                                                 \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                               \
        gemm_qkvza_mq4g256v2_wmma_gfx1100_mw##NW##_lds(                                       \
            const char* __restrict__ A_qkv,                                                  \
            const char* __restrict__ A_z,                                                    \
            const char* __restrict__ A_beta,                                                 \
            const char* __restrict__ A_alpha,                                                \
            const _Float16* __restrict__ X,                                                  \
            float* __restrict__ Y_qkv,                                                       \
            float* __restrict__ Y_z,                                                         \
            float* __restrict__ Y_beta,                                                      \
            float* __restrict__ Y_alpha,                                                     \
            int qkv_m,                                                                       \
            int z_m,                                                                         \
            int beta_m,                                                                      \
            int alpha_m,                                                                     \
            int K,                                                                           \
            int N) {                                                                         \
        const int tid = threadIdx.x;                                                         \
        const int lane = tid & 31;                                                           \
        const int wave_id = tid >> 5;                                                        \
        const int ml = lane & 15;                                                            \
        const int total_m = qkv_m + z_m + beta_m + alpha_m;                                  \
        const int row_start = blockIdx.x * 16;                                               \
        const int n_block_start = blockIdx.y * (16 * (NW));                                  \
        const int out_col = n_block_start + wave_id * 16 + ml;                               \
        const int safe_batch = (out_col < N) ? out_col : 0;                                  \
        const int groups_per_row = K / 256;                                                  \
                                                                                             \
        /* Tile-major [kt][row][ki]: 16*16*16 half = 8192 B. */                              \
        __shared__ __align__(32) _Float16 w_lds[16][16][16];                                  \
                                                                                             \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                      \
                                                                                             \
        for (int g = 0; g < groups_per_row; g++) {                                           \
            /* 128 logical staging jobs: 16 rows x 8 segments. */                            \
            for (int job = tid; job < 128; job += 32 * (NW)) {                               \
                const int row = job / 8;                                                     \
                const int segment = job % 8;                                                 \
                const int safe_row =                                                         \
                    (row_start + row < total_m) ? (row_start + row) : (total_m - 1);          \
                const char* A;                                                               \
                int local_row;                                                               \
                if (safe_row < qkv_m) {                                                      \
                    A = A_qkv;                                                               \
                    local_row = safe_row;                                                    \
                } else if (safe_row < qkv_m + z_m) {                                         \
                    A = A_z;                                                                 \
                    local_row = safe_row - qkv_m;                                            \
                } else if (safe_row < qkv_m + z_m + beta_m) {                                \
                    A = A_beta;                                                              \
                    local_row = safe_row - (qkv_m + z_m);                                    \
                } else {                                                                     \
                    A = A_alpha;                                                             \
                    local_row = safe_row - (qkv_m + z_m + beta_m);                           \
                }                                                                            \
                const char* gp =                                                             \
                    A + (long long)local_row * groups_per_row * 136 + (long long)g * 136;     \
                /* One relevant dual-half V2 header for this segment's K half. */            \
                const unsigned int h =                                                       \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                    \
                const _Float16 sc_h =                                                        \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));  \
                const _Float16 zp_h =                                                        \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));      \
                /* Four u32 payload loads (16 B); group stride 136 alternates 16B align. */  \
                const unsigned int pk0 = *(const unsigned int*)(gp + 8 + segment * 16 + 0);  \
                const unsigned int pk1 = *(const unsigned int*)(gp + 8 + segment * 16 + 4);  \
                const unsigned int pk2 = *(const unsigned int*)(gp + 8 + segment * 16 + 8);  \
                const unsigned int pk3 = *(const unsigned int*)(gp + 8 + segment * 16 + 12); \
                const int kt0 = 2 * segment;                                                 \
                const int kt1 = kt0 + 1;                                                     \
                half16_t frag;                                                               \
                frag[0] = sc_h * (_Float16)(float)((pk0 >> 0) & 0xFu) + zp_h;                \
                frag[1] = sc_h * (_Float16)(float)((pk0 >> 4) & 0xFu) + zp_h;                \
                frag[2] = sc_h * (_Float16)(float)((pk0 >> 8) & 0xFu) + zp_h;                \
                frag[3] = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;               \
                frag[4] = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;               \
                frag[5] = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;               \
                frag[6] = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;               \
                frag[7] = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;               \
                frag[8] = sc_h * (_Float16)(float)((pk1 >> 0) & 0xFu) + zp_h;                \
                frag[9] = sc_h * (_Float16)(float)((pk1 >> 4) & 0xFu) + zp_h;                \
                frag[10] = sc_h * (_Float16)(float)((pk1 >> 8) & 0xFu) + zp_h;               \
                frag[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;              \
                frag[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;              \
                frag[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;              \
                frag[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;              \
                frag[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;              \
                *(half16_t*)(&w_lds[kt0][row][0]) = frag;                                    \
                frag[0] = sc_h * (_Float16)(float)((pk2 >> 0) & 0xFu) + zp_h;                \
                frag[1] = sc_h * (_Float16)(float)((pk2 >> 4) & 0xFu) + zp_h;                \
                frag[2] = sc_h * (_Float16)(float)((pk2 >> 8) & 0xFu) + zp_h;                \
                frag[3] = sc_h * (_Float16)(float)((pk2 >> 12) & 0xFu) + zp_h;               \
                frag[4] = sc_h * (_Float16)(float)((pk2 >> 16) & 0xFu) + zp_h;               \
                frag[5] = sc_h * (_Float16)(float)((pk2 >> 20) & 0xFu) + zp_h;               \
                frag[6] = sc_h * (_Float16)(float)((pk2 >> 24) & 0xFu) + zp_h;               \
                frag[7] = sc_h * (_Float16)(float)((pk2 >> 28) & 0xFu) + zp_h;               \
                frag[8] = sc_h * (_Float16)(float)((pk3 >> 0) & 0xFu) + zp_h;                \
                frag[9] = sc_h * (_Float16)(float)((pk3 >> 4) & 0xFu) + zp_h;                \
                frag[10] = sc_h * (_Float16)(float)((pk3 >> 8) & 0xFu) + zp_h;               \
                frag[11] = sc_h * (_Float16)(float)((pk3 >> 12) & 0xFu) + zp_h;              \
                frag[12] = sc_h * (_Float16)(float)((pk3 >> 16) & 0xFu) + zp_h;              \
                frag[13] = sc_h * (_Float16)(float)((pk3 >> 20) & 0xFu) + zp_h;              \
                frag[14] = sc_h * (_Float16)(float)((pk3 >> 24) & 0xFu) + zp_h;              \
                frag[15] = sc_h * (_Float16)(float)((pk3 >> 28) & 0xFu) + zp_h;              \
                *(half16_t*)(&w_lds[kt1][row][0]) = frag;                                    \
            }                                                                                \
            /* All threads reach both barriers every group — no early returns. */            \
            __syncthreads();                                                                 \
                                                                                             \
            /* K2 ascending: kt pairs 0,1; 2,3; ... 14,15. One float8 acc / wave. */         \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;         \
            _Pragma("unroll")                                                                \
            for (int kt = 0; kt < 16; kt += 2) {                                             \
                half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]);                      \
                half16_t b_reg = *(const half16_t*)(xg + kt * 16);                           \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);         \
                a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]);                           \
                b_reg = *(const half16_t*)(xg + (kt + 1) * 16);                              \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);         \
            }                                                                                \
                                                                                             \
            __syncthreads();                                                                 \
        }                                                                                    \
                                                                                             \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. Overwrite (=). */      \
        if (out_col < N) {                                                                   \
            _Pragma("unroll")                                                                \
            for (int j = 0; j < 8; j++) {                                                    \
                const int out_row = row_start + 2 * j + (lane >> 4);                          \
                if (out_row < total_m) {                                                     \
                    float* Y;                                                                \
                    int proj_row;                                                            \
                    int out_stride;                                                          \
                    if (out_row < qkv_m) {                                                   \
                        Y = Y_qkv;                                                           \
                        proj_row = out_row;                                                  \
                        out_stride = qkv_m;                                                  \
                    } else if (out_row < qkv_m + z_m) {                                      \
                        Y = Y_z;                                                             \
                        proj_row = out_row - qkv_m;                                          \
                        out_stride = z_m;                                                    \
                    } else if (out_row < qkv_m + z_m + beta_m) {                             \
                        Y = Y_beta;                                                          \
                        proj_row = out_row - (qkv_m + z_m);                                  \
                        out_stride = beta_m;                                                 \
                    } else {                                                                 \
                        Y = Y_alpha;                                                         \
                        proj_row = out_row - (qkv_m + z_m + beta_m);                         \
                        out_stride = alpha_m;                                                \
                    }                                                                        \
                    Y[(long long)out_col * out_stride + proj_row] = acc[j];                  \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
    }

GEN_QKVZA_MW_LDS(4)
GEN_QKVZA_MW_LDS(8)
GEN_QKVZA_MW_LDS(12)
GEN_QKVZA_MW_LDS(16)
