// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1201 multi-wave same-row LDS gate+up for MQ4G256V2 (qt=44).
// Arithmetic / header / contiguous-C authority:
//   gemm_gate_up_mq4g256v2_wmma.gfx12.hip  (base gfx12 half8 K4 kernel)
// Stage structure authority:
//   gemm_gate_up_mq4g256v2_wmma_gfx1100_mw_lds.hip
//
// One workgroup owns a single 16-row global tile spanning gate then up
// (total_m = gate_m + up_m). NW waves own distinct adjacent 16-column N
// tiles. For each 256-weight group the block cooperatively stages the
// dequantized 16x256 half weight tile into static LDS (tile-major
// [kt][row][ki], 8 KiB = 4096 fp16), routing each safe global row to
// A_gate or A_up, then every wave reuses that A fragment against its
// private X columns loaded from global memory.
//
// Staging: 128 logical jobs independent of MW
// (job=tid; job<128; job+=32*NW). job row=job/8, segment=job%8. Each job
// owns 32 adjacent weights (two K-tiles) via four aligned u32 payload
// loads and one dual-half V2 header (gp+0 for segment<4, gp+4 for
// segment>=4). Exact authority dequant:
//   sc_h * (_Float16)(float)(nibble) + zp_h
// Store two complete K tiles tile-major. All 32*NW threads execute both
// __syncthreads per group; no early returns. M tails duplicate row
// total_m-1 into invalid LDS rows (discarded by out_row<total_m). Partial
// N lanes and whole tail waves read X batch 0 and discard via out_col<N.
//
// After barrier: each wave loads A half8 at [kt][m_lane][k_grp*8], loads
// wave-private B half8 directly from global X, issues gfx12 WMMA in exact
// K4 order (kt 0..15 step 4). Output: overwrite (=) to separate Y_gate /
// Y_up with correct strides; gfx12 contiguous C map row=8*k_grp+j.
//
// Grid:  [ceil(total_m/16), ceil(N/(16*NW)), 1]
// Block: [32*NW, 1, 1]. Static LDS 8192 B; dynamic shared_mem=0.
// Instantiated: mw4/mw8/mw12 (blocks 128/256/384, N tiles 64/128/192).
//
// Compile guard: exact-gfx1201. This TU must be compiled ONLY with
//   hipcc --offload-arch=gfx1201 (defines __gfx1201__). Family macros
//   __gfx12__ are NOT sufficient — gfx1200 would select the wrong WMMA path.
//
// ABI / Symbols (composer-wireable, production-neutral):
//   gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw4
//   gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw8
//   gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw12
//   Args: (A_gate, A_up, X, Y_gate, Y_up, gate_m, up_m, K, N)

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "exact-gfx1201: gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds.hip requires --offload-arch=gfx1201"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_GATE_UP_MW_LDS(NW)                                                                 \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                                 \
        gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw##NW(                                     \
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
        const int m_lane = lane & 15;                                                          \
        const int k_grp = lane >> 4;                                                           \
        const int total_m = gate_m + up_m;                                                     \
        const int row_start = blockIdx.x * 16;                                                 \
        const int n_block_start = blockIdx.y * (16 * (NW));                                    \
        const int out_col = n_block_start + wave_id * 16 + m_lane;                             \
        const int safe_batch = (out_col < N) ? out_col : 0;                                    \
        const int groups_per_row = K / 256;                                                    \
                                                                                               \
        /* Dequantized weight LDS, tile-major [kt][row][ki]: 16*16*16 half = 8192 B. */       \
        __shared__ __align__(32) _Float16 w_lds[16][16][16];                                    \
                                                                                               \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                        \
                                                                                               \
        for (int g = 0; g < groups_per_row; g++) {                                             \
            /* 128 logical staging jobs: 16 rows × 8 segments (independent of NW). */          \
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
                /* One dual-half V2 header for this segment's K half. */                       \
                const unsigned int h =                                                         \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                      \
                const _Float16 sc_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));    \
                const _Float16 zp_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));        \
                /* Four u32 payload loads (16 B); two complete K tiles. */                     \
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
            /* A from dequant LDS half8; B from global X; exact K4 order. */                   \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;           \
            _Pragma("unroll")                                                                  \
            for (int kt = 0; kt < 16; kt += 4) {                                               \
                half8_t a_a = *(const half8_t*)(&w_lds[kt + 0][m_lane][k_grp * 8]);            \
                half8_t a_b = *(const half8_t*)(&w_lds[kt + 1][m_lane][k_grp * 8]);            \
                half8_t a_c = *(const half8_t*)(&w_lds[kt + 2][m_lane][k_grp * 8]);            \
                half8_t a_d = *(const half8_t*)(&w_lds[kt + 3][m_lane][k_grp * 8]);            \
                half8_t b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);               \
                half8_t b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);               \
                half8_t b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);               \
                half8_t b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);               \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_a, b_a, acc);         \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_b, b_b, acc);         \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_c, b_c, acc);         \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_d, b_d, acc);         \
            }                                                                                  \
                                                                                               \
            __syncthreads();                                                                   \
        }                                                                                      \
                                                                                               \
        /* gfx12 wave32 WMMA: acc[j] = C[8*k_grp + j][m_lane]. Overwrite (=). */               \
        if (out_col < N) {                                                                     \
            _Pragma("unroll")                                                                  \
            for (int j = 0; j < 8; j++) {                                                      \
                const int out_row = row_start + 8 * k_grp + j;                                  \
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
GEN_GATE_UP_MW_LDS(12)

#undef GEN_GATE_UP_MW_LDS
