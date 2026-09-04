// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 split-K LDS residual for MQ4G256V2 (qt=44), DFlash verify tier.
// Arithmetic / header / interleaved-C authority:
//   gemm_mq4g256v2_residual_wmma.hip  (base gfx11 half16 K2 kernel)
//
// One workgroup owns a single 16x16 output tile
// (row_start = blockIdx.x*16, batch_start = blockIdx.y*16). KW waves own
// disjoint K-ranges of that SAME tile: wave w owns groups
// [w*G/KW, (w+1)*G/KW) where G = K/256. Each wave runs the base kernel's
// exact per-group loop (dual headers, DQ macro, 16 WMMA per group) on its
// K-range into its private float8 acc, then the accs reduce through LDS in
// fixed wave order and wave 0 applies the single Y += once.
//
// Grid:  [ceil(M/16), ceil(N/16), 1]
// Block: [32*KW, 1, 1]. Static LDS KW*1 KiB (red[KW][8][32] floats);
// dynamic shared_mem=0. Instantiated: ks2/ks4/ks8.
// Production policy (exact gfx1100, non-replay/non-capture, N<=16):
// small-batch residual shapes (DFlash verify). Launcher requires
// K % 256 == 0, G % KW == 0, G >= KW (else fall back to a smaller KW or base).
// Output: Y += sum with unique owner; gfx11 interleaved C mapping.
// No atomics, no global partial buffer, no early return before the barrier
// (M/N tails masked inside, as mw_lds does). Reduction order w=0..KW-1 is
// fixed, hence deterministic; fp32 association differs from the base kernel
// so bit-exactness is NOT claimed (parity gate: relL2 <= 1e-5).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define GEN_RESID_KSPLIT_LDS(KW)                                                              \
    extern "C" __launch_bounds__(32 * (KW), 1) __global__ void                                \
        gemm_mq4g256v2_residual_wmma_gfx1100_ks##KW##_lds(                                    \
            const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
            int M, int K, int N) {                                                            \
        const int tid = threadIdx.x;                                                          \
        const int lane = tid & 31;                                                            \
        const int wave_id = tid >> 5;                                                         \
        const int ml = lane & 15;                                                             \
        const int row_start = blockIdx.x * 16;                                                \
        const int batch_start = blockIdx.y * 16;                                              \
        /* Same tile for every wave; tails duplicate row M-1 / batch 0 (discarded). */        \
        const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);               \
        const int out_col = batch_start + ml;                                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                   \
        const int groups_per_row = K / 256;                                                   \
        const int groups_per_wave = groups_per_row / (KW);                                    \
        const int g_begin = wave_id * groups_per_wave;                                        \
        const int g_end = g_begin + groups_per_wave;                                          \
        const char* row_base = A + (long long)safe_row * groups_per_row * 136;                \
        const _Float16* x_base = X + (long long)safe_batch * K;                                \
                                                                                              \
        /* Base-kernel register footprint: one float8 acc + one half16 a/b per lane. */       \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                       \
                                                                                              \
        for (int g = g_begin; g < g_end; g++) {                                               \
            const char* gp = row_base + g * 136;                                              \
            const unsigned int hA = *(const unsigned int*)(gp);                               \
            const unsigned int hB = *(const unsigned int*)(gp + 4);                            \
            const _Float16 sc0 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));      \
            const _Float16 zp0 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));          \
            const _Float16 sc1 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));      \
            const _Float16 zp1 =                                                              \
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));          \
            const _Float16* xg = x_base + g * 256;                                            \
                                                                                              \
            _Pragma("unroll")                                                                 \
            for (int kt = 0; kt < 16; kt++) {                                                 \
                const int k_off = kt * 16;                                                    \
                const _Float16 sc_h = (kt < 8) ? sc0 : sc1;                                   \
                const _Float16 zp_h = (kt < 8) ? zp0 : zp1;                                   \
                                                                                              \
                unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);                \
                unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);            \
                                                                                              \
                half16_t a_reg;                                                               \
                a_reg[0] = sc_h * (_Float16)(float)((pk0 >> 0) & 0xFu) + zp_h;                \
                a_reg[1] = sc_h * (_Float16)(float)((pk0 >> 4) & 0xFu) + zp_h;                \
                a_reg[2] = sc_h * (_Float16)(float)((pk0 >> 8) & 0xFu) + zp_h;                \
                a_reg[3] = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;               \
                a_reg[4] = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;               \
                a_reg[5] = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;               \
                a_reg[6] = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;               \
                a_reg[7] = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;               \
                a_reg[8] = sc_h * (_Float16)(float)((pk1 >> 0) & 0xFu) + zp_h;                \
                a_reg[9] = sc_h * (_Float16)(float)((pk1 >> 4) & 0xFu) + zp_h;                \
                a_reg[10] = sc_h * (_Float16)(float)((pk1 >> 8) & 0xFu) + zp_h;               \
                a_reg[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;              \
                a_reg[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;              \
                a_reg[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;              \
                a_reg[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;              \
                a_reg[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;              \
                                                                                              \
                half16_t b_reg = *(const half16_t*)(xg + k_off);                              \
                                                                                              \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);          \
            }                                                                                 \
        }                                                                                     \
                                                                                              \
        /* KW waves x 8 acc lanes x 32 lanes: KW KiB. Lane-consecutive: bank-clean. */        \
        __shared__ float red[KW][8][32];                                                      \
        _Pragma("unroll")                                                                     \
        for (int j = 0; j < 8; j++)                                                           \
            red[wave_id][j][lane] = acc[j];                                                   \
        /* All threads reach the barrier every launch — no early returns. */                  \
        __syncthreads();                                                                      \
                                                                                              \
        /* Wave 0 sums in FIXED order w=0..KW-1 (deterministic), single Y += owner. */        \
        if (wave_id == 0) {                                                                   \
            float8_t sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                   \
            for (int w = 0; w < (KW); w++) {                                                  \
                _Pragma("unroll")                                                             \
                for (int j = 0; j < 8; j++)                                                   \
                    sum[j] += red[w][j][lane];                                                \
            }                                                                                 \
            /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. */                  \
            if (out_col < N) {                                                                \
                _Pragma("unroll")                                                             \
                for (int j = 0; j < 8; j++) {                                                 \
                    const int out_row = row_start + 2 * j + (lane >> 4);                       \
                    if (out_row < M)                                                          \
                        Y[(long long)out_col * M + out_row] += sum[j];                        \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    }

GEN_RESID_KSPLIT_LDS(2)
GEN_RESID_KSPLIT_LDS(4)
GEN_RESID_KSPLIT_LDS(8)
// ── exp/ks4-prefetch: software-pipelined ks4 (KILL EXPERIMENT) ────────────────
// Hypothesis: the gfx1100 N<=16 residual GEMM is MEMORY-LATENCY-bound (too few
// outstanding weight loads per CU), not bandwidth- or dequant-bound. These two
// kw=4-only variants pipeline the ks4 group loop with a whole-group lookahead:
// at loop top the raw bytes (2 header u32 + 32 payload u32) for a FUTURE group
// are burst-loaded into lane-private registers, then the CURRENT group is
// dequantized + 16x WMMA from previously-loaded registers. The burst puts ~34
// independent loads in flight during the 16 WMMA of compute; ks4 serializes
// load->use with ~600 ns DRAM latency behind it.
//   ks4_pf_lds  (depth 2): 1 group lookahead, +34 live u32 per wave.
//   ks4_pf4_lds (depth 4): 3 group lookahead, +102 live u32 per wave.
// Bit-exactness contract: dequant arithmetic, WMMA operands/order, split-K
// reduce, and interleaved C mapping are textually identical to ks4 above, so
// output must be BIT-IDENTICAL to ks4 (not just relL2). Any bitwise diff is a
// pipeline bug. Same 48 B kernarg ABI, same grid/block (ceil(M/16) x
// ceil(N/16), 128 threads), same static LDS (red[4][8][32] = 4 KiB).
// All prefetch arrays are indexed by loop-unrolled constants only, so the
// compiler must keep them in VGPRs (scratch != 0 means the pipeline spilled
// and the measurement is void).

#define PF_LOAD_GROUP(S, GIDX)                                                   \
    do {                                                                        \
        const long long _pf_goff##S = (long long)(GIDX) * 136;                   \
        const char* _pf_gp##S = row_base + _pf_goff##S;                          \
        hA##S = *(const unsigned int*)(_pf_gp##S);                               \
        hB##S = *(const unsigned int*)(_pf_gp##S + 4);                           \
        _Pragma("unroll")                                                       \
        for (int _pi = 0; _pi < 32; _pi++)                                      \
            pk##S[_pi] = *((const unsigned int*)(_pf_gp##S + 8) + _pi);          \
    } while (0)

#define PF_CONSUME_GROUP(S, GIDX)                                               \
    do {                                                                        \
        const _Float16 _pf_sc0 =                                                \
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hA##S & 0xFFFFu))); \
        const _Float16 _pf_zp0 =                                                \
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hA##S >> 16))); \
        const _Float16 _pf_sc1 =                                                \
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hB##S & 0xFFFFu))); \
        const _Float16 _pf_zp1 =                                                \
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hB##S >> 16))); \
        const _Float16* _pf_xg = x_base + (long long)(GIDX) * 256;               \
        _Pragma("unroll")                                                       \
        for (int kt = 0; kt < 16; kt++) {                                       \
            const int k_off = kt * 16;                                          \
            const _Float16 sc_h = (kt < 8) ? _pf_sc0 : _pf_sc1;                  \
            const _Float16 zp_h = (kt < 8) ? _pf_zp0 : _pf_zp1;                  \
            unsigned int qw0 = pk##S[2 * kt];                                    \
            unsigned int qw1 = pk##S[2 * kt + 1];                                \
            half16_t a_reg;                                                     \
            a_reg[0] = sc_h * (_Float16)(float)((qw0 >> 0) & 0xFu) + zp_h;       \
            a_reg[1] = sc_h * (_Float16)(float)((qw0 >> 4) & 0xFu) + zp_h;       \
            a_reg[2] = sc_h * (_Float16)(float)((qw0 >> 8) & 0xFu) + zp_h;       \
            a_reg[3] = sc_h * (_Float16)(float)((qw0 >> 12) & 0xFu) + zp_h;      \
            a_reg[4] = sc_h * (_Float16)(float)((qw0 >> 16) & 0xFu) + zp_h;      \
            a_reg[5] = sc_h * (_Float16)(float)((qw0 >> 20) & 0xFu) + zp_h;      \
            a_reg[6] = sc_h * (_Float16)(float)((qw0 >> 24) & 0xFu) + zp_h;      \
            a_reg[7] = sc_h * (_Float16)(float)((qw0 >> 28) & 0xFu) + zp_h;      \
            a_reg[8] = sc_h * (_Float16)(float)((qw1 >> 0) & 0xFu) + zp_h;       \
            a_reg[9] = sc_h * (_Float16)(float)((qw1 >> 4) & 0xFu) + zp_h;       \
            a_reg[10] = sc_h * (_Float16)(float)((qw1 >> 8) & 0xFu) + zp_h;      \
            a_reg[11] = sc_h * (_Float16)(float)((qw1 >> 12) & 0xFu) + zp_h;     \
            a_reg[12] = sc_h * (_Float16)(float)((qw1 >> 16) & 0xFu) + zp_h;     \
            a_reg[13] = sc_h * (_Float16)(float)((qw1 >> 20) & 0xFu) + zp_h;     \
            a_reg[14] = sc_h * (_Float16)(float)((qw1 >> 24) & 0xFu) + zp_h;     \
            a_reg[15] = sc_h * (_Float16)(float)((qw1 >> 28) & 0xFu) + zp_h;     \
            half16_t b_reg = *(const half16_t*)(_pf_xg + k_off);                 \
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
        }                                                                       \
    } while (0)

#define PF_REDUCE_4()                                                           \
    __shared__ float red[4][8][32];                                             \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < 8; j++)                                                 \
        red[wave_id][j][lane] = acc[j];                                         \
    __syncthreads();                                                            \
    if (wave_id == 0) {                                                         \
        float8_t sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};         \
        for (int w = 0; w < 4; w++) {                                           \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 8; j++)                                         \
                sum[j] += red[w][j][lane];                                      \
        }                                                                       \
        if (out_col < N) {                                                      \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 8; j++) {                                       \
                const int out_row = row_start + 2 * j + (lane >> 4);             \
                if (out_row < M)                                                \
                    Y[(long long)out_col * M + out_row] += sum[j];              \
            }                                                                   \
        }                                                                       \
    }

extern "C" __launch_bounds__(128, 1) __global__ void                            \
    gemm_mq4g256v2_residual_wmma_gfx1100_ks4_pf_lds(                            \
        const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
        int M, int K, int N) {                                                  \
    const int tid = threadIdx.x;                                                \
    const int lane = tid & 31;                                                  \
    const int wave_id = tid >> 5;                                               \
    const int ml = lane & 15;                                                   \
    const int row_start = blockIdx.x * 16;                                      \
    const int batch_start = blockIdx.y * 16;                                    \
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);     \
    const int out_col = batch_start + ml;                                       \
    const int safe_batch = (out_col < N) ? out_col : 0;                         \
    const int groups_per_row = K / 256;                                         \
    const int groups_per_wave = groups_per_row / 4;                             \
    const int g_begin = wave_id * groups_per_wave;                              \
    const int g_end = g_begin + groups_per_wave;                                \
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;      \
    const _Float16* x_base = X + (long long)safe_batch * K;                      \
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};             \
    unsigned int hA0, hB0, hA1, hB1;                                            \
    unsigned int pk0[32], pk1[32];                                              \
    PF_LOAD_GROUP(0, g_begin);                                                  \
    for (int g = g_begin; g < g_end; g++) {                                     \
        if (((g - g_begin) & 1) == 0) {                                         \
            if (g + 1 < g_end)                                                  \
                PF_LOAD_GROUP(1, g + 1);                                        \
            PF_CONSUME_GROUP(0, g);                                             \
        } else {                                                                \
            if (g + 1 < g_end)                                                  \
                PF_LOAD_GROUP(0, g + 1);                                        \
            PF_CONSUME_GROUP(1, g);                                             \
        }                                                                       \
    }                                                                           \
    PF_REDUCE_4();                                                              \
}

extern "C" __launch_bounds__(128, 1) __global__ void                            \
    gemm_mq4g256v2_residual_wmma_gfx1100_ks4_pf4_lds(                           \
        const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
        int M, int K, int N) {                                                  \
    const int tid = threadIdx.x;                                                \
    const int lane = tid & 31;                                                  \
    const int wave_id = tid >> 5;                                               \
    const int ml = lane & 15;                                                   \
    const int row_start = blockIdx.x * 16;                                      \
    const int batch_start = blockIdx.y * 16;                                    \
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);     \
    const int out_col = batch_start + ml;                                       \
    const int safe_batch = (out_col < N) ? out_col : 0;                         \
    const int groups_per_row = K / 256;                                         \
    const int groups_per_wave = groups_per_row / 4;                             \
    const int g_begin = wave_id * groups_per_wave;                              \
    const int g_end = g_begin + groups_per_wave;                                \
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;      \
    const _Float16* x_base = X + (long long)safe_batch * K;                      \
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};             \
    unsigned int hA0, hB0, hA1, hB1, hA2, hB2, hA3, hB3;                         \
    unsigned int pk0[32], pk1[32], pk2[32], pk3[32];                            \
    PF_LOAD_GROUP(0, g_begin);                                                  \
    if (g_begin + 1 < g_end)                                                    \
        PF_LOAD_GROUP(1, g_begin + 1);                                          \
    if (g_begin + 2 < g_end)                                                    \
        PF_LOAD_GROUP(2, g_begin + 2);                                          \
    for (int g = g_begin; g < g_end; g++) {                                     \
        const int phase = (g - g_begin) & 3;                                    \
        if (g + 3 < g_end) {                                                    \
            if (phase == 0)                                                     \
                PF_LOAD_GROUP(3, g + 3);                                        \
            else if (phase == 1)                                                \
                PF_LOAD_GROUP(0, g + 3);                                        \
            else if (phase == 2)                                                \
                PF_LOAD_GROUP(1, g + 3);                                        \
            else                                                                \
                PF_LOAD_GROUP(2, g + 3);                                        \
        }                                                                       \
        if (phase == 0)                                                         \
            PF_CONSUME_GROUP(0, g);                                             \
        else if (phase == 1)                                                    \
            PF_CONSUME_GROUP(1, g);                                             \
        else if (phase == 2)                                                    \
            PF_CONSUME_GROUP(2, g);                                             \
        else                                                                    \
            PF_CONSUME_GROUP(3, g);                                             \
    }                                                                           \
    PF_REDUCE_4();                                                              \
}
