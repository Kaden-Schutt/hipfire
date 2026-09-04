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
// ── xlds: slab-synchronized X-in-LDS residual (DFlash verify tier) ─────────────
// Goal: remove the per-WMMA X-fragment drain on gfx1100 at N<=16. Every ks
// variant issues 2x global_load_b128 of the X (B-operand) fragment + a full
// `s_waitcnt vmcnt(0)` immediately before EACH 16x16 WMMA (~1,600 drains per
// launch, each wave re-reading the identical X from DRAM). Weight-side
// prefetch (9x MLP) changed nothing, proving weight latency is not the bound.
//
// Structure (A: slab-synchronized, gfx12-ldsstage-shaped): one workgroup owns
// a single 16x16 output tile (row_start = blockIdx.x*16, batch_start =
// blockIdx.y*16) with 8 waves (256 threads). K is walked in 512-wide slabs
// (slab = 2x G256 groups). Per slab:
//   1. ALL 256 threads cooperatively stage the slab's X tile
//      (16 batch rows x 512 K fp16 = 16 KiB) from global to LDS
//      (each thread moves 32 fp16 = 4x dwordx4 of its batch row),
//      then __syncthreads;
//   2. each wave prefetches its 64-K quarter's weights (2 header u32 +
//      4 fragments x 2 payload u32 = 10 u32) from global into VGPRs;
//   3. each wave consumes its quarter from LDS: 4x (ds_read B + base-kernel
//      DQ dequant + wmma) with NO global load in the loop body, so the
//      compiler has no vmcnt source between consecutive WMMAs;
//      then __syncthreads before the next slab overwrites the X tile.
// Weights stay per-wave global loads (proven not to be the bound); only the
// X drain is removed. Per-wave K association = wave w owns slab offset
// w*64 (group_in_slab = wave>>2, quarter = wave&3, header quarter<2 ? hA :
// hB, fragments quarter*4..quarter*4+3), reduced wave-0..7 in fixed order.
//
// Arithmetic / header / interleaved-C authority: gemm_mq4g256v2_residual_wmma.hip
// (base): dual fp16 headers, kt<8 -> (s0,z0) else (s1,z1) == quarter<2 ? hA :
// hB, DQ macro verbatim, half16 full-16-K-per-lane B, acc[j] =
// C[2*j+(lane>>4)][lane&15], fused Y += once by wave 0.
//
// Grid:  [ceil(M/16), ceil(N/16), 1]
// Block: [256, 1, 1]. Static LDS 16 KiB (staged_x[16][512] fp16) + 8 KiB
// (red[8][8][32] floats) = 24 KiB; dynamic shared_mem=0. Requires K % 512 == 0
// and K > 0 (launcher falls back to ks4 / ks table / base otherwise).
// Deterministic (fixed wave-order LDS reduction, no atomics) hence
// capture-safe; same 48 B kernarg ABI as ks. fp32 association differs from
// base AND from ks (slab-quarter partition), so bit-exactness is NOT claimed
// (parity gate: relL2 <= 5e-5 vs base, no farther from f64 truth than base).
// Kill switch: HIPFIRE_RESIDUAL_XLDS_OFF=1 restores ks4 on the tier path.
// Compile with `hipcc --offload-arch=gfx1100`.
typedef unsigned int __attribute__((ext_vector_type(4))) resid_xlds_uint4v_t;
__launch_bounds__(256, 2)
extern "C" __global__ void gemm_mq4g256v2_residual_wmma_gfx1100_xlds(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,
    int M, int K, int N) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    // Uniform over the workgroup: safe to return before any barrier.
    if (row_start >= M || batch_start >= N) return;
    const int ml = lane & 15;
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);
    const int out_col = batch_start + ml;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const int slabs = K / 512;
    const char* row_base = A + (long long)safe_row * row_bytes;
    // 16 batch rows x 512 K fp16 X slab tile + 8-wave fp32 acc reduce buffer.
    __shared__ __align__(16) _Float16 staged_x[16 * 512];
    __shared__ float red[8][8][32];
    // Cooperative X-fill assignment: thread tid stages 32 fp16 (4x dwordx4)
    // of batch slot sb = tid>>4 at K-chunk st = tid&15. Tails replicate row 0.
    const int sb = tid >> 4;
    const int st = tid & 15;
    const int sb_row = (batch_start + sb < N) ? (batch_start + sb) : 0;
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // This wave's fixed 64-K quarter of every slab (uniform per wave).
    const int group_in_slab = wave >> 2;
    const int quarter = wave & 3;
    const int frag_base = quarter * 4;
    for (int slab = 0; slab < slabs; ++slab) {
        // (1) Stage this slab's X tile cooperatively, then barrier. The only
        // global X traffic of the launch; vmcnt waits live here, at the slab
        // boundary, never in the consume loop below.
        const _Float16* x_src = X + (long long)sb_row * K + slab * 512 + st * 32;
        _Float16* x_dst = staged_x + sb * 512 + st * 32;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            ((resid_xlds_uint4v_t*)x_dst)[i] = ((const resid_xlds_uint4v_t*)x_src)[i];
        __syncthreads();
        // (2) Prefetch this wave's quarter weights into VGPRs (10 u32). Kept
        // in a separate fully-unrolled loop over scalar temporaries so the
        // consume loop below issues no global loads of its own.
        const char* gp = row_base + (long long)(slab * 2 + group_in_slab) * 136;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (quarter < 2) ? hA : hB;
        const _Float16 sc_h =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const _Float16 zp_h =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const int f0 = frag_base;
        unsigned int pk0_0 = *(const unsigned int*)(gp + 8 + (f0 + 0) * 8);
        unsigned int pk1_0 = *(const unsigned int*)(gp + 8 + (f0 + 0) * 8 + 4);
        unsigned int pk0_1 = *(const unsigned int*)(gp + 8 + (f0 + 1) * 8);
        unsigned int pk1_1 = *(const unsigned int*)(gp + 8 + (f0 + 1) * 8 + 4);
        unsigned int pk0_2 = *(const unsigned int*)(gp + 8 + (f0 + 2) * 8);
        unsigned int pk1_2 = *(const unsigned int*)(gp + 8 + (f0 + 2) * 8 + 4);
        unsigned int pk0_3 = *(const unsigned int*)(gp + 8 + (f0 + 3) * 8);
        unsigned int pk1_3 = *(const unsigned int*)(gp + 8 + (f0 + 3) * 8 + 4);
        // (3) Consume the quarter from LDS: B fragments come from staged_x
        // via ds_load (lgkmcnt, never vmcnt); the body is ds_load +
        // base-kernel DQ VALU + wmma. Weight payloads were prefetched to
        // VGPRs above, so the loop issues no global loads of its own.
        const _Float16* b_base = staged_x + ml * 512 + wave * 64;
        {
            half16_t a_reg;
            #define RESID_XLDS_DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            RESID_XLDS_DQ(0, pk0_0, 0); RESID_XLDS_DQ(1, pk0_0, 4);
            RESID_XLDS_DQ(2, pk0_0, 8); RESID_XLDS_DQ(3, pk0_0, 12);
            RESID_XLDS_DQ(4, pk0_0, 16); RESID_XLDS_DQ(5, pk0_0, 20);
            RESID_XLDS_DQ(6, pk0_0, 24); RESID_XLDS_DQ(7, pk0_0, 28);
            RESID_XLDS_DQ(8, pk1_0, 0); RESID_XLDS_DQ(9, pk1_0, 4);
            RESID_XLDS_DQ(10, pk1_0, 8); RESID_XLDS_DQ(11, pk1_0, 12);
            RESID_XLDS_DQ(12, pk1_0, 16); RESID_XLDS_DQ(13, pk1_0, 20);
            RESID_XLDS_DQ(14, pk1_0, 24); RESID_XLDS_DQ(15, pk1_0, 28);
            half16_t b_reg = *(const half16_t*)(b_base + 0);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
            RESID_XLDS_DQ(0, pk0_1, 0); RESID_XLDS_DQ(1, pk0_1, 4);
            RESID_XLDS_DQ(2, pk0_1, 8); RESID_XLDS_DQ(3, pk0_1, 12);
            RESID_XLDS_DQ(4, pk0_1, 16); RESID_XLDS_DQ(5, pk0_1, 20);
            RESID_XLDS_DQ(6, pk0_1, 24); RESID_XLDS_DQ(7, pk0_1, 28);
            RESID_XLDS_DQ(8, pk1_1, 0); RESID_XLDS_DQ(9, pk1_1, 4);
            RESID_XLDS_DQ(10, pk1_1, 8); RESID_XLDS_DQ(11, pk1_1, 12);
            RESID_XLDS_DQ(12, pk1_1, 16); RESID_XLDS_DQ(13, pk1_1, 20);
            RESID_XLDS_DQ(14, pk1_1, 24); RESID_XLDS_DQ(15, pk1_1, 28);
            b_reg = *(const half16_t*)(b_base + 16);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
            RESID_XLDS_DQ(0, pk0_2, 0); RESID_XLDS_DQ(1, pk0_2, 4);
            RESID_XLDS_DQ(2, pk0_2, 8); RESID_XLDS_DQ(3, pk0_2, 12);
            RESID_XLDS_DQ(4, pk0_2, 16); RESID_XLDS_DQ(5, pk0_2, 20);
            RESID_XLDS_DQ(6, pk0_2, 24); RESID_XLDS_DQ(7, pk0_2, 28);
            RESID_XLDS_DQ(8, pk1_2, 0); RESID_XLDS_DQ(9, pk1_2, 4);
            RESID_XLDS_DQ(10, pk1_2, 8); RESID_XLDS_DQ(11, pk1_2, 12);
            RESID_XLDS_DQ(12, pk1_2, 16); RESID_XLDS_DQ(13, pk1_2, 20);
            RESID_XLDS_DQ(14, pk1_2, 24); RESID_XLDS_DQ(15, pk1_2, 28);
            b_reg = *(const half16_t*)(b_base + 32);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
            RESID_XLDS_DQ(0, pk0_3, 0); RESID_XLDS_DQ(1, pk0_3, 4);
            RESID_XLDS_DQ(2, pk0_3, 8); RESID_XLDS_DQ(3, pk0_3, 12);
            RESID_XLDS_DQ(4, pk0_3, 16); RESID_XLDS_DQ(5, pk0_3, 20);
            RESID_XLDS_DQ(6, pk0_3, 24); RESID_XLDS_DQ(7, pk0_3, 28);
            RESID_XLDS_DQ(8, pk1_3, 0); RESID_XLDS_DQ(9, pk1_3, 4);
            RESID_XLDS_DQ(10, pk1_3, 8); RESID_XLDS_DQ(11, pk1_3, 12);
            RESID_XLDS_DQ(12, pk1_3, 16); RESID_XLDS_DQ(13, pk1_3, 20);
            RESID_XLDS_DQ(14, pk1_3, 24); RESID_XLDS_DQ(15, pk1_3, 28);
            #undef RESID_XLDS_DQ
            b_reg = *(const half16_t*)(b_base + 48);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
        // All waves must finish consuming the slab before the cooperative
        // fill overwrites it on the next iteration.
        __syncthreads();
    }
    // 8 waves x 8 acc lanes x 32 lanes: 8 KiB. Lane-consecutive: bank-clean.
    #pragma unroll
    for (int j = 0; j < 8; j++)
        red[wave][j][lane] = acc[j];
    __syncthreads();
    // Wave 0 sums in FIXED order w=0..7 (deterministic), single Y += owner.
    if (wave == 0) {
        float8_t sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < 8; w++) {
            #pragma unroll
            for (int j = 0; j < 8; j++)
                sum[j] += red[w][j][lane];
        }
        // RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15].
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 2 * j + (lane >> 4);
                if (out_row < M)
                    Y[(long long)out_col * M + out_row] += sum[j];
            }
        }
    }
}
