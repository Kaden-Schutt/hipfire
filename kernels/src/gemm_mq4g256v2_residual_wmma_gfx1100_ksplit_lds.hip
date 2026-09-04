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

// ---- ks4-radiowave lowering hooks (S1-kernel-hooks) ----
// Frozen-contract levers, driven via -D (HIPFIRE_KS4_DEFINES in production):
//   KS4_W_LOADFORM  0=flat (default)  1=BUFFER_B32  2=ALIGNED_B128 (weight stream)
//   KS4_X_LOADFORM  0=flat (default)  1=BUFFER_B32  2=ALIGNED_B128 (X stream)
//   KS4_VGPR_CAP    0=none (default)  else amdgpu_num_vgpr(N) on ks2/ks4/ks8
//   KS4_UNROLL      1=today, no pragma (default)  else _Pragma(unroll N) on group loop
// LOADFORM==0 expands to today's exact flat dereferences (byte-identical default).
// Non-zero forms use the radiowave buffer-resource lowering; authority is
// crates/radiowave/include/radiowave/hip.h:42-64 (same descriptor word
// 0x31004000, same raw_buffer_load builtins). Helpers are defined inline here
// so the production compile needs no extra -I, only -D defines.
// W-form-2 loads the containing 16B chunk (byte_off & ~15) and extracts the
// lane, so every B128 is 16B-aligned by construction despite the 136B group
// stride (Slice 3 parity gate still applies to every non-zero form).
#ifndef KS4_W_LOADFORM
#define KS4_W_LOADFORM 0
#endif
#ifndef KS4_X_LOADFORM
#define KS4_X_LOADFORM 0
#endif
#ifndef KS4_VGPR_CAP
#define KS4_VGPR_CAP 0
#endif
#ifndef KS4_UNROLL
#define KS4_UNROLL 1
#endif

#if KS4_VGPR_CAP > 0
#define KS4_VGPR_ATTR __attribute__((amdgpu_num_vgpr(KS4_VGPR_CAP)))
#else
#define KS4_VGPR_ATTR
#endif

#define KS4_STR_(x) #x
#define KS4_STR(x) KS4_STR_(x)
#if KS4_UNROLL > 1
#define KS4_GROUP_UNROLL _Pragma(KS4_STR(unroll KS4_UNROLL))
#else
#define KS4_GROUP_UNROLL
#endif

#if KS4_W_LOADFORM != 0 || KS4_X_LOADFORM != 0
typedef unsigned int __attribute__((ext_vector_type(4))) ks4rw_u32x4_t;
typedef unsigned int __attribute__((ext_vector_type(8))) ks4rw_u32x8_t;
typedef int __attribute__((ext_vector_type(4))) ks4rw_i32x4_t;
typedef __amdgpu_buffer_rsrc_t ks4rw_rsrc_t;

__device__ __forceinline__ ks4rw_rsrc_t ks4rw_make_rsrc(const unsigned int* p) {
    // Same descriptor word as radiowave::buffer_resource (hip.h:42-45).
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<unsigned int*>(p), 0, 0xffffffffu, 0x31004000u);
}

__device__ __forceinline__ unsigned int ks4rw_buf_u32(ks4rw_rsrc_t r, unsigned int word_off) {
    return (unsigned int)__builtin_amdgcn_raw_buffer_load_b32(r, word_off * 4u, 0, 0);
}

__device__ __forceinline__ ks4rw_u32x4_t ks4rw_buf_u32x4_byte(ks4rw_rsrc_t r, unsigned int byte_off) {
    const ks4rw_i32x4_t w = __builtin_amdgcn_raw_buffer_load_b128(r, byte_off, 0, 0);
    return __builtin_bit_cast(ks4rw_u32x4_t, w);
}

__device__ __forceinline__ unsigned int ks4rw_buf_u32x4_lane(ks4rw_rsrc_t r, unsigned int byte_off) {
    const ks4rw_u32x4_t c = ks4rw_buf_u32x4_byte(r, byte_off & ~15u);
    return c[(byte_off >> 2) & 3u];
}

__device__ __forceinline__ half16_t ks4rw_x_b32(ks4rw_rsrc_t r, unsigned int half_off) {
    const unsigned int wo = half_off >> 1;
    ks4rw_u32x8_t w;
    w[0] = ks4rw_buf_u32(r, wo + 0);
    w[1] = ks4rw_buf_u32(r, wo + 1);
    w[2] = ks4rw_buf_u32(r, wo + 2);
    w[3] = ks4rw_buf_u32(r, wo + 3);
    w[4] = ks4rw_buf_u32(r, wo + 4);
    w[5] = ks4rw_buf_u32(r, wo + 5);
    w[6] = ks4rw_buf_u32(r, wo + 6);
    w[7] = ks4rw_buf_u32(r, wo + 7);
    return __builtin_bit_cast(half16_t, w);
}

__device__ __forceinline__ half16_t ks4rw_x_b128(ks4rw_rsrc_t r, unsigned int half_off) {
    const unsigned int wo = half_off >> 1;
    const ks4rw_u32x4_t lo = ks4rw_buf_u32x4_byte(r, (wo + 0) * 4u);
    const ks4rw_u32x4_t hi = ks4rw_buf_u32x4_byte(r, (wo + 4) * 4u);
    ks4rw_u32x8_t w;
    w[0] = lo[0];
    w[1] = lo[1];
    w[2] = lo[2];
    w[3] = lo[3];
    w[4] = hi[0];
    w[5] = hi[1];
    w[6] = hi[2];
    w[7] = hi[3];
    return __builtin_bit_cast(half16_t, w);
}
#endif

#if KS4_W_LOADFORM == 1
#define KS4_W_U32(P) (ks4rw_buf_u32(ks4rw_w_rsrc, (unsigned int)(((const char*)(P) - row_base) >> 2)))
#elif KS4_W_LOADFORM == 2
#define KS4_W_U32(P) (ks4rw_buf_u32x4_lane(ks4rw_w_rsrc, (unsigned int)((const char*)(P) - row_base)))
#else
#define KS4_W_U32(P) (*(const unsigned int*)(P))
#endif

#if KS4_X_LOADFORM == 1
#define KS4_X_F16(P) (ks4rw_x_b32(ks4rw_x_rsrc, (unsigned int)((const _Float16*)(P) - x_base)))
#elif KS4_X_LOADFORM == 2
#define KS4_X_F16(P) (ks4rw_x_b128(ks4rw_x_rsrc, (unsigned int)((const _Float16*)(P) - x_base)))
#else
#define KS4_X_F16(P) (*(const half16_t*)(P))
#endif

#if KS4_W_LOADFORM != 0
#define KS4_W_HOIST const ks4rw_rsrc_t ks4rw_w_rsrc = ks4rw_make_rsrc((const unsigned int*)row_base);
#else
#define KS4_W_HOIST
#endif
#if KS4_X_LOADFORM != 0
#define KS4_X_HOIST const ks4rw_rsrc_t ks4rw_x_rsrc = ks4rw_make_rsrc((const unsigned int*)x_base);
#else
#define KS4_X_HOIST
#endif

#define GEN_RESID_KSPLIT_LDS(KW)                                                              \
    extern "C" __launch_bounds__(32 * (KW), 1) KS4_VGPR_ATTR __global__ void                   \
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
        KS4_W_HOIST                                                                            \
        KS4_X_HOIST                                                                            \
                                                                                              \
        /* Base-kernel register footprint: one float8 acc + one half16 a/b per lane. */       \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                       \
                                                                                              \
        KS4_GROUP_UNROLL                                                                       \
        for (int g = g_begin; g < g_end; g++) {                                               \
            const char* gp = row_base + g * 136;                                              \
            const unsigned int hA = KS4_W_U32(gp);                                            \
            const unsigned int hB = KS4_W_U32(gp + 4);                                        \
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
                unsigned int pk0 = KS4_W_U32(gp + 8 + k_off / 2);                             \
                unsigned int pk1 = KS4_W_U32(gp + 8 + k_off / 2 + 4);                         \
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
                half16_t b_reg = KS4_X_F16(xg + k_off);                                          \
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
