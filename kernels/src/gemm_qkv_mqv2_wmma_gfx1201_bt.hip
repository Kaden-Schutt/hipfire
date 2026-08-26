// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Generic MQ{2,3,5,6}G256V2 QKV batch-tiled WMMA — exact-gfx1201 BT8 only.
// Self-contained (no project-local includes). Compile ONLY with
//   hipcc --offload-arch=gfx1201
//
// Authority:
//   MQ4 gfx1201 QKV BT K4 loop: gemm_qkv_mq4g256v2_wmma_gfx1201_bt.hip (BV=8)
//   Contiguous gfx12 C map:     acc[j] -> row 8*k_grp+j  (same as gfx12 bases)
//   V2 payload: 8 B dual-FP16 header + contiguous bit-packed codes
//               GROUP_BYTES = 8 + 32*BITS; half header selected by kt<8
//
// Symbols (extern "C"):
//   gemm_qkv_mq2g256v2_wmma_gfx1201_bt8
//   gemm_qkv_mq3g256v2_wmma_gfx1201_bt8
//   gemm_qkv_mq5g256v2_wmma_gfx1201_bt8
//   gemm_qkv_mq6g256v2_wmma_gfx1201_bt8
// Grid: [ceil(total_m/16), ceil(N/128)]  Block: [32]  LDS: 0
// ABI: (A_q, A_k, A_v, X, Y_q, Y_k, Y_v, q_m, k_m, v_m, K, N)
// Overwrite Y (not +=). Decode one half8 weight fragment per lane/kt; reuse
// across the 8 independent N-tile accumulators.

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "exact-gfx1201: gemm_qkv_mqv2_wmma_gfx1201_bt.hip requires --offload-arch=gfx1201"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// Generic V2 code unpack: 8 contiguous codes starting at absolute index code0
// within the 256-wide group payload (after the 8-byte dual-FP16 header).
// Per code: read current byte, and the next only when (bit_in_byte+BITS)>8.
template <int BITS>
__device__ __forceinline__ half8_t decode8(const unsigned char* __restrict__ codes,
                                           int code0,
                                           _Float16 sc_h,
                                           _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int bit_pos     = (code0 + i) * BITS;
        const int byte_i      = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned v = (unsigned)codes[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= (unsigned)codes[byte_i + 1] << 8;
        const unsigned q = (v >> bit_in_byte) & qmask;
        a[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return a;
}

// BT8 body: K4 unroll, weight decode once per (kt+ki), reuse across BV=8 N tiles.
template <int BITS>
__device__ __forceinline__ void gemm_qkv_mqv2_bt8_body(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m, int k_m, int v_m, int K, int N) {
    constexpr int GROUP_BYTES = 8 + 32 * BITS;
    constexpr int BV = 8;

    const int total_m = q_m + k_m + v_m;
    const int rs  = (int)blockIdx.x * 16;
    const int bs  = (int)blockIdx.y * (16 * BV);
    const int tid = (int)threadIdx.x;
    if (rs >= total_m || bs >= N) return;

    const int ml = tid & 15;
    const int kg = tid >> 4;
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1);

    const char* A;
    int lrow;
    if (sr < q_m) {
        A = A_q;
        lrow = sr;
    } else if (sr < q_m + k_m) {
        A = A_k;
        lrow = sr - q_m;
    } else {
        A = A_v;
        lrow = sr - (q_m + k_m);
    }

    const int gpr = K / 256;
    const char* rb = A + (long long)lrow * (long long)gpr * (long long)GROUP_BYTES;

    int sb[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }

    float8_t acc[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b)
        acc[b] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < gpr; ++g) {
        const char* gp = rb + (long long)g * (long long)GROUP_BYTES;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const unsigned char* codes = (const unsigned char*)(gp + 8);

        const _Float16* xg[BV];
#pragma unroll
        for (int b = 0; b < BV; ++b)
            xg[b] = X + (long long)sb[b] * (long long)K + (long long)g * 256;

        // K4: kt in {0,4,8,12} — each quartet stays inside one half-header.
        for (int kt = 0; kt < 16; kt += 4) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int code0 = (kt + ki) * 16 + kg * 8;
                // One weight fragment; reuse across 8 N-tile accumulators.
                const half8_t a = decode8<BITS>(codes, code0, sc_h, zp_h);
#pragma unroll
                for (int b = 0; b < BV; ++b) {
                    const half8_t bb =
                        *(const half8_t*)(xg[b] + (kt + ki) * 16 + kg * 8);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]);
                }
            }
        }
    }

    // Contiguous gfx12 C map: out_row = rs + 8*kg + j
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) {
                    float* Y;
                    int pr, os;
                    if (orow < q_m) {
                        Y  = Y_q;
                        pr = orow;
                        os = q_m;
                    } else if (orow < q_m + k_m) {
                        Y  = Y_k;
                        pr = orow - q_m;
                        os = k_m;
                    } else {
                        Y  = Y_v;
                        pr = orow - (q_m + k_m);
                        os = v_m;
                    }
                    Y[(long long)oc * (long long)os + pr] = acc[b][j];
                }
            }
        }
    }
}

#define GEN_QKV_MQV2_BT8(BITS)                                                                 \
    extern "C" __launch_bounds__(32, 2) __global__ void gemm_qkv_mq##BITS##g256v2_wmma_gfx1201_bt8( \
        const char* __restrict__ A_q,                                                          \
        const char* __restrict__ A_k,                                                          \
        const char* __restrict__ A_v,                                                          \
        const _Float16* __restrict__ X,                                                        \
        float* __restrict__ Y_q,                                                               \
        float* __restrict__ Y_k,                                                               \
        float* __restrict__ Y_v,                                                               \
        int q_m, int k_m, int v_m, int K, int N) {                                             \
        gemm_qkv_mqv2_bt8_body<BITS>(A_q, A_k, A_v, X, Y_q, Y_k, Y_v, q_m, k_m, v_m, K, N);  \
    }

GEN_QKV_MQV2_BT8(2)
GEN_QKV_MQV2_BT8(3)
GEN_QKV_MQV2_BT8(5)
GEN_QKV_MQV2_BT8(6)

#undef GEN_QKV_MQV2_BT8
