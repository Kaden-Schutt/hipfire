// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
using hipfire_qkv_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_qkv_f32x4 = float __attribute__((ext_vector_type(4)));
#endif

#ifndef HIPFIRE_QKV_KERNEL_NAME
#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq3g256v2
#endif

#if defined(HIPFIRE_QKV_WITH_BIAS)
#define HIPFIRE_QKV_BIAS_ARGS \
    , const float* __restrict__ bias_q \
    , const float* __restrict__ bias_k \
    , const float* __restrict__ bias_v
#define HIPFIRE_QKV_BIAS_DECL const float* bias;
#define HIPFIRE_QKV_BIAS_Q bias = bias_q;
#define HIPFIRE_QKV_BIAS_K bias = bias_k;
#define HIPFIRE_QKV_BIAS_V bias = bias_v;
#else
#define HIPFIRE_QKV_BIAS_ARGS
#define HIPFIRE_QKV_BIAS_DECL
#define HIPFIRE_QKV_BIAS_Q
#define HIPFIRE_QKV_BIAS_K
#define HIPFIRE_QKV_BIAS_V
#endif

// 3-way fused HFQ4-G256 GEMV for the Qwen3.5 FullAttention layer preamble.
// One kernel launch performs all three projections from a single shared
// x_rot input:
//
//   y_q = W_q · x_rot   (M=q_m, n_heads × head_dim × 2 — Q+gate interleaved)
//   y_k = W_k · x_rot   (M=k_m, n_kv_heads × head_dim)
//   y_v = W_v · x_rot   (M=v_m, n_kv_heads × head_dim)
//
// Same 4x-unroll inner loop and tail handling as gemv_hfq4g256.hip
// (5302926 accumulator-interleave fix). Grid = q_m + k_m + v_m.
//
// Arch coverage: works on every RDNA generation (gfx1010 / gfx1013 /
// gfx1030 / gfx1100+). See fused_qkvza_hfq4g256.hip for the full
// rationale on the unified 4-accumulator inner loop and the per-arch
// register-budget assumptions baked into launch_bounds(32, 16).
//
// Bit-exact with three sequential gemv_hfq4g256 calls.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_QKV_KERNEL_NAME(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const float* __restrict__ x,
    float*       __restrict__ y_q,
    float*       __restrict__ y_k,
    float*       __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K HIPFIRE_QKV_BIAS_ARGS
) {
    const int gid = blockIdx.x;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    // Route to one of the three projections.
    const char* A;
    float* y;
    int row;
    HIPFIRE_QKV_BIAS_DECL
    if (gid < q_m) {
        A = A_q; y = y_q; row = gid; HIPFIRE_QKV_BIAS_Q
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; row = gid - q_m; HIPFIRE_QKV_BIAS_K
    } else {
        A = A_v; y = y_v; row = gid - q_m - k_m; HIPFIRE_QKV_BIAS_V
    }

    const int tid = threadIdx.x;
#if defined(HIPFIRE_RDNA3_QKV_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 104;
    const int boff = tid * 3;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif
#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
    const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(x), 0, 0xffffffffu, 0x31004000u);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 104;
        const char* gp1 = gp0 + 104;
        const char* gp2 = gp1 + 104;
        const char* gp3 = gp2 + 104;

        const unsigned int hA_sc0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 104);
        const unsigned int hB_sc0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 104 + 4);
        const unsigned int hs_sc0 = (tid < 16) ? hA_sc0 : hB_sc0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 >> 16)));
        const unsigned int hA_sc1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 104);
        const unsigned int hB_sc1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 104 + 4);
        const unsigned int hs_sc1 = (tid < 16) ? hA_sc1 : hB_sc1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 >> 16)));
        const unsigned int hA_sc2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 104);
        const unsigned int hB_sc2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 104 + 4);
        const unsigned int hs_sc2 = (tid < 16) ? hA_sc2 : hB_sc2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 >> 16)));
        const unsigned int hA_sc3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 104);
        const unsigned int hB_sc3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 104 + 4);
        const unsigned int hs_sc3 = (tid < 16) ? hA_sc3 : hB_sc3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 >> 16)));
        const unsigned char* d_pk0 = (const unsigned char*)(gp0 + 8); unsigned int pk0 = (unsigned int)d_pk0[boff] | ((unsigned int)d_pk0[boff+1] << 8) | ((unsigned int)d_pk0[boff+2] << 16);
        const unsigned char* d_pk1 = (const unsigned char*)(gp1 + 8); unsigned int pk1 = (unsigned int)d_pk1[boff] | ((unsigned int)d_pk1[boff+1] << 8) | ((unsigned int)d_pk1[boff+2] << 16);
        const unsigned char* d_pk2 = (const unsigned char*)(gp2 + 8); unsigned int pk2 = (unsigned int)d_pk2[boff] | ((unsigned int)d_pk2[boff+1] << 8) | ((unsigned int)d_pk2[boff+2] << 16);
        const unsigned char* d_pk3 = (const unsigned char*)(gp3 + 8); unsigned int pk3 = (unsigned int)d_pk3[boff] | ((unsigned int)d_pk3[boff+1] << 8) | ((unsigned int)d_pk3[boff+2] << 16);

        int base = g * 256 + tid * 8;

#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
        const hipfire_qkv_f32x4 xv0a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 0) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv0b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 4) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv1a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 256) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv1b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 260) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv2a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 512) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv2b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 516) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv3a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 768) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv3b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 772) * 4, 0, 0));
        #define DOG_V(pk, sc, zp, a, x0, x1, x2, x3, x4, x5, x6, x7) \
            (a) += (sc * (float)((pk) & 7u)         + zp) * (x0) \
                 + (sc * (float)(((pk) >> 3 ) & 7u)  + zp) * (x1) \
                 + (sc * (float)(((pk) >> 6 ) & 7u)  + zp) * (x2) \
                 + (sc * (float)(((pk) >> 9 ) & 7u) + zp) * (x3) \
                 + (sc * (float)(((pk) >> 12 ) & 7u) + zp) * (x4) \
                 + (sc * (float)(((pk) >> 15 ) & 7u) + zp) * (x5) \
                 + (sc * (float)(((pk) >> 18 ) & 7u) + zp) * (x6) \
                 + (sc * (float)(((pk) >> 21 ) & 7u) + zp) * (x7)
        DOG_V(pk0, sc0, zp0, acc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
        DOG_V(pk1, sc1, zp1, acc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
        DOG_V(pk2, sc2, zp2, acc2, xv2a.x,xv2a.y,xv2a.z,xv2a.w,xv2b.x,xv2b.y,xv2b.z,xv2b.w);
        DOG_V(pk3, sc3, zp3, acc3, xv3a.x,xv3a.y,xv3a.z,xv3a.w,xv3b.x,xv3b.y,xv3b.z,xv3b.w);
        #undef DOG_V
#else
        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 7u)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 3 ) & 7u)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 6 ) & 7u)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 9 ) & 7u) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 12 ) & 7u) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 15 ) & 7u) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 18 ) & 7u) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 21 ) & 7u) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base,       acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
#endif
    }

    // Tail handling: preserve acc[gi] interleaving. See 5302926 bug report.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 7u)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 3 ) & 7u) + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 6 ) & 7u) + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 9 ) & 7u) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 12 ) & 7u) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 15 ) & 7u) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 18 ) & 7u) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 21 ) & 7u) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 104;
        const unsigned int hA_sc = *(const unsigned int*)(gp);
        const unsigned int hB_sc = *(const unsigned int*)(gp + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8); unsigned int pk = (unsigned int)d[boff] | ((unsigned int)d[boff+1] << 8) | ((unsigned int)d[boff+2] << 16);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 104;
        const unsigned int hA_sc = *(const unsigned int*)(gp);
        const unsigned int hB_sc = *(const unsigned int*)(gp + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8); unsigned int pk = (unsigned int)d[boff] | ((unsigned int)d[boff+1] << 8) | ((unsigned int)d[boff+2] << 16);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 104;
        const unsigned int hA_sc = *(const unsigned int*)(gp);
        const unsigned int hB_sc = *(const unsigned int*)(gp + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8); unsigned int pk = (unsigned int)d[boff] | ((unsigned int)d[boff+1] << 8) | ((unsigned int)d[boff+2] << 16);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

#if defined(HIPFIRE_QKV_WITH_BIAS)
    if (tid == 0) y[row] = acc + (bias ? bias[row] : 0.0f);
#else
    if (tid == 0) y[row] = acc;
#endif
}
