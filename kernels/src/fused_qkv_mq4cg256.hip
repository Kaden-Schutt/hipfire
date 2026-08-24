// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#endif
#ifndef HIPFIRE_WEIGHT_RSRC
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) (*(const unsigned int*)(ptr))
#endif


#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
using hipfire_qkv_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_qkv_f32x4 = float __attribute__((ext_vector_type(4)));
#endif

#ifndef HIPFIRE_QKV_KERNEL_NAME
#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq4cg256
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
    const int tid = threadIdx.x;
#if defined(HIPFIRE_RDNA3_QKV_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    // Planar MQ4C: each weight tensor is its own blob [payload m*gpr*136][header interleaved].
    // Header for (row,g) lives at group base gp (stride 136 B: 4 B header + 4 B pad + 128 B payload).
    const char* A;
    float* y;
    int row;
    unsigned int row_weight_offset;
    HIPFIRE_QKV_BIAS_DECL
    if (gid < q_m) {
        A = A_q; y = y_q; row = gid; HIPFIRE_QKV_BIAS_Q
        row_weight_offset = (unsigned int)((long long)row * groups_per_row * 136);
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; row = gid - q_m; HIPFIRE_QKV_BIAS_K
        row_weight_offset = (unsigned int)((long long)row * groups_per_row * 136);
    } else {
        A = A_v; y = y_v; row = gid - q_m - k_m; HIPFIRE_QKV_BIAS_V
        row_weight_offset = (unsigned int)((long long)row * groups_per_row * 136);
    }
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
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
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int hdr0__ = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp0, row_weight_offset + g * 136);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hdr0__ & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hdr0__ >> 16)));
        const unsigned int hdr1__ = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp1, row_weight_offset + (g + 1) * 136);
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hdr1__ & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hdr1__ >> 16)));
        const unsigned int hdr2__ = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp2, row_weight_offset + (g + 2) * 136);
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hdr2__ & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hdr2__ >> 16)));
        const unsigned int hdr3__ = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp3, row_weight_offset + (g + 3) * 136);
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hdr3__ & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hdr3__ >> 16)));

        unsigned int pk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp0 + 8 + boff, row_weight_offset + g * 136 + 8 + boff);
        unsigned int pk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp1 + 8 + boff, row_weight_offset + (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp2 + 8 + boff, row_weight_offset + (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp3 + 8 + boff, row_weight_offset + (g + 3) * 136 + 8 + boff);
#else
        const unsigned int hdr0__ = *(const unsigned int*)(gp0);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hdr0__ & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hdr0__ >> 16)));
        const unsigned int hdr1__ = *(const unsigned int*)(gp1);
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hdr1__ & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hdr1__ >> 16)));
        const unsigned int hdr2__ = *(const unsigned int*)(gp2);
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hdr2__ & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hdr2__ >> 16)));
        const unsigned int hdr3__ = *(const unsigned int*)(gp3);
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hdr3__ & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hdr3__ >> 16)));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);
#endif

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
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * (x0) \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (x1) \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (x2) \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (x3) \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (x4) \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (x5) \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (x6) \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (x7)
        DOG_V(pk0, sc0, zp0, acc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
        DOG_V(pk1, sc1, zp1, acc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
        DOG_V(pk2, sc2, zp2, acc2, xv2a.x,xv2a.y,xv2a.z,xv2a.w,xv2b.x,xv2b.y,xv2b.z,xv2b.w);
        DOG_V(pk3, sc3, zp3, acc3, xv3a.x,xv3a.y,xv3a.z,xv3a.w,xv3b.x,xv3b.y,xv3b.z,xv3b.w);
        #undef DOG_V
#else
        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base,       acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
#endif
    }

    // Tail handling: preserve acc[gi] interleaving. See 5302926 bug report.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int hdr__ = *(const unsigned int*)(gp);
        float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        const unsigned int hdr__ = *(const unsigned int*)(gp);
        float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int hdr__ = *(const unsigned int*)(gp);
        float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
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