// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

// HFQ4-G256 GEMV with fused SCALED residual add: y[row] += scale * (A[row] · x).
//
// Two kernel variants — identical math, differ only in how the scale is
// delivered:
//   _scaled_cpu     — scale is a kernarg f32, passed from the host. Used
//                     for the MoE routed-expert accumulator (weight comes
//                     from CPU top-K selection).
//   _scaled_gpu     — scale is read from c_buf[0] of a 1-element device
//                     buffer. Used for the MoE shared-expert accumulator
//                     (scale = sigmoid(shared_expert_gate · x), computed
//                     entirely on-device to avoid a D2H sync).
//
// Identical math to gemv_hfq4g256_residual except the final write multiplies
// acc by `scale` before the += — bit-exact with running the baseline kernel
// followed by an independent scaled_add.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_scaled_cpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float scale
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += scale * acc;   // scaled residual add
}

// GPU-scalar variant — identical except `scale` is loaded from a device
// buffer. Avoids the D2H sync that a CPU-scalar version would force when
// the scale is already computed on-device (e.g. sigmoid(gate)).
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_scaled_gpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ c_buf,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += c_buf[0] * acc;   // scaled residual add, gpu-scalar
}

// Variant: same math as _scaled_gpu but applies sigmoid to c_buf[0] before
// scaling. Lets the caller skip a separate `sigmoid_f32` launch on the 1-elem
// shared-expert gate scalar — one less launch per MoE layer (~40 launches
// saved per forward on A3B). Bit-equivalent to sigmoid_f32 followed by
// _scaled_gpu up to single-precision float precision.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_sigmoid_scaled_gpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ c_buf,
    int M, int K
) {
#if HIPFIRE_RDNA3_SIGMOID_ROWS4
    // gfx1100 R4: amortize each activation fetch across four independent
    // output rows. The per-row four-chain accumulation order is unchanged
    // from R2; only the amount of row-level reuse inside a wave increases.
    const int row0 = blockIdx.x * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const bool have_row3 = row3 < M;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
    const char* row_ptr3 = A + (long long)(have_row3 ? row3 : row0) * row_stride;
    const int boff = tid * 4;
    const auto row_rsrc0 = HIPFIRE_WEIGHT_RSRC(row_ptr0);
    const auto row_rsrc1 = HIPFIRE_WEIGHT_RSRC(row_ptr1);
    const auto row_rsrc2 = HIPFIRE_WEIGHT_RSRC(row_ptr2);
    const auto row_rsrc3 = HIPFIRE_WEIGHT_RSRC(row_ptr3);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    float dcc0 = 0.0f, dcc1 = 0.0f, dcc2 = 0.0f, dcc3 = 0.0f;
    float ecc0 = 0.0f, ecc1 = 0.0f, ecc2 = 0.0f, ecc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8_R4(b) \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]

    #define DOG_X8_R4(rsrc, gp, group_offset, a) do { \
        float sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32((rsrc), (gp), (group_offset))); \
        float zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32((rsrc), (gp) + 4, (group_offset) + 4)); \
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            (rsrc), (gp) + 8 + boff, (group_offset) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    #define DOG_ROWS4(group_index, chain) do { \
        const int goff = (group_index) * 136; \
        DOG_X8_R4(row_rsrc0, row_ptr0 + goff, goff, acc##chain); \
        if (have_row1) DOG_X8_R4(row_rsrc1, row_ptr1 + goff, goff, bcc##chain); \
        if (have_row2) DOG_X8_R4(row_rsrc2, row_ptr2 + goff, goff, dcc##chain); \
        if (have_row3) DOG_X8_R4(row_rsrc3, row_ptr3 + goff, goff, ecc##chain); \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;
        { LOAD_X8_R4(base0);       DOG_ROWS4(g, 0); }
        { LOAD_X8_R4(base0 + 256); DOG_ROWS4(g + 1, 1); }
        { LOAD_X8_R4(base0 + 512); DOG_ROWS4(g + 2, 2); }
        { LOAD_X8_R4(base0 + 768); DOG_ROWS4(g + 3, 3); }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8_R4(base);
        DOG_ROWS4(g, 0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8_R4(base);
        DOG_ROWS4(g, 1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8_R4(base);
        DOG_ROWS4(g, 2);
    }
    #undef DOG_ROWS4
    #undef DOG_X8_R4
    #undef LOAD_X8_R4

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    float dcc = (dcc0 + dcc1) + (dcc2 + dcc3);
    float ecc = (ecc0 + ecc1) + (ecc2 + ecc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
        bcc += __shfl_down(bcc, offset);
        dcc += __shfl_down(dcc, offset);
        ecc += __shfl_down(ecc, offset);
    }

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_buf[0]));
        if (have_row3) {
            float4 yy = *reinterpret_cast<float4*>(y + row0);
            yy.x += gate * acc;
            yy.y += gate * bcc;
            yy.z += gate * dcc;
            yy.w += gate * ecc;
            *reinterpret_cast<float4*>(y + row0) = yy;
        } else {
            y[row0] += gate * acc;
            if (have_row1) y[row1] += gate * bcc;
            if (have_row2) y[row2] += gate * dcc;
        }
    }
#else
    // R14c3_sig_r2_vrw: keep the confirmed two-row x/gate reuse mapping,
    // then update the paired residual rows with one aligned float2
    // read-modify-write. This probes output access width without extending
    // the row-reuse ladder into the known r4 noise/loss zone.
    const int row0 = blockIdx.x * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
#if HIPFIRE_RDNA3_SIGMOID_M2048
    const bool have_row1 = true;
#else
    const bool have_row1 = row1 < M;
#endif
    const int tid = threadIdx.x;

#if HIPFIRE_RDNA3_SIGMOID_K512
    // The gfx1100 A3B shared-down decode shape is fixed at K=512. Keeping the
    // public ABI unchanged lets replay reuse the same kernarg while LLVM can
    // delete the empty quad loop and dynamic tail predicates.
    const int groups_per_row = 2;
#else
    const int groups_per_row = K / 256;
#endif
#if HIPFIRE_RDNA3_SIGMOID_M2048
    const long long row_stride = 272;
#else
    const long long row_stride = (long long)groups_per_row * 136;
#endif
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 4;
    const auto row_rsrc0 = HIPFIRE_WEIGHT_RSRC(row_ptr0);
    const auto row_rsrc1 = HIPFIRE_WEIGHT_RSRC(row_ptr1);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if HIPFIRE_RDNA3_SIGMOID_HOIST_X16
    // gfx1100 fixed-shape producer/consumer schedule: make both activation
    // groups resident before issuing either row's weight stream. This gives
    // LLVM independent VMEM to overlap with the two row-header phases while
    // preserving each accumulator's exact group and FMA order.
    const int base0 = tid * 8;
    const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
    const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
    const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
    const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);

    #define DOG_X8_FIXED(rsrc, gp, group_offset, a, xa, xb) do { \
        float sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32((rsrc), (gp), (group_offset))); \
        float zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32((rsrc), (gp) + 4, (group_offset) + 4)); \
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            (rsrc), (gp) + 8 + boff, (group_offset) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)          + zp) * (xa).x \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (xa).y \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (xa).z \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (xa).w \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (xb).x \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (xb).y \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (xb).z \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (xb).w; \
    } while (0)

    DOG_X8_FIXED(row_rsrc0, row_ptr0, 0, acc0, xv0a, xv0b);
    DOG_X8_FIXED(row_rsrc1, row_ptr1, 0, bcc0, xv0a, xv0b);
    DOG_X8_FIXED(row_rsrc0, row_ptr0 + 136, 136, acc1, xv1a, xv1b);
    DOG_X8_FIXED(row_rsrc1, row_ptr1 + 136, 136, bcc1, xv1a, xv1b);
    #undef DOG_X8_FIXED
#else
    #define LOAD_X8(b) \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]

    #define DOG_X8(rsrc, gp, group_offset, a) do { \
        float sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32((rsrc), (gp), (group_offset))); \
        float zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32((rsrc), (gp) + 4, (group_offset) + 4)); \
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            (rsrc), (gp) + 8 + boff, (group_offset) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        {
            LOAD_X8(base0);
            DOG_X8(row_rsrc0, row_ptr0 + g * 136, g * 136, acc0);
            if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + g * 136, g * 136, bcc0);
        }
        {
            LOAD_X8(base0 + 256);
            DOG_X8(row_rsrc0, row_ptr0 + (g + 1) * 136, (g + 1) * 136, acc1);
            if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + (g + 1) * 136, (g + 1) * 136, bcc1);
        }
        {
            LOAD_X8(base0 + 512);
            DOG_X8(row_rsrc0, row_ptr0 + (g + 2) * 136, (g + 2) * 136, acc2);
            if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + (g + 2) * 136, (g + 2) * 136, bcc2);
        }
        {
            LOAD_X8(base0 + 768);
            DOG_X8(row_rsrc0, row_ptr0 + (g + 3) * 136, (g + 3) * 136, acc3);
            if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + (g + 3) * 136, (g + 3) * 136, bcc3);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_rsrc0, row_ptr0 + g * 136, g * 136, acc0);
        if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + g * 136, g * 136, bcc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_rsrc0, row_ptr0 + g * 136, g * 136, acc1);
        if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + g * 136, g * 136, bcc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_rsrc0, row_ptr0 + g * 136, g * 136, acc2);
        if (have_row1) DOG_X8(row_rsrc1, row_ptr1 + g * 136, g * 136, bcc2);
    }
    #undef DOG_X8
    #undef LOAD_X8
#endif

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_buf[0]));
#if HIPFIRE_RDNA3_SIGMOID_M2048
        float2 yy = *reinterpret_cast<float2*>(y + row0);
        yy.x += gate * acc;
        yy.y += gate * bcc;
        *reinterpret_cast<float2*>(y + row0) = yy;
#else
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += gate * acc;
            yy.y += gate * bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += gate * acc;
        }
#endif
    }
#endif
}

// N-batched variant of `gemv_hfq4g256_residual_sigmoid_scaled_gpu`.
// Grid = (M, N, 1). Each (row, token) block runs the same HFQ4G256
// GEMV body over the token's rotated input row; the final write applies
// sigmoid(c_buf[token]) as the scale and atomicAdd's into y[token×M..].
//
// Used by the batched MoE FFN's shared-expert down projection to avoid
// launching the single-token sigmoid_scaled kernel N times per MoE layer.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
    const char* __restrict__ A,
    const float* __restrict__ x_batch,     // [N × K]
    float* __restrict__ y_batch,           // [N × M]
    const float* __restrict__ c_batch,     // [N]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bid = blockIdx.y;
    const int tid = threadIdx.x;

    const float* __restrict__ x = x_batch + (long long)bid * K;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_batch[bid]));
        y_batch[(long long)bid * M + row] += gate * acc;
    }
}
