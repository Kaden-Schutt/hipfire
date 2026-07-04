// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

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
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row0_ptr = A + (long long)row0 * row_bytes;
    const char* row1_ptr = row1 < M ? row0_ptr + row_bytes : row0_ptr;
    const int boff = tid * 4;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* g00 = row0_ptr + g * 136;
        const char* g01 = g00 + 136;
        const char* g02 = g01 + 136;
        const char* g03 = g02 + 136;
        const char* g10 = row1_ptr + g * 136;
        const char* g11 = g10 + 136;
        const char* g12 = g11 + 136;
        const char* g13 = g12 + 136;

        float sc00 = __builtin_bit_cast(float, *(const unsigned int*)(g00));
        float zp00 = __builtin_bit_cast(float, *(const unsigned int*)(g00 + 4));
        float sc01 = __builtin_bit_cast(float, *(const unsigned int*)(g01));
        float zp01 = __builtin_bit_cast(float, *(const unsigned int*)(g01 + 4));
        float sc02 = __builtin_bit_cast(float, *(const unsigned int*)(g02));
        float zp02 = __builtin_bit_cast(float, *(const unsigned int*)(g02 + 4));
        float sc03 = __builtin_bit_cast(float, *(const unsigned int*)(g03));
        float zp03 = __builtin_bit_cast(float, *(const unsigned int*)(g03 + 4));
        float sc10 = __builtin_bit_cast(float, *(const unsigned int*)(g10));
        float zp10 = __builtin_bit_cast(float, *(const unsigned int*)(g10 + 4));
        float sc11 = __builtin_bit_cast(float, *(const unsigned int*)(g11));
        float zp11 = __builtin_bit_cast(float, *(const unsigned int*)(g11 + 4));
        float sc12 = __builtin_bit_cast(float, *(const unsigned int*)(g12));
        float zp12 = __builtin_bit_cast(float, *(const unsigned int*)(g12 + 4));
        float sc13 = __builtin_bit_cast(float, *(const unsigned int*)(g13));
        float zp13 = __builtin_bit_cast(float, *(const unsigned int*)(g13 + 4));

        unsigned int pk00 = *(const unsigned int*)(g00 + 8 + boff);
        unsigned int pk01 = *(const unsigned int*)(g01 + 8 + boff);
        unsigned int pk02 = *(const unsigned int*)(g02 + 8 + boff);
        unsigned int pk03 = *(const unsigned int*)(g03 + 8 + boff);
        unsigned int pk10 = *(const unsigned int*)(g10 + 8 + boff);
        unsigned int pk11 = *(const unsigned int*)(g11 + 8 + boff);
        unsigned int pk12 = *(const unsigned int*)(g12 + 8 + boff);
        unsigned int pk13 = *(const unsigned int*)(g13 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG2(pk0, sc0, zp0, pk1, sc1, zp1, b, a0, a1) do { \
            float xv = x[(b)]; \
            (a0) += ((sc0) * (float)((pk0) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)((pk1) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 1]; \
            (a0) += ((sc0) * (float)(((pk0) >> 4) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 4) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 2]; \
            (a0) += ((sc0) * (float)(((pk0) >> 8) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 8) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 3]; \
            (a0) += ((sc0) * (float)(((pk0) >> 12) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 12) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 4]; \
            (a0) += ((sc0) * (float)(((pk0) >> 16) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 16) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 5]; \
            (a0) += ((sc0) * (float)(((pk0) >> 20) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 20) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 6]; \
            (a0) += ((sc0) * (float)(((pk0) >> 24) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 24) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 7]; \
            (a0) += ((sc0) * (float)(((pk0) >> 28) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 28) & 0xFu) + (zp1)) * xv; \
        } while (0)

        DOG2(pk00, sc00, zp00, pk10, sc10, zp10, base,       acc00, acc10);
        DOG2(pk01, sc01, zp01, pk11, sc11, zp11, base + 256, acc01, acc11);
        DOG2(pk02, sc02, zp02, pk12, sc12, zp12, base + 512, acc02, acc12);
        DOG2(pk03, sc03, zp03, pk13, sc13, zp13, base + 768, acc03, acc13);
        #undef DOG2
    }

    #define TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, b, a0, a1) do { \
        float xv = x[(b)]; \
        (a0) += ((sc0) * (float)((pk0) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)((pk1) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 1]; \
        (a0) += ((sc0) * (float)(((pk0) >> 4) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 4) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 2]; \
        (a0) += ((sc0) * (float)(((pk0) >> 8) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 8) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 3]; \
        (a0) += ((sc0) * (float)(((pk0) >> 12) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 12) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 4]; \
        (a0) += ((sc0) * (float)(((pk0) >> 16) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 16) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 5]; \
        (a0) += ((sc0) * (float)(((pk0) >> 20) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 20) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 6]; \
        (a0) += ((sc0) * (float)(((pk0) >> 24) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 24) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 7]; \
        (a0) += ((sc0) * (float)(((pk0) >> 28) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 28) & 0xFu) + (zp1)) * xv; \
    } while (0)

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp0 = row0_ptr + g * 136;
        const char* gp1 = row1_ptr + g * 136;
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, base, acc00, acc10);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp0 = row0_ptr + g * 136;
        const char* gp1 = row1_ptr + g * 136;
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, base, acc01, acc11);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp0 = row0_ptr + g * 136;
        const char* gp1 = row1_ptr + g * 136;
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, base, acc02, acc12);
    }
    #undef TAIL_DOG2

    float acc0 = (acc00 + acc01) + (acc02 + acc03);
    float acc1 = (acc10 + acc11) + (acc12 + acc13);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down(acc0, offset);
        acc1 += __shfl_down(acc1, offset);
    }

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_buf[0]));
        y[row0] += gate * acc0;
        if (row1 < M) y[row1] += gate * acc1;
    }
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
