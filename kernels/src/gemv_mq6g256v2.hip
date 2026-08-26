// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MQ6G256V2_KERNEL
#define HIPFIRE_MQ6G256V2_KERNEL gemv_mq6g256v2
#endif

#ifndef HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#endif

// MQ6-G256 v2 GEMV: neutral Magnum V2 (qt=47).
// Per-group 200 B: [0..2) fp16 s0, [2..4) fp16 z0, [4..6) fp16 s1, [6..8) fp16 z1,
// [8..200) 192 B legacy 6-bit payload (4 weights per 3 bytes, 4/3 B).
// Half 0 governs q[0..128), half 1 q[128..256); reconstruction q*f32(s[h])+f32(z[h]).
// K%256 required. Same 4-accumulator interleaving as gemv_mq4g256v2 for byte-exact.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MQ6G256V2_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_MQ6G256V2_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const unsigned int row_weight_offset =
        (unsigned int)((long long)row * groups_per_row * 200);
    const int boff = tid * 6;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    #if defined(HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER)
        #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #else
        #define LOAD_WEIGHT_HEADER(ptr, offset) \
            HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, (offset))
    #endif
    #define LOAD_WEIGHT_PACKED_BYTE(ptr, offset) HIPFIRE_WEIGHT_LOAD_U8(weight_rsrc, (offset))
#else
    #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #define LOAD_WEIGHT_PACKED_BYTE(ptr, offset) (*(const unsigned char*)(ptr))
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const char* gp1 = gp0 + 200;
        const char* gp2 = gp1 + 200;
        const char* gp3 = gp2 + 200;

        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA0 = LOAD_WEIGHT_HEADER(gp0, goff);
        const unsigned int hB0 = LOAD_WEIGHT_HEADER(gp0 + 4, goff + 4u);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = LOAD_WEIGHT_HEADER(gp1, goff + 200u);
        const unsigned int hB1 = LOAD_WEIGHT_HEADER(gp1 + 4, goff + 204u);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = LOAD_WEIGHT_HEADER(gp2, goff + 400u);
        const unsigned int hB2 = LOAD_WEIGHT_HEADER(gp2 + 4, goff + 404u);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = LOAD_WEIGHT_HEADER(gp3, goff + 600u);
        const unsigned int hB3 = LOAD_WEIGHT_HEADER(gp3 + 4, goff + 604u);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        int base = g * 256 + tid * 8;

#define MQ6_DOG(gp, goff_base, sc, zp, b, a) do { \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        int q0 = b0 & 63; \
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2); \
        int q2 = (b1 >> 4) | ((b2 & 3) << 4); \
        int q3 = b2 >> 2; \
        int q4 = b3 & 63; \
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2); \
        int q6 = (b4 >> 4) | ((b5 & 3) << 4); \
        int q7 = b5 >> 2; \
        (a) += (sc * (float)q0 + zp) * x[(b)] \
             + (sc * (float)q1 + zp) * x[(b) + 1] \
             + (sc * (float)q2 + zp) * x[(b) + 2] \
             + (sc * (float)q3 + zp) * x[(b) + 3] \
             + (sc * (float)q4 + zp) * x[(b) + 4] \
             + (sc * (float)q5 + zp) * x[(b) + 5] \
             + (sc * (float)q6 + zp) * x[(b) + 6] \
             + (sc * (float)q7 + zp) * x[(b) + 7]; \
    } while(0)

        MQ6_DOG(gp0, goff, sc0, zp0, base, acc0);
        MQ6_DOG(gp1, goff + 200u, sc1, zp1, base + 256, acc1);
        MQ6_DOG(gp2, goff + 400u, sc2, zp2, base + 512, acc2);
        MQ6_DOG(gp3, goff + 600u, sc3, zp3, base + 768, acc3);
#undef MQ6_DOG
    }

#define MQ6_TAIL_DOG(gp, goff_, sc, zp, b, a) do { \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        int q0 = b0 & 63; \
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2); \
        int q2 = (b1 >> 4) | ((b2 & 3) << 4); \
        int q3 = b2 >> 2; \
        int q4 = b3 & 63; \
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2); \
        int q6 = (b4 >> 4) | ((b5 & 3) << 4); \
        int q7 = b5 >> 2; \
        (a) += (sc * (float)q0 + zp) * x[(b)] \
             + (sc * (float)q1 + zp) * x[(b) + 1] \
             + (sc * (float)q2 + zp) * x[(b) + 2] \
             + (sc * (float)q3 + zp) * x[(b) + 3] \
             + (sc * (float)q4 + zp) * x[(b) + 4] \
             + (sc * (float)q5 + zp) * x[(b) + 5] \
             + (sc * (float)q6 + zp) * x[(b) + 6] \
             + (sc * (float)q7 + zp) * x[(b) + 7]; \
    } while(0)

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 200;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        int base = g * 256 + tid * 8;
        MQ6_TAIL_DOG(gp, goff, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 200;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        int base = g * 256 + tid * 8;
        MQ6_TAIL_DOG(gp, goff, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 200;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        int base = g * 256 + tid * 8;
        MQ6_TAIL_DOG(gp, goff, sc, zp, base, acc2);
    }
#undef MQ6_TAIL_DOG
#undef LOAD_WEIGHT_PACKED_BYTE
#undef LOAD_WEIGHT_HEADER

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// x-batched variant not needed for MQ6; dense Qwen uses decode + prefill paths above.
