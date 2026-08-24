// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MQ5G256V2_KERNEL
#define HIPFIRE_MQ5G256V2_KERNEL gemv_mq5g256v2
#endif

#ifndef HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#endif

// MQ5G256V2 GEMV: MagnumQuant V2 (qt=48) dual-fp16 header + 5-bit payload.
// Per-group 168 B: [0..2) fp16 s0, [2..4) fp16 z0, [4..6) fp16 s1, [6..8) fp16 z1,
// [8..168) 160 B 5-bit payload (8 weights per 5 bytes). Half 0 q[0..128), half 1 q[128..256).
// Reconstruction q*f32(s[h])+f32(z[h]). K%256 required. Payload identical to MQ5G256 (qt=31).

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MQ5G256V2_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_MQ5G256V2_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 168;
    const unsigned int row_weight_offset =
        (unsigned int)((long long)row * groups_per_row * 168);
    const int boff = tid * 5;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    #if defined(HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER)
        #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #else
        #define LOAD_WEIGHT_HEADER(ptr, offset) \
            HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, (offset))
    #endif
    #define LOAD_WEIGHT_PACKED5(ptr, offset) \
        HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, (offset))
#else
    #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #define LOAD_WEIGHT_PACKED5(ptr, offset) (*(const unsigned int*)(ptr))
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 168;
        const char* gp1 = gp0 + 168;
        const char* gp2 = gp1 + 168;
        const char* gp3 = gp2 + 168;

        const unsigned int goff = row_weight_offset + (unsigned int)g * 168u;
        const unsigned int hA0 = LOAD_WEIGHT_HEADER(gp0, goff);
        const unsigned int hB0 = LOAD_WEIGHT_HEADER(gp0 + 4, goff + 4u);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = LOAD_WEIGHT_HEADER(gp1, goff + 168u);
        const unsigned int hB1 = LOAD_WEIGHT_HEADER(gp1 + 4, goff + 172u);
        // Actually goff+168 and +172? Wait gp1 offset is goff+168, so gp1+4 is goff+172
        // Fix: recompute correctly
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = LOAD_WEIGHT_HEADER(gp2, goff + 336u);
        const unsigned int hB2 = LOAD_WEIGHT_HEADER(gp2 + 4, goff + 340u);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = LOAD_WEIGHT_HEADER(gp3, goff + 504u);
        const unsigned int hB3 = LOAD_WEIGHT_HEADER(gp3 + 4, goff + 508u);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        // 5-bit payload: 8 weights per 5 bytes per thread
        const unsigned char* d0 = (const unsigned char*)(gp0 + 8 + boff);
        const unsigned char* d1 = (const unsigned char*)(gp1 + 8 + boff);
        const unsigned char* d2 = (const unsigned char*)(gp2 + 8 + boff);
        const unsigned char* d3 = (const unsigned char*)(gp3 + 8 + boff);
        unsigned char b00 = d0[0], b01 = d0[1], b02 = d0[2], b03 = d0[3], b04 = d0[4];
        unsigned char b10 = d1[0], b11 = d1[1], b12 = d1[2], b13 = d1[3], b14 = d1[4];
        unsigned char b20 = d2[0], b21 = d2[1], b22 = d2[2], b23 = d2[3], b24 = d2[4];
        unsigned char b30 = d3[0], b31 = d3[1], b32 = d3[2], b33 = d3[3], b34 = d3[4];

        int base = g * 256 + tid * 8;

        #define DOG5(sc, zp, b0,b1,b2,b3,b4, base, acc) \
            (acc) += (sc * (float)( (b0) & 0x1f) + zp) * x[(base)] \
                   + (sc * (float)( (((b0)>>5)&0x07) | (((b1)&0x03)<<3) ) + zp) * x[(base)+1] \
                   + (sc * (float)( ((b1)>>2) & 0x1f ) + zp) * x[(base)+2] \
                   + (sc * (float)( (((b1)>>7)&0x01) | (((b2)&0x0f)<<1) ) + zp) * x[(base)+3] \
                   + (sc * (float)( (((b2)>>4)&0x0f) | (((b3)&0x01)<<4) ) + zp) * x[(base)+4] \
                   + (sc * (float)( ((b3)>>1) & 0x1f ) + zp) * x[(base)+5] \
                   + (sc * (float)( (((b3)>>6)&0x03) | (((b4)&0x07)<<2) ) + zp) * x[(base)+6] \
                   + (sc * (float)( ((b4)>>3) & 0x1f ) + zp) * x[(base)+7]

        DOG5(sc0, zp0, b00,b01,b02,b03,b04, base, acc0);
        DOG5(sc1, zp1, b10,b11,b12,b13,b14, base + 256, acc1);
        DOG5(sc2, zp2, b20,b21,b22,b23,b24, base + 512, acc2);
        DOG5(sc3, zp3, b30,b31,b32,b33,b34, base + 768, acc3);
        #undef DOG5
    }

    #define TAIL_DOG5(sc, zp, b0,b1,b2,b3,b4, base, acc) \
        (acc) += (sc * (float)( (b0) & 0x1f) + zp) * x[(base)] \
               + (sc * (float)( (((b0)>>5)&0x07) | (((b1)&0x03)<<3) ) + zp) * x[(base)+1] \
               + (sc * (float)( ((b1)>>2) & 0x1f ) + zp) * x[(base)+2] \
               + (sc * (float)( (((b1)>>7)&0x01) | (((b2)&0x0f)<<1) ) + zp) * x[(base)+3] \
               + (sc * (float)( (((b2)>>4)&0x0f) | (((b3)&0x01)<<4) ) + zp) * x[(base)+4] \
               + (sc * (float)( ((b3)>>1) & 0x1f ) + zp) * x[(base)+5] \
               + (sc * (float)( (((b3)>>6)&0x03) | (((b4)&0x07)<<2) ) + zp) * x[(base)+6] \
               + (sc * (float)( ((b4)>>3) & 0x1f ) + zp) * x[(base)+7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 168;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 168u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4];
        int base = g * 256 + tid * 8;
        TAIL_DOG5(sc, zp, b0,b1,b2,b3,b4, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 168;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 168u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4];
        int base = g * 256 + tid * 8;
        TAIL_DOG5(sc, zp, b0,b1,b2,b3,b4, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 168;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 168u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4];
        int base = g * 256 + tid * 8;
        TAIL_DOG5(sc, zp, b0,b1,b2,b3,b4, base, acc2);
    }
    #undef TAIL_DOG5
    #undef LOAD_WEIGHT_PACKED5
    #undef LOAD_WEIGHT_HEADER

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// x-batched variant
#if defined(HIPFIRE_MQ5G256V2_XBATCH)

#ifndef HIPFIRE_MQ5G256V2_XBATCH_MAX
#define HIPFIRE_MQ5G256V2_XBATCH_MAX 4
#endif

__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_MQ5G256V2_XBATCH_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int B
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_MQ5G256V2_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 168;
    const int boff = tid * 5;

    float acc0[HIPFIRE_MQ5G256V2_XBATCH_MAX];
    float acc1[HIPFIRE_MQ5G256V2_XBATCH_MAX];
    float acc2[HIPFIRE_MQ5G256V2_XBATCH_MAX];
    float acc3[HIPFIRE_MQ5G256V2_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_MQ5G256V2_XBATCH_MAX; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 168;
        const char* gp1 = gp0 + 168;
        const char* gp2 = gp1 + 168;
        const char* gp3 = gp2 + 168;

        const unsigned int hA0 = *(const unsigned int*)(gp0);
        const unsigned int hB0 = *(const unsigned int*)(gp0 + 4);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = *(const unsigned int*)(gp1);
        const unsigned int hB1 = *(const unsigned int*)(gp1 + 4);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = *(const unsigned int*)(gp2);
        const unsigned int hB2 = *(const unsigned int*)(gp2 + 4);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = *(const unsigned int*)(gp3);
        const unsigned int hB3 = *(const unsigned int*)(gp3 + 4);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        const unsigned char* d0 = (const unsigned char*)(gp0 + 8 + boff);
        const unsigned char* d1 = (const unsigned char*)(gp1 + 8 + boff);
        const unsigned char* d2 = (const unsigned char*)(gp2 + 8 + boff);
        const unsigned char* d3 = (const unsigned char*)(gp3 + 8 + boff);
        unsigned char b00=d0[0], b01=d0[1], b02=d0[2], b03=d0[3], b04=d0[4];
        unsigned char b10=d1[0], b11=d1[1], b12=d1[2], b13=d1[3], b14=d1[4];
        unsigned char b20=d2[0], b21=d2[1], b22=d2[2], b23=d2[3], b24=d2[4];
        unsigned char b30=d3[0], b31=d3[1], b32=d3[2], b33=d3[3], b34=d3[4];
        int base = g * 256 + tid * 8;
        // Hoist x for each b in inner loop — weight decode hoisted, x varied
        for (int b = 0; b < HIPFIRE_MQ5G256V2_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            #define DOG5_B(sc,zp,b0,b1,b2,b3,b4, xb, base, acc) \
                (acc) += (sc * (float)((b0)&0x1f)+zp)*xb[(base)] \
                       + (sc * (float)( (((b0)>>5)&0x07)|(((b1)&0x03)<<3) )+zp)*xb[(base)+1] \
                       + (sc * (float)( ((b1)>>2)&0x1f)+zp)*xb[(base)+2] \
                       + (sc * (float)( (((b1)>>7)&0x01)|(((b2)&0x0f)<<1) )+zp)*xb[(base)+3] \
                       + (sc * (float)( (((b2)>>4)&0x0f)|(((b3)&0x01)<<4) )+zp)*xb[(base)+4] \
                       + (sc * (float)( ((b3)>>1)&0x1f)+zp)*xb[(base)+5] \
                       + (sc * (float)( (((b3)>>6)&0x03)|(((b4)&0x07)<<2) )+zp)*xb[(base)+6] \
                       + (sc * (float)( ((b4)>>3)&0x1f)+zp)*xb[(base)+7]
            DOG5_B(sc0,zp0,b00,b01,b02,b03,b04, xb, base, acc0[b]);
            DOG5_B(sc1,zp1,b10,b11,b12,b13,b14, xb, base+256, acc1[b]);
            DOG5_B(sc2,zp2,b20,b21,b22,b23,b24, xb, base+512, acc2[b]);
            DOG5_B(sc3,zp3,b30,b31,b32,b33,b34, xb, base+768, acc3[b]);
            #undef DOG5_B
        }
    }
    // tail - simplified: reuse same pattern but per b loop
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 168;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4];
        int base = g * 256 + tid * 8;
        for (int b=0;b<HIPFIRE_MQ5G256V2_XBATCH_MAX;b++){ if(b>=B) break;
            const float* xb = x + (long long)b*K;
            #define TAIL5(sc,zp,b0,b1,b2,b3,b4, xb, base, acc) \
                (acc) += (sc*(float)((b0)&0x1f)+zp)*xb[(base)] \
                       + (sc*(float)( (((b0)>>5)&0x07)|(((b1)&0x03)<<3) )+zp)*xb[(base)+1] \
                       + (sc*(float)( ((b1)>>2)&0x1f)+zp)*xb[(base)+2] \
                       + (sc*(float)( (((b1)>>7)&0x01)|(((b2)&0x0f)<<1) )+zp)*xb[(base)+3] \
                       + (sc*(float)( (((b2)>>4)&0x0f)|(((b3)&0x01)<<4) )+zp)*xb[(base)+4] \
                       + (sc*(float)( ((b3)>>1)&0x1f)+zp)*xb[(base)+5] \
                       + (sc*(float)( (((b3)>>6)&0x03)|(((b4)&0x07)<<2) )+zp)*xb[(base)+6] \
                       + (sc*(float)( ((b4)>>3)&0x1f)+zp)*xb[(base)+7]
            TAIL5(sc,zp,b0,b1,b2,b3,b4, xb, base, acc0[b]);
            #undef TAIL5
        }
    }
    if (tail >= 2) {
        const int g = (quads<<2)+1;
        const char* gp = row_ptr + g * 168;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid<16)?hA:hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4];
        int base = g*256 + tid*8;
        for(int b=0;b<HIPFIRE_MQ5G256V2_XBATCH_MAX;b++){ if(b>=B) break;
            const float* xb = x + (long long)b*K;
            TAIL5(sc,zp,b0,b1,b2,b3,b4, xb, base, acc1[b]);
        }
    }
    if (tail >= 3) {
        const int g = (quads<<2)+2;
        const char* gp = row_ptr + g * 168;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid<16)?hA:hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4];
        int base = g*256 + tid*8;
        for(int b=0;b<HIPFIRE_MQ5G256V2_XBATCH_MAX;b++){ if(b>=B) break;
            const float* xb = x + (long long)b*K;
            TAIL5(sc,zp,b0,b1,b2,b3,b4, xb, base, acc2[b]);
        }
    }

    for (int b=0;b<HIPFIRE_MQ5G256V2_XBATCH_MAX;b++){
        if(b>=B) break;
        float acc = (acc0[b]+acc1[b])+(acc2[b]+acc3[b]);
        for(int offset=16; offset>0; offset>>=1) acc += __shfl_down(acc, offset);
        if(tid==0) y[(long long)b*M + row] = acc;
    }
}
#endif
