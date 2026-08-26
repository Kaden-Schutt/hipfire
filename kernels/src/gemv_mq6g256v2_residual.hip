// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) (*(const unsigned int*)(ptr))
#endif

#ifndef HIPFIRE_RESIDUAL_KERNEL
#define HIPFIRE_RESIDUAL_KERNEL gemv_mq6g256v2_residual
#endif

#if defined(__gfx1200__) || defined(__gfx1201__)
__launch_bounds__(32, 32)
#else
__launch_bounds__(32, 16)
#endif
extern "C" __global__ void HIPFIRE_RESIDUAL_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 6;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#define MQ6_UNPACK_AND_ACC(gp, sc, zp, acc_slot, base_off) do { \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5]; \
        int q0=b0 &63; int q1=(b0>>6)|((b1&0xF)<<2); int q2=(b1>>4)|((b2&3)<<4); int q3=b2>>2; \
        int q4=b3 &63; int q5=(b3>>6)|((b4&0xF)<<2); int q6=(b4>>4)|((b5&3)<<4); int q7=b5>>2; \
        (acc_slot) += (sc*(float)q0+zp)*x[(base_off)] + (sc*(float)q1+zp)*x[(base_off)+1] + (sc*(float)q2+zp)*x[(base_off)+2] + (sc*(float)q3+zp)*x[(base_off)+3] \
                    + (sc*(float)q4+zp)*x[(base_off)+4] + (sc*(float)q5+zp)*x[(base_off)+5] + (sc*(float)q6+zp)*x[(base_off)+6] + (sc*(float)q7+zp)*x[(base_off)+7]; \
    } while(0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        // Load headers dual-row
        const unsigned int hA0a = *(const unsigned int*)(row_ptr0 + g*200);
        const unsigned int hB0a = *(const unsigned int*)(row_ptr0 + g*200 +4);
        float sc0a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA0a:hB0a) &0xFFFFu)));
        float zp0a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA0a:hB0a)>>16)));
        const unsigned int hA0b = *(const unsigned int*)(row_ptr1 + g*200);
        const unsigned int hB0b = *(const unsigned int*)(row_ptr1 + g*200 +4);
        float sc0b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA0b:hB0b) &0xFFFFu)));
        float zp0b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA0b:hB0b)>>16)));

        const unsigned int hA1a = *(const unsigned int*)(row_ptr0 + (g+1)*200);
        const unsigned int hB1a = *(const unsigned int*)(row_ptr0 + (g+1)*200+4);
        float sc1a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA1a:hB1a) &0xFFFFu)));
        float zp1a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA1a:hB1a)>>16)));
        const unsigned int hA1b = *(const unsigned int*)(row_ptr1 + (g+1)*200);
        const unsigned int hB1b = *(const unsigned int*)(row_ptr1 + (g+1)*200+4);
        float sc1b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA1b:hB1b) &0xFFFFu)));
        float zp1b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA1b:hB1b)>>16)));

        const unsigned int hA2a = *(const unsigned int*)(row_ptr0 + (g+2)*200);
        const unsigned int hB2a = *(const unsigned int*)(row_ptr0 + (g+2)*200+4);
        float sc2a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA2a:hB2a) &0xFFFFu)));
        float zp2a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA2a:hB2a)>>16)));
        const unsigned int hA2b = *(const unsigned int*)(row_ptr1 + (g+2)*200);
        const unsigned int hB2b = *(const unsigned int*)(row_ptr1 + (g+2)*200+4);
        float sc2b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA2b:hB2b) &0xFFFFu)));
        float zp2b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA2b:hB2b)>>16)));

        const unsigned int hA3a = *(const unsigned int*)(row_ptr0 + (g+3)*200);
        const unsigned int hB3a = *(const unsigned int*)(row_ptr0 + (g+3)*200+4);
        float sc3a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA3a:hB3a) &0xFFFFu)));
        float zp3a = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA3a:hB3a)>>16)));
        const unsigned int hA3b = *(const unsigned int*)(row_ptr1 + (g+3)*200);
        const unsigned int hB3b = *(const unsigned int*)(row_ptr1 + (g+3)*200+4);
        float sc3b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA3b:hB3b) &0xFFFFu)));
        float zp3b = __half2float(__ushort_as_half((unsigned short)(((tid<16)?hA3b:hB3b)>>16)));

        // Use x hoist via direct loads for simplicity: we already have xv* but need per element mapping
        // We'll do inline acc using xv* values
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc0 += (sc0a*(float)q0+zp0a)*xv0a.x + (sc0a*(float)q1+zp0a)*xv0a.y + (sc0a*(float)q2+zp0a)*xv0a.z + (sc0a*(float)q3+zp0a)*xv0a.w
                  + (sc0a*(float)q4+zp0a)*xv0b.x + (sc0a*(float)q5+zp0a)*xv0b.y + (sc0a*(float)q6+zp0a)*xv0b.z + (sc0a*(float)q7+zp0a)*xv0b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc0 += (sc0b*(float)q0+zp0b)*xv0a.x + (sc0b*(float)q1+zp0b)*xv0a.y + (sc0b*(float)q2+zp0b)*xv0a.z + (sc0b*(float)q3+zp0b)*xv0a.w
                  + (sc0b*(float)q4+zp0b)*xv0b.x + (sc0b*(float)q5+zp0b)*xv0b.y + (sc0b*(float)q6+zp0b)*xv0b.z + (sc0b*(float)q7+zp0b)*xv0b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + (g+1)*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc1 += (sc1a*(float)q0+zp1a)*xv1a.x + (sc1a*(float)q1+zp1a)*xv1a.y + (sc1a*(float)q2+zp1a)*xv1a.z + (sc1a*(float)q3+zp1a)*xv1a.w
                  + (sc1a*(float)q4+zp1a)*xv1b.x + (sc1a*(float)q5+zp1a)*xv1b.y + (sc1a*(float)q6+zp1a)*xv1b.z + (sc1a*(float)q7+zp1a)*xv1b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + (g+1)*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc1 += (sc1b*(float)q0+zp1b)*xv1a.x + (sc1b*(float)q1+zp1b)*xv1a.y + (sc1b*(float)q2+zp1b)*xv1a.z + (sc1b*(float)q3+zp1b)*xv1a.w
                  + (sc1b*(float)q4+zp1b)*xv1b.x + (sc1b*(float)q5+zp1b)*xv1b.y + (sc1b*(float)q6+zp1b)*xv1b.z + (sc1b*(float)q7+zp1b)*xv1b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + (g+2)*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc2 += (sc2a*(float)q0+zp2a)*xv2a.x + (sc2a*(float)q1+zp2a)*xv2a.y + (sc2a*(float)q2+zp2a)*xv2a.z + (sc2a*(float)q3+zp2a)*xv2a.w
                  + (sc2a*(float)q4+zp2a)*xv2b.x + (sc2a*(float)q5+zp2a)*xv2b.y + (sc2a*(float)q6+zp2a)*xv2b.z + (sc2a*(float)q7+zp2a)*xv2b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + (g+2)*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc2 += (sc2b*(float)q0+zp2b)*xv2a.x + (sc2b*(float)q1+zp2b)*xv2a.y + (sc2b*(float)q2+zp2b)*xv2a.z + (sc2b*(float)q3+zp2b)*xv2a.w
                  + (sc2b*(float)q4+zp2b)*xv2b.x + (sc2b*(float)q5+zp2b)*xv2b.y + (sc2b*(float)q6+zp2b)*xv2b.z + (sc2b*(float)q7+zp2b)*xv2b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + (g+3)*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc3 += (sc3a*(float)q0+zp3a)*xv3a.x + (sc3a*(float)q1+zp3a)*xv3a.y + (sc3a*(float)q2+zp3a)*xv3a.z + (sc3a*(float)q3+zp3a)*xv3a.w
                  + (sc3a*(float)q4+zp3a)*xv3b.x + (sc3a*(float)q5+zp3a)*xv3b.y + (sc3a*(float)q6+zp3a)*xv3b.z + (sc3a*(float)q7+zp3a)*xv3b.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + (g+3)*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc3 += (sc3b*(float)q0+zp3b)*xv3a.x + (sc3b*(float)q1+zp3b)*xv3a.y + (sc3b*(float)q2+zp3b)*xv3a.z + (sc3b*(float)q3+zp3b)*xv3a.w
                  + (sc3b*(float)q4+zp3b)*xv3b.x + (sc3b*(float)q5+zp3b)*xv3b.y + (sc3b*(float)q6+zp3b)*xv3b.z + (sc3b*(float)q7+zp3b)*xv3b.w;
        }
    }

    // tails (up to 3 groups)
    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        const unsigned int ha0 = *(const unsigned int*)(row_ptr0 + g*200);
        const unsigned int hb0 = *(const unsigned int*)(row_ptr0 + g*200+4);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha0:hb0) &0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha0:hb0)>>16)));
        const unsigned int ha1 = *(const unsigned int*)(row_ptr1 + g*200);
        const unsigned int hb1 = *(const unsigned int*)(row_ptr1 + g*200+4);
        float sc1 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha1:hb1) &0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha1:hb1)>>16)));
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc0 += (sc0*(float)q0+zp0)*xa.x + (sc0*(float)q1+zp0)*xa.y + (sc0*(float)q2+zp0)*xa.z + (sc0*(float)q3+zp0)*xa.w + (sc0*(float)q4+zp0)*xb.x + (sc0*(float)q5+zp0)*xb.y + (sc0*(float)q6+zp0)*xb.z + (sc0*(float)q7+zp0)*xb.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc0 += (sc1*(float)q0+zp1)*xa.x + (sc1*(float)q1+zp1)*xa.y + (sc1*(float)q2+zp1)*xa.z + (sc1*(float)q3+zp1)*xa.w + (sc1*(float)q4+zp1)*xb.x + (sc1*(float)q5+zp1)*xb.y + (sc1*(float)q6+zp1)*xb.z + (sc1*(float)q7+zp1)*xb.w;
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        const unsigned int ha0 = *(const unsigned int*)(row_ptr0 + g*200);
        const unsigned int hb0 = *(const unsigned int*)(row_ptr0 + g*200+4);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha0:hb0) &0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha0:hb0)>>16)));
        const unsigned int ha1 = *(const unsigned int*)(row_ptr1 + g*200);
        const unsigned int hb1 = *(const unsigned int*)(row_ptr1 + g*200+4);
        float sc1 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha1:hb1) &0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha1:hb1)>>16)));
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc1 += (sc0*(float)q0+zp0)*xa.x + (sc0*(float)q1+zp0)*xa.y + (sc0*(float)q2+zp0)*xa.z + (sc0*(float)q3+zp0)*xa.w + (sc0*(float)q4+zp0)*xb.x + (sc0*(float)q5+zp0)*xb.y + (sc0*(float)q6+zp0)*xb.z + (sc0*(float)q7+zp0)*xb.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc1 += (sc1*(float)q0+zp1)*xa.x + (sc1*(float)q1+zp1)*xa.y + (sc1*(float)q2+zp1)*xa.z + (sc1*(float)q3+zp1)*xa.w + (sc1*(float)q4+zp1)*xb.x + (sc1*(float)q5+zp1)*xb.y + (sc1*(float)q6+zp1)*xb.z + (sc1*(float)q7+zp1)*xb.w;
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        const unsigned int ha0 = *(const unsigned int*)(row_ptr0 + g*200);
        const unsigned int hb0 = *(const unsigned int*)(row_ptr0 + g*200+4);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha0:hb0) &0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha0:hb0)>>16)));
        const unsigned int ha1 = *(const unsigned int*)(row_ptr1 + g*200);
        const unsigned int hb1 = *(const unsigned int*)(row_ptr1 + g*200+4);
        float sc1 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha1:hb1) &0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(((tid<16)?ha1:hb1)>>16)));
        {
            const unsigned char* d = (const unsigned char*)(row_ptr0 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            acc2 += (sc0*(float)q0+zp0)*xa.x + (sc0*(float)q1+zp0)*xa.y + (sc0*(float)q2+zp0)*xa.z + (sc0*(float)q3+zp0)*xa.w + (sc0*(float)q4+zp0)*xb.x + (sc0*(float)q5+zp0)*xb.y + (sc0*(float)q6+zp0)*xb.z + (sc0*(float)q7+zp0)*xb.w;
        }
        {
            const unsigned char* d = (const unsigned char*)(row_ptr1 + g*200 + 8 + boff);
            unsigned char b0=d[0], b1=d[1], b2=d[2], b3=d[3], b4=d[4], b5=d[5];
            int q0=b0 &63, q1=(b0>>6)|((b1&0xF)<<2), q2=(b1>>4)|((b2&3)<<4), q3=b2>>2, q4=b3 &63, q5=(b3>>6)|((b4&0xF)<<2), q6=(b4>>4)|((b5&3)<<4), q7=b5>>2;
            bcc2 += (sc1*(float)q0+zp1)*xa.x + (sc1*(float)q1+zp1)*xa.y + (sc1*(float)q2+zp1)*xa.z + (sc1*(float)q3+zp1)*xa.w + (sc1*(float)q4+zp1)*xb.x + (sc1*(float)q5+zp1)*xb.y + (sc1*(float)q6+zp1)*xb.z + (sc1*(float)q7+zp1)*xb.w;
        }
    }
#undef MQ6_UNPACK_AND_ACC
    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1) bcc += __shfl_down(bcc, offset);
    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += acc;
            yy.y += bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += acc;
        }
    }
}
