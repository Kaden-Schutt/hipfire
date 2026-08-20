// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_RESIDUAL_KERNEL
#define HIPFIRE_RESIDUAL_KERNEL gemv_mq5g256v2_residual
#endif

// MQ5G256V2 residual: y[row] += A[row]·x (168 B/group, 5-bit, dual-fp16)

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_RESIDUAL_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 168;
    const int boff = tid * 5;
    float acc0=0,acc1=0,acc2=0,acc3=0;
    const int quads = groups_per_row >>2;
    const int tail = groups_per_row &3;
    for(int q=0;q<quads;q++){
        const int g=q<<2;
        const char* gp0=row_ptr+g*168;
        const char* gp1=gp0+168;
        const char* gp2=gp1+168;
        const char* gp3=gp2+168;
        const unsigned int hA0=*(const unsigned int*)(gp0);
        const unsigned int hB0=*(const unsigned int*)(gp0+4);
        const unsigned int hs0=(tid<16)?hA0:hB0;
        float sc0=__half2float(__ushort_as_half((unsigned short)(hs0&0xFFFFu)));
        float zp0=__half2float(__ushort_as_half((unsigned short)(hs0>>16)));
        const unsigned int hA1=*(const unsigned int*)(gp1);
        const unsigned int hB1=*(const unsigned int*)(gp1+4);
        const unsigned int hs1=(tid<16)?hA1:hB1;
        float sc1=__half2float(__ushort_as_half((unsigned short)(hs1&0xFFFFu)));
        float zp1=__half2float(__ushort_as_half((unsigned short)(hs1>>16)));
        const unsigned int hA2=*(const unsigned int*)(gp2);
        const unsigned int hB2=*(const unsigned int*)(gp2+4);
        const unsigned int hs2=(tid<16)?hA2:hB2;
        float sc2=__half2float(__ushort_as_half((unsigned short)(hs2&0xFFFFu)));
        float zp2=__half2float(__ushort_as_half((unsigned short)(hs2>>16)));
        const unsigned int hA3=*(const unsigned int*)(gp3);
        const unsigned int hB3=*(const unsigned int*)(gp3+4);
        const unsigned int hs3=(tid<16)?hA3:hB3;
        float sc3=__half2float(__ushort_as_half((unsigned short)(hs3&0xFFFFu)));
        float zp3=__half2float(__ushort_as_half((unsigned short)(hs3>>16)));
        const unsigned char* d0=(const unsigned char*)(gp0+8+boff);
        const unsigned char* d1=(const unsigned char*)(gp1+8+boff);
        const unsigned char* d2=(const unsigned char*)(gp2+8+boff);
        const unsigned char* d3=(const unsigned char*)(gp3+8+boff);
        unsigned char b00=d0[0],b01=d0[1],b02=d0[2],b03=d0[3],b04=d0[4];
        unsigned char b10=d1[0],b11=d1[1],b12=d1[2],b13=d1[3],b14=d1[4];
        unsigned char b20=d2[0],b21=d2[1],b22=d2[2],b23=d2[3],b24=d2[4];
        unsigned char b30=d3[0],b31=d3[1],b32=d3[2],b33=d3[3],b34=d3[4];
        int base=g*256+tid*8;
        #define DOG5(sc,zp,b0,b1,b2,b3,b4, base, acc) \
            acc += (sc*(float)((b0)&0x1f)+zp)*x[(base)] + (sc*(float)( (((b0)>>5)&0x07)|(((b1)&0x03)<<3) )+zp)*x[(base)+1] + (sc*(float)( ((b1)>>2)&0x1f)+zp)*x[(base)+2] + (sc*(float)( (((b1)>>7)&0x01)|(((b2)&0x0f)<<1) )+zp)*x[(base)+3] + (sc*(float)( (((b2)>>4)&0x0f)|(((b3)&0x01)<<4) )+zp)*x[(base)+4] + (sc*(float)( ((b3)>>1)&0x1f)+zp)*x[(base)+5] + (sc*(float)( (((b3)>>6)&0x03)|(((b4)&0x07)<<2) )+zp)*x[(base)+6] + (sc*(float)( ((b4)>>3)&0x1f)+zp)*x[(base)+7]
        DOG5(sc0,zp0,b00,b01,b02,b03,b04, base, acc0);
        DOG5(sc1,zp1,b10,b11,b12,b13,b14, base+256, acc1);
        DOG5(sc2,zp2,b20,b21,b22,b23,b24, base+512, acc2);
        DOG5(sc3,zp3,b30,b31,b32,b33,b34, base+768, acc3);
        #undef DOG5
    }
    #define TAIL5(sc,zp,b0,b1,b2,b3,b4, base, acc) \
        acc += (sc*(float)((b0)&0x1f)+zp)*x[(base)] + (sc*(float)( (((b0)>>5)&0x07)|(((b1)&0x03)<<3) )+zp)*x[(base)+1] + (sc*(float)( ((b1)>>2)&0x1f)+zp)*x[(base)+2] + (sc*(float)( (((b1)>>7)&0x01)|(((b2)&0x0f)<<1) )+zp)*x[(base)+3] + (sc*(float)( (((b2)>>4)&0x0f)|(((b3)&0x01)<<4) )+zp)*x[(base)+4] + (sc*(float)( ((b3)>>1)&0x1f)+zp)*x[(base)+5] + (sc*(float)( (((b3)>>6)&0x03)|(((b4)&0x07)<<2) )+zp)*x[(base)+6] + (sc*(float)( ((b4)>>3)&0x1f)+zp)*x[(base)+7]
    if(tail>=1){
        const int g=quads<<2;
        const char* gp=row_ptr+g*168;
        const unsigned int hA=*(const unsigned int*)(gp);
        const unsigned int hB=*(const unsigned int*)(gp+4);
        const unsigned int hs=(tid<16)?hA:hB;
        float sc=__half2float(__ushort_as_half((unsigned short)(hs&0xFFFFu)));
        float zp=__half2float(__ushort_as_half((unsigned short)(hs>>16)));
        const unsigned char* d=(const unsigned char*)(gp+8+boff);
        unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4];
        int base=g*256+tid*8;
        TAIL5(sc,zp,b0,b1,b2,b3,b4, base, acc0);
    }
    if(tail>=2){
        const int g=(quads<<2)+1;
        const char* gp=row_ptr+g*168;
        const unsigned int hA=*(const unsigned int*)(gp);
        const unsigned int hB=*(const unsigned int*)(gp+4);
        const unsigned int hs=(tid<16)?hA:hB;
        float sc=__half2float(__ushort_as_half((unsigned short)(hs&0xFFFFu)));
        float zp=__half2float(__ushort_as_half((unsigned short)(hs>>16)));
        const unsigned char* d=(const unsigned char*)(gp+8+boff);
        unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4];
        int base=g*256+tid*8;
        TAIL5(sc,zp,b0,b1,b2,b3,b4, base, acc1);
    }
    if(tail>=3){
        const int g=(quads<<2)+2;
        const char* gp=row_ptr+g*168;
        const unsigned int hA=*(const unsigned int*)(gp);
        const unsigned int hB=*(const unsigned int*)(gp+4);
        const unsigned int hs=(tid<16)?hA:hB;
        float sc=__half2float(__ushort_as_half((unsigned short)(hs&0xFFFFu)));
        float zp=__half2float(__ushort_as_half((unsigned short)(hs>>16)));
        const unsigned char* d=(const unsigned char*)(gp+8+boff);
        unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4];
        int base=g*256+tid*8;
        TAIL5(sc,zp,b0,b1,b2,b3,b4, base, acc2);
    }
    #undef TAIL5
    float acc=(acc0+acc1)+(acc2+acc3);
    for(int off=16;off>0;off>>=1) acc+=__shfl_down(acc,off);
    if(tid==0) y[row] += acc;
}
