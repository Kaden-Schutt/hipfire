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

// MQ5G256V2 multi-row GEMV: one warp computes R output rows.
// Mirrors gemv_mq4g256v2_multirow.hip with 168 B stride and 5-bit payload.
// R ∈ {2,4,8}, grid = ceil(M/R), block 32.

template<int R>
__device__ __forceinline__ void gemv_mq5g256v2_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 168;
    const int boff = tid * 5;
    const char* row_ptrs[R];
    #pragma unroll
    for (int r=0;r<R;r++){
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }
    float acc[R][4];
    #pragma unroll
    for(int r=0;r<R;r++) acc[r][0]=acc[r][1]=acc[r][2]=acc[r][3]=0.0f;
    const int quads = groups_per_row >>2;
    const int tail = groups_per_row &3;
    for(int q=0;q<quads;q++){
        const int g = q<<2;
        const int base = g*256 + tid*8;
        const float x00=x[base+0], x01=x[base+1], x02=x[base+2], x03=x[base+3], x04=x[base+4], x05=x[base+5], x06=x[base+6], x07=x[base+7];
        const float x10=x[base+256+0], x11=x[base+256+1], x12=x[base+256+2], x13=x[base+256+3], x14=x[base+256+4], x15=x[base+256+5], x16=x[base+256+6], x17=x[base+256+7];
        const float x20=x[base+512+0], x21=x[base+512+1], x22=x[base+512+2], x23=x[base+512+3], x24=x[base+512+4], x25=x[base+512+5], x26=x[base+512+6], x27=x[base+512+7];
        const float x30=x[base+768+0], x31=x[base+768+1], x32=x[base+768+2], x33=x[base+768+3], x34=x[base+768+4], x35=x[base+768+5], x36=x[base+768+6], x37=x[base+768+7];
        #pragma unroll
        for(int r=0;r<R;r++){
            const char* gp0 = row_ptrs[r] + g*168;
            const char* gp1 = gp0 + 168;
            const char* gp2 = gp1 + 168;
            const char* gp3 = gp2 + 168;
            const unsigned int hA0 = *(const unsigned int*)(gp0);
            const unsigned int hB0 = *(const unsigned int*)(gp0+4);
            const unsigned int hs0 = (tid<16)?hA0:hB0;
            float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
            float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0>>16)));
            const unsigned int hA1 = *(const unsigned int*)(gp1);
            const unsigned int hB1 = *(const unsigned int*)(gp1+4);
            const unsigned int hs1 = (tid<16)?hA1:hB1;
            float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
            float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1>>16)));
            const unsigned int hA2 = *(const unsigned int*)(gp2);
            const unsigned int hB2 = *(const unsigned int*)(gp2+4);
            const unsigned int hs2 = (tid<16)?hA2:hB2;
            float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
            float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2>>16)));
            const unsigned int hA3 = *(const unsigned int*)(gp3);
            const unsigned int hB3 = *(const unsigned int*)(gp3+4);
            const unsigned int hs3 = (tid<16)?hA3:hB3;
            float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
            float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3>>16)));
            const unsigned char* d0 = (const unsigned char*)(gp0+8+boff);
            const unsigned char* d1 = (const unsigned char*)(gp1+8+boff);
            const unsigned char* d2 = (const unsigned char*)(gp2+8+boff);
            const unsigned char* d3 = (const unsigned char*)(gp3+8+boff);
            unsigned char b00=d0[0],b01=d0[1],b02=d0[2],b03=d0[3],b04=d0[4];
            unsigned char b10=d1[0],b11=d1[1],b12=d1[2],b13=d1[3],b14=d1[4];
            unsigned char b20=d2[0],b21=d2[1],b22=d2[2],b23=d2[3],b24=d2[4];
            unsigned char b30=d3[0],b31=d3[1],b32=d3[2],b33=d3[3],b34=d3[4];
            #define DOG5_T(sc,zp,b0,b1,b2,b3,b4, x0,x1,x2,x3,x4,x5,x6,x7, acc) \
                acc += (sc*(float)((b0)&0x1f)+zp)*(x0) + (sc*(float)( (((b0)>>5)&0x07)|(((b1)&0x03)<<3) )+zp)*(x1) + (sc*(float)( ((b1)>>2)&0x1f)+zp)*(x2) + (sc*(float)( (((b1)>>7)&0x01)|(((b2)&0x0f)<<1) )+zp)*(x3) + (sc*(float)( (((b2)>>4)&0x0f)|(((b3)&0x01)<<4) )+zp)*(x4) + (sc*(float)( ((b3)>>1)&0x1f)+zp)*(x5) + (sc*(float)( (((b3)>>6)&0x03)|(((b4)&0x07)<<2) )+zp)*(x6) + (sc*(float)( ((b4)>>3)&0x1f)+zp)*(x7)
            DOG5_T(sc0,zp0,b00,b01,b02,b03,b04, x00,x01,x02,x03,x04,x05,x06,x07, acc[r][0]);
            DOG5_T(sc1,zp1,b10,b11,b12,b13,b14, x10,x11,x12,x13,x14,x15,x16,x17, acc[r][1]);
            DOG5_T(sc2,zp2,b20,b21,b22,b23,b24, x20,x21,x22,x23,x24,x25,x26,x27, acc[r][2]);
            DOG5_T(sc3,zp3,b30,b31,b32,b33,b34, x30,x31,x32,x33,x34,x35,x36,x37, acc[r][3]);
            #undef DOG5_T
        }
    }
    #define TAIL5(sc,zp,b0,b1,b2,b3,b4, x0,x1,x2,x3,x4,x5,x6,x7, acc) \
        acc += (sc*(float)((b0)&0x1f)+zp)*(x0) + (sc*(float)( (((b0)>>5)&0x07)|(((b1)&0x03)<<3) )+zp)*(x1) + (sc*(float)( ((b1)>>2)&0x1f)+zp)*(x2) + (sc*(float)( (((b1)>>7)&0x01)|(((b2)&0x0f)<<1) )+zp)*(x3) + (sc*(float)( (((b2)>>4)&0x0f)|(((b3)&0x01)<<4) )+zp)*(x4) + (sc*(float)( ((b3)>>1)&0x1f)+zp)*(x5) + (sc*(float)( (((b3)>>6)&0x03)|(((b4)&0x07)<<2) )+zp)*(x6) + (sc*(float)( ((b4)>>3)&0x1f)+zp)*(x7)
    if(tail>=1){
        const int g = quads<<2;
        const int base = g*256 + tid*8;
        const float tx0=x[base+0],tx1=x[base+1],tx2=x[base+2],tx3=x[base+3],tx4=x[base+4],tx5=x[base+5],tx6=x[base+6],tx7=x[base+7];
        #pragma unroll
        for(int r=0;r<R;r++){
            const char* gp = row_ptrs[r] + g*168;
            const unsigned int hA=*(const unsigned int*)(gp);
            const unsigned int hB=*(const unsigned int*)(gp+4);
            const unsigned int hs=(tid<16)?hA:hB;
            float sc=__half2float(__ushort_as_half((unsigned short)(hs &0xFFFFu)));
            float zp=__half2float(__ushort_as_half((unsigned short)(hs>>16)));
            const unsigned char* d=(const unsigned char*)(gp+8+boff);
            unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4];
            TAIL5(sc,zp,b0,b1,b2,b3,b4, tx0,tx1,tx2,tx3,tx4,tx5,tx6,tx7, acc[r][0]);
        }
    }
    if(tail>=2){
        const int g = (quads<<2)+1;
        const int base = g*256 + tid*8;
        const float tx0=x[base+0],tx1=x[base+1],tx2=x[base+2],tx3=x[base+3],tx4=x[base+4],tx5=x[base+5],tx6=x[base+6],tx7=x[base+7];
        #pragma unroll
        for(int r=0;r<R;r++){
            const char* gp = row_ptrs[r] + g*168;
            const unsigned int hA=*(const unsigned int*)(gp);
            const unsigned int hB=*(const unsigned int*)(gp+4);
            const unsigned int hs=(tid<16)?hA:hB;
            float sc=__half2float(__ushort_as_half((unsigned short)(hs &0xFFFFu)));
            float zp=__half2float(__ushort_as_half((unsigned short)(hs>>16)));
            const unsigned char* d=(const unsigned char*)(gp+8+boff);
            unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4];
            TAIL5(sc,zp,b0,b1,b2,b3,b4, tx0,tx1,tx2,tx3,tx4,tx5,tx6,tx7, acc[r][1]);
        }
    }
    if(tail>=3){
        const int g = (quads<<2)+2;
        const int base = g*256 + tid*8;
        const float tx0=x[base+0],tx1=x[base+1],tx2=x[base+2],tx3=x[base+3],tx4=x[base+4],tx5=x[base+5],tx6=x[base+6],tx7=x[base+7];
        #pragma unroll
        for(int r=0;r<R;r++){
            const char* gp = row_ptrs[r] + g*168;
            const unsigned int hA=*(const unsigned int*)(gp);
            const unsigned int hB=*(const unsigned int*)(gp+4);
            const unsigned int hs=(tid<16)?hA:hB;
            float sc=__half2float(__ushort_as_half((unsigned short)(hs &0xFFFFu)));
            float zp=__half2float(__ushort_as_half((unsigned short)(hs>>16)));
            const unsigned char* d=(const unsigned char*)(gp+8+boff);
            unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4];
            TAIL5(sc,zp,b0,b1,b2,b3,b4, tx0,tx1,tx2,tx3,tx4,tx5,tx6,tx7, acc[r][2]);
        }
    }
    #undef TAIL5
    #pragma unroll
    for(int r=0;r<R;r++){
        float s=(acc[r][0]+acc[r][1])+(acc[r][2]+acc[r][3]);
        for(int off=16;off>0;off>>=1) s+=__shfl_down(s,off);
        int row=row_base+r;
        if(tid==0 && row<M) y[row]=s;
    }
}
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq5g256v2_multirow_r2(const char*__restrict__ A,const float*__restrict__ x,float*__restrict__ y,int M,int K){ gemv_mq5g256v2_multirow_body<2>(A,x,y,M,K); }
__launch_bounds__(32, 10)
extern "C" __global__ void gemv_mq5g256v2_multirow_r4(const char*__restrict__ A,const float*__restrict__ x,float*__restrict__ y,int M,int K){ gemv_mq5g256v2_multirow_body<4>(A,x,y,M,K); }
__launch_bounds__(32, 6)
extern "C" __global__ void gemv_mq5g256v2_multirow_r8(const char*__restrict__ A,const float*__restrict__ x,float*__restrict__ y,int M,int K){ gemv_mq5g256v2_multirow_body<8>(A,x,y,M,K); }
