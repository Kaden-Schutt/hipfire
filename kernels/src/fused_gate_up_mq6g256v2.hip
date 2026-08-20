// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define MQ6_GATE(sc,zp,gp_,base_,acc_) do { const unsigned char* d=(const unsigned char*)((gp_)+8+boff); unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4],b5=d[5]; int q0=b0&63,q1=(b0>>6)|((b1&0xF)<<2),q2=(b1>>4)|((b2&3)<<4),q3=b2>>2,q4=b3&63,q5=(b3>>6)|((b4&0xF)<<2),q6=(b4>>4)|((b5&3)<<4),q7=b5>>2; (acc_)+=(sc*(float)q0+zp)*x[(base_)]+(sc*(float)q1+zp)*x[(base_)+1]+(sc*(float)q2+zp)*x[(base_)+2]+(sc*(float)q3+zp)*x[(base_)+3]+(sc*(float)q4+zp)*x[(base_)+4]+(sc*(float)q5+zp)*x[(base_)+5]+(sc*(float)q6+zp)*x[(base_)+6]+(sc*(float)q7+zp)*x[(base_)+7]; } while(0)
#define MQ6_GTAIL(sc,zp,gp_,base_,acc_) do { const unsigned char* d=(const unsigned char*)((gp_)+8+boff); unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4],b5=d[5]; int q0=b0&63,q1=(b0>>6)|((b1&0xF)<<2),q2=(b1>>4)|((b2&3)<<4),q3=b2>>2,q4=b3&63,q5=(b3>>6)|((b4&0xF)<<2),q6=(b4>>4)|((b5&3)<<4),q7=b5>>2; (acc_)+=(sc*(float)q0+zp)*x[(base_)]+(sc*(float)q1+zp)*x[(base_)+1]+(sc*(float)q2+zp)*x[(base_)+2]+(sc*(float)q3+zp)*x[(base_)+3]+(sc*(float)q4+zp)*x[(base_)+4]+(sc*(float)q5+zp)*x[(base_)+5]+(sc*(float)q6+zp)*x[(base_)+6]+(sc*(float)q7+zp)*x[(base_)+7]; } while(0)
__launch_bounds__(32, 16)
extern "C" __global__ void fused_gate_up_mq6g256v2(const char* __restrict__ A_gate, const char* __restrict__ A_up, const float* __restrict__ x, float* __restrict__ y_gate, float* __restrict__ y_up, int gate_m, int up_m, int K){
    const int gid=blockIdx.x; const int tid=threadIdx.x;
    const char* A; float* y; int local_row; if(gid<gate_m){A=A_gate; y=y_gate; local_row=gid;} else {A=A_up; y=y_up; local_row=gid-gate_m;}
    const int gpr=K/256; const char* row_ptr=A+(long long)local_row*gpr*200; const int boff=tid*6;
    float acc0=0,acc1=0,acc2=0,acc3=0; const int quads=gpr>>2, tail=gpr&3;
    for(int q=0;q<quads;q++){ int g=q<<2; const char* gp0=row_ptr+g*200; const char* gp1=gp0+200; const char* gp2=gp1+200; const char* gp3=gp2+200;
        const unsigned int hA0=*(const unsigned int*)gp0, hB0=*(const unsigned int*)(gp0+4); float sc0=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA0:hB0)&0xFFFFu))); float zp0=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA0:hB0)>>16)));
        const unsigned int hA1=*(const unsigned int*)gp1, hB1=*(const unsigned int*)(gp1+4); float sc1=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA1:hB1)&0xFFFFu))); float zp1=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA1:hB1)>>16)));
        const unsigned int hA2=*(const unsigned int*)gp2, hB2=*(const unsigned int*)(gp2+4); float sc2=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA2:hB2)&0xFFFFu))); float zp2=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA2:hB2)>>16)));
        const unsigned int hA3=*(const unsigned int*)gp3, hB3=*(const unsigned int*)(gp3+4); float sc3=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA3:hB3)&0xFFFFu))); float zp3=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA3:hB3)>>16)));
        int base=g*256+tid*8;
        MQ6_GATE(sc0,zp0,gp0,base,acc0);
        MQ6_GATE(sc1,zp1,gp1,base+256,acc1);
        MQ6_GATE(sc2,zp2,gp2,base+512,acc2);
        MQ6_GATE(sc3,zp3,gp3,base+768,acc3);
    }
    if(tail>=1){int g=quads<<2; const char* gp=row_ptr+g*200; const unsigned int hA=*(const unsigned int*)gp, hB=*(const unsigned int*)(gp+4); float sc=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA:hB)&0xFFFFu))); float zp=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA:hB)>>16))); int base=g*256+tid*8; MQ6_GTAIL(sc,zp,gp,base,acc0);}
    if(tail>=2){int g=(quads<<2)+1; const char* gp=row_ptr+g*200; const unsigned int hA=*(const unsigned int*)gp, hB=*(const unsigned int*)(gp+4); float sc=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA:hB)&0xFFFFu))); float zp=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA:hB)>>16))); int base=g*256+tid*8; MQ6_GTAIL(sc,zp,gp,base,acc1);}
    if(tail>=3){int g=(quads<<2)+2; const char* gp=row_ptr+g*200; const unsigned int hA=*(const unsigned int*)gp, hB=*(const unsigned int*)(gp+4); float sc=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA:hB)&0xFFFFu))); float zp=__half2float(__ushort_as_half((unsigned short)(((tid<16)?hA:hB)>>16))); int base=g*256+tid*8; MQ6_GTAIL(sc,zp,gp,base,acc2);}
    float acc=(acc0+acc1)+(acc2+acc3); for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o); if(tid==0) y[local_row]=acc;
}
#undef MQ6_GATE
#undef MQ6_GTAIL
