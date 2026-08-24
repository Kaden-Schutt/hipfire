// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
#define MQ6_DQ6(i,dw,sh) a[i]=sc*(_Float16)(float)(((dw)>>(sh))&63u)+zp
#define MQ6_DQ6_8(d0,d1) do { MQ6_DQ6(0,(d0),0); MQ6_DQ6(1,(d0),6); MQ6_DQ6(2,(d0),12); MQ6_DQ6(3,(d0),18); MQ6_DQ6(4,(d1),0); MQ6_DQ6(5,(d1),6); MQ6_DQ6(6,(d1),12); MQ6_DQ6(7,(d1),18); } while(0)
#define MQ6_HDR_LOAD(gp) const unsigned int hA=*(const unsigned int*)(gp); const unsigned int hB=*(const unsigned int*)(gp+4); const _Float16 sc0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA&0xFFFFu))); const _Float16 zp0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA>>16))); const _Float16 sc1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB&0xFFFFu))); const _Float16 zp1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB>>16)))
#define MQ6_HDR_SEL _Float16 sc=(kt<8)?sc0:sc1; _Float16 zp=(kt<8)?zp0:zp1;
#define GEN_RES_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_residual_wmma_gfx12_bt##BV(const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M,int K,int N){ \
    const int rs=blockIdx.x*16, bs=blockIdx.y*(16*BV), tid=threadIdx.x; if(rs>=M||bs>=N) return; const int ml=tid&15, kg=tid>>4; const int sr=(rs+ml<M)?(rs+ml):(M-1); const int gpr=K/256; const char* rb=A+(long long)sr*gpr*200; int sb[BV]; for(int b=0;b<BV;b++){int t=bs+b*16+ml; sb[b]=(t<N)?t:0;} float8_t acc[BV]; for(int b=0;b<BV;b++) acc[b]=float8_t{0,0,0,0,0,0,0,0}; for(int g=0;g<gpr;g++){const char* gp=rb+g*200; MQ6_HDR_LOAD(gp); const _Float16* xg[BV]; for(int b=0;b<BV;b++) xg[b]=X+(long long)sb[b]*K+g*256; for(int kt=0;kt<16;kt+=4){ unsigned int d0a=*(const unsigned int*)(gp+8+(kt+0)*12+kg*6), d1a=*(const unsigned int*)(gp+8+(kt+0)*12+kg*6+3); unsigned int d0b=*(const unsigned int*)(gp+8+(kt+1)*12+kg*6), d1b=*(const unsigned int*)(gp+8+(kt+1)*12+kg*6+3); unsigned int d0c=*(const unsigned int*)(gp+8+(kt+2)*12+kg*6), d1c=*(const unsigned int*)(gp+8+(kt+2)*12+kg*6+3); unsigned int d0d=*(const unsigned int*)(gp+8+(kt+3)*12+kg*6), d1d=*(const unsigned int*)(gp+8+(kt+3)*12+kg*6+3); _Pragma("unroll") for(int ki=0;ki<4;ki++){ MQ6_HDR_SEL half8_t a; if(ki==0) MQ6_DQ6_8(d0a,d1a); else if(ki==1) MQ6_DQ6_8(d0b,d1b); else if(ki==2) MQ6_DQ6_8(d0c,d1c); else MQ6_DQ6_8(d0d,d1d); _Pragma("unroll") for(int b=0;b<BV;b++){ half8_t bb=*(const half8_t*)(xg[b]+(kt+ki)*16+kg*8); acc[b]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,bb,acc[b]); } } } } for(int b=0;b<BV;b++){ const int oc=bs+b*16+ml; if(oc<N) for(int j=0;j<8;j++){ int orow=rs+8*kg+j; if(orow<M) Y[(long long)oc*M+orow]+=acc[b][j]; } } \
}
GEN_RES_BT(4)
GEN_RES_BT(8)
GEN_RES_BT(12)
