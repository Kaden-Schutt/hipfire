// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq6g256v2_residual_wmma_gfx12(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,
    int M, int K, int N
){
    const int row_start=blockIdx.x*16, batch_start=blockIdx.y*16;
    const int tid=threadIdx.x;
    if(row_start>=M||batch_start>=N) return;
    const int m_lane=tid &15, k_grp=tid>>4;
    const int my_row=row_start+m_lane;
    const int safe_row=(my_row<M)?my_row:(M-1);
    const int groups_per_row=K/256;
    const long long row_stride=(long long)groups_per_row*200;
    const char* row_base=A+safe_row*row_stride;
    const int safe_batch=(batch_start+m_lane<N)?(batch_start+m_lane):0;
    float8_t acc={0,0,0,0,0,0,0,0};
    for(int g=0;g<groups_per_row;g++){
        const char* gp=row_base+g*200;
        const unsigned int hA=*(const unsigned int*)(gp);
        const unsigned int hB=*(const unsigned int*)(gp+4);
        _Float16 sc0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA&0xFFFFu)));
        _Float16 zp0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA>>16)));
        _Float16 sc1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB&0xFFFFu)));
        _Float16 zp1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB>>16)));
        const _Float16* x_base=X+(long long)safe_batch*K+g*256;
        #define DQ6(i,dw,sh) a_reg[i]=sc_h*(_Float16)(float)(((dw)>>(sh))&63u)+zp_h
        #define DQ6_8(d0,d1) DQ6(0,(d0),0); DQ6(1,(d0),6); DQ6(2,(d0),12); DQ6(3,(d0),18); DQ6(4,(d1),0); DQ6(5,(d1),6); DQ6(6,(d1),12); DQ6(7,(d1),18)
        for(int kt=0;kt<16;kt+=4){
            _Float16 sc_h=(kt<8)?sc0:sc1; _Float16 zp_h=(kt<8)?zp0:zp1;
            const int bb_a=(kt+0)*12 + k_grp*6, bb_b=(kt+1)*12 + k_grp*6, bb_c=(kt+2)*12 + k_grp*6, bb_d=(kt+3)*12 + k_grp*6;
            const unsigned char* dp_a=(const unsigned char*)(gp+8+bb_a); const unsigned char* dp_b=(const unsigned char*)(gp+8+bb_b); const unsigned char* dp_c=(const unsigned char*)(gp+8+bb_c); const unsigned char* dp_d=(const unsigned char*)(gp+8+bb_d);
            unsigned int d0a=*(const unsigned int*)dp_a, d1a=*(const unsigned int*)(dp_a+3); unsigned int d0b=*(const unsigned int*)dp_b, d1b=*(const unsigned int*)(dp_b+3); unsigned int d0c=*(const unsigned int*)dp_c, d1c=*(const unsigned int*)(dp_c+3); unsigned int d0d=*(const unsigned int*)dp_d, d1d=*(const unsigned int*)(dp_d+3);
            half8_t b_a=*(const half8_t*)(x_base+(kt+0)*16+k_grp*8); half8_t b_b=*(const half8_t*)(x_base+(kt+1)*16+k_grp*8); half8_t b_c=*(const half8_t*)(x_base+(kt+2)*16+k_grp*8); half8_t b_d=*(const half8_t*)(x_base+(kt+3)*16+k_grp*8);
            half8_t a_reg; DQ6_8(d0a,d1a); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_a,acc); DQ6_8(d0b,d1b); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_b,acc); DQ6_8(d0c,d1c); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_c,acc); DQ6_8(d0d,d1d); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_d,acc);
        }
        #undef DQ6_8
        #undef DQ6
    }
    const int out_col=batch_start+m_lane;
    if(out_col<N){
        #pragma unroll
        for(int j=0;j<8;j++){
            int out_row=row_start+8*k_grp+j;
            if(out_row<M) Y[(long long)out_col*M+out_row]+=acc[j];
        }
    }
}
