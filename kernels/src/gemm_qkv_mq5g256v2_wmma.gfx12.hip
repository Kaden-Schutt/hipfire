// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_mq5g256v2_wmma_gfx12(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m, int k_m, int v_m, int K, int N
){
    const int total_m = q_m+k_m+v_m;
    const int row_start=blockIdx.x*16;
    const int batch_start=blockIdx.y*16;
    const int tid=threadIdx.x;
    if(row_start>=total_m) return;
    if(batch_start>=N) return;
    const int m_lane=tid&15;
    const int k_grp=tid>>4;
    const int my_row=row_start+m_lane;
    const int safe_row=(my_row<total_m)?my_row:(total_m-1);
    const char* A; int local_row;
    if(safe_row<q_m){ A=A_q; local_row=safe_row; }
    else if(safe_row<q_m+k_m){ A=A_k; local_row=safe_row-q_m; }
    else { A=A_v; local_row=safe_row-q_m-k_m; }
    const int groups_per_row=K/256;
    const long long row_stride=(long long)groups_per_row*168;
    const char* row_base=A+local_row*row_stride;
    const int safe_batch=(batch_start+m_lane<N)?(batch_start+m_lane):0;
    float8_t acc={0};
    for(int g=0; g<groups_per_row; g++){
        const char* gp=row_base+g*168;
        const unsigned int hA=*(const unsigned int*)(gp);
        const unsigned int hB=*(const unsigned int*)(gp+4);
        const _Float16 sc0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA&0xFFFFu)));
        const _Float16 zp0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA>>16)));
        const _Float16 sc1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB&0xFFFFu)));
        const _Float16 zp1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB>>16)));
        const _Float16* x_base=X+(long long)safe_batch*K+g*256;
        #define DQ5(dp, sc_h, zp_h, a_reg) { uint32_t _d0=*(const uint32_t*)(dp); uint8_t _d4=*((const unsigned char*)(dp)+4); uint64_t _bits=((uint64_t)_d0)|((uint64_t)_d4<<32); a_reg[0]=sc_h*(_Float16)(float)((_bits>>0)&31u)+zp_h; a_reg[1]=sc_h*(_Float16)(float)((_bits>>5)&31u)+zp_h; a_reg[2]=sc_h*(_Float16)(float)((_bits>>10)&31u)+zp_h; a_reg[3]=sc_h*(_Float16)(float)((_bits>>15)&31u)+zp_h; a_reg[4]=sc_h*(_Float16)(float)((_bits>>20)&31u)+zp_h; a_reg[5]=sc_h*(_Float16)(float)((_bits>>25)&31u)+zp_h; a_reg[6]=sc_h*(_Float16)(float)((_bits>>30)&31u)+zp_h; a_reg[7]=sc_h*(_Float16)(float)((_bits>>35)&31u)+zp_h; }
        for(int kt=0; kt<16; kt+=4){
            const _Float16 sc_h0=(kt<8)?sc0:sc1; const _Float16 zp_h0=(kt<8)?zp0:zp1;
            const _Float16 sc_h1=(kt+1<8)?sc0:sc1; const _Float16 zp_h1=(kt+1<8)?zp0:zp1;
            const _Float16 sc_h2=(kt+2<8)?sc0:sc1; const _Float16 zp_h2=(kt+2<8)?zp0:zp1;
            const _Float16 sc_h3=(kt+3<8)?sc0:sc1; const _Float16 zp_h3=(kt+3<8)?zp0:zp1;
            int off_a=(kt+0)*10+k_grp*5, off_b=(kt+1)*10+k_grp*5, off_c=(kt+2)*10+k_grp*5, off_d=(kt+3)*10+k_grp*5;
            const unsigned char* dp_a=(const unsigned char*)(gp+8+off_a);
            const unsigned char* dp_b=(const unsigned char*)(gp+8+off_b);
            const unsigned char* dp_c=(const unsigned char*)(gp+8+off_c);
            const unsigned char* dp_d=(const unsigned char*)(gp+8+off_d);
            half8_t b_a=*(const half8_t*)(x_base+(kt+0)*16+k_grp*8);
            half8_t b_b=*(const half8_t*)(x_base+(kt+1)*16+k_grp*8);
            half8_t b_c=*(const half8_t*)(x_base+(kt+2)*16+k_grp*8);
            half8_t b_d=*(const half8_t*)(x_base+(kt+3)*16+k_grp*8);
            half8_t a_reg;
            DQ5(dp_a,sc_h0,zp_h0,a_reg); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_a,acc);
            DQ5(dp_b,sc_h1,zp_h1,a_reg); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_b,acc);
            DQ5(dp_c,sc_h2,zp_h2,a_reg); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_c,acc);
            DQ5(dp_d,sc_h3,zp_h3,a_reg); acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg,b_d,acc);
        }
        #undef DQ5
    }
    const int out_col=batch_start+m_lane;
    if(out_col<N){
        #pragma unroll
        for(int j=0;j<8;j++){
            int out_row=row_start+8*k_grp+j;
            if(out_row<total_m){
                float* Y; int proj_row, out_stride;
                if(out_row<q_m){ Y=Y_q; proj_row=out_row; out_stride=q_m; }
                else if(out_row<q_m+k_m){ Y=Y_k; proj_row=out_row-q_m; out_stride=k_m; }
                else { Y=Y_v; proj_row=out_row-q_m-k_m; out_stride=v_m; }
                Y[(long long)out_col*out_stride+proj_row]=acc[j];
            }
        }
    }
}
