// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4-G256-GL multirow GEMV — qt=40, global Lloyd codebook.
//
// Mirrors gemv_hfq4g256_multirow.hip's memory-level parallelism lever:
// one warp computes R rows, hoisting x across rows and issuing the R rows'
// weight loads back-to-back so LLVM can clause them.
//
// Format (SoA, 130 B/group = 4.0625 bpw):
//   [0 .. M*gpr*128)          4-bit indices, 128 B/group, row-major (row,group)
//   [M*gpr*128 .. +M*gpr*2)   fp16 per-block scales, 2 B/group, row-major (row,group)
// gpr = K/256, K%256==0. Nibble packing identical to MQ4G256.
// Reconstruction: w = f32(scale) * GL_CB4[code]  (CB in LDS, branchless).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Templated body shared by R=4 and R=8. R=2 has a specialized unrolled
// body below for VGPR/occupancy (mirrors hfq multirow R13c1 structure).
template<int R>
__device__ __forceinline__ void gemv_mq4g256gl_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    float cb0,  float cb1,  float cb2,  float cb3,
    float cb4,  float cb5,  float cb6,  float cb7,
    float cb8,  float cb9,  float cb10, float cb11,
    float cb12, float cb13, float cb14, float cb15,
    int M, int K)
{
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;
    const int gpr = K / 256;
    const long long idx_stride = (long long)gpr * 128;
    const __half* scales_base = (const __half*)(A + (long long)M * gpr * 128);
    const int boff = tid * 4;

    // LDS: [16 f32 codebook][R*gpr f32 scales]
    extern __shared__ float lds[];
    float* cbt = lds;
    float* sc = lds + 16;
    if (tid < 16) {
        const float cbv[16] = {cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7,
                               cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15};
        cbt[tid] = cbv[tid];
    }
    // Hoist R*gpr scales — R rows are consecutive so this is one contiguous
    // run of halfs. Clamp OOB rows to row_base for last partial wave.
    const int total_sc = R * gpr;
    for (int i = tid; i < total_sc; i += 32) {
        int r = i / gpr;
        int g = i % gpr;
        int src_row = row_base + r;
        if (src_row >= M) src_row = row_base;
        sc[i] = __half2float(scales_base[(long long)src_row * gpr + g]);
    }
    __syncthreads();

    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        int row = row_base + r;
        int safe = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe * idx_stride;
    }

    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    // Macros for x hoist via float4 (2 VMEM per 8 floats vs 8 scalar)
    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);

        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp0 = row_ptrs[r] + g * 128;
            const char* gp1 = gp0 + 128;
            const char* gp2 = gp1 + 128;
            const char* gp3 = gp2 + 128;

            float sc0 = sc[r * gpr + g + 0];
            float sc1 = sc[r * gpr + g + 1];
            float sc2 = sc[r * gpr + g + 2];
            float sc3 = sc[r * gpr + g + 3];

            unsigned int pk0 = *(const unsigned int*)(gp0 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + boff);

            unsigned int q00 =  pk0        & 0xFu;
            unsigned int q01 = (pk0 >> 4)  & 0xFu;
            unsigned int q02 = (pk0 >> 8)  & 0xFu;
            unsigned int q03 = (pk0 >> 12) & 0xFu;
            unsigned int q04 = (pk0 >> 16) & 0xFu;
            unsigned int q05 = (pk0 >> 20) & 0xFu;
            unsigned int q06 = (pk0 >> 24) & 0xFu;
            unsigned int q07 = (pk0 >> 28) & 0xFu;

            unsigned int q10 =  pk1        & 0xFu;
            unsigned int q11 = (pk1 >> 4)  & 0xFu;
            unsigned int q12 = (pk1 >> 8)  & 0xFu;
            unsigned int q13 = (pk1 >> 12) & 0xFu;
            unsigned int q14 = (pk1 >> 16) & 0xFu;
            unsigned int q15 = (pk1 >> 20) & 0xFu;
            unsigned int q16 = (pk1 >> 24) & 0xFu;
            unsigned int q17 = (pk1 >> 28) & 0xFu;

            unsigned int q20 =  pk2        & 0xFu;
            unsigned int q21 = (pk2 >> 4)  & 0xFu;
            unsigned int q22 = (pk2 >> 8)  & 0xFu;
            unsigned int q23 = (pk2 >> 12) & 0xFu;
            unsigned int q24 = (pk2 >> 16) & 0xFu;
            unsigned int q25 = (pk2 >> 20) & 0xFu;
            unsigned int q26 = (pk2 >> 24) & 0xFu;
            unsigned int q27 = (pk2 >> 28) & 0xFu;

            unsigned int q30 =  pk3        & 0xFu;
            unsigned int q31 = (pk3 >> 4)  & 0xFu;
            unsigned int q32 = (pk3 >> 8)  & 0xFu;
            unsigned int q33 = (pk3 >> 12) & 0xFu;
            unsigned int q34 = (pk3 >> 16) & 0xFu;
            unsigned int q35 = (pk3 >> 20) & 0xFu;
            unsigned int q36 = (pk3 >> 24) & 0xFu;
            unsigned int q37 = (pk3 >> 28) & 0xFu;

            // Branchless LDS lookup — identical FMA order to single-row kernel:
            // scale * sum(cbt[q]*x)
            acc[r][0] += sc0 * (cbt[q00]*xa0 + cbt[q01]*xa1 + cbt[q02]*xa2 + cbt[q03]*xa3
                              + cbt[q04]*xa4 + cbt[q05]*xa5 + cbt[q06]*xa6 + cbt[q07]*xa7);
            acc[r][1] += sc1 * (cbt[q10]*xb0 + cbt[q11]*xb1 + cbt[q12]*xb2 + cbt[q13]*xb3
                              + cbt[q14]*xb4 + cbt[q15]*xb5 + cbt[q16]*xb6 + cbt[q17]*xb7);
            acc[r][2] += sc2 * (cbt[q20]*xc0 + cbt[q21]*xc1 + cbt[q22]*xc2 + cbt[q23]*xc3
                              + cbt[q24]*xc4 + cbt[q25]*xc5 + cbt[q26]*xc6 + cbt[q27]*xc7);
            acc[r][3] += sc3 * (cbt[q30]*xd0 + cbt[q31]*xd1 + cbt[q32]*xd2 + cbt[q33]*xd3
                              + cbt[q34]*xd4 + cbt[q35]*xd5 + cbt[q36]*xd6 + cbt[q37]*xd7);
        }
    }

    #undef LOAD_X8_INTO

    // Tail — preserve 4-way slot mapping acc[r][g%4]
    #define TAIL_LOAD_X(b) \
        const float4 _tv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 _tv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float _x0 = _tv0.x, _x1 = _tv0.y, _x2 = _tv0.z, _x3 = _tv0.w; \
        const float _x4 = _tv1.x, _x5 = _tv1.y, _x6 = _tv1.z, _x7 = _tv1.w

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp = row_ptrs[r] + g * 128;
            float scc = sc[r * gpr + g];
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu;
            unsigned int qq1 = (pk >> 4)  & 0xFu;
            unsigned int qq2 = (pk >> 8)  & 0xFu;
            unsigned int qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu;
            unsigned int qq5 = (pk >> 20) & 0xFu;
            unsigned int qq6 = (pk >> 24) & 0xFu;
            unsigned int qq7 = (pk >> 28) & 0xFu;
            acc[r][0] += scc * (cbt[qq0]*_x0 + cbt[qq1]*_x1 + cbt[qq2]*_x2 + cbt[qq3]*_x3
                              + cbt[qq4]*_x4 + cbt[qq5]*_x5 + cbt[qq6]*_x6 + cbt[qq7]*_x7);
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp = row_ptrs[r] + g * 128;
            float scc = sc[r * gpr + g];
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu;
            unsigned int qq1 = (pk >> 4)  & 0xFu;
            unsigned int qq2 = (pk >> 8)  & 0xFu;
            unsigned int qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu;
            unsigned int qq5 = (pk >> 20) & 0xFu;
            unsigned int qq6 = (pk >> 24) & 0xFu;
            unsigned int qq7 = (pk >> 28) & 0xFu;
            acc[r][1] += scc * (cbt[qq0]*_x0 + cbt[qq1]*_x1 + cbt[qq2]*_x2 + cbt[qq3]*_x3
                              + cbt[qq4]*_x4 + cbt[qq5]*_x5 + cbt[qq6]*_x6 + cbt[qq7]*_x7);
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp = row_ptrs[r] + g * 128;
            float scc = sc[r * gpr + g];
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu;
            unsigned int qq1 = (pk >> 4)  & 0xFu;
            unsigned int qq2 = (pk >> 8)  & 0xFu;
            unsigned int qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu;
            unsigned int qq5 = (pk >> 20) & 0xFu;
            unsigned int qq6 = (pk >> 24) & 0xFu;
            unsigned int qq7 = (pk >> 28) & 0xFu;
            acc[r][2] += scc * (cbt[qq0]*_x0 + cbt[qq1]*_x1 + cbt[qq2]*_x2 + cbt[qq3]*_x3
                              + cbt[qq4]*_x4 + cbt[qq5]*_x5 + cbt[qq6]*_x6 + cbt[qq7]*_x7);
        }
    }
    #undef TAIL_LOAD_X

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] = s;
    }
}

// Specialized R=2 body — same math but explicit a0..b3 accumulators to
// reduce VGPR pressure vs the generic R=2 instantiation (mirrors HFQ R2).
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4g256gl_multirow_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    float cb0,  float cb1,  float cb2,  float cb3,
    float cb4,  float cb5,  float cb6,  float cb7,
    float cb8,  float cb9,  float cb10, float cb11,
    float cb12, float cb13, float cb14, float cb15,
    int M, int K)
{
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;
    const int gpr = K / 256;
    const long long idx_stride = (long long)gpr * 128;
    const __half* scales_base = (const __half*)(A + (long long)M * gpr * 128);
    const int boff = tid * 4;

    extern __shared__ float lds[];
    float* cbt = lds;
    float* sc = lds + 16;
    if (tid < 16) {
        const float cbv[16] = {cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7,
                               cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15};
        cbt[tid] = cbv[tid];
    }
    // R=2 contiguous scales: 2*gpr halfs
    for (int i = tid; i < 2 * gpr; i += 32) {
        int r = i / gpr;
        int g = i % gpr;
        int src_row = (r == 0) ? row0 : (have_row1 ? row1 : row0);
        sc[i] = __half2float(scales_base[(long long)src_row * gpr + g]);
    }
    __syncthreads();

    const char* row_ptr0 = A + (long long)row0 * idx_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * idx_stride;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;
        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);

        // Group g
        {
            unsigned int pk0 = *(const unsigned int*)(row_ptr0 + g*128 + boff);
            unsigned int pk1 = *(const unsigned int*)(row_ptr1 + g*128 + boff);
            float sc0 = sc[0 * gpr + g];
            float sc1 = sc[1 * gpr + g];
            unsigned int qa0 =  pk0        & 0xFu, qa1 = (pk0 >> 4) & 0xFu, qa2 = (pk0 >> 8) & 0xFu, qa3 = (pk0 >>12) & 0xFu;
            unsigned int qa4 = (pk0 >>16) & 0xFu, qa5 = (pk0 >>20) & 0xFu, qa6 = (pk0 >>24) & 0xFu, qa7 = (pk0 >>28) & 0xFu;
            unsigned int qb0 =  pk1        & 0xFu, qb1 = (pk1 >> 4) & 0xFu, qb2 = (pk1 >> 8) & 0xFu, qb3 = (pk1 >>12) & 0xFu;
            unsigned int qb4 = (pk1 >>16) & 0xFu, qb5 = (pk1 >>20) & 0xFu, qb6 = (pk1 >>24) & 0xFu, qb7 = (pk1 >>28) & 0xFu;
            a0 += sc0 * (cbt[qa0]*xa0 + cbt[qa1]*xa1 + cbt[qa2]*xa2 + cbt[qa3]*xa3 + cbt[qa4]*xa4 + cbt[qa5]*xa5 + cbt[qa6]*xa6 + cbt[qa7]*xa7);
            b0 += sc1 * (cbt[qb0]*xa0 + cbt[qb1]*xa1 + cbt[qb2]*xa2 + cbt[qb3]*xa3 + cbt[qb4]*xa4 + cbt[qb5]*xa5 + cbt[qb6]*xa6 + cbt[qb7]*xa7);
        }
        // Group g+1
        {
            unsigned int pk0 = *(const unsigned int*)(row_ptr0 + (g+1)*128 + boff);
            unsigned int pk1 = *(const unsigned int*)(row_ptr1 + (g+1)*128 + boff);
            float sc0 = sc[0 * gpr + g+1];
            float sc1 = sc[1 * gpr + g+1];
            unsigned int qa0 =  pk0        & 0xFu, qa1 = (pk0 >> 4) & 0xFu, qa2 = (pk0 >> 8) & 0xFu, qa3 = (pk0 >>12) & 0xFu;
            unsigned int qa4 = (pk0 >>16) & 0xFu, qa5 = (pk0 >>20) & 0xFu, qa6 = (pk0 >>24) & 0xFu, qa7 = (pk0 >>28) & 0xFu;
            unsigned int qb0 =  pk1        & 0xFu, qb1 = (pk1 >> 4) & 0xFu, qb2 = (pk1 >> 8) & 0xFu, qb3 = (pk1 >>12) & 0xFu;
            unsigned int qb4 = (pk1 >>16) & 0xFu, qb5 = (pk1 >>20) & 0xFu, qb6 = (pk1 >>24) & 0xFu, qb7 = (pk1 >>28) & 0xFu;
            a1 += sc0 * (cbt[qa0]*xb0 + cbt[qa1]*xb1 + cbt[qa2]*xb2 + cbt[qa3]*xb3 + cbt[qa4]*xb4 + cbt[qa5]*xb5 + cbt[qa6]*xb6 + cbt[qa7]*xb7);
            b1 += sc1 * (cbt[qb0]*xb0 + cbt[qb1]*xb1 + cbt[qb2]*xb2 + cbt[qb3]*xb3 + cbt[qb4]*xb4 + cbt[qb5]*xb5 + cbt[qb6]*xb6 + cbt[qb7]*xb7);
        }
        // Group g+2
        {
            unsigned int pk0 = *(const unsigned int*)(row_ptr0 + (g+2)*128 + boff);
            unsigned int pk1 = *(const unsigned int*)(row_ptr1 + (g+2)*128 + boff);
            float sc0 = sc[0 * gpr + g+2];
            float sc1 = sc[1 * gpr + g+2];
            unsigned int qa0 =  pk0        & 0xFu, qa1 = (pk0 >> 4) & 0xFu, qa2 = (pk0 >> 8) & 0xFu, qa3 = (pk0 >>12) & 0xFu;
            unsigned int qa4 = (pk0 >>16) & 0xFu, qa5 = (pk0 >>20) & 0xFu, qa6 = (pk0 >>24) & 0xFu, qa7 = (pk0 >>28) & 0xFu;
            unsigned int qb0 =  pk1        & 0xFu, qb1 = (pk1 >> 4) & 0xFu, qb2 = (pk1 >> 8) & 0xFu, qb3 = (pk1 >>12) & 0xFu;
            unsigned int qb4 = (pk1 >>16) & 0xFu, qb5 = (pk1 >>20) & 0xFu, qb6 = (pk1 >>24) & 0xFu, qb7 = (pk1 >>28) & 0xFu;
            a2 += sc0 * (cbt[qa0]*xc0 + cbt[qa1]*xc1 + cbt[qa2]*xc2 + cbt[qa3]*xc3 + cbt[qa4]*xc4 + cbt[qa5]*xc5 + cbt[qa6]*xc6 + cbt[qa7]*xc7);
            b2 += sc1 * (cbt[qb0]*xc0 + cbt[qb1]*xc1 + cbt[qb2]*xc2 + cbt[qb3]*xc3 + cbt[qb4]*xc4 + cbt[qb5]*xc5 + cbt[qb6]*xc6 + cbt[qb7]*xc7);
        }
        // Group g+3
        {
            unsigned int pk0 = *(const unsigned int*)(row_ptr0 + (g+3)*128 + boff);
            unsigned int pk1 = *(const unsigned int*)(row_ptr1 + (g+3)*128 + boff);
            float sc0 = sc[0 * gpr + g+3];
            float sc1 = sc[1 * gpr + g+3];
            unsigned int qa0 =  pk0        & 0xFu, qa1 = (pk0 >> 4) & 0xFu, qa2 = (pk0 >> 8) & 0xFu, qa3 = (pk0 >>12) & 0xFu;
            unsigned int qa4 = (pk0 >>16) & 0xFu, qa5 = (pk0 >>20) & 0xFu, qa6 = (pk0 >>24) & 0xFu, qa7 = (pk0 >>28) & 0xFu;
            unsigned int qb0 =  pk1        & 0xFu, qb1 = (pk1 >> 4) & 0xFu, qb2 = (pk1 >> 8) & 0xFu, qb3 = (pk1 >>12) & 0xFu;
            unsigned int qb4 = (pk1 >>16) & 0xFu, qb5 = (pk1 >>20) & 0xFu, qb6 = (pk1 >>24) & 0xFu, qb7 = (pk1 >>28) & 0xFu;
            a3 += sc0 * (cbt[qa0]*xd0 + cbt[qa1]*xd1 + cbt[qa2]*xd2 + cbt[qa3]*xd3 + cbt[qa4]*xd4 + cbt[qa5]*xd5 + cbt[qa6]*xd6 + cbt[qa7]*xd7);
            b3 += sc1 * (cbt[qb0]*xd0 + cbt[qb1]*xd1 + cbt[qb2]*xd2 + cbt[qb3]*xd3 + cbt[qb4]*xd4 + cbt[qb5]*xd5 + cbt[qb6]*xd6 + cbt[qb7]*xd7);
        }
    }
    #undef LOAD_X8_INTO

    #define TAIL_X(b) \
        const float4 _tv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 _tv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float _x0=_tv0.x,_x1=_tv0.y,_x2=_tv0.z,_x3=_tv0.w; const float _x4=_tv1.x,_x5=_tv1.y,_x6=_tv1.z,_x7=_tv1.w

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g*256 + tid*8;
        TAIL_X(base);
        unsigned int pk0 = *(const unsigned int*)(row_ptr0 + g*128 + boff);
        unsigned int pk1 = *(const unsigned int*)(row_ptr1 + g*128 + boff);
        float sc0 = sc[0*gpr+g]; float sc1 = sc[1*gpr+g];
        unsigned int qa0=pk0&0xFu,qa1=(pk0>>4)&0xFu,qa2=(pk0>>8)&0xFu,qa3=(pk0>>12)&0xFu,qa4=(pk0>>16)&0xFu,qa5=(pk0>>20)&0xFu,qa6=(pk0>>24)&0xFu,qa7=(pk0>>28)&0xFu;
        unsigned int qb0=pk1&0xFu,qb1=(pk1>>4)&0xFu,qb2=(pk1>>8)&0xFu,qb3=(pk1>>12)&0xFu,qb4=(pk1>>16)&0xFu,qb5=(pk1>>20)&0xFu,qb6=(pk1>>24)&0xFu,qb7=(pk1>>28)&0xFu;
        a0 += sc0 * (cbt[qa0]*_x0 + cbt[qa1]*_x1 + cbt[qa2]*_x2 + cbt[qa3]*_x3 + cbt[qa4]*_x4 + cbt[qa5]*_x5 + cbt[qa6]*_x6 + cbt[qa7]*_x7);
        b0 += sc1 * (cbt[qb0]*_x0 + cbt[qb1]*_x1 + cbt[qb2]*_x2 + cbt[qb3]*_x3 + cbt[qb4]*_x4 + cbt[qb5]*_x5 + cbt[qb6]*_x6 + cbt[qb7]*_x7);
    }
    if (tail >= 2) {
        const int g = (quads<<2)+1;
        const int base = g*256 + tid*8;
        TAIL_X(base);
        unsigned int pk0 = *(const unsigned int*)(row_ptr0 + g*128 + boff);
        unsigned int pk1 = *(const unsigned int*)(row_ptr1 + g*128 + boff);
        float sc0 = sc[0*gpr+g]; float sc1 = sc[1*gpr+g];
        unsigned int qa0=pk0&0xFu,qa1=(pk0>>4)&0xFu,qa2=(pk0>>8)&0xFu,qa3=(pk0>>12)&0xFu,qa4=(pk0>>16)&0xFu,qa5=(pk0>>20)&0xFu,qa6=(pk0>>24)&0xFu,qa7=(pk0>>28)&0xFu;
        unsigned int qb0=pk1&0xFu,qb1=(pk1>>4)&0xFu,qb2=(pk1>>8)&0xFu,qb3=(pk1>>12)&0xFu,qb4=(pk1>>16)&0xFu,qb5=(pk1>>20)&0xFu,qb6=(pk1>>24)&0xFu,qb7=(pk1>>28)&0xFu;
        a1 += sc0 * (cbt[qa0]*_x0 + cbt[qa1]*_x1 + cbt[qa2]*_x2 + cbt[qa3]*_x3 + cbt[qa4]*_x4 + cbt[qa5]*_x5 + cbt[qa6]*_x6 + cbt[qa7]*_x7);
        b1 += sc1 * (cbt[qb0]*_x0 + cbt[qb1]*_x1 + cbt[qb2]*_x2 + cbt[qb3]*_x3 + cbt[qb4]*_x4 + cbt[qb5]*_x5 + cbt[qb6]*_x6 + cbt[qb7]*_x7);
    }
    if (tail >= 3) {
        const int g = (quads<<2)+2;
        const int base = g*256 + tid*8;
        TAIL_X(base);
        unsigned int pk0 = *(const unsigned int*)(row_ptr0 + g*128 + boff);
        unsigned int pk1 = *(const unsigned int*)(row_ptr1 + g*128 + boff);
        float sc0 = sc[0*gpr+g]; float sc1 = sc[1*gpr+g];
        unsigned int qa0=pk0&0xFu,qa1=(pk0>>4)&0xFu,qa2=(pk0>>8)&0xFu,qa3=(pk0>>12)&0xFu,qa4=(pk0>>16)&0xFu,qa5=(pk0>>20)&0xFu,qa6=(pk0>>24)&0xFu,qa7=(pk0>>28)&0xFu;
        unsigned int qb0=pk1&0xFu,qb1=(pk1>>4)&0xFu,qb2=(pk1>>8)&0xFu,qb3=(pk1>>12)&0xFu,qb4=(pk1>>16)&0xFu,qb5=(pk1>>20)&0xFu,qb6=(pk1>>24)&0xFu,qb7=(pk1>>28)&0xFu;
        a2 += sc0 * (cbt[qa0]*_x0 + cbt[qa1]*_x1 + cbt[qa2]*_x2 + cbt[qa3]*_x3 + cbt[qa4]*_x4 + cbt[qa5]*_x5 + cbt[qa6]*_x6 + cbt[qa7]*_x7);
        b2 += sc1 * (cbt[qb0]*_x0 + cbt[qb1]*_x1 + cbt[qb2]*_x2 + cbt[qb3]*_x3 + cbt[qb4]*_x4 + cbt[qb5]*_x5 + cbt[qb6]*_x6 + cbt[qb7]*_x7);
    }
    #undef TAIL_X

    float s0 = (a0 + a1) + (a2 + a3);
    float s1 = (b0 + b1) + (b2 + b3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down(s0, offset);
        s1 += __shfl_down(s1, offset);
    }
    if (tid == 0) {
        y[row0] = s0;
        if (have_row1) y[row1] = s1;
    }
}

__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4g256gl_multirow_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    float cb0,  float cb1,  float cb2,  float cb3,
    float cb4,  float cb5,  float cb6,  float cb7,
    float cb8,  float cb9,  float cb10, float cb11,
    float cb12, float cb13, float cb14, float cb15,
    int M, int K)
{
    gemv_mq4g256gl_multirow_body<4>(A, x, y, cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15, M, K);
}

__launch_bounds__(32, 6)
extern "C" __global__ void gemv_mq4g256gl_multirow_r8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    float cb0,  float cb1,  float cb2,  float cb3,
    float cb4,  float cb5,  float cb6,  float cb7,
    float cb8,  float cb9,  float cb10, float cb11,
    float cb12, float cb13, float cb14, float cb15,
    int M, int K)
{
    gemv_mq4g256gl_multirow_body<8>(A, x, y, cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15, M, K);
}
