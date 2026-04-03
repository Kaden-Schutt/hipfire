#include <hip/hip_runtime.h>

// RDNA4-optimized HFQ6-G256 GEMV.
// 2x group unroll + software prefetch. ~42 VGPRs (RDNA4 budget: 96 at 16 waves).

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq6g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int pairs = groups_per_row / 2;
    const int remainder = groups_per_row & 1;

    for (int gp = 0; gp < pairs; gp++) {
        const int g = gp * 2;
        const char* gptr0 = row_ptr + g * 200;
        const char* gptr1 = gptr0 + 200;

        // Prefetch 2 groups ahead (400 bytes)
        if (gp + 1 < pairs) {
            __builtin_amdgcn_s_prefetch_data((const void*)(gptr0 + 400), 0);
        }

        // Group 0
        float scale0 = __builtin_bit_cast(float, *(const unsigned int*)(gptr0));
        float zero0  = __builtin_bit_cast(float, *(const unsigned int*)(gptr0 + 4));
        const unsigned char* d0 = (const unsigned char*)(gptr0 + 8);
        int base0 = g * 256 + tid * 8;
        int boff0 = tid * 6;

        unsigned char b0=d0[boff0], b1=d0[boff0+1], b2=d0[boff0+2];
        unsigned char b3=d0[boff0+3], b4=d0[boff0+4], b5=d0[boff0+5];
        acc0 += (scale0 * (float)(b0 & 63) + zero0) * x[base0]
              + (scale0 * (float)((b0>>6)|((b1&0xF)<<2)) + zero0) * x[base0+1]
              + (scale0 * (float)((b1>>4)|((b2&3)<<4)) + zero0) * x[base0+2]
              + (scale0 * (float)(b2>>2) + zero0) * x[base0+3]
              + (scale0 * (float)(b3 & 63) + zero0) * x[base0+4]
              + (scale0 * (float)((b3>>6)|((b4&0xF)<<2)) + zero0) * x[base0+5]
              + (scale0 * (float)((b4>>4)|((b5&3)<<4)) + zero0) * x[base0+6]
              + (scale0 * (float)(b5>>2) + zero0) * x[base0+7];

        // Group 1
        float scale1 = __builtin_bit_cast(float, *(const unsigned int*)(gptr1));
        float zero1  = __builtin_bit_cast(float, *(const unsigned int*)(gptr1 + 4));
        const unsigned char* d1 = (const unsigned char*)(gptr1 + 8);
        int base1 = (g+1) * 256 + tid * 8;
        int boff1 = tid * 6;

        b0=d1[boff1]; b1=d1[boff1+1]; b2=d1[boff1+2];
        b3=d1[boff1+3]; b4=d1[boff1+4]; b5=d1[boff1+5];
        acc1 += (scale1 * (float)(b0 & 63) + zero1) * x[base1]
              + (scale1 * (float)((b0>>6)|((b1&0xF)<<2)) + zero1) * x[base1+1]
              + (scale1 * (float)((b1>>4)|((b2&3)<<4)) + zero1) * x[base1+2]
              + (scale1 * (float)(b2>>2) + zero1) * x[base1+3]
              + (scale1 * (float)(b3 & 63) + zero1) * x[base1+4]
              + (scale1 * (float)((b3>>6)|((b4&0xF)<<2)) + zero1) * x[base1+5]
              + (scale1 * (float)((b4>>4)|((b5&3)<<4)) + zero1) * x[base1+6]
              + (scale1 * (float)(b5>>2) + zero1) * x[base1+7];
    }

    if (remainder) {
        const int g = groups_per_row - 1;
        const char* gptr = row_ptr + g * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* d = (const unsigned char*)(gptr + 8);
        int base = g * 256 + tid * 8;
        int boff = tid * 6;

        unsigned char b0=d[boff], b1=d[boff+1], b2=d[boff+2];
        unsigned char b3=d[boff+3], b4=d[boff+4], b5=d[boff+5];
        acc0 += (scale * (float)(b0 & 63) + zero) * x[base]
              + (scale * (float)((b0>>6)|((b1&0xF)<<2)) + zero) * x[base+1]
              + (scale * (float)((b1>>4)|((b2&3)<<4)) + zero) * x[base+2]
              + (scale * (float)(b2>>2) + zero) * x[base+3]
              + (scale * (float)(b3 & 63) + zero) * x[base+4]
              + (scale * (float)((b3>>6)|((b4&0xF)<<2)) + zero) * x[base+5]
              + (scale * (float)((b4>>4)|((b5&3)<<4)) + zero) * x[base+6]
              + (scale * (float)(b5>>2) + zero) * x[base+7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
