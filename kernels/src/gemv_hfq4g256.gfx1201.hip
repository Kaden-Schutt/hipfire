#include <hip/hip_runtime.h>

// RDNA4-optimized HFQ4-G256 GEMV.
// Differences from baseline:
//   - 2x group unroll (process 2 groups per iteration, 16 weights/thread)
//   - Software prefetch: s_prefetch_data 2 groups ahead (~272 bytes)
//   - Higher VGPR budget: ~36 VGPRs (RDNA4 allows 96 at max 16-wave occupancy)
//   - Targets gfx1200/gfx1201 (Navi 44/48, RDNA4)

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Process 2 groups per iteration for better ILP
    const int pairs = groups_per_row / 2;
    const int remainder = groups_per_row & 1;

    for (int gp = 0; gp < pairs; gp++) {
        const int g = gp * 2;
        const char* gptr0 = row_ptr + g * 136;
        const char* gptr1 = gptr0 + 136;

        // Prefetch 2 groups ahead (272 bytes, well within 32KB distance)
        if (gp + 1 < pairs) {
            __builtin_amdgcn_s_prefetch_data((const void*)(gptr0 + 272), 0);
        }

        // Group 0
        float scale0 = __builtin_bit_cast(float, *(const unsigned int*)(gptr0));
        float zero0  = __builtin_bit_cast(float, *(const unsigned int*)(gptr0 + 4));
        const unsigned char* nib0 = (const unsigned char*)(gptr0 + 8);
        int base0 = g * 256 + tid * 8;
        int boff0 = tid * 4;

        unsigned char b0 = nib0[boff0];
        unsigned char b1 = nib0[boff0 + 1];
        unsigned char b2 = nib0[boff0 + 2];
        unsigned char b3 = nib0[boff0 + 3];

        acc0 += (scale0 * (float)(b0 & 0xF) + zero0) * x[base0]
              + (scale0 * (float)(b0 >> 4)  + zero0) * x[base0 + 1]
              + (scale0 * (float)(b1 & 0xF) + zero0) * x[base0 + 2]
              + (scale0 * (float)(b1 >> 4)  + zero0) * x[base0 + 3]
              + (scale0 * (float)(b2 & 0xF) + zero0) * x[base0 + 4]
              + (scale0 * (float)(b2 >> 4)  + zero0) * x[base0 + 5]
              + (scale0 * (float)(b3 & 0xF) + zero0) * x[base0 + 6]
              + (scale0 * (float)(b3 >> 4)  + zero0) * x[base0 + 7];

        // Group 1
        float scale1 = __builtin_bit_cast(float, *(const unsigned int*)(gptr1));
        float zero1  = __builtin_bit_cast(float, *(const unsigned int*)(gptr1 + 4));
        const unsigned char* nib1 = (const unsigned char*)(gptr1 + 8);
        int base1 = (g + 1) * 256 + tid * 8;
        int boff1 = tid * 4;

        b0 = nib1[boff1];
        b1 = nib1[boff1 + 1];
        b2 = nib1[boff1 + 2];
        b3 = nib1[boff1 + 3];

        acc1 += (scale1 * (float)(b0 & 0xF) + zero1) * x[base1]
              + (scale1 * (float)(b0 >> 4)  + zero1) * x[base1 + 1]
              + (scale1 * (float)(b1 & 0xF) + zero1) * x[base1 + 2]
              + (scale1 * (float)(b1 >> 4)  + zero1) * x[base1 + 3]
              + (scale1 * (float)(b2 & 0xF) + zero1) * x[base1 + 4]
              + (scale1 * (float)(b2 >> 4)  + zero1) * x[base1 + 5]
              + (scale1 * (float)(b3 & 0xF) + zero1) * x[base1 + 6]
              + (scale1 * (float)(b3 >> 4)  + zero1) * x[base1 + 7];
    }

    // Handle odd remainder group
    if (remainder) {
        const int g = groups_per_row - 1;
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nib = (const unsigned char*)(gptr + 8);
        int base = g * 256 + tid * 8;
        int boff = tid * 4;

        unsigned char b0 = nib[boff];
        unsigned char b1 = nib[boff + 1];
        unsigned char b2 = nib[boff + 2];
        unsigned char b3 = nib[boff + 3];

        acc0 += (scale * (float)(b0 & 0xF) + zero) * x[base]
              + (scale * (float)(b0 >> 4)  + zero) * x[base + 1]
              + (scale * (float)(b1 & 0xF) + zero) * x[base + 2]
              + (scale * (float)(b1 >> 4)  + zero) * x[base + 3]
              + (scale * (float)(b2 & 0xF) + zero) * x[base + 4]
              + (scale * (float)(b2 >> 4)  + zero) * x[base + 5]
              + (scale * (float)(b3 & 0xF) + zero) * x[base + 6]
              + (scale * (float)(b3 >> 4)  + zero) * x[base + 7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
