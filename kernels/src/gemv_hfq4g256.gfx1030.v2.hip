#include <hip/hip_runtime.h>

// RDNA2 variant 2: high-occupancy
// Tests the hypothesis: does RDNA2's Infinity Cache make 20 waves unnecessary?
// launch_bounds(32,20) like gfx1010 but WITH 2x group unroll + dual accumulators.
// Must fit in ~51 VGPRs (1024 VGPRs / 20 waves). The unroll adds ~4 VGPRs over
// baseline (scale1/zero1/acc1 + extra pointer), so ~22 total. Should fit easily.
// If this beats v1, Infinity Cache doesn't change the occupancy calculus.

__launch_bounds__(32, 20)
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

    const int pairs = groups_per_row / 2;
    const int remainder = groups_per_row & 1;

    for (int gp = 0; gp < pairs; gp++) {
        const int g = gp * 2;
        const char* gptr0 = row_ptr + g * 136;
        const char* gptr1 = gptr0 + 136;

        // Group 0: reuse byte vars to minimize live VGPRs
        {
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gptr0));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gptr0 + 4));
            const unsigned char* nib = (const unsigned char*)(gptr0 + 8);
            int base = g * 256 + tid * 8;
            int boff = tid * 4;

            unsigned char b0 = nib[boff];
            unsigned char b1 = nib[boff + 1];
            unsigned char b2 = nib[boff + 2];
            unsigned char b3 = nib[boff + 3];

            acc0 += (sc * (float)(b0 & 0xF) + zp) * x[base]
                  + (sc * (float)(b0 >> 4)  + zp) * x[base + 1]
                  + (sc * (float)(b1 & 0xF) + zp) * x[base + 2]
                  + (sc * (float)(b1 >> 4)  + zp) * x[base + 3]
                  + (sc * (float)(b2 & 0xF) + zp) * x[base + 4]
                  + (sc * (float)(b2 >> 4)  + zp) * x[base + 5]
                  + (sc * (float)(b3 & 0xF) + zp) * x[base + 6]
                  + (sc * (float)(b3 >> 4)  + zp) * x[base + 7];
        }

        // Group 1
        {
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gptr1));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gptr1 + 4));
            const unsigned char* nib = (const unsigned char*)(gptr1 + 8);
            int base = (g + 1) * 256 + tid * 8;
            int boff = tid * 4;

            unsigned char b0 = nib[boff];
            unsigned char b1 = nib[boff + 1];
            unsigned char b2 = nib[boff + 2];
            unsigned char b3 = nib[boff + 3];

            acc1 += (sc * (float)(b0 & 0xF) + zp) * x[base]
                  + (sc * (float)(b0 >> 4)  + zp) * x[base + 1]
                  + (sc * (float)(b1 & 0xF) + zp) * x[base + 2]
                  + (sc * (float)(b1 >> 4)  + zp) * x[base + 3]
                  + (sc * (float)(b2 & 0xF) + zp) * x[base + 4]
                  + (sc * (float)(b2 >> 4)  + zp) * x[base + 5]
                  + (sc * (float)(b3 & 0xF) + zp) * x[base + 6]
                  + (sc * (float)(b3 >> 4)  + zp) * x[base + 7];
        }
    }

    if (remainder) {
        const int g = groups_per_row - 1;
        const char* gptr = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nib = (const unsigned char*)(gptr + 8);
        int base = g * 256 + tid * 8;
        int boff = tid * 4;

        unsigned char b0 = nib[boff];
        unsigned char b1 = nib[boff + 1];
        unsigned char b2 = nib[boff + 2];
        unsigned char b3 = nib[boff + 3];

        acc0 += (sc * (float)(b0 & 0xF) + zp) * x[base]
              + (sc * (float)(b0 >> 4)  + zp) * x[base + 1]
              + (sc * (float)(b1 & 0xF) + zp) * x[base + 2]
              + (sc * (float)(b1 >> 4)  + zp) * x[base + 3]
              + (sc * (float)(b2 & 0xF) + zp) * x[base + 4]
              + (sc * (float)(b2 >> 4)  + zp) * x[base + 5]
              + (sc * (float)(b3 & 0xF) + zp) * x[base + 6]
              + (sc * (float)(b3 >> 4)  + zp) * x[base + 7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
