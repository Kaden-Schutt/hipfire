#include <hip/hip_runtime.h>

// RDNA2 variant 3: wide-unroll
// Maximizes ALU throughput per wave at the cost of occupancy.
// launch_bounds(32,12) → ~85 VGPRs budget, 4x group unroll with 4 accumulators.
// 12 waves/SIMD is well below max but each wave does 4x the work.
// Theory: on bandwidth-bound GEMV, the Infinity Cache keeps data flowing
// fast enough that fewer waves with deeper pipelines wins.

__launch_bounds__(32, 12)
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
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Process 4 groups per iteration (32 weights/thread/iter)
    const int quads = groups_per_row / 4;
    const int tail = groups_per_row - quads * 4;

    for (int gq = 0; gq < quads; gq++) {
        const int g = gq * 4;

        #pragma unroll 4
        for (int gi = 0; gi < 4; gi++) {
            const char* gptr = row_ptr + (g + gi) * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            const unsigned char* nib = (const unsigned char*)(gptr + 8);
            int base = (g + gi) * 256 + tid * 8;
            int boff = tid * 4;

            unsigned char b0 = nib[boff];
            unsigned char b1 = nib[boff + 1];
            unsigned char b2 = nib[boff + 2];
            unsigned char b3 = nib[boff + 3];

            // Distribute across 4 accumulators for max ILP
            float* acc = (gi == 0) ? &acc0 : (gi == 1) ? &acc1 : (gi == 2) ? &acc2 : &acc3;

            *acc += (sc * (float)(b0 & 0xF) + zp) * x[base]
                  + (sc * (float)(b0 >> 4)  + zp) * x[base + 1]
                  + (sc * (float)(b1 & 0xF) + zp) * x[base + 2]
                  + (sc * (float)(b1 >> 4)  + zp) * x[base + 3]
                  + (sc * (float)(b2 & 0xF) + zp) * x[base + 4]
                  + (sc * (float)(b2 >> 4)  + zp) * x[base + 5]
                  + (sc * (float)(b3 & 0xF) + zp) * x[base + 6]
                  + (sc * (float)(b3 >> 4)  + zp) * x[base + 7];
        }
    }

    // Handle 0-3 tail groups
    for (int gi = 0; gi < tail; gi++) {
        const int g = quads * 4 + gi;
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

    float acc = acc0 + acc1 + acc2 + acc3;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
