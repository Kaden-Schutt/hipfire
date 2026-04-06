#include <hip/hip_runtime.h>

// RDNA2 variant 5: cache-aggressive
// Restructures memory access for Infinity Cache line alignment.
// Navi 21 Infinity Cache: 128MB, 16-way associative, 64B cache lines.
// Key insight: HFQ4-G256 group = 136 bytes = 2.125 cache lines. The 8-byte
// scale/zero header straddles a line boundary at byte 64. By loading the
// header first and the 128B nibble body as two aligned 64B halves, we avoid
// partial-line fetches and maximize IC hit rate.
//
// Also: process groups in a strided order so that adjacent warps access
// non-overlapping cache sets, reducing IC eviction pressure.

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

    // 2x unroll with cache-line-aware access pattern
    const int pairs = groups_per_row / 2;
    const int remainder = groups_per_row & 1;

    for (int gp = 0; gp < pairs; gp++) {
        const int g = gp * 2;

        // Pre-load both group headers (8B each) — these sit at 136-byte boundaries.
        // Loading both before touching nibble data gives the memory controller
        // time to fetch the nibble cache lines while we compute addresses.
        const char* gptr0 = row_ptr + g * 136;
        const char* gptr1 = gptr0 + 136;

        float scale0 = __builtin_bit_cast(float, *(const unsigned int*)(gptr0));
        float zero0  = __builtin_bit_cast(float, *(const unsigned int*)(gptr0 + 4));
        float scale1 = __builtin_bit_cast(float, *(const unsigned int*)(gptr1));
        float zero1  = __builtin_bit_cast(float, *(const unsigned int*)(gptr1 + 4));

        // Group 0 nibbles: offset 8 within 136B group = start at 64B-aligned + 8
        // Thread tid reads 4 bytes at offset tid*4 within the 128B nibble region.
        // With 32 threads × 4 bytes = 128B total = exactly 2 cache lines.
        const unsigned char* nib0 = (const unsigned char*)(gptr0 + 8);
        int boff = tid * 4;

        // Use 32-bit packed load for 4 consecutive bytes (naturally aligned
        // since tid*4 is always 4-byte-aligned within the nibble region)
        unsigned int pk0 = *(const unsigned int*)(nib0 + boff);
        unsigned char b0_0 = pk0 & 0xFF;
        unsigned char b0_1 = (pk0 >> 8) & 0xFF;
        unsigned char b0_2 = (pk0 >> 16) & 0xFF;
        unsigned char b0_3 = (pk0 >> 24) & 0xFF;

        int base0 = g * 256 + tid * 8;

        // Factored computation: scale * sum(nib*x) + zero * sum(x)
        // Saves 6 FP32 multiplies vs naive per-element (scale*nib+zero)*x
        float nx0 = (float)(b0_0 & 0xF) * x[base0]
                   + (float)(b0_0 >> 4)  * x[base0 + 1]
                   + (float)(b0_1 & 0xF) * x[base0 + 2]
                   + (float)(b0_1 >> 4)  * x[base0 + 3]
                   + (float)(b0_2 & 0xF) * x[base0 + 4]
                   + (float)(b0_2 >> 4)  * x[base0 + 5]
                   + (float)(b0_3 & 0xF) * x[base0 + 6]
                   + (float)(b0_3 >> 4)  * x[base0 + 7];

        float sx0 = x[base0] + x[base0+1] + x[base0+2] + x[base0+3]
                   + x[base0+4] + x[base0+5] + x[base0+6] + x[base0+7];

        acc0 += scale0 * nx0 + zero0 * sx0;

        // Group 1 — same pattern, already prefetched by header loads above
        const unsigned char* nib1 = (const unsigned char*)(gptr1 + 8);
        unsigned int pk1 = *(const unsigned int*)(nib1 + boff);
        unsigned char b1_0 = pk1 & 0xFF;
        unsigned char b1_1 = (pk1 >> 8) & 0xFF;
        unsigned char b1_2 = (pk1 >> 16) & 0xFF;
        unsigned char b1_3 = (pk1 >> 24) & 0xFF;

        int base1 = (g + 1) * 256 + tid * 8;

        float nx1 = (float)(b1_0 & 0xF) * x[base1]
                   + (float)(b1_0 >> 4)  * x[base1 + 1]
                   + (float)(b1_1 & 0xF) * x[base1 + 2]
                   + (float)(b1_1 >> 4)  * x[base1 + 3]
                   + (float)(b1_2 & 0xF) * x[base1 + 4]
                   + (float)(b1_2 >> 4)  * x[base1 + 5]
                   + (float)(b1_3 & 0xF) * x[base1 + 6]
                   + (float)(b1_3 >> 4)  * x[base1 + 7];

        float sx1 = x[base1] + x[base1+1] + x[base1+2] + x[base1+3]
                   + x[base1+4] + x[base1+5] + x[base1+6] + x[base1+7];

        acc1 += scale1 * nx1 + zero1 * sx1;
    }

    if (remainder) {
        const int g = groups_per_row - 1;
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nib = (const unsigned char*)(gptr + 8);
        int boff = tid * 4;

        unsigned int pk = *(const unsigned int*)(nib + boff);
        unsigned char b0 = pk & 0xFF;
        unsigned char b1 = (pk >> 8) & 0xFF;
        unsigned char b2 = (pk >> 16) & 0xFF;
        unsigned char b3 = (pk >> 24) & 0xFF;

        int base = g * 256 + tid * 8;

        float nx = (float)(b0 & 0xF) * x[base]
                 + (float)(b0 >> 4)  * x[base + 1]
                 + (float)(b1 & 0xF) * x[base + 2]
                 + (float)(b1 >> 4)  * x[base + 3]
                 + (float)(b2 & 0xF) * x[base + 4]
                 + (float)(b2 >> 4)  * x[base + 5]
                 + (float)(b3 & 0xF) * x[base + 6]
                 + (float)(b3 >> 4)  * x[base + 7];

        float sx = x[base] + x[base+1] + x[base+2] + x[base+3]
                 + x[base+4] + x[base+5] + x[base+6] + x[base+7];

        acc0 += scale * nx + zero * sx;
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
