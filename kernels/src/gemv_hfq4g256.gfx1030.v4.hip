#include <hip/hip_runtime.h>

// RDNA2 variant 4: dp4a-packed
// Uses v_dot4_i32_i8 (dp4a) to compute 4 int8 multiply-accumulates per instruction.
// RDNA2 (gfx1030) has native dp4a support via __builtin_amdgcn_sdot4.
//
// Challenge: x is FP32, dp4a needs int8. We quantize x per-thread-tile (8 elements)
// to int8, use 2x dp4a for the nib*x_q dot product, then rescale to FP32.
// Nibbles are 0-15 (positive in signed int8), x_q is [-127,+127].
//
// Trade-off: quantization overhead + ~0.4% noise vs 2 dp4a replacing 8 FP32 FMAs.
// This tests whether RDNA2's dp4a throughput overcomes the setup cost.
// The factored formulation (scale * sum(nib*x) + zero * sum(x)) reduces the
// float multiply count regardless.

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

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nib = (const unsigned char*)(gptr + 8);
        int base = g * 256 + tid * 8;
        int boff = tid * 4;

        // Packed 32-bit load for 4 consecutive nibble bytes
        unsigned int pk = *(const unsigned int*)(nib + boff);
        unsigned char b0 = pk & 0xFF;
        unsigned char b1 = (pk >> 8) & 0xFF;
        unsigned char b2 = (pk >> 16) & 0xFF;
        unsigned char b3 = (pk >> 24) & 0xFF;

        // Load 8 x values
        float x0 = x[base];     float x1 = x[base + 1];
        float x2 = x[base + 2]; float x3 = x[base + 3];
        float x4 = x[base + 4]; float x5 = x[base + 5];
        float x6 = x[base + 6]; float x7 = x[base + 7];

        // sum(x) for zero-point correction — independent of dp4a path
        float sum_x = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

        // Find thread-local amax for x quantization
        float amax = fmaxf(fmaxf(fmaxf(fabsf(x0), fabsf(x1)),
                                  fmaxf(fabsf(x2), fabsf(x3))),
                           fmaxf(fmaxf(fabsf(x4), fabsf(x5)),
                                  fmaxf(fabsf(x6), fabsf(x7))));

        // Avoid division by zero; if all x are zero, result is zero anyway
        float x_scale = amax * (1.0f / 127.0f);
        float inv_x_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        // Quantize x to signed int8
        int xq0 = __float2int_rn(x0 * inv_x_scale);
        int xq1 = __float2int_rn(x1 * inv_x_scale);
        int xq2 = __float2int_rn(x2 * inv_x_scale);
        int xq3 = __float2int_rn(x3 * inv_x_scale);
        int xq4 = __float2int_rn(x4 * inv_x_scale);
        int xq5 = __float2int_rn(x5 * inv_x_scale);
        int xq6 = __float2int_rn(x6 * inv_x_scale);
        int xq7 = __float2int_rn(x7 * inv_x_scale);

        // Pack nibbles (even indices 0,2,4,6 and odd indices 1,3,5,7) into int8x4
        // Nibbles are 0-15, safe as signed int8 (positive range)
        int nib_even = (int)(b0 & 0xF)
                     | ((int)(b1 & 0xF) << 8)
                     | ((int)(b2 & 0xF) << 16)
                     | ((int)(b3 & 0xF) << 24);

        int nib_odd  = (int)(b0 >> 4)
                     | ((int)(b1 >> 4) << 8)
                     | ((int)(b2 >> 4) << 16)
                     | ((int)(b3 >> 4) << 24);

        // Pack quantized x values matching the nibble grouping
        int xq_even = (xq0 & 0xFF) | ((xq2 & 0xFF) << 8)
                    | ((xq4 & 0xFF) << 16) | ((xq6 & 0xFF) << 24);

        int xq_odd  = (xq1 & 0xFF) | ((xq3 & 0xFF) << 8)
                    | ((xq5 & 0xFF) << 16) | ((xq7 & 0xFF) << 24);

        // dp4a: 2 instructions compute dot(nib[0,2,4,6], xq[0,2,4,6])
        //        + dot(nib[1,3,5,7], xq[1,3,5,7])
        int dot_sum = __builtin_amdgcn_sdot4(nib_even, xq_even, 0, false);
        dot_sum = __builtin_amdgcn_sdot4(nib_odd, xq_odd, dot_sum, false);

        // Rescale: dot_sum ≈ sum(nib_i * x_i) / x_scale
        float nib_dot_x = (float)dot_sum * x_scale;

        // Final: acc += scale * sum(nib*x) + zero * sum(x)
        acc += scale * nib_dot_x + zero * sum_x;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
