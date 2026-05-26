// MagnumQuant 4-bit GEMV with per-block 16-entry fp16 Lloyd-Max codebook.
//
// Block layout (160 B/group, +17.6% bandwidth over uniform MQ4's 136 B):
//   bytes [0..32)   : 16 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [32..160) : 128 bytes packed 4-bit indices (low nibble = idx[2i],
//                     high nibble = idx[2i+1]), same nibble layout as uniform MQ4.
//
// Per-thread work: 8 weights / thread × 32 threads = 256 weights / block.
// Reconstruction is `cb[q]` — register-resident 16-entry codebook lookup.
// This is the slow generic chip variant; LDS-codebook fast path mirrors
// gemv_mq3g256_lloyd.gfx1100.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256_lloyd(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 160;
    const int boff = tid * 4;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 160;

        // Load 16 fp16 codebook entries → 16 floats (register resident).
        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb0  = __half2float(cb_h[0]);
        float cb1  = __half2float(cb_h[1]);
        float cb2  = __half2float(cb_h[2]);
        float cb3  = __half2float(cb_h[3]);
        float cb4  = __half2float(cb_h[4]);
        float cb5  = __half2float(cb_h[5]);
        float cb6  = __half2float(cb_h[6]);
        float cb7  = __half2float(cb_h[7]);
        float cb8  = __half2float(cb_h[8]);
        float cb9  = __half2float(cb_h[9]);
        float cb10 = __half2float(cb_h[10]);
        float cb11 = __half2float(cb_h[11]);
        float cb12 = __half2float(cb_h[12]);
        float cb13 = __half2float(cb_h[13]);
        float cb14 = __half2float(cb_h[14]);
        float cb15 = __half2float(cb_h[15]);

        const unsigned char* data = (const unsigned char*)(gptr + 32);

        // 8 × 4-bit nibble unpack (identical to uniform MQ4):
        //   data[boff + i] = idx[2i] (low) | idx[2i+1] (high)
        unsigned int b0 = (unsigned int)data[boff];
        unsigned int b1 = (unsigned int)data[boff + 1];
        unsigned int b2 = (unsigned int)data[boff + 2];
        unsigned int b3 = (unsigned int)data[boff + 3];

        unsigned int q0 =  b0       & 0xFu;
        unsigned int q1 = (b0 >> 4) & 0xFu;
        unsigned int q2 =  b1       & 0xFu;
        unsigned int q3 = (b1 >> 4) & 0xFu;
        unsigned int q4 =  b2       & 0xFu;
        unsigned int q5 = (b2 >> 4) & 0xFu;
        unsigned int q6 =  b3       & 0xFu;
        unsigned int q7 = (b3 >> 4) & 0xFu;

        int base_idx = g * 256 + tid * 8;

        // 16-entry codebook lookup as a binary chain. Compiler folds to
        // either a sequence of conditional moves or a small jump table on
        // RDNA3; either way the lookup stays register-resident.
        #define CB_LOOKUP4(q) (                                                          \
            (q) ==  0 ? cb0  : (q) ==  1 ? cb1  : (q) ==  2 ? cb2  : (q) ==  3 ? cb3  :  \
            (q) ==  4 ? cb4  : (q) ==  5 ? cb5  : (q) ==  6 ? cb6  : (q) ==  7 ? cb7  :  \
            (q) ==  8 ? cb8  : (q) ==  9 ? cb9  : (q) == 10 ? cb10 : (q) == 11 ? cb11 :  \
            (q) == 12 ? cb12 : (q) == 13 ? cb13 : (q) == 14 ? cb14 : cb15                \
        )

        acc += CB_LOOKUP4(q0) * x[base_idx]
             + CB_LOOKUP4(q1) * x[base_idx + 1]
             + CB_LOOKUP4(q2) * x[base_idx + 2]
             + CB_LOOKUP4(q3) * x[base_idx + 3]
             + CB_LOOKUP4(q4) * x[base_idx + 4]
             + CB_LOOKUP4(q5) * x[base_idx + 5]
             + CB_LOOKUP4(q6) * x[base_idx + 6]
             + CB_LOOKUP4(q7) * x[base_idx + 7];

        #undef CB_LOOKUP4
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
