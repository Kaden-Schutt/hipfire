#include <hip/hip_runtime.h>

// HFQ3-G256 GEMV with fused residual add: y[row] += A[row] · x.
// Default (arch-agnostic) kernel matching the 4-accumulator pairwise-combine
// ordering of gemv_hfq3g256.gfx1100.hip so that cross-arch byte-exactness is
// preserved against MQ3 quality-gate baselines.
//
// Identical math to gemv_hfq3g256 (single-row per warp, 32 threads × 8 weights
// per group, packed uint24 unpack of 8×3 bits) except the final write
// accumulates into y instead of overwriting. Used for wo and w_down
// projections in the Qwen3.5 forward path to eliminate the alloc+gemv+add
// fallback in the MQ3 arm of weight_gemv_residual.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq3g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 104;
    const int boff = tid * 3;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 104;
        const char* gp1 = gp0 + 104;
        const char* gp2 = gp1 + 104;
        const char* gp3 = gp2 + 104;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        const unsigned char* d0 = (const unsigned char*)(gp0 + 8) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 8) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 8) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 8) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

        int base = g * 256 + tid * 8;

        #define DOG3(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 7u)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 3) & 7u)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 6) & 7u)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 9) & 7u)  + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 12) & 7u) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 15) & 7u) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 18) & 7u) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 21) & 7u) + zp) * x[(b) + 7]

        DOG3(pk0, sc0, zp0, base, acc0);
        DOG3(pk1, sc1, zp1, base + 256, acc1);
        DOG3(pk2, sc2, zp2, base + 512, acc2);
        DOG3(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG3
    }

    #define TAIL_DOG3(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 7u)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 3) & 7u)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 6) & 7u)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 9) & 7u)  + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 12) & 7u) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 15) & 7u) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 18) & 7u) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 21) & 7u) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 104;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8) | ((unsigned int)d[2] << 16);
        int base = g * 256 + tid * 8;
        TAIL_DOG3(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 104;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8) | ((unsigned int)d[2] << 16);
        int base = g * 256 + tid * 8;
        TAIL_DOG3(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 104;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8) | ((unsigned int)d[2] << 16);
        int base = g * 256 + tid * 8;
        TAIL_DOG3(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG3

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
