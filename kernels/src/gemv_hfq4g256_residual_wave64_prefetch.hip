#include <hip/hip_runtime.h>

// Software-pipelined variant of gemv_hfq4g256_residual_wave64. Identical
// math; the only structural change is across-quad weight prefetch — quad
// q+1's 12 dwords (4×sc, 4×zp, 4×packed) are issued *before* quad q's
// compute chain runs, so the L2 fills overlap with the FMA chain instead
// of stalling the load unit each iteration.
//
// Motivation (gfx906 / MI50):
//   The non-prefetched kernel hits 39.5 % L2 hit rate per PMC L2CacheHit
//   pass — meaning ~60 % of weight fetches miss to HBM. The activation
//   x[] easily fits L2; the misses are weights, streamed-once per WG with
//   no reuse within a token. Issuing q+1's loads while q computes shifts
//   weight fetches into the L2-hit regime by the time the compute hits
//   them.
//
// Layout / topology unchanged: block=[64,1,1], 2 rows/WG (warp_id=0,1),
// 4-quad interleave with 4 accumulators. Bit-exact with the base kernel
// modulo IEEE FMA reordering tolerance.
//
// VGPR cost: +12 dwords for the staged-load buffer. Base kernel uses 29
// VGPR; this variant should land around 41-44 VGPR (still 5-6 waves/SIMD
// on gfx906, well above HBM-saturating).

extern "C" __global__ void gemv_hfq4g256_residual_wave64_prefetch(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = lane * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    #define LOAD_QUAD(q_idx, sc0v, zp0v, sc1v, zp1v, sc2v, zp2v, sc3v, zp3v, pk0v, pk1v, pk2v, pk3v) \
        do { \
            const int g_ = (q_idx) << 2; \
            const char* gp0_ = row_ptr + g_ * 136; \
            const char* gp1_ = gp0_ + 136; \
            const char* gp2_ = gp1_ + 136; \
            const char* gp3_ = gp2_ + 136; \
            (sc0v) = __builtin_bit_cast(float, *(const unsigned int*)(gp0_));     \
            (zp0v) = __builtin_bit_cast(float, *(const unsigned int*)(gp0_ + 4)); \
            (sc1v) = __builtin_bit_cast(float, *(const unsigned int*)(gp1_));     \
            (zp1v) = __builtin_bit_cast(float, *(const unsigned int*)(gp1_ + 4)); \
            (sc2v) = __builtin_bit_cast(float, *(const unsigned int*)(gp2_));     \
            (zp2v) = __builtin_bit_cast(float, *(const unsigned int*)(gp2_ + 4)); \
            (sc3v) = __builtin_bit_cast(float, *(const unsigned int*)(gp3_));     \
            (zp3v) = __builtin_bit_cast(float, *(const unsigned int*)(gp3_ + 4)); \
            (pk0v) = *(const unsigned int*)(gp0_ + 8 + boff); \
            (pk1v) = *(const unsigned int*)(gp1_ + 8 + boff); \
            (pk2v) = *(const unsigned int*)(gp2_ + 8 + boff); \
            (pk3v) = *(const unsigned int*)(gp3_ + 8 + boff); \
        } while (0)

    if (quads > 0) {
        // Prologue: load quad 0 into the staged registers.
        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        unsigned int pk0, pk1, pk2, pk3;
        LOAD_QUAD(0, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3, pk0, pk1, pk2, pk3);

        // Steady state: compute(q) then prefetch(q+1) into next-iter regs.
        // Compiler's job is to schedule the q+1 loads early in the body so
        // they overlap the 4×8 = 32 FMAs of the DOG chain.
        for (int q = 0; q < quads - 1; q++) {
            // Stage q+1 first — gives the load unit a head start on
            // returning the data while compute runs.
            float n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3;
            unsigned int n_pk0, n_pk1, n_pk2, n_pk3;
            LOAD_QUAD(q + 1, n_sc0, n_zp0, n_sc1, n_zp1, n_sc2, n_zp2, n_sc3, n_zp3,
                             n_pk0, n_pk1, n_pk2, n_pk3);

            int base = (q << 2) * 256 + lane * 8;
            DOG(pk0, sc0, zp0, base,         acc0);
            DOG(pk1, sc1, zp1, base + 256,   acc1);
            DOG(pk2, sc2, zp2, base + 512,   acc2);
            DOG(pk3, sc3, zp3, base + 768,   acc3);

            sc0 = n_sc0; zp0 = n_zp0; sc1 = n_sc1; zp1 = n_zp1;
            sc2 = n_sc2; zp2 = n_zp2; sc3 = n_sc3; zp3 = n_zp3;
            pk0 = n_pk0; pk1 = n_pk1; pk2 = n_pk2; pk3 = n_pk3;
        }

        // Epilogue: compute the last quad whose data was prefetched in the
        // prior loop iteration's tail (or the prologue, if quads == 1).
        {
            int base = ((quads - 1) << 2) * 256 + lane * 8;
            DOG(pk0, sc0, zp0, base,         acc0);
            DOG(pk1, sc1, zp1, base + 256,   acc1);
            DOG(pk2, sc2, zp2, base + 512,   acc2);
            DOG(pk3, sc3, zp3, base + 768,   acc3);
        }
    }
    #undef DOG
    #undef LOAD_QUAD

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + lane * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + lane * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + lane * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] += acc;
}
