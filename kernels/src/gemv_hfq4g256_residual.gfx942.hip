#include <hip/hip_runtime.h>

// gfx942 (MI300X CDNA3) optimized GEMV with LDS-cached input vector.
//
// Improvements vs gemv_hfq4g256_residual_wave64 (the prior gfx94x path):
//   * 256 threads per WG = 4 wave64s × 2 logical 32-lane stripes = 8 stripes
//     → 8 output rows per WG (vs 2 in wave64 base)
//   * Grid = (M + 7) / 8 → 4× fewer WGs → better CU saturation
//     (M=5120 example: 640 WGs vs 2560 across 304 CUs)
//   * Cooperative LDS load of x[K] → each WG reads x ONCE then 8 rows reuse.
//     Cuts effective x-HBM traffic 4× vs prior kernel (which has 2 stripes
//     reading x independently per WG → no LDS sharing)
//   * Dynamic LDS (sized by launch param), handles K up to whatever fits
//
// Inner math is byte-identical to gemv_hfq4g256_residual_wave64: same
// 4-accumulator group-interleave, same DOG/TAIL_DOG, same pairwise
// combine, same 32-lane __shfl_down chain. Only the loads-from-x change
// from global to LDS.
//
// __shfl_down semantics on wave64: shfl operates within the wave64 (64
// lanes). With 4 wave64s per WG, each wave64 reduces independently. The
// chain offset=16/8/4/2/1 produces TWO outputs per wave64 (lane 0 holds
// sum of lanes 0..31, lane 32 holds sum of lanes 32..63). 4 wave64s × 2
// outputs = 8 row outputs per WG, matching the 8-stripe geometry.

extern "C" __global__ __launch_bounds__(256)
void gemv_hfq4g256_residual_gfx942(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float lds_x[];

    const int tid = threadIdx.x;

    // Cooperative load of x[K] into LDS. 256 threads, ceil(K/256) iters.
    for (int i = tid; i < K; i += 256) {
        lds_x[i] = x[i];
    }
    __syncthreads();

    // Per-stripe geometry: 8 stripes of 32 lanes each, indexed across the
    // 4 wave64s as (wave_id, stripe_in_wave) → flat stripe in [0, 7].
    const int wave_id = tid >> 6;          // 0..3 (which wave64)
    const int lane_in_wave = tid & 63;     // 0..63 within wave64
    const int stripe_in_wave = lane_in_wave >> 5;   // 0 or 1
    const int lane = lane_in_wave & 31;    // 0..31 within stripe
    const int stripe = (wave_id << 1) | stripe_in_wave;  // 0..7

    const int row = blockIdx.x * 8 + stripe;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = lane * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + lane * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * lds_x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * lds_x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * lds_x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * lds_x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * lds_x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * lds_x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * lds_x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * lds_x[(b) + 7]

        DOG(pk0, sc0, zp0, base,       acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * lds_x[(b)]     \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * lds_x[(b) + 1] \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * lds_x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * lds_x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * lds_x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * lds_x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * lds_x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * lds_x[(b) + 7]

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
    // wave64 __shfl_down chain — operates within the wave (64 lanes).
    // Lane 0 ends up holding the sum of stripe-0 (lanes 0..31), lane 32
    // ends up holding the sum of stripe-1 (lanes 32..63). Same shfl chain
    // as the wave64 base kernel; just runs in 4 waves in parallel here.
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] += acc;
}
