#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1100 (RDNA3) MQ4G256-Lloyd GEMV: 4× group unroll + LDS-resident
// 16-entry codebook + SINGLE linear accumulator.
//
// Per group (160 B): 32 B fp16 codebook (16 entries) + 128 B data
// (256 weights packed as 128 nibble-pair bytes — low nibble = idx[2i],
// high nibble = idx[2i+1]). 32 threads × 8 weights/group = 256 weights.
// Each thread owns 4 bytes of packed 4-bit weights at byte_off = 32 + tid*4.
//
// LDS layout: 64 fp32 entries (256 B), one row per group:
//   cb_lds[ 0..15] = group 0's 16 codebook entries
//   cb_lds[16..31] = group 1's
//   cb_lds[32..47] = group 2's
//   cb_lds[48..63] = group 3's
// Two-phase cooperative load: phase 1 covers groups 0/1 (lds=0..31),
// phase 2 covers groups 2/3 (lds=32..63). Each phase = 32 threads × 1
// fp16 read with one __syncthreads barrier between.
//
// **Single accumulator, NOT multi-accumulator.** Unlike
// gemv_mq3g256_lloyd.gfx1100.hip (which uses 4 separate `acc0..acc3`
// registers and merges them as `(acc0+acc1)+(acc2+acc3)` after the K4
// loop), this kernel sums all four group contributions per quad-iter
// into a single linear `acc` register. Empirically required: the K4 +
// multi-accumulator + (acc0+acc1)+(acc2+acc3) merge produced a 1.7%
// PPL drift on Qwen3.5-9B real weights vs the slow generic kernel,
// while passing the synthetic parity test at fp32 reorder tolerance.
// Single-acc is byte-equal to the slow kernel at PPL precision, with
// the same speed (~31 tok/s on gfx1151 vs slow's 9.3 tok/s) and lower
// VGPR pressure (71 vs 74). Why this asymmetry vs MQ3-Lloyd is open
// (#182 follow-up); do NOT "optimize" this kernel back to the
// multi-accumulator pattern without re-validating PPL byte-equality.

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

    __shared__ float cb_lds[64];

    float acc = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // ----- main K4 loop with single accumulator -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 160;
        const char* gp1 = gp0 + 160;
        const char* gp2 = gp1 + 160;
        const char* gp3 = gp2 + 160;

        // Cooperative two-phase load: 32 threads × 1 fp16 → 32 LDS entries
        // per phase. Phase 1 covers groups 0/1, phase 2 covers groups 2/3.
        {
            const int g1 = tid >> 4;
            const int e1 = tid & 0xF;
            const __half* cb_h1 = (const __half*)(gp0 + g1 * 160);
            cb_lds[(g1 << 4) + e1] = __half2float(cb_h1[e1]);
        }
        __syncthreads();
        {
            const int g2 = (tid >> 4) + 2;
            const int e2 = tid & 0xF;
            const __half* cb_h2 = (const __half*)(gp0 + g2 * 160);
            cb_lds[(g2 << 4) + e2] = __half2float(cb_h2[e2]);
        }
        __syncthreads();

        // Read packed-uint32 indices for all 4 groups.
        const unsigned char* d0 = (const unsigned char*)(gp0 + 32) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 32) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 32) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 32) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8)
                         | ((unsigned int)d0[2] << 16) | ((unsigned int)d0[3] << 24);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8)
                         | ((unsigned int)d1[2] << 16) | ((unsigned int)d1[3] << 24);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8)
                         | ((unsigned int)d2[2] << 16) | ((unsigned int)d2[3] << 24);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8)
                         | ((unsigned int)d3[2] << 16) | ((unsigned int)d3[3] << 24);

        int base = g * 256 + tid * 8;

        #define DOG4_LDS(pk, lds, b) ( \
              cb_lds[(lds) +  ((pk)        & 0xFu)] * x[(b)]      \
            + cb_lds[(lds) + (((pk) >> 4)  & 0xFu)] * x[(b) + 1]  \
            + cb_lds[(lds) + (((pk) >> 8)  & 0xFu)] * x[(b) + 2]  \
            + cb_lds[(lds) + (((pk) >> 12) & 0xFu)] * x[(b) + 3]  \
            + cb_lds[(lds) + (((pk) >> 16) & 0xFu)] * x[(b) + 4]  \
            + cb_lds[(lds) + (((pk) >> 20) & 0xFu)] * x[(b) + 5]  \
            + cb_lds[(lds) + (((pk) >> 24) & 0xFu)] * x[(b) + 6]  \
            + cb_lds[(lds) + (((pk) >> 28) & 0xFu)] * x[(b) + 7])

        // Single accumulator, linear order — same accumulation order as the
        // slow generic kernel.
        acc += DOG4_LDS(pk0,  0, base);
        acc += DOG4_LDS(pk1, 16, base + 256);
        acc += DOG4_LDS(pk2, 32, base + 512);
        acc += DOG4_LDS(pk3, 48, base + 768);
        #undef DOG4_LDS
    }

    // ----- tail loop -----
    // Reuses cb_lds[0..15] for one group's codebook. Lanes 0..15 each load
    // one fp16; barrier; everyone reads.
    #define TAIL_LOAD_AND_DOT(g_expr) do {                                     \
        const int g = (g_expr);                                                 \
        const char* gp = row_ptr + g * 160;                                     \
        if (tid < 16) {                                                         \
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);               \
        }                                                                       \
        __syncthreads();                                                        \
        const unsigned char* d = (const unsigned char*)(gp + 32) + boff;        \
        unsigned int pk = (unsigned int)d[0]                                    \
                        | ((unsigned int)d[1] << 8)                             \
                        | ((unsigned int)d[2] << 16)                            \
                        | ((unsigned int)d[3] << 24);                           \
        int base = g * 256 + tid * 8;                                           \
        acc += cb_lds[ (pk)        & 0xFu] * x[base]                            \
             + cb_lds[((pk) >> 4)  & 0xFu] * x[base + 1]                        \
             + cb_lds[((pk) >> 8)  & 0xFu] * x[base + 2]                        \
             + cb_lds[((pk) >> 12) & 0xFu] * x[base + 3]                        \
             + cb_lds[((pk) >> 16) & 0xFu] * x[base + 4]                        \
             + cb_lds[((pk) >> 20) & 0xFu] * x[base + 5]                        \
             + cb_lds[((pk) >> 24) & 0xFu] * x[base + 6]                        \
             + cb_lds[((pk) >> 28) & 0xFu] * x[base + 7];                       \
        __syncthreads();                                                        \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2);
    #undef TAIL_LOAD_AND_DOT

    // Wave reduction.
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
