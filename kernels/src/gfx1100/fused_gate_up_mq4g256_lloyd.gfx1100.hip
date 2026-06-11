#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1100 (RDNA3) Fused Gate+Up MQ4-Lloyd: two GEMVs in one launch (gate +
// up for the FFN). Mirrors fused_gate_up_mq3g256_lloyd.gfx1100.hip's gid-
// demux pattern, with the K4 + 64-slot LDS-codebook + SINGLE-accumulator
// inner body from gemv_mq4g256_lloyd.gfx1100.hip. See that file's header
// for why single-accumulator is required for MQ4-Lloyd.

__launch_bounds__(32, 16)
extern "C" __global__ void fused_gate_up_mq4g256_lloyd(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)local_row * groups_per_row * 160;
    const int boff = tid * 4;

    __shared__ float cb_lds[64];

    float acc = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 160;
        const char* gp1 = gp0 + 160;
        const char* gp2 = gp1 + 160;
        const char* gp3 = gp2 + 160;

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

        // Single accumulator — mandatory for MQ4-Lloyd (see header note).
        acc += DOG4_LDS(pk0,  0, base);
        acc += DOG4_LDS(pk1, 16, base + 256);
        acc += DOG4_LDS(pk2, 32, base + 512);
        acc += DOG4_LDS(pk3, 48, base + 768);
        #undef DOG4_LDS
    }

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

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[local_row] = acc;
}
