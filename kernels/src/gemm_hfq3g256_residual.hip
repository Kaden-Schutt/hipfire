#include <hip/hip_runtime.h>

// Batched HFQ3-G256 GEMM with fused residual add:
//
//   for b in 0..N:
//     Y[b, row] += A[row] · X[b]         (residual accumulation)
//
// HFQ3 sibling of gemm_hfq4g256_residual.hip — same dispatch shape and
// batching tile; only the weight unpack differs (104 B/group, tid*3 byte
// offset, uint24 byte-combine for 8 3-bit trits per group per thread).
// See gemm_qkv_hfq3g256.hip header for the rationale on the byte-level
// combine vs uint32 load.
//
// Byte-exact with running gemv_hfq3g256_residual N times when N=1.

#define BATCH_TILE 8

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_hfq3g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ X,   // [batch_size, K], row-major
    float* __restrict__ Y,         // [batch_size, M], row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 104;
    const int boff = tid * 3;

    float acc0[BATCH_TILE], acc1[BATCH_TILE], acc2[BATCH_TILE], acc3[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
        acc2[b] = 0.0f;
        acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

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

        float w0_0 = sc0 * (float)((pk0      ) & 7u) + zp0;
        float w0_1 = sc0 * (float)((pk0 >>  3) & 7u) + zp0;
        float w0_2 = sc0 * (float)((pk0 >>  6) & 7u) + zp0;
        float w0_3 = sc0 * (float)((pk0 >>  9) & 7u) + zp0;
        float w0_4 = sc0 * (float)((pk0 >> 12) & 7u) + zp0;
        float w0_5 = sc0 * (float)((pk0 >> 15) & 7u) + zp0;
        float w0_6 = sc0 * (float)((pk0 >> 18) & 7u) + zp0;
        float w0_7 = sc0 * (float)((pk0 >> 21) & 7u) + zp0;

        float w1_0 = sc1 * (float)((pk1      ) & 7u) + zp1;
        float w1_1 = sc1 * (float)((pk1 >>  3) & 7u) + zp1;
        float w1_2 = sc1 * (float)((pk1 >>  6) & 7u) + zp1;
        float w1_3 = sc1 * (float)((pk1 >>  9) & 7u) + zp1;
        float w1_4 = sc1 * (float)((pk1 >> 12) & 7u) + zp1;
        float w1_5 = sc1 * (float)((pk1 >> 15) & 7u) + zp1;
        float w1_6 = sc1 * (float)((pk1 >> 18) & 7u) + zp1;
        float w1_7 = sc1 * (float)((pk1 >> 21) & 7u) + zp1;

        float w2_0 = sc2 * (float)((pk2      ) & 7u) + zp2;
        float w2_1 = sc2 * (float)((pk2 >>  3) & 7u) + zp2;
        float w2_2 = sc2 * (float)((pk2 >>  6) & 7u) + zp2;
        float w2_3 = sc2 * (float)((pk2 >>  9) & 7u) + zp2;
        float w2_4 = sc2 * (float)((pk2 >> 12) & 7u) + zp2;
        float w2_5 = sc2 * (float)((pk2 >> 15) & 7u) + zp2;
        float w2_6 = sc2 * (float)((pk2 >> 18) & 7u) + zp2;
        float w2_7 = sc2 * (float)((pk2 >> 21) & 7u) + zp2;

        float w3_0 = sc3 * (float)((pk3      ) & 7u) + zp3;
        float w3_1 = sc3 * (float)((pk3 >>  3) & 7u) + zp3;
        float w3_2 = sc3 * (float)((pk3 >>  6) & 7u) + zp3;
        float w3_3 = sc3 * (float)((pk3 >>  9) & 7u) + zp3;
        float w3_4 = sc3 * (float)((pk3 >> 12) & 7u) + zp3;
        float w3_5 = sc3 * (float)((pk3 >> 15) & 7u) + zp3;
        float w3_6 = sc3 * (float)((pk3 >> 18) & 7u) + zp3;
        float w3_7 = sc3 * (float)((pk3 >> 21) & 7u) + zp3;

        const int base = g * 256 + tid * 8;

        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const float* xb = X + (long long)(batch_start + b) * K;
            const float x0 = xb[base    ];
            const float x1 = xb[base + 1];
            const float x2 = xb[base + 2];
            const float x3 = xb[base + 3];
            const float x4 = xb[base + 4];
            const float x5 = xb[base + 5];
            const float x6 = xb[base + 6];
            const float x7 = xb[base + 7];

            acc0[b] += w0_0 * x0 + w0_1 * x1 + w0_2 * x2 + w0_3 * x3
                     + w0_4 * x4 + w0_5 * x5 + w0_6 * x6 + w0_7 * x7;

            const float x1_0 = xb[base + 256    ];
            const float x1_1 = xb[base + 256 + 1];
            const float x1_2 = xb[base + 256 + 2];
            const float x1_3 = xb[base + 256 + 3];
            const float x1_4 = xb[base + 256 + 4];
            const float x1_5 = xb[base + 256 + 5];
            const float x1_6 = xb[base + 256 + 6];
            const float x1_7 = xb[base + 256 + 7];

            acc1[b] += w1_0 * x1_0 + w1_1 * x1_1 + w1_2 * x1_2 + w1_3 * x1_3
                     + w1_4 * x1_4 + w1_5 * x1_5 + w1_6 * x1_6 + w1_7 * x1_7;

            const float x2_0 = xb[base + 512    ];
            const float x2_1 = xb[base + 512 + 1];
            const float x2_2 = xb[base + 512 + 2];
            const float x2_3 = xb[base + 512 + 3];
            const float x2_4 = xb[base + 512 + 4];
            const float x2_5 = xb[base + 512 + 5];
            const float x2_6 = xb[base + 512 + 6];
            const float x2_7 = xb[base + 512 + 7];

            acc2[b] += w2_0 * x2_0 + w2_1 * x2_1 + w2_2 * x2_2 + w2_3 * x2_3
                     + w2_4 * x2_4 + w2_5 * x2_5 + w2_6 * x2_6 + w2_7 * x2_7;

            const float x3_0 = xb[base + 768    ];
            const float x3_1 = xb[base + 768 + 1];
            const float x3_2 = xb[base + 768 + 2];
            const float x3_3 = xb[base + 768 + 3];
            const float x3_4 = xb[base + 768 + 4];
            const float x3_5 = xb[base + 768 + 5];
            const float x3_6 = xb[base + 768 + 6];
            const float x3_7 = xb[base + 768 + 7];

            acc3[b] += w3_0 * x3_0 + w3_1 * x3_1 + w3_2 * x3_2 + w3_3 * x3_3
                     + w3_4 * x3_4 + w3_5 * x3_5 + w3_6 * x3_6 + w3_7 * x3_7;
        }
    }

    #define TAIL_BODY(acc_arr)                                                     \
        {                                                                           \
            const char* gp = row_ptr + g * 104;                                     \
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));       \
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));   \
            const unsigned char* d = (const unsigned char*)(gp + 8) + boff;         \
            unsigned int pk = (unsigned int)d[0]                                    \
                            | ((unsigned int)d[1] << 8)                             \
                            | ((unsigned int)d[2] << 16);                           \
            float w0 = sc * (float)((pk      ) & 7u) + zp;                          \
            float w1 = sc * (float)((pk >>  3) & 7u) + zp;                          \
            float w2 = sc * (float)((pk >>  6) & 7u) + zp;                          \
            float w3 = sc * (float)((pk >>  9) & 7u) + zp;                          \
            float w4 = sc * (float)((pk >> 12) & 7u) + zp;                          \
            float w5 = sc * (float)((pk >> 15) & 7u) + zp;                          \
            float w6 = sc * (float)((pk >> 18) & 7u) + zp;                          \
            float w7 = sc * (float)((pk >> 21) & 7u) + zp;                          \
            const int base = g * 256 + tid * 8;                                     \
            _Pragma("unroll")                                                       \
            for (int b = 0; b < BATCH_TILE; b++) {                                  \
                if (b >= local_bs) break;                                           \
                const float* xb = X + (long long)(batch_start + b) * K;             \
                (acc_arr)[b] += w0 * xb[base    ] + w1 * xb[base + 1]               \
                              + w2 * xb[base + 2] + w3 * xb[base + 3]               \
                              + w4 * xb[base + 4] + w5 * xb[base + 5]               \
                              + w6 * xb[base + 6] + w7 * xb[base + 7];              \
            }                                                                       \
        }

    if (tail >= 1) {
        const int g = quads << 2;
        TAIL_BODY(acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        TAIL_BODY(acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        TAIL_BODY(acc2);
    }
    #undef TAIL_BODY

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row] += val;
    }
}
