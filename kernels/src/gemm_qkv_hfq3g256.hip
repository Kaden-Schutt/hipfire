#include <hip/hip_runtime.h>

// Batched 3-way fused HFQ3-G256 GEMM for the Qwen3.5 FullAttention preamble.
// HFQ3 sibling of gemm_qkv_hfq4g256.hip — same dispatch shape, batching
// tile, and ILP pattern; only the weight unpack differs:
//
//   HFQ4 group: 136 B (8B scale/zero + 128B 4-bit nibbles, 4 B/thread)
//   HFQ3 group: 104 B (8B scale/zero + 96B  3-bit trits,   3 B/thread)
//
// Per-thread byte offset is tid*3 (not tid*4). Each thread reads 3 bytes
// per group and unpacks 8 trits via a uint24 combine + 3-bit-shift cascade
// (& 7u, >> 3, >> 6, …, >> 21). Cannot use a uint32 load here because the
// last thread's 3-byte slice ends at byte 95 of the 96-byte data region —
// a uint32 read would step 1 byte past the group end and either OOB on the
// last group of the row or pollute the next group's scale/zero header on
// earlier groups.
//
// Byte-exact with running the single-token HFQ3 GEMV N times when N=1,
// since the 4-accumulator interleave + pairwise final combine matches the
// gfx1100 / generic HFQ3 GEMV pattern.

#define BATCH_TILE 8

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_qkv_hfq3g256(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const float* __restrict__ X,     // [N × K]
    float*       __restrict__ Y_q,   // [N × q_m]
    float*       __restrict__ Y_k,   // [N × k_m]
    float*       __restrict__ Y_v,   // [N × v_m]
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = N - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < q_m) {
        A = A_q; Y = Y_q; row = gid;                out_stride = q_m;
    } else if (gid < q_m + k_m) {
        A = A_k; Y = Y_k; row = gid - q_m;          out_stride = k_m;
    } else {
        A = A_v; Y = Y_v; row = gid - q_m - k_m;    out_stride = v_m;
    }

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

            acc0[b] += w0_0 * xb[base    ] + w0_1 * xb[base + 1]
                     + w0_2 * xb[base + 2] + w0_3 * xb[base + 3]
                     + w0_4 * xb[base + 4] + w0_5 * xb[base + 5]
                     + w0_6 * xb[base + 6] + w0_7 * xb[base + 7];

            acc1[b] += w1_0 * xb[base + 256    ] + w1_1 * xb[base + 256 + 1]
                     + w1_2 * xb[base + 256 + 2] + w1_3 * xb[base + 256 + 3]
                     + w1_4 * xb[base + 256 + 4] + w1_5 * xb[base + 256 + 5]
                     + w1_6 * xb[base + 256 + 6] + w1_7 * xb[base + 256 + 7];

            acc2[b] += w2_0 * xb[base + 512    ] + w2_1 * xb[base + 512 + 1]
                     + w2_2 * xb[base + 512 + 2] + w2_3 * xb[base + 512 + 3]
                     + w2_4 * xb[base + 512 + 4] + w2_5 * xb[base + 512 + 5]
                     + w2_6 * xb[base + 512 + 6] + w2_7 * xb[base + 512 + 7];

            acc3[b] += w3_0 * xb[base + 768    ] + w3_1 * xb[base + 768 + 1]
                     + w3_2 * xb[base + 768 + 2] + w3_3 * xb[base + 768 + 3]
                     + w3_4 * xb[base + 768 + 4] + w3_5 * xb[base + 768 + 5]
                     + w3_6 * xb[base + 768 + 6] + w3_7 * xb[base + 768 + 7];
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
        if (tid == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
