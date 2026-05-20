// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// K-tiled batched HFQ4-G256 GEMM with fused residual add.
//
// Processes ROWS_PER_BLOCK=4 rows per block (4 waves of 32 threads each).
// For each weight-group quad, the X activation tile is loaded cooperatively
// into LDS by all 128 threads, then each wave reads X from LDS while loading
// its own row's weight data from VRAM. This shares X across rows via LDS
// instead of each row re-reading from L2 independently.
//
// Grid: [ceil(M/ROWS_PER_BLOCK), ceil(batch_size/BATCH_TILE), 1]
// Block: [128, 1, 1]  (4 waves)
// LDS: BATCH_TILE * 4 * 256 * 4 = 32 KB (one quad of X for all batch elements)

#define BATCH_TILE 8
#define ROWS_PER_BLOCK 4

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_hfq4g256_residual_kt(
    const char* __restrict__ A,
    const float* __restrict__ X,   // [batch_size, K]
    float* __restrict__ Y,         // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const int wave_id = threadIdx.x >> 5;   // 0..3
    const int lane = threadIdx.x & 31;      // 0..31
    const int my_row = row_base + wave_id;

    const int groups_per_row = K / 256;
    const int boff = lane * 4;  // byte offset in nibble array

    // Each wave accumulates for its own row
    float acc0[BATCH_TILE], acc1[BATCH_TILE], acc2[BATCH_TILE], acc3[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    // LDS: X tile for one quad (4 groups × 256 elements × BATCH_TILE batch)
    // Layout: lds_x[b][group_in_quad * 256 + k] where b=0..BATCH_TILE-1
    // Total: BATCH_TILE * 1024 floats = 32 KB
    __shared__ float lds_x[BATCH_TILE * 1024];

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base_k = g * 256;

        // ── Cooperative X load: all 128 threads load X tile into LDS ──
        // Total elements: BATCH_TILE * 1024 = 8192 floats
        for (int i = threadIdx.x; i < local_bs * 1024; i += 128) {
            int b = i / 1024;
            int k_off = i % 1024;
            lds_x[b * 1024 + k_off] = X[(long long)(batch_start + b) * K + base_k + k_off];
        }
        __syncthreads();

        // ── Each wave loads its row's weights from VRAM ──
        if (my_row < M) {
            const char* row_ptr = A + (long long)my_row * groups_per_row * 136;
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

            float w0_0 = sc0*(float)((pk0    )&0xFu)+zp0, w0_1 = sc0*(float)((pk0>> 4)&0xFu)+zp0;
            float w0_2 = sc0*(float)((pk0>> 8)&0xFu)+zp0, w0_3 = sc0*(float)((pk0>>12)&0xFu)+zp0;
            float w0_4 = sc0*(float)((pk0>>16)&0xFu)+zp0, w0_5 = sc0*(float)((pk0>>20)&0xFu)+zp0;
            float w0_6 = sc0*(float)((pk0>>24)&0xFu)+zp0, w0_7 = sc0*(float)((pk0>>28)&0xFu)+zp0;

            float w1_0 = sc1*(float)((pk1    )&0xFu)+zp1, w1_1 = sc1*(float)((pk1>> 4)&0xFu)+zp1;
            float w1_2 = sc1*(float)((pk1>> 8)&0xFu)+zp1, w1_3 = sc1*(float)((pk1>>12)&0xFu)+zp1;
            float w1_4 = sc1*(float)((pk1>>16)&0xFu)+zp1, w1_5 = sc1*(float)((pk1>>20)&0xFu)+zp1;
            float w1_6 = sc1*(float)((pk1>>24)&0xFu)+zp1, w1_7 = sc1*(float)((pk1>>28)&0xFu)+zp1;

            float w2_0 = sc2*(float)((pk2    )&0xFu)+zp2, w2_1 = sc2*(float)((pk2>> 4)&0xFu)+zp2;
            float w2_2 = sc2*(float)((pk2>> 8)&0xFu)+zp2, w2_3 = sc2*(float)((pk2>>12)&0xFu)+zp2;
            float w2_4 = sc2*(float)((pk2>>16)&0xFu)+zp2, w2_5 = sc2*(float)((pk2>>20)&0xFu)+zp2;
            float w2_6 = sc2*(float)((pk2>>24)&0xFu)+zp2, w2_7 = sc2*(float)((pk2>>28)&0xFu)+zp2;

            float w3_0 = sc3*(float)((pk3    )&0xFu)+zp3, w3_1 = sc3*(float)((pk3>> 4)&0xFu)+zp3;
            float w3_2 = sc3*(float)((pk3>> 8)&0xFu)+zp3, w3_3 = sc3*(float)((pk3>>12)&0xFu)+zp3;
            float w3_4 = sc3*(float)((pk3>>16)&0xFu)+zp3, w3_5 = sc3*(float)((pk3>>20)&0xFu)+zp3;
            float w3_6 = sc3*(float)((pk3>>24)&0xFu)+zp3, w3_7 = sc3*(float)((pk3>>28)&0xFu)+zp3;

            const int k_off = lane * 8;  // offset within the 1024-element quad

            #pragma unroll
            for (int b = 0; b < BATCH_TILE; b++) {
                if (b >= local_bs) break;
                const float* xb = lds_x + b * 1024 + k_off;

                acc0[b] += w0_0*xb[0] + w0_1*xb[1] + w0_2*xb[2] + w0_3*xb[3]
                         + w0_4*xb[4] + w0_5*xb[5] + w0_6*xb[6] + w0_7*xb[7];

                acc1[b] += w1_0*xb[256] + w1_1*xb[257] + w1_2*xb[258] + w1_3*xb[259]
                         + w1_4*xb[260] + w1_5*xb[261] + w1_6*xb[262] + w1_7*xb[263];

                acc2[b] += w2_0*xb[512] + w2_1*xb[513] + w2_2*xb[514] + w2_3*xb[515]
                         + w2_4*xb[516] + w2_5*xb[517] + w2_6*xb[518] + w2_7*xb[519];

                acc3[b] += w3_0*xb[768] + w3_1*xb[769] + w3_2*xb[770] + w3_3*xb[771]
                         + w3_4*xb[772] + w3_5*xb[773] + w3_6*xb[774] + w3_7*xb[775];
            }
        }
        __syncthreads();
    }

    // Tail groups: fall back to direct L2 reads (at most 3 groups, not worth LDS)
    if (my_row < M && tail > 0) {
        const char* row_ptr = A + (long long)my_row * groups_per_row * 136;

        #define TAIL_BODY(acc_arr)                                                  \
            {                                                                        \
                const char* gp = row_ptr + g * 136;                                  \
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));    \
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);             \
                float tw0 = sc*(float)((pk    )&0xFu)+zp, tw1 = sc*(float)((pk>> 4)&0xFu)+zp; \
                float tw2 = sc*(float)((pk>> 8)&0xFu)+zp, tw3 = sc*(float)((pk>>12)&0xFu)+zp; \
                float tw4 = sc*(float)((pk>>16)&0xFu)+zp, tw5 = sc*(float)((pk>>20)&0xFu)+zp; \
                float tw6 = sc*(float)((pk>>24)&0xFu)+zp, tw7 = sc*(float)((pk>>28)&0xFu)+zp; \
                const int base = g * 256 + lane * 8;                                 \
                _Pragma("unroll")                                                    \
                for (int b = 0; b < BATCH_TILE; b++) {                               \
                    if (b >= local_bs) break;                                        \
                    const float* xb = X + (long long)(batch_start + b) * K;           \
                    (acc_arr)[b] += tw0*xb[base  ] + tw1*xb[base+1]                   \
                                  + tw2*xb[base+2] + tw3*xb[base+3]                   \
                                  + tw4*xb[base+4] + tw5*xb[base+5]                   \
                                  + tw6*xb[base+6] + tw7*xb[base+7];                  \
                }                                                                    \
            }

        if (tail >= 1) { const int g = quads << 2; TAIL_BODY(acc0); }
        if (tail >= 2) { const int g = (quads << 2) + 1; TAIL_BODY(acc1); }
        if (tail >= 3) { const int g = (quads << 2) + 2; TAIL_BODY(acc2); }
        #undef TAIL_BODY
    }

    // Intra-wave reduction + write (only for valid rows)
    if (my_row < M && local_bs > 0) {
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down(val, offset);
            if (lane == 0) Y[(long long)(batch_start + b) * M + my_row] += val;
        }
    }
}
