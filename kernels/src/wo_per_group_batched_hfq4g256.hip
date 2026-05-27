// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// DeepSeek V4 per-group O-LoRA batched GEMV — HFQ4-G256 weights (MQ4 prerotated).
//
// Block-diagonal HFQ4 GEMV. wo_a has natural per-group structure:
//   wo_a: [G, M, K] HFQ4G256-packed   (each [M, K] block is independent)
//   x:    [B, G, K] FWHT-rotated      (MQ4 contract: prerotated input)
//   y:    [B, G, M]
//
// Per (b, g, r):
//   y[b, g, r] = Σ_k dequant(W[g, r, k]) · x[b, g, k]
//
// Single launch in place of `B × G` separate gemv_mq4g256_prerotated
// calls. This collapses ~11k dispatch calls/chunk (B=32, G=8, 43 layers)
// down to 43.
//
// HFQ4G256 layout (per 256-element group):
//   bytes 0..3   = scale  (F32)
//   bytes 4..7   = zero   (F32)
//   bytes 8..135 = 32 × u32 of packed 4-bit weights (8 weights per u32)
// Total: 136 bytes per group.
//
// Grid:  [M, B, G]
// Block: [32, 1, 1]
//
// DeepSeek V4 typical: G=8, M=o_lora_rank=1024, K=per_group_in=4096, B≤max_batch.

__launch_bounds__(32, 20)
extern "C" __global__ void wo_per_group_batched_hfq4g256(
    const char*  __restrict__ wo_a,   // [G * M * K / 256 * 136] bytes
    const float* __restrict__ x_in,   // [B, G, K]
    float*       __restrict__ y_out,  // [B, G, M]
    int G,
    int M,
    int K,
    int batch_size
) {
    const int r = blockIdx.x;     // output row within group
    const int b = blockIdx.y;     // batch index
    const int g = blockIdx.z;     // group index
    if (r >= M || b >= batch_size || g >= G) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long per_group_bytes = (long long)M * groups_per_row * 136;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr = wo_a + (long long)g * per_group_bytes + (long long)r * row_stride;
    const float* x_row = x_in + ((size_t)b * G + g) * K;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int gp = q << 2;
        const int base = gp * 256 + tid * 8;
        const float x00 = x_row[base + 0],   x01 = x_row[base + 1],
                    x02 = x_row[base + 2],   x03 = x_row[base + 3],
                    x04 = x_row[base + 4],   x05 = x_row[base + 5],
                    x06 = x_row[base + 6],   x07 = x_row[base + 7];
        const float x10 = x_row[base + 256 + 0], x11 = x_row[base + 256 + 1],
                    x12 = x_row[base + 256 + 2], x13 = x_row[base + 256 + 3],
                    x14 = x_row[base + 256 + 4], x15 = x_row[base + 256 + 5],
                    x16 = x_row[base + 256 + 6], x17 = x_row[base + 256 + 7];
        const float x20 = x_row[base + 512 + 0], x21 = x_row[base + 512 + 1],
                    x22 = x_row[base + 512 + 2], x23 = x_row[base + 512 + 3],
                    x24 = x_row[base + 512 + 4], x25 = x_row[base + 512 + 5],
                    x26 = x_row[base + 512 + 6], x27 = x_row[base + 512 + 7];
        const float x30 = x_row[base + 768 + 0], x31 = x_row[base + 768 + 1],
                    x32 = x_row[base + 768 + 2], x33 = x_row[base + 768 + 3],
                    x34 = x_row[base + 768 + 4], x35 = x_row[base + 768 + 5],
                    x36 = x_row[base + 768 + 6], x37 = x_row[base + 768 + 7];

        const char* w0 = row_ptr + gp * 136;
        const char* w1 = w0 + 136;
        const char* w2 = w1 + 136;
        const char* w3 = w2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(w0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(w0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(w1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(w1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(w2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(w2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(w3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(w3 + 4));

        unsigned int pk0 = *(const unsigned int*)(w0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(w1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(w2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(w3 + 8 + boff);

        acc0 += (sc0 * (float)((pk0 >>  0) & 0xFu) + zp0) * x00
              + (sc0 * (float)((pk0 >>  4) & 0xFu) + zp0) * x01
              + (sc0 * (float)((pk0 >>  8) & 0xFu) + zp0) * x02
              + (sc0 * (float)((pk0 >> 12) & 0xFu) + zp0) * x03
              + (sc0 * (float)((pk0 >> 16) & 0xFu) + zp0) * x04
              + (sc0 * (float)((pk0 >> 20) & 0xFu) + zp0) * x05
              + (sc0 * (float)((pk0 >> 24) & 0xFu) + zp0) * x06
              + (sc0 * (float)((pk0 >> 28) & 0xFu) + zp0) * x07;
        acc1 += (sc1 * (float)((pk1 >>  0) & 0xFu) + zp1) * x10
              + (sc1 * (float)((pk1 >>  4) & 0xFu) + zp1) * x11
              + (sc1 * (float)((pk1 >>  8) & 0xFu) + zp1) * x12
              + (sc1 * (float)((pk1 >> 12) & 0xFu) + zp1) * x13
              + (sc1 * (float)((pk1 >> 16) & 0xFu) + zp1) * x14
              + (sc1 * (float)((pk1 >> 20) & 0xFu) + zp1) * x15
              + (sc1 * (float)((pk1 >> 24) & 0xFu) + zp1) * x16
              + (sc1 * (float)((pk1 >> 28) & 0xFu) + zp1) * x17;
        acc2 += (sc2 * (float)((pk2 >>  0) & 0xFu) + zp2) * x20
              + (sc2 * (float)((pk2 >>  4) & 0xFu) + zp2) * x21
              + (sc2 * (float)((pk2 >>  8) & 0xFu) + zp2) * x22
              + (sc2 * (float)((pk2 >> 12) & 0xFu) + zp2) * x23
              + (sc2 * (float)((pk2 >> 16) & 0xFu) + zp2) * x24
              + (sc2 * (float)((pk2 >> 20) & 0xFu) + zp2) * x25
              + (sc2 * (float)((pk2 >> 24) & 0xFu) + zp2) * x26
              + (sc2 * (float)((pk2 >> 28) & 0xFu) + zp2) * x27;
        acc3 += (sc3 * (float)((pk3 >>  0) & 0xFu) + zp3) * x30
              + (sc3 * (float)((pk3 >>  4) & 0xFu) + zp3) * x31
              + (sc3 * (float)((pk3 >>  8) & 0xFu) + zp3) * x32
              + (sc3 * (float)((pk3 >> 12) & 0xFu) + zp3) * x33
              + (sc3 * (float)((pk3 >> 16) & 0xFu) + zp3) * x34
              + (sc3 * (float)((pk3 >> 20) & 0xFu) + zp3) * x35
              + (sc3 * (float)((pk3 >> 24) & 0xFu) + zp3) * x36
              + (sc3 * (float)((pk3 >> 28) & 0xFu) + zp3) * x37;
    }

    // Tail groups (1..3 leftover, must accumulate into matching slot).
    if (tail >= 1) {
        const int gp = quads << 2;
        const int base = gp * 256 + tid * 8;
        const char* w = row_ptr + gp * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(w));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(w + 4));
        unsigned int pk = *(const unsigned int*)(w + 8 + boff);
        acc0 += (sc * (float)((pk >>  0) & 0xFu) + zp) * x_row[base + 0]
              + (sc * (float)((pk >>  4) & 0xFu) + zp) * x_row[base + 1]
              + (sc * (float)((pk >>  8) & 0xFu) + zp) * x_row[base + 2]
              + (sc * (float)((pk >> 12) & 0xFu) + zp) * x_row[base + 3]
              + (sc * (float)((pk >> 16) & 0xFu) + zp) * x_row[base + 4]
              + (sc * (float)((pk >> 20) & 0xFu) + zp) * x_row[base + 5]
              + (sc * (float)((pk >> 24) & 0xFu) + zp) * x_row[base + 6]
              + (sc * (float)((pk >> 28) & 0xFu) + zp) * x_row[base + 7];
    }
    if (tail >= 2) {
        const int gp = (quads << 2) + 1;
        const int base = gp * 256 + tid * 8;
        const char* w = row_ptr + gp * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(w));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(w + 4));
        unsigned int pk = *(const unsigned int*)(w + 8 + boff);
        acc1 += (sc * (float)((pk >>  0) & 0xFu) + zp) * x_row[base + 0]
              + (sc * (float)((pk >>  4) & 0xFu) + zp) * x_row[base + 1]
              + (sc * (float)((pk >>  8) & 0xFu) + zp) * x_row[base + 2]
              + (sc * (float)((pk >> 12) & 0xFu) + zp) * x_row[base + 3]
              + (sc * (float)((pk >> 16) & 0xFu) + zp) * x_row[base + 4]
              + (sc * (float)((pk >> 20) & 0xFu) + zp) * x_row[base + 5]
              + (sc * (float)((pk >> 24) & 0xFu) + zp) * x_row[base + 6]
              + (sc * (float)((pk >> 28) & 0xFu) + zp) * x_row[base + 7];
    }
    if (tail >= 3) {
        const int gp = (quads << 2) + 2;
        const int base = gp * 256 + tid * 8;
        const char* w = row_ptr + gp * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(w));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(w + 4));
        unsigned int pk = *(const unsigned int*)(w + 8 + boff);
        acc2 += (sc * (float)((pk >>  0) & 0xFu) + zp) * x_row[base + 0]
              + (sc * (float)((pk >>  4) & 0xFu) + zp) * x_row[base + 1]
              + (sc * (float)((pk >>  8) & 0xFu) + zp) * x_row[base + 2]
              + (sc * (float)((pk >> 12) & 0xFu) + zp) * x_row[base + 3]
              + (sc * (float)((pk >> 16) & 0xFu) + zp) * x_row[base + 4]
              + (sc * (float)((pk >> 20) & 0xFu) + zp) * x_row[base + 5]
              + (sc * (float)((pk >> 24) & 0xFu) + zp) * x_row[base + 6]
              + (sc * (float)((pk >> 28) & 0xFu) + zp) * x_row[base + 7];
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (tid == 0) {
        y_out[((size_t)b * G + g) * M + r] = acc;
    }
}
