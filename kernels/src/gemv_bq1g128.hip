// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// BQ1G128 decode GEMV: 18-byte blocks of 128 binary weights.
// Layout per block: [0..2) fp16 scale d, [2..18) 16 code bytes, 8 codes each.
// Lane t owns nibble t of the code array (byte t>>1, low/high half by t&1), so
// two adjacent lanes share a byte and a wave covers one block.
//
// Do not read the GiB/s-of-weight-bytes figure as headroom here: a block ships
// 128 weights in 16 bytes but still pulls 128 fp32 of x (512 bytes), so this
// kernel is x-load and ALU bound. The sign is applied by XOR-ing straight into
// the float's sign bit, which avoids the per-weight select the branchy form
// generated -- that select, not the weight traffic, is why binary was slower
// than ternary in wall-clock at equal M,K despite reading half the bytes.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bq1g128(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 18;
    const char* row_ptr = A + (long long)row * row_bytes;

    const int byte_off = 2 + (tid >> 1);
    const int sh = (tid & 1) * 4;

    // bit==1 -> +v, bit==0 -> -v, with no branch: flip the sign bit when the
    // code bit is clear. ~b puts "is clear" in the low position for free.
    #define BQ1_SGN(nb, i, v) \
        __uint_as_float(__float_as_uint(v) ^ ((((nb) >> (i)) & 1u) << 31))

    #define BQ1_BLOCK_DOT(gp, xoff, acc)                                     \
        do {                                                                 \
            const float d = (float)(*(const _Float16*)(gp));                 \
            const unsigned int b = *((const unsigned char*)((gp) + byte_off));\
            const unsigned int nb = (~b) >> sh;                              \
            const float4 xv = *(const float4*)((x) + (xoff));                \
            (acc) += d * ( BQ1_SGN(nb, 0, xv.x) + BQ1_SGN(nb, 1, xv.y)       \
                         + BQ1_SGN(nb, 2, xv.z) + BQ1_SGN(nb, 3, xv.w) );    \
        } while (0)

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int g = 0;
    const int xbase = tid * 4;
    for (; g + 4 <= groups_per_row; g += 4) {
        const char* gp0 = row_ptr + g * 18;
        BQ1_BLOCK_DOT(gp0,          g * 128 + xbase,       acc0);
        BQ1_BLOCK_DOT(gp0 + 18,     (g + 1) * 128 + xbase, acc1);
        BQ1_BLOCK_DOT(gp0 + 36,     (g + 2) * 128 + xbase, acc2);
        BQ1_BLOCK_DOT(gp0 + 54,     (g + 3) * 128 + xbase, acc3);
    }
    for (; g < groups_per_row; g++) {
        BQ1_BLOCK_DOT(row_ptr + g * 18, g * 128 + xbase, acc0);
    }
    #undef BQ1_BLOCK_DOT
    #undef BQ1_SGN

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}

// ── x-batched BQ1-G128 ────────────────────────────────────────────────────
//
// Binary sibling of the TQ2-G128 x-batch kernel: y[b][row] = A[row] . x[b]
// for b in 0..B, reading each weight row ONCE rather than once per b. See
// gemv_tq2g128.hip for why that reuse is what makes prefill a real GEMM.
#if defined(HIPFIRE_BQ1G128_XBATCH)

#ifndef HIPFIRE_BQ1G128_XBATCH_MAX
#define HIPFIRE_BQ1G128_XBATCH_MAX 4
#endif

__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_BQ1G128_XBATCH_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 18;
    const int xbase = tid * 4;
    const int byte_off = 2 + (tid >> 1);
    const int sh = (tid & 1) * 4;

    float acc0[HIPFIRE_BQ1G128_XBATCH_MAX];
    float acc1[HIPFIRE_BQ1G128_XBATCH_MAX];
    float acc2[HIPFIRE_BQ1G128_XBATCH_MAX];
    float acc3[HIPFIRE_BQ1G128_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_BQ1G128_XBATCH_MAX; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // Same branchless sign as the scalar kernel: flip the float's sign bit
    // when the code bit is clear.
    #define BQ1_XSGN(nb, i, v) \
        __uint_as_float(__float_as_uint(v) ^ ((((nb) >> (i)) & 1u) << 31))

    #define BQ1_XDOG(nb, d, xoff, a)                                        \
        do {                                                                \
            const float4 xv = *(const float4*)(xb + (xoff));                \
            (a) += (d) * ( BQ1_XSGN(nb, 0, xv.x) + BQ1_XSGN(nb, 1, xv.y)    \
                         + BQ1_XSGN(nb, 2, xv.z) + BQ1_XSGN(nb, 3, xv.w) ); \
        } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 18;
        const char* gp1 = gp0 + 18;
        const char* gp2 = gp1 + 18;
        const char* gp3 = gp2 + 18;
        const float d0 = (float)(*(const _Float16*)gp0);
        const float d1 = (float)(*(const _Float16*)gp1);
        const float d2 = (float)(*(const _Float16*)gp2);
        const float d3 = (float)(*(const _Float16*)gp3);
        const unsigned int n0 = (~(unsigned int)*((const unsigned char*)(gp0 + byte_off))) >> sh;
        const unsigned int n1 = (~(unsigned int)*((const unsigned char*)(gp1 + byte_off))) >> sh;
        const unsigned int n2 = (~(unsigned int)*((const unsigned char*)(gp2 + byte_off))) >> sh;
        const unsigned int n3 = (~(unsigned int)*((const unsigned char*)(gp3 + byte_off))) >> sh;
        const int base = g * 128 + xbase;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_BQ1G128_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            BQ1_XDOG(n0, d0, base,       acc0[b]);
            BQ1_XDOG(n1, d1, base + 128, acc1[b]);
            BQ1_XDOG(n2, d2, base + 256, acc2[b]);
            BQ1_XDOG(n3, d3, base + 384, acc3[b]);
        }
    }

    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + g * 18;
        const float d = (float)(*(const _Float16*)gp);
        const unsigned int nb = (~(unsigned int)*((const unsigned char*)(gp + byte_off))) >> sh;
        const int base = g * 128 + xbase;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_BQ1G128_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            if (t == 0)      { BQ1_XDOG(nb, d, base, acc0[b]); }
            else if (t == 1) { BQ1_XDOG(nb, d, base, acc1[b]); }
            else             { BQ1_XDOG(nb, d, base, acc2[b]); }
        }
    }
    #undef BQ1_XDOG
    #undef BQ1_XSGN

    #pragma unroll
    for (int b = 0; b < HIPFIRE_BQ1G128_XBATCH_MAX; b++) {
        if (b >= B) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) y[(long long)b * M + row] = acc;
    }
}

#endif  // HIPFIRE_BQ1G128_XBATCH
