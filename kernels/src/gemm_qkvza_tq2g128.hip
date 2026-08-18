// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched 4-way fused TQ2-G128 GEMM for the Qwen3.5 DeltaNet LA preamble.
// Ternary counterpart of gemm_qkvza_hfq4g256:
//
//   Y_qkv[b]   = W_qkv   . X[b]      for b in 0..N
//   Y_z[b]     = W_z     . X[b]
//   Y_beta[b]  = W_beta  . X[b]
//   Y_alpha[b] = W_alpha . X[b]
//
// MEASURED AND NOT WIRED IN -- kept as the record of a rejected design.
//
// The hypothesis was that fusion explained the remaining prefill gap to
// qwen3.6-27b.mq4: mq4's prefill kernel does all four matrices in ONE grid
// (gemm_qkvza_hfq4g256, BATCH_TILE=8) while our path issues four launches.
// Ternary also reads 0.27 B/weight against mq4's 0.53, so fusing looked like
// it should overtake mq4 outright.
//
// It does not, and the head-to-head against mq4's own kernel at IDENTICAL
// m/K/N (bench_qkvza_ternary_vs_mq4) says why. Two separate causes:
//
// 1. FIXED -- loop structure. The first version gave each of the four groups
//    in a quad its own batch loop. mq4 hoists all four groups' loads and
//    decodes above ONE batch loop, so the compiler interleaves them and the
//    bounds check is paid once instead of four times. That alone moved
//    ternary/mq4 from 0.41x to 0.93x at N=32.
//
// 2. OPEN -- load granularity, which is inherent to the format. A TQ2G128
//    group's code array is 32 B, i.e. exactly 1 byte per lane of a 32-lane
//    wave, so the wave can only issue BYTE loads and touches 32 B of a 64 B
//    cache line. HFQ4G256's code array is 128 B = 4 B/lane, so mq4 issues
//    DWORD loads over full lines. Ternary reads half the bytes but needs
//    twice the requests at half the line efficiency.
//
//    The fix is a lane remap: let each lane take 4 code bytes (16 weights) of
//    ONE group so 8 lanes cover a group and a wave covers 4 groups at dword
//    granularity, carrying 4 scales. Not attempted yet.
//
// Current standing at the Bonsai LA shapes (K=5120, m=[6144,4096,128,128]),
// ternary/mq4 with the same kernel design: 0.93x at N=32, 0.36x at N=128,
// 0.15x at N=512.
//
// Still NOT wired in: even at 0.93x this loses to four gemm_tq2g128_wmma
// launches, because mq4's fused kernel uses no WMMA, no dp4a and no LDS.
//
// Each block handles (1 output row of one of the 4 projections) x (up to
// BATCH_TILE tokens). Grid = (total_m) x ceil(N / BATCH_TILE).
//
// Lane t owns code byte t of a 34-byte block, i.e. weights 4t..4t+3, so a
// wave covers one 128-weight group -- the same split the scalar GEMV uses.
// The 4 codes in a byte are decoded ONCE and reused across the token tile;
// that reuse is the point of batching.
//
// Overwrite semantics (y[b][row] = acc), matching the GEMV reference.

#ifndef HIPFIRE_QKVZA_TQ2_BATCH_TILE
#define HIPFIRE_QKVZA_TQ2_BATCH_TILE 8
#endif
#define BT HIPFIRE_QKVZA_TQ2_BATCH_TILE

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_qkvza_tq2g128(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ X,        // [N x K]
    float*       __restrict__ Y_qkv,    // [N x qkv_m]
    float*       __restrict__ Y_z,      // [N x z_m]
    float*       __restrict__ Y_beta,   // [N x beta_m]
    float*       __restrict__ Y_alpha,  // [N x alpha_m]
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BT;
    int local_bs = N - batch_start;
    if (local_bs > BT) local_bs = BT;
    if (local_bs <= 0) return;

    // Route this workgroup to its target matrix + output row + output stride.
    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < qkv_m) {
        A = A_qkv;   Y = Y_qkv;   row = gid;                          out_stride = qkv_m;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     Y = Y_z;     row = gid - qkv_m;                  out_stride = z_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  Y = Y_beta;  row = gid - (qkv_m + z_m);          out_stride = beta_m;
    } else {
        A = A_alpha; Y = Y_alpha; row = gid - (qkv_m + z_m + beta_m); out_stride = alpha_m;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 34;
    const int xbase = tid * 4;

    // Four accumulator sets over g mod 4, matching the scalar kernel's
    // interleave so the pairwise combine below reproduces its FMA order.
    float acc0[BT], acc1[BT], acc2[BT], acc3[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    // Hoist ALL FOUR groups' loads + decodes above ONE batch loop, exactly as
    // gemm_qkvza_hfq4g256 does. The first version of this kernel gave each
    // group its own batch loop (four per quad); that quadrupled the bounds-
    // check branching and stopped the compiler interleaving the groups, and
    // measured 9x slower than mq4 at identical shapes. Structure is the whole
    // difference -- the format was never the problem.
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 34;
        const char* gp1 = gp0 + 34;
        const char* gp2 = gp1 + 34;
        const char* gp3 = gp2 + 34;

        const float d0 = (float)(*(const _Float16*)gp0);
        const float d1 = (float)(*(const _Float16*)gp1);
        const float d2 = (float)(*(const _Float16*)gp2);
        const float d3 = (float)(*(const _Float16*)gp3);

        const unsigned int c0 = *((const unsigned char*)(gp0 + 2 + tid));
        const unsigned int c1 = *((const unsigned char*)(gp1 + 2 + tid));
        const unsigned int c2 = *((const unsigned char*)(gp2 + 2 + tid));
        const unsigned int c3 = *((const unsigned char*)(gp3 + 2 + tid));

        #define TQ2_W(d, c, sh) ((d) * (float)((int)(((c) >> (sh)) & 0x3) - 1))
        const float w0_0 = TQ2_W(d0, c0, 0), w0_1 = TQ2_W(d0, c0, 2);
        const float w0_2 = TQ2_W(d0, c0, 4), w0_3 = TQ2_W(d0, c0, 6);
        const float w1_0 = TQ2_W(d1, c1, 0), w1_1 = TQ2_W(d1, c1, 2);
        const float w1_2 = TQ2_W(d1, c1, 4), w1_3 = TQ2_W(d1, c1, 6);
        const float w2_0 = TQ2_W(d2, c2, 0), w2_1 = TQ2_W(d2, c2, 2);
        const float w2_2 = TQ2_W(d2, c2, 4), w2_3 = TQ2_W(d2, c2, 6);
        const float w3_0 = TQ2_W(d3, c3, 0), w3_1 = TQ2_W(d3, c3, 2);
        const float w3_2 = TQ2_W(d3, c3, 4), w3_3 = TQ2_W(d3, c3, 6);
        #undef TQ2_W

        const int base = g * 128 + xbase;

        #pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b >= local_bs) break;
            const float* xb = X + (long long)(batch_start + b) * K;
            acc0[b] += w0_0 * xb[base      ] + w0_1 * xb[base       + 1]
                     + w0_2 * xb[base   + 2] + w0_3 * xb[base       + 3];
            acc1[b] += w1_0 * xb[base + 128] + w1_1 * xb[base + 128 + 1]
                     + w1_2 * xb[base + 130] + w1_3 * xb[base + 128 + 3];
            acc2[b] += w2_0 * xb[base + 256] + w2_1 * xb[base + 256 + 1]
                     + w2_2 * xb[base + 258] + w2_3 * xb[base + 256 + 3];
            acc3[b] += w3_0 * xb[base + 384] + w3_1 * xb[base + 384 + 1]
                     + w3_2 * xb[base + 386] + w3_3 * xb[base + 384 + 3];
        }
    }

    // Tail groups accumulate into acc[g % 4], matching the interleave the
    // combine below assumes.
    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + g * 34;
        const float d = (float)(*(const _Float16*)gp);
        const unsigned int c = *((const unsigned char*)(gp + 2 + tid));
        const float tw0 = d * (float)((int)( c       & 0x3) - 1);
        const float tw1 = d * (float)((int)((c >> 2) & 0x3) - 1);
        const float tw2 = d * (float)((int)((c >> 4) & 0x3) - 1);
        const float tw3 = d * (float)((int)((c >> 6) & 0x3) - 1);
        const int base = g * 128 + xbase;
        #pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b >= local_bs) break;
            const float* xb = X + (long long)(batch_start + b) * K;
            const float v = tw0 * xb[base] + tw1 * xb[base + 1]
                          + tw2 * xb[base + 2] + tw3 * xb[base + 3];
            if (t == 0)      { acc0[b] += v; }
            else if (t == 1) { acc1[b] += v; }
            else             { acc2[b] += v; }
        }
    }

    #pragma unroll
    for (int b = 0; b < BT; b++) {
        if (b >= local_bs) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) {
            Y[(long long)(batch_start + b) * out_stride + row] = acc;
        }
    }
}
