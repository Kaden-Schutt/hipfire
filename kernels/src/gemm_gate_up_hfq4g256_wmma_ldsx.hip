// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Gate 0 ISA prototype — LDS-staged X variant of gate_up.
//
// PURPOSE: ISA inspection only. Not for benchmarking, not for correctness
// testing, not for production. The kernel below should be compilable on
// gfx1100 via:
//   hipcc --genco -O3 --offload-arch=gfx1100 -save-temps -S \
//         gate_up_ldsx_proto.hip -o gate_up_ldsx_proto.hsaco
//
// What we want to learn from the ISA dump:
//   1. Does the compiler keep weight loads (the per-row A scale/zp/nibble
//      reads) at the TOP of the inner loop, or does the s_barrier push
//      them below it? If pushed below, the existing software-pipelining
//      structure is destroyed and the variant is dead.
//   2. Does the cooperative global→LDS path emit a recognizable
//      ds_write pattern, or does it serialize through a register
//      round-trip with an effective vmcnt stall on the LDS write path?
//   3. Does the WMMA path show ds_read followed closely by v_wmma, or is
//      there an lgkmcnt stall between them that offsets the saved vmcnt?
//   4. What VGPR count does the compiler report? Baseline is 80.
//
// Design choices vs. v1 (per the post-review v2 plan):
//   - Per-K-tile-pair LDS slab (1 KB per stage, 2 KB double-buffered)
//     to preserve occupancy.
//   - Cooperative load partition that splits the (16 batches × 32 K)
//     tile evenly across 32 lanes — one half16_t per lane, no wave
//     redundancy across the high/low lane halves.
//   - Layout [stage][n=batch][k] which costs ~16 cycles LDS service
//     (per the GLM bank analysis) but matches the dequant-then-WMMA
//     pattern without extra indexing math.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma_ldsx(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    // Per-K-tile-pair, double-buffered.
    // Layout: [stage][n=batch col 0..15][k=0..31]. 2 × 16 × 32 × 2 = 2048 bytes.
    __shared__ _Float16 lds_x[2][16][32];

    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // Per-row routing (unchanged from baseline)
    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);
    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate; local_row = safe_row;
    } else {
        A = A_up; local_row = safe_row - gate_m;
    }
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    // Cooperative-load mapping: lane L loads
    //   batch_n      = L >> 1                 (lanes {2L, 2L+1} both touch batch L)
    //   k_half       = L & 1                  (lane mod 2 picks low / high half of pair)
    //   load address = X[batch_start + (L>>1), g*256 + kt*16 + (L&1)*16 .. +15]
    const int load_batch_idx  = batch_start + (tid >> 1);
    const int load_batch_safe = (load_batch_idx < N) ? load_batch_idx : 0;
    const int load_k_half     = (tid & 1) * 16;

    float8_t acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = X + (long long)load_batch_safe * K + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int stage = (kt >> 1) & 1;

            // ===== Cooperative global → LDS for this K-tile-pair =====
            // Each lane loads its own half16_t (32 bytes) of unique X data.
            // 32 lanes × 32 bytes = 1024 bytes = full per-pair X tile.
            half16_t x_my = *(const half16_t*)(xg + kt * 16 + load_k_half);
            #pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                lds_x[stage][tid >> 1][load_k_half + kk] = x_my[kk];
            }
            __syncthreads();

            // ===== Dequant A weights for both K-tiles in the pair =====
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            // ===== Read X from LDS for B operands =====
            // Lane L reads its batch column (L & 15) for both K-tiles.
            const int wmma_n = tid & 15;
            half16_t b_a;
            half16_t b_b;
            #pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                b_a[kk] = lds_x[stage][wmma_n][kk];
                b_b[kk] = lds_x[stage][wmma_n][kk + 16];
            }

            // ===== Dequant + WMMA A =====
            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // ===== Dequant + WMMA B =====
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // ===== Output (unchanged from baseline) =====
    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y; int proj_row; int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate; proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;   proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
