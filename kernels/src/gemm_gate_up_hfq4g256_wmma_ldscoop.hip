// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// LDSCOOP variant: cooperative LDS weight staging for better DRAM
// coalescing on gfx1100 RDNA3 wave32 WMMA.
//
// Why: the base gemm_gate_up_hfq4g256_wmma kernel has each thread load
// weights from ITS OWN row independently (different rows = different
// cache lines for the same warp instruction). Per warp, that's 16
// scattered cache-line loads = poor coalescing. Observed at M=27648
// N=3: 305 GB/s = 32% of 960 GB/s peak BW.
//
// Fix: all 32 threads cooperatively load ONE row's weights at a time
// into LDS, with coalesced loads (32 threads × 4 bytes = 128 bytes /
// cache line per warp instruction). Cycle through all 16 rows. WMMA
// loop then reads weights from LDS (fast on-chip).
//
// Per group LDS layout: w_lds[16][136] = 2176 bytes
// Per group load pattern: 16 outer iter × 32 thread × 4 bytes/uint
//   = 128 bytes per outer iter (1-2 cache lines coalesced).
//   Last 8 bytes (offset 128..135) covered by tid < 2 in second sub-iter.
//
// Same WMMA C-mapping as k2/k4 siblings (canonical wave32 layout).
// Byte-exact-up-to-fp-rounding (different load order may shift FP
// accumulation by a few ULPs; argmax-stable).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma_ldscoop(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route MY ROW to gate or up projection.
    const char* A_my;
    int local_my_row;
    if (safe_row < gate_m) { A_my = A_gate; local_my_row = safe_row; }
    else                   { A_my = A_up;   local_my_row = safe_row - gate_m; }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;

    // Routes for ALL 16 rows in this block — needed for cooperative loads
    // since each load iteration pulls a different row's bytes.
    // For row r in 0..15: src_base[r] = (A_for_row_r) + local_row_r * row_stride
    // We compute this lazily inside the load loop per row r.

    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;

    // LDS staging: 16 rows × 136 bytes = 2176 bytes per group.
    __shared__ char w_lds[16 * 136];

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        // ── Cooperative load: 16 rows × 136 bytes into LDS ──
        // Per row: 32 threads × 4 bytes = 128 bytes coalesced (1-2 cache
        // lines). Need 2 such warp loads to cover 136 bytes: first 128B
        // by all 32 threads, then 8B by tid < 2.
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            const int global_row = row_start + r;
            const int safe_global = (global_row < total_m) ? global_row : (total_m - 1);
            const char* Ar;
            int local_r;
            if (safe_global < gate_m) { Ar = A_gate; local_r = safe_global; }
            else                       { Ar = A_up;   local_r = safe_global - gate_m; }
            const char* src = Ar + local_r * row_stride + g * 136;
            char* dst = &w_lds[r * 136];

            // Coalesced load A: all 32 threads load bytes [tid*4 .. tid*4+4)
            // covering offsets 0..127. 32 × 4 = 128 bytes coalesced.
            *((unsigned int*)(dst + tid * 4)) =
                *((const unsigned int*)(src + tid * 4));

            // Tail bytes 128..135 (8 bytes total = 2 uints).
            // tid 0 loads offset 128..131, tid 1 loads 132..135.
            if (tid < 2) {
                *((unsigned int*)(dst + 128 + tid * 4)) =
                    *((const unsigned int*)(src + 128 + tid * 4));
            }
        }
        __syncthreads();

        // Now read MY row's weights from LDS.
        const char* gp = &w_lds[(tid & 15) * 136];
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_g = x_base + g * 256;

        // K-tile loop: 16 K-tiles per group, k2 software pipeline mirrors
        // the base wmma kernel (load-ahead + dequant + WMMA).
        #pragma unroll
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b_a = *(const half16_t*)(x_g + k_off_a);
            half16_t b_b = *(const half16_t*)(x_g + k_off_b);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }

        __syncthreads();  // Before next group overwrites LDS
    }

    // --- Output (same as base k2) ---
    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;          out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m; out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
