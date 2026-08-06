// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused HFQ4-G256 GEMM for the Qwen3.5 FFN
// preamble (gate + up projections).  gfx1100+ only (RDNA3 wave32 WMMA).
//
// K2 variant: 2x K-tile unroll with software pipelining.
// Processes 2 K-tiles (32 K-elements) per loop body. Loads weights+X for
// tile B while WMMA on tile A is executing, enabling the compiler to
// overlap loads with compute (target: s_waitcnt vmcnt(2)).
//
// Same structure as gemm_gate_up_hfq4g256 but uses WMMA 16x16x16 tiles
// instead of scalar FMA.  X is pre-converted to FP16.
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)].  Block: [32].  LDS: 0.
//
// Each 16-row tile may straddle the gate/up boundary.  Per-thread output
// routing resolves which Y pointer (Y_gate or Y_up) each row writes to.
// Overwrite semantics (y = acc, not +=).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,        // [N x K], FP16
    float*         __restrict__ Y_gate,    // [N x gate_m]
    float*         __restrict__ Y_up,      // [N x up_m]
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row = row_start + (tid & 15);

    // Clamp row into valid range for weight loads (OOB threads will be
    // masked at output time).  Route to gate or up projection.
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);
    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate;
        local_row = safe_row;
    } else {
        A = A_up;
        local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    // Clamp batch index for the B-matrix (X) load.
    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            // Load both X tiles early (consecutive addresses — prefetcher-friendly)
            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            // Dequant tile A
            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            // WMMA A — b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // Dequant tile B (reuse a_reg register)
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            // WMMA B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // --- Output ---
    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // So row = 2*j + (tid>>4), col (batch) = tid & 15.
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                // Route this row to the correct output matrix.
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;              out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m;     out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}

// Experimental cold-weight LDS staging variant for fused gate+up on gfx11
// (RDNA3/3.5). Sister of gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage: same
// slab/wave structure and gate/up row routing; gfx11 half16_t operands +
// interleaved C-map + _w32 builtin. Opt-in via HIPFIRE_HFQ4G256_LDSSTAGE=1;
// K%512==0. OVERWRITE semantics (y = sum, not +=).
typedef unsigned int __attribute__((ext_vector_type(4))) uint4v_t;

__launch_bounds__(256, 4)
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma_ldsstage(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= total_m || batch_start >= N) return;

    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int safe_batch = (batch_start + m_lane < N)
        ? (batch_start + m_lane) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const int slabs = K / 512;

    __shared__ __align__(16) unsigned char staged_weights[16][272];
    __shared__ float partials[8 * 32 * 8];

    const int load_row = tid / 17;
    const int load_vec = tid - load_row * 17;
    const int safe_load_row = (row_start + load_row < total_m)
        ? (row_start + load_row) : (total_m - 1);
    const char* A_load;
    int local_load_row;
    if (safe_load_row < gate_m) {
        A_load = A_gate;  local_load_row = safe_load_row;
    } else {
        A_load = A_up;    local_load_row = safe_load_row - gate_m;
    }

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int slab = 0; slab < slabs; ++slab) {
        const char* src0 = A_load + (long long)local_load_row * row_stride
            + slab * 272 + load_vec * 16;
        unsigned char* dst0 = staged_weights[load_row] + load_vec * 16;
        *(uint4v_t*)dst0 = *(const uint4v_t*)src0;

        if (tid < 16) {
            const int vec = tid + 1;
            const int safe_row15 = (row_start + 15 < total_m)
                ? (row_start + 15) : (total_m - 1);
            const char* A15;
            int local_row15;
            if (safe_row15 < gate_m) {
                A15 = A_gate;  local_row15 = safe_row15;
            } else {
                A15 = A_up;    local_row15 = safe_row15 - gate_m;
            }
            const char* src1 = A15 + (long long)local_row15 * row_stride
                + slab * 272 + vec * 16;
            unsigned char* dst1 = staged_weights[15] + vec * 16;
            *(uint4v_t*)dst1 = *(const uint4v_t*)src1;
        }
        __syncthreads();

        const int group_in_slab = wave >> 2;
        const int quarter_in_group = wave & 3;
        const unsigned char* gp = staged_weights[m_lane] + group_in_slab * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + slab * 512 + wave * 64;
        const int packed_base = quarter_in_group * 32;

        // gfx11 WMMA: half16_t A/B (full 16-K per lane). Four 16-K tiles
        // cover the wave's 64-wide K slice of the 512-wide slab.
        #define DQ16_LDS(pk0, pk1) \
            a_reg[0]  = sc_h * (_Float16)(float)(((pk0) >>  0) & 0xFu) + zp_h; \
            a_reg[1]  = sc_h * (_Float16)(float)(((pk0) >>  4) & 0xFu) + zp_h; \
            a_reg[2]  = sc_h * (_Float16)(float)(((pk0) >>  8) & 0xFu) + zp_h; \
            a_reg[3]  = sc_h * (_Float16)(float)(((pk0) >> 12) & 0xFu) + zp_h; \
            a_reg[4]  = sc_h * (_Float16)(float)(((pk0) >> 16) & 0xFu) + zp_h; \
            a_reg[5]  = sc_h * (_Float16)(float)(((pk0) >> 20) & 0xFu) + zp_h; \
            a_reg[6]  = sc_h * (_Float16)(float)(((pk0) >> 24) & 0xFu) + zp_h; \
            a_reg[7]  = sc_h * (_Float16)(float)(((pk0) >> 28) & 0xFu) + zp_h; \
            a_reg[8]  = sc_h * (_Float16)(float)(((pk1) >>  0) & 0xFu) + zp_h; \
            a_reg[9]  = sc_h * (_Float16)(float)(((pk1) >>  4) & 0xFu) + zp_h; \
            a_reg[10] = sc_h * (_Float16)(float)(((pk1) >>  8) & 0xFu) + zp_h; \
            a_reg[11] = sc_h * (_Float16)(float)(((pk1) >> 12) & 0xFu) + zp_h; \
            a_reg[12] = sc_h * (_Float16)(float)(((pk1) >> 16) & 0xFu) + zp_h; \
            a_reg[13] = sc_h * (_Float16)(float)(((pk1) >> 20) & 0xFu) + zp_h; \
            a_reg[14] = sc_h * (_Float16)(float)(((pk1) >> 24) & 0xFu) + zp_h; \
            a_reg[15] = sc_h * (_Float16)(float)(((pk1) >> 28) & 0xFu) + zp_h

        {
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + packed_base + 0);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + packed_base + 4);
            const half16_t b_reg = *(const half16_t*)(xg + 0);
            half16_t a_reg;
            DQ16_LDS(pk0, pk1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
        {
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + packed_base + 8);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + packed_base + 12);
            const half16_t b_reg = *(const half16_t*)(xg + 16);
            half16_t a_reg;
            DQ16_LDS(pk0, pk1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
        {
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + packed_base + 16);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + packed_base + 20);
            const half16_t b_reg = *(const half16_t*)(xg + 32);
            half16_t a_reg;
            DQ16_LDS(pk0, pk1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
        {
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + packed_base + 24);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + packed_base + 28);
            const half16_t b_reg = *(const half16_t*)(xg + 48);
            half16_t a_reg;
            DQ16_LDS(pk0, pk1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
        #undef DQ16_LDS

        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        partials[(wave * 32 + lane) * 8 + j] = acc[j];
    }
    __syncthreads();

    // Wave 0 reduction with gate_up OVERWRITE semantics (y = sum, not +=).
    // gfx11 C-map: acc[j] = C[2*j + (lane>>4)][lane & 15].
    if (wave == 0) {
        const int out_col = batch_start + m_lane;
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sum = partials[(0 * 32 + lane) * 8 + j];
                #pragma unroll
                for (int w = 1; w < 8; ++w) {
                    sum += partials[(w * 32 + lane) * 8 + j];
                }
                const int out_row = row_start + 2 * j + k_grp;
                if (out_row < total_m) {
                    float* Y;
                    int proj_row;
                    int out_stride;
                    if (out_row < gate_m) {
                        Y = Y_gate;  proj_row = out_row;           out_stride = gate_m;
                    } else {
                        Y = Y_up;    proj_row = out_row - gate_m;  out_stride = up_m;
                    }
                    Y[(long long)out_col * out_stride + proj_row] = sum;
                }
            }
        }
    }
}
