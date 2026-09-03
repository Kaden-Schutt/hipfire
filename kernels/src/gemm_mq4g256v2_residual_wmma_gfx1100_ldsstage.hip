// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1100 (RDNA3) port of the gfx12 LDS-staged residual design
// (`gemm_mq4g256v2_residual_wmma_gfx12_ldsstage` in
// gemm_mq4g256v2_residual_wmma.gfx12.hip) for MQ4G256V2 (qt=44), DFlash
// verify tier (N<=16).
//
// Faithful port: identical cooperative slab fill (all 256 threads stage one
// 16-row x 512-K RAW quantized slab, `staged_weights[16][272]` = 4352 B, via
// coalesced dwordx4 loads with `load_row = tid/17`, `load_vec = tid%17`, plus
// a 16-thread tail for row 15), identical LDS layout, identical 8-wave K
// partition (wave = 64-wide K slice of the 512 slab: `group_in_slab =
// wave>>2`, `quarter = wave&3`, header `quarter<2 ? hA : hB`), identical
// `partials[8*32*8]` + wave-0 fixed-order reduce, `__syncthreads` before and
// after consumption. Requires K % 512 == 0 (launcher-enforced).
//
// Per-wave consume loop is gfx11-shaped (the whole port): each wave's 64 K =
// 4 x 16-wide fragments; for each, a `half16_t` a_reg is built from LDS via
// the base kernel's DQ macro (16 nibbles from two u32 loads at
// `gp + 8 + frag*8`, header kt<8 within the group => quarter 0..1 -> s0/z0,
// 2..3 -> s1/z1, same select as gfx12), `half16_t b_reg` is loaded from X,
// and `acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc)`
// (no `_gfx12` suffix; each lane holds a FULL 16-K fragment, kABKLane=1).
// Output uses the gfx11 interleaved C mapping from the base kernel
// (`gemm_mq4g256v2_residual_wmma.hip`): acc[j] = C[2*j + (lane>>4)][lane&15].
//
// Grid:  [ceil(M/16), ceil(N/16), 1]
// Block: [256, 1, 1]. Static LDS 4352 + 8192 = 12544 B; dynamic shared_mem=0.
//
// Reorders FP32 K accumulation (disjoint 64-wide slices per wave, fixed
// wave-0..7 LDS reduction), so bit-exactness vs the base kernel is NOT
// claimed (parity gate: relL2 <= 5e-5). Deterministic (fixed reduction
// order, no atomics) hence capture-safe. Compile with
// `hipcc --offload-arch=gfx1100`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4v_t;

__launch_bounds__(256, 4)
extern "C" __global__ void gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int m_lane = lane & 15;
    const int safe_batch = (batch_start + m_lane < batch_size)
        ? (batch_start + m_lane) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const int slabs = K / 512;

    // Two packed G256 groups per row, 16 rows: 16 * 272 = 4352 B.
    // The logical 272-vector flat fill uses dwordx4 loads. Vectors 0..254
    // cover rows 0..14, vector 255 starts row 15, and lanes 0..15 load its
    // remaining 16 vectors. Within each row, consecutive threads access
    // consecutive 16-byte addresses.
    __shared__ __align__(16) unsigned char staged_weights[16][272];
    __shared__ float partials[8 * 32 * 8];

    const int load_row = tid / 17;
    const int load_vec = tid - load_row * 17;
    const int safe_load_row = (row_start + load_row < M)
        ? (row_start + load_row) : (M - 1);

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int slab = 0; slab < slabs; ++slab) {
        const char* src0 = A + (long long)safe_load_row * row_bytes
            + slab * 272 + load_vec * 16;
        unsigned char* dst0 = staged_weights[load_row] + load_vec * 16;
        *(uint4v_t*)dst0 = *(const uint4v_t*)src0;

        if (tid < 16) {
            const int vec = tid + 1;
            const int safe_row15 = (row_start + 15 < M) ? (row_start + 15) : (M - 1);
            const char* src1 = A + (long long)safe_row15 * row_bytes
                + slab * 272 + vec * 16;
            unsigned char* dst1 = staged_weights[15] + vec * 16;
            *(uint4v_t*)dst1 = *(const uint4v_t*)src1;
        }
        __syncthreads();

        const int group_in_slab = wave >> 2;
        const int quarter_in_group = wave & 3;
        const unsigned char* gp = staged_weights[m_lane] + group_in_slab * 136;
        // quarter 0..1 -> weights 0..127 (half0); quarter 2..3 -> 128..255 (half1).
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (quarter_in_group < 2) ? hA : hB;
        const _Float16 sc_h =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const _Float16 zp_h =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const _Float16* xg = x_base + slab * 512 + wave * 64;
        const int frag_base = quarter_in_group * 4;

        // Deliberately NOT unrolled (`unroll 1`): a rolled 4-iteration loop
        // keeps exactly one fragment's pk0/pk1/a_reg/b_reg live at a time
        // (plus the loop-carried float8 acc and sc_h/zp_h scalars). A fully
        // unrolled body holds 4 fragments' temporaries simultaneously and
        // costs 126 VGPRs -> 1 WG/CU; the rolled form targets <= 64 VGPRs
        // (2 WGs/CU). A 4-trip loop is negligible at this shape.
        #pragma unroll 1
        for (int f = 0; f < 4; ++f) {
            // Global 16-wide fragment within the group; 8 bytes = 2 u32.
            const int frag = frag_base + f;
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + frag * 8);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + frag * 8 + 4);

            // Base-kernel DQ macro verbatim (half16, full 16-K per lane).
            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(xg + f * 16);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }

        // All waves must finish consuming the slab before the cooperative fill
        // overwrites it on the next iteration.
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        partials[(wave * 32 + lane) * 8 + j] = acc[j];
    }
    __syncthreads();

    // Wave 0 performs a deterministic, explicitly ordered reduction.
    // gfx11 interleaved C mapping: acc[j] = C[2*j + (lane>>4)][lane & 15].
    if (wave == 0) {
        const int out_col = batch_start + m_lane;
        if (out_col < batch_size) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sum = partials[(0 * 32 + lane) * 8 + j];
                #pragma unroll
                for (int w = 1; w < 8; ++w) {
                    sum += partials[(w * 32 + lane) * 8 + j];
                }
                const int out_row = row_start + 2 * j + (lane >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] += sum;
                }
            }
        }
    }
}
