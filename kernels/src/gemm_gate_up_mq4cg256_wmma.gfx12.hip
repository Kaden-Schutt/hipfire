// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused MQ4C-G256 GEMM for the Qwen3.5
// FFN preamble (gate + up projections).
//
// **gfx12 (RDNA4) variant — K4 unroll**, issue #65 lever 2. Port of
// gemm_gate_up_hfq4g256_wmma.gfx12.hip to qt=45 (MQ4C / v1.5).
//
// Pad mq4c (qt=45) — 136 B/group (4 B header + 4 B pad + 128 B payload):
//   header (fp16 scale + fp16 zero) in header plane, payload nibbles in
//   payload plane (16-byte aligned). w = q * scale + zero — ONE affine grid
//   per 256; no half-select.

//
// **gfx12 (RDNA4) variant — K4 unroll**, issue #65 lever 2. The original
// PR #62 ship used K2 (lifted from gfx11). RDNA4 has higher memory
// latency / less aggressive scheduler than RDNA3 at this kernel shape;
// deeper unroll gives the compiler a longer pipeline to hide loads
// behind WMMA execution. Measured 2.0× on R9700 9B pp=128 gate_up
// (1842 → 922 µs/call, 39 → 80 GiB/s effective).
//
// Same lane decomposition as the K2 baseline — the only structural
// change is processing 4 K-tiles per loop body instead of 2.
//
// gfx12 vs gfx11 differences (carried over from the K2 baseline):
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//      (gfx11 uses _w32 without the _gfx12 suffix).
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t.
//      Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (kABKLane=2):
//        lane (tid >> 4) = 0  -> carries K = [0..7]  of the 16-K tile
//        lane (tid >> 4) = 1  -> carries K = [8..15] of the 16-K tile
//      Each lane loads HALF as much per WMMA tile as gfx11.
//   4. C-output mapping (validated 2026-04-27 in the QKV scaffold's
//      channel-test on R9700):
//        gfx11: acc[j] = C[2*j + (tid>>4)][tid & 15]    (rows interleaved)
//        gfx12: acc[j] = C[8*(tid>>4) + j][tid & 15]    (rows contiguous —
//                                                        group 0: rows 0..7,
//                                                        group 1: rows 8..15)
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.
//
// Each 16-row tile may straddle the gate/up boundary. Per-thread output
// routing resolves which Y pointer (Y_gate or Y_up) each row writes to.
// Overwrite semantics (y = acc, not +=).
//
// K constraint: K must be a multiple of 256 (the MQ4C-G256 group size).
// The K4 unroll lives entirely inside a single group's 16 K-tiles —
// it doesn't impose any additional K constraint.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_mq4cg256_wmma_gfx12(
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

    // Lane decomposition (gfx12 wave32 WMMA):
    //   m_lane = tid & 15   (0..15)  -> row index within the 16-row tile
    //   k_grp  = tid >> 4   (0 or 1) -> which K half-tile this lane carries
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix — each with OWN header-plane base.
    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate;  local_row = safe_row;            
    } else {
        A = A_up;    local_row = safe_row - gate_m;   
    }

    const int gpr = K / 256;
    const char* row_base = A + (long long)local_row * gpr * 136;

    // B operand indexes BATCH (N) dimension by m_lane (CK kBNLane=16).
    // Each lane holds the 8-K slice of its batch row corresponding to k_grp.
    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < gpr; g++) {
        const char* gp = row_base + g * 136;
        const unsigned int hdr__ = *(const unsigned int*)(gp);
        _Float16 sc_h = __ushort_as_half((unsigned short)(hdr__ & 0xFFFFu));
        _Float16 zp_h = __ushort_as_half((unsigned short)(hdr__ >> 16));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;


        // K4 unroll (issue #65 lever 2): 4 K-tiles per loop body, all loads
        // up-front so the compiler can deep-pipeline the WMMA chain. 4
        // iterations cover K=256.
        #define DQ8(pk) \
            a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

        for (int kt = 0; kt < 16; kt += 4) {
            const int k_off_a = (kt + 0) * 16;
            const int k_off_b = (kt + 1) * 16;
            const int k_off_c = (kt + 2) * 16;
            const int k_off_d = (kt + 3) * 16;

            // 4 weight reads, all up-front. Planar: no +4 (payload aligned).
            unsigned int pka = *(const unsigned int*)(gp + 8 + k_off_a / 2 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(gp + 8 + k_off_b / 2 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(gp + 8 + k_off_c / 2 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(gp + 8 + k_off_d / 2 + k_grp * 4);


            // 4 X reads, all up-front. ~64 bytes/lane = 16 VGPR live across
            // the 4 dequant+WMMA pairs below.
            half8_t b_a = *(const half8_t*)(x_base + k_off_a + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + k_off_b + k_grp * 8);
            half8_t b_c = *(const half8_t*)(x_base + k_off_c + k_grp * 8);
            half8_t b_d = *(const half8_t*)(x_base + k_off_d + k_grp * 8);

            half8_t a_reg;
            DQ8(pka);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ8
    }

    // --- Output ---
    // gfx12 wave32 WMMA C-mapping: lane group 0 holds rows 0..7,
    // group 1 holds rows 8..15 (validated on R9700 via the QKV scaffold).
    const int out_col = batch_start + m_lane;  // batch index
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < total_m) {
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

// Experimental cold-weight LDS staging variant for fused gate+up.
// Same slab/wave structure as residual_ldsstage; routes rows across A_gate/A_up
// and overwrites Y_gate/Y_up. Opt-in via HIPFIRE_MQ4CG256_LDSSTAGE=1; K%512==0.
// Pad: 136 B/group, slab = 272 B/row (2×136).
typedef unsigned int __attribute__((ext_vector_type(4))) uint4v_t;

__launch_bounds__(256, 4)
extern "C" __global__ void gemm_gate_up_mq4cg256_wmma_gfx12_ldsstage(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int tid = threadIdx.x;
    const int wave = tid>>5;
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
    const int gpr = K / 256;
    const int slabs = K / 512;
    // Header planes for each tensor (planar layout)

    // Two packed G256 groups per row, 16 rows: 16 * 272 = 4352 B.
    // Pad: 136 B/group (header inline + 4 B pad + 128 B payload), slab = 272 B/row.
    // Fill uses 16 dwordx4 loads covering 0..255, plus a 16 B tail at 256.
    __shared__ __align__(16) unsigned char staged_weights[16][272];
    __shared__ float partials[8 * 32 * 8];


    const int load_row = tid >> 4;          // 0..15
    const int load_vec = tid & 15;          // 0..15  (16 B * 16 = 256 B bulk, plus 16 B tail)
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
        const char* src0 = A_load + (long long)local_load_row * gpr * 136
            + (long long)slab * 272 + load_vec * 16;
        unsigned char* dst0 = staged_weights[load_row] + load_vec * 16;
        *(uint4v_t*)dst0 = *(const uint4v_t*)src0;
        if (tid < 16) {
            const int safe_tail_row = (row_start + tid < total_m)
                ? (row_start + tid) : (total_m - 1);
            const char* A_tail;
            int local_tail_row;
            if (safe_tail_row < gate_m) {
                A_tail = A_gate;  local_tail_row = safe_tail_row;
            } else {
                A_tail = A_up;    local_tail_row = safe_tail_row - gate_m;
            }
            const char* src1 = A_tail + (long long)local_tail_row * gpr * 136 + (long long)slab * 272 + 256;
            unsigned char* dst1 = staged_weights[tid] + 256;
            *(uint4v_t*)dst1 = *(const uint4v_t*)src1;
        }
        __syncthreads();

        const int group_in_slab = wave >> 2;
        const int quarter_in_group = wave & 3;
        const unsigned char* gp = staged_weights[m_lane] + group_in_slab * 136;
        const unsigned int hdr__ = *(const unsigned int*)(gp);
        _Float16 sc_h = __ushort_as_half((unsigned short)(hdr__ & 0xFFFFu));
        _Float16 zp_h = __ushort_as_half((unsigned short)(hdr__ >> 16));
        const _Float16* xg = x_base + slab * 512 + wave * 64;
        const int packed_base = quarter_in_group * 32;

        #define DQ8_LDS(pk) \
            a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

        // Pad: header at gp[0..4), payload at gp+8.
        const unsigned int pka = *(const unsigned int*)(gp + 8 + packed_base + 0 + k_grp * 4);
        const unsigned int pkb = *(const unsigned int*)(gp + 8 + packed_base + 8 + k_grp * 4);
        const unsigned int pkc = *(const unsigned int*)(gp + 8 + packed_base + 16 + k_grp * 4);
        const unsigned int pkd = *(const unsigned int*)(gp + 8 + packed_base + 24 + k_grp * 4);


        const half8_t b_a = *(const half8_t*)(xg + 0 + k_grp * 8);
        const half8_t b_b = *(const half8_t*)(xg + 16 + k_grp * 8);
        const half8_t b_c = *(const half8_t*)(xg + 32 + k_grp * 8);
        const half8_t b_d = *(const half8_t*)(xg + 48 + k_grp * 8);

        half8_t a_reg;
        DQ8_LDS(pka);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
        DQ8_LDS(pkb);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
        DQ8_LDS(pkc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
        DQ8_LDS(pkd);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        #undef DQ8_LDS

        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        partials[(wave * 32 + lane) * 8 + j] = acc[j];
    }
    __syncthreads();

    // Wave 0 reduction with gate_up OVERWRITE semantics (y = sum, not +=).
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
                const int out_row = row_start + 8 * k_grp + j;
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
