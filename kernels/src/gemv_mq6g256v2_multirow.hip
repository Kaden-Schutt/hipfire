// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
using hipfire_lm_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_lm_f32x4 = float __attribute__((ext_vector_type(4)));
#endif

// MQ6-G256 v2 multi-row GEMV: one warp computes R output rows.
// Neutral V2: 200 B/group, header per half, 6-bit payload 4/3B.
// Default (non-RDNA3) variant used by archs that fall through to the
// default gemv_mq6g256v2 dispatch path: gfx1010, gfx1013 (BC-250), etc.
// Each thread keeps R×4 accumulators. x values are hoisted once per
// group-quad and reused across all R rows.
//
// ───────────────────────────────────────────────────────────────────────────
// STATUS: R=2 is a modest win on gfx1010 / gfx1013 (BC-250). Enable via
// HIPFIRE_GEMV_ROWS ∈ {2, 4, 8}; the dispatch layer picks this kernel on
// non-RDNA3 archs when the env var is set (or when the arch-specific
// default is R=2 — see dispatch.rs).
//
// Empirical result on BC-250 / gfx1013 (runs as gfx1010), 24 CUs,
// 224 GiB/s peak DRAM, Qwen3.5 0.8B MQ4, bench_qwen35_mq4 200 gen:
//
//   R=1 (baseline): 212.6 tok/s  (108.7 GiB/s, 48.5% of peak)
//   R=2           : 218.4 tok/s  (111.7 GiB/s, 49.9% of peak)  +2.7%
//   R=4           : 187.9 tok/s  ( 96.1 GiB/s)                 -12.3%
//   R=8           : 188.1 tok/s  ( 96.2 GiB/s)                 -12.2%
//
// VGPR counts (gfx1010 compile, -O3): R=2=53, R=4=81, R=8=137.
// Register-file budget on RDNA1 is 1024 VGPRs/SIMD (not 1536 like
// RDNA3/4), so launch_bounds have to be tighter:
//   R=2: launch_bounds(32, 18) → ceil(1024/56) = 18 waves/SIMD (vs 20 for R=1)
//   R=4: launch_bounds(32, 10) → ceil(1024/84) = 12 waves/SIMD
//   R=8: launch_bounds(32,  6) → ceil(1024/140) = 7 waves/SIMD
//
// R=2 wins because the x-value hoist amortizes 32 float loads per
// group-quad across 2 rows while only dropping occupancy from 20 → 18
// waves/SIMD. R=4 and R=8 drop occupancy too far to pay for the reuse.
//
// Contrast with gfx1100 (RX 7900 XTX, 960 GiB/s peak), measured on the
// gemv_mq6g256v2_multirow.gfx1100.hip variant: R=1 is already near-peak
// BW-bound (577 GiB/s on 9B ≈ 60% of peak) and multi-row under-subscribes
// the wave scheduler. R=2 was actually slightly NEGATIVE on gfx1100.
// The inversion is real: lower-BW targets benefit from R=2 reuse, higher-BW
// targets are already BW-saturated and lose from reduced wave count.
// ───────────────────────────────────────────────────────────────────────────
//
// Grid = ceil(M / R) workgroups × 32 threads.
//
// Tail handling: the main loop processes groups 4 at a time into
// acc[r][g%4]. Tail groups MUST accumulate into the SAME slot (acc[r][g%4])
// to preserve the 4-way interleaving that the combine step assumes.
// Dumping all tail groups into acc[r][0] broke 4B/0.8B MQ4 output on the
// single-row gfx1100 kernel — see commit 5302926 for the full story.

// The body is templated on R and shared between entry points.
template<int R>
__device__ __forceinline__ void gemv_mq6g256v2_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    const int boff = tid * 6;

    // Row pointers. For partial final workgroups (row_base + r >= M) we
    // clamp the pointer to row_base so reads stay in-bounds; the write is
    // masked separately by the row_valid check.
    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }

    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // Main loop: 4 groups per iteration.
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base = g * 256 + tid * 8;

        // Hoist x loads — identical across all R rows. The compiler will
        // keep these in registers for the duration of the row loop.
        const float x00 = x[base + 0],   x01 = x[base + 1],
                    x02 = x[base + 2],   x03 = x[base + 3],
                    x04 = x[base + 4],   x05 = x[base + 5],
                    x06 = x[base + 6],   x07 = x[base + 7];
        const float x10 = x[base + 256 + 0], x11 = x[base + 256 + 1],
                    x12 = x[base + 256 + 2], x13 = x[base + 256 + 3],
                    x14 = x[base + 256 + 4], x15 = x[base + 256 + 5],
                    x16 = x[base + 256 + 6], x17 = x[base + 256 + 7];
        const float x20 = x[base + 512 + 0], x21 = x[base + 512 + 1],
                    x22 = x[base + 512 + 2], x23 = x[base + 512 + 3],
                    x24 = x[base + 512 + 4], x25 = x[base + 512 + 5],
                    x26 = x[base + 512 + 6], x27 = x[base + 512 + 7];
        const float x30 = x[base + 768 + 0], x31 = x[base + 768 + 1],
                    x32 = x[base + 768 + 2], x33 = x[base + 768 + 3],
                    x34 = x[base + 768 + 4], x35 = x[base + 768 + 5],
                    x36 = x[base + 768 + 6], x37 = x[base + 768 + 7];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp0 = row_ptrs[r] + g * 200;
            const char* gp1 = gp0 + 200;
            const char* gp2 = gp1 + 200;
            const char* gp3 = gp2 + 200;

            const unsigned int hA0 = *(const unsigned int*)(gp0);
            const unsigned int hB0 = *(const unsigned int*)(gp0 + 4);
            const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
            float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
            float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
            const unsigned int hA1 = *(const unsigned int*)(gp1);
            const unsigned int hB1 = *(const unsigned int*)(gp1 + 4);
            const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
            float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
            float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
            const unsigned int hA2 = *(const unsigned int*)(gp2);
            const unsigned int hB2 = *(const unsigned int*)(gp2 + 4);
            const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
            float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
            float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
            const unsigned int hA3 = *(const unsigned int*)(gp3);
            const unsigned int hB3 = *(const unsigned int*)(gp3 + 4);
            const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
            float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
            float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

            // 6-bit payload: 6 bytes per thread = 8 weights
            const unsigned char* d0 = (const unsigned char*)(gp0 + 8 + boff);
            const unsigned char* d1 = (const unsigned char*)(gp1 + 8 + boff);
            const unsigned char* d2 = (const unsigned char*)(gp2 + 8 + boff);
            const unsigned char* d3 = (const unsigned char*)(gp3 + 8 + boff);
            // q0..q7 unpack for each group
                        #define MQ6_ACC_TMPL(d_, sc_, zp_, acc_, x0_,x1_,x2_,x3_,x4_,x5_,x6_,x7_) do{ const unsigned char* _d=(d_); unsigned char b0=_d[0],b1=_d[1],b2=_d[2],b3=_d[3],b4=_d[4],b5=_d[5]; int q0=b0&63,q1=(b0>>6)|((b1&0xF)<<2),q2=(b1>>4)|((b2&3)<<4),q3=b2>>2,q4=b3&63,q5=(b3>>6)|((b4&0xF)<<2),q6=(b4>>4)|((b5&3)<<4),q7=b5>>2; (acc_)+=(sc_*(float)q0+zp_)*x0_+(sc_*(float)q1+zp_)*x1_+(sc_*(float)q2+zp_)*x2_+(sc_*(float)q3+zp_)*x3_+(sc_*(float)q4+zp_)*x4_+(sc_*(float)q5+zp_)*x5_+(sc_*(float)q6+zp_)*x6_+(sc_*(float)q7+zp_)*x7_; }while(0)
            MQ6_ACC_TMPL(d0, sc0, zp0, acc[r][0], x00,x01,x02,x03,x04,x05,x06,x07);
            MQ6_ACC_TMPL(d1, sc1, zp1, acc[r][1], x10,x11,x12,x13,x14,x15,x16,x17);
            MQ6_ACC_TMPL(d2, sc2, zp2, acc[r][2], x20,x21,x22,x23,x24,x25,x26,x27);
            MQ6_ACC_TMPL(d3, sc3, zp3, acc[r][3], x30,x31,x32,x33,x34,x35,x36,x37);
            #undef MQ6_ACC_TMPL
        }
    }

    // Tail: per-row, accumulate into acc[r][gi] to preserve the 4-way
    // interleaving (same reason as the single-row kernel — 5302926).
    #define TAIL_LOAD_X(b) \
        const float txv0 = x[(b) + 0], txv1 = x[(b) + 1], \
                    txv2 = x[(b) + 2], txv3 = x[(b) + 3], \
                    txv4 = x[(b) + 4], txv5 = x[(b) + 5], \
                    txv6 = x[(b) + 6], txv7 = x[(b) + 7]
    #define TAIL_DOG_MQ6(dptr, sc, zp, a) do { \
        const unsigned char* dd = (const unsigned char*)(dptr); \
        unsigned char b0=dd[0], b1=dd[1], b2=dd[2], b3=dd[3], b4=dd[4], b5=dd[5]; \
        int q0=b0 &63; int q1=(b0>>6)|((b1&0xF)<<2); int q2=(b1>>4)|((b2&3)<<4); int q3=b2>>2; \
        int q4=b3 &63; int q5=(b3>>6)|((b4&0xF)<<2); int q6=(b4>>4)|((b5&3)<<4); int q7=b5>>2; \
        (a) += (sc*(float)q0+zp)*txv0 + (sc*(float)q1+zp)*txv1 + (sc*(float)q2+zp)*txv2 + (sc*(float)q3+zp)*txv3 \
             + (sc*(float)q4+zp)*txv4 + (sc*(float)q5+zp)*txv5 + (sc*(float)q6+zp)*txv6 + (sc*(float)q7+zp)*txv7; \
    } while(0)

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 200;
            const unsigned int hA = *(const unsigned int*)(gp);
            const unsigned int hB = *(const unsigned int*)(gp + 4);
            const unsigned int hs = (tid < 16) ? hA : hB;
            float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            TAIL_DOG_MQ6(gp + 8 + boff, sc, zp, acc[r][0]);
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 200;
            const unsigned int hA = *(const unsigned int*)(gp);
            const unsigned int hB = *(const unsigned int*)(gp + 4);
            const unsigned int hs = (tid < 16) ? hA : hB;
            float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            TAIL_DOG_MQ6(gp + 8 + boff, sc, zp, acc[r][1]);
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 200;
            const unsigned int hA = *(const unsigned int*)(gp);
            const unsigned int hB = *(const unsigned int*)(gp + 4);
            const unsigned int hs = (tid < 16) ? hA : hB;
            float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            TAIL_DOG_MQ6(gp + 8 + boff, sc, zp, acc[r][2]);
        }
    }
    #undef TAIL_LOAD_X
    #undef TAIL_DOG

    // Per-row warp reduction and write-back.
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] = s;
    }
}

// ── extern "C" entry points per R ──
// Each gets its own launch_bounds because VGPR pressure scales with R.

// R13c1 lever (gfx1201): re-hoist full 32-wide x per group-quad on the
// specialized R=2 branchless body (R12 kept sequential float4×2 x loads).
// BOD: L2-resident/mem-busy occ~89% mem_busy~93% vgpr=96. R12 sequential
// x was meant to free VGPRs but profile kept vgpr=96 / SATURATED — so the
// live-x cut bought no occupancy and serializes each dual-row DOG pair
// behind a fresh x load. Hypothesis: hoist all four groups' x once
// (float4 packs), then fire eight branchless DOGs with all x resident —
// same math (4-way acc[g%4] + pairwise combine, 5302926) and same
// branchless dual-row clamp, but removes x-load → DOG serialization so
// dual-issue can cover more of the weight-load latency under L2-hit.
// R=2 goes through the SHARED TEMPLATE, like R=4 and R=8.
//
// It used to have its own hand-written dual-row body — an lm-head experiment
// carrying raw-buffer VMEM loads, an X-buffer SRD, and a compile-time K=2048
// specialization, all selected by macros that only fire on gfx1100/gfx1201.
// On any other target (gfx1151 among them) every one of those #ifs took its
// `#else`, leaving a pointer-load path that was never validated: it produced
// KLD 10.64 / PPL 248,303 against 0.0071 for the R=4 and R=8 template bodies
// on the same weights. The entry point was unreferenced by dispatch, so
// nothing caught it.
//
// If the specialization is revived for gfx1201/gfx1100, gate it on those
// targets explicitly and keep this template as the fallback — do not let a
// tuned variant become the only implementation of a width again.
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq6g256v2_multirow_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_mq6g256v2_multirow_body<2>(A, x, y, M, K);
}

__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq6g256v2_multirow_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_mq6g256v2_multirow_body<4>(A, x, y, M, K);
}

// R=8 drops occupancy to 8 waves/SIMD (192 VGPR budget) because 32
// accumulators + 32 x values + 16 row pointers would blow launch_bounds(32,16).
__launch_bounds__(32, 6)
extern "C" __global__ void gemv_mq6g256v2_multirow_r8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_mq6g256v2_multirow_body<8>(A, x, y, M, K);
}
