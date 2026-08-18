// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_RSRC
#define HIPFIRE_WEIGHT_RSRC(base) 0
#endif
#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, off) (*(const unsigned int*)(ptr))
#endif


#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
using hipfire_lm_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_lm_f32x4 = float __attribute__((ext_vector_type(4)));
#endif

// HFQ4-G256 multi-row GEMV: one warp computes R output rows.
// Default (non-RDNA3) variant used by archs that fall through to the
// default gemv_mq4cg256 dispatch path: gfx1010, gfx1013 (BC-250), etc.
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
// gemv_mq4cg256_multirow.gfx1100.hip variant: R=1 is already near-peak
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
__device__ __forceinline__ void gemv_mq4cg256_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 128;
    const char* hplane = A + (long long)M * groups_per_row * 128;
    const int boff = tid * 4;

    // Row pointers. For partial final workgroups (row_base + r >= M) we
    // clamp the pointer to row_base so reads stay in-bounds; the write is
    // masked separately by the row_valid check.
    const char* row_ptrs[R];
    const char* row_hptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
        row_hptrs[r] = hplane + (long long)safe_row * groups_per_row * 4;
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
            const char* gp0 = row_ptrs[r] + g * 128;
            const char* gp1 = gp0 + 128;
            const char* gp2 = gp1 + 128;
            const char* gp3 = gp2 + 128;

            const unsigned int hdr0__ = *(const unsigned int*)(row_hptrs[r] + g * 4);
            float sc0 = __half2float(__ushort_as_half((unsigned short)(hdr0__ & 0xFFFFu)));
            float zp0 = __half2float(__ushort_as_half((unsigned short)(hdr0__ >> 16)));
            const unsigned int hdr1__ = *(const unsigned int*)(row_hptrs[r] + (g + 1) * 4);
            float sc1 = __half2float(__ushort_as_half((unsigned short)(hdr1__ & 0xFFFFu)));
            float zp1 = __half2float(__ushort_as_half((unsigned short)(hdr1__ >> 16)));
            const unsigned int hdr2__ = *(const unsigned int*)(row_hptrs[r] + (g + 2) * 4);
            float sc2 = __half2float(__ushort_as_half((unsigned short)(hdr2__ & 0xFFFFu)));
            float zp2 = __half2float(__ushort_as_half((unsigned short)(hdr2__ >> 16)));
            const unsigned int hdr3__ = *(const unsigned int*)(row_hptrs[r] + (g + 3) * 4);
            float sc3 = __half2float(__ushort_as_half((unsigned short)(hdr3__ & 0xFFFFu)));
            float zp3 = __half2float(__ushort_as_half((unsigned short)(hdr3__ >> 16)));

            unsigned int pk0 = *(const unsigned int*)(gp0 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + boff);

            acc[r][0] += (sc0 * (float)((pk0 >>  0) & 0xFu) + zp0) * x00
                       + (sc0 * (float)((pk0 >>  4) & 0xFu) + zp0) * x01
                       + (sc0 * (float)((pk0 >>  8) & 0xFu) + zp0) * x02
                       + (sc0 * (float)((pk0 >> 12) & 0xFu) + zp0) * x03
                       + (sc0 * (float)((pk0 >> 16) & 0xFu) + zp0) * x04
                       + (sc0 * (float)((pk0 >> 20) & 0xFu) + zp0) * x05
                       + (sc0 * (float)((pk0 >> 24) & 0xFu) + zp0) * x06
                       + (sc0 * (float)((pk0 >> 28) & 0xFu) + zp0) * x07;
            acc[r][1] += (sc1 * (float)((pk1 >>  0) & 0xFu) + zp1) * x10
                       + (sc1 * (float)((pk1 >>  4) & 0xFu) + zp1) * x11
                       + (sc1 * (float)((pk1 >>  8) & 0xFu) + zp1) * x12
                       + (sc1 * (float)((pk1 >> 12) & 0xFu) + zp1) * x13
                       + (sc1 * (float)((pk1 >> 16) & 0xFu) + zp1) * x14
                       + (sc1 * (float)((pk1 >> 20) & 0xFu) + zp1) * x15
                       + (sc1 * (float)((pk1 >> 24) & 0xFu) + zp1) * x16
                       + (sc1 * (float)((pk1 >> 28) & 0xFu) + zp1) * x17;
            acc[r][2] += (sc2 * (float)((pk2 >>  0) & 0xFu) + zp2) * x20
                       + (sc2 * (float)((pk2 >>  4) & 0xFu) + zp2) * x21
                       + (sc2 * (float)((pk2 >>  8) & 0xFu) + zp2) * x22
                       + (sc2 * (float)((pk2 >> 12) & 0xFu) + zp2) * x23
                       + (sc2 * (float)((pk2 >> 16) & 0xFu) + zp2) * x24
                       + (sc2 * (float)((pk2 >> 20) & 0xFu) + zp2) * x25
                       + (sc2 * (float)((pk2 >> 24) & 0xFu) + zp2) * x26
                       + (sc2 * (float)((pk2 >> 28) & 0xFu) + zp2) * x27;
            acc[r][3] += (sc3 * (float)((pk3 >>  0) & 0xFu) + zp3) * x30
                       + (sc3 * (float)((pk3 >>  4) & 0xFu) + zp3) * x31
                       + (sc3 * (float)((pk3 >>  8) & 0xFu) + zp3) * x32
                       + (sc3 * (float)((pk3 >> 12) & 0xFu) + zp3) * x33
                       + (sc3 * (float)((pk3 >> 16) & 0xFu) + zp3) * x34
                       + (sc3 * (float)((pk3 >> 20) & 0xFu) + zp3) * x35
                       + (sc3 * (float)((pk3 >> 24) & 0xFu) + zp3) * x36
                       + (sc3 * (float)((pk3 >> 28) & 0xFu) + zp3) * x37;
        }
    }

    // Tail: per-row, accumulate into acc[r][gi] to preserve the 4-way
    // interleaving (same reason as the single-row kernel — 5302926).
    #define TAIL_LOAD_X(b) \
        const float txv0 = x[(b) + 0], txv1 = x[(b) + 1], \
                    txv2 = x[(b) + 2], txv3 = x[(b) + 3], \
                    txv4 = x[(b) + 4], txv5 = x[(b) + 5], \
                    txv6 = x[(b) + 6], txv7 = x[(b) + 7]
    #define TAIL_DOG(pk, sc, zp, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * txv0  \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * txv1 \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * txv2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * txv3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * txv4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * txv5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * txv6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * txv7

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 128;
            const unsigned int hdr__ = *(const unsigned int*)(row_hptrs[r] + g * 4);
            float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16)));
            unsigned int pk = *(const unsigned int*)(gp + boff);
            TAIL_DOG(pk, sc, zp, acc[r][0]);
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 128;
            const unsigned int hdr__ = *(const unsigned int*)(row_hptrs[r] + g * 4);
            float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16)));
            unsigned int pk = *(const unsigned int*)(gp + boff);
            TAIL_DOG(pk, sc, zp, acc[r][1]);
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 128;
            const unsigned int hdr__ = *(const unsigned int*)(row_hptrs[r] + g * 4);
            float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16)));
            unsigned int pk = *(const unsigned int*)(gp + boff);
            TAIL_DOG(pk, sc, zp, acc[r][2]);
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
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4cg256_multirow_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_GFX1151_LM_HEAD_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const long long row_stride = (long long)groups_per_row * 128;
    // Clamp OOB row1 to row0 so weight reads stay in-bounds; store is masked.
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* hplane = A + (long long)M * groups_per_row * 128;
    const unsigned int hplane_off = (unsigned int)((long long)M * groups_per_row * 128);
    const int boff = tid * 4;
    const unsigned int row_weight_offset0 = (unsigned int)((long long)row0 * row_stride);
    const unsigned int row_weight_offset1 =
        (unsigned int)((long long)(have_row1 ? row1 : row0) * row_stride);
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // The full lm-head matrix fits the raw-buffer 32-bit offset range; one SRD
    // avoids extending the live descriptor set in this 94-VGPR kernel.
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif
#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
    const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(x), 0, 0xffffffffu, 0x31004000u);
#endif

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
    #define LOAD_X8_INTO(b, pfx) \
        const hipfire_lm_f32x4 pfx##v0 = __builtin_bit_cast( \
            hipfire_lm_f32x4, __builtin_amdgcn_raw_buffer_load_b128( \
                x_rsrc, (unsigned int)(b) * 4u, 0, 0)); \
        const hipfire_lm_f32x4 pfx##v1 = __builtin_bit_cast( \
            hipfire_lm_f32x4, __builtin_amdgcn_raw_buffer_load_b128( \
                x_rsrc, ((unsigned int)(b) + 4u) * 4u, 0, 0)); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w
#else
    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w
#endif

#if defined(HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER)
    // Uniform per-group headers are cheaper on the scalar/global path. Only
    // the lane-divergent packed dword uses temporal buffer VMEM.
    #define LOAD_LM_HEADER(gp, weight_offset) (*(const unsigned int*)(hplane + (((gp) - A) >> 5)))
#else
    #define LOAD_LM_HEADER(gp, weight_offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, hplane + (((gp) - A) >> 5), hplane_off + ((weight_offset) >> 5))
#endif
    #define LOAD_LM_PACKED(gp, weight_offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)

    #define DOG_X8(gp, weight_offset, a, pfx) do { \
        const unsigned int hdr__ = LOAD_LM_HEADER(gp, weight_offset); \
        float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16))); \
        unsigned int pk = LOAD_LM_PACKED((gp) + boff, (weight_offset) + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * pfx##0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * pfx##1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * pfx##2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * pfx##3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * pfx##4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * pfx##5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * pfx##6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * pfx##7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist all 32 x values for this group-quad before any DOG.
        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);
#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
        asm volatile("" : :
            "v"(xav0), "v"(xav1), "v"(xbv0), "v"(xbv1),
            "v"(xcv0), "v"(xcv1), "v"(xdv0), "v"(xdv1) : "memory");
#endif

        // Branchless dual-row DOG with all x resident (no mid-quad x reload).
        DOG_X8(row_ptr0 + g * 128, row_weight_offset0 + g * 128, a0, xa);
        DOG_X8(row_ptr1 + g * 128, row_weight_offset1 + g * 128, b0, xa);
        DOG_X8(row_ptr0 + (g + 1) * 128, row_weight_offset0 + (g + 1) * 128, a1, xb);
        DOG_X8(row_ptr1 + (g + 1) * 128, row_weight_offset1 + (g + 1) * 128, b1, xb);
        DOG_X8(row_ptr0 + (g + 2) * 128, row_weight_offset0 + (g + 2) * 128, a2, xc);
        DOG_X8(row_ptr1 + (g + 2) * 128, row_weight_offset1 + (g + 2) * 128, b2, xc);
        DOG_X8(row_ptr0 + (g + 3) * 128, row_weight_offset0 + (g + 3) * 128, a3, xd);
        DOG_X8(row_ptr1 + (g + 3) * 128, row_weight_offset1 + (g + 3) * 128, b3, xd);
    }

    // Tail uses sequential x (at most 3 groups; hoist buys little).
    #define LOAD_X8(b) \
        const float4 xv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 xv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float x0 = xv0.x, x1 = xv0.y, x2 = xv0.z, x3 = xv0.w; \
        const float x4 = xv1.x, x5 = xv1.y, x6 = xv1.z, x7 = xv1.w

    #define DOG_X8_TAIL(gp, weight_offset, a) do { \
        const unsigned int hdr__ = LOAD_LM_HEADER(gp, weight_offset); \
        float sc = __half2float(__ushort_as_half((unsigned short)(hdr__ & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(hdr__ >> 16))); \
        unsigned int pk = LOAD_LM_PACKED((gp) + boff, (weight_offset) + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(row_ptr0 + g * 128, row_weight_offset0 + g * 128, a0);
        DOG_X8_TAIL(row_ptr1 + g * 128, row_weight_offset1 + g * 128, b0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(row_ptr0 + g * 128, row_weight_offset0 + g * 128, a1);
        DOG_X8_TAIL(row_ptr1 + g * 128, row_weight_offset1 + g * 128, b1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(row_ptr0 + g * 128, row_weight_offset0 + g * 128, a2);
        DOG_X8_TAIL(row_ptr1 + g * 128, row_weight_offset1 + g * 128, b2);
    }
    #undef DOG_X8_TAIL
    #undef LOAD_X8
    #undef DOG_X8
    #undef LOAD_X8_INTO
    #undef LOAD_LM_PACKED
    #undef LOAD_LM_HEADER

    float s0 = (a0 + a1) + (a2 + a3);
    float s1 = (b0 + b1) + (b2 + b3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down(s0, offset);
        s1 += __shfl_down(s1, offset);
    }

    if (tid == 0) {
        y[row0] = s0;
        if (have_row1) y[row1] = s1;
    }
}

__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4cg256_multirow_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_mq4cg256_multirow_body<4>(A, x, y, M, K);
}

// R=8 drops occupancy to 8 waves/SIMD (192 VGPR budget) because 32
// accumulators + 32 x values + 16 row pointers would blow launch_bounds(32,16).
__launch_bounds__(32, 6)
extern "C" __global__ void gemv_mq4cg256_multirow_r8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_mq4cg256_multirow_body<8>(A, x, y, M, K);
}
