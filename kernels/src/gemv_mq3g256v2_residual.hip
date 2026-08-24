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

#ifndef HIPFIRE_RESIDUAL_KERNEL
#define HIPFIRE_RESIDUAL_KERNEL gemv_mq3g256v2_residual
#endif

// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
//
// VARIANT R36c1 res_w_preload:
// BOD residual mixed (occ~23%, mem_busy~45%, L2~78%, vgpr=72): post
// R15 hoist_x32, free micro-schedule (DOG order / LDS-x) is exhausted —
// TLP already hides latency. Gate_up R35c2 won by fully separating the
// weight MEMORY phase from the DOG VALU phase after x-hoist: sequential
// sc/zp/pk streams for the whole group-quad, then a pure FMA burst.
//
// Hypothesis: same separation helps residual's dual-row weight stream
// (2 rows × 4 groups = 8 headers). Baseline interleaves load+FMA per
// DOG; here load all 8 sc/zp/pk first (row0 g0..g3 then row1 g0..g3 —
// sequential along each row_ptr), then 8 DOG FMAs with identical math
// and acc slots. Value-preserving. Expect VGPR up modestly (hold 8
// headers); if DROPPED-WAVES dominates, this path is closed for residual.

#if defined(__gfx1200__) || defined(__gfx1201__)
// RDNA4-only occupancy probe: force the compiler to target 32 resident
// one-wave blocks (64 VGPR/thread budget) on this low-occupancy kernel.
// The non-gfx12 device code remains byte-identical to the baseline.
__launch_bounds__(32, 32)
#else
__launch_bounds__(32, 16)
#endif
extern "C" __global__ void HIPFIRE_RESIDUAL_KERNEL(
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

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Exact Qwen3.6 A3B decode contract. Making both quads static lets LLVM
    // schedule the two independent row chains across the full K extent and
    // removes the dead generic tail machinery on gfx1151.
    const int groups_per_row = 16;
#else
    const int groups_per_row = K / 256;
#endif
    const long long row_stride = (long long)groups_per_row * 104;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 3;
    const unsigned int row_weight_offset0 = (unsigned int)((long long)row0 * row_stride);
    const unsigned int row_weight_offset1 =
        (unsigned int)((long long)(have_row1 ? row1 : row0) * row_stride);
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // One matrix-wide SRD keeps the dual-row kernel at 94 VGPR / zero scratch.
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 7u)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 3 ) & 7u)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 6 ) & 7u)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 9 ) & 7u) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 12 ) & 7u) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 15 ) & 7u) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 18 ) & 7u) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 21 ) & 7u) + (zp)) * (x7); \
    } while (0)

    #if HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER
    #define HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset) (*(const unsigned int*)(gp))
    #define HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset) (*(const unsigned int*)((gp) + 4))
    #else
    #define HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)
    #define HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, (gp) + 4, (weight_offset) + 4)
    #endif

    #define LOAD_W(gp, weight_offset, sc, zp, pk) do { \
        const unsigned int hA = HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset); \
        const unsigned int hB = HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        (pk) = ((unsigned int)((const unsigned char*)((gp)+8))[boff] | ((unsigned int)((const unsigned char*)((gp)+8))[boff+1] <<8) | ((unsigned int)((const unsigned char*)((gp)+8))[boff+2] <<16)); \
    } while (0)

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Keep the fixed four-quad bound without expanding the body. Full unroll
    // doubles the instruction footprint; factor-two unroll spills private=16
    // and is not admissible for direct gfx1151 PM4 replay.
    #pragma unroll 1
#endif
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist full 32-wide x for this quad (4 groups × 8) via float4 pairs.
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        // Weight MEMORY phase: sequential along row0 then row1 (8 headers).
        float sc0a, zp0a, sc1a, zp1a, sc2a, zp2a, sc3a, zp3a;
        float sc0b, zp0b, sc1b, zp1b, sc2b, zp2b, sc3b, zp3b;
        unsigned int pk0a, pk1a, pk2a, pk3a, pk0b, pk1b, pk2b, pk3b;
        LOAD_W(row_ptr0 + g * 104, row_weight_offset0 + g * 104, sc0a, zp0a, pk0a);
        LOAD_W(row_ptr0 + (g + 1) * 104, row_weight_offset0 + (g + 1) * 104, sc1a, zp1a, pk1a);
        LOAD_W(row_ptr0 + (g + 2) * 104, row_weight_offset0 + (g + 2) * 104, sc2a, zp2a, pk2a);
        LOAD_W(row_ptr0 + (g + 3) * 104, row_weight_offset0 + (g + 3) * 104, sc3a, zp3a, pk3a);
        LOAD_W(row_ptr1 + g * 104, row_weight_offset1 + g * 104, sc0b, zp0b, pk0b);
        LOAD_W(row_ptr1 + (g + 1) * 104, row_weight_offset1 + (g + 1) * 104, sc1b, zp1b, pk1b);
        LOAD_W(row_ptr1 + (g + 2) * 104, row_weight_offset1 + (g + 2) * 104, sc2b, zp2b, pk2b);
        LOAD_W(row_ptr1 + (g + 3) * 104, row_weight_offset1 + (g + 3) * 104, sc3b, zp3b, pk3b);

        // Pure FMA phase: same dual-row DOG order and acc slots as baseline.
        DOG_X8(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);

        DOG_X8(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);

        DOG_X8(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);

        DOG_X8(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
        DOG_X8(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 104, row_weight_offset0 + g * 104, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 104, row_weight_offset1 + g * 104, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 104, row_weight_offset0 + g * 104, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 104, row_weight_offset1 + g * 104, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 104, row_weight_offset0 + g * 104, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 104, row_weight_offset1 + g * 104, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_W
    #undef HIPFIRE_RESIDUAL_LOAD_SC
    #undef HIPFIRE_RESIDUAL_LOAD_ZP
    #undef DOG_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += acc;
            yy.y += bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += acc;
        }
    }
}
