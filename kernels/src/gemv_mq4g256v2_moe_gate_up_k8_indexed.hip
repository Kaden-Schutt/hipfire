// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // __ushort_as_half / __half2float for the qt44 f16 header

// MQ4G256V2 (qt=44) sister of `gemv_hfq4g256_moe_gate_up_k8_indexed`.
//
// Transcribed from that kernel. The staging, tiling, dual gate/up stream,
// wavefront reduction and output layout are identical line for line. The ONLY
// difference is the group header decode.
//
// Why this kernel has to exist: qt44 had no MoE GEMV of any kind, so
// `routed_indexable_*` could never be true for it and a qt44 A3B MoE model
// fell to the resident CPU-fallback decode path instead of the device-indexed
// one.
//
// This is not hypothetical. The PUBLISHED artifact
// hipfire-models/ornith1.5-35b-a3b ships routed experts as qt44 — 20,651 of
// its 21,093 tensors, read from its HFQ index — so the shipped model has been
// decoding through the fallback path.
//
// Group layout, 136 B stride for BOTH formats:
//
//   qt13 MQ4G256      [0..4)  f32 scale (all 256 weights)
//                     [4..8)  f32 min   (all 256 weights)
//                     [8..136) 128 nibble bytes
//
//   qt44 MQ4G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..136) 128 nibble bytes   (identical packing)
//
// Affine-grid selection is LANE-CONSTANT. Thread `tid` reads the u32 at
// `boff = tid*4` within the payload, which covers weights [tid*8, tid*8+8) —
// wholly inside one 128-weight half for every tid, and the same half in every
// group. So the header offset is computed ONCE outside the loops:
//   tid  0..15 → weights   0..127 → grid 0 → header at gp+0
//   tid 16..31 → weights 128..255 → grid 1 → header at gp+4
// This makes the qt44 header read strictly cheaper than qt13's: one 4-byte
// load instead of two.
//
// Dequant math is f32, matching `gemv_mq4g256v2.hip` (the dense qt44 GEMV):
// the f16 scale/zero are widened to float and the accumulation is identical to
// qt13's. Note this differs from the qt44 grouped GEMM, which accumulates the
// dequant in fp16 — the two paths have always had different precision
// conventions and are not expected to be bit-identical to each other.
//
// Degenerate groups: the encoder writes scale bits 0 and all nibbles 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.

// Kernel-name macro: default keeps the incumbent exported symbol. Candidate
// modules redefine HIPFIRE_MQ4V2_GATE_UP_KERNEL before including this TU.
#ifndef HIPFIRE_MQ4V2_GATE_UP_KERNEL
#define HIPFIRE_MQ4V2_GATE_UP_KERNEL gemv_mq4g256v2_moe_gate_up_k8_indexed
#endif

#if defined(HIPFIRE_MQ4V2_GATE_UP_NOLDS)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 16)
#endif
extern "C" __global__ void HIPFIRE_MQ4V2_GATE_UP_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int mi = M >> 1;
    const int row = blockIdx.x;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_MQ4V2_GATE_UP_NOLDS)
    // Candidate exact-shape arm (K=2048 gfx1100): no LDS staging, no barrier.
    // Dead rows (row >= mi) exit before any work; x is read from global only.
    // One-wave/row arithmetic below is unchanged vs the incumbent path.
    if (row >= mi) return;
    const float* __restrict__ xsrc = x;
#else
    // Stage x into LDS before any early-return so all 32 lanes hit the barrier.
    // Cap at 2048 (A3B hidden); larger K falls back to global x (value-identical).
    __shared__ float xs[2048];
    const bool use_lds = (K > 0 && K <= 2048);
    if (use_lds) {
        float4* __restrict__ xs4 = reinterpret_cast<float4*>(xs);
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        if (K == 2048) {
            // n4=512, stride 32 → 16 independent float4 copies per lane.
            xs4[tid + 0 * 32]  = x4[tid + 0 * 32];
            xs4[tid + 1 * 32]  = x4[tid + 1 * 32];
            xs4[tid + 2 * 32]  = x4[tid + 2 * 32];
            xs4[tid + 3 * 32]  = x4[tid + 3 * 32];
            xs4[tid + 4 * 32]  = x4[tid + 4 * 32];
            xs4[tid + 5 * 32]  = x4[tid + 5 * 32];
            xs4[tid + 6 * 32]  = x4[tid + 6 * 32];
            xs4[tid + 7 * 32]  = x4[tid + 7 * 32];
            xs4[tid + 8 * 32]  = x4[tid + 8 * 32];
            xs4[tid + 9 * 32]  = x4[tid + 9 * 32];
            xs4[tid + 10 * 32] = x4[tid + 10 * 32];
            xs4[tid + 11 * 32] = x4[tid + 11 * 32];
            xs4[tid + 12 * 32] = x4[tid + 12 * 32];
            xs4[tid + 13 * 32] = x4[tid + 13 * 32];
            xs4[tid + 14 * 32] = x4[tid + 14 * 32];
            xs4[tid + 15 * 32] = x4[tid + 15 * 32];
        } else {
            const int n4 = K >> 2;
            for (int i = tid; i < n4; i += 32) {
                xs4[i] = x4[i];
            }
        }
        __syncthreads();
    }

    if (row >= mi) return;

    const float* __restrict__ xsrc = use_lds ? xs : x;
#endif

    // Read expert_id from device and index into the pointer table.
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;
    // Lane-constant affine-grid header offset — see the note above.
    const int hoff = (tid < 16) ? 0 : 4;

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(xsrc + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(xsrc + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    #define DOG_X8(gp, a, pfx) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
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

        // Hoist all 32 x values for this group-quad from LDS/global before DOG.
        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);

        // Gate+up dual-stream DOG with all x resident (no mid-quad x reload).
        DOG_X8(gate_ptr + g * 136, gacc0, xa);
        DOG_X8(up_ptr + g * 136, uacc0, xa);
        DOG_X8(gate_ptr + (g + 1) * 136, gacc1, xb);
        DOG_X8(up_ptr + (g + 1) * 136, uacc1, xb);
        DOG_X8(gate_ptr + (g + 2) * 136, gacc2, xc);
        DOG_X8(up_ptr + (g + 2) * 136, uacc2, xc);
        DOG_X8(gate_ptr + (g + 3) * 136, gacc3, xd);
        DOG_X8(up_ptr + (g + 3) * 136, uacc3, xd);
    }

    // Tail: at most 3 groups; sequential load is fine.
    #define LOAD_X8(b) \
        const float4 xv0 = *reinterpret_cast<const float4*>(xsrc + (b)); \
        const float4 xv1 = *reinterpret_cast<const float4*>(xsrc + (b) + 4); \
        const float x0 = xv0.x, x1 = xv0.y, x2 = xv0.z, x3 = xv0.w; \
        const float x4 = xv1.x, x5 = xv1.y, x6 = xv1.z, x7 = xv1.w

    #define DOG_X8_TAIL(gp, a) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
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
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(gate_ptr + g * 136, gacc0);
        DOG_X8_TAIL(up_ptr + g * 136, uacc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(gate_ptr + g * 136, gacc1);
        DOG_X8_TAIL(up_ptr + g * 136, uacc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(gate_ptr + g * 136, gacc2);
        DOG_X8_TAIL(up_ptr + g * 136, uacc2);
    }
    #undef DOG_X8_TAIL
    #undef LOAD_X8
    #undef DOG_X8
    #undef LOAD_X8_INTO

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        gacc += __shfl_down(gacc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        uacc += __shfl_down(uacc, offset);

    if (tid == 0) {
        y_gate[krank * mi + row] = gacc;
        y_up[krank * mi + row] = uacc;
    }
}
