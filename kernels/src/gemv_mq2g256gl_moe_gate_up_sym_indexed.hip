// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for MQ2-GL, SYMMETRIC-codebook variant.
// Perf rewrite of gemv_mq2g256gl_moe_gate_up_k8_indexed (same format, same
// SoA blob, same grid/block, same output split) — only the inner-loop math
// and the codebook kernel-arg count change.
//
// ---------------------------------------------------------------------------
// Why this exists
// ---------------------------------------------------------------------------
// The stock GL kernel reads 11% fewer weight bytes than the per-block
// MQ2-Lloyd kernel (64 B/group vs 72 B/group) yet measured 3.1% SLOWER. Cause:
// it parks the 4-entry global codebook in LDS and issues one ds_read_b32 PER
// WEIGHT (8 per group per thread). On a latency-bound decode GEMV that is a
// memory-dependent chain the scheduler cannot hide — the saved DRAM bytes are
// paid back at the LDS pipe.
//
// Fix: make the codebook EXACTLY SYMMETRIC and stop looking it up at all.
// Measured 2026-08-04 on 28.3M real a3b post-FWHT expert weights: the fitted
// global codebook costs +2.35% NRMSE vs per-block Lloyd; forcing it to the
// exact-symmetric textbook Lloyd–Max Gaussian levels costs +2.37% — a 0.02%
// NRMSE delta for the whole trick. (GL_CB2 is already symmetric to 4 decimals:
// [-1.5104, -0.4528, +0.4528, +1.5104].)
//
// With c = {-b, -a, +a, +b} the group dot product factors by MAGNITUDE CLASS:
//
//     sum_i c[q_i] * x_i  =  a * sum(sign_i * x_i | |c[q_i]| = a)
//                         +  b * sum(sign_i * x_i | |c[q_i]| = b)
//
// so the inner loop does NO codebook load and NO multiply: one XOR into x's
// sign bit, then an add into one of two accumulators picked by the magnitude
// bit. Two multiplies per 256-block (one per class) replace eight FMAs plus
// eight ds_reads per group.
//
// ---------------------------------------------------------------------------
// Bit algebra — DERIVED FROM THE ENCODER, read this before touching the macro
// ---------------------------------------------------------------------------
// crates/hipfire-quantize/src/main.rs::gl_encode_block picks the nearest
// neighbour in the ASCENDING array GL_CB2, so the code→value map is:
//
//     q  bits(b1,b0)   value    sign      magnitude
//     0     0 0         -b     negative       b
//     1     0 1         -a     negative       a
//     2     1 0         +a     positive       a
//     3     1 1         +b     positive       b
//
//   * SIGN  is bit1: negate iff b1 == 0.  (NOT "sign = bit1 means negative" —
//     bit1 SET is the positive half.)
//   * MAGNITUDE is the GRAY bit b0 ^ b1: |c| = a iff (b0 ^ b1) == 1.
//     It is NOT bit0. q=0 (b0=0) is the LARGE level and q=1 (b0=1) is the
//     small one; a plain bit0 test swaps a and b on exactly half the codes and
//     produces silent garbage that still looks numerically plausible.
//
// Both masks fall out of the packed 16-bit word with constant shifts:
//   ns = ~pk                → bit(2j+1) of ns is "negate weight j"
//   gr = pk ^ (pk >> 1)     → bit(2j)   of gr is "|c| = a for weight j"
// and (2j+1) + (30 - 2j) == 31, so `(ns << (30-2j)) & 0x80000000` lands the
// negate flag directly on the IEEE-754 sign bit. XOR-ing that into
// __float_as_uint(x) is an exact negation for every finite x (and for ±0 and
// denormals) — no rounding, no v_sub.
//
// The class split uses an AND mask rather than a branch or a cndmask pair:
//   ta = xi & am        → sign-applied x when |c| = a, else +0.0f
//   tb = xi ^ ta        → sign-applied x when |c| = b, else +0.0f
// Adding a bit-pattern +0.0f is exact, so nothing is perturbed by the split.
//
// The per-block fp16 scale still factors out of the whole group exactly as in
// the stock GL kernel — it multiplies BOTH class sums (w = scale * c[q]), so
// it stays one multiply per class per group and never enters the inner loop.
//
// ---------------------------------------------------------------------------
// NOT BIT-IDENTICAL to gemv_mq2g256gl_moe_gate_up_k8_indexed
// ---------------------------------------------------------------------------
// Same value, different summation order, so expect FMA-order epsilon:
//   * the reference sums 8 terms c[q_j]*x_j left-to-right into one sub-total;
//     here the 8 terms are partitioned into two class sums, each summed
//     left-to-right, and the codebook constant is applied ONCE at the very end
//     (a*(x0+x3+...) instead of (a*x0)+(a*x3)+...).
//   * the reference folds the scale as `acc += s * sub` (one FMA per group);
//     here it is two FMAs per group, one per class, and cb_a/cb_b multiply the
//     fully-reduced per-thread totals.
// Do not use this kernel for a byte-parity check against the stock GL kernel.
// It is also not numerically identical to the FITTED codebook — by design, the
// caller must pass the symmetrized levels.
//
// Kernel args differ from the stock GL kernel: TWO scalar floats (cb_a, cb_b,
// the POSITIVE half-levels, 0 < cb_a < cb_b) instead of four. They occupy the
// same slot — after every pointer, before M and K.
//
// Expert blob layout (SoA, unchanged from the stock GL kernel):
//   [0 .. M*gpr*64)                     : packed 2-bit indices, 64 B/group
//   [M*gpr*64 .. M*gpr*64 + M*gpr*2)    : fp16 per-block scales, row-major
// Index packing matches MQ2-Lloyd: 4 codes/byte, little-endian, so thread tid
// owns bytes [2*tid, 2*tid+1] == codes [8*tid .. 8*tid+8) of the group and
// pairs code j with x_rot[g*256 + tid*8 + j].
//
// X must be FWHT-pre-rotated by the caller — identical convention to every
// other MQ*/MFP* kernel; nothing is rotated here.
//
// Grid: [M, k_top, 1], block [32]. Output split: row<M/2 → y_gate, ≥M/2 → y_up.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256gl_moe_gate_up_sym_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x_rot,                     // [K] (FWHT-rotated)
    float* __restrict__ y_gate,                          // [k_top × mi]
    float* __restrict__ y_up,                            // [k_top × mi]
    float cb_a, float cb_b,                              // symmetric half-levels
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;

    // No LDS, no __syncthreads(), no codebook staging at all: cb_a/cb_b are
    // kernel-arg SGPRs and are touched exactly twice, after the K loop.
    const char* idx_base   = A + (long long)row * groups_per_row * 64;
    const __half* scale_base = reinterpret_cast<const __half*>(
        A + (long long)M * groups_per_row * 64) + (long long)row * groups_per_row;

    const int boff = tid * 2;  // 2 bytes of indices per group per thread

    // Eight independent accumulator chains: 4 K-unrolled groups × 2 magnitude
    // classes. accA* collect sign-applied, scale-applied x for |c| = cb_a.
    float accA0 = 0.0f, accA1 = 0.0f, accA2 = 0.0f, accA3 = 0.0f;
    float accB0 = 0.0f, accB1 = 0.0f, accB2 = 0.0f, accB3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // One weight: XOR the negate flag onto x's sign bit, then route it to the
    // class-a or class-b partial with an AND mask. J is a literal so every
    // shift folds at compile time.
    #define SPLIT1(ns_, gr_, b_, J, sa, sb) do {                              \
        unsigned int xi_ = __float_as_uint(x_rot[(b_) + (J)]);                \
        xi_ ^= (((ns_) << (30 - 2 * (J))) & 0x80000000u);                     \
        const unsigned int am_ = 0u - (((gr_) >> (2 * (J))) & 1u);            \
        const unsigned int ta_ = xi_ & am_;                                   \
        (sa) += __uint_as_float(ta_);                                         \
        (sb) += __uint_as_float(xi_ ^ ta_);                                   \
    } while (0)

    // One 256-group's worth of this thread's 8 weights → two class partials.
    #define SPLIT8(pk, b_, sa, sb) do {                                       \
        const unsigned int ns_ = ~(pk);            /* bit(2j+1): negate  */   \
        const unsigned int gr_ = (pk) ^ ((pk) >> 1); /* bit(2j): |c|=a   */   \
        (sa) = 0.0f; (sb) = 0.0f;                                             \
        SPLIT1(ns_, gr_, b_, 0, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 1, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 2, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 3, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 4, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 5, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 6, sa, sb);                                      \
        SPLIT1(ns_, gr_, b_, 7, sa, sb);                                      \
    } while (0)

    // ----- main K4 loop — no barrier, no LDS, no multiply per weight -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const unsigned char* d0 = (const unsigned char*)(idx_base + (g    ) * 64) + boff;
        const unsigned char* d1 = (const unsigned char*)(idx_base + (g + 1) * 64) + boff;
        const unsigned char* d2 = (const unsigned char*)(idx_base + (g + 2) * 64) + boff;
        const unsigned char* d3 = (const unsigned char*)(idx_base + (g + 3) * 64) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8);

        const float s0 = __half2float(scale_base[g]);
        const float s1 = __half2float(scale_base[g + 1]);
        const float s2 = __half2float(scale_base[g + 2]);
        const float s3 = __half2float(scale_base[g + 3]);

        const int base = g * 256 + tid * 8;
        float pa0, pb0, pa1, pb1, pa2, pb2, pa3, pb3;
        SPLIT8(pk0, base,       pa0, pb0);
        SPLIT8(pk1, base + 256, pa1, pb1);
        SPLIT8(pk2, base + 512, pa2, pb2);
        SPLIT8(pk3, base + 768, pa3, pb3);

        // Scale factors out of the whole group: w = s * c[q], and c[q] is
        // ±cb_a / ±cb_b, so s multiplies both class sums identically.
        accA0 += s0 * pa0;  accB0 += s0 * pb0;
        accA1 += s1 * pa1;  accB1 += s1 * pb1;
        accA2 += s2 * pa2;  accB2 += s2 * pb2;
        accA3 += s3 * pa3;  accB3 += s3 * pb3;
    }

    // ----- tail (1, 2, or 3 leftover groups) -----
    #define TAIL(g_expr, accA, accB) do {                                      \
        const int g = (g_expr);                                                \
        const unsigned char* d = (const unsigned char*)(idx_base + g * 64) + boff; \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8);      \
        const float s = __half2float(scale_base[g]);                           \
        const int base = g * 256 + tid * 8;                                    \
        float pa, pb; SPLIT8(pk, base, pa, pb);                                \
        (accA) += s * pa;  (accB) += s * pb;                                   \
    } while (0)

    if (tail >= 1) TAIL( quads << 2,      accA0, accB0);
    if (tail >= 2) TAIL((quads << 2) + 1, accA1, accB1);
    if (tail >= 3) TAIL((quads << 2) + 2, accA2, accB2);
    #undef TAIL
    #undef SPLIT8
    #undef SPLIT1

    // Two multiplies for the whole row-slice: the codebook is applied ONCE,
    // to the fully-folded per-class totals.
    const float sumA = (accA0 + accA1) + (accA2 + accA3);
    const float sumB = (accB0 + accB1) + (accB2 + accB3);
    float acc = cb_a * sumA + cb_b * sumB;

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
