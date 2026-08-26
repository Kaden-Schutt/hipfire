// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1100 (RDNA3) MQ2G256-Lloyd GEMV with fused residual add: y[row] += A[row] · x.
// Same K4 unroll + LDS-resident codebook structure as the MQ2-Lloyd MoE
// gate_up kernel (gemv_mq2g256_lloyd_moe_gate_up_indexed.hip); only the final
// write differs (+= into y[row] instead of the gate/up split store). Mirrors
// gemv_mq3g256_lloyd_residual.gfx1100.hip one-for-one, which is the same
// K4+LDS treatment for the 3-bit format.
//
// Rationale (inherited from the MQ3 gfx1100 sibling): 4 group accumulators
// break the FMA dependency chain, and staging the codebook in LDS replaces the
// baseline's divergent ternary decision tree with one ds_read_b32 per weight.
// Used by `weight_gemv_residual`'s MQ2-Lloyd arm to eliminate the
// alloc + gemv + add_inplace_f32 + free fallback chain.
//
// Differences vs the MQ3-Lloyd gfx1100 residual — the ONLY things that change
// between the two formats:
//   * group stride 112 B -> 72 B (so the quad steps are +72/+144/+216)
//   * codebook 8 × fp16 (16 B) -> 4 × fp16 (8 B); the packed-index region
//     starts at gptr + 8, not gptr + 16
//   * LDS table 32 slots (4 groups × 8) -> 16 slots (4 groups × 4); group g's
//     codebook lives at cb_lds[g*4 .. g*4+4], so the per-accumulator LDS bases
//     are 0/4/8/12 instead of 0/8/16/24
//   * the cooperative load needs an `if (tid < 16)` guard (MQ3's 32-slot table
//     is filled by all 32 lanes unguarded); the tail load guard is
//     `if (tid < 4)` instead of `if (tid < 8)`. Without the guard, lanes 16..31
//     would compute (tid>>2) ∈ 4..7 and read past the 4-group window.
//   * index width 3 bit -> 2 bit: 64 B of indices per 256-group instead of
//     96 B, so each thread owns 2 bytes (boff = tid*2) instead of 3, `pk` is a
//     16-bit LE assembly instead of 24-bit, and the 8 field shifts are
//     0,2,4,6,8,10,12,14 with mask 3u instead of 0,3,…,21 with mask 7u. There
//     is no cross-byte straddle at 2 bpw.
// Everything else — grid [M,1,1] / block [32,1,1], the (long long) row stride
// cast, x indexing (g*256 + tid*8, ascending), the quads/tail split with tail
// groups folding into acc0/acc1/acc2, the `(acc0+acc1)+(acc2+acc3)` fold, the
// 5-step wave32 __shfl_down reduction, and the tid==0 guard — is identical.
//
// launch_bounds(32, 20) matches the rest of the MQ2-Lloyd family (MQ3-Lloyd
// targets 16). RDNA3 caps at 16 waves/SIMD, so hipcc emits a benign
// "failed to meet occupancy target" -Wpass-failed warning here, exactly as the
// existing MQ2-Lloyd kernels already do. Occupancy is a perf knob only.
//
// NOTE: the K4 reassociation means this variant is NOT bit-identical to the
// single-accumulator baseline gemv_mq2g256_lloyd_residual.hip (FMA-order
// epsilon) — the same caveat the MQ3-Lloyd gfx1100 pair carries.
//
// X must be FWHT-256 pre-rotated by the caller; this kernel never rotates.
//
// Block layout (72 B / 256 weights):
//   bytes [0..8)   : 4 × fp16 codebook entries (sorted ascending at quant time)
//   bytes [8..72)  : 64 B packed 2-bit indices, 4 per byte, little-endian
//                    within the byte — byte i holds weights 4i..4i+4 at bit
//                    offsets 0,2,4,6. Thread tid owns bytes 2*tid, 2*tid+1 =
//                    weights 8*tid..8*tid+8 = x[base .. base+8].
//
// LDS layout: 16 fp32 entries (64 B). Per quad iteration:
//   cb_lds[ 0.. 3] = group 0's 4 codebook entries (fp16 → fp32)
//   cb_lds[ 4.. 7] = group 1's
//   cb_lds[ 8..11] = group 2's
//   cb_lds[12..15] = group 3's
// Cooperative load: thread tid<16 loads entry (tid & 3) of group (tid >> 2)
// from gp0 + (tid >> 2) * 72 + (tid & 3) * 2.
//
// fp32 in LDS (not fp16): 4 fp32 entries span 4 distinct banks (32 banks × 4 B)
// so bank pressure is trivial; 64 B is nothing on 64 KB LDS.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 72;
    const int boff = tid * 2;  // 2 bytes of indices per group per thread

    // 16-slot LDS codebook: 4 groups × 4 entries.
    __shared__ float cb_lds[16];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // ----- main K4 loop -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;

        // Cooperative load: 16 threads × 1 fp16 → 16 fp32 entries in LDS.
        // thread tid<16 → group (tid >> 2), entry (tid & 3).
        if (tid < 16) {
            const __half* cb_h =
                (const __half*)(gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb_h[tid & 3]);
        }
        __syncthreads();

        // Load 4 packed-uint16 indices (one per group, this thread's 2 bytes).
        const unsigned char* d0 = (const unsigned char*)(gp0 + 8) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 8) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 8) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 8) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8);

        int base = g * 256 + tid * 8;

        #define DOG2_LDS(pk, lds, b, a)                                  \
            (a) += cb_lds[(lds) + ( (pk)        & 3u)] * x[(b)]          \
                 + cb_lds[(lds) + (((pk) >>  2) & 3u)] * x[(b) + 1]      \
                 + cb_lds[(lds) + (((pk) >>  4) & 3u)] * x[(b) + 2]      \
                 + cb_lds[(lds) + (((pk) >>  6) & 3u)] * x[(b) + 3]      \
                 + cb_lds[(lds) + (((pk) >>  8) & 3u)] * x[(b) + 4]      \
                 + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x[(b) + 5]      \
                 + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x[(b) + 6]      \
                 + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x[(b) + 7]

        DOG2_LDS(pk0,  0, base,        acc0);
        DOG2_LDS(pk1,  4, base + 256,  acc1);
        DOG2_LDS(pk2,  8, base + 512,  acc2);
        DOG2_LDS(pk3, 12, base + 768,  acc3);
        #undef DOG2_LDS
    }

    // ----- tail loop (1, 2, or 3 leftover groups) -----
    // Each tail iteration reuses cb_lds[0..3] for one group's codebook.
    // Only the first 4 lanes write; barrier; everyone reads. Tail group i
    // lands in acc[(quads*4 + i) & 3] = acc[i] (since quads*4 % 4 == 0).
    #define TAIL_LOAD_AND_DOT(g_expr, acc) do {                                 \
        const int g = (g_expr);                                                  \
        const char* gp = row_ptr + g * 72;                                       \
        if (tid < 4) {                                                           \
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);                \
        }                                                                        \
        __syncthreads();                                                         \
        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;          \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8);        \
        int base = g * 256 + tid * 8;                                            \
        (acc) += cb_lds[ (pk)        & 3u] * x[base]                             \
               + cb_lds[((pk) >>  2) & 3u] * x[base + 1]                         \
               + cb_lds[((pk) >>  4) & 3u] * x[base + 2]                         \
               + cb_lds[((pk) >>  6) & 3u] * x[base + 3]                         \
               + cb_lds[((pk) >>  8) & 3u] * x[base + 4]                         \
               + cb_lds[((pk) >> 10) & 3u] * x[base + 5]                         \
               + cb_lds[((pk) >> 12) & 3u] * x[base + 6]                         \
               + cb_lds[((pk) >> 14) & 3u] * x[base + 7];                        \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2,       acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1,  acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2,  acc2);
    #undef TAIL_LOAD_AND_DOT

    // Wave reduction.
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
