// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// QTIP-3 GEMV: trellis-coded 3-bit weights, fused on-the-fly decode + matvec.
// Phase C2 (NEXT-STEPS.md). The bandwidth lever — 0.391 B/weight (26% < MQ4).
//
// Block layout (100 B/group, produced by qtip::pack_qtip3_group):
//   bytes [0..4)   : f32 per-group scale
//   bytes [4..100) : 96 bytes packed 3-bit trellis symbols (256 weights),
//                    contiguous little-endian bitstream — symbol i sits at
//                    bit 3*i, identical packing to uniform MQ3 (so the byte
//                    layout is shared; only the *meaning* differs).
//
// QTIP decode (the key difference from MQ3/Lloyd, where each weight is an
// independent codebook index): the value is `scale * codebook[state_i]`, where
// `state_i` is the **bitshift-trellis state** = the last STATE_BITS(=12) code
// bits = the 4 most-recent 3-bit symbols ending at position i. Because the
// state is a fixed 4-symbol sliding window, decode is embarrassingly PARALLEL:
// each output only needs its own symbol plus the 3 preceding ones. The codebook
// is COMPUTED (QTIP 1MAD hash + a baked zero-mean/unit-var renorm affine), not
// stored — so there is no per-group table in the weight stream and ~zero LDS.
// These constants make the kernel decode bit-identical to the offline
// `qtip::build_codebook` / `decode_group_bits(.., bits=3)` that was PPL-
// validated (qtip3-sim PPL 15.20).
//
// Weights are stored in the FWHT-rotated frame (incoherence); the caller passes
// FWHT-rotated x (shared `mq_rotate_x` path, as MQ4/MFP4 do), so the dot product
// is taken in the rotated basis — mathematically equivalent to un-rotated.
//
// Per-thread work: 8 weights / thread × 32 threads (wave32) = 256 weights/block.

#include <hip/hip_runtime.h>

// build_codebook() renorm affine over the 2^12 states (computed in Rust, baked):
//   codebook[s] = (decode_1mad_raw(s) - QTIP_MEAN) * QTIP_INV_STD
#define QTIP_MEAN     (-0.0071028476f)
#define QTIP_INV_STD  (0.9959345134f)
#define QTIP_STATE_MASK 0xFFFu   // 12-bit trellis state

// QTIP "1MAD" computed codebook: state (12-bit) -> ~N(0,1) raw value.
// One mul-add + a 4-byte central-limit sum. Matches qtip::decode_1mad exactly
// (same integer constants), then the baked affine renormalizes.
__device__ __forceinline__ float qtip_decode(unsigned int state) {
    unsigned int x = state & 0xFFFFFFFFu;
    x = x * 34038481u + 76625530u;           // wrapping mul-add (u32)
    unsigned int bs = (x & 0xFFu) + ((x >> 8) & 0xFFu)
                    + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    float raw = ((float)bs - 510.0f) * (1.0f / 147.800537109375f);
    return (raw - QTIP_MEAN) * QTIP_INV_STD;
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_qtip3g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 100;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 100;
        const float scale = *reinterpret_cast<const float*>(gptr);
        const unsigned char* data = (const unsigned char*)(gptr + 4);

        // Own 24-bit symbol window (8 symbols): symbol j = (pk >> 3j) & 7.
        const int boff = tid * 3;
        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        // 3 preceding symbols (global tid*8-3,-2,-1) = symbols 5,6,7 of the
        // previous chunk. For tid==0 they are leading zeros (decoder starts at
        // state 0). s_m3 = pos-3, s_m2 = pos-2, s_m1 = pos-1.
        unsigned int s_m3 = 0u, s_m2 = 0u, s_m1 = 0u;
        if (tid > 0) {
            const int pboff = (tid - 1) * 3;
            unsigned int ppk = (unsigned int)data[pboff]
                             | ((unsigned int)data[pboff + 1] << 8)
                             | ((unsigned int)data[pboff + 2] << 16);
            s_m3 = (ppk >> 15) & 7u; // symbol 5 of prev chunk
            s_m2 = (ppk >> 18) & 7u; // symbol 6
            s_m1 = (ppk >> 21) & 7u; // symbol 7
        }

        // Symbols of this chunk.
        unsigned int q0 = pk & 7u;
        unsigned int q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u;
        unsigned int q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u;
        unsigned int q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u;
        unsigned int q7 = (pk >> 21) & 7u;

        // state_i = (s[i-3]<<9 | s[i-2]<<6 | s[i-1]<<3 | s[i]) & 0xFFF — the last
        // 4 symbols ending at i (same recurrence as decode_group_bits).
        #define ST(a,b,c,d) ((((a)<<9)|((b)<<6)|((c)<<3)|(d)) & QTIP_STATE_MASK)
        unsigned int st0 = ST(s_m3, s_m2, s_m1, q0);
        unsigned int st1 = ST(s_m2, s_m1, q0,  q1);
        unsigned int st2 = ST(s_m1, q0,  q1,  q2);
        unsigned int st3 = ST(q0,   q1,  q2,  q3);
        unsigned int st4 = ST(q1,   q2,  q3,  q4);
        unsigned int st5 = ST(q2,   q3,  q4,  q5);
        unsigned int st6 = ST(q3,   q4,  q5,  q6);
        unsigned int st7 = ST(q4,   q5,  q6,  q7);
        #undef ST

        const int base_idx = g * 256 + tid * 8;
        acc += scale * qtip_decode(st0) * x[base_idx]
             + scale * qtip_decode(st1) * x[base_idx + 1]
             + scale * qtip_decode(st2) * x[base_idx + 2]
             + scale * qtip_decode(st3) * x[base_idx + 3]
             + scale * qtip_decode(st4) * x[base_idx + 4]
             + scale * qtip_decode(st5) * x[base_idx + 5]
             + scale * qtip_decode(st6) * x[base_idx + 6]
             + scale * qtip_decode(st7) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
