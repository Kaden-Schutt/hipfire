// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// QTIP-4 GEMV: trellis-coded 4-bit weights, fused on-the-fly decode + matvec.
// The 4-bit sibling of gemv_qtip3g256 — 0.516 B/weight (marginally < MQ4's
// 0.531, with the trellis codebook instead of a uniform grid). Same computed
// 1MAD codebook and STATE_BITS=12 trellis; only the per-symbol width (4 vs 3)
// and the on-disk packing change.
//
// Block layout (132 B/group, produced by qtip::pack_qtip4_group):
//   bytes [0..4)   : f32 per-group scale
//   bytes [4..132) : 128 bytes nibble-packed 4-bit trellis symbols (256 weights),
//                    two symbols per byte (low nibble = even index) — the MQ4/HFQ4
//                    nibble convention.
//
// QTIP decode: value = `scale * codebook[state_i]`, where `state_i` is the
// bitshift-trellis state = the last STATE_BITS(=12) code bits. At 4 bits/symbol
// that is the **3** most-recent symbols ending at position i (12/4 = 3), vs the 4
// symbols of the 3-bit kernel. Decode stays embarrassingly parallel: each output
// needs only its own symbol plus the 2 preceding ones. The codebook is COMPUTED
// (QTIP 1MAD hash + baked zero-mean/unit-var renorm), identical to the 3-bit
// kernel and to the offline `qtip::build_codebook` / `decode_group_bits(.., 4)`.
//
// Weights are stored in the FWHT-rotated frame; the caller passes FWHT-rotated x
// (shared `mq_rotate_x` path), so the dot product is taken in the rotated basis.
//
// Per-thread work: 8 weights / thread × 32 threads (wave32) = 256 weights/block.

#include <hip/hip_runtime.h>

// build_codebook() renorm affine over the 2^12 states (computed in Rust, baked);
// identical to gemv_qtip3g256 — the codebook depends only on STATE_BITS.
#define QTIP_MEAN     (-0.0071028476f)
#define QTIP_INV_STD  (0.9959345134f)
#define QTIP_STATE_MASK 0xFFFu   // 12-bit trellis state

// QTIP "1MAD" computed codebook: state (12-bit) -> ~N(0,1). Matches
// qtip::decode_1mad exactly, then the baked affine renormalizes.
__device__ __forceinline__ float qtip_decode(unsigned int state) {
    unsigned int x = state & 0xFFFFFFFFu;
    x = x * 34038481u + 76625530u;           // wrapping mul-add (u32)
    unsigned int bs = (x & 0xFFu) + ((x >> 8) & 0xFFu)
                    + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    float raw = ((float)bs - 510.0f) * (1.0f / 147.800537109375f);
    return (raw - QTIP_MEAN) * QTIP_INV_STD;
}

// state_i = (s[i-2]<<8 | s[i-1]<<4 | s[i]) & 0xFFF — the last 3 symbols ending at
// i (same recurrence as decode_group_bits with bits=4; leading positions use 0).
#define ST4(a,b,c) ((((a)<<8)|((b)<<4)|(c)) & QTIP_STATE_MASK)

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_qtip4g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 132;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 132;
        const float scale = *reinterpret_cast<const float*>(gptr);
        const unsigned char* data = (const unsigned char*)(gptr + 4);

        // Own 32-bit symbol window (8 nibbles): symbol j = (pk >> 4j) & 0xf.
        const int boff = tid * 4;
        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16)
                        | ((unsigned int)data[boff + 3] << 24);

        // 2 preceding symbols (global tid*8-2, -1) = nibbles 6,7 of the previous
        // thread's chunk. For tid==0 they are leading zeros (decoder starts at 0).
        unsigned int s_m2 = 0u, s_m1 = 0u;
        if (tid > 0) {
            const int pboff = (tid - 1) * 4;
            unsigned int ppk = (unsigned int)data[pboff]
                             | ((unsigned int)data[pboff + 1] << 8)
                             | ((unsigned int)data[pboff + 2] << 16)
                             | ((unsigned int)data[pboff + 3] << 24);
            s_m2 = (ppk >> 24) & 0xfu; // nibble 6 of prev chunk (pos-2)
            s_m1 = (ppk >> 28) & 0xfu; // nibble 7 of prev chunk (pos-1)
        }

        unsigned int q0 = pk & 0xfu;
        unsigned int q1 = (pk >> 4) & 0xfu;
        unsigned int q2 = (pk >> 8) & 0xfu;
        unsigned int q3 = (pk >> 12) & 0xfu;
        unsigned int q4 = (pk >> 16) & 0xfu;
        unsigned int q5 = (pk >> 20) & 0xfu;
        unsigned int q6 = (pk >> 24) & 0xfu;
        unsigned int q7 = (pk >> 28) & 0xfu;

        unsigned int st0 = ST4(s_m2, s_m1, q0);
        unsigned int st1 = ST4(s_m1, q0,  q1);
        unsigned int st2 = ST4(q0,   q1,  q2);
        unsigned int st3 = ST4(q1,   q2,  q3);
        unsigned int st4 = ST4(q2,   q3,  q4);
        unsigned int st5 = ST4(q3,   q4,  q5);
        unsigned int st6 = ST4(q4,   q5,  q6);
        unsigned int st7 = ST4(q5,   q6,  q7);

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

// Residual variant: y[row] += W·x (o_proj/down_proj into the residual stream).
// Identical decode; only the final store accumulates.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_qtip4g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 132;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 132;
        const float scale = *reinterpret_cast<const float*>(gptr);
        const unsigned char* data = (const unsigned char*)(gptr + 4);

        const int boff = tid * 4;
        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16)
                        | ((unsigned int)data[boff + 3] << 24);

        unsigned int s_m2 = 0u, s_m1 = 0u;
        if (tid > 0) {
            const int pboff = (tid - 1) * 4;
            unsigned int ppk = (unsigned int)data[pboff]
                             | ((unsigned int)data[pboff + 1] << 8)
                             | ((unsigned int)data[pboff + 2] << 16)
                             | ((unsigned int)data[pboff + 3] << 24);
            s_m2 = (ppk >> 24) & 0xfu;
            s_m1 = (ppk >> 28) & 0xfu;
        }

        unsigned int q0 = pk & 0xfu;
        unsigned int q1 = (pk >> 4) & 0xfu;
        unsigned int q2 = (pk >> 8) & 0xfu;
        unsigned int q3 = (pk >> 12) & 0xfu;
        unsigned int q4 = (pk >> 16) & 0xfu;
        unsigned int q5 = (pk >> 20) & 0xfu;
        unsigned int q6 = (pk >> 24) & 0xfu;
        unsigned int q7 = (pk >> 28) & 0xfu;

        unsigned int st0 = ST4(s_m2, s_m1, q0);
        unsigned int st1 = ST4(s_m1, q0,  q1);
        unsigned int st2 = ST4(q0,   q1,  q2);
        unsigned int st3 = ST4(q1,   q2,  q3);
        unsigned int st4 = ST4(q2,   q3,  q4);
        unsigned int st5 = ST4(q3,   q4,  q5);
        unsigned int st6 = ST4(q4,   q5,  q6);
        unsigned int st7 = ST4(q5,   q6,  q7);

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

    if (tid == 0) y[row] += acc;
}
