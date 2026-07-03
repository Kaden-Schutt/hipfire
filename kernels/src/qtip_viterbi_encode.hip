// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// GPU QTIP trellis ENCODER (offline). Full Viterbi over the 12-bit state space,
// one thread-block per 256-weight group. The CPU encoder (beam_encode_group_bits)
// is the offline bottleneck (~1h/1B @ 25 cores); this replaces it with a
// massively-parallel GPU pass.
//
// Key design choice vs the docs/todo beam plan: STATE_BITS=12 → only 4096 states,
// small enough to enumerate FULLY. So instead of beam-search + a hard per-position
// top-k prune, run exact Viterbi: dp[4096] per position, each new state's cost =
// d² + min over its 2^bits predecessors. No top-k primitive, and OPTIMAL (≥ beam
// quality). dp ping-pongs in LDS (2×4096×4 = 32 KB); backpointers (predecessor
// selector, u8) stream to global scratch; a sequential backtrack recovers symbols.
//
// Transition (matches decode_group_bits): s_new = ((s_prev<<bits)|sym) & 0xFFF, so
//   sym       = s_new & (2^bits-1)
//   the 2^bits predecessors of s_new are s_prev = (h<<(12-bits)) | (s_new>>bits),
//     h in [0, 2^bits)   (h is the predecessor's high `bits` bits, lost in the shift)
// The emission cost at s_new is (w[pos] - scale·codebook[s_new])² — depends only on
// s_new, so it's added once per state. Codebook = the computed 1MAD hash + baked
// renorm, identical to gemv_qtip3g256 / qtip::build_codebook.
//
// Encode scale = per-group RMS (group_scale seed), computed in-kernel; the caller
// refines to the closed-form optimal_scale after (as the CPU pipeline does).
//
// blockDim MUST be 256 (the RMS reduction + weight staging assume it).

#include <hip/hip_runtime.h>

#define QTIP_MEAN     (-0.0071028476f)
#define QTIP_INV_STD  (0.9959345134f)
#define QTIP_NSTATES  4096
#define QTIP_BIG      1e30f
// 12-bit rotate. `dp` is indexed by ROTL(state, bits) so a state's 2^bits
// predecessors (which differ in s_prev's top `bits` bits) map to contiguous LDS
// slots in distinct banks, killing the stride-2^(12-bits) 16-way bank conflict.
// hs = 12 - bits (passed to avoid recompute).
#define QTIP_ROTL(x, b, hs) (((unsigned)((x) << (b)) | (unsigned)((x) >> (hs))) & 0xFFFu)
#define QTIP_ROTR(x, b, hs) (((unsigned)((x) >> (b)) | (unsigned)((x) << (hs))) & 0xFFFu)

// state (12-bit) -> ~N(0,1) computed codebook. Matches qtip::decode_1mad + renorm.
__device__ __forceinline__ float qtip_decode(unsigned int state) {
    unsigned int x = state & 0xFFFFFFFFu;
    x = x * 34038481u + 76625530u;
    unsigned int bs = (x & 0xFFu) + ((x >> 8) & 0xFFu)
                    + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    float raw = ((float)bs - 510.0f) * (1.0f / 147.800537109375f);
    return (raw - QTIP_MEAN) * QTIP_INV_STD;
}

// W: [n_groups*256] FWHT-rotated weights (row-major groups).
// symbols_out: [n_groups*256] u8 (0..2^bits-1).
// backptr: [n_groups*256*4096] u8 predecessor selector h.
// scales_out: [n_groups] f32 RMS seed scale used for the encode.
extern "C" __global__ __launch_bounds__(256) void qtip_viterbi_encode(
    const float* __restrict__ W,
    unsigned char* __restrict__ symbols_out,
    unsigned char* __restrict__ backptr,
    float* __restrict__ scales_out,
    int n_groups,
    int bits
) {
    const int g = blockIdx.x;
    if (g >= n_groups) return;
    const int tid = threadIdx.x;      // blockDim.x == 256
    const float* w = W + (long long)g * 256;
    unsigned char* bp = backptr + (long long)g * 256 * QTIP_NSTATES;

    __shared__ float dp[2][QTIP_NSTATES]; // 32 KB ping-pong
    __shared__ float wl[256];             // staged group weights
    __shared__ float red[256];            // RMS reduction scratch

    // Stage weights + accumulate sum of squares (one weight per thread).
    float v = w[tid];
    wl[tid] = v;
    red[tid] = v * v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = sqrtf(red[0] / 256.0f);
    if (!(scale > 0.0f)) scale = 1.0f;
    if (tid == 0) scales_out[g] = scale;

    const int numsym = 1 << bits;
    const int hshift = 12 - bits;
    const int symmask = numsym - 1;

    // dp init: only the leading-zero start state is reachable.
    for (int s = tid; s < QTIP_NSTATES; s += 256) dp[0][s] = (s == 0) ? 0.0f : QTIP_BIG;
    __syncthreads();

    int cur = 0;
    for (int pos = 0; pos < 256; pos++) {
        const float wv = wl[pos];
        const int nxt = cur ^ 1;
        for (int s_new = tid; s_new < QTIP_NSTATES; s_new += 256) {
            const float cbv = scale * qtip_decode((unsigned int)s_new);
            const float diff = wv - cbv;
            const float d2 = diff * diff;
            // The 2^bits predecessors of s_new are, in the permuted `dp` layout,
            // the contiguous slots [pbase, pbase+2^bits) — one per LDS bank.
            const int pbase = (s_new >> bits) << bits;
            float best = QTIP_BIG;
            int besth = 0;
            for (int h = 0; h < numsym; h++) {
                const float c = dp[cur][pbase + h];
                if (c < best) { best = c; besth = h; }
            }
            dp[nxt][QTIP_ROTL(s_new, bits, hshift)] = best + d2;
            bp[(long long)pos * QTIP_NSTATES + s_new] = (unsigned char)besth;
        }
        __syncthreads();
        cur = nxt;
    }

    // Backtrack (thread 0): argmin over the permuted final dp → perm index, mapped
    // back to the actual state; then walk predecessors (bp is actual-indexed).
    if (tid == 0) {
        int pidx = 0;
        float best = QTIP_BIG;
        for (int p = 0; p < QTIP_NSTATES; p++) {
            if (dp[cur][p] < best) { best = dp[cur][p]; pidx = p; }
        }
        int state = (int)QTIP_ROTR(pidx, bits, hshift);
        for (int pos = 255; pos >= 0; pos--) {
            symbols_out[(long long)g * 256 + pos] = (unsigned char)(state & symmask);
            const int h = bp[(long long)pos * QTIP_NSTATES + state];
            state = (h << hshift) | (state >> bits);
        }
    }
}
