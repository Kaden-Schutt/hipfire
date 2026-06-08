// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Batched (block-parallel) LFM2 gated short-conv: forward, state-advance, and
// backward. Processes Bb block positions of ONE sequence in parallel, seeded by
// conv_state (the K-1 bx history from the committed prefix). Used by the DFlash
// LFM2 draft forward (a block of B positions) and by the trainer.
//
// Forward (per output position j, channel c):
//   bx_p   = B_gate_p * xin_p                         (independent per pos — NO recurrence)
//   conv_j = sum_{t=0..K-1} bx_{j-(K-1)+t} * weight[c*K+t]   (weight[0]=oldest)
//   out_j  = C_gate_j * conv_j
// Window positions p<0 read conv_state[c, (K-1)+p] (oldest at idx 0).
//
// bcx layout: [Bb, 3*C] = (B_gate | C_gate | xin) per position. conv_state is
// READ-ONLY in the forward/backward; conv1d_gated_state_advance_f32 writes the
// post-block state (race-free, separate launch).

#include <hip/hip_runtime.h>

#define LFM2_CONV_KMAX 8

// ───────────────────────────── forward ──────────────────────────────
extern "C" __global__ void conv1d_gated_batched_f32(
    const float* __restrict__ bcx,    // [Bb, 3*C]
    const float* __restrict__ state,  // [C, K-1]  prefix history (oldest..newest), RO
    const float* __restrict__ weight, // [C, K]
    float* __restrict__ out_y,        // [Bb, C]
    int Bb, int C, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Bb * C) return;
    int j = idx / C, c = idx % C;
    const int hist = K - 1;
    float c_gate = bcx[(long long)j * (3 * C) + C + c];
    float acc = 0.0f;
    for (int t = 0; t < K; t++) {
        int p = j - hist + t;
        float bx;
        if (p >= 0) { const float* r = bcx + (long long)p * (3 * C); bx = r[c] * r[2 * C + c]; }
        else        { bx = state[(long long)c * hist + (hist + p)]; }
        acc += bx * weight[c * K + t];
    }
    out_y[idx] = c_gate * acc;
}

// ─────────────────────────── state advance ──────────────────────────
extern "C" __global__ void conv1d_gated_state_advance_f32(
    const float* __restrict__ bcx,       // [Bb, 3*C]
    const float* __restrict__ state_in,  // [C, K-1]  (read only when Bb < K-1)
    float* __restrict__ state_out,       // [C, K-1]
    int Bb, int C, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hist = K - 1;
    if (idx >= C * hist) return;
    int c = idx / hist, t = idx % hist;
    int p = Bb - hist + t;
    float bx;
    if (p >= 0) { const float* r = bcx + (long long)p * (3 * C); bx = r[c] * r[2 * C + c]; }
    else        { bx = state_in[(long long)c * hist + (hist + p)]; }
    state_out[(long long)c * hist + t] = bx;
}

// ───────────────────── backward 1: d_conv + d_C_gate ─────────────────
// per (j,c): recompute conv_j; d_C_gate_j = g*conv_j → d_bcx; d_conv = g*C_gate.
extern "C" __global__ void conv1d_gated_bwd_dconv_f32(
    const float* __restrict__ bcx,    // [Bb,3C]
    const float* __restrict__ state,  // [C,K-1]
    const float* __restrict__ weight, // [C,K]
    const float* __restrict__ g_out,  // [Bb,C]  upstream grad
    float* __restrict__ d_conv,       // [Bb,C]  out (intermediate)
    float* __restrict__ d_bcx,        // [Bb,3C] out (C_gate slice)
    int Bb, int C, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Bb * C) return;
    int j = idx / C, c = idx % C;
    const int hist = K - 1;
    float c_gate = bcx[(long long)j * (3 * C) + C + c];
    float conv = 0.0f;
    for (int t = 0; t < K; t++) {
        int p = j - hist + t;
        float bx;
        if (p >= 0) { const float* r = bcx + (long long)p * (3 * C); bx = r[c] * r[2 * C + c]; }
        else        { bx = state[(long long)c * hist + (hist + p)]; }
        conv += bx * weight[c * K + t];
    }
    float g = g_out[idx];
    d_conv[idx] = g * c_gate;
    d_bcx[(long long)j * (3 * C) + C + c] = g * conv;
}

// ───────────── backward 2: d_bx → d_B_gate/d_xin (block) | d_state (prefix) ──
// grid = (Bb+hist)*C; thread pp in [0,Bb+hist) → window pos p = pp-hist in [-hist,Bb).
extern "C" __global__ void conv1d_gated_bwd_dbx_f32(
    const float* __restrict__ bcx,    // [Bb,3C]
    const float* __restrict__ weight, // [C,K]
    const float* __restrict__ d_conv, // [Bb,C]
    float* __restrict__ d_bcx,        // [Bb,3C] out (B_gate + xin slices)
    float* __restrict__ d_state,      // [C,K-1] out (prefix bx grads)
    int Bb, int C, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hist = K - 1;
    if (idx >= (Bb + hist) * C) return;
    int pp = idx / C, c = idx % C;
    int p = pp - hist;                         // window pos in [-hist, Bb)
    int j0 = p > 0 ? p : 0;
    int j1 = p + hist; if (j1 > Bb - 1) j1 = Bb - 1;
    float dbx = 0.0f;
    for (int j = j0; j <= j1; j++)
        dbx += d_conv[(long long)j * C + c] * weight[c * K + (p - j + hist)];
    if (p >= 0) {
        const float* r = bcx + (long long)p * (3 * C);
        d_bcx[(long long)p * (3 * C) + c]         = dbx * r[2 * C + c]; // d_B_gate = d_bx*xin
        d_bcx[(long long)p * (3 * C) + 2 * C + c] = dbx * r[c];         // d_xin   = d_bx*B_gate
    } else {
        d_state[(long long)c * hist + (hist + p)] = dbx;
    }
}

// ───────────────────── backward 3: d_weight ──────────────────────────
// per (c,t): d_weight[c,t] = sum_j d_conv[j,c] * bx_{j-hist+t}
extern "C" __global__ void conv1d_gated_bwd_dweight_f32(
    const float* __restrict__ bcx,    // [Bb,3C]
    const float* __restrict__ state,  // [C,K-1]
    const float* __restrict__ d_conv, // [Bb,C]
    float* __restrict__ d_weight,     // [C,K]
    int Bb, int C, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * K) return;
    int c = idx / K, t = idx % K;
    const int hist = K - 1;
    float dw = 0.0f;
    for (int j = 0; j < Bb; j++) {
        int p = j - hist + t;
        float bx;
        if (p >= 0) { const float* r = bcx + (long long)p * (3 * C); bx = r[c] * r[2 * C + c]; }
        else        { bx = state[(long long)c * hist + (hist + p)]; }
        dw += d_conv[(long long)j * C + c] * bx;
    }
    d_weight[idx] = dw;
}
