// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Indexed-MoE low-rank weight-error correction (LQER/CALDERA): per routed
// expert e, add U_e·(V_e·x) to the expert GEMV output, where U_e[M,r]·V_e[r,K]
// is the offline rank-r SVD of the quant error E = W_fp − dequant(W_q). This is
// the "2-WMMA weight treatment" term: it composes with ANY expert format
// (QTIP3/OQ4/…) — run after the main indexed-MoE GEMV. Two stages so V·x (shared
// across the M output rows) is computed once:
//   proj:   t[krank,:] = V_e · x          [k_top × r]
//   expand: out[krank,row] += U_e[row,:]·t[krank,:]
// U,V are per-expert f32 row-major, indexed by a device pointer table (like the
// weight blobs). r is small (16–32); ~1% extra compute over the main GEMV.

// Stage 1: t[krank, i] = Σ_j V_e[i,j]·x_in[j].  Grid (k_top), block 32 (lane i<R).
// `x_stride` selects the input layout:
//   0 → all routed experts share one [K] input vector `x` (gate_up: the routed
//       token's rotated hidden state is the same for every expert).
//   K → each routed slot has its own [K] input row at `x + krank*K` (down: the
//       per-expert silu(gate)·up intermediate differs per expert).
__launch_bounds__(32)
extern "C" __global__ void gemv_lowrank_moe_proj(
    const unsigned long long* __restrict__ expert_v_ptrs, // [n_exp] → V_e[r,K] f32
    const int* __restrict__ topk_indices,                  // [k_top]
    const float* __restrict__ x,                           // [K] or [k_top × K]
    float* __restrict__ t,                                 // [k_top × R]
    int R, int K, int x_stride
) {
    const int krank = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= R) return;
    const int e = topk_indices[krank];
    const float* __restrict__ vrow = reinterpret_cast<const float*>(expert_v_ptrs[e]) + (long long)i * K;
    const float* __restrict__ xin = x + (long long)krank * x_stride;
    float acc = 0.0f;
    for (int j = 0; j < K; j++) acc += vrow[j] * xin[j];
    t[krank * R + i] = acc;
}

// Stage 2: out[krank, row] += Σ_{i<R} U_e[row,i]·t[krank,i].  Grid (M, k_top),
// block 32 (lane i<R, wave-reduce). Accumulates into the existing expert output.
__launch_bounds__(32)
extern "C" __global__ void gemv_lowrank_moe_expand(
    const unsigned long long* __restrict__ expert_u_ptrs, // [n_exp] → U_e[M,r] f32
    const int* __restrict__ topk_indices,                  // [k_top]
    const float* __restrict__ t,                           // [k_top × R]
    float* __restrict__ out,                               // [k_top × M] (+=)
    int M, int R
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int i = threadIdx.x;
    const int e = topk_indices[krank];
    const float* __restrict__ urow = reinterpret_cast<const float*>(expert_u_ptrs[e]) + (long long)row * R;
    float part = (i < R) ? urow[i] * t[krank * R + i] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down(part, off);
    if (i == 0) out[(long long)krank * M + row] += part;
}
