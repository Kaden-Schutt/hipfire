// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Fused unscatter + SwiGLU + clamp for the DeepSeek V4 grouped MoE
// gate_up pipeline. Phase D1 of the prefill catch-up plan.
//
// Replaces the back-to-back pair:
//
//   1) moe_gate_up_unscatter_k8     (scatter [m_total, 2*mi] → gate/up [N, K_TOP, mi])
//   2) deepseek4_silu_mul_clamp_f32 (silu(gate) * up clamped → moe_gate_batch)
//
// with a single per-(slot, m_col) kernel that, for valid slots, decodes
// the (token, k_rank) destination directly from `sorted_slot_index` and
// writes `silu(clamp(gate)) * clamp(up)` into `moe_gate_batch` —
// eliminating the intermediate `moe_up_batch` buffer entirely and one
// kernel launch per routed-MoE layer.
//
// Math per upstream DeepSeek V4 `inference/model.py:Expert.forward` —
// asymmetric clamp:
//   gate = min(gate, +limit)        (one-sided, cap from above only)
//   up   = clamp(up, -limit, +limit) (symmetric)
//   out  = silu(gate) * up
//
// Shapes:
//   Y_grouped         : [m_total × (2*mi)] f32 — rows 0..mi gate, mi..2*mi up
//   sorted_slot_index : [m_total] i32 — flat = token*K_TOP + k_rank, or -1
//   moe_gate_batch    : [N × K_TOP × mi] f32 (output)
//
// Padding slots (sorted_slot_index = -1) are skipped — downstream
// FWHT-rotate only reads valid (token, k_rank) tuples.
//
// Grid:  (ceil(mi/256), m_total)
// Block: [256]
// LDS:   0

#include <hip/hip_runtime.h>

extern "C" __global__ void moe_unscatter_silu_clamp_k8(
    const float* __restrict__ Y_grouped,
    const int*   __restrict__ sorted_slot_index,
    float*       __restrict__ moe_gate_batch,
    int mi, int K_TOP, int m_total,
    float swiglu_limit
) {
    const int slot = blockIdx.y;
    if (slot >= m_total) return;
    const int flat = sorted_slot_index[slot];
    if (flat < 0) return; // padding slot — downstream does not consume.

    const int token = flat / K_TOP;
    const int krank = flat - token * K_TOP;

    const int m_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_col >= mi) return;

    const long long y_grp_base = (long long)slot * 2 * mi;
    const long long out_base   = (long long)token * K_TOP * mi + (long long)krank * mi;

    float g = Y_grouped[y_grp_base + m_col];
    float u = Y_grouped[y_grp_base + mi + m_col];

    // DeepSeek V4 asymmetric clamp.
    if (g >  swiglu_limit) g =  swiglu_limit;
    if (u >  swiglu_limit) u =  swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;

    // SwiGLU: silu(g) * u.
    const float silu_g = g / (1.0f + __expf(-g));
    moe_gate_batch[out_base + m_col] = silu_g * u;
}
