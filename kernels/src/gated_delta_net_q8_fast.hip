// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fast variant of gated_delta_net_q8 for the default MQ4/HFQ4 path.
// Supports EF residual (sigma-delta) but always does single-end requant
// OUTSIDE the per-token loop. No per-token requant mode.
//
// Why: the full kernel moved the requant inside the loop to support
// per-token requant (PARO path). This increased register pressure because
// the compiler must carry the requant live-set (q0..q3, scale, rng, EF
// temporaries) across the entire loop body. On gfx906 wave64 (256 VGPRs/file)
// this causes spill-to-scratch: 108→172 bytes per workitem (+60%), costing
// +38% per dispatch. This variant avoids that by keeping the loop lean.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HD 128
#define TILE_ROWS 4

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_q8_fast(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int frame,
    __half* __restrict__ s_ef_residual
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];  // 4 KiB

    signed char* sq_base = s_q8 + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        int r = i / HD;
        float scale = s_scales[h * HD + row_start + r];
        S_tile[i] = scale * (float)sq_base[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[tid], k1 = k_t[tid+32], k2 = k_t[tid+64], k3 = k_t[tid+96];
        float q0 = q_t[tid], q1 = q_t[tid+32], q2 = q_t[tid+64], q3 = q_t[tid+96];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD;
            float s0 = sr[tid], s1 = sr[tid+32], s2 = sr[tid+64], s3 = sr[tid+96];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[tid] = s0; sr[tid+32] = s1; sr[tid+64] = s2; sr[tid+96] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    // Requant FP32 → Q8 — OUTSIDE the token loop.
    // Only the final state is quantized (intermediate tokens stay FP32 in LDS).
    const bool use_ef = (s_ef_residual != nullptr);
    for (int r = 0; r < TILE_ROWS; r++) {
        int rr = row_start + r;
        float* sr = S_tile + r * HD;
        float s0 = sr[tid], s1 = sr[tid+32], s2 = sr[tid+64], s3 = sr[tid+96];

        __half* efr = use_ef ? (s_ef_residual + h * HD * HD + rr * HD) : nullptr;
        if (use_ef) {
            s0 += __half2float(efr[tid]);
            s1 += __half2float(efr[tid+32]);
            s2 += __half2float(efr[tid+64]);
            s3 += __half2float(efr[tid+96]);
        }

        float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
        for (int o = 16; o > 0; o >>= 1)
            my_max = fmaxf(my_max, __shfl_xor(my_max, o));

        float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;

        signed char* dst = sq_base + r * HD;
        if (use_ef) {
            // Deterministic round-to-nearest with EF residual update.
            float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
            float qf0 = fminf(fmaxf(rintf(s0 * inv_s), -128.0f), 127.0f);
            float qf1 = fminf(fmaxf(rintf(s1 * inv_s), -128.0f), 127.0f);
            float qf2 = fminf(fmaxf(rintf(s2 * inv_s), -128.0f), 127.0f);
            float qf3 = fminf(fmaxf(rintf(s3 * inv_s), -128.0f), 127.0f);
            efr[tid]      = __float2half(s0 - qf0 * scale);
            efr[tid+32]   = __float2half(s1 - qf1 * scale);
            efr[tid+64]   = __float2half(s2 - qf2 * scale);
            efr[tid+96]   = __float2half(s3 - qf3 * scale);
            dst[tid]      = (signed char)qf0;
            dst[tid+32]   = (signed char)qf1;
            dst[tid+64]   = (signed char)qf2;
            dst[tid+96]   = (signed char)qf3;
            if (tid == 0) s_scales[h * HD + rr] = scale;
        } else {
            // Stochastic rounding.
            unsigned int rng = (unsigned int)(h * 1664525u + rr * 1013904223u
                             + tid * 214013u + frame * 2531011u);
            #define STOCH_ROUND(val) ({ \
                rng = rng * 1664525u + 1013904223u; \
                float noise = (float)(rng >> 8) * (1.0f / 16777216.0f); \
                (signed char)fminf(fmaxf(floorf((val) * inv_s + noise), -128.0f), 127.0f); \
            })
            dst[tid]      = STOCH_ROUND(s0);
            dst[tid+32]   = STOCH_ROUND(s1);
            dst[tid+64]   = STOCH_ROUND(s2);
            dst[tid+96]   = STOCH_ROUND(s3);
            #undef STOCH_ROUND
            if (tid == 0) {
                float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
                s_scales[h * HD + rr] = scale;
            }
        }
    }
}
