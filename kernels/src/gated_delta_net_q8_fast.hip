// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fast variant of gated_delta_net_q8 for the default MQ4/HFQ4 path.
// Supports EF residual (sigma-delta) but always does single-end requant
// OUTSIDE the per-token loop. No per-token requant mode.
//
// VARIANT R14c1 gdn_lds_float4_pack:
// BOD: mem-active(mixed) occ~17% mem_busy~33% L2~86% vgpr=64 LDS=2048.
// Baseline stores each lane's 4 HD-slices (tid, tid+32, tid+64, tid+96)
// as four scalar LDS floats → 4 loads + 4 stores per row per token.
// Pack as float4 S_f4[row*32 + tid] (same 2048 B footprint): one b128-class
// LDS transaction per row touch. Math identical (same 4-way partials, same
// shfl_down reduce tree, same update/requant). Hypothesis: LDS scalar
// multi-issue is on the critical path of the per-row state machine under
// low-occ decode (n_tokens=1); packing raises effective LDS bandwidth
// without raising VGPR (still 4 live floats, just as a vector).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HD 128

#ifndef HIPFIRE_GDN_TILE_ROWS
#define HIPFIRE_GDN_TILE_ROWS 4
#endif

#define TILE_ROWS HIPFIRE_GDN_TILE_ROWS

#ifndef HIPFIRE_GDN_QK_HEAD_DIV
#define HIPFIRE_GDN_QK_HEAD_DIV 1
#endif

#ifndef HIPFIRE_GDN_MIN_BLOCKS
#define HIPFIRE_GDN_MIN_BLOCKS 8
#endif

#ifndef HIPFIRE_GDN_BLOCK_SIZE
#define HIPFIRE_GDN_BLOCK_SIZE 32
#endif

#ifndef HIPFIRE_GDN_WAVES_PER_BLOCK
#define HIPFIRE_GDN_WAVES_PER_BLOCK 1
#endif

#ifndef HIPFIRE_GDN_KERNEL
#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_fast
#endif

#if defined(HIPFIRE_GFX1151_GDN_DPP_REDUCE)
// Preserve the shipping shfl_down reduction order while replacing its LDS
// bpermutes with gfx11 VALU cross-lane operations. Only lane 0 is consumed.
// row_shl:N gives lane i the value from lane i+N; after each addition the next
// DPP therefore observes the same partial sums as offsets 16,8,4,2,1 in the
// original tree.
__device__ __forceinline__ float hipfire_gdn_reduce_sum_lane0_dpp(float value) {
    int bits = __builtin_bit_cast(int, value);
    const int other_half = __builtin_amdgcn_permlanex16(
        bits, bits, 0, 0, false, false);
    value += __builtin_bit_cast(float, other_half);

    bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float,
        __builtin_amdgcn_mov_dpp(bits, 0x108, 0xf, 0xf, true));
    bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float,
        __builtin_amdgcn_mov_dpp(bits, 0x104, 0xf, 0xf, true));
    bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float,
        __builtin_amdgcn_mov_dpp(bits, 0x102, 0xf, 0xf, true));
    bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float,
        __builtin_amdgcn_mov_dpp(bits, 0x101, 0xf, 0xf, true));
    return value;
}

__device__ __forceinline__ float hipfire_gdn_read_lane0(float value) {
    return __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(
        __builtin_bit_cast(int, value)));
}
#endif

extern "C" __launch_bounds__(HIPFIRE_GDN_BLOCK_SIZE, HIPFIRE_GDN_MIN_BLOCKS)
__global__ void HIPFIRE_GDN_KERNEL(
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
    __half* __restrict__ s_ef_residual,
    const int* __restrict__ row_slot,   // [n_tokens] or nullptr = legacy (all slot 0)
    int s_stride_elems                  // elements between consecutive slots' S state
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
#if HIPFIRE_GDN_WAVES_PER_BLOCK == 2
    const int wave = threadIdx.x >> 5;
    const int tid = threadIdx.x & 31;
#else
    const int tid = threadIdx.x;
#endif
    const int stride = n_heads * HD;
    const int qk_heads = n_heads / HIPFIRE_GDN_QK_HEAD_DIV;
    const int qk_stride = qk_heads * HD;
    const int qk_head = h / HIPFIRE_GDN_QK_HEAD_DIV;
#if HIPFIRE_GDN_WAVES_PER_BLOCK == 2
    const int row_start = tile * (TILE_ROWS * 2) + wave * TILE_ROWS;
#else
    const int row_start = tile * TILE_ROWS;
#endif

    // Packed layout: S_f4[r * 32 + tid] = {S[r][tid], S[r][tid+32],
    // S[r][tid+64], S[r][tid+96]} — same 4*128 floats as the scalar tile.
#if HIPFIRE_GDN_WAVES_PER_BLOCK == 2
    __shared__ float4 S_f4[TILE_ROWS * 2 * 32];
    __shared__ float qk_f32[HD * 2];
#define HIPFIRE_GDN_S_INDEX(r, lane) (((wave * TILE_ROWS + (r)) * 32) + (lane))
#else
    __shared__ float4 S_f4[TILE_ROWS * 32];
#define HIPFIRE_GDN_S_INDEX(r, lane) (((r) * 32) + (lane))
#endif

    // Each slot owns an independent, fixed-size S state. There is no shared
    // arena and no ragged length here, unlike KV — just a stride. In legacy
    // mode row_slot is null, the offset is 0, and addressing is unchanged.
    const int dn_slot = (row_slot != nullptr) ? row_slot[0] : 0;
    const long long s_off = (long long)dn_slot * (long long)s_stride_elems;

    signed char* sq_base = s_q8 + s_off + h * HD * HD + row_start * HD;
    for (int r = 0; r < TILE_ROWS; r++) {
        const float scale = s_scales[s_off + h * HD + row_start + r];
        const signed char* src = sq_base + r * HD;
        S_f4[HIPFIRE_GDN_S_INDEX(r, tid)] = make_float4(
            scale * (float)src[tid],
            scale * (float)src[tid + 32],
            scale * (float)src[tid + 64],
            scale * (float)src[tid + 96]);
    }

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * qk_stride + qk_head * HD;
        const float* k_t = k + t * qk_stride + qk_head * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

#if HIPFIRE_GDN_WAVES_PER_BLOCK == 2
        const int wg_tid = threadIdx.x;
        for (int i = wg_tid; i < HD; i += HIPFIRE_GDN_BLOCK_SIZE) {
            qk_f32[i] = q_t[i];
            qk_f32[HD + i] = k_t[i];
        }
        __syncthreads();
        float k0 = qk_f32[HD + tid], k1 = qk_f32[HD + tid+32];
        float k2 = qk_f32[HD + tid+64], k3 = qk_f32[HD + tid+96];
        float q0 = qk_f32[tid], q1 = qk_f32[tid+32];
        float q2 = qk_f32[tid+64], q3 = qk_f32[tid+96];
#else
        float k0 = k_t[tid], k1 = k_t[tid+32], k2 = k_t[tid+64], k3 = k_t[tid+96];
        float q0 = q_t[tid], q1 = q_t[tid+32], q2 = q_t[tid+64], q3 = q_t[tid+96];
#endif
#if TILE_ROWS == 4
        const float4 v_rows = *reinterpret_cast<const float4*>(v_t + row_start);
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float4 s = S_f4[HIPFIRE_GDN_S_INDEX(r, tid)];
            float s0 = s.x, s1 = s.y, s2 = s.z, s3 = s.w;

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
#if defined(HIPFIRE_GFX1151_GDN_DPP_REDUCE)
            kv = hipfire_gdn_read_lane0(hipfire_gdn_reduce_sum_lane0_dpp(kv));
#else
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);
#endif

            const float v_row =
                (r == 0) ? v_rows.x :
                (r == 1) ? v_rows.y :
                (r == 2) ? v_rows.z : v_rows.w;
            float delta = (v_row - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            S_f4[HIPFIRE_GDN_S_INDEX(r, tid)] = make_float4(s0, s1, s2, s3);

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
#if defined(HIPFIRE_GFX1151_GDN_DPP_REDUCE)
            out_v = hipfire_gdn_reduce_sum_lane0_dpp(out_v);
#else
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
#endif
            if (tid == 0) {
                if (r == 0) out0 = out_v;
                else if (r == 1) out1 = out_v;
                else if (r == 2) out2 = out_v;
                else out3 = out_v;
            }
        }
        if (tid == 0) {
            *reinterpret_cast<float4*>(output + t * stride + h * HD + row_start) =
                make_float4(out0, out1, out2, out3);
        }
#elif TILE_ROWS == 8
#pragma clang loop unroll(disable)
        for (int r = 0; r < TILE_ROWS; r++) {
            float4 s = S_f4[r * 32 + tid];
            float s0 = s.x, s1 = s.y, s2 = s.z, s3 = s.w;

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
#if defined(HIPFIRE_GFX1151_GDN_DPP_REDUCE)
            kv = hipfire_gdn_read_lane0(hipfire_gdn_reduce_sum_lane0_dpp(kv));
#else
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);
#endif

            const float v_row = v_t[row_start + r];
            float delta = (v_row - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            S_f4[r * 32 + tid] = make_float4(s0, s1, s2, s3);

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
#if defined(HIPFIRE_GFX1151_GDN_DPP_REDUCE)
            out_v = hipfire_gdn_reduce_sum_lane0_dpp(out_v);
#else
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
#endif
            if (tid == 0)
                output[t * stride + h * HD + row_start + r] = out_v;
        }
#else
#error "unsupported HIPFIRE_GDN_TILE_ROWS"
#endif
#if HIPFIRE_GDN_WAVES_PER_BLOCK == 2
        if (t + 1 < n_tokens)
            __syncthreads();
#endif
    }

    // Requant FP32 → Q8 — OUTSIDE the token loop.
    // Only the final state is quantized (intermediate tokens stay FP32 in LDS).
    const bool use_ef = (s_ef_residual != nullptr);
    for (int r = 0; r < TILE_ROWS; r++) {
        int rr = row_start + r;
        float4 s = S_f4[HIPFIRE_GDN_S_INDEX(r, tid)];
        float s0 = s.x, s1 = s.y, s2 = s.z, s3 = s.w;

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
            if (tid == 0) s_scales[s_off + h * HD + rr] = scale;
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
                s_scales[s_off + h * HD + rr] = scale;
            }
        }
    }

#undef HIPFIRE_GDN_S_INDEX
}
