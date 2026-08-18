// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// DeepSeek-V4 parent checkpoint: fused FP8 (E4M3 OCP) + UE8M0 scale
// quantize→dequantize simulation, in-place on a BF16 activation tensor.
//
// Reference (`.codeinsight+research/ds4-parent-ref/inference/kernel.py`):
//   fast_log2_ceil        :22-27
//   fast_pow2             :30-33
//   fast_round_scale      :36-37
//   act_quant_kernel body :41-102  (inplace=True, round_scale=True path)
//     amax = reduce_absmax over group                 :77
//     amax = max(amax, 1e-4)                         :79
//     s    = fast_round_scale(amax, 1/448)           :80-81
//     q    = e4m3_rne(clamp(x/s, ±448))              :88-89
//     y    = bf16( f32(q) * s )                      :86-90
//   act_quant wrapper     :105-125  (block runtime arg; N % block == 0)
// Contract: local://ds4-parent-contract.md §3.
//
// E4M3 rounding path
// ------------------
// MI300X (gfx942) `__builtin_amdgcn_cvt_pk_fp8_f32` implements the *FNUZ*
// encoding (max finite 240, NaN = 0x80), NOT OCP `float8_e4m3fn`
// (max 448, NaN = 0x7F/0xFF). Empirically confirmed on this host's
// mi300x ROCm 7.14: saturating 448/500 both encode as 0x80 under the
// bare builtin, and magnitudes past the FNUZ range collapse. HIP's
// `__hip_cvt_float_to_fp8(..., __HIP_E4M3)` fast-path on this SKU still
// routes through the same FNUZ builtin after a software clamp to ±448 —
// and on this agent the OCP class path aborted the queue — so we do NOT
// trust it here.
//
// Therefore this kernel implements OCP E4M3 RNE + saturate-to-finite
// in software (`e4m3_rne_ocp` below), bit-compatible with the host-side
// `cast_to_f8<float, /*is_fnuz=*/false>(…, wm=3, we=4, clip=true)` in
// `amd_hip_fp8.h`. Decode is the contract's pure LUT form
// (`e4m3_to_f32`). Incorrect rounding here is a silent accuracy bug:
// no coherence check would catch a FNUZ-vs-OCP mixup.
//
// Launch geometry
// ---------------
//   block = [64, 1, 1]   one native CDNA wave
//   grid  = [rows, last_dim / block_size, 1]
// One workgroup owns one (row, group) pair. For block_size ∈ {64, 128}
// every lane holds `block_size/64` elements; absmax is a wave XOR-reduce
// over the per-lane local max, then every lane writes its elements back.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ── bit-exact scale helpers (kernel.py:22-37) ─────────────────────────
__device__ __forceinline__ int fast_log2_ceil(float x) {
    unsigned b = __builtin_bit_cast(unsigned, x);
    int e = (int)((b >> 23) & 0xFFu);
    unsigned m = b & 0x7FFFFFu;
    return e - 127 + (m != 0u ? 1 : 0);
}

__device__ __forceinline__ float fast_pow2(int n) {
    return __builtin_bit_cast(float, (unsigned)(n + 127) << 23);
}

__device__ __forceinline__ float fast_round_scale(float amax, float max_inv) {
    return fast_pow2(fast_log2_ceil(amax * max_inv));
}

// ── OCP float8_e4m3fn decode (contract §2.4) ──────────────────────────
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    float v;
    if (e == 0) {
        // subnormal: (-1)^s * 2^-6 * (m/8) = m * 2^-9
        v = (float)m * 0x1p-9f;
    } else if (e == 0xF && m == 0x7) {
        // only NaN encoding; should not appear after saturating quant
        v = __int_as_float(0x7FC00000u);
    } else {
        // 2^(e-7) * (1 + m/8); exp field in f32 = (e - 7) + 127 = e + 120
        const float pow2 = __builtin_bit_cast(float, (unsigned)(e + 120) << 23);
        v = pow2 * (1.0f + (float)m * 0.125f);
    }
    return s ? -v : v;
}

// ── OCP E4M3 RNE + saturate-to-finite (software) ──────────────────────
// Specialised from amd_hip_fp8.h cast_to_f8<float, is_fnuz=false>
// with wm=3, we=4, clip=true. RNE midpoints: sticky/midpoint detected
// BEFORE the denorm right-shift so residual bits below the half-ULP
// mark cannot masquerade as a tie (see header comment at the midpoint
// test). On a pure midpoint the LSB kept is forced even.
__device__ __forceinline__ unsigned char e4m3_rne_ocp(float v) {
    const unsigned x = __builtin_bit_cast(unsigned, v);
    const unsigned sign = x >> 31;
    const unsigned absx = x & 0x7FFFFFFFu;
    const int exponent = (int)((x >> 23) & 0xFFu);
    unsigned long long mantissa = (unsigned long long)(x & 0x7FFFFFu);

    // OCP E4M3: no Inf; NaN input → signed NaN (0x7F / 0xFF).
    if ((x & 0x7F800000u) == 0x7F800000u) {
        return (unsigned char)((sign << 7) | 0x7F);
    }
    // Saturate-to-finite: |v| > 448 → ± max finite of same sign (0x7E).
    // 448.0f bit pattern = 0x43E00000.
    if (absx > 0x43E00000u) {
        return (unsigned char)((sign << 7) | 0x7E);
    }
    if (absx == 0u) {
        return (unsigned char)(sign << 7);  // preserve signed zero
    }

    constexpr int wm = 3;
    constexpr int we = 4;
    constexpr int mfmt = 23;
    constexpr int f8_bias = (1 << (we - 1)) - 1;          // 7
    constexpr int f8_denormal_act_exponent = 1 - f8_bias; // -6

    int act_exponent;
    int exponent_diff;
    if (exponent == 0) {
        // f32 denormal
        act_exponent = exponent - 127 + 1;
        exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {
        act_exponent = exponent - 127;
        if (act_exponent <= f8_denormal_act_exponent) {
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        } else {
            exponent_diff = 0;
        }
        mantissa += (1ull << mfmt);  // implicit 1
    }

    // Midpoint / sticky MUST be sampled before any right shift.
    const int shift = mfmt - wm + exponent_diff;  // bits dropped after align
    bool midpoint = false;
    if (shift >= 1 && shift < 64) {
        const unsigned long long half = 1ull << (shift - 1);
        const unsigned long long mask = (1ull << shift) - 1ull;
        midpoint = (mantissa & mask) == half;
    }

    if (exponent_diff > 0) {
        mantissa >>= (unsigned)exponent_diff;
    } else if (exponent_diff == -1) {
        mantissa <<= 1;
    }

    const bool implicit_one = (mantissa & (1ull << mfmt)) != 0ull;
    int f8_exponent =
        (act_exponent + exponent_diff) + f8_bias - (implicit_one ? 0 : 1);

    const unsigned long long drop_mask = (1ull << (mfmt - wm)) - 1ull;
    const bool odd = (mantissa & (1ull << (mfmt - wm))) != 0ull;
    // RNE: add the dropped bits' contribution; on a pure midpoint force even.
    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa) & drop_mask;

    // Overflow from rounding.
    if (f8_exponent == 0) {
        if (mantissa & (1ull << mfmt)) {
            f8_exponent = 1;  // denorm → normal
        }
    } else {
        if (mantissa & (1ull << (mfmt + 1))) {
            mantissa >>= 1;
            f8_exponent++;
        }
    }
    mantissa >>= (mfmt - wm);

    constexpr int max_exp = (1 << we) - 1;  // 15
    if (f8_exponent > max_exp) {
        // clip=true → max finite
        return (unsigned char)((sign << 7) | 0x7E);
    }
    if (f8_exponent == 0 && mantissa == 0ull) {
        return (unsigned char)(sign << 7);
    }
    mantissa &= (1ull << wm) - 1ull;
    return (unsigned char)((sign << 7) | ((unsigned)f8_exponent << wm) | (unsigned)mantissa);
}

// ── kernel ────────────────────────────────────────────────────────────
// x: [rows, last_dim] BF16 row-major. block ∈ {64, 128} (validated host-side).
extern "C" __global__ __launch_bounds__(64)
void act_quant_fp8_ue8m0_inplace_gfx942(
    __hip_bfloat16* __restrict__ x,
    int rows,
    int last_dim,
    int block
) {
    const int row = (int)blockIdx.x;
    const int group = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;  // 0..63
    if (row >= rows) return;

    const int group_base = group * block;
    if (group_base >= last_dim) return;

    // Elements per lane: block/64 ∈ {1, 2}.
    const int per_lane = block >> 6;  // block is 64 or 128
    float vals[2];
    float local_max = 0.0f;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        vals[i] = 0.0f;
        if (i < per_lane) {
            const int col = group_base + lane * per_lane + i;
            // col is in-range when last_dim % block == 0 (host-enforced).
            const float v = __bfloat162float(x[(long long)row * last_dim + col]);
            vals[i] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }

    // Wave-wide absmax via XOR shuffle (native wave64).
    float amax = local_max;
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 64));
    }
    amax = fmaxf(amax, 1e-4f);                         // kernel.py:79
    const float s = fast_round_scale(amax, 1.0f / 448.0f);  // :80-81
    const float inv_s = 1.0f / s;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (i < per_lane) {
            const int col = group_base + lane * per_lane + i;
            const float scaled = fminf(fmaxf(vals[i] * inv_s, -448.0f), 448.0f);
            const unsigned char q = e4m3_rne_ocp(scaled);
            const float dequant = e4m3_to_f32(q) * s;
            x[(long long)row * last_dim + col] = __float2bfloat16(dequant);
        }
    }
}
