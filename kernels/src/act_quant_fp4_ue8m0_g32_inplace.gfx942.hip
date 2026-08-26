// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// DeepSeek-V4 parent checkpoint: fused FP4 (E2M1) + UE8M0 scale
// quantize→dequantize simulation, in-place on a BF16 activation tensor.
// Groups of 32 along the last dim (the only group size the parent uses).
//
// Reference (`.codeinsight+research/ds4-parent-ref/inference/kernel.py`):
//   fast_log2_ceil        :22-27
//   fast_pow2             :30-33
//   fast_round_scale      :36-37
//   fp4_quant_kernel body :129-183  (inplace=True path)
//     amax = reduce_absmax over group of 32           :161
//     amax = max(amax, 6 * 2^-126)                    :163
//     s    = fast_round_scale(amax, 1/6)              :164
//     q    = e2m1_rne(clamp(x/s, ±6))                 :169-170
//     y    = bf16( f32(q) * s )                       :167-171
//   fp4_act_quant wrapper :186-200
// Contract: local://ds4-parent-contract.md §3.
//
// e2m1_rne
// --------
// Round-to-nearest-even onto the 8 magnitudes
//   {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
// with sign preserved. Implemented as a branchless midpoint cascade —
// no library FP4 type, no LUT search that would hide the tie rule.
//
// Tie-breaking at the exact midpoints (halfway between two neighbours):
//   0.25 → 0.0   (0 even, 0.5 odd in the code index)
//   0.75 → 1.0   (0.5 odd, 1.0 even)
//   1.25 → 1.0   (1.0 even, 1.5 odd)
//   1.75 → 2.0   (1.5 odd, 2.0 even)
//   2.5  → 2.0   (2.0 even, 3.0 odd)
//   3.5  → 4.0   (3.0 odd, 4.0 even)
//   5.0  → 4.0   (4.0 even, 6.0 odd)   ← note: 5 is the mid of 4 and 6
// "Even" here means the even code index in {0,.5,1,1.5,2,3,4,6}
// (indices 0,2,4,6), matching IEEE-style RNE on this discrete set.
// Values above 6 saturate to 6 (host also clamps before rounding).
//
// Launch geometry
// ---------------
//   block = [32, 1, 1]   one group = one 32-lane cohort (half-wave on CDNA)
//   grid  = [rows, last_dim / 32, 1]
// Lane L owns element L of the group. Absmax via XOR-shuffle over 32.

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

// E2M1 magnitude LUT (contract §2.4 / code indices 0..7).
// Indexed by the rounded code; sign applied by the caller.
__device__ __forceinline__ float e2m1_mag_from_code(int code) {
    // code in 0..7
    const float mag[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    return mag[code];
}

// Branchless RNE onto {0,.5,1,1.5,2,3,4,6}. `x` is already clamped to
// [0, 6] (sign handled outside). Returns the magnitude code 0..7.
//
// Decision boundaries sit at the midpoints; on an exact midpoint we take
// the even code (RNE). Boundaries used (left-closed / right-open except
// at a midpoint where the even side wins):
//
//   [0, 0.25) ∪ {0.25} → 0
//   (0.25, 0.75)       → 1   (.5)
//   {0.75} ∪ (0.75, 1.25) ∪ {1.25} → 2   (1.0)
//   (1.25, 1.75)       → 3   (1.5)
//   {1.75} ∪ (1.75, 2.5) ∪ {2.5}  → 4   (2.0)
//   (2.5, 3.5)         → 5   (3.0)
//   {3.5} ∪ (3.5, 5.0) ∪ {5.0}    → 6   (4.0)
//   (5.0, 6.0]         → 7   (6.0)
//
// Which collapses to thresholds at the midpoints with `<=` on the side
// of the even code:
//   x <= 0.25 → 0
//   x <  0.75 → 1
//   x <= 1.25 → 2
//   x <  1.75 → 3
//   x <= 2.5  → 4
//   x <  3.5  → 5
//   x <= 5.0  → 6
//   else      → 7
__device__ __forceinline__ int e2m1_rne_code(float ax) {
    // Branchless select cascade. Exact midpoints land on the even code
    // via the `<=` / `<` choice documented above.
    int c = 7;
    c = (ax <= 5.0f)  ? 6 : c;
    c = (ax <  3.5f)  ? 5 : c;
    c = (ax <= 2.5f)  ? 4 : c;
    c = (ax <  1.75f) ? 3 : c;
    c = (ax <= 1.25f) ? 2 : c;
    c = (ax <  0.75f) ? 1 : c;
    c = (ax <= 0.25f) ? 0 : c;
    return c;
}

// ── kernel ────────────────────────────────────────────────────────────
extern "C" __global__ __launch_bounds__(32)
void act_quant_fp4_ue8m0_g32_inplace_gfx942(
    __hip_bfloat16* __restrict__ x,
    int rows,
    int last_dim
) {
    const int row = (int)blockIdx.x;
    const int group = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;  // 0..31
    if (row >= rows) return;

    const int col = group * 32 + lane;
    if (col >= last_dim) return;

    const long long idx = (long long)row * last_dim + col;
    const float v = __bfloat162float(x[idx]);
    float amax = fabsf(v);

    // 32-lane absmax reduce (half-wave). XOR mask stays inside 0..31.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    // kernel.py:163 — floor so a zero group still produces a finite scale.
    amax = fmaxf(amax, 6.0f * 0x1p-126f);
    const float s = fast_round_scale(amax, 1.0f / 6.0f);  // :164
    const float inv_s = 1.0f / s;

    float scaled = v * inv_s;
    scaled = fminf(fmaxf(scaled, -6.0f), 6.0f);

    const float ax = fabsf(scaled);
    const int code = e2m1_rne_code(ax);
    const float mag = e2m1_mag_from_code(code);
    // Preserve sign, including signed zero via copysign.
    const float q = copysignf(mag, scaled);
    const float dequant = q * s;
    x[idx] = __float2bfloat16(dequant);
}
