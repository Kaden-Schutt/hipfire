#include <hip/hip_runtime.h>

// Dual-output MagnumQuant FWHT rotation: produces BOTH F32 and FP8 (E4M3)
// outputs in one launch. gfx12 (RDNA4) only — uses cvt_pk_fp8_f32.
//
// MOTIVATION: the FP8-dot4 decode GEMV (`gemv_hfp4g32_fp8_gfx12`)
// expects FP8-packed activations. With the standalone `mq_rotate_x` +
// `pack_f32_to_fp8` pre-pass approach, every weight_gemv(MFP4G32) call
// in production decode had to invalidate the FP8 cache (mq_x_rot's
// pointer is stable but data turns over per rotation) and re-pack —
// putting the pack launch on the critical path (~5-10 µs) and
// exceeding the dot4 savings (~3-5 µs) → 0.93× net regression on
// 9B Qwen 3.5 measured 2026-05-10.
//
// FUSING the FP8 pack into the rotation kernel itself adds only ~K bytes
// of extra global writes (1 byte per K-element) and 4 cvt_pk_fp8_f32
// VALU ops per thread per K-group — negligible compared to a saved
// kernel launch. The fallback F32 output is preserved so non-FP8
// consumers (gemv_hfp4g32 fallback path, any FP16-path that reads
// mq_x_rot directly) keep working unchanged.
//
// LAYOUT: same as mq_rotate_x except an additional `x_out_fp8`
// (1 byte per K-element). x_out and x_out_fp8 hold the same numerical
// values; FP8 is E4M3-rounded F32.
//
// Grid: [groups_total, batch, 1]. Block: [32]. LDS: 0.

extern "C" __launch_bounds__(32, 16)
__global__ void mq_rotate_x_dual_fp8_gfx12(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    unsigned char* __restrict__ x_out_fp8,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* x_in_b  = x_in  + batch_off;
          float* x_out_b = x_out + batch_off;
          unsigned char* x_out_fp8_b = x_out_fp8 + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float v0 = x_in_b[base]     * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];
    float v4 = x_in_b[base + 4] * signs1[d0 + 4];
    float v5 = x_in_b[base + 5] * signs1[d0 + 5];
    float v6 = x_in_b[base + 6] * signs1[d0 + 6];
    float v7 = x_in_b[base + 7] * signs1[d0 + 7];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    // Final scale + sign — produce F32 values first, then write both
    // F32 and FP8 from the same registers.
    v0 *= s * signs2[d0];
    v1 *= s * signs2[d0 + 1];
    v2 *= s * signs2[d0 + 2];
    v3 *= s * signs2[d0 + 3];
    v4 *= s * signs2[d0 + 4];
    v5 *= s * signs2[d0 + 5];
    v6 *= s * signs2[d0 + 6];
    v7 *= s * signs2[d0 + 7];

    x_out_b[base    ] = v0;
    x_out_b[base + 1] = v1;
    x_out_b[base + 2] = v2;
    x_out_b[base + 3] = v3;
    x_out_b[base + 4] = v4;
    x_out_b[base + 5] = v5;
    x_out_b[base + 6] = v6;
    x_out_b[base + 7] = v7;

    // FP8: 8 floats → 2 uint32 (8 packed E4M3 bytes) via 4 cvt_pk_fp8_f32.
    unsigned int lo = 0u, hi = 0u;
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(v4, v5, hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(v6, v7, hi, true);
    *(unsigned int*)(x_out_fp8_b + base    ) = lo;
    *(unsigned int*)(x_out_fp8_b + base + 4) = hi;
}
