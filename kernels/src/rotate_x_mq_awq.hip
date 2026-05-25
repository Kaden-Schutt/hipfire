#include <hip/hip_runtime.h>

// AWQ-aware variant of `mq_rotate_x` (in `gemv_mq4g256.hip`).
//
// Phase A Stage A — F2 follow-up. The original `mq_rotate_x` is used as the
// input-rotation stage before the post-projection linear (o_proj / out_proj),
// which Stage A's AWQ whitelist excluded because no AWQ-aware rotate existed
// for that path. This kernel completes the AWQ math `(W·s) · (x/s) = W·x`
// by dividing each input element by `awq_scale[i]` BEFORE the FWHT — exact
// mirror of `fused_rmsnorm_mq_rotate_awq.hip` Phase 1c.
//
// Backward compatibility: this kernel is dispatched ONLY when the upcoming
// linear carries an `awq_scale` GpuTensor. Otherwise the standard
// `mq_rotate_x` runs. Both produce identical output on .hfq files that
// don't carry AWQ scales.
//
// Layout: identical to `mq_rotate_x` aside from the extra divide. See that
// kernel's preamble for FWHT details.
//
// awq_scale: 1D FP32 buffer length K (host-side F16 → F32 conversion happens
// in the loader; see hfq.rs::load_awq_scale).

extern "C" __launch_bounds__(32, 16)
__global__ void rotate_x_mq_awq(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ awq_scale,   // [K], FP32
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

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // AWQ divide + signs1 gather. The divide happens BEFORE the FWHT to
    // mirror the offline weight side `W' = W · diag(s)` — quantize-time
    // pre-scaling cancels the runtime per-channel divide so that
    // `(W·s) · (x/s) = W·x`. awq_scale is 1D length-K, indexed in the
    // unrotated x basis (not FWHT-rotated).
    float v0 = (x_in_b[base    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x_in_b[base + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x_in_b[base + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x_in_b[base + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x_in_b[base + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x_in_b[base + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x_in_b[base + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x_in_b[base + 7] / awq_scale[base + 7]) * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4
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

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle.
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
    x_out_b[base    ] = v0 * s * signs2[d0    ];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * s * signs2[d0 + 7];
}
