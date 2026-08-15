#include <hip/hip_runtime.h>

// Kernel B of the two-kernel fused_rmsnorm_mq_rotate split for gfx942.
//
// Consumes rms_in[batch] from Kernel A (rmsnorm_reduce_gfx942) and emits
// the normalized + FWHT-rotated output. One WG processes ONE group of
// 256 elements for ONE batch element.
//
// Geometry:
//   block = [64, 1, 1]   = 1 wave64 (lanes 0..31 active per current FWHT)
//   grid  = [groups_per_row, batch, 1]   = [K/256, batch, 1]
//   LDS   = 0 (single-wave, all in registers)
//
// Math: byte-identical to fused_rmsnorm_mq_rotate's Phase 1c + Phase 2
// for the single group this WG owns. Same 3-stage local butterfly +
// 5-stage ds_swizzle wave butterfly, same 1/16 scale, same signs1/2
// application order.
//
// Lanes 32..63 are idle (FWHT math is wave32-shaped). This matches the
// existing kernel's per-warp logic verbatim and keeps math byte-exact.

extern "C" __global__ __launch_bounds__(64)
void rotate_with_rms_gfx942(
    const float* __restrict__ x,          // [batch, K]
    const float* __restrict__ weight,     // [K]
    const float* __restrict__ signs1,     // [K]
    const float* __restrict__ signs2,     // [K]
    const float* __restrict__ rms_in,     // [batch]
    float* __restrict__ x_rot,            // [batch, K]
    int K
) {
    const int lane_id = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;   // wave64 launch but only lanes 0..31 do work

    const int group = blockIdx.x;
    const int batch = blockIdx.y;

    const long long batch_off = (long long)batch * K;
    const float* x_b     = x     + batch_off;
          float* x_rot_b = x_rot + batch_off;

    const float rms = rms_in[batch];

    const int d0 = lane_id * 8;
    const int base = group * 256 + d0;

    // Apply Phase 1c (x * weight * rms) then Phase 2 (* signs1) in the SAME
    // left-associative order as the baseline kernel to preserve FP rounding:
    //   baseline: x_shared[i] = x_b[i] * weight[i] * rms;   // = ((x*w)*r)
    //             v = x_shared[i] * signs1[i];              // = ((x*w)*r) * s
    // We compute (((x*w)*r)*s) directly here per element.
    #define LOAD_V(idx, vname) \
        float vname; { \
            float _xw = x_b[base + (idx)] * weight[d0 + (idx)]; \
            float _xwr = _xw * rms; \
            vname = _xwr * signs1[d0 + (idx)]; \
        }
    LOAD_V(0, v0)
    LOAD_V(1, v1)
    LOAD_V(2, v2)
    LOAD_V(3, v3)
    LOAD_V(4, v4)
    LOAD_V(5, v5)
    LOAD_V(6, v6)
    LOAD_V(7, v7)
    #undef LOAD_V

    // Local butterfly: strides 1, 2, 4 (within each thread's 8 registers).
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

    // Wave butterfly via ds_swizzle (32-lane group; wave-local).
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while (0)
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
    x_rot_b[base]     = v0 * s * signs2[d0];
    x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
}
