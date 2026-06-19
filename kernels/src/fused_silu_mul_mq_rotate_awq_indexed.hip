#include <hip/hip_runtime.h>

// Per-expert (indexed) AWQ variant of `fused_silu_mul_mq_rotate` for the MoE
// down_proj input stage. Mirrors `fused_silu_mul_mq_rotate_awq` exactly, except
// the AWQ scale is selected PER BATCH ROW (per routed expert) from a device
// pointer table — each token's K_TOP experts have a different down.awq_scale.
//
// grid = [groups_total = K/256, K_TOP, 1]; blockIdx.y = krank (the routed-expert
// rank). gate/up/x_rot are [K_TOP × K] row-major (the per-expert SwiGLU outputs).
//   awq_scale = expert_down_awq_ptrs[topk_indices[krank]]   (the selected expert's [K] f32 vector)
// Pattern matches the indexed gate_up GEMV's `expert_ptrs[topk_indices[krank]]`.
//
// AWQ math: (W·s) · (silu(g)*u / s) = W · silu(g)*u. The /s divide happens
// AFTER silu*up and BEFORE signs1 + FWHT (the unrotated feature basis), exactly
// as the dense kernel does. On .hfq files without down AWQ this kernel is not
// dispatched (the plain `fused_silu_mul_mq_rotate` runs).

extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate_awq_indexed(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const unsigned long long* __restrict__ expert_down_awq_ptrs, // [n_exp] device ptrs to [K] f32
    const int* __restrict__ topk_indices,                        // [K_TOP]
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    // Per-expert select: this batch row's routed expert id → its awq_scale base.
    const int krank = blockIdx.y;
    const int expert_id = topk_indices[krank];
    const float* __restrict__ awq_scale =
        reinterpret_cast<const float*>(expert_down_awq_ptrs[expert_id]);

    const long long batch_off = (long long)krank * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
          float* x_rot_b = x_rot + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: silu(gate)*up, AWQ divide (per-expert scale), signs1.
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = (SILU_MUL(g0, u0) / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (SILU_MUL(g1, u1) / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (SILU_MUL(g2, u2) / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (SILU_MUL(g3, u3) / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (SILU_MUL(g4, u4) / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (SILU_MUL(g5, u5) / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (SILU_MUL(g6, u6) / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (SILU_MUL(g7, u7) / awq_scale[base + 7]) * signs1[d0 + 7];
    #undef SILU_MUL

    // Local butterfly: strides 1, 2, 4.
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

    // Wave butterfly via ds_swizzle.
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
    x_rot_b[base    ] = v0 * s * signs2[d0    ];
    x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
}
