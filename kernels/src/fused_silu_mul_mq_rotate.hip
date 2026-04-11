#include <hip/hip_runtime.h>

// Fused SwiGLU + FWHT rotation: rotate(silu(gate) * up).
// Used as the w_down input stage for MQ4 weights — replaces the sequence:
//   silu_mul_f32(gate, up, ffn_hidden)   // ~9 µs launch
//   mq_rotate_x(ffn_hidden, x_rot)       // ~9 µs launch
// with a single kernel.
//
// Parallelism is per 256-element group, same as mq_rotate_x — each workgroup
// owns its own group and applies the per-group butterfly. Unlike the siluxin
// GEMV experiment this has NO redundant compute: each element's silu is
// computed exactly once across the whole forward, matching the standalone
// silu_mul_f32 kernel.
//
// Structure matches mq_rotate_x exactly — only phase 1 differs (read
// gate[i]+up[i] and compute silu*up, vs read x[i]).

extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    // Batch row: grid.y is the batch dim. For decode grid=[groups_total, 1, 1]
    // and blockIdx.y = 0 (byte-identical to pre-batching). For prefill,
    // grid=[groups_total, N, 1]; gate, up, and x_rot are [N × K] row-major.
    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
          float* x_rot_b = x_rot + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: read gate + up, compute silu(gate)*up, apply signs1.
    // silu(z) = z / (1 + exp(-z)).
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = SILU_MUL(g0, u0) * signs1[d0];
    float v1 = SILU_MUL(g1, u1) * signs1[d0 + 1];
    float v2 = SILU_MUL(g2, u2) * signs1[d0 + 2];
    float v3 = SILU_MUL(g3, u3) * signs1[d0 + 3];
    float v4 = SILU_MUL(g4, u4) * signs1[d0 + 4];
    float v5 = SILU_MUL(g5, u5) * signs1[d0 + 5];
    float v6 = SILU_MUL(g6, u6) * signs1[d0 + 6];
    float v7 = SILU_MUL(g7, u7) * signs1[d0 + 7];
    #undef SILU_MUL

    // Local butterfly: strides 1, 2, 4 — identical to mq_rotate_x.
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

    // Wave butterfly via ds_swizzle — same as mq_rotate_x.
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
    x_rot_b[base]     = v0 * s * signs2[d0];
    x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
}
