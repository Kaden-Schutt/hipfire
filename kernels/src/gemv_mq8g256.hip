#include <hip/hip_runtime.h>

// MagnumQuant MQ8 GEMV: FWHT-rotated symmetric INT8 with dp4a.
// x_rot is PRE-ROTATED and PRE-QUANTIZED to INT8 by the caller.
// Inner loop uses v_dot4_i32_i8 (sdot4) for 4x VALU throughput vs FP32.
// Both operands are signed int8 (Q8_1 activations + symmetric MQ8 weights),
// so sdot4 (gfx906+, dot2-insts) is correct. sudot4 (RDNA3+, dot8-insts)
// is the mixed-mode form for unsigned-weight quants and is gratuitous here;
// it also fails to compile on gfx906 since the iu8 instruction is RDNA3+.
//
// Weight format per group (258 bytes for 256 elements):
//   [0:2]   f16 scale
//   [2:258] int8[256] quantized FWHT-rotated weights
//
// x_rot format per group:
//   x_rot_q8[256] int8 (quantized rotated x)
//   x_rot_scale[1] float (per-group scale for x)

// f16 → f32 helper
__device__ __forceinline__ float f16_load(const unsigned char* p) {
    unsigned short bits = p[0] | (p[1] << 8);
    unsigned int sign = (bits >> 15) & 1;
    unsigned int exp = (bits >> 10) & 0x1F;
    unsigned int frac = bits & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        // Denorm
        float f = frac * (1.0f / 1024.0f);
        return sign ? -f * (1.0f / 16384.0f) : f * (1.0f / 16384.0f);
    }
    if (exp == 31) return sign ? -__int_as_float(0x7F800000) : __int_as_float(0x7F800000);
    unsigned int f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    return __int_as_float(f32_bits);
}

// 4x unrolled dp4a GEMV: 4 groups in-flight, 8 dp4a per iteration.
// MQ8 is bandwidth-bound (258 B/group) — 4x is the sweet spot.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq8g256(
    const unsigned char* __restrict__ A,
    const signed char* __restrict__ x_q8,
    const float* __restrict__ x_scales,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int bpg = 258;
    const unsigned char* row_ptr = A + (long long)row * groups_per_row * bpg;
    const int w_off = tid * 8;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int quads = groups_per_row / 4;
    const int tail = groups_per_row - quads * 4;

    for (int q = 0; q < quads; q++) {
        const int g = q * 4;

        #define LOAD_DP4A(i, acc_var) { \
            const unsigned char* gp = row_ptr + (g + (i)) * bpg; \
            float ws = f16_load(gp); \
            float xs = x_scales[g + (i)]; \
            const signed char* wb = (const signed char*)(gp + 2); \
            const signed char* xb = x_q8 + (g + (i)) * 256; \
            int wp0 = *(const int*)(wb + w_off); \
            int wp1 = *(const int*)(wb + w_off + 4); \
            int xp0 = *(const int*)(xb + w_off); \
            int xp1 = *(const int*)(xb + w_off + 4); \
            int dot = __builtin_amdgcn_sdot4(wp0, xp0, 0, false); \
            dot = __builtin_amdgcn_sdot4(wp1, xp1, dot, false); \
            acc_var += ws * xs * (float)dot; \
        }

        LOAD_DP4A(0, acc0) LOAD_DP4A(1, acc1) LOAD_DP4A(2, acc2) LOAD_DP4A(3, acc3)
        #undef LOAD_DP4A
    }

    for (int gi = 0; gi < tail; gi++) {
        const int g = quads * 4 + gi;
        const unsigned char* gp = row_ptr + g * bpg;
        float ws = f16_load(gp);
        float xs = x_scales[g];
        const signed char* wb = (const signed char*)(gp + 2);
        const signed char* xb = x_q8 + g * 256;
        int wp0 = *(const int*)(wb + w_off);
        int wp1 = *(const int*)(wb + w_off + 4);
        int xp0 = *(const int*)(xb + w_off);
        int xp1 = *(const int*)(xb + w_off + 4);
        int dot = __builtin_amdgcn_sdot4(wp0, xp0, 0, false);
        dot = __builtin_amdgcn_sdot4(wp1, xp1, dot, false);
        acc0 += ws * xs * (float)dot;
    }

    float acc = (acc0+acc1)+(acc2+acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// Pre-rotate and quantize x to INT8 for dp4a.
// One block per 256-element group.
extern "C" __launch_bounds__(32, 16)
__global__ void mq8_rotate_quantize_x(
    const float* __restrict__ x_in,
    signed char* __restrict__ x_q8,    // output: INT8 quantized rotated x
    float* __restrict__ x_scales,      // output: per-group scales
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    if (group >= K / 256) return;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Load and apply signs1
    float v0 = x_in[base]     * signs1[d0];
    float v1 = x_in[base + 1] * signs1[d0 + 1];
    float v2 = x_in[base + 2] * signs1[d0 + 2];
    float v3 = x_in[base + 3] * signs1[d0 + 3];
    float v4 = x_in[base + 4] * signs1[d0 + 4];
    float v5 = x_in[base + 5] * signs1[d0 + 5];
    float v6 = x_in[base + 6] * signs1[d0 + 6];
    float v7 = x_in[base + 7] * signs1[d0 + 7];

    // Local butterfly
    float t;
    t = v0; v0 += v1; v1 = t - v1;
    t = v2; v2 += v3; v3 = t - v3;
    t = v4; v4 += v5; v5 = t - v5;
    t = v6; v6 += v7; v7 = t - v7;
    t = v0; v0 += v2; v2 = t - v2;
    t = v1; v1 += v3; v3 = t - v3;
    t = v4; v4 += v6; v6 = t - v6;
    t = v5; v5 += v7; v7 = t - v7;
    t = v0; v0 += v4; v4 = t - v4;
    t = v1; v1 += v5; v5 = t - v5;
    t = v2; v2 += v6; v6 = t - v6;
    t = v3; v3 += v7; v7 = t - v7;

    // Wave butterfly via ds_swizzle
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)
    HBFLY8(0x041F, 1); HBFLY8(0x081F, 2); HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8); HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    // Apply scale and signs2
    const float s = 0.0625f;
    v0 *= s * signs2[d0];     v1 *= s * signs2[d0 + 1];
    v2 *= s * signs2[d0 + 2]; v3 *= s * signs2[d0 + 3];
    v4 *= s * signs2[d0 + 4]; v5 *= s * signs2[d0 + 5];
    v6 *= s * signs2[d0 + 6]; v7 *= s * signs2[d0 + 7];

    // Find max absolute value across the group for symmetric quantization
    __shared__ float smax[32];
    float my_max = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))),
                         fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7))));
    for (int off = 16; off > 0; off >>= 1)
        my_max = fmaxf(my_max, __shfl_xor(my_max, off));

    float group_scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
    float inv_scale = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;

    if (tid == 0) x_scales[group] = group_scale;

    // Quantize to INT8
    signed char* out = x_q8 + group * 256 + d0;
    out[0] = (signed char)fminf(fmaxf(rintf(v0 * inv_scale), -128.0f), 127.0f);
    out[1] = (signed char)fminf(fmaxf(rintf(v1 * inv_scale), -128.0f), 127.0f);
    out[2] = (signed char)fminf(fmaxf(rintf(v2 * inv_scale), -128.0f), 127.0f);
    out[3] = (signed char)fminf(fmaxf(rintf(v3 * inv_scale), -128.0f), 127.0f);
    out[4] = (signed char)fminf(fmaxf(rintf(v4 * inv_scale), -128.0f), 127.0f);
    out[5] = (signed char)fminf(fmaxf(rintf(v5 * inv_scale), -128.0f), 127.0f);
    out[6] = (signed char)fminf(fmaxf(rintf(v6 * inv_scale), -128.0f), 127.0f);
    out[7] = (signed char)fminf(fmaxf(rintf(v7 * inv_scale), -128.0f), 127.0f);
}
