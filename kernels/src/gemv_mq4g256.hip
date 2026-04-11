#include <hip/hip_runtime.h>

// MagnumQuant GEMV: FWHT-rotated HFQ4-G256.
// x_rot is PRE-ROTATED by the caller (one FWHT pass on x, not per-row).
// The inner loop is identical to standard HFQ4-G256 — zero rotation overhead.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,   // PRE-ROTATED x (FWHT already applied)
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        acc += (scale * (float)(b0 & 0xF) + zero) * x_rot[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x_rot[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x_rot[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x_rot[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x_rot[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x_rot[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x_rot[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x_rot[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// Rotate x in-place: applies FWHT to each 256-element group.
// Called ONCE before the GEMV, not per-row.
// 32 threads per group, 8 elements per thread, ds_swizzle butterfly.
//
// Grid: [groups_total, batch, 1]. For decode the caller passes batch=1 and
// blockIdx.y is always 0 (byte-identical to the pre-batching version).
// For prefill, batch=N; each block processes one group × one batch row.
// x_in / x_out are laid out as [batch × K] row-major.
extern "C" __launch_bounds__(32, 16)
__global__ void mq_rotate_x(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
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

    float v0 = x_in_b[base]     * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];
    float v4 = x_in_b[base + 4] * signs1[d0 + 4];
    float v5 = x_in_b[base + 5] * signs1[d0 + 5];
    float v6 = x_in_b[base + 6] * signs1[d0 + 6];
    float v7 = x_in_b[base + 7] * signs1[d0 + 7];

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

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle
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
    x_out_b[base]     = v0 * s * signs2[d0];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * s * signs2[d0 + 7];
}
