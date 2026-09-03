// Escha-W2 activation transforms.
//   in : xh = f16( H128(x * rin) * RS )
//   out: y  = f16( H128(mid) * RS * rout )
//
// H128 is the UNNORMALISED 128-point Walsh-Hadamard in Sylvester (natural)
// order. Escha folds its sign flips into rin/rout, so unlike hipfire's
// gemv_mq4g128 there is no sign-seed stage here.
//
// A pruned output channel (rout == 0) must come out EXACTLY zero — do not
// reorder the final multiply in a way that could produce -0.0 or a denormal.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ESCHA_RS 0.0883883476f   // 1/sqrt(128)

// This ROCm toolchain's __float2half(-0.0f) returns +0.0 (bits 0x0000)
// instead of the sign-preserving -0.0 (0x8000): confirmed in isolation with
// a standalone multiply kernel (see task-8-report.md), and it is the ONLY
// divergence from escha_ref across the whole G3 sweep — every non-zero
// value, including values that underflow fp16 (e.g. 1e-10, 1e-30, either
// sign), round correctly. escha_ref's `f16_rne` uses `half::f16::from_f32`,
// which does preserve the sign of an exact zero (a pruned `rout` channel
// multiplying a negative pre-scale value yields -0.0 there), so the G3 gate
// requires bit-exact agreement including that sign. Route only the exact
// -0.0f or 0.0f case around __float2half by hand: the top 16 bits of an
// IEEE-754 float zero (sign + 8 exponent + top 8 mantissa, all zero except
// sign) collapse exactly onto the fp16 zero's sign+exponent+mantissa layout,
// so `bits >> 16` is the correct fp16 encoding for either signed zero.
__device__ __forceinline__ __half f2h_rne(float v) {
    if (v == 0.0f) {
        unsigned int bits = __float_as_uint(v);
        return __ushort_as_half((unsigned short)(bits >> 16));
    }
    return __float2half(v);
}

__device__ __forceinline__ void h128_block(float* v) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

// One block per 128-channel group. 128 threads cooperate via LDS.
extern "C" __global__ void escha_h128_in(
    const float* __restrict__ x, const float* __restrict__ rin,
    __half* __restrict__ xh, int n) {
    __shared__ float s[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    s[t] = x[idx] * rin[idx];
    __syncthreads();
    if (t == 0) h128_block(s);
    __syncthreads();
    xh[idx] = f2h_rne(s[t] * ESCHA_RS);
}

extern "C" __global__ void escha_h128_out(
    const float* __restrict__ mid, const float* __restrict__ rout,
    __half* __restrict__ y, int n) {
    __shared__ float s[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    s[t] = mid[idx];
    __syncthreads();
    if (t == 0) h128_block(s);
    __syncthreads();
    y[idx] = f2h_rne(s[t] * ESCHA_RS * rout[idx]);
}
