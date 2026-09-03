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
//
// These kernels run on every token, on both sides of every escha matmul —
// the hot decode path, unlike escha_decode_tiles (once per expert at load
// time) — so the butterfly is parallel across all 128 threads of the block,
// not a single thread doing all 7 stages serially. See h128_parallel below.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ESCHA_RS 0.0883883476f   // 1/sqrt(128)

// NOT a general "__float2half(-0.0f) is broken on this toolchain" bug — a
// bare literal -0.0f converts correctly (confirmed both as an immediate and
// loaded from device memory: 0x8000, correct). The trigger is narrower: on
// this ROCm 7.2.2/gfx1151 build, __float2half loses the sign bit only when
// its argument is a RUNTIME MULTIPLY-PRODUCED exact zero — the shape this
// kernel's own `v * RS` / `v * RS * rout` produces when a pruned (`rout ==
// 0`) channel meets a negative `v`. A reviewer independently reproduced this
// standalone against `/opt/rocm-7.2.2/bin/hipcc --offload-arch=gfx1151`:
//
//   __float2half(-0.0f)                   = 0x8000  (literal, CORRECT)
//   __float2half(-0.0f)                   = 0x8000  (loaded from memory, CORRECT)
//   __float2half(-1e-9f)                  = 0x8000  (CORRECT)
//   __float2half(-1e-30f)                 = 0x8000  (CORRECT)
//   __float2half(a[i]*c[i])  where product == -0.0  -> 0x0000  (WRONG, want 0x8000)
//   __float2half(a[i]*CONST*c[i])  (this kernel's exact shape) -> 0x0000  (WRONG)
//   float pre = a*b*c; __float2half(pre)  -> 0x0000  (WRONG)
//
// Critically, a tiny NONZERO underflowing product computed the identical way
// DOES preserve sign (-1e-10 * 1e-10 -> 0x8000, correct) — so the defect is
// scoped to exact-zero products, not underflow in general. That's why the
// `v == 0.0f` guard below is correctly targeted at exact zero and must NOT
// be broadened to "small magnitude" or any other range.
//
// Auditor's test, if you're checking another __float2half call site for the
// same bug: converting a bare -0.0f literal will NOT reproduce it and will
// wrongly look safe. You must reproduce a -0.0 as the runtime result of a
// multiply (or chain of multiplies) feeding directly into __float2half.
//
// This is the ONLY divergence from escha_ref across the whole G3 sweep —
// every non-zero value, including values that underflow fp16 (e.g. 1e-10,
// 1e-30, either sign), round correctly. escha_ref's `f16_rne` uses
// `half::f16::from_f32`, which does preserve the sign of an exact zero (a
// pruned `rout` channel multiplying a negative pre-scale value yields -0.0
// there), so the G3 gate requires bit-exact agreement including that sign.
// Route only the exact -0.0f or 0.0f case around __float2half by hand: the
// top 16 bits of an IEEE-754 float zero (sign + 8 exponent + top 8 mantissa,
// all zero except sign) collapse exactly onto the fp16 zero's
// sign+exponent+mantissa layout, so `bits >> 16` is the correct fp16
// encoding for either signed zero.
//
// Is the workaround needed for CORRECTNESS, or only for bit-exactness with
// the CPU oracle? Only the latter. IEEE-754 defines -0.0 == 0.0, and nothing
// downstream of this kernel in the escha pipeline divides by an activation
// (which is the only operation where the sign of a zero would change a
// result), so the pruned-channel contract — "a pruned channel reads back as
// exactly zero" — holds numerically with EITHER sign of zero. Model output
// would be identical without this workaround. It exists purely to keep this
// kernel bit-for-bit against `escha_ref` for the G3 gate; a port of this
// workaround to another kernel should be justified by its own bit-exactness
// requirement, not assumed to be needed for correctness.
__device__ __forceinline__ __half f2h_rne(float v) {
    if (v == 0.0f) {
        unsigned int bits = __float_as_uint(v);
        return __ushort_as_half((unsigned short)(bits >> 16));
    }
    return __float2half(v);
}

// Parallel 128-point Hadamard butterfly, one thread per element, 7 stages
// with a `__syncthreads()` between each.
//
// Ping-pongs between two LDS buffers instead of updating `bufA` in place:
// the naive single-thread version reads `v[j]`/`v[j+h]` and writes both
// before any other pair is touched, so within one stage a pair's inputs are
// always the PREVIOUS stage's outputs. Doing this in place with 128 threads
// racing would let a "high" thread (writing v[j+h]) read v[j] after some
// other "low" thread has already overwritten it, or vice versa. Ping-ponging
// reads every stage entirely from the buffer the previous stage finished
// (and `__syncthreads()`-published), so every thread's inputs are the exact
// pre-stage values regardless of scheduling — same operand pairing as the
// reference's sequential stage-by-stage sweep, just computed out of order,
// and each pair's `a+b`/`a-b` depends only on that pair's own two operands.
// Order-independence across pairs is what makes this bit-exact with the
// naive kernel and with escha_ref: no reassociation, no changed rounding,
// only a different (but result-identical) schedule of the same 64 adds and
// 64 subtracts per stage.
__device__ __forceinline__ float* h128_parallel(float* bufA, float* bufB, int t) {
    float* cur = bufA;
    float* nxt = bufB;
#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
        int period = h << 1;
        bool low = (t & (period - 1)) < h;
        int j = low ? t : (t - h);
        float a = cur[j];
        float b = cur[j + h];
        nxt[t] = low ? (a + b) : (a - b);
        __syncthreads();
        float* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }
    return cur;
}

// One block per 128-channel group. 128 threads cooperate via LDS.
extern "C" __global__ void escha_h128_in(
    const float* __restrict__ x, const float* __restrict__ rin,
    __half* __restrict__ xh, int n) {
    __shared__ float bufA[128];
    __shared__ float bufB[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    bufA[t] = x[idx] * rin[idx];
    __syncthreads();
    float* out = h128_parallel(bufA, bufB, t);
    xh[idx] = f2h_rne(out[t] * ESCHA_RS);
}

extern "C" __global__ void escha_h128_out(
    const float* __restrict__ mid, const float* __restrict__ rout,
    __half* __restrict__ y, int n) {
    __shared__ float bufA[128];
    __shared__ float bufB[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    bufA[t] = mid[idx];
    __syncthreads();
    float* out = h128_parallel(bufA, bufB, t);
    y[idx] = f2h_rne(out[t] * ESCHA_RS * rout[idx]);
}
