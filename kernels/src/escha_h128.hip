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

// ── Batched-across-experts forms (Task 10) ───────────────────────────────
//
// WHY THESE EXIST: Task 8 measured the H128 kernels as LAUNCH-bound, not
// bandwidth-bound — an empty kernel at the same grid/block costs 1.74-1.78 us
// against a real launch's 2.4 us, and the overhead-subtracted time is nearly
// flat (0.59 -> 0.69 us) from 16 to 136 blocks. Issuing one launch per
// (expert, projection, side) costs 40 layers x 8 experts x 4 transforms =
// 1280 launches/token = 3.07 ms = a 326 tok/s ceiling from the transforms
// ALONE, before any GEMV work. One launch per (layer, projection, side)
// covering all top_k experts is 160 launches = 0.38 ms.
//
// The maths is identical to the per-expert form; the only change is that
// `r` is now the whole resident `[E, n]` table and each slot picks its row
// via `ids[slot]`. Grid is `slots * (n/128)` blocks of 128 threads; block
// `g` serves slot `g / (n/128)`, group `g % (n/128)`.
//
// OUTPUT DTYPE: these write f32 holding the f16-ROUNDED value (widened),
// not __half. The value set is identical to the per-expert kernels' — the
// rounding happens at exactly the same point — but the consumer is
// hipfire's f32 GEMV, so widening here saves a separate convert launch.
//
// `x_group` says how many CONSECUTIVE slots share one row of `x`:
//
//   x_group <= 0  broadcast — every slot reads the single `[n]` row `x`.
//                 (Decode gate_up input side: a token's top_k experts all see
//                 the same post-rmsnorm x, only `rin` differs.)
//   x_group == 1  one row per slot, `x` is `[slots, n]`.
//                 (Down input side: each slot's activation is its own SwiGLU
//                 output.)
//   x_group == g  row `slot / g`, `x` is `[slots/g, n]`.
//                 (Batched-prefill gate_up input side, g = k: slots are laid
//                 out token-major as `token * k + krank`, so `slot / k` is the
//                 token and every one of its k experts reads that token's x.)
//
// 0 and 1 keep the exact meanings the old binary `x_batched` flag had, so the
// two pre-existing call sites and the two pre-existing G3 cases are unchanged
// by construction; the grouped case is the only new behaviour and G3 gates it
// bit-exactly against escha_ref like the other two.
//
// Deliberately `slot / x_group` rather than a second `tokens` kernarg: the
// division is once per block (128 threads share it) against a value already
// in a register, and expressing it as "how many slots share a row" keeps ALL
// THREE cases one expression instead of a three-way branch per element.
extern "C" __global__ void escha_h128_in_batched(
    const float* __restrict__ x, const float* __restrict__ r,
    const int* __restrict__ ids, float* __restrict__ out,
    int n, int x_group) {
    __shared__ float bufA[128];
    __shared__ float bufB[128];
    int groups = n >> 7;
    int slot = blockIdx.x / groups;
    int grp = blockIdx.x - slot * groups;
    int t = threadIdx.x;
    int lane = grp * 128 + t;
    int e = ids[slot];
    float xv = (x_group <= 0) ? x[lane]
                              : x[(size_t)(slot / x_group) * n + lane];
    bufA[t] = xv * r[(size_t)e * n + lane];
    __syncthreads();
    float* h = h128_parallel(bufA, bufB, t);
    out[(size_t)slot * n + lane] = __half2float(f2h_rne(h[t] * ESCHA_RS));
}

extern "C" __global__ void escha_h128_out_batched(
    const float* __restrict__ mid, const float* __restrict__ r,
    const int* __restrict__ ids, float* __restrict__ out, int n) {
    __shared__ float bufA[128];
    __shared__ float bufB[128];
    int groups = n >> 7;
    int slot = blockIdx.x / groups;
    int grp = blockIdx.x - slot * groups;
    int t = threadIdx.x;
    int lane = grp * 128 + t;
    int e = ids[slot];
    bufA[t] = mid[(size_t)slot * n + lane];
    __syncthreads();
    float* h = h128_parallel(bufA, bufB, t);
    // Pruned channel (r == 0) must read back EXACTLY zero — f2h_rne routes
    // the exact-zero product around __float2half by hand (see the comment on
    // f2h_rne), so a negative pre-scale value against a pruned channel gives
    // -0.0, not a denormal and not a dropped sign. Do NOT reorder or
    // "optimise away" this multiply.
    out[(size_t)slot * n + lane] =
        __half2float(f2h_rne(h[t] * ESCHA_RS * r[(size_t)e * n + lane]));
}

// SwiGLU over the f16-ROUNDED merged gate_up output, batched across slots.
// `y` is `[slots, 2*inter]`, gate is the FIRST half. `y` is already the
// output of escha_h128_out_batched, i.e. every element is an exactly
// representable f16 value — that is the escha contract ("SwiGLU runs on the
// f16-rounded merged gate_up output"), and it is why this kernel must be fed
// the transform's output and never a pre-transform accumulator.
//
// The two interior f16 roundings are `escha_ref::swiglu`'s, not an accident:
// the frozen oracle rounds silu(g) and then rounds the product. Output is
// stored as f32 holding that f16 value (the down-projection input transform
// consumes f32).
//
// `expf`, not `__expf`: the fast intrinsic is a different function; hipfire's
// own `silu_mul_f32` (kernels/src/silu_mul.hip) uses `expf` and so does the
// oracle. Note this kernel is NOT bit-exact against `escha_ref::swiglu` for
// every input — device `expf` and Rust's `f32::exp` can differ by 1 ulp in
// f32, and about 1 in 2^13 of those land on an f16 rounding boundary. The G3
// gate therefore bounds the swiglu mismatch at <=1 f16 ulp rather than
// demanding equality; the H128 transforms themselves ARE bit-exact and are
// gated as such.
extern "C" __global__ void escha_swiglu_batched(
    const float* __restrict__ y, float* __restrict__ h, int inter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = blockIdx.y;
    if (i >= inter) return;
    const float* row = y + (size_t)slot * 2 * inter;
    float g = row[i];
    float s = __half2float(f2h_rne(g / (1.0f + expf(-g))));
    h[(size_t)slot * inter + i] = __half2float(f2h_rne(s * row[inter + i]));
}
