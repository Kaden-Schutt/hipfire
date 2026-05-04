#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// SmoothQuant-style per-channel activation preshift for the gfx12 iu4 K=32
// path:
//
//   y[t][c] = (x[t][c] - mu_a[c]) * inv_s_a[c]
//
// `mu_a` and `inv_s_a` are stored as FP16 to keep the sidecar small. The
// preshift writes FP32 because the downstream Q4_1 quantizer
// (`quantize_q4_1_mmq_ds4_gfx12`) reads FP32. Per-call cost: one HBM read
// of x, one HBM write of y, plus a tiny constant-table read of mu/inv_s
// (sized to k = 4096-12288 channels = 16-48 KB FP16, easily cached).
//
// Threading: same as the Q4_1 quantizer — 4 elements per thread, blockDim
// = 256, grid = ((K+1023)/1024, N, 1). Each thread reads/writes 1 float4
// into [t][c..c+4] for c = (blockIdx.x * 256 + threadIdx.x) * 4. mu/inv_s
// are read aligned at the same c offsets — stride is k FP16 (= 2 bytes) so
// 4 channels = 8 bytes = one float2 of FP16 pairs.
//
// Stub for non-gfx12 archs to keep the cross-arch JIT happy. Calibrated
// iu4 dispatch is gfx12-only at present; other archs that wire the
// SmoothQuant idea later should add their own preshift kernel matching
// their preferred quantizer's input shape.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void activation_preshift_gfx12(
    const float*,
    const __half*,
    const __half*,
    float*,
    int,
    int
) {}

#else

extern "C" __global__ void activation_preshift_gfx12(
    const float* __restrict__ X,        // [N x K] FP32 row-major
    const __half* __restrict__ mu_a,    // [K] FP16
    const __half* __restrict__ inv_s_a, // [K] FP16 (1 / s_a, with floor)
    float* __restrict__ Y,              // [N x K] FP32 row-major (out)
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || c0 >= K) return;

    // Bulk-load 4 channels of mu / inv_s as FP16 pairs.
    float mu0, mu1, mu2, mu3;
    float is0, is1, is2, is3;
    if (c0 + 3 < K) {
        const __half2 mu_lo = *(const __half2*)(mu_a + c0);
        const __half2 mu_hi = *(const __half2*)(mu_a + c0 + 2);
        const __half2 is_lo = *(const __half2*)(inv_s_a + c0);
        const __half2 is_hi = *(const __half2*)(inv_s_a + c0 + 2);
        const float2 mu_lo_f = __half22float2(mu_lo);
        const float2 mu_hi_f = __half22float2(mu_hi);
        const float2 is_lo_f = __half22float2(is_lo);
        const float2 is_hi_f = __half22float2(is_hi);
        mu0 = mu_lo_f.x; mu1 = mu_lo_f.y; mu2 = mu_hi_f.x; mu3 = mu_hi_f.y;
        is0 = is_lo_f.x; is1 = is_lo_f.y; is2 = is_hi_f.x; is3 = is_hi_f.y;
    } else {
        mu0 = (c0 + 0 < K) ? __half2float(mu_a[c0 + 0]) : 0.0f;
        mu1 = (c0 + 1 < K) ? __half2float(mu_a[c0 + 1]) : 0.0f;
        mu2 = (c0 + 2 < K) ? __half2float(mu_a[c0 + 2]) : 0.0f;
        mu3 = (c0 + 3 < K) ? __half2float(mu_a[c0 + 3]) : 0.0f;
        is0 = (c0 + 0 < K) ? __half2float(inv_s_a[c0 + 0]) : 1.0f;
        is1 = (c0 + 1 < K) ? __half2float(inv_s_a[c0 + 1]) : 1.0f;
        is2 = (c0 + 2 < K) ? __half2float(inv_s_a[c0 + 2]) : 1.0f;
        is3 = (c0 + 3 < K) ? __half2float(inv_s_a[c0 + 3]) : 1.0f;
    }

    const long long base = (long long)token * K + c0;
    if (c0 + 3 < K) {
        const float4 xi = *(const float4*)(X + base);
        float4 yi;
        yi.x = (xi.x - mu0) * is0;
        yi.y = (xi.y - mu1) * is1;
        yi.z = (xi.z - mu2) * is2;
        yi.w = (xi.w - mu3) * is3;
        *(float4*)(Y + base) = yi;
    } else {
        if (c0 + 0 < K) Y[base + 0] = (X[base + 0] - mu0) * is0;
        if (c0 + 1 < K) Y[base + 1] = (X[base + 1] - mu1) * is1;
        if (c0 + 2 < K) Y[base + 2] = (X[base + 2] - mu2) * is2;
        if (c0 + 3 < K) Y[base + 3] = (X[base + 3] - mu3) * is3;
    }
}

#endif
