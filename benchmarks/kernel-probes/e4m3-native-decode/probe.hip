// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Can radiowave's native gfx12 OCP FP8 path replace the hand-rolled E4M3
// decode that 29 kernels in kernels/src/ currently inline?
//
// Three questions, in order — a "no" on either of the first two stops the
// consolidation regardless of speed:
//   1. Is the native conversion BIT-IDENTICAL to the hand-rolled decode over
//      the reachable domain? The wire format's correctness depends on the
//      kernel agreeing exactly with the Rust encoder (fp8.rs::e4m3_decode).
//   2. Is it fewer instructions?
//   3. Is the decode hot enough for either to matter?
//
// Reachable domain note: hipfire uses E4M3 for SCALES only. The encoder
// (e4m3_encode_roundup) emits codes 1..=126 with bit 7 clear, so 0x7F (the OCP
// NaN slot) and every negative code are unreachable in practice. The
// hand-rolled decode ignores bit 7 entirely and maps 0x7F to 448.0; a
// spec-correct native converter will disagree on exactly those. This probe
// reports both the full-256 sweep and the reachable subset separately so the
// distinction is explicit rather than assumed.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1201 -std=c++17 probe.hip -o probe
// ISA:
//   hipcc -O3 --offload-arch=gfx1201 -std=c++17 --cuda-device-only -S probe.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

// ── variant A: the idiom currently inlined across 29 kernels ────────────────
// Verbatim from kernels/src/turbo_common.h::cvt_e4m3_scale_to_f32_dq.
__device__ __forceinline__ float e4m3_handrolled(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f; // defensive: NaN code -> max finite (never emitted)
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

// ── variant B: native OCP path (radiowave Fp8Format::E4M3Ocp) ───────────────
#if defined(__gfx1200__) || defined(__gfx1201__)
#define RW_NATIVE_FP8 1
#else
#define RW_NATIVE_FP8 0
#endif

__device__ __forceinline__ float e4m3_native(unsigned char b) {
#if RW_NATIVE_FP8
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
#else
    return e4m3_handrolled(b);
#endif
}

// Exhaustive 256-code sweep, both variants, one thread per code.
__global__ void sweep(float* hand, float* nat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256) return;
    hand[i] = e4m3_handrolled((unsigned char)i);
    nat[i] = e4m3_native((unsigned char)i);
}

// Throughput shape: one scale decode per 32-weight block, matching the
// MFP4-G32-E8 GEMV inner loop (the hottest real consumer).
template <int VARIANT>
__global__ void __launch_bounds__(32, 16) decode_cost(
    const unsigned char* __restrict__ codes, float* __restrict__ out, int blocks) {
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int b = tid; b < blocks; b += 32) {
        const unsigned char c = codes[b];
        const float s = (VARIANT == 0) ? e4m3_handrolled(c) : e4m3_native(c);
        // Consume the scale the way a GEMV would: 32 weights per block.
        acc = fmaf(s, (float)(b & 31), acc);
    }
    out[blockIdx.x * 32 + tid] = acc;
}

#define HC(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            printf("HIP error %s line %d\n", hipGetErrorString(e_), __LINE__); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main() {
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("arch=%s\n", prop.gcnArchName);
    printf("native OCP FP8 compiled in: %s\n\n", RW_NATIVE_FP8 ? "yes" : "NO (non-gfx12)");

    // ── Q1: bit-exactness ───────────────────────────────────────────────────
    float *dh, *dn;
    HC(hipMalloc(&dh, 256 * sizeof(float)));
    HC(hipMalloc(&dn, 256 * sizeof(float)));
    hipLaunchKernelGGL(sweep, dim3(8), dim3(32), 0, 0, dh, dn);
    HC(hipDeviceSynchronize());
    std::vector<float> h(256), n(256);
    HC(hipMemcpy(h.data(), dh, 256 * sizeof(float), hipMemcpyDeviceToHost));
    HC(hipMemcpy(n.data(), dn, 256 * sizeof(float), hipMemcpyDeviceToHost));

    int diff_all = 0, diff_reach = 0;
    printf("first disagreements (code: handrolled vs native):\n");
    int shown = 0;
    for (int i = 0; i < 256; ++i) {
        const bool bitsame =
            (std::isnan(h[i]) && std::isnan(n[i])) ||
            (__builtin_bit_cast(uint32_t, h[i]) == __builtin_bit_cast(uint32_t, n[i]));
        if (bitsame) continue;
        ++diff_all;
        // Reachable = what e4m3_encode_roundup can actually emit: 1..=126.
        const bool reachable = (i >= 1 && i <= 126);
        if (reachable) ++diff_reach;
        if (shown < 8) {
            printf("  0x%02X: %-14g vs %-14g%s\n", i, h[i], n[i],
                   reachable ? "   <-- REACHABLE" : "");
            ++shown;
        }
    }
    if (diff_all == 0) printf("  (none)\n");
    printf("\n  mismatches over all 256 codes : %d\n", diff_all);
    printf("  mismatches over reachable 1..126: %d  %s\n\n", diff_reach,
           diff_reach == 0 ? "<- SAFE to substitute" : "<- UNSAFE");

    // ── Q3: decode throughput ───────────────────────────────────────────────
    const int blocks = 1 << 16;
    std::vector<unsigned char> codes(blocks);
    for (int i = 0; i < blocks; ++i) codes[i] = (unsigned char)(1 + (i * 37) % 126);
    unsigned char* dc;
    float* dout;
    HC(hipMalloc(&dc, blocks));
    HC(hipMalloc(&dout, 4096 * 32 * sizeof(float)));
    HC(hipMemcpy(dc, codes.data(), blocks, hipMemcpyHostToDevice));

    auto bench = [&](int variant) {
        for (int i = 0; i < 50; ++i) {
            if (variant == 0)
                hipLaunchKernelGGL((decode_cost<0>), dim3(4096), dim3(32), 0, 0, dc, dout, blocks);
            else
                hipLaunchKernelGGL((decode_cost<1>), dim3(4096), dim3(32), 0, 0, dc, dout, blocks);
        }
        hipDeviceSynchronize();
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        std::vector<float> t;
        for (int r = 0; r < 9; ++r) {
            hipEventRecord(a);
            for (int i = 0; i < 64; ++i) {
                if (variant == 0)
                    hipLaunchKernelGGL((decode_cost<0>), dim3(4096), dim3(32), 0, 0, dc, dout, blocks);
                else
                    hipLaunchKernelGGL((decode_cost<1>), dim3(4096), dim3(32), 0, 0, dc, dout, blocks);
            }
            hipEventRecord(b);
            hipEventSynchronize(b);
            float ms;
            hipEventElapsedTime(&ms, a, b);
            t.push_back(ms / 64);
        }
        std::sort(t.begin(), t.end());
        hipEventDestroy(a);
        hipEventDestroy(b);
        return (double)t[t.size() / 2];
    };

    // Interleaved warmup so DPM state does not alias into the delta.
    bench(0);
    bench(1);
    const double th = bench(0);
    const double tn = bench(1);
    printf("decode throughput (median of 9, 64 batched launches each):\n");
    printf("  hand-rolled : %.5f ms\n", th);
    printf("  native OCP  : %.5f ms   (%+.2f%%)\n", tn, (tn / th - 1.0) * 100.0);
    printf("\n%s\n", tn < th ? "native is faster" : "hand-rolled is faster or equal");
    return 0;
}
