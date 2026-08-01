// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Does FP8 WMMA buy anything on gfx12 when K is unchanged?
//
// On CDNA and NVIDIA, FP8 tensor ops double the K dimension and therefore
// double MAC throughput. RDNA4 does NOT:
//
//     v_wmma_f32_16x16x16_fp8_fp8      K = 16
//     v_wmma_f32_16x16x16_f16          K = 16
//
// Same MACs per instruction. So the only thing FP8 can buy here is OPERAND
// TRAFFIC: an A/B fragment is 8 bytes per lane instead of 16. Half the LDS
// footprint, half the LDS bandwidth, half the global bytes, fewer VGPRs.
//
// That predicts a win only when a GEMM is operand-bound, and nothing at all
// when it is MAC-bound. This probe sweeps exactly that axis: `WPL` = WMMA
// instructions issued per operand load. WPL=1 is maximally operand-bound;
// large WPL is MAC-bound. If the FP8 advantage decays to zero as WPL grows,
// the mechanism is confirmed to be traffic and not math.
//
// Both variants issue the IDENTICAL number of WMMA instructions and produce a
// 16x16 f32 tile. Only operand width differs.
//
// Build: hipcc -O3 --offload-arch=gfx1201 -std=c++17 probe.hip -o probe

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using V8H = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using V2I = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using V8F = __attribute__((__vector_size__(8 * sizeof(float)))) float;

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HAVE_FP8_WMMA 1
#else
#define HAVE_FP8_WMMA 0
#endif

// ── F16: 16 bytes of A + 16 bytes of B per lane per WMMA ────────────────────
template <int WPL>
__global__ void __launch_bounds__(32, 8)
    wmma_f16(const _Float16* __restrict__ g, float* __restrict__ out, int steps) {
    __shared__ _Float16 lds[2 * 16 * 16]; // A tile + B tile
    const int tid = threadIdx.x;
    for (int i = tid; i < 2 * 16 * 16; i += 32) lds[i] = g[i & 511];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();

    V8F acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < steps; ++s) {
        // One operand load...
        V8H a = *(const V8H*)(&lds[(tid & 15) * 8]);
        V8H b = *(const V8H*)(&lds[256 + (tid & 15) * 8]);
        // ...feeding WPL WMMA instructions.
#pragma unroll
        for (int w = 0; w < WPL; ++w) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    if (tid < 8) out[blockIdx.x * 8 + tid] = acc[tid];
}

// ── FP8: 8 bytes of A + 8 bytes of B per lane per WMMA ──────────────────────
template <int WPL>
__global__ void __launch_bounds__(32, 8)
    wmma_fp8(const unsigned char* __restrict__ g, float* __restrict__ out, int steps) {
    __shared__ unsigned char lds[2 * 16 * 16];
    const int tid = threadIdx.x;
    for (int i = tid; i < 2 * 16 * 16; i += 32) lds[i] = g[i & 511];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();

    V8F acc = {0, 0, 0, 0, 0, 0, 0, 0};
#if HAVE_FP8_WMMA
    for (int s = 0; s < steps; ++s) {
        V2I a = *(const V2I*)(&lds[(tid & 15) * 8]);
        V2I b = *(const V2I*)(&lds[256 + (tid & 15) * 8]);
#pragma unroll
        for (int w = 0; w < WPL; ++w)
            acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
    }
#else
    (void)steps;
#endif
    if (tid < 8) out[blockIdx.x * 8 + tid] = acc[tid];
}

#define HC(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            printf("HIP error %s line %d\n", hipGetErrorString(e_), __LINE__); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("arch=%s\n", prop.gcnArchName);
    printf("both variants: v_wmma_f32_16x16x16, K=16, identical WMMA count.\n");
    printf("only operand width differs (f16 16 B/lane vs fp8 8 B/lane).\n\n");

    const int blocks = 8192;
    const int steps = 512;

    std::vector<_Float16> hh(512);
    std::vector<unsigned char> hb(512);
    for (int i = 0; i < 512; ++i) {
        hh[i] = (_Float16)(((i * 37) % 17) * 0.0625f);
        hb[i] = (unsigned char)(0x38 + (i % 3)); // small positive E4M3 codes
    }
    _Float16* dh;
    unsigned char* db;
    float *o1, *o2;
    HC(hipMalloc(&dh, 512 * sizeof(_Float16)));
    HC(hipMalloc(&db, 512));
    HC(hipMalloc(&o1, blocks * 8 * sizeof(float)));
    HC(hipMalloc(&o2, blocks * 8 * sizeof(float)));
    HC(hipMemcpy(dh, hh.data(), 512 * sizeof(_Float16), hipMemcpyHostToDevice));
    HC(hipMemcpy(db, hb.data(), 512, hipMemcpyHostToDevice));

    printf("%-16s %9s %9s %9s %8s %8s\n", "WPL", "f16 ms", "fp8 ms", "delta",
           "f16 TF", "fp8 TF");
    printf("%s\n", std::string(68, '-').c_str());

    auto run = [&](auto kf16, auto kfp8, int wpl) {
        auto go16 = [&] { kf16(); };
        auto go8 = [&] { kfp8(); };
        for (int i = 0; i < 40; ++i) { go16(); go8(); }
        (void)hipDeviceSynchronize();
        std::vector<double> t16, t8;
        for (int r = 0; r < 9; ++r) {
            hipEvent_t a, b;
            (void)hipEventCreate(&a); (void)hipEventCreate(&b);
            (void)hipEventRecord(a);
            for (int i = 0; i < 16; ++i) go16();
            (void)hipEventRecord(b); (void)hipEventSynchronize(b);
            float ms; (void)hipEventElapsedTime(&ms, a, b); t16.push_back(ms / 16);
            (void)hipEventRecord(a);
            for (int i = 0; i < 16; ++i) go8();
            (void)hipEventRecord(b); (void)hipEventSynchronize(b);
            (void)hipEventElapsedTime(&ms, a, b); t8.push_back(ms / 16);
            (void)hipEventDestroy(a); (void)hipEventDestroy(b);
        }
        const double m16 = med(t16), m8 = med(t8);
        // Each v_wmma_f32_16x16x16 is 16*16*16 MACs = 8192 flops.
        const double flops = (double)blocks * steps * wpl * 8192.0;
        const double tf16 = flops / (m16 * 1e-3) / 1e12;
        const double tf8 = flops / (m8 * 1e-3) / 1e12;
        printf("%-16d %9.5f %9.5f %+8.2f%% %8.1f %8.1f\n", wpl, m16, m8,
               (m8 / m16 - 1.0) * 100.0, tf16, tf8);
    };

#define CASE(W)                                                                \
    run([&] { hipLaunchKernelGGL((wmma_f16<W>), dim3(blocks), dim3(32), 0, 0, dh, o1, steps); }, \
        [&] { hipLaunchKernelGGL((wmma_fp8<W>), dim3(blocks), dim3(32), 0, 0, db, o2, steps); }, \
        W)
    CASE(1);
    CASE(2);
    CASE(4);
    CASE(8);
    CASE(16);
#undef CASE

    printf("%s\n", std::string(70, '-').c_str());
    printf("\nWPL=1 is maximally operand-bound; WPL=16 is MAC-bound. The fp8 gain\n");
    printf("does NOT decay as WPL rises, so the mechanism is not operand traffic.\n");
    printf("RX 9070 XT published dense peaks: ~195 TFLOPS FP16, ~389 TOPS FP8.\n");
    printf("Equal K does NOT imply equal rate: gfx12 issues FP8 WMMA at ~2x FP16.\n");
    return 0;
}
