// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Device/host parity check for kernels/src/convert_f32_to_bf16.hip.
//
// The GPU convert must be bit-identical to the host oracle
// `round_to_bf16` (crates/hipfire-arch-deepseek4/src/parent/codec.rs:92-110),
// because calibration stages activations on device while the reference /
// quantizer path rounds on host. A half-ULP disagreement silently perturbs
// every captured activation.
//
// Build: hipcc --offload-arch=gfx942 -O3 scripts/cvt_bf16_parity.hip -o cvt_parity

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

#include "../kernels/src/convert_f32_to_bf16.hip"

// Host oracle — transcribed from codec.rs round_to_bf16.
static uint16_t host_round_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {   // NaN
        uint32_t sign = (bits >> 16) & 0x8000u;
        return (uint16_t)(sign | 0x7fc0u);
    }
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t lower = bits & 0xffffu;
    uint32_t round_bit = (lower >> 15) & 1u;
    uint32_t sticky = (lower & 0x7fffu) != 0u ? 1u : 0u;
    uint32_t top = bits >> 16;
    if (round_bit == 1u && (sticky == 1u || lsb == 1u)) top += 1u;
    return (uint16_t)(top & 0xffffu);
}

static float f32_from_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

int main() {
    std::vector<float> in;

    // ── Adversarial cases: exact ties, tie-to-even both parities, boundaries ──
    in.push_back(0.0f);
    in.push_back(-0.0f);
    in.push_back(1.0f);
    in.push_back(-1.0f);
    in.push_back(f32_from_bits(0x7f800000u));   // +Inf
    in.push_back(f32_from_bits(0xff800000u));   // -Inf
    in.push_back(f32_from_bits(0x7fc00000u));   // quiet NaN
    in.push_back(f32_from_bits(0xffc00001u));   // signalling-ish NaN, sign set
    in.push_back(f32_from_bits(0x7f7fffffu));   // max finite f32 -> overflows to Inf in bf16
    in.push_back(f32_from_bits(0x00000001u));   // smallest denormal
    in.push_back(f32_from_bits(0x80000001u));   // -smallest denormal

    // Exact ties (lower == 0x8000, sticky == 0): must round to EVEN.
    // lsb == 0 -> stay; lsb == 1 -> round up. Both parities covered.
    for (uint32_t topbits : {0x3f800000u, 0x3f810000u, 0x40000000u, 0x40010000u,
                             0xbf800000u, 0xbf810000u}) {
        in.push_back(f32_from_bits(topbits | 0x8000u));
    }
    // Just below / just above a tie (sticky set) — must round by magnitude.
    for (uint32_t topbits : {0x3f800000u, 0x3f810000u}) {
        in.push_back(f32_from_bits(topbits | 0x7fffu));
        in.push_back(f32_from_bits(topbits | 0x8001u));
    }

    // ── Dense sweep over exponents and mantissa low bits ──
    for (uint32_t e = 100; e < 160; ++e) {
        for (uint32_t m = 0; m < 512; ++m) {
            uint32_t mant = (m * 8191u) & 0x7fffffu;   // spread across mantissa
            in.push_back(f32_from_bits((e << 23) | mant));
            in.push_back(f32_from_bits(0x80000000u | (e << 23) | mant));
        }
    }

    const int n = (int)in.size();
    float*    d_src = nullptr;
    uint16_t* d_dst = nullptr;
    if (hipMalloc(&d_src, (size_t)n * 4) != hipSuccess) { printf("hipMalloc src failed\n"); return 2; }
    if (hipMalloc(&d_dst, (size_t)n * 2) != hipSuccess) { printf("hipMalloc dst failed\n"); return 2; }
    hipMemcpy(d_src, in.data(), (size_t)n * 4, hipMemcpyHostToDevice);
    hipMemset(d_dst, 0xAB, (size_t)n * 2);

    int block = 256;
    int grid  = (n + block - 1) / block;
    if (grid > 1024) grid = 1024;                 // exercise the grid-stride loop
    hipLaunchKernelGGL(convert_f32_to_bf16, dim3(grid), dim3(block), 0, 0,
                       d_src, (__bf16*)d_dst, n);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("kernel launch failed: %s\n", hipGetErrorString(err)); return 2; }

    std::vector<uint16_t> got(n);
    hipMemcpy(got.data(), d_dst, (size_t)n * 2, hipMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        uint16_t want = host_round_to_bf16(in[i]);
        if (got[i] != want) {
            if (mismatches < 12) {
                uint32_t b; memcpy(&b, &in[i], 4);
                printf("  MISMATCH i=%d src=0x%08x (%g)  gpu=0x%04x  host=0x%04x\n",
                       i, b, (double)in[i], got[i], want);
            }
            ++mismatches;
        }
    }

    hipFree(d_src);
    hipFree(d_dst);

    printf("convert_f32_to_bf16 parity: %d cases, %d mismatches -> %s\n",
           n, mismatches, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0 ? 0 : 1;
}
