// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// The wave32 reduction tail: __shfl_down vs DPP.
//
// 267 kernels in kernels/src/ end with the same 5-step shuffle reduction:
//
//     for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
//
// On gfx1201 that lowers to FIVE `ds_bpermute_b32`, each separated by a full
// `s_wait_dscnt 0x0` — i.e. five serialized LDS round-trips, in the VALU-idle
// tail of every GEMV row. DPP does cross-lane in the VALU with no LDS traffic
// and no dscnt wait.
//
// This matters more than it looks for small-K GEMV: at K=1024 the MQ4 inner
// loop is only 4 group iterations, so a 5-step LDS tail is a real fraction of
// the kernel rather than a rounding error.
//
// Correctness is checked first — a faster reduction that reduces wrong is
// worthless, and DPP bound_ctrl/row_mask semantics are easy to get subtly
// wrong at the row boundary.
//
// Build: hipcc -O3 --offload-arch=gfx1201 -std=c++17 probe.hip -o probe

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

// ── A: the idiom in 267 files ───────────────────────────────────────────────
__device__ __forceinline__ float reduce_shfl(float acc) {
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    return acc;
}

// ── B: DPP row_xmask butterfly within each row of 16, then one cross-row
// shuffle. row_xmask:N is a lane-XOR inside the 16-lane row, so every lane
// ends with its row's total (unlike row_shr, which lands the sum in the LAST
// lane — the bug the correctness gate caught on the first attempt).
//
// Cost: 4 VALU DPP ops + ONE ds_bpermute, versus five ds_bpermute each
// separated by a full s_wait_dscnt.
#define DPP_XMASK_ADD(x, n)                                                    \
    (x) += __builtin_bit_cast(                                                 \
        float, __builtin_amdgcn_update_dpp(                                    \
                   0, __builtin_bit_cast(int, (x)), 0x160 | (n), 0xF, 0xF, true))

__device__ __forceinline__ float reduce_dpp(float acc) {
    DPP_XMASK_ADD(acc, 1);
    DPP_XMASK_ADD(acc, 2);
    DPP_XMASK_ADD(acc, 4);
    DPP_XMASK_ADD(acc, 8);
    // Rows 0 and 1 each hold their own total; one shuffle folds them.
    acc += __shfl_xor(acc, 16);
    return acc;
}

__global__ void check(const float* in, float* out_shfl, float* out_dpp) {
    const int tid = threadIdx.x;
    const float v = in[blockIdx.x * 32 + tid];
    const float a = reduce_shfl(v);
    const float b = reduce_dpp(v);
    if (tid == 0) {
        out_shfl[blockIdx.x] = a;
        out_dpp[blockIdx.x] = b;
    }
}

// Realistic shape: MQ4 GEMV at K=1024 is 4 group iterations then the tail.
template <int VARIANT>
__global__ void __launch_bounds__(32, 16)
    gemv_like(const float* __restrict__ x, float* __restrict__ y, int iters) {
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int g = 0; g < iters; ++g) acc = fmaf(x[g * 32 + tid], (float)(g + 1), acc);
    acc = (VARIANT == 0) ? reduce_shfl(acc) : reduce_dpp(acc);
    if (tid == 0) y[blockIdx.x] = acc;
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
    printf("arch=%s\n\n", prop.gcnArchName);

    // ── correctness ─────────────────────────────────────────────────────────
    const int nblk = 4096;
    std::vector<float> h(nblk * 32);
    // Values that are NOT exactly representable in f32. An earlier version
    // used n/128.0f, where every partial sum is exact, so ANY reduction order
    // agreed and the probe reported a false bit-exactness. DPP row_xmask and
    // __shfl_down associate in OPPOSITE stride order (1,2,4,8,16 vs 16,8,4,2,1),
    // so on real data they differ at the ULP level. That is reassociation, not
    // a bug -- but it must be measured, not assumed away.
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (float)((double)((i * 2654435761u) % 1000003u) / 7919.0 - 63.1);
    float *din, *ds, *dd;
    HC(hipMalloc(&din, h.size() * 4));
    HC(hipMalloc(&ds, nblk * 4));
    HC(hipMalloc(&dd, nblk * 4));
    HC(hipMemcpy(din, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(check, dim3(nblk), dim3(32), 0, 0, din, ds, dd);
    HC(hipDeviceSynchronize());
    std::vector<float> a(nblk), b(nblk);
    HC(hipMemcpy(a.data(), ds, nblk * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(b.data(), dd, nblk * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    double worst = 0;
    for (int i = 0; i < nblk; ++i) {
        const double d = std::abs((double)a[i] - (double)b[i]);
        worst = std::max(worst, d);
        // Reassociation differs, so exact equality is not the bar.
        if (d > 1e-3 * std::abs((double)a[i]) + 1e-4) ++bad;
    }
    printf("correctness: %d/%d blocks disagree (worst abs delta %.3e)  %s\n\n", bad, nblk,
           worst, bad == 0 ? "ok" : "FAIL");
    if (bad) return 1;

    // ── throughput ──────────────────────────────────────────────────────────
    float* dy;
    HC(hipMalloc(&dy, nblk * 4));
    auto bench = [&](int variant, int iters) {
        auto go = [&] {
            if (variant == 0)
                hipLaunchKernelGGL((gemv_like<0>), dim3(nblk), dim3(32), 0, 0, din, dy, iters);
            else
                hipLaunchKernelGGL((gemv_like<1>), dim3(nblk), dim3(32), 0, 0, din, dy, iters);
        };
        for (int i = 0; i < 60; ++i) go();
        (void)hipDeviceSynchronize();
        hipEvent_t s, e;
        (void)hipEventCreate(&s);
        (void)hipEventCreate(&e);
        std::vector<float> t;
        for (int r = 0; r < 9; ++r) {
            (void)hipEventRecord(s);
            for (int i = 0; i < 64; ++i) go();
            (void)hipEventRecord(e);
            (void)hipEventSynchronize(e);
            float ms;
            (void)hipEventElapsedTime(&ms, s, e);
            t.push_back(ms / 64);
        }
        std::sort(t.begin(), t.end());
        (void)hipEventDestroy(s);
        (void)hipEventDestroy(e);
        return (double)t[t.size() / 2];
    };

    printf("%-38s %11s %11s %9s\n", "inner-loop depth (groups)", "shfl ms", "dpp ms", "delta");
    printf("%s\n", std::string(72, '-').c_str());
    for (int iters : {4, 8, 16, 32}) {
        bench(0, iters); // interleaved warmup, DPM settle
        bench(1, iters);
        const double ts = bench(0, iters);
        const double td = bench(1, iters);
        printf("%-38d %11.5f %11.5f %+8.2f%%\n", iters, ts, td, (td / ts - 1.0) * 100.0);
    }
    printf("\nK=1024 MQ4/RWQ4 GEMV corresponds to 4 groups — the top row.\n");
    return 0;
}
