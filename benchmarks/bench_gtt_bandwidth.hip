// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GPU read-bandwidth microbench: device-local VRAM vs mapped host/GTT memory.
// Compile: hipcc -O3 --offload-arch=gfx1151 -o /tmp/bench_gtt_bandwidth benchmarks/bench_gtt_bandwidth.hip
// Run:    /tmp/bench_gtt_bandwidth [MiB=512] [passes=8] [runs=7]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t error_ = (call);                                             \
        if (error_ != hipSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,           \
                         hipGetErrorString(error_));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

extern "C" __launch_bounds__(256) __global__ void
read_bandwidth(const uint4 *__restrict__ src, uint64_t *__restrict__ sums,
               size_t vectors, unsigned passes) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    uint4 acc = {0, 0, 0, 0};

    for (unsigned pass = 0; pass < passes; ++pass) {
        for (size_t i = tid; i < vectors; i += stride) {
            const uint4 value = src[i];
            acc.x += value.x;
            acc.y += value.y;
            acc.z += value.z;
            acc.w += value.w;
        }
    }
    sums[tid] = static_cast<uint64_t>(acc.x) + acc.y + acc.z + acc.w;
}

static float measure(const uint4 *src, uint64_t *sums, size_t vectors,
                     unsigned passes, unsigned blocks, hipEvent_t start,
                     hipEvent_t stop) {
    HIP_CHECK(hipEventRecord(start));
    read_bandwidth<<<blocks, 256>>>(src, sums, vectors, passes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
}

static double median_gbps(std::vector<float> times, size_t bytes,
                          unsigned passes) {
    std::sort(times.begin(), times.end());
    const double ms = times[times.size() / 2];
    return static_cast<double>(bytes) * passes / (ms * 1.0e6);
}

int main(int argc, char **argv) {
    const size_t mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 512;
    const unsigned passes = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 8;
    const unsigned runs = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 7;
    if (mib == 0 || passes == 0 || runs == 0) {
        std::fprintf(stderr, "MiB, passes, and runs must be non-zero\n");
        return 2;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    const size_t bytes = mib * 1024 * 1024;
    const size_t vectors = bytes / sizeof(uint4);
    const unsigned blocks = std::min<unsigned>(
        prop.multiProcessorCount * 8,
        static_cast<unsigned>((vectors + 255) / 256));
    const size_t sum_bytes = static_cast<size_t>(blocks) * 256 * sizeof(uint64_t);

    uint4 *vram = nullptr;
    void *gtt_host = nullptr;
    uint4 *gtt_device = nullptr;
    uint64_t *sums = nullptr;
    HIP_CHECK(hipMalloc(&vram, bytes));
    HIP_CHECK(hipHostMalloc(&gtt_host, bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(
        reinterpret_cast<void **>(&gtt_device), gtt_host, 0));
    HIP_CHECK(hipMalloc(&sums, sum_bytes));

    HIP_CHECK(hipMemset(vram, 0xa5, bytes));
    std::memset(gtt_host, 0xa5, bytes); // faults in every host/GTT page

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < 2; ++i) {
        (void)measure(vram, sums, vectors, passes, blocks, start, stop);
        (void)measure(gtt_device, sums, vectors, passes, blocks, start, stop);
    }

    std::vector<float> vram_ms, gtt_ms;
    vram_ms.reserve(runs);
    gtt_ms.reserve(runs);
    for (unsigned run = 0; run < runs; ++run) {
        if ((run & 1) == 0) {
            vram_ms.push_back(measure(vram, sums, vectors, passes, blocks, start, stop));
            gtt_ms.push_back(measure(gtt_device, sums, vectors, passes, blocks, start, stop));
        } else {
            gtt_ms.push_back(measure(gtt_device, sums, vectors, passes, blocks, start, stop));
            vram_ms.push_back(measure(vram, sums, vectors, passes, blocks, start, stop));
        }
    }

    std::vector<uint64_t> vram_sums(sum_bytes / sizeof(uint64_t));
    std::vector<uint64_t> gtt_sums(vram_sums.size());
    (void)measure(vram, sums, vectors, passes, blocks, start, stop);
    HIP_CHECK(hipMemcpy(vram_sums.data(), sums, sum_bytes, hipMemcpyDeviceToHost));
    (void)measure(gtt_device, sums, vectors, passes, blocks, start, stop);
    HIP_CHECK(hipMemcpy(gtt_sums.data(), sums, sum_bytes, hipMemcpyDeviceToHost));
    const bool verified = vram_sums == gtt_sums;
    const uint64_t checksum =
        std::accumulate(vram_sums.begin(), vram_sums.end(), uint64_t{0});

    const double vram_gbps = median_gbps(vram_ms, bytes, passes);
    const double gtt_gbps = median_gbps(gtt_ms, bytes, passes);
    std::printf("GPU: %s (%d CUs)\n", prop.name, prop.multiProcessorCount);
    std::printf("Working set: %zu MiB x %u passes, %u runs\n", mib, passes, runs);
    std::printf("VRAM read: %.2f GB/s\n", vram_gbps);
    std::printf("GTT  read: %.2f GB/s\n", gtt_gbps);
    std::printf("GTT/VRAM: %.3fx\n", gtt_gbps / vram_gbps);
    std::printf("Checksum: 0x%016llx (%s)\n",
                static_cast<unsigned long long>(checksum),
                verified ? "match" : "MISMATCH");
    std::printf("BENCH_VRAM_GB_S=%.2f\nBENCH_GTT_GB_S=%.2f\nBENCH_GTT_VRAM_RATIO=%.6f\n",
                vram_gbps, gtt_gbps, gtt_gbps / vram_gbps);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(sums));
    HIP_CHECK(hipHostFree(gtt_host));
    HIP_CHECK(hipFree(vram));
    return verified ? 0 : 1;
}
