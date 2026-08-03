// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

// Compile with radiowave/hip.h force-included. This probe exercises the same
// buffer-resource helpers used by Radiowave code objects, then checks the
// result through the ordinary HIP runtime on the selected device.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" __global__ void radiowave_portable_buffer_probe(
    const std::uint32_t* input, std::uint32_t* output) {
    const auto input_resource = radiowave::buffer_resource(input);
    const auto output_resource = radiowave::buffer_resource(output);
    const std::uint32_t value =
        radiowave::buffer_load_u32(input_resource, 0);
    radiowave::buffer_store_u32(output_resource, 0, value + 1);
}

static bool hip_ok(hipError_t status, const char* operation) {
    if (status == hipSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 hipGetErrorString(status));
    return false;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: portable_buffer_probe DEVICE\n");
        return 2;
    }
    const int device = std::atoi(argv[1]);
    hipDeviceProp_t properties{};
    if (!hip_ok(hipGetDeviceProperties(&properties, device),
                "hipGetDeviceProperties") ||
        !hip_ok(hipSetDevice(device), "hipSetDevice")) {
        return 3;
    }

    std::uint32_t* device_input = nullptr;
    std::uint32_t* device_output = nullptr;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&device_input),
                          sizeof(*device_input)),
                "hipMalloc(input)") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&device_output),
                          sizeof(*device_output)),
                "hipMalloc(output)")) {
        return 3;
    }

    const std::uint32_t input = 41;
    std::uint32_t output = 0;
    bool ok = hip_ok(hipMemcpy(device_input, &input, sizeof(input),
                               hipMemcpyHostToDevice),
                     "hipMemcpy(input)") &&
              hip_ok(hipMemset(device_output, 0, sizeof(output)),
                     "hipMemset(output)");
    if (ok) {
        hipLaunchKernelGGL(radiowave_portable_buffer_probe, dim3(1), dim3(1),
                           0, 0, device_input, device_output);
        ok = hip_ok(hipGetLastError(), "kernel launch") &&
             hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") &&
             hip_ok(hipMemcpy(&output, device_output, sizeof(output),
                              hipMemcpyDeviceToHost),
                    "hipMemcpy(output)");
    }

    ok = hip_ok(hipFree(device_output), "hipFree(output)") && ok;
    ok = hip_ok(hipFree(device_input), "hipFree(input)") && ok;
    std::printf("device=%d arch=%s output=%u\n", device,
                properties.gcnArchName, output);
    return ok && output == 42 ? 0 : 4;
}
