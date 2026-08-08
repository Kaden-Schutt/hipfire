// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kaden Schutt

// Serialized host-mapped coherent visibility oracle for a recovered gfx1100 +
// gfx1151 pair. This is deliberately NOT a GTT-GEM benchmark, a concurrent
// transport, or a performance-admission test. It only detects whether writes
// made through one device's mapping become exactly visible through the other
// device's mapping after host serialization and system-scope fences.
//
// Do not run this while either GPU is unhealthy. Build explicitly with:
//   hipcc -O3 -std=c++17 \
//     -DHIPFIRE_BENCH_SERIAL_HOST_MAPPED_TRANSPORT=1 \
//     --offload-arch=gfx1100 --offload-arch=gfx1151 \
//     benchmarks/bench_shared_gtt_transport.hip \
//     -o bench_shared_gtt_transport
//
// Execution additionally requires --confirm-gpus-recovered. Without that
// acknowledgement, main exits before any device query, allocation, or launch.
// As with every HIP executable, the loader may register embedded fat binaries
// before main; therefore do not execute this binary at all on an unhealthy host.

#if !defined(HIPFIRE_BENCH_SERIAL_HOST_MAPPED_TRANSPORT) || \
    HIPFIRE_BENCH_SERIAL_HOST_MAPPED_TRANSPORT != 1
#error "Define HIPFIRE_BENCH_SERIAL_HOST_MAPPED_TRANSPORT=1 to build this run-later diagnostic"
#endif

#include <hip/hip_runtime.h>

// Keep the hazardous transport primitives that invalidated earlier experiments
// unspellable below this point. This diagnostic has one host allocation, one
// producer at a time, one consumer at a time, and host-side serialization.
#pragma GCC poison hipDeviceEnablePeerAccess
#pragma GCC poison hipDeviceCanAccessPeer
#pragma GCC poison hipMemcpyPeer
#pragma GCC poison hipMemcpyPeerAsync
#pragma GCC poison hipMallocManaged
#pragma GCC poison hipHostRegister
#pragma GCC poison hipIpcGetMemHandle
#pragma GCC poison hipIpcOpenMemHandle
#pragma GCC poison hipImportExternalMemory
#pragma GCC poison hipStreamWaitEvent

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultBytes = 16 * 1024;
constexpr size_t kMaxBytes = 16 * 1024 * 1024;
constexpr uint32_t kDefaultIterations = 1;
constexpr uint32_t kMaxIterations = 64;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxProducerBlocks = 64;
constexpr uint64_t kNoMismatch = std::numeric_limits<uint64_t>::max();

struct ConsumerResult {
    uint64_t checksum;
    uint64_t mismatches;
    uint64_t first_bad_word;
};

struct DeviceEndpoint {
    int ordinal = -1;
    const char* label = nullptr;
    std::string expected_arch;
    std::string expected_bdf;
    hipDeviceProp_t properties{};
    uint32_t* mapped = nullptr;
};

enum class Direction {
    Both,
    Gfx1100ToGfx1151,
    Gfx1151ToGfx1100,
};

struct Options {
    bool confirm_recovered = false;
    bool show_help = false;
    int gfx1100_device = 0;
    int gfx1151_device = 1;
    std::string gfx1100_bdf = "0000:66:00.0";
    std::string gfx1151_bdf = "0000:bf:00.0";
    size_t bytes = kDefaultBytes;
    uint32_t iterations = kDefaultIterations;
    Direction direction = Direction::Both;
};

__host__ __device__ uint32_t pattern_word(uint64_t tag, uint64_t word_index) {
    uint64_t value = tag + 0x9e3779b97f4a7c15ULL * (word_index + 1);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<uint32_t>(value ^ (value >> 32));
}

__global__ void producer_write(uint32_t* shared, size_t words, uint64_t tag) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    size_t word = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (word < words) {
        shared[word] = pattern_word(tag, word);
        word += stride;
    }
    __threadfence_system();
}

__global__ void consumer_check(const uint32_t* shared,
                               size_t words,
                               uint64_t tag,
                               ConsumerResult* result) {
    __shared__ uint64_t checksums[kThreads];
    __shared__ uint64_t mismatches[kThreads];
    __shared__ uint64_t first_bad[kThreads];

    __threadfence_system();

    uint64_t checksum = 0;
    uint64_t mismatch_count = 0;
    uint64_t first_mismatch = kNoMismatch;
    for (size_t word = threadIdx.x; word < words; word += blockDim.x) {
        const uint32_t observed = shared[word];
        const uint32_t expected = pattern_word(tag, word);
        checksum += observed;
        if (observed != expected) {
            ++mismatch_count;
            first_mismatch = min(first_mismatch, static_cast<uint64_t>(word));
        }
    }

    checksums[threadIdx.x] = checksum;
    mismatches[threadIdx.x] = mismatch_count;
    first_bad[threadIdx.x] = first_mismatch;
    __syncthreads();

    for (uint32_t width = blockDim.x / 2; width != 0; width /= 2) {
        if (threadIdx.x < width) {
            checksums[threadIdx.x] += checksums[threadIdx.x + width];
            mismatches[threadIdx.x] += mismatches[threadIdx.x + width];
            first_bad[threadIdx.x] = min(first_bad[threadIdx.x], first_bad[threadIdx.x + width]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result->checksum = checksums[0];
        result->mismatches = mismatches[0];
        result->first_bad_word = first_bad[0];
    }
}

bool check_hip(hipError_t status, const char* operation) {
    if (status == hipSuccess) {
        return true;
    }
    std::cerr << "HIP error in " << operation << ": " << hipGetErrorString(status) << '\n';
    return false;
}

#define HIP_REQUIRE(operation) \
    do { \
        if (!check_hip((operation), #operation)) { \
            return false; \
        } \
    } while (false)

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --confirm-gpus-recovered [options]\n"
        << "\n"
        << "Run-later serialized host-mapped coherent visibility oracle.\n"
        << "Without confirmation, main performs no device query, allocation, or launch.\n"
        << "The HIP loader still registers fat binaries before main: do not execute this\n"
        << "binary at all while either GPU is unhealthy.\n"
        << "\n"
        << "Options:\n"
        << "  --gfx1100-device N       HIP ordinal (default: 0)\n"
        << "  --gfx1151-device N       HIP ordinal (default: 1)\n"
        << "  --gfx1100-bdf BDF        Exact PCI identity (default: 0000:66:00.0)\n"
        << "  --gfx1151-bdf BDF        Exact PCI identity (default: 0000:bf:00.0)\n"
        << "  --bytes N                4-byte multiple, at most 16 MiB (default: 16384)\n"
        << "  --iters N                1..64 (default: 1)\n"
        << "  --direction VALUE        both, 1100to1151, or 1151to1100\n"
        << "  --help                   Print this message; HIP loader registration still occurs\n";
}

bool parse_unsigned(const char* text, uint64_t* value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (*end != '\0') {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options* options) {
    auto take_value = [&](int* index, const char* option) -> const char* {
        if (*index + 1 >= argc) {
            std::cerr << option << " requires a value\n";
            return nullptr;
        }
        return argv[++(*index)];
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--confirm-gpus-recovered") {
            options->confirm_recovered = true;
        } else if (argument == "--help") {
            options->show_help = true;
        } else if (argument == "--gfx1100-device" || argument == "--gfx1151-device") {
            const char* text = take_value(&index, argument.c_str());
            uint64_t value = 0;
            if (text == nullptr || !parse_unsigned(text, &value) || value > std::numeric_limits<int>::max()) {
                std::cerr << "Invalid ordinal for " << argument << '\n';
                return false;
            }
            if (argument == "--gfx1100-device") {
                options->gfx1100_device = static_cast<int>(value);
            } else {
                options->gfx1151_device = static_cast<int>(value);
            }
        } else if (argument == "--gfx1100-bdf" || argument == "--gfx1151-bdf") {
            const char* text = take_value(&index, argument.c_str());
            if (text == nullptr) {
                return false;
            }
            if (argument == "--gfx1100-bdf") {
                options->gfx1100_bdf = text;
            } else {
                options->gfx1151_bdf = text;
            }
        } else if (argument == "--bytes" || argument == "--iters") {
            const char* text = take_value(&index, argument.c_str());
            uint64_t value = 0;
            if (text == nullptr || !parse_unsigned(text, &value)) {
                std::cerr << "Invalid integer for " << argument << '\n';
                return false;
            }
            if (argument == "--bytes") {
                if (value > std::numeric_limits<size_t>::max()) {
                    std::cerr << "Byte count is not representable\n";
                    return false;
                }
                options->bytes = static_cast<size_t>(value);
            } else {
                if (value > std::numeric_limits<uint32_t>::max()) {
                    std::cerr << "Iteration count is not representable\n";
                    return false;
                }
                options->iterations = static_cast<uint32_t>(value);
            }
        } else if (argument == "--direction") {
            const char* text = take_value(&index, argument.c_str());
            if (text == nullptr) {
                return false;
            }
            const std::string value = text;
            if (value == "both") {
                options->direction = Direction::Both;
            } else if (value == "1100to1151") {
                options->direction = Direction::Gfx1100ToGfx1151;
            } else if (value == "1151to1100") {
                options->direction = Direction::Gfx1151ToGfx1100;
            } else {
                std::cerr << "Invalid --direction value: " << value << '\n';
                return false;
            }
        } else {
            std::cerr << "Unknown option: " << argument << '\n';
            return false;
        }
    }

    if (options->bytes < sizeof(uint32_t) || options->bytes > kMaxBytes ||
        options->bytes % sizeof(uint32_t) != 0) {
        std::cerr << "--bytes must be a 4-byte multiple in [4, " << kMaxBytes << "]\n";
        return false;
    }
    if (options->iterations == 0 || options->iterations > kMaxIterations) {
        std::cerr << "--iters must be in [1, " << kMaxIterations << "]\n";
        return false;
    }
    if (options->gfx1100_device == options->gfx1151_device) {
        std::cerr << "The two endpoints must use distinct HIP ordinals\n";
        return false;
    }
    return true;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string arch_atom(const char* gcn_arch_name) {
    const std::string full_name = gcn_arch_name == nullptr ? "" : gcn_arch_name;
    const size_t feature_separator = full_name.find(':');
    return full_name.substr(0, feature_separator);
}

std::string pci_bdf(const hipDeviceProp_t& properties) {
    std::ostringstream bdf;
    bdf << std::hex << std::setfill('0')
        << std::setw(4) << properties.pciDomainID << ':'
        << std::setw(2) << properties.pciBusID << ':'
        << std::setw(2) << properties.pciDeviceID << ".0";
    return lowercase(bdf.str());
}

bool validate_endpoint(DeviceEndpoint* endpoint, int device_count) {
    if (endpoint->ordinal < 0 || endpoint->ordinal >= device_count) {
        std::cerr << endpoint->label << " ordinal " << endpoint->ordinal
                  << " is outside the available device range [0, " << device_count << ")\n";
        return false;
    }

    HIP_REQUIRE(hipGetDeviceProperties(&endpoint->properties, endpoint->ordinal));
    const std::string observed_arch = arch_atom(endpoint->properties.gcnArchName);
    const std::string observed_bdf = pci_bdf(endpoint->properties);
    const std::string expected_bdf = lowercase(endpoint->expected_bdf);
    if (observed_arch != endpoint->expected_arch) {
        std::cerr << endpoint->label << " arch mismatch: expected " << endpoint->expected_arch
                  << ", observed " << observed_arch << '\n';
        return false;
    }
    if (observed_bdf != expected_bdf) {
        std::cerr << endpoint->label << " BDF mismatch: expected " << expected_bdf
                  << ", observed " << observed_bdf << '\n';
        return false;
    }

    std::cout << "IDENTITY_OK label=" << endpoint->label
              << " ordinal=" << endpoint->ordinal
              << " arch=" << observed_arch
              << " bdf=" << observed_bdf
              << " name=\"" << endpoint->properties.name << "\"\n";
    return true;
}

uint64_t direction_tag(uint32_t direction_id, uint32_t iteration) {
    return 0x4846504952450000ULL ^ (static_cast<uint64_t>(direction_id) << 32) ^
           static_cast<uint64_t>(iteration + 1);
}

double elapsed_microseconds(std::chrono::steady_clock::time_point begin,
                            std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if (samples.size() % 2 != 0) {
        return samples[middle];
    }
    return (samples[middle - 1] + samples[middle]) / 2.0;
}

bool run_direction(const DeviceEndpoint& producer,
                   const DeviceEndpoint& consumer,
                   uint32_t direction_id,
                   uint32_t* host_words,
                   size_t word_count,
                   uint32_t iterations) {
    ConsumerResult* device_result = nullptr;
    if (!check_hip(hipSetDevice(consumer.ordinal), "hipSetDevice(consumer)")) {
        return false;
    }
    if (!check_hip(hipMalloc(reinterpret_cast<void**>(&device_result), sizeof(ConsumerResult)),
                   "hipMalloc(consumer result)")) {
        return false;
    }

    std::vector<double> producer_us;
    std::vector<double> host_verify_us;
    std::vector<double> consumer_us;
    std::vector<double> end_to_end_us;
    producer_us.reserve(iterations);
    host_verify_us.reserve(iterations);
    consumer_us.reserve(iterations);
    end_to_end_us.reserve(iterations);

    const uint32_t producer_blocks = static_cast<uint32_t>(std::max<size_t>(
        1, std::min<size_t>(kMaxProducerBlocks, (word_count + kThreads - 1) / kThreads)));

    bool passed = true;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const uint64_t tag = direction_tag(direction_id, iteration);
        for (size_t word = 0; word < word_count; ++word) {
            host_words[word] = ~pattern_word(tag, word);
        }

        const auto end_to_end_begin = std::chrono::steady_clock::now();
        if (!check_hip(hipSetDevice(producer.ordinal), "hipSetDevice(producer)")) {
            passed = false;
            break;
        }
        const auto producer_begin = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(producer_write,
                           dim3(producer_blocks),
                           dim3(kThreads),
                           0,
                           0,
                           producer.mapped,
                           word_count,
                           tag);
        if (!check_hip(hipGetLastError(), "producer_write launch") ||
            !check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(producer)")) {
            passed = false;
            break;
        }
        const auto producer_end = std::chrono::steady_clock::now();

        const auto host_verify_begin = producer_end;
        uint64_t expected_checksum = 0;
        uint64_t first_host_mismatch = kNoMismatch;
        for (size_t word = 0; word < word_count; ++word) {
            const uint32_t expected = pattern_word(tag, word);
            expected_checksum += expected;
            if (host_words[word] != expected && first_host_mismatch == kNoMismatch) {
                first_host_mismatch = word;
            }
        }
        const auto host_verify_end = std::chrono::steady_clock::now();
        if (first_host_mismatch != kNoMismatch) {
            std::cerr << "HOST_VISIBILITY_FAIL direction=" << producer.label << "->"
                      << consumer.label << " iteration=" << iteration
                      << " first_bad_word=" << first_host_mismatch
                      << " expected=" << pattern_word(tag, first_host_mismatch)
                      << " observed=" << host_words[first_host_mismatch] << '\n';
            passed = false;
            break;
        }

        if (!check_hip(hipSetDevice(consumer.ordinal), "hipSetDevice(consumer)")) {
            passed = false;
            break;
        }
        ConsumerResult observed{};
        const auto consumer_begin = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(consumer_check,
                           dim3(1),
                           dim3(kThreads),
                           0,
                           0,
                           consumer.mapped,
                           word_count,
                           tag,
                           device_result);
        if (!check_hip(hipGetLastError(), "consumer_check launch") ||
            !check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(consumer)") ||
            !check_hip(hipMemcpy(&observed,
                                 device_result,
                                 sizeof(observed),
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy(consumer result D2H)")) {
            passed = false;
            break;
        }
        const auto consumer_end = std::chrono::steady_clock::now();

        if (observed.mismatches != 0 || observed.first_bad_word != kNoMismatch ||
            observed.checksum != expected_checksum) {
            std::cerr << "CONSUMER_VISIBILITY_FAIL direction=" << producer.label << "->"
                      << consumer.label << " iteration=" << iteration
                      << " mismatches=" << observed.mismatches
                      << " first_bad_word=" << observed.first_bad_word
                      << " expected_checksum=" << expected_checksum
                      << " observed_checksum=" << observed.checksum << '\n';
            passed = false;
            break;
        }

        producer_us.push_back(elapsed_microseconds(producer_begin, producer_end));
        host_verify_us.push_back(elapsed_microseconds(host_verify_begin, host_verify_end));
        consumer_us.push_back(elapsed_microseconds(consumer_begin, consumer_end));
        end_to_end_us.push_back(elapsed_microseconds(end_to_end_begin, consumer_end));
        std::cout << "ITERATION_OK direction=" << producer.label << "->" << consumer.label
                  << " iteration=" << iteration
                  << " bytes=" << word_count * sizeof(uint32_t)
                  << " producer_us=" << producer_us.back()
                  << " host_verify_us=" << host_verify_us.back()
                  << " consumer_us=" << consumer_us.back()
                  << " end_to_end_us=" << end_to_end_us.back() << '\n';
    }

    if (!check_hip(hipSetDevice(consumer.ordinal), "hipSetDevice(consumer cleanup)")) {
        passed = false;
    } else if (!check_hip(hipFree(device_result), "hipFree(consumer result)")) {
        passed = false;
    }

    if (passed) {
        std::cout << "SERIAL_HOST_MAPPED_OK direction=" << producer.label << "->"
                  << consumer.label
                  << " bytes=" << word_count * sizeof(uint32_t)
                  << " iterations=" << iterations
                  << " producer_us_median=" << median(producer_us)
                  << " host_verify_us_median=" << median(host_verify_us)
                  << " consumer_us_median=" << median(consumer_us)
                  << " end_to_end_us_median=" << median(end_to_end_us) << '\n';
    }
    return passed;
}

bool map_for_endpoint(DeviceEndpoint* endpoint, void* host_allocation) {
    HIP_REQUIRE(hipSetDevice(endpoint->ordinal));
    HIP_REQUIRE(hipHostGetDevicePointer(reinterpret_cast<void**>(&endpoint->mapped),
                                        host_allocation,
                                        0));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return 2;
    }
    if (options.show_help) {
        print_usage(argv[0]);
        return 0;
    }
    if (!options.confirm_recovered) {
        std::cerr << "Refusing to initialize HIP: both GPUs must be recovered, then pass "
                     "--confirm-gpus-recovered explicitly.\n";
        return 2;
    }

    int device_count = 0;
    if (!check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount")) {
        return 3;
    }

    DeviceEndpoint gfx1100{
        options.gfx1100_device, "gfx1100", "gfx1100", options.gfx1100_bdf, {}, nullptr};
    DeviceEndpoint gfx1151{
        options.gfx1151_device, "gfx1151", "gfx1151", options.gfx1151_bdf, {}, nullptr};
    if (!validate_endpoint(&gfx1100, device_count) || !validate_endpoint(&gfx1151, device_count)) {
        return 3;
    }

    void* host_allocation = nullptr;
    if (!check_hip(hipSetDevice(gfx1100.ordinal), "hipSetDevice(gfx1100 allocation owner)") ||
        !check_hip(hipHostMalloc(&host_allocation,
                                options.bytes,
                                hipHostMallocPortable | hipHostMallocMapped |
                                    hipHostMallocCoherent),
                   "hipHostMalloc(portable|mapped|coherent)")) {
        return 3;
    }

    bool passed = map_for_endpoint(&gfx1100, host_allocation) &&
                  map_for_endpoint(&gfx1151, host_allocation);
    auto* host_words = static_cast<uint32_t*>(host_allocation);
    const size_t word_count = options.bytes / sizeof(uint32_t);
    if (passed && (options.direction == Direction::Both ||
                   options.direction == Direction::Gfx1100ToGfx1151)) {
        passed = run_direction(gfx1100,
                               gfx1151,
                               0x11001151U,
                               host_words,
                               word_count,
                               options.iterations);
    }
    if (passed && (options.direction == Direction::Both ||
                   options.direction == Direction::Gfx1151ToGfx1100)) {
        passed = run_direction(gfx1151,
                               gfx1100,
                               0x11511100U,
                               host_words,
                               word_count,
                               options.iterations);
    }

    if (!check_hip(hipHostFree(host_allocation), "hipHostFree")) {
        passed = false;
    }
    if (!passed) {
        return 4;
    }

    std::cout << "SERIAL_HOST_MAPPED_ALL_OK=1\n";
    return 0;
}
