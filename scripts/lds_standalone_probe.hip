#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err__ = (call);                                             \
        if (err__ != hipSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                \
                         hipGetErrorString(err__), static_cast<int>(err__));   \
            return 2;                                                          \
        }                                                                      \
    } while (0)

template <int TILE, int ACTIVE, int ITERS>
__global__ void lds_probe(float* out, int stride) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float sum = 0.0f;

    for (int it = 0; it < ITERS; ++it) {
        if (tx < ACTIVE && ty < ACTIVE) {
            As[ty][tx] = static_cast<float>((blockIdx.x + 1) * (ty + 1) + tx + it);
            Bs[ty][tx] = static_cast<float>((blockIdx.y + 1) * (tx + 1) - ty + it);
        }
        __syncthreads();

        if (tx < ACTIVE && ty < ACTIVE) {
#pragma unroll
            for (int kk = 0; kk < ACTIVE; ++kk) {
                sum += As[ty][kk] * Bs[kk][tx];
            }
        }
        __syncthreads();
    }

    if (tx < ACTIVE && ty < ACTIVE) {
        const int idx = ((blockIdx.y * gridDim.x + blockIdx.x) * stride) +
                        ty * ACTIVE + tx;
        out[idx] = sum;
    }
}

template <int TILE, int ITERS>
__global__ void lds_probe_nomask(float* out, int stride) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float sum = 0.0f;

    for (int it = 0; it < ITERS; ++it) {
        As[ty][tx] = static_cast<float>((blockIdx.x + 1) * (ty + 1) + tx + it);
        Bs[ty][tx] = static_cast<float>((blockIdx.y + 1) * (tx + 1) - ty + it);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    const int idx = ((blockIdx.y * gridDim.x + blockIdx.x) * stride) +
                    ty * TILE + tx;
    out[idx] = sum;
}

#ifndef HIPFIRE_LDS_STANDALONE_NO_MAIN
static int run_variant(const char* variant, int launches, int grid_x, int grid_y) {
    float* out = nullptr;
    const size_t elems = static_cast<size_t>(grid_x) * grid_y * 256;
    HIP_CHECK(hipMalloc(&out, elems * sizeof(float)));
    HIP_CHECK(hipMemset(out, 0, elems * sizeof(float)));

    dim3 grid(grid_x, grid_y, 1);
    dim3 block;

    for (int i = 0; i < launches; ++i) {
        if (std::strcmp(variant, "tile5") == 0) {
            block = dim3(5, 5, 1);
            hipLaunchKernelGGL((lds_probe<5, 5, 128>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 128>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile5_i512") == 0) {
            block = dim3(5, 5, 1);
            hipLaunchKernelGGL((lds_probe<5, 5, 512>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i512") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 512>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i256") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 256>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i320") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 320>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i336") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 336>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i352") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 352>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i384") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe<6, 6, 384>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_i512_nomask") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe_nomask<6, 512>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile6_nomask") == 0) {
            block = dim3(6, 6, 1);
            hipLaunchKernelGGL((lds_probe_nomask<6, 128>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile8") == 0) {
            block = dim3(8, 8, 1);
            hipLaunchKernelGGL((lds_probe<8, 8, 128>), grid, block, 0, 0, out, 64);
        } else if (std::strcmp(variant, "tile16") == 0) {
            block = dim3(16, 16, 1);
            hipLaunchKernelGGL((lds_probe<16, 16, 128>), grid, block, 0, 0, out, 256);
        } else if (std::strcmp(variant, "active4_block8") == 0) {
            block = dim3(8, 8, 1);
            hipLaunchKernelGGL((lds_probe<4, 4, 128>), grid, block, 0, 0, out, 64);
        } else {
            std::fprintf(stderr, "unknown variant: %s\n", variant);
            hipFree(out);
            return 2;
        }

        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess) {
            std::fprintf(stderr, "launch %d failed: %s (%d)\n", i,
                         hipGetErrorString(launch_err),
                         static_cast<int>(launch_err));
            hipFree(out);
            return 3;
        }

        hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            std::fprintf(stderr, "sync %d failed: %s (%d)\n", i,
                         hipGetErrorString(sync_err),
                         static_cast<int>(sync_err));
            hipFree(out);
            return 4;
        }
    }

    float first = 0.0f;
    HIP_CHECK(hipMemcpy(&first, out, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(out));
    std::printf("variant=%s launches=%d grid=%dx%d first=%f OK\n",
                variant, launches, grid_x, grid_y, first);
    return 0;
}

int main(int argc, char** argv) {
    const char* variant = argc > 1 ? argv[1] : "tile6";
    int launches = argc > 2 ? std::atoi(argv[2]) : 100;
    int grid_x = argc > 3 ? std::atoi(argv[3]) : 64;
    int grid_y = argc > 4 ? std::atoi(argv[4]) : 64;
    return run_variant(variant, launches, grid_x, grid_y);
}
#endif
