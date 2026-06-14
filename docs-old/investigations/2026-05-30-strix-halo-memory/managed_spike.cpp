// Managed-memory oversubscription spike for gfx1151 (Strix Halo).
// Allocates `GB` of hipMallocManaged (can exceed the 96GB VRAM carveout),
// writes every byte from a kernel (forces page residency), reads it back,
// reports bandwidth. Green = managed/unified path is viable for hipfire.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    printf("ERR %s: %s\n",#x,hipGetErrorString(e)); fflush(stdout); return 1; } } while(0)

__global__ void writek(uint8_t* p, size_t n) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    for (; i < n; i += stride) p[i] = (uint8_t)(i & 0x7f);
}
__global__ void sumk(const uint8_t* p, size_t n, unsigned long long* out) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    unsigned long long acc = 0;
    for (; i < n; i += stride) acc += p[i];
    atomicAdd(out, acc);
}

int main(int argc, char** argv) {
    double gb = argc > 1 ? atof(argv[1]) : 100.0;
    size_t n = (size_t)(gb * 1073741824.0);
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("dev: %s (%s)\n", prop.name, prop.gcnArchName);
    size_t fb=0, tb=0; hipMemGetInfo(&fb, &tb);
    printf("hipMemGetInfo: free=%.1f total=%.1f GB\n", fb/1.073741824e9, tb/1.073741824e9);

    uint8_t* p = nullptr;
    printf("hipMallocManaged %.1f GB (n=%zu)...\n", gb, n); fflush(stdout);
    double a0 = now_s();
    hipError_t e = hipMallocManaged((void**)&p, n, hipMemAttachGlobal);
    if (e != hipSuccess) { printf("ALLOC FAIL: %s\n", hipGetErrorString(e)); return 2; }
    printf("  alloc ok %.2fs\n", now_s()-a0); fflush(stdout);

    int th = 256, bl = 8192;
    double t1 = now_s();
    hipLaunchKernelGGL(writek, dim3(bl), dim3(th), 0, 0, p, n);
    CK(hipDeviceSynchronize());
    double ws = now_s()-t1;
    printf("WRITE %.1fGB: %.2fs = %.1f GB/s\n", gb, ws, gb/ws); fflush(stdout);

    unsigned long long* d; CK(hipMallocManaged((void**)&d, 8)); *d = 0;
    double t2 = now_s();
    hipLaunchKernelGGL(sumk, dim3(bl), dim3(th), 0, 0, p, n, d);
    CK(hipDeviceSynchronize());
    double rs = now_s()-t2;
    printf("READ  %.1fGB: %.2fs = %.1f GB/s  sum=%llu\n", gb, rs, gb/rs, *d); fflush(stdout);

    hipFree(p); hipFree(d);
    printf("OK %.1f GB\n", gb);
    return 0;
}
