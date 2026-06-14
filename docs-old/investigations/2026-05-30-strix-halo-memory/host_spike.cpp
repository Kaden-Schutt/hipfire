// hipHostMalloc zero-copy spike for gfx1151 (Strix Halo APU).
// Allocates pinned host memory (system RAM), maps it to a device pointer, and
// has a kernel write/read it directly — no hipMemcpy. Measures the effective
// GPU<->host-pinned bandwidth. Green + decent BW = the viable >carveout path
// (capacity then bounded by system-RAM size / BIOS carveout split).
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
    double gb = argc > 1 ? atof(argv[1]) : 20.0;
    size_t n = (size_t)(gb * 1073741824.0);
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("dev: %s (%s)\n", prop.name, prop.gcnArchName);

    uint8_t* hp = nullptr;
    printf("hipHostMalloc %.1f GB (mapped)...\n", gb); fflush(stdout);
    double a0 = now_s();
    hipError_t e = hipHostMalloc((void**)&hp, n, hipHostMallocMapped);
    if (e != hipSuccess) { printf("HOST ALLOC FAIL: %s\n", hipGetErrorString(e)); return 2; }
    printf("  alloc ok %.2fs\n", now_s()-a0); fflush(stdout);

    uint8_t* dp = nullptr;
    CK(hipHostGetDevicePointer((void**)&dp, hp, 0));
    printf("  device ptr ok (host=%p dev=%p)\n", (void*)hp, (void*)dp); fflush(stdout);

    int th = 256, bl = 8192;
    double t1 = now_s();
    hipLaunchKernelGGL(writek, dim3(bl), dim3(th), 0, 0, dp, n);
    CK(hipDeviceSynchronize());
    double ws = now_s()-t1;
    printf("WRITE %.1fGB (gpu->host-pinned): %.2fs = %.1f GB/s\n", gb, ws, gb/ws); fflush(stdout);

    unsigned long long* d; CK(hipMalloc((void**)&d, 8)); CK(hipMemset(d, 0, 8));
    double t2 = now_s();
    hipLaunchKernelGGL(sumk, dim3(bl), dim3(th), 0, 0, dp, n, d);
    CK(hipDeviceSynchronize());
    unsigned long long hsum = 0; CK(hipMemcpy(&hsum, d, 8, hipMemcpyDeviceToHost));
    double rs = now_s()-t2;
    printf("READ  %.1fGB (gpu<-host-pinned): %.2fs = %.1f GB/s  sum=%llu\n", gb, rs, gb/rs, hsum);
    printf("host coherent check: hp[12345]=%d (expect %d)\n", (int)hp[12345], (int)(12345 & 0x7f));

    hipHostFree(hp); hipFree(d);
    printf("OK %.1f GB host-pinned zero-copy\n", gb);
    return 0;
}
