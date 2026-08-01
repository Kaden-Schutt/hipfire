// Large-buffer device HBM bandwidth microbench for gfx942 VF
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d: %s\n", #cmd, __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

// Pure streaming triad: a[i] = b[i] + scalar * c[i]
__global__ void triad_kernel(float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ c,
                             float scalar,
                             size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    a[i] = b[i] + scalar * c[i];
  }
}

// Pure large-buffer read-sum (2 reads if we also write result; here write one scalar via atomic)
__global__ void read_sum_kernel(const float* __restrict__ x, double* __restrict__ out, size_t n) {
  // grid-stride partial sum into shared, then atomicAdd
  __shared__ double smem[256];
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  double acc = 0.0;
  for (; i < n; i += stride) acc += (double)x[i];
  smem[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if ((int)tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, smem[0]);
}

// Pure copy: dst = src
__global__ void copy_kernel(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

static double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2) return v[n/2];
  return 0.5 * (v[n/2 - 1] + v[n/2]);
}

int main(int argc, char** argv) {
  // Default: 8 GiB per buffer, 30 timed iters after 5 warmup
  size_t gib = 8;
  int warm = 5;
  int iters = 30;
  int streams = 1;
  if (argc > 1) gib = (size_t)atoi(argv[1]);
  if (argc > 2) iters = atoi(argv[2]);
  if (argc > 3) warm = atoi(argv[3]);
  if (argc > 4) streams = atoi(argv[4]);
  if (streams < 1) streams = 1;
  if (streams > 4) streams = 4;

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device=%s gcnArchName=%s multiProcessorCount=%d totalGlobalMem=%zu\n",
         prop.name, prop.gcnArchName, prop.multiProcessorCount, (size_t)prop.totalGlobalMem);
  fflush(stdout);

  const size_t n = gib * (size_t(1) << 30) / sizeof(float); // elements per buffer
  const size_t bytes = n * sizeof(float);
  printf("buffer_elems=%zu buffer_bytes=%zu (%.3f GiB) warm=%d iters=%d streams=%d\n",
         n, bytes, bytes / (1024.0*1024.0*1024.0), warm, iters, streams);
  fflush(stdout);

  // Allocate: for triad need 3 buffers; for multi-stream replicate
  std::vector<float*> d_a(streams), d_b(streams), d_c(streams);
  std::vector<hipStream_t> st(streams);
  for (int s = 0; s < streams; s++) {
    HIP_CHECK(hipStreamCreate(&st[s]));
    HIP_CHECK(hipMalloc(&d_a[s], bytes));
    HIP_CHECK(hipMalloc(&d_b[s], bytes));
    HIP_CHECK(hipMalloc(&d_c[s], bytes));
    HIP_CHECK(hipMemsetAsync(d_a[s], 0, bytes, st[s]));
    HIP_CHECK(hipMemsetAsync(d_b[s], 1, bytes, st[s]));
    HIP_CHECK(hipMemsetAsync(d_c[s], 2, bytes, st[s]));
  }
  HIP_CHECK(hipDeviceSynchronize());

  const int block = 256;
  // Fill device: many waves
  const int grid = prop.multiProcessorCount * 32; // oversubscribe CUs
  printf("launch grid=%d block=%d\n", grid, block);
  fflush(stdout);

  auto run_kernel_family = [&](const char* name, int mode /*0=copy,1=triad,2=readsum*/) {
    // For readsum need out buffer
    std::vector<double*> d_out(streams, nullptr);
    if (mode == 2) {
      for (int s = 0; s < streams; s++) {
        HIP_CHECK(hipMalloc(&d_out[s], sizeof(double)));
      }
    }

    // Warm
    for (int w = 0; w < warm; w++) {
      for (int s = 0; s < streams; s++) {
        if (mode == 0) {
          hipLaunchKernelGGL(copy_kernel, dim3(grid), dim3(block), 0, st[s], d_a[s], d_b[s], n);
        } else if (mode == 1) {
          hipLaunchKernelGGL(triad_kernel, dim3(grid), dim3(block), 0, st[s], d_a[s], d_b[s], d_c[s], 1.1f, n);
        } else {
          HIP_CHECK(hipMemsetAsync(d_out[s], 0, sizeof(double), st[s]));
          hipLaunchKernelGGL(read_sum_kernel, dim3(grid), dim3(block), 0, st[s], d_b[s], d_out[s], n);
        }
      }
      HIP_CHECK(hipDeviceSynchronize());
    }

    std::vector<double> times_ms;
    times_ms.reserve(iters);
    for (int it = 0; it < iters; it++) {
      hipEvent_t t0, t1;
      HIP_CHECK(hipEventCreate(&t0));
      HIP_CHECK(hipEventCreate(&t1));
      HIP_CHECK(hipEventRecord(t0, 0));
      for (int s = 0; s < streams; s++) {
        if (mode == 0) {
          hipLaunchKernelGGL(copy_kernel, dim3(grid), dim3(block), 0, st[s], d_a[s], d_b[s], n);
        } else if (mode == 1) {
          hipLaunchKernelGGL(triad_kernel, dim3(grid), dim3(block), 0, st[s], d_a[s], d_b[s], d_c[s], 1.1f, n);
        } else {
          HIP_CHECK(hipMemsetAsync(d_out[s], 0, sizeof(double), st[s]));
          hipLaunchKernelGGL(read_sum_kernel, dim3(grid), dim3(block), 0, st[s], d_b[s], d_out[s], n);
        }
      }
      HIP_CHECK(hipEventRecord(t1, 0));
      HIP_CHECK(hipEventSynchronize(t1));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
      times_ms.push_back((double)ms);
      HIP_CHECK(hipEventDestroy(t0));
      HIP_CHECK(hipEventDestroy(t1));
    }

    // Bytes moved per iteration across all streams:
    // copy: 2 * bytes * streams (read+write)
    // triad: 3 * bytes * streams
    // readsum: 1 * bytes * streams (read only; ignore tiny atomic out)
    double bytes_moved = 0;
    if (mode == 0) bytes_moved = 2.0 * (double)bytes * streams;
    else if (mode == 1) bytes_moved = 3.0 * (double)bytes * streams;
    else bytes_moved = 1.0 * (double)bytes * streams;

    double med = median(times_ms);
    double mn = *std::min_element(times_ms.begin(), times_ms.end());
    double mx = *std::max_element(times_ms.begin(), times_ms.end());
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / times_ms.size();
    double gbs_med = (bytes_moved / (med * 1e-3)) / 1e9;
    double gbs_max = (bytes_moved / (mn * 1e-3)) / 1e9;
    double gbs_mean = (bytes_moved / (mean * 1e-3)) / 1e9;
    double ratio_53 = gbs_med / 5300.0;

    printf("RESULT name=%s streams=%d bytes_moved_per_iter=%.0f\n", name, streams, bytes_moved);
    printf("  time_ms median=%.4f mean=%.4f min=%.4f max=%.4f\n", med, mean, mn, mx);
    printf("  GB/s median=%.2f mean=%.2f peak_from_min=%.2f\n", gbs_med, gbs_mean, gbs_max);
    printf("  ratio_to_5.3TBs_spec median=%.4f (%.2f%%)\n", ratio_53, 100.0*ratio_53);
    fflush(stdout);

    if (mode == 2) {
      for (int s = 0; s < streams; s++) HIP_CHECK(hipFree(d_out[s]));
    }
  };

  run_kernel_family("copy", 0);
  run_kernel_family("triad", 1);
  run_kernel_family("read_sum", 2);

  // Also hipMemcpyDtoD large buffer bandwidth
  {
    float *src = d_b[0], *dst = d_a[0];
    for (int w = 0; w < warm; w++) {
      HIP_CHECK(hipMemcpyDtoD(dst, src, bytes));
    }
    std::vector<double> times_ms;
    for (int it = 0; it < iters; it++) {
      hipEvent_t t0, t1;
      HIP_CHECK(hipEventCreate(&t0));
      HIP_CHECK(hipEventCreate(&t1));
      HIP_CHECK(hipEventRecord(t0, 0));
      HIP_CHECK(hipMemcpyDtoDAsync(dst, src, bytes, st[0]));
      HIP_CHECK(hipEventRecord(t1, 0));
      HIP_CHECK(hipEventSynchronize(t1));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
      times_ms.push_back(ms);
      HIP_CHECK(hipEventDestroy(t0));
      HIP_CHECK(hipEventDestroy(t1));
    }
    double med = median(times_ms);
    // DtoD is read+write
    double bytes_moved = 2.0 * (double)bytes;
    double gbs = (bytes_moved / (med * 1e-3)) / 1e9;
    printf("RESULT name=hipMemcpyDtoD streams=1 bytes_moved_per_iter=%.0f\n", bytes_moved);
    printf("  time_ms median=%.4f\n", med);
    printf("  GB/s median=%.2f ratio_to_5.3TBs=%.4f\n", gbs, gbs/5300.0);
    fflush(stdout);
  }

  for (int s = 0; s < streams; s++) {
    HIP_CHECK(hipFree(d_a[s]));
    HIP_CHECK(hipFree(d_b[s]));
    HIP_CHECK(hipFree(d_c[s]));
    HIP_CHECK(hipStreamDestroy(st[s]));
  }
  printf("DONE\n");
  return 0;
}
