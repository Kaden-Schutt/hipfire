// Measure HIP kernel dispatch floor on gfx942
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
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

__global__ void null_kernel() {}

__global__ void add_one_kernel(int* p) {
  if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(p, 1);
}

static double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2) return v[n/2];
  return 0.5 * (v[n/2 - 1] + v[n/2]);
}

static void report(const char* name, int launches, const std::vector<double>& times_ms) {
  std::vector<double> t = times_ms;
  double med = median(t);
  double mn = *std::min_element(t.begin(), t.end());
  double mx = *std::max_element(t.begin(), t.end());
  double mean = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
  // per-launch us
  double us_med = (med * 1000.0) / launches;
  double us_mean = (mean * 1000.0) / launches;
  double us_min = (mn * 1000.0) / launches;
  // ceiling at 2795 launches/token
  double tok_s_med = 1e6 / (2795.0 * us_med);
  double tok_s_min = 1e6 / (2795.0 * us_min); // optimistic (min time)
  printf("RESULT name=%s launches_per_trial=%d trials=%zu\n", name, launches, times_ms.size());
  printf("  trial_ms median=%.4f mean=%.4f min=%.4f max=%.4f\n", med, mean, mn, mx);
  printf("  per_launch_us median=%.4f mean=%.4f min=%.4f\n", us_med, us_mean, us_min);
  printf("  launch_bound_tok_s_at_2795 median=%.2f optimistic_from_min=%.2f\n", tok_s_med, tok_s_min);
  fflush(stdout);
}

int main(int argc, char** argv) {
  int launches = 5000;
  int warm_passes = 2;
  int trials = 10;
  if (argc > 1) launches = atoi(argv[1]);
  if (argc > 2) trials = atoi(argv[2]);
  if (argc > 3) warm_passes = atoi(argv[3]);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device=%s gcnArchName=%s CUs=%d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
  printf("launches=%d trials=%d warm_passes=%d\n", launches, trials, warm_passes);
  fflush(stdout);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int* d_counter = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  HIP_CHECK(hipMemset(d_counter, 0, sizeof(int)));

  auto bench = [&](const char* name, bool use_null, dim3 grid, dim3 block) {
    // warm
    for (int w = 0; w < warm_passes; w++) {
      for (int i = 0; i < launches; i++) {
        if (use_null) hipLaunchKernelGGL(null_kernel, grid, block, 0, stream);
        else hipLaunchKernelGGL(add_one_kernel, grid, block, 0, stream, d_counter);
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    std::vector<double> times;
    times.reserve(trials);
    for (int t = 0; t < trials; t++) {
      hipEvent_t t0, t1;
      HIP_CHECK(hipEventCreate(&t0));
      HIP_CHECK(hipEventCreate(&t1));
      // Also wall clock for host-side view
      HIP_CHECK(hipEventRecord(t0, stream));
      for (int i = 0; i < launches; i++) {
        if (use_null) hipLaunchKernelGGL(null_kernel, grid, block, 0, stream);
        else hipLaunchKernelGGL(add_one_kernel, grid, block, 0, stream, d_counter);
      }
      HIP_CHECK(hipEventRecord(t1, stream));
      HIP_CHECK(hipEventSynchronize(t1));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
      times.push_back(ms);
      HIP_CHECK(hipEventDestroy(t0));
      HIP_CHECK(hipEventDestroy(t1));
    }
    report(name, launches, times);
  };

  // null kernel empty grid? use 1x1x1
  bench("null_kernel_1x1x1", true, dim3(1), dim3(1));
  bench("add_one_1x1x1", false, dim3(1), dim3(1));
  // also a slightly larger trivial kernel
  bench("null_kernel_64x64", true, dim3(64), dim3(64));
  // back-to-back with hipDeviceSynchronize after each launch (serialized drain)
  {
    for (int w = 0; w < warm_passes; w++) {
      for (int i = 0; i < 200; i++) {
        hipLaunchKernelGGL(null_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
      }
    }
    const int nser = 1000;
    std::vector<double> times;
    for (int t = 0; t < trials; t++) {
      hipEvent_t t0, t1;
      HIP_CHECK(hipEventCreate(&t0));
      HIP_CHECK(hipEventCreate(&t1));
      HIP_CHECK(hipEventRecord(t0, stream));
      for (int i = 0; i < nser; i++) {
        hipLaunchKernelGGL(null_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
      }
      HIP_CHECK(hipEventRecord(t1, stream));
      HIP_CHECK(hipEventSynchronize(t1));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
      times.push_back(ms);
      HIP_CHECK(hipEventDestroy(t0));
      HIP_CHECK(hipEventDestroy(t1));
    }
    report("null_kernel_1x1x1_sync_each", nser, times);
  }

  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipStreamDestroy(stream));
  printf("DONE\n");
  return 0;
}
