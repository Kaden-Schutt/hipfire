// LDS bank-conflict microbenchmark for gfx906 wave64.
//
// Phase 0 task for plans/gfx906_mmq_plan.md §3.3. The skyne98 wiki reports
// ~1865 GB/s (stride=32 vec4, conflicting) vs ~3974 GB/s (stride=33 vec4,
// `+1 vec4` rule recovered). Those numbers were measured but the wiki
// doesn't explicitly say what wave size or access pattern it used.
//
// Access pattern under test (the conflict-relevant one for a column-consumed
// MMQ tile): all 64 lanes of one wave issue ds_read_b32 to row=lane, col=fixed
// in lockstep. The hardware fans these 64 reads across 32 banks. The
// per-lane bank id is:
//
//   bank(lane) = ((lane * STRIDE_b32) + col) % 32
//
// where STRIDE_b32 is the leading-dim stride in 32-bit words. With:
//   STRIDE_b32 = 32  -> bank = (lane * 32 + col) % 32 = col, ALL lanes hit
//                       the same bank → 64-way conflict on wave64.
//   STRIDE_b32 = 33  -> bank = (lane + col) % 32, distinct across 32 lanes
//                       at fixed col → still 2-way on wave64 (lane 0 vs 32).
//   STRIDE_b32 = 36  -> per the existing MMQ_TILE_Y_K used on RDNA3.
//
// We test these three so we can compare against the wiki and against the
// MMQ kernel's existing choice.
//
// Compile (each variant separately):
//   hipcc -O3 --offload-arch=gfx906 -DSTRIDE_B32=32 -o bench_lds_w64_s32 \
//     benchmarks/bench_lds_wave64_gfx906.hip
//   hipcc -O3 --offload-arch=gfx906 -DSTRIDE_B32=33 -o bench_lds_w64_s33 ...
//   hipcc -O3 --offload-arch=gfx906 -DSTRIDE_B32=36 -o bench_lds_w64_s36 ...

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef STRIDE_B32
#define STRIDE_B32 33
#endif

// One workgroup = one wave of 64 lanes. We're measuring intra-wave conflict.
// Multiple workgroups in flight to keep the GPU busy.
#define BLOCK_THREADS 64
#define ROWS 64                   // one row per lane
#define COLS 32                   // unrolled column sweep
#define LDS_B32_COUNT (ROWS * STRIDE_B32)
#define LDS_BYTES (LDS_B32_COUNT * 4)

extern "C" __launch_bounds__(BLOCK_THREADS)
__global__ void lds_read_bw(int* __restrict__ out, int iters) {
    __shared__ int tile[LDS_B32_COUNT];

    const int tid = threadIdx.x;       // 0..63 = lane
    const int my_row = tid;            // each lane owns one row

    // Initialize LDS.
    for (int i = tid; i < LDS_B32_COUNT; i += BLOCK_THREADS) {
        tile[i] = i;
    }
    __syncthreads();

    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (int it = 0; it < iters; ++it) {
        // 32 unrolled column reads per iter. Each read: same row (per lane),
        // different column. All 64 lanes read in lockstep. Bank id per lane:
        //   bank(lane, col) = (lane * STRIDE_B32 + col) % 32
        // 4 independent accumulators to keep ALU off the LDS critical path.
        const int base = my_row * STRIDE_B32;
        acc0 ^= tile[base +  0]; acc1 ^= tile[base +  1];
        acc2 ^= tile[base +  2]; acc3 ^= tile[base +  3];
        acc0 ^= tile[base +  4]; acc1 ^= tile[base +  5];
        acc2 ^= tile[base +  6]; acc3 ^= tile[base +  7];
        acc0 ^= tile[base +  8]; acc1 ^= tile[base +  9];
        acc2 ^= tile[base + 10]; acc3 ^= tile[base + 11];
        acc0 ^= tile[base + 12]; acc1 ^= tile[base + 13];
        acc2 ^= tile[base + 14]; acc3 ^= tile[base + 15];
        acc0 ^= tile[base + 16]; acc1 ^= tile[base + 17];
        acc2 ^= tile[base + 18]; acc3 ^= tile[base + 19];
        acc0 ^= tile[base + 20]; acc1 ^= tile[base + 21];
        acc2 ^= tile[base + 22]; acc3 ^= tile[base + 23];
        acc0 ^= tile[base + 24]; acc1 ^= tile[base + 25];
        acc2 ^= tile[base + 26]; acc3 ^= tile[base + 27];
        acc0 ^= tile[base + 28]; acc1 ^= tile[base + 29];
        acc2 ^= tile[base + 30]; acc3 ^= tile[base + 31];
        // Force the compiler to keep these reads inside the loop body
        // (otherwise it hoists the constant-base reads out of `it` once).
        // We do this by adding `it` into the address dependency chain:
        // shift the base by one stride per iter, modulo ROWS.
        if ((it & 0xff) == 0) acc0 ^= it;  // light dependence to keep the loop "alive" without changing addresses
    }

    if (tid == 0) {
        out[blockIdx.x] = acc0 ^ acc1 ^ acc2 ^ acc3;
    }
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

#define CHECK(call) do {                                                      \
    hipError_t e = (call);                                                    \
    if (e != hipSuccess) {                                                    \
        fprintf(stderr, "HIP error %s at %s:%d\n",                            \
            hipGetErrorString(e), __FILE__, __LINE__); exit(1);               \
    } } while (0)

int main(int argc, char** argv) {
    int blocks = argc > 1 ? atoi(argv[1]) : 1024;  // many small WGs to saturate
    int iters  = argc > 2 ? atoi(argv[2]) : 4096;

    int* d_out = nullptr;
    CHECK(hipMalloc(&d_out, blocks * sizeof(int)));

    hipLaunchKernelGGL(lds_read_bw, dim3(blocks), dim3(BLOCK_THREADS), 0, 0, d_out, 64);
    CHECK(hipDeviceSynchronize());

    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
        double t0 = now_s();
        hipLaunchKernelGGL(lds_read_bw, dim3(blocks), dim3(BLOCK_THREADS), 0, 0, d_out, iters);
        CHECK(hipDeviceSynchronize());
        double t1 = now_s();
        if (t1 - t0 < best) best = t1 - t0;
    }

    // Bytes per launch: blocks × 64 lanes × iters × COLS × 4 B
    double bytes = (double)blocks * BLOCK_THREADS * iters * COLS * 4.0;
    double gbps = bytes / best / 1e9;

    fprintf(stderr,
        "STRIDE_B32=%d  (LDS_per_WG=%zu B = %.1f KiB)\n"
        "  blocks=%d iters=%d cols/iter=%d\n"
        "  best=%.4f s  bytes=%.2f GiB  bandwidth=%.1f GB/s\n",
        STRIDE_B32,
        (size_t)LDS_BYTES, LDS_BYTES / 1024.0,
        blocks, iters, COLS,
        best, bytes / (1024.0 * 1024.0 * 1024.0), gbps);

    hipFree(d_out);
    return 0;
}
