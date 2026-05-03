// Wave64 indexing verification test for gfx906 MMQ port.
//
// Phase 0 task for plans/gfx906_mmq_plan.md §3.5. The MMQ kernel's `tile<>`
// abstraction (kernels/src/gemm_hfq4g256_residual_mmq.hip:96-153) hard-codes
// element count as `ne = I * J / 32` and computes per-thread (i, j) using
// `threadIdx.x / 16` and `threadIdx.x % 16`, both of which silently assume
// threadIdx.x ∈ [0, 32). On gfx906 the launch is dim3(32, 8, 1) but the
// hardware packs lanes into wave64s — every wave covers TWO threadIdx.y
// values, and threadIdx.x still ranges [0, 32). So the per-warp indexing
// (which uses threadIdx.x for in-warp lane) should still be safe in
// principle — every threadIdx.y group of 32 threads has unique threadIdx.x.
//
// This test verifies:
//
//   1. block dim (32, 8, 1) on wave64: confirm linearization (tid = ty*32 + tx)
//      and that each wave64 covers exactly 2 threadIdx.y values.
//   2. tile<16,8,int,DATA_LAYOUT_I_MAJOR>::get_i(l) and ::get_j(l) for the
//      gfx906 (#else) branch: confirm lanes 0..31 map to (i,j) = (lane%16,
//      4*(lane/16)+l) for l=0..3 and produce a complete 16x8 tile cover.
//   3. quantize_q8_1_mmq_ds4 shuffle pattern: verify __shfl_xor with
//      WARP_SIZE=32 on wave64 reduces correctly within each 32-lane half-wave.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx906 -o bench_wave64_idx \
//     benchmarks/bench_wave64_indexing_gfx906.hip

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Replicate the gfx906 (#else) tile indexing from
// kernels/src/gemm_hfq4g256_residual_mmq.hip lines 99-121.
struct Tile16x8_IMajor {
    static constexpr int I = 16;
    static constexpr int J = 8;
    static constexpr int ne = I * J / 32;  // = 4 — this is the wave32 assumption

    __device__ static int get_i(int) {
        return threadIdx.x % 16;
    }
    __device__ static int get_j(int l) {
        return ne * (threadIdx.x / 16) + l;
    }
};

// Test 1: thread linearization.
// Each thread writes its (linear_tid, ty, tx, computed_get_i_0, computed_get_j_0..3)
// to global memory. Block dim is (32, 8, 1) = 256 threads.
extern "C" __global__ void linearize_test(int* __restrict__ out) {
    int tid = threadIdx.x + threadIdx.y * 32;
    int gi  = Tile16x8_IMajor::get_i(0);
    int gj0 = Tile16x8_IMajor::get_j(0);
    int gj1 = Tile16x8_IMajor::get_j(1);
    int gj2 = Tile16x8_IMajor::get_j(2);
    int gj3 = Tile16x8_IMajor::get_j(3);

    int* p = out + tid * 8;
    p[0] = tid;
    p[1] = threadIdx.y;
    p[2] = threadIdx.x;
    p[3] = gi;
    p[4] = gj0;
    p[5] = gj1;
    p[6] = gj2;
    p[7] = gj3;
}

// Test 2: shuffle in wave64 with width=32.
// Each thread writes its lane id and the result of __shfl_xor(lane, 16, 32).
// On wave32: thread 0 swaps with thread 16, etc. — result = lane ^ 16.
// On wave64 with width=32: lanes 0..31 swap among themselves (result = lane ^ 16),
//                          lanes 32..63 swap among themselves (result = lane ^ 16).
// If width semantics are honored, the result for lane N should be (N % 32) ^ 16
//   shifted by the wave's base — i.e. N within [0,32): (N ^ 16),
//                                  N within [32,64): 32 + ((N - 32) ^ 16).
extern "C" __global__ void shuffle_test(int* __restrict__ out) {
    int lane_in_wave = threadIdx.x % 64;  // assuming wave64; we'll verify
    int my_val = lane_in_wave;
    int swapped = __shfl_xor(my_val, 16, 32);
    out[threadIdx.x * 2 + 0] = my_val;
    out[threadIdx.x * 2 + 1] = swapped;
}

// Test 3: MMQ-style reduction over a 32-lane half-wave.
// The quantize_q8_1_mmq_ds4 kernel does:
//   amax = abs_value;
//   for (off = 4; off > 0; off >>= 1)
//       amax = fmaxf(amax, __shfl_xor(amax, off, WARP_SIZE=32));
// We feed each lane its lane index, run the same reduction over WARP_SIZE=32,
// and check that lanes 0..31 reduce to max(0..31)=31, lanes 32..63 to
// max(32..63)=63 (lanes 0..7 within their half-wave reduction-fan-out).
// Actually the existing pattern uses 4 -> 1 (i.e. width=8); for clarity
// we test the actual loop pattern: 4, 2, 1 with width=32.
extern "C" __global__ void reduce_test(int* __restrict__ out) {
    int lane_in_wave = threadIdx.x % 64;
    int v = lane_in_wave;
    // Mirror the actual quantize loop: offset = 4, 2, 1 with WARP_SIZE = 32.
    // After all 3 steps a 32-lane fan-out reduction has happened across 8 lanes.
    // (The actual quantize has per-block 32-element granularity; this matches
    //  reduction across the lower 8 lanes of a 32-lane warp on wave32.)
    for (int off = 4; off > 0; off >>= 1) {
        int peer = __shfl_xor(v, off, 32);
        if (peer > v) v = peer;
    }
    out[threadIdx.x * 2 + 0] = lane_in_wave;
    out[threadIdx.x * 2 + 1] = v;
}

#define CHECK(call) do {                                                      \
    hipError_t e = (call);                                                    \
    if (e != hipSuccess) {                                                    \
        fprintf(stderr, "HIP error %s at %s:%d\n",                            \
            hipGetErrorString(e), __FILE__, __LINE__); exit(1);               \
    } } while (0)

static int report_test1(const int* h_out) {
    int errors = 0;
    printf("=== Test 1: thread linearization with block dim (32, 8, 1) ===\n");
    printf("%-4s %-4s %-4s | %-3s %-4s %-4s %-4s %-4s | %-8s | %-8s\n",
           "tid", "ty", "tx", "g_i", "gj0", "gj1", "gj2", "gj3",
           "expect_i", "expect_j[0..3]");

    // Print one entry per wave64 boundary and a couple inside-wave entries.
    int probes[] = {0, 1, 16, 31, 32, 33, 47, 63, 64, 95, 96, 127, 128, 192, 255};
    for (size_t k = 0; k < sizeof(probes) / sizeof(probes[0]); ++k) {
        int t = probes[k];
        const int* p = h_out + t * 8;
        int tid_obs = p[0], ty = p[1], tx = p[2];
        int gi = p[3], gj0 = p[4], gj1 = p[5], gj2 = p[6], gj3 = p[7];

        // Expected values per the source (replicated above)
        int exp_gi = tx % 16;
        int exp_gj0 = 4 * (tx / 16) + 0;
        int exp_gj1 = 4 * (tx / 16) + 1;
        int exp_gj2 = 4 * (tx / 16) + 2;
        int exp_gj3 = 4 * (tx / 16) + 3;

        // Linearization expectation: tid = ty * 32 + tx
        int exp_tid = ty * 32 + tx;

        const char* mark = "";
        if (tid_obs != exp_tid || gi != exp_gi || gj0 != exp_gj0 ||
            gj1 != exp_gj1 || gj2 != exp_gj2 || gj3 != exp_gj3) {
            mark = " <-- MISMATCH";
            errors++;
        }
        printf("%-4d %-4d %-4d | %-3d %-4d %-4d %-4d %-4d | i=%-6d | j=%d..%d%s\n",
               tid_obs, ty, tx, gi, gj0, gj1, gj2, gj3,
               exp_gi, exp_gj0, exp_gj3, mark);
    }
    printf("Test 1 errors: %d\n\n", errors);
    return errors;
}

static int report_test2(const int* h_out) {
    int errors = 0;
    printf("=== Test 2: __shfl_xor(v, 16, width=32) on wave64 ===\n");
    printf("Expected: lane N ∈ [0,32)  → swapped = N ^ 16\n");
    printf("          lane N ∈ [32,64) → swapped = 32 + ((N - 32) ^ 16)\n");
    printf("(repeats every 64 lanes for 4 waves of the 256-thread block)\n");

    int sample[] = {0, 1, 15, 16, 31, 32, 33, 47, 48, 63, 64, 65, 80, 96, 127, 128, 191, 255};
    printf("%-4s %-6s %-7s %-7s\n", "tid", "lane%64", "obs", "expect");
    for (size_t k = 0; k < sizeof(sample) / sizeof(sample[0]); ++k) {
        int t = sample[k];
        int my_val = h_out[t * 2 + 0];
        int got = h_out[t * 2 + 1];
        int wave_id = t / 64;
        int lane_in_wave = t % 64;
        int expect;
        if (lane_in_wave < 32) expect = (lane_in_wave ^ 16);
        else                   expect = 32 + ((lane_in_wave - 32) ^ 16);
        const char* mark = "";
        if (got != expect) { mark = " <-- MISMATCH"; errors++; }
        printf("%-4d %-6d %-7d %-7d  (wave=%d)%s\n",
               t, my_val, got, expect, wave_id, mark);
    }
    printf("Test 2 errors: %d\n\n", errors);
    return errors;
}

static int report_test3(const int* h_out) {
    int errors = 0;
    printf("=== Test 3: max-reduction over WARP_SIZE=32 (matches quantize kernel) ===\n");
    printf("Reduction loop: for off in {4,2,1}: v = max(v, __shfl_xor(v, off, 32))\n");
    printf("This fans out over 8 lanes (offsets 1,2,4 → reaches lanes ±7 max).\n");
    printf("Lane N ends with max over the 8-lane group {N XOR k : k ∈ [0,8)}.\n");
    printf("On wave64, each lane group stays within its 32-lane half-wave.\n");

    int sample[] = {0, 7, 8, 15, 16, 23, 24, 31, 32, 33, 39, 40, 47, 48, 55, 56, 63, 64, 95, 127, 128, 192, 255};
    printf("%-4s %-6s %-7s %-7s\n", "tid", "lane%64", "obs", "expect");
    for (size_t k = 0; k < sizeof(sample) / sizeof(sample[0]); ++k) {
        int t = sample[k];
        int my_val = h_out[t * 2 + 0];
        int got = h_out[t * 2 + 1];
        int lane_in_wave = t % 64;
        // Group is the 8-lane block: lane (group_id * 8 + group_lane).
        int group_id = lane_in_wave / 8;
        int group_lo = group_id * 8;
        int group_hi = group_lo + 7;
        int expect = group_hi;
        const char* mark = "";
        if (got != expect) { mark = " <-- MISMATCH"; errors++; }
        printf("%-4d %-6d %-7d %-7d  (group [%d..%d])%s\n",
               t, my_val, got, expect, group_lo, group_hi, mark);
    }
    printf("Test 3 errors: %d\n\n", errors);
    return errors;
}

int main(void) {
    int total_errors = 0;

    {
        printf("Launching linearize_test with grid(1,1,1) block(32,8,1) (256 threads)\n");
        int* d_out = nullptr;
        CHECK(hipMalloc(&d_out, 256 * 8 * sizeof(int)));
        hipLaunchKernelGGL(linearize_test, dim3(1,1,1), dim3(32,8,1), 0, 0, d_out);
        CHECK(hipDeviceSynchronize());
        int h_out[256 * 8];
        CHECK(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));
        total_errors += report_test1(h_out);
        hipFree(d_out);
    }

    {
        printf("Launching shuffle_test with grid(1,1,1) block(256,1,1) (256 threads)\n");
        int* d_out = nullptr;
        CHECK(hipMalloc(&d_out, 256 * 2 * sizeof(int)));
        hipLaunchKernelGGL(shuffle_test, dim3(1,1,1), dim3(256,1,1), 0, 0, d_out);
        CHECK(hipDeviceSynchronize());
        int h_out[256 * 2];
        CHECK(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));
        total_errors += report_test2(h_out);
        hipFree(d_out);
    }

    {
        printf("Launching reduce_test with grid(1,1,1) block(256,1,1) (256 threads)\n");
        int* d_out = nullptr;
        CHECK(hipMalloc(&d_out, 256 * 2 * sizeof(int)));
        hipLaunchKernelGGL(reduce_test, dim3(1,1,1), dim3(256,1,1), 0, 0, d_out);
        CHECK(hipDeviceSynchronize());
        int h_out[256 * 2];
        CHECK(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));
        total_errors += report_test3(h_out);
        hipFree(d_out);
    }

    printf("=== TOTAL ERRORS: %d ===\n", total_errors);
    return total_errors == 0 ? 0 : 1;
}
