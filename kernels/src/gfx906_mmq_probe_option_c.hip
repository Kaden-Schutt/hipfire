#include <hip/hip_runtime.h>
#include <stdint.h>

// Phase 2 probe — sync-frequency reduction (plan §P2).
//
// Topology: 64-K x_qs window covers 4 sub-iters' worth of x data per LDS load.
// The 4 sub-iters then run as straight-line code (no __syncthreads between
// them), against resident x_qs and resident tile_y. Each HFQ4 group runs
// 2 windows × (load + 4× compute) = 2 reload events per group, vs the
// shipping body.cuh's 8 reload events per group. 4× sync reduction.
//
// LDS budget at mmq_x=64:
//   x_qs:   128 rows × 32 ints × 4 B = 16,384 B  (16 KiB)
//   x_dm:   128 × float2             =  1,024 B
//   tile_y: 64 × 36 × 4 B            =  9,216 B
//   total                            = 26,624 B  (26 KiB) — fits 2 WGs/CU
//
// vs current shipping (X_STRIDE=8, 32-K window): x_qs = 4 KiB, total = 14 KiB
//
// Risk being measured: did the 4× inner-unroll cause register pressure
// to spill? If vgpr_count ≤ 128 with 0 spills (or ≤ 200), proceed. The
// previous Phase 2b session's full-8 unroll caused 986 spills at x64.
// Going to 4 should be much safer but needs measurement.

#define MMQ_X 64
#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32

// 64-K window = 32 ints/row (vs shipping's 8 ints/row for 32-K window).
#define X_STRIDE_64K 32
#define Y_STRIDE 36

extern "C" __global__ void __launch_bounds__(256, 2)
gemm_hfq4g256_mmq_probe_option_c(
    const char* __restrict__ A,
    const char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ int x_qs[MMQ_Y * X_STRIDE_64K];   // 128 * 32 * 4 = 16,384 B
    __shared__ float x_dm[MMQ_Y * 2];            //               = 1,024 B
    __shared__ int tile_y[MMQ_X * Y_STRIDE];     //  64 * 36 * 4 = 9,216 B

    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;  // 0..255
    const int i0  = threadIdx.x;                           // 0..63 (lane)
    const int j0_base = threadIdx.y * (MMQ_X / MMQ_NWARPS); // 0,16,32,48

    // Per-thread accumulator (32 floats, same as Option B).
    float sum[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) sum[k] = (float)tid;

    // Outer "K loop": each iteration represents a 64-K window.
    // (Real kernel runs 2 iterations per HFQ4 group × groups_per_row.
    //  Probe runs 8 to give the compiler an unrollable shape and to
    //  match the shipping body's per-group work count.)
    for (int win = 0; win < 8; ++win) {

        // ── Load phase: 64-K of x (32 ints/row) + Q8_1 block of y ──
        if (tid < 128) {
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                x_qs[tid * X_STRIDE_64K + k] = (int)A[tid * 32 + k + win * 4];
            }
            x_dm[tid * 2 + 0] = (float)B[tid + win];
            x_dm[tid * 2 + 1] = (float)B[tid + win + 1];
        }
        if (tid < 64) {
            #pragma unroll
            for (int k = 0; k < Y_STRIDE; ++k) {
                tile_y[tid * Y_STRIDE + k] = (int)B[tid * Y_STRIDE + k + win];
            }
        }
        __syncthreads();  // sync 1: producers → consumers

        // ── Compute phase: 4 sub-iters straight-line, no syncs between ──
        // This is the load-bearing change vs Option B. Option B had
        // __syncthreads at the bottom of every sub_iter; here we run
        // 4 sub-iters back-to-back against resident LDS.
        //
        // j0 inner loop: rolled (#pragma unroll 1) to keep live-range
        // pressure manageable. Per Phase 2b's earlier observation
        // (commit c022682), full j0 unroll causes 200+ register spills
        // at mmq_x=64. The shipping body.cuh runs j0 rolled (8 stride)
        // and we mirror that here.
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int v_x0 = ((volatile int*)x_qs)[i0 * X_STRIDE_64K + sub * 8 + 0];
            int v_x1 = ((volatile int*)x_qs)[i0 * X_STRIDE_64K + sub * 8 + 4];

            #pragma unroll 1
            for (int j0_off = 0; j0_off < 16; ++j0_off) {
                int j0 = j0_base + j0_off;
                int v_y0 = ((volatile int*)tile_y)[j0 * Y_STRIDE + 4 + sub * 8 + i0 / 8];
                int v_y1 = ((volatile int*)tile_y)[j0 * Y_STRIDE + 4 + sub * 8 + 4 + i0 / 8];

                asm volatile (
                    "v_dot4_i32_i8 %0, %1, %2, %0\n\t"
                    "v_dot4_i32_i8 %3, %4, %5, %3\n\t"
                    : "+v"(sum[j0_off * 2 + 0]), "+v"(sum[j0_off * 2 + 1])
                    : "v"(v_x0), "v"(v_y0), "v"(v_x1), "v"(v_y1)
                );
            }
        }
        __syncthreads();  // sync 2: before next window's load
    }

    // Write-back (prevents dead-code elimination of `sum`).
    if (tid < M) {
        C[tid] = sum[0] + sum[15] + sum[31];
    }
}
