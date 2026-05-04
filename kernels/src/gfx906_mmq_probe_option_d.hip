#include <hip/hip_runtime.h>
#include <stdint.h>

// Phase 2 probe — even more aggressive: 256-K window per HFQ4 group.
// One LDS load covers the whole group, 8 sub-iters run as straight-line
// code, only 2 syncs per group (one per Q8_1 block tile_y switch).
//
// LDS budget at mmq_x=64:
//   x_qs:   128 rows × 64 ints × 4 B = 32,768 B  (32 KiB)
//   x_dm:   128 × float2             =  1,024 B
//   tile_y: 64 × 36 × 4 B            =  9,216 B
//   total                            = 43,008 B  (42 KiB) — EXCEEDS 32 KiB cap
//
// This would force 1 WG/CU. Probe records whether VGPR/spill budget
// at least permits the 1-WG/CU fallback (Phase 2a Option A).

#define MMQ_X 64
#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64

#define X_STRIDE_256K 64
#define Y_STRIDE 36

extern "C" __global__ void __launch_bounds__(256, 1)  // NOTE: 1 WG/CU
gemm_hfq4g256_mmq_probe_option_d(
    const char* __restrict__ A,
    const char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ int x_qs[MMQ_Y * X_STRIDE_256K];  // 128 * 64 * 4 = 32,768 B
    __shared__ float x_dm[MMQ_Y * 2];            //               =  1,024 B
    __shared__ int tile_y[MMQ_X * Y_STRIDE];     //  64 * 36 * 4 =  9,216 B

    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;
    const int i0  = threadIdx.x;
    const int j0_base = threadIdx.y * (MMQ_X / MMQ_NWARPS);

    float sum[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) sum[k] = (float)tid;

    // Outer loop = HFQ4 groups. Each iter loads a full 256-K window
    // and runs 8 sub-iters back-to-back (with one mid-loop tile_y
    // switch between blocks A and B).
    for (int kg = 0; kg < 4; ++kg) {
        // Load full group: 64 ints/row × 128 rows = 8,192 ints.
        // 256 threads × 32 ints/thread = 8,192. Each thread loads 32.
        if (tid < 128) {
            #pragma unroll
            for (int k = 0; k < 64; ++k) {
                x_qs[tid * X_STRIDE_256K + k] = (int)A[tid * 64 + k + kg];
            }
            x_dm[tid * 2 + 0] = (float)B[tid + kg];
            x_dm[tid * 2 + 1] = (float)B[tid + kg + 1];
        }
        // Q8_1 block A
        if (tid < 64) {
            #pragma unroll
            for (int k = 0; k < Y_STRIDE; ++k) {
                tile_y[tid * Y_STRIDE + k] = (int)B[tid * Y_STRIDE + k + kg];
            }
        }
        __syncthreads();  // sync 1: producers → consumers

        // Sub-iters 0..3 against block A
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int v_x0 = ((volatile int*)x_qs)[i0 * X_STRIDE_256K + sub * 8 + 0];
            int v_x1 = ((volatile int*)x_qs)[i0 * X_STRIDE_256K + sub * 8 + 4];
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

        // Switch tile_y to block B
        __syncthreads();  // sync 2: prep for tile_y B reload
        if (tid < 64) {
            #pragma unroll
            for (int k = 0; k < Y_STRIDE; ++k) {
                tile_y[tid * Y_STRIDE + k] = (int)B[tid * Y_STRIDE + k + kg + 100];
            }
        }
        __syncthreads();  // sync 3: producers → consumers (block B ready)

        // Sub-iters 4..7 against block B (reads x_qs[..., sub*8+32+...])
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int v_x0 = ((volatile int*)x_qs)[i0 * X_STRIDE_256K + 32 + sub * 8 + 0];
            int v_x1 = ((volatile int*)x_qs)[i0 * X_STRIDE_256K + 32 + sub * 8 + 4];
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
    }

    if (tid < M) {
        C[tid] = sum[0] + sum[15] + sum[31];
    }
}
