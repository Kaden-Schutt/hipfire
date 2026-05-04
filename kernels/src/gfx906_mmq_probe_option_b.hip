#include <hip/hip_runtime.h>
#include <stdint.h>

#define MMQ_X 64
#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32

// Option B: 32-K streaming x_qs.
// 32 K-elements unpacked = 8 ints/row. With X_STRIDE_STREAM=16 the per-row
// LDS footprint is 16 ints × 4 B = 64 B (16-byte aligned, headroom for
// ds_read_b128). Total x_qs = 128 rows × 16 ints × 4 B = 8 KiB.
//
// NOTE (2026-05-04): the shipping body.cuh uses X_STRIDE=8 (no headroom)
// because the dp4a path only reads 8 ints/row. This probe stub keeps =16
// for historical comparison with phase2a_probe_results.md.

#define X_STRIDE_STREAM 16
#define Y_STRIDE 36

extern "C" __global__ void __launch_bounds__(256, 2)
gemm_hfq4g256_mmq_probe_option_b(
    const char* __restrict__ A,
    const char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 1. LDS Allocation (Option B: Streaming)
    __shared__ int x_qs[MMQ_Y * X_STRIDE_STREAM]; // 128 * 16 * 4 = 8,192 B
    __shared__ float x_dm[MMQ_Y * 2];             // 128 * 2 * 4 = 1,024 B
    __shared__ int tile_y[MMQ_X * Y_STRIDE];        // 64 * 36 * 4 = 9,216 B
    // Total LDS = 8,192 + 1,024 + 9,216 = 18,432 B = 18 KiB.
    // 18 KiB * 2 WGs = 36 KiB < 64 KiB. PASSED.
    
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;
    const int i0 = threadIdx.x;
    const int j0_base = threadIdx.y * (MMQ_X / MMQ_NWARPS);
    
    float sum[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) sum[k] = (float)tid;
    
    // kg loop would be here in real kernel.
    // sub_iter loop (8x) for Option B streaming.
    for (int sub_iter = 0; sub_iter < 8; ++sub_iter) {
        
        if (tid < 128) {
            // Load 32-K chunk into LDS
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                x_qs[tid * X_STRIDE_STREAM + k] = (int)A[tid + k];
            }
            x_dm[tid] = (float)B[tid];
        }
        if (tid < 64) {
            tile_y[tid * Y_STRIDE] = (int)B[tid + 128];
        }
        __syncthreads();

        int v_x = ((volatile int*)x_qs)[i0 * X_STRIDE_STREAM + (sub_iter % 8)];
        
        #pragma unroll
        for (int j0_off = 0; j0_off < 16; ++j0_off) {
            int j0 = j0_base + j0_off;
            int v_y0 = ((volatile int*)tile_y)[j0 * Y_STRIDE + 4 + i0 / 2];
            int v_y1 = ((volatile int*)tile_y)[j0 * Y_STRIDE + 4 + i0 / 2 + 16];
            
            asm volatile (
                "v_dot4_i32_i8 %0, %1, %2, %0\n\t"
                "v_dot4_i32_i8 %3, %1, %4, %3\n\t"
                : "+v"(sum[j0_off * 2 + 0]), "+v"(sum[j0_off * 2 + 1])
                : "v"(v_x), "v"(v_y0), "v"(v_y1)
            );
        }
        
        __syncthreads();
    }
    
    if (tid < M) {
        C[tid] = sum[0] + sum[31];
    }
}
