#include <hip/hip_runtime.h>
#include <stdint.h>

#define MMQ_X 64
#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32

// Alignment padding for ds_read_b128 alignment
#define X_STRIDE 68
#define Y_STRIDE 36

extern "C" __global__ void __launch_bounds__(256, 2)
gemm_hfq4g256_mmq_probe_gfx906(
    const char* __restrict__ A,
    const char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Phase 2a Probe: Measure VGPRs and LDS for the nwarps=4 redesign.
    
    // 1. LDS Allocation (Check Gate 3)
    __shared__ int x_qs[MMQ_Y * X_STRIDE]; // 128 * 68 * 4 = 34,816 B
    __shared__ float x_dm[MMQ_Y * 2];      // 128 * 2 * 4 = 1,024 B
    __shared__ int tile_y[MMQ_X * Y_STRIDE]; // 64 * 36 * 4 = 9,216 B
    
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255
    const int i0 = threadIdx.x; // 0..63
    const int j0_base = threadIdx.y * (MMQ_X / MMQ_NWARPS); // 0, 16, 32, 48
    
    // 2. Accumulators (Check Gate 1)
    float sum[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) sum[k] = (float)tid;
    
    // Dummy loop to force compute and register pressure
    for (int sub_iter = 0; sub_iter < 8; ++sub_iter) {
        
        // DUMMY WORK to ensure LDS is not optimized away
        if (tid < 128) {
            x_qs[tid + sub_iter] = (int)A[tid + sub_iter];
            x_dm[tid] = (float)B[tid];
        }
        if (tid < 64) {
            tile_y[tid * Y_STRIDE] = (int)B[tid + 128];
        }
        __syncthreads();

        // REALISTIC LOADS
        // Using volatile to prevent optimization
        int v_x = ((volatile int*)x_qs)[i0 + sub_iter * X_STRIDE];
        float sc_x = ((volatile float*)x_dm)[i0 * 2];
        float zp_x = ((volatile float*)x_dm)[i0 * 2 + 1];
        
        #pragma unroll
        for (int j0_off = 0; j0_off < 16; ++j0_off) {
            int j0 = j0_base + j0_off;
            
            // Per-warp tile access
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
    
    // Write back to avoid optimization
    if (tid < M) {
        C[tid] = sum[0] + sum[31];
    }
}
