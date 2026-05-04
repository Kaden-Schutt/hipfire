#include <hip/hip_runtime.h>
#include <stdint.h>

#define MMQ_X 64
#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32

// Option B: 32-K streaming x_qs
// (128 rows * 32 K-elements / 4 elements-per-int) = 1024 ints.
// But we need padding for ds_read_b128 alignment (X_STRIDE=68)
// Wait, 32 K-elements is 32/4 = 8 ints.
// If X_STRIDE is 68, it covers 68 * 4 = 272 K-elements.
// My previous calc for Option B: "x_qs = 128 rows * 32 K-elements... no"
// Stock's Q4_0 x_qs size = mmq_y * MMQ_TILE_NE_K = 128 * 32 = 4096 ints?
// No, K elements are nibbles. 128 rows * 32 K-elements = 4096 nibbles = 2048 bytes.
// Wait, stock's calc for Q4_0: mmq_y * MMQ_TILE_NE_K + mmq_y.
// 128 * 32 + 128 = 4224 ints.
// Why ints? Because they unpack into ints for dp4a.
// So 128 rows * 32 elements * 1 int/element (if unpacked) = 4096 ints.
// Our unpacked x_qs stride was 68. 128 * 68 = 8704 ints.
// 8704 * 4 = 34,816 B. (This was the 256-K case).
// 32-K case: 128 rows * 8 ints-per-row (since 32/4 = 8).
// 128 * 8 = 1024 ints = 4096 B.
// Plus padding for alignment? If we use X_STRIDE=68 we are back to 34KB.
// If we want to stream 32-K, we only need X_STRIDE=8 or 12 or 16.
// Let's use X_STRIDE=16 (64 bytes, 16-byte aligned).
// 128 * 16 * 4 = 8192 B = 8 KiB.

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
