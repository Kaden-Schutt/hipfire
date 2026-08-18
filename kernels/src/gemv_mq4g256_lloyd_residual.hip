// MagnumQuant 4-bit GEMV (Lloyd codebook) with fused residual add.
// y[row] += A[row] · x.
//
// Same body as gemv_mq4g256_lloyd.hip (16-way ternary lookup; baseline for
// non-RDNA3 archs). Only the final write differs (+= instead of =).
// Mirrors gemv_mq3g256_lloyd_residual.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256_lloyd_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 160;
    const int boff = tid * 4;

    __shared__ float cbt[16];

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 160;

        if (tid < 16) {
            cbt[tid] = __half2float(((const __half*)gptr)[tid]);
        }
        __syncthreads();

        const unsigned char* data = (const unsigned char*)(gptr + 32);

        unsigned int b0 = (unsigned int)data[boff];
        unsigned int b1 = (unsigned int)data[boff + 1];
        unsigned int b2 = (unsigned int)data[boff + 2];
        unsigned int b3 = (unsigned int)data[boff + 3];

        unsigned int q0 =  b0       & 0xFu;
        unsigned int q1 = (b0 >> 4) & 0xFu;
        unsigned int q2 =  b1       & 0xFu;
        unsigned int q3 = (b1 >> 4) & 0xFu;
        unsigned int q4 =  b2       & 0xFu;
        unsigned int q5 = (b2 >> 4) & 0xFu;
        unsigned int q6 =  b3       & 0xFu;
        unsigned int q7 = (b3 >> 4) & 0xFu;

        int base_idx = g * 256 + tid * 8;

        acc += cbt[q0] * x[base_idx]
             + cbt[q1] * x[base_idx + 1]
             + cbt[q2] * x[base_idx + 2]
             + cbt[q3] * x[base_idx + 3]
             + cbt[q4] * x[base_idx + 4]
             + cbt[q5] * x[base_idx + 5]
             + cbt[q6] * x[base_idx + 6]
             + cbt[q7] * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
