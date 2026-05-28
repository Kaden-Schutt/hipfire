#include <hip/hip_runtime.h>

// HFQ3/MQ3-G256 batched GEMV with fused sigmoid-scaled residual add:
//   y_batch[bid, row] += sigmoid(c_batch[bid]) * (A[row] . x_batch[bid])
//
// This is the HFQ3 104 B/group sibling of the HFQ4/HFQ6 shared-expert down
// kernels used by batched MoE prefill. MQ3 uses the same storage as HFQ3; the
// caller pre-rotates x_batch when weights are MQ3.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched(
    const char* __restrict__ A,
    const float* __restrict__ x_batch,     // [N x K]
    float* __restrict__ y_batch,           // [N x M]
    const float* __restrict__ c_batch,     // [N]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bid = blockIdx.y;
    const int tid = threadIdx.x;

    const float* __restrict__ x = x_batch + (long long)bid * K;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 104;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * 104;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* data = (const unsigned char*)(gp + 8);

        // 256 weights / 32 threads = 8 weights per thread.
        // 3-bit packing: 8 weights per 3 bytes.
        const int base = g * 256 + tid * 8;
        const int byte_off = tid * 3;

        unsigned int b0 = data[byte_off + 0];
        unsigned int b1 = data[byte_off + 1];
        unsigned int b2 = data[byte_off + 2];

        int q0 = b0 & 7;
        int q1 = (b0 >> 3) & 7;
        int q2 = ((b0 >> 6) | (b1 << 2)) & 7;
        int q3 = (b1 >> 1) & 7;
        int q4 = (b1 >> 4) & 7;
        int q5 = ((b1 >> 7) | (b2 << 1)) & 7;
        int q6 = (b2 >> 2) & 7;
        int q7 = (b2 >> 5) & 7;

        acc += (scale * (float)q0 + zero) * x[base]
             + (scale * (float)q1 + zero) * x[base + 1]
             + (scale * (float)q2 + zero) * x[base + 2]
             + (scale * (float)q3 + zero) * x[base + 3]
             + (scale * (float)q4 + zero) * x[base + 4]
             + (scale * (float)q5 + zero) * x[base + 5]
             + (scale * (float)q6 + zero) * x[base + 6]
             + (scale * (float)q7 + zero) * x[base + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_batch[bid]));
        y_batch[(long long)bid * M + row] += gate * acc;
    }
}
