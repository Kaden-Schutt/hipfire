#include <hip/hip_runtime.h>

// HFQ6-G256 batched GEMV with fused sigmoid-scaled residual add:
//   y_batch[bid, row] += sigmoid(c_batch[bid]) * (A[row] . x_batch[bid])
//
// Mirrors `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` exactly in
// kernel shape (grid = M × N, block = 32, one warp per (row, token)) but
// reads HFQ6's 200 B / group layout (4 B scale + 4 B zero + 192 B packed
// 6-bit nibbles, 4 weights per 3 bytes — see `gemv_hfq6g256_residual.hip`
// for the wave32 single-token reference).
//
// Used by the batched MoE FFN shared-expert `down` projection when the
// shared.down weight is MQ6 (12 of 40 layers in AWQ A3B fall into this
// case). MQ6G256 shares storage with HFQ6G256 — the caller applies the
// FWHT rotation upstream, same convention as MQ4 / HFQ4.
//
// No atomicAdd needed: each (bid, row) block writes a unique
// y_batch[bid * M + row] cell. Matches the HFQ4 batched sibling.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
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
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 4 weights per 3 bytes -> 8 weights = 6 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 6;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];
        unsigned char b5 = data[byte_off + 5];

        // 4 weights from first 3 bytes
        int q0 = b0 & 63;
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        int q3 = b2 >> 2;
        // 4 weights from next 3 bytes
        int q4 = b3 & 63;
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        int q7 = b5 >> 2;

        acc += (scale * (float)q0 + zero) * x[base_idx]
             + (scale * (float)q1 + zero) * x[base_idx + 1]
             + (scale * (float)q2 + zero) * x[base_idx + 2]
             + (scale * (float)q3 + zero) * x[base_idx + 3]
             + (scale * (float)q4 + zero) * x[base_idx + 4]
             + (scale * (float)q5 + zero) * x[base_idx + 5]
             + (scale * (float)q6 + zero) * x[base_idx + 6]
             + (scale * (float)q7 + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_batch[bid]));
        y_batch[(long long)bid * M + row] += gate * acc;
    }
}
