#include <hip/hip_runtime.h>

// HFQ5-G256 batched GEMV with fused sigmoid-scaled residual add:
//   y_batch[bid, row] += sigmoid(c_batch[bid]) * (A[row] . x_batch[bid])
//
// Mirrors `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` exactly in
// kernel shape (grid = M × N, block = 32, one warp per (row, token)) but
// reads HFQ5's 168 B / group layout (4 B scale + 4 B zero + 160 B packed
// 5-bit codes, 8 weights per 5 bytes — see `gemv_hfq5g256_residual.hip`
// for the wave32 single-token reference).
//
// Used by the batched MoE FFN shared-expert `down` projection when the
// shared.down weight is MQ5 (12 of 40 layers in AWQ A3B fall into this
// case). MQ5G256 shares storage with HFQ5G256 — the caller applies the
// FWHT rotation upstream, same convention as MQ4 / HFQ4.
//
// No atomicAdd needed: each (bid, row) block writes a unique
// y_batch[bid * M + row] cell. Matches the HFQ4 batched sibling.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq5g256_residual_sigmoid_scaled_gpu_batched(
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
    const int row_bytes = groups_per_row * 168;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 168;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 8 weights per 5 bytes -> 8 weights = 5 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 5;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];

        int q0 =  b0 & 0x1f;
        int q1 = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3);
        int q2 =  (b1 >> 2) & 0x1f;
        int q3 = ((b1 >> 7) & 0x01) | ((b2 & 0x0f) << 1);
        int q4 = ((b2 >> 4) & 0x0f) | ((b3 & 0x01) << 4);
        int q5 =  (b3 >> 1) & 0x1f;
        int q6 = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2);
        int q7 =  (b4 >> 3) & 0x1f;

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
