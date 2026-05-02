// PFlash per-block scoring on Q8_0 K cache.
//
// One workgroup per output block. Each thread strides through kv_dim,
// dequantizes K[pos][d] for every position in its block, accumulates
// the per-thread (block_mean . last_k, ||block_mean||^2, ||last_k||^2)
// fragment, then a 256-thread shared-memory reduction folds the per-
// thread fragments into one cosine score per block.
//
// Q8_0 layout (matches engine::llama::KvCache::new_gpu_q8):
//   per position, per head: blocks_per_head = head_dim / 32
//   per block: 2 bytes f16 scale + 32 bytes int8 values
//   per head bytes: blocks_per_head * 34
//   per position bytes: n_kv_heads * per-head bytes
//
// The dequant of the last position's K is recomputed by every thread
// inside its kv_dim stride — small enough (head_dim/32 = 4 reads per
// thread on Qwen3-0.6B's head_dim=128) that the simple form is faster
// than a separate pre-dequant kernel + extra global memory traffic.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void pflash_score_q8_kv_blocks(
    const unsigned char* __restrict__ k_cache,
    float* __restrict__ scores,
    int n_pos,
    int n_kv_heads,
    int head_dim,
    int block_size,
    int n_blocks,
    int last_pos
) {
    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    int block_start = block_idx * block_size;
    int block_end = block_start + block_size;
    if (block_end > n_pos) block_end = n_pos;
    int block_len = block_end - block_start;
    if (block_len <= 0) {
        if (threadIdx.x == 0) scores[block_idx] = 0.0f;
        return;
    }

    int kv_dim = n_kv_heads * head_dim;
    int blocks_per_head = head_dim / 32;
    int bytes_per_head = blocks_per_head * 34;
    int bytes_per_pos = n_kv_heads * bytes_per_head;

    __shared__ float s_dot[256];
    __shared__ float s_nb[256];
    __shared__ float s_nl[256];

    int tid = threadIdx.x;

    float dot = 0.0f;
    float nb = 0.0f;
    float nl = 0.0f;

    for (int d = tid; d < kv_dim; d += 256) {
        int h = d / head_dim;
        int dim_in_head = d - h * head_dim;
        int blk_in_head = dim_in_head / 32;
        int j_in_blk = dim_in_head - blk_in_head * 32;

        // last position K[d]
        const unsigned char* last_blk =
            k_cache + last_pos * bytes_per_pos + h * bytes_per_head + blk_in_head * 34;
        unsigned short last_scale_bits =
            (unsigned short)last_blk[0] | ((unsigned short)last_blk[1] << 8);
        __half last_scale_h;
        last_scale_h = *reinterpret_cast<const __half*>(&last_scale_bits);
        float last_scale = __half2float(last_scale_h);
        float last_v = (float)((signed char)last_blk[2 + j_in_blk]) * last_scale;

        // Block-mean K[d]: sum over positions in this block, divide.
        float sum = 0.0f;
        for (int pos = block_start; pos < block_end; ++pos) {
            const unsigned char* blk_bytes =
                k_cache + pos * bytes_per_pos + h * bytes_per_head + blk_in_head * 34;
            unsigned short scale_bits =
                (unsigned short)blk_bytes[0] | ((unsigned short)blk_bytes[1] << 8);
            __half scale_h;
            scale_h = *reinterpret_cast<const __half*>(&scale_bits);
            float scale = __half2float(scale_h);
            sum += (float)((signed char)blk_bytes[2 + j_in_blk]) * scale;
        }
        float mean = sum / (float)block_len;

        dot += mean * last_v;
        nb += mean * mean;
        nl += last_v * last_v;
    }

    s_dot[tid] = dot;
    s_nb[tid] = nb;
    s_nl[tid] = nl;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
            s_nb[tid] += s_nb[tid + stride];
            s_nl[tid] += s_nl[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = sqrtf(s_nb[0]) * sqrtf(s_nl[0]);
        if (denom < 1e-12f) denom = 1e-12f;
        scores[block_idx] = s_dot[0] / denom;
    }
}
