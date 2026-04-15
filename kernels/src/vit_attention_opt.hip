#include <hip/hip_runtime.h>

// Optimized bidirectional attention for the vision encoder.
// Computes: out[qi, head] = softmax(Q[qi] @ K^T / sqrt(d)) @ V
//
// Improvements over vit_attention_f32:
//   1. Tiled K loading into shared memory — coalesced reads, reuse across threads
//   2. 4 queries per block — reduces block count by 4x (fewer launches, better occupancy)
//   3. Pre-loaded Q in registers — avoids repeated global reads
//
// Grid: [n_heads, ceil(N/4)]
// Block: [256]
// LDS: 128 * head_dim * 4 bytes (K tile) + N * 4 bytes (scores) + 256 * 4 (workspace)
//      ≈ 128*72*4 + 784*4 + 256*4 ≈ 41KB — fits in gfx1100's 64KB LDS

#define QPB 2  // queries per block
#define K_TILE 64

__launch_bounds__(256, 2)
extern "C" __global__ void vit_attention_opt(
    const float* __restrict__ qkv,   // [N, 3*hidden] row-major, interleaved Q/K/V
    float* __restrict__ out,          // [N, hidden] row-major
    int N, int hidden, int num_heads, int head_dim, float scale
) {
    const int head = blockIdx.x;
    const int qi_base = blockIdx.y * QPB;
    const int tid = threadIdx.x;

    if (head >= num_heads) return;

    const int stride = 3 * hidden;
    const int q_off = head * head_dim;
    const int k_off = hidden + head * head_dim;
    const int v_off = 2 * hidden + head * head_dim;

    extern __shared__ float smem[];
    float* k_tile = smem;                    // [K_TILE * head_dim]
    float* scores = smem + K_TILE * head_dim; // [N] — shared across QPB queries
    float* ws = scores + N;                  // [256] workspace for reduction

    for (int qi_off = 0; qi_off < QPB; qi_off++) {
        int qi = qi_base + qi_off;
        if (qi >= N) continue;

        // Load Q into a local buffer (read once)
        float q_reg[72];  // up to head_dim=72
        const float* q_ptr = qkv + qi * stride + q_off;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            q_reg[d] = q_ptr[d];
        }
        __syncthreads();

        // Phase 1: Compute attention scores using tiled K loading
        for (int tile_start = 0; tile_start < N; tile_start += K_TILE) {
            int tile_end = tile_start + K_TILE;
            if (tile_end > N) tile_end = N;
            int tile_len = tile_end - tile_start;

            // Cooperative load K tile (coalesced)
            for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
                int pos = i / head_dim;
                int d = i % head_dim;
                k_tile[i] = qkv[(tile_start + pos) * stride + k_off + d];
            }
            __syncthreads();

            // Each thread computes dot products for a subset of tile positions
            for (int t = tid; t < tile_len; t += blockDim.x) {
                float dot = 0.0f;
                const float* kt = k_tile + t * head_dim;
                #pragma unroll 8
                for (int d = 0; d < head_dim; d++) {
                    dot += q_reg[d] * kt[d];
                }
                scores[tile_start + t] = dot * scale;
            }
            __syncthreads();
        }

        // Phase 2: Softmax (unchanged from original — it's already correct)
        // Find max
        float local_max = -1e30f;
        for (int j = tid; j < N; j += blockDim.x) {
            local_max = fmaxf(local_max, scores[j]);
        }
        ws[tid] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
            __syncthreads();
        }
        float max_val = ws[0];
        __syncthreads();

        // Exp + sum
        float local_sum = 0.0f;
        for (int j = tid; j < N; j += blockDim.x) {
            float e = expf(scores[j] - max_val);
            scores[j] = e;
            local_sum += e;
        }
        ws[tid] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] += ws[tid + s];
            __syncthreads();
        }
        float sum_val = ws[0];
        __syncthreads();

        // Normalize scores
        for (int j = tid; j < N; j += blockDim.x) {
            scores[j] /= sum_val;
        }
        __syncthreads();

        // Phase 3: Weighted V sum using tiled V loading
        float* out_row = out + qi * hidden + head * head_dim;

        // Initialize output
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_row[d] = 0.0f;
        }
        __syncthreads();

        for (int tile_start = 0; tile_start < N; tile_start += K_TILE) {
            int tile_end = tile_start + K_TILE;
            if (tile_end > N) tile_end = N;
            int tile_len = tile_end - tile_start;

            // Load V tile (reuse K tile memory)
            for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
                int pos = i / head_dim;
                int d = i % head_dim;
                k_tile[i] = qkv[(tile_start + pos) * stride + v_off + d];
            }
            __syncthreads();

            // Weighted sum for this tile
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float val = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    val += scores[tile_start + t] * k_tile[t * head_dim + d];
                }
                out_row[d] += val;
            }
            __syncthreads();
        }
    }
}
