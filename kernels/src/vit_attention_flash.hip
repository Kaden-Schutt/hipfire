#include <hip/hip_runtime.h>

// Optimized bidirectional attention for the vision encoder.
// Computes: out[qi, head] = softmax(Q[qi] @ K^T / sqrt(d)) @ V
//
// Improvements over vit_attention_f32:
//   1. Tiled K/V loading: loads K tiles into shared memory, reuses Q across tile
//   2. Online softmax: no full scores array needed (saves N * sizeof(float) shared mem)
//   3. Multiple queries per block: amortizes kernel launch overhead
//
// Grid: [n_heads, ceil(N / queries_per_block)]
// Block: [256]
// LDS: tile_size * head_dim * sizeof(float) for K tile + workspace

#define TILE_SIZE 64      // K/V tile size (positions per tile)
#define QUERIES_PER_BLK 4 // Query positions processed per block

__launch_bounds__(256, 2)
extern "C" __global__ void vit_attention_flash(
    const float* __restrict__ qkv,   // [N, 3*hidden] row-major, interleaved Q/K/V
    float* __restrict__ out,          // [N, hidden] row-major
    int N, int hidden, int num_heads, int head_dim, float scale
) {
    const int head = blockIdx.x;
    const int query_group = blockIdx.y;  // which group of queries
    const int tid = threadIdx.x;

    if (head >= num_heads) return;

    const int stride = 3 * hidden;
    const int q_off = head * head_dim;
    const int k_off = hidden + head * head_dim;
    const int v_off = 2 * hidden + head * head_dim;

    // Shared memory: K tile [TILE_SIZE * head_dim] + workspace
    extern __shared__ float smem[];
    float* k_tile = smem;                              // [TILE_SIZE * head_dim]
    float* v_tile = smem + TILE_SIZE * head_dim;       // [TILE_SIZE * head_dim]

    // Process multiple query positions per block
    for (int qi_offset = 0; qi_offset < QUERIES_PER_BLK; qi_offset++) {
        int qi = query_group * QUERIES_PER_BLK + qi_offset;
        if (qi >= N) continue;

        const float* q_ptr = qkv + qi * stride + q_off;

        // Online softmax state
        float row_max = -1e30f;
        float row_sum = 0.0f;
        float acc[16];  // accumulator for output (up to head_dim=16 per thread)
        #pragma unroll
        for (int d = 0; d < 16; d++) acc[d] = 0.0f;

        // Process K/V in tiles
        for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
            int tile_end = tile_start + TILE_SIZE;
            if (tile_end > N) tile_end = N;
            int tile_len = tile_end - tile_start;

            // Cooperative load K tile into shared memory
            // Each thread loads multiple elements
            for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
                int pos = i / head_dim;
                int d = i % head_dim;
                k_tile[i] = qkv[(tile_start + pos) * stride + k_off + d];
            }
            // Cooperative load V tile
            for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
                int pos = i / head_dim;
                int d = i % head_dim;
                v_tile[i] = qkv[(tile_start + pos) * stride + v_off + d];
            }
            __syncthreads();

            // Compute dot products against K tile (from shared memory)
            // Each thread handles a subset of tile positions
            for (int t = tid; t < tile_len; t += blockDim.x) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    dot += q_ptr[d] * k_tile[t * head_dim + d];
                }
                float s = dot * scale;

                // Online softmax update
                float new_max = fmaxf(row_max, s);
                float old_correction = expf(row_max - new_max);
                float new_score = expf(s - new_max);

                // Update accumulator with correction
                #pragma unroll
                for (int dd = 0; dd < 16; dd++) {
                    if (dd < head_dim) {
                        acc[dd] = acc[dd] * old_correction + new_score * v_tile[t * head_dim + dd];
                    }
                }
                row_sum = row_sum * old_correction + new_score;
                row_max = new_max;
            }

            __syncthreads();
        }

        // Final normalize and write output
        float* out_row = out + qi * hidden + head * head_dim;
        float inv_sum = 1.0f / row_sum;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_row[d] = acc[d % 16] * inv_sum;
        }
    }
}
