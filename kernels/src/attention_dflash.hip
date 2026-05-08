// Non-causal cross-attention kernel for DFlash draft forward.
//
// Each query position attends to ALL key/value positions (non-causal,
// bidirectional). Supports GQA where n_heads > n_kv_heads. Q/K/V are
// separate buffers — K can be longer than Q. Grid: [n_heads, B, 1],
// one block per (head, query-position).
//
// Layout (row-major, flat):
//   q    : [B  * n_heads    * head_dim]
//   k    : [L  * n_kv_heads * head_dim]
//   v    : [L  * n_kv_heads * head_dim]
//   out  : [B  * n_heads    * head_dim]
//
// rep = n_heads / n_kv_heads (must divide evenly).
//
// PR3 step 2 follow-up: this is a tiled online-softmax (FlashAttention-style)
// implementation. Replaces the prior single-pass `scores[L]` allocation that
// overflowed gfx1100/RDNA3's 64 KB LDS budget at L > ~16128. With tiling,
// LDS = (tile_size + block_size + head_dim) * 4 bytes — independent of L.
// Single-tile case (tile_size = L when L ≤ tile_size) is mathematically
// equivalent to the prior kernel up to FP non-associativity in the same
// reduce trees. Multi-tile case carries a running (max, sum, output)
// triplet across tiles, mathematically identical to single-pass softmax
// modulo FP order.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_dflash_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale,
    int tile_size
) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int qi = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (head >= n_heads || qi >= B) return;

    int rep = n_heads / n_kv_heads;
    int kv_head = head / rep;
    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    // LDS layout: [tile_scores[tile_size]] [ws[nthreads]] [out_run[head_dim]]
    float* tile_scores = sdata;                                // [tile_size]
    float* ws          = sdata + tile_size;                    // [nthreads]
    float* out_run     = sdata + tile_size + nthreads;         // [head_dim]

    const float* q_ptr = q + qi * q_stride + head * head_dim;

    // Initialize running output to 0.
    for (int d = tid; d < head_dim; d += nthreads) {
        out_run[d] = 0.0f;
    }
    __syncthreads();

    // Online-softmax running state.
    float m_run = -1e30f;  // running max of all scores seen so far
    float l_run = 0.0f;    // running sum of exp(scores - m_run)

    int n_tiles = (L + tile_size - 1) / tile_size;
    for (int t = 0; t < n_tiles; ++t) {
        int tile_start = t * tile_size;
        int tile_len = L - tile_start;
        if (tile_len > tile_size) tile_len = tile_size;

        // Phase A: compute scores for this tile, track tile-local max.
        float local_max = -1e30f;
        for (int j = tid; j < tile_len; j += nthreads) {
            int abs_j = tile_start + j;
            const float* k_ptr = k + abs_j * kv_stride + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_ptr[d] * k_ptr[d];
            }
            float s = dot * scale;
            tile_scores[j] = s;
            if (s > local_max) local_max = s;
        }
        ws[tid] = local_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float a = ws[tid];
                float b = ws[tid + s];
                ws[tid] = a > b ? a : b;
            }
            __syncthreads();
        }
        float tile_max = ws[0];
        __syncthreads();

        // Phase B: rescale running state to the new global max.
        //   m_new   = max(m_run, tile_max)
        //   alpha   = exp(m_run - m_new)             (= 1 if m_run won, < 1 otherwise)
        //   l_new   = alpha * l_run + sum_j exp(score_j - m_new)
        //   out_new = alpha * out_run + sum_j exp(score_j - m_new) * V[j]
        // First tile (m_run = -inf): alpha = exp(-inf - finite) = 0, so the
        // running state correctly reduces to fresh-tile contribution.
        float m_new = m_run > tile_max ? m_run : tile_max;
        float alpha = expf(m_run - m_new);

        // Convert tile_scores in place to exp(score - m_new), summing.
        float local_sum = 0.0f;
        for (int j = tid; j < tile_len; j += nthreads) {
            float e = expf(tile_scores[j] - m_new);
            tile_scores[j] = e;
            local_sum += e;
        }
        ws[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] += ws[tid + s];
            __syncthreads();
        }
        float tile_sum = ws[0];
        __syncthreads();

        // Update running denominator.
        l_run = alpha * l_run + tile_sum;

        // Update running output: each thread updates the head_dim slots it owns.
        for (int d = tid; d < head_dim; d += nthreads) {
            float scaled_old = alpha * out_run[d];
            float contrib = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
                int abs_j = tile_start + j;
                contrib += tile_scores[j] * v[abs_j * kv_stride + kv_head * head_dim + d];
            }
            out_run[d] = scaled_old + contrib;
        }
        __syncthreads();

        m_run = m_new;
    }

    // Finalize: out = out_run / l_run.
    float* out_ptr = out + qi * q_stride + head * head_dim;
    float inv_l = 1.0f / l_run;
    for (int d = tid; d < head_dim; d += nthreads) {
        out_ptr[d] = out_run[d] * inv_l;
    }
}
