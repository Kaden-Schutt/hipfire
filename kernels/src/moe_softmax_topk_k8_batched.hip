#include <hip/hip_runtime.h>

// Batched MoE router softmax + top-8 + (optional) renormalize.
//
// Identical per-block algorithm to `moe_softmax_topk_renorm_k8` — one
// workgroup per token. Extends to N tokens by launching N workgroups:
// `blockIdx.x` selects the token row. All shared-memory reductions stay
// within a single block, so the only change from the single-token kernel
// is pointer striding by `bid * n_exp` (logits) or `bid * K_TOP`
// (outputs).
//
// Drop-in batched replacement for the per-token MoE top-K used in
// prefill. Enables `forward_prefill_chunk` to route MoE layers in one
// launch instead of N.

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      8

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_softmax_topk_renorm_k8_batched(
    const float* __restrict__ logits_in,   // [N × n_exp]
    int*         __restrict__ topk_idx,    // [N × 8]
    float*       __restrict__ topk_w,      // [N × 8]
    int n_exp,
    int norm_topk                          // 0 = keep raw probs, 1 = renormalize
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const float* __restrict__ row_logits = logits_in + (long long)bid * n_exp;
    int*   __restrict__ row_idx = topk_idx + bid * K_TOP;
    float* __restrict__ row_w   = topk_w   + bid * K_TOP;

    // ── 1. Load logits into shared memory ──
    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? row_logits[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    // ── 2. Softmax over n_exp ──
    float my_max = -INFINITY;
    for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
        my_max = fmaxf(my_max, smem_v[i]);
    }
    reduce_v[tid] = my_max;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_v[tid] = fmaxf(reduce_v[tid], reduce_v[tid + s]);
        __syncthreads();
    }
    const float max_val = reduce_v[0];

    float my_sum = 0.0f;
    for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
        float e = expf(smem_v[i] - max_val);
        smem_v[i] = e;
        my_sum += e;
    }
    __syncthreads();
    reduce_v[tid] = my_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_v[tid] += reduce_v[tid + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_v[0];

    for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
        smem_v[i] = smem_v[i] * inv_sum;
    }
    __syncthreads();

    // ── 3. Iterative top-K: 8 rounds of argmax + mask ──
    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float v = smem_v[i];
            if (v > best_v) { best_v = v; best_i = smem_i[i]; }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float va = reduce_v[tid];
                float vb = reduce_v[tid + s];
                if (vb > va) {
                    reduce_v[tid] = vb;
                    reduce_i[tid] = reduce_i[tid + s];
                }
            }
            __syncthreads();
        }
        const int   winner = reduce_i[0];
        const float winner_w = reduce_v[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k]   = winner_w;
        }
        if (winner >= 0 && winner < n_exp) {
            if ((winner & (BLOCK_SIZE - 1)) == tid) {
                smem_v[winner] = -INFINITY;
            }
        }
        __syncthreads();
    }

    // ── 4. Optional renormalize of the 8 picked probs ──
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < K_TOP; k++) s += picked_w[k];
        if (norm_topk && s > 0.0f) {
            const float inv = 1.0f / s;
            for (int k = 0; k < K_TOP; k++) picked_w[k] *= inv;
        }
    }
    __syncthreads();

    // ── 5. Write output ──
    // Clamp the winner to a valid expert range before write-out. See the
    // single-token kernel for the full rationale (closes #152).
    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        row_idx[tid] = safe_idx;
        row_w[tid]   = safe_w;
    }
}
