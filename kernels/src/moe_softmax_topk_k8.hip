#include <hip/hip_runtime.h>

// MoE router softmax + top-8 + (optional) renormalize, in a single launch.
// Reads [n_exp] logits (up to 1024, bounded by one workgroup = 1024 threads),
// writes [8] i32 indices and [8] f32 weights to device buffers — no D2H
// sync required. Replaces the CPU-side softmax/sort/renorm that would
// otherwise block hipGraph capture.
//
// One workgroup, 256 threads. Supports n_exp up to 1024 by looping (each
// thread holds up to 4 logits). For n_exp == 256 (A3B) each thread handles
// one logit and the loops degenerate.
//
// Algorithm:
//   1. Load logits into shared memory, pad with -INF.
//   2. Numerically-stable softmax (find max → exp(l - max) → sum → divide).
//   3. Iteratively pick top-K: argmax reduction, record (idx, prob),
//      then replace the winning slot with -INF so the next iteration
//      picks a different expert.
//   4. If norm_topk_prob, renormalize the 8 weights to sum to 1.

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      8

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_softmax_topk_renorm_k8(
    const float* __restrict__ logits_in,   // [n_exp]
    int*         __restrict__ topk_idx,    // [8]
    float*       __restrict__ topk_w,      // [8]
    int n_exp,
    int norm_topk                           // 0 = keep raw probs, 1 = renormalize
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];

    const int tid = threadIdx.x;

    // ── 1. Load logits into shared memory ──
    // Each thread may hold up to ceil(n_exp / BLOCK_SIZE) values.
    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? logits_in[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    // ── 2. Softmax over n_exp ──
    // 2a. Find max (for numerical stability).
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

    // 2b. Exp and sum.
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
    const float sum_exp = reduce_v[0];
    const float inv_sum = 1.0f / sum_exp;

    // 2c. Divide → smem_v now holds softmaxed probs.
    for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
        smem_v[i] = smem_v[i] * inv_sum;
    }
    __syncthreads();

    // ── 3. Iterative top-K: 8 rounds of argmax + mask ──
    for (int k = 0; k < K_TOP; k++) {
        // Per-thread partial argmax over this thread's slice of smem_v.
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float v = smem_v[i];
            if (v > best_v) { best_v = v; best_i = smem_i[i]; }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();

        // Tree reduction — compare (value, index) pairs.
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
        // Mask out the winner so the next round picks a different expert.
        // Only one thread (the one holding that slot) does the write.
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
    // Clamp the winner to a valid expert range before write-out. Under MQ4
    // expert quant on 122B-A10B, a NaN in the softmax can cause the argmax
    // reduction to select the sentinel `-1` (or any garbage); without this
    // guard the dispatch picks an out-of-bounds expert and the matmul
    // either GPU-faults (aperture violation) or reads stale weights.
    // Clamp: if invalid, route to expert 0 with weight 0 — output stays
    // mathematically correct (zero contribution) and dispatch stays in-bounds.
    // Closes #152.
    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        topk_idx[tid] = safe_idx;
        topk_w[tid]   = safe_w;
    }
}
