#include <hip/hip_runtime.h>

// Per-row softmax-prob gather, temperature-scaled.
//
// For each row r in [0, n_rows):
//   row_logits = logits[r * vocab : (r+1) * vocab]
//   inv_temp   = 1 / temperature
//   row_max    = max(row_logits[i] * inv_temp)
//   row_sum    = sum_i exp(row_logits[i] * inv_temp - row_max)
//   target_e   = exp(row_logits[indices[r]] * inv_temp - row_max)
//   probs_out[r] = target_e / row_sum
//
// Numerically stable max-subtraction softmax. Returns one prob per row.
//
// Use cases (MTP residual-acceptance sampling spec-decode):
//   - n_rows = 1: gather p_draft(c_k) for a single draft sample
//   - n_rows = K: batched gather of p_target(c_k) for K candidates against
//     trunk verify logits (one prob per verify position)
//
// Launch: n_rows blocks × 256 threads. Each block processes one row with
// parallel reduce over vocab. Shared mem: 1 KB scratch + 4 B target.
//
// Output bytes per call: n_rows × 4 = tiny D2H (~20 bytes for K=5).
// Replaces the 6 MB host-side D2H + softmax loop with a few-microsecond
// kernel + tiny D2H.

extern "C" __global__ void softmax_prob_gather_batched(
    const float* __restrict__ logits,    // [n_rows × vocab]
    const int* __restrict__ indices,     // [n_rows] target token ID per row
    float* __restrict__ probs_out,       // [n_rows] output prob
    int vocab,
    float temperature                     // must be > 0
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int idx = indices[row];
    const float inv_temp = 1.0f / temperature;
    const float* row_logits = logits + (size_t)row * (size_t)vocab;

    __shared__ float scratch[256];
    __shared__ float s_target;

    // Phase 1: row max of (logit * inv_temp).
    float local_max = -INFINITY;
    for (int i = tid; i < vocab; i += nt) {
        float v = row_logits[i] * inv_temp;
        if (v > local_max) local_max = v;
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();

    // Init target slot (default 0 if idx is out of range — caller bug).
    if (tid == 0) s_target = 0.0f;
    __syncthreads();

    // Phase 2: sum exp + capture target exp.
    // Exactly one thread will hit i == idx (the one where tid == idx % nt).
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += nt) {
        float e = expf(row_logits[i] * inv_temp - row_max);
        local_sum += e;
        if (i == idx) s_target = e;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float row_sum = scratch[0];
    __syncthreads();

    // Thread 0 writes.
    if (tid == 0) {
        float p = (row_sum > 0.0f) ? (s_target / row_sum) : 0.0f;
        probs_out[row] = p;
    }
}
