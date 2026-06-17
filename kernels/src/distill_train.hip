// SPDX-License-Identifier: Apache-2.0
// hipfire-train — KL distillation loss (fwd+bwd), fp32. Phase 2 Q1.
//
// Soft-target analogue of cross-entropy: the target is a full teacher
// distribution p_t[row, V] (probabilities) rather than a one-hot. Per row:
//   q       = softmax(student_logits)
//   loss[r] = KL(p_t ‖ q) = Σ_v p_t (log p_t - log q)
//   d_logits = q - p_t        (gradient of the soft cross-entropy term)
//
// One block per row, shared-mem reductions (max, sumexp, KL). Sum-reduction
// gradient — caller divides by row count for a mean.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(64)
__global__ void distill_kl_train(
    const float* __restrict__ student, // [rows, V]
    const float* __restrict__ teacher_p, // [rows, V] (probabilities)
    float* __restrict__ loss,          // [rows]
    float* __restrict__ d_logits,      // [rows, V]
    int rows, int V
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* sr = student + (size_t)r * V;
    const float* pr = teacher_p + (size_t)r * V;
    float* dr = d_logits + (size_t)r * V;

    __shared__ float red[64];
    // max
    float m = -3.402823e38f;
    for (int i = tid; i < V; i += nt) m = fmaxf(m, sr[i]);
    red[tid] = m;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();
    // sumexp
    float se = 0.0f;
    for (int i = tid; i < V; i += nt) se += expf(sr[i] - mx);
    red[tid] = se;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float lse = mx + logf(red[0]);
    __syncthreads();
    // KL = Σ p_t (log p_t - (s - lse)) ; d = exp(s-lse) - p_t
    float kl = 0.0f;
    for (int i = tid; i < V; i += nt) {
        float q = expf(sr[i] - lse);
        float p = pr[i];
        dr[i] = q - p;
        if (p > 1e-12f) kl += p * (logf(p) - (sr[i] - lse));
    }
    red[tid] = kl;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    if (tid == 0) loss[r] = red[0];
}
