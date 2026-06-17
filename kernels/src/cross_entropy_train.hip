// SPDX-License-Identifier: Apache-2.0
// hipfire-train — fused cross-entropy (logsoftmax + NLL) forward+backward.
//
// One block per row (token). Given logits[row, V] and an integer target:
//   lse        = logsumexp(logits)
//   loss[row]  = lse - logits[target]                 (0 if target == ignore)
//   d_logits   = softmax(logits) - onehot(target)     (0 if target == ignore)
//
// Targets arrive as f32 (exact for vocab < 2^24); ignore_index likewise. This
// is a SUM-reduction gradient: the caller divides by the valid-token count for
// a mean loss. Forward and backward are fused because they always run together
// and share the softmax.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(64)
__global__ void cross_entropy_train(
    const float* __restrict__ logits, // [rows, V]
    const float* __restrict__ targets, // [rows]  (integer-valued f32)
    float* __restrict__ loss,         // [rows]   out
    float* __restrict__ d_logits,     // [rows, V] out
    int rows, int V, int ignore_index
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* lr = logits + (size_t)r * V;
    float* dr = d_logits + (size_t)r * V;
    int tgt = (int)targets[r];

    if (tgt == ignore_index) {
        if (tid == 0) loss[r] = 0.0f;
        for (int i = tid; i < V; i += nt) dr[i] = 0.0f;
        return;
    }

    __shared__ float red[64];
    // max
    float m = -3.402823e38f;
    for (int i = tid; i < V; i += nt) m = fmaxf(m, lr[i]);
    red[tid] = m;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();
    // sum exp
    float se = 0.0f;
    for (int i = tid; i < V; i += nt) se += expf(lr[i] - mx);
    red[tid] = se;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float sumexp = red[0];
    float lse = mx + logf(sumexp);
    float inv = 1.0f / sumexp;

    if (tid == 0) loss[r] = lse - lr[tgt];
    for (int i = tid; i < V; i += nt) {
        float p = expf(lr[i] - mx) * inv;
        dr[i] = p - (i == tgt ? 1.0f : 0.0f);
    }
}
