// SPDX-License-Identifier: Apache-2.0
// hipfire-train — causal mask for attention scores (fp32).
//
// scores is [seq_q, seq_k] row-major. For query row i, keys j > i are future
// positions: set them to a large negative so softmax zeros them. The masked
// value is a constant (no dependence on Q/K), so the backward needs no special
// handling — softmax_bwd yields ~0 gradient there.

#include <hip/hip_runtime.h>

extern "C" __global__ void causal_mask_train(
    float* __restrict__ scores, // [seq_q, seq_k]
    int seq_q, int seq_k
) {
    int i = blockIdx.x; // query row
    if (i >= seq_q) return;
    for (int j = threadIdx.x; j < seq_k; j += blockDim.x) {
        if (j > i) scores[(size_t)i * seq_k + j] = -1.0e30f;
    }
}
