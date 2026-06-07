// Logit soft-capping (Gemma 2/3/4 final-logit softcap).
//
//   out[i] = tanh(x[i] / cap) * cap
//
// Applied in-place over the flat logits vector (262144 on Gemma 4).
// Used only on final LM-head outputs before sampling — Gemma 4 has NO
// per-attention-layer softcap (unlike Gemma 2).
//
// Grid: 1D over N; block 256 threads.
#include <hip/hip_runtime.h>

extern "C" __global__ void logit_softcap_f32(
    float* __restrict__ x,   // in-place [n]
    int n,
    float cap                // e.g. 30.0
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.0f / cap;
    x[i] = tanhf(x[i] * inv_cap) * cap;
}
