#include <hip/hip_runtime.h>

// conv1d + silu + Q/K/V split fused into one kernel.
//
// Replaces the sequence
//   conv1d_silu_f32(dn_conv_out, dn_qkv, weight, state, qkv_dim);
//   memcpy_dtod_at(q_out, 0, dn_conv_out, 0,           k_dim * 4);
//   memcpy_dtod_at(k_out, 0, dn_conv_out, k_dim*4,     k_dim * 4);
//   memcpy_dtod_at(v_out, 0, dn_conv_out, k_dim*2*4,   v_dim * 4);
//
// with a single launch. Each thread processes one channel: does the 1D
// conv + SiLU in registers (bit-identical to conv1d_silu.hip), then
// writes to q_out / k_out / v_out depending on channel index.
//
// Channel layout: [Q (k_dim) | K (k_dim) | V (v_dim)] in the `input`
// buffer, matching the wqkv projection output. Total n_channels =
// 2*k_dim + v_dim.
//
// Batched mode: `n_tokens` > 1 processes a sequence of N tokens through
// the conv in order. The state ring buffer is advanced N times inside
// the kernel (exactly the same state trajectory as calling the kernel
// N times sequentially). `input` is [N × n_channels], `q_out` is
// [N × k_dim], `k_out` is [N × k_dim], `v_out` is [N × v_dim], all
// row-major. For decode, `n_tokens = 1` gives byte-identical behaviour
// to the pre-batching version (the loop runs once).

extern "C" __global__ void conv1d_silu_split_f32(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_tokens
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    // Loop over tokens, performing exactly the same math and memory ops as
    // the single-token version did before the loop was added. Keeping the
    // weight[] indexing in-place (not hoisted) is LOAD-BEARING for
    // byte-exact output against the single-token baseline — hoisting the
    // loads changes the compiler's FMA fusion decisions and nudges the
    // numerics by ~1 ULP, which cascades over 2k+ greedy tokens.
    for (int t = 0; t < n_tokens; t++) {
        // Same math as conv1d_silu.hip — 4-tap causal depthwise conv + SiLU.
        float x = input[t * n_channels + c];
        float s0 = state[c * 3];
        float s1 = state[c * 3 + 1];
        float s2 = state[c * 3 + 2];

        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

        // SiLU
        float result = y / (1.0f + expf(-y));

        // Update ring buffer state
        state[c * 3 + 2] = s1;
        state[c * 3 + 1] = s0;
        state[c * 3]     = x;

        // Write to Q/K/V based on channel index.
        if (c < k_dim) {
            q_out[t * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[t * k_dim + (c - k_dim)] = result;
        } else {
            v_out[t * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}
