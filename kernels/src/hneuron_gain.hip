// SPDX-License-Identifier: Apache-2.0
// hipfire — H-Neuron intervention gain kernel. See LICENSE / NOTICE.
//
// Per-neuron activation gain for the H-Neurons intervention sweep. Scales each
// FFN neuron's activation in place by this layer's per-neuron gain:
//
//   ffn[t, j] *= mask[j]   for t in [0, positions), j in [0, inter)
//
// `mask[j]` holds the gain for neuron j (1.0 for non-H-Neurons; the swept scalar
// for H-Neurons). Applied to the down_proj INPUT before down_proj, on every
// position (prefill + decode). Register-only, no LDS (gfx1103 fault class).

#include <hip/hip_runtime.h>

extern "C" __global__ void hneuron_gain(float* __restrict__ ffn,       // [positions*inter] in place
                                        const float* __restrict__ mask, // [inter] this layer's gain
                                        int positions, int inter) {
    long total = (long)positions * (long)inter;
    long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (idx >= total) {
        return;
    }
    int j = (int)(idx % (long)inter);
    ffn[idx] *= mask[j];
}
