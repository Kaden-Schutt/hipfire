// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// H-Neurons CETT GPU-side reduction (arXiv 2512.01797). Replaces the per-layer
// host download of the FFN activations + output, which forced ~2 device syncs
// per layer and serialized the capture forward. Both kernels are register-tiled
// with NO LDS and no cross-thread reductions — deliberately avoiding the
// gfx1103 LDS/GDS-VM fault class (see docs/plans/gfx1103-lds-hip719-*).

// Per-token L2 norm of the down_proj output: out_norm[t] = ||down_out[t,:]||.
// One thread per token (positions are few, ~tens); each thread sums squares
// over `hidden` serially in a register. No reduction, no LDS.
extern "C" __global__ void cett_out_norm(const float *__restrict__ down_out, // [positions * hidden]
                                         float *__restrict__ out_norm,        // [positions]
                                         int positions, int hidden) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= positions)
        return;
    const float *row = down_out + (size_t)t * (size_t)hidden;
    float sum = 0.0f;
    for (int h = 0; h < hidden; h++) {
        float v = row[h];
        sum += v * v;
    }
    out_norm[t] = sqrtf(sum);
}

// Accumulate per-neuron CETT summed over the response tokens into `sums`:
//   sums[j] += sum_{t in [resp_start, positions)} |act[t,j]| * col_norm[j]
//              / (out_norm[t] + 1e-8)
// One thread per neuron j; it accumulates over the (few) response tokens in a
// register and does a single read-modify-write into sums[j]. Each j is owned by
// exactly one thread, so no atomics are needed. No LDS.
extern "C" __global__ void cett_accumulate(const float *__restrict__ act,      // [positions * intermediate]
                                           const float *__restrict__ col_norm, // [intermediate]
                                           const float *__restrict__ out_norm, // [positions]
                                           float *__restrict__ sums,           // [intermediate]
                                           int positions, int intermediate, int resp_start) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= intermediate)
        return;
    float cn = col_norm[j];
    float acc = 0.0f;
    for (int t = resp_start; t < positions; t++) {
        float a = act[(size_t)t * (size_t)intermediate + j];
        float inv = 1.0f / (out_norm[t] + 1e-8f);
        acc += fabsf(a) * cn * inv;
    }
    sums[j] += acc;
}
