// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Gemma 4 E-series PLE activation for batched prefill:
//
//   hidden[row, col] = gelu_tanh(gate[row, col])
//                    * projection_all[row, layer, col]
//
// projection_all is [batch, n_layers * ple_dim]. Using grid.y for the row
// avoids integer division in the hot elementwise path.
extern "C" __global__ void gemma4_ple_gelu_mul_strided_f32(
    const float* __restrict__ gate,
    const float* __restrict__ projection_all,
    float* __restrict__ hidden,
    int ple_dim,
    int packed_dim,
    int layer_offset,
    int batch
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || col >= ple_dim) return;

    int dense_idx = row * ple_dim + col;
    int packed_idx = row * packed_dim + layer_offset + col;
    float v = gate[dense_idx];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    float activated = 0.5f * v * (1.0f + tanhf(inner));
    hidden[dense_idx] = activated * projection_all[packed_idx];
}
