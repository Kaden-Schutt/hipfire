// SPDX-License-Identifier: Apache-2.0
// hipfire — RoughQuant sparse-correction gather/scatter.
//
// Support kernels for the roughquant real packed format's protected-channel
// correction (y += R_S · x_S). The correction is a small dense GEMV over the |S|
// protected channels; these kernels move data into/out of that contiguous |S|-wide
// space:
//   rq_gather_f32     — xs[j] = x[idx[j]]            (reader: gather x[S], pad with 0)
//   rq_scatter_add_f32 — y[idx[j]] += c[j]           (writer: scatter c into y[S])

#include <hip/hip_runtime.h>

// dst[j] = (j < n_idx) ? src[idx[j]] : 0   for j < n_out  (n_out ≥ n_idx, padded).
extern "C" __global__ void rq_gather_f32(
    const float* __restrict__ src,
    const int* __restrict__ idx,
    float* __restrict__ dst,
    int n_idx, int n_out
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    dst[j] = (j < n_idx) ? src[idx[j]] : 0.0f;
}

// y[idx[j]] += c[j]  for j < n_idx.
extern "C" __global__ void rq_scatter_add_f32(
    float* __restrict__ y,
    const int* __restrict__ idx,
    const float* __restrict__ c,
    int n_idx
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_idx) return;
    y[idx[j]] += c[j];
}
