// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — arithmetic-identical gfx1201 batched HC control projection.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define HC_CTRL 24
#define HC_THREADS 256
#define HC_WAVES (HC_THREADS / 32)

// One workgroup owns one token row and all 24 control rows. The established
// kernel launches 24 workgroups per token; each reloads X and recomputes its
// norm. This geometry retains the established scalar accumulation, shuffle,
// and eight-wave LDS reduction order for every control output while loading X
// once and evaluating the shared norm once.
extern "C" __global__ __launch_bounds__(HC_THREADS)
void hc_compute_control_batched_fused24_gfx1201(
    const float* __restrict__ x_flat,
    const _Float16* __restrict__ w_fn,
    const _Float16* __restrict__ base,
    float* __restrict__ c,
    int n_ctrl,
    int x_dim,
    int batch_size
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size || n_ctrl != HC_CTRL) return;

    const float* x_row = x_flat + (size_t)b * x_dim;
    float dot[HC_CTRL];
    #pragma unroll
    for (int ctrl = 0; ctrl < HC_CTRL; ++ctrl) dot[ctrl] = 0.0f;
    float sq = 0.0f;

    for (int d = tid; d < x_dim; d += HC_THREADS) {
        const float xv = x_row[d];
        #pragma unroll
        for (int ctrl = 0; ctrl < HC_CTRL; ++ctrl)
            dot[ctrl] += xv * (float)w_fn[(size_t)ctrl * x_dim + d];
        sq += xv * xv;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int ctrl = 0; ctrl < HC_CTRL; ++ctrl)
            dot[ctrl] += __shfl_down(dot[ctrl], offset);
        sq += __shfl_down(sq, offset);
    }

    __shared__ float partial_dot[HC_CTRL][HC_WAVES];
    __shared__ float partial_sq[HC_WAVES];
    __shared__ float inv_rms;
    if ((tid & 31) == 0) {
        const int wave = tid >> 5;
        #pragma unroll
        for (int ctrl = 0; ctrl < HC_CTRL; ++ctrl)
            partial_dot[ctrl][wave] = dot[ctrl];
        partial_sq[wave] = sq;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int wave = 0; wave < HC_WAVES; ++wave)
            total_sq += partial_sq[wave];
        inv_rms = rsqrtf(total_sq / (float)x_dim + 1.0e-6f);
    }
    float total_dot = 0.0f;
    if (tid < HC_CTRL) {
        #pragma unroll
        for (int wave = 0; wave < HC_WAVES; ++wave)
            total_dot += partial_dot[tid][wave];
    }
    // The norm publication is independent of the dot reduction, but both
    // must complete before the established final multiply/add.
    __syncthreads();
    if (tid < HC_CTRL) {
        c[(size_t)b * HC_CTRL + tid] =
            total_dot * inv_rms + (float)base[tid];
    }
}
