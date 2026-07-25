// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// DeepSeek V4 mHC control finalizer.  This preserves the operation order of:
//   hc_apply_alpha
//   hc_pre_post_sigmoid_scale_f32
//   hc_sinkhorn_4x4
// while keeping the 24-element control vector resident across one wave and
// replacing three dispatches with one.
extern "C" __global__ __launch_bounds__(32, 1)
void hc_finalize_control(
    float*        __restrict__ c,
    const __half* __restrict__ alpha,
    const __half* __restrict__ base,
    float hc_eps,
    float post_scale,
    int sinkhorn_iters
) {
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;

    if (tid < 24) {
        int seg;
        if      (tid < 4) seg = 0;
        else if (tid < 8) seg = 1;
        else              seg = 2;
        float a = __half2float(alpha[seg]);
        float b = __half2float(base[tid]);
        float ci = c[tid];
        c[tid] = a * ci + (1.0f - a) * b;
    }
    __syncthreads();

    if (tid < 4) {
        float z = c[tid];
        c[tid] = 1.0f / (1.0f + __expf(-z)) + hc_eps;
    } else if (tid < 8) {
        float z = c[tid];
        c[tid] = post_scale * (1.0f / (1.0f + __expf(-z)));
    }
    __syncthreads();

    if (tid < 16) {
        float v = c[8 + tid];

        float rm = v;
        rm = fmaxf(rm, __shfl_xor(rm, 1));
        rm = fmaxf(rm, __shfl_xor(rm, 2));
        v = __expf(v - rm);
        float rs = v;
        rs += __shfl_xor(rs, 1);
        rs += __shfl_xor(rs, 2);
        v = v / rs + hc_eps;

        float cs = v;
        cs += __shfl_xor(cs, 4);
        cs += __shfl_xor(cs, 8);
        cs += hc_eps;
        v = v / cs;

        for (int it = 0; it < sinkhorn_iters - 1; ++it) {
            rs = v;
            rs += __shfl_xor(rs, 1);
            rs += __shfl_xor(rs, 2);
            rs += hc_eps;
            v = v / rs;

            cs = v;
            cs += __shfl_xor(cs, 4);
            cs += __shfl_xor(cs, 8);
            cs += hc_eps;
            v = v / cs;
        }
        c[8 + tid] = v;
    }
}
