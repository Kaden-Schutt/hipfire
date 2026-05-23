// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Hyper-Connections 4-stream residual mix (DeepSeek V4 Phase 3).
//
// Each layer mixes the 4 residual streams via a Sinkhorn-normalised
// 4×4 gating matrix produced by `hc_sinkhorn_4x4`, then adds the
// layer transform's output scaled by per-stream scaling factors.
//
//   x_out[s, d] = sum_t(A[s, t] * x_in[t, d]) + scale[s] * transform_out[d]
//
// Signature (subject to revision):
//
//   x_in           : [hc_mult = 4, hidden = 4096]   fp16, in
//   A              : [4, 4]                          fp32, in (post-Sinkhorn)
//   scale          : [4]                             fp32, in
//   transform_out  : [hidden]                        fp16, in
//   x_out          : [hc_mult, hidden]               fp16, out
//
// Strategy: one block per `hidden / BLOCK_HIDDEN` chunk × per output
// stream. 4 streams × `hidden / BLOCK_HIDDEN` blocks total per layer.
// Each thread handles one hidden-dim element.
//
// Status: STUB. Single-threaded reference; production version
// vectorises hidden loads + uses smem for A and scale.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HC_MULT 4
#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_mix_4stream(
    const float* __restrict__ x_in,           // [HC_MULT, hidden] F32 (DeepSeek V4 convention)
    const float* __restrict__ A,               // [HC_MULT, HC_MULT]
    const float* __restrict__ scale,           // [HC_MULT]
    const float* __restrict__ transform_out,   // [hidden] F32
    float*       __restrict__ x_out,           // [HC_MULT, hidden] F32
    int hidden
) {
    const int stream_out = blockIdx.y;
    const int tid = threadIdx.x;
    const int d   = blockIdx.x * BLOCK_HIDDEN + tid;
    if (d >= hidden || stream_out >= HC_MULT) return;

    // Sum over input streams via gating matrix row A[stream_out, :].
    float acc = 0.0f;
    #pragma unroll
    for (int s_in = 0; s_in < HC_MULT; ++s_in) {
        float gate = A[stream_out * HC_MULT + s_in];
        float val  = x_in[s_in * hidden + d];
        acc += gate * val;
    }
    // Add transform's output scaled by this stream's scale factor.
    acc += scale[stream_out] * transform_out[d];

    x_out[stream_out * hidden + d] = acc;
}
