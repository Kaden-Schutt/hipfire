// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Hyper-Connections control-vector computation (DeepSeek V4 Phase 3).
//
// Upstream DeepSeek V4 (model.py:Block.hc_pre):
//   x = x.flatten(2).float()                                    # [hidden * hc_mult]
//   rsqrt = torch.rsqrt(x.square().mean(-1) + norm_eps)         # SCALAR
//   mixes = F.linear(x, hc_fn) * rsqrt                          # then + base in sinkhorn
//
// We fuse the (X·W) GEMV with the rsqrt + base-add into one launch:
//   1. Each block computes one output entry's dot product (X·W)_ctrl
//      AND accumulates sum-of-squares for the rsqrt in the same pass.
//   2. Reduce both, compute rsqrt = 1/sqrt(mean(x^2) + eps).
//   3. c[ctrl] = acc_dot * rsqrt + base[ctrl].
//
// DeepSeek V4 per-layer `hc_attn_fn` / `hc_ffn_fn` are [24, 16384] (n_ctrl=24,
// x_dim=hidden*hc_mult=16384). hc_base is [24] F16; output c is F32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Overridable so a wider-block variant can be built from the same source.
// The finalize kernel's critical path is ONE block's full x_dim reduction:
// blocks 1..n_ctrl-1 spin on block 0's rsqrt, and the last block to finish
// runs the sinkhorn epilogue. Fewer loop iterations per block therefore
// releases every spinner sooner, on top of the raw occupancy gain
// (24 blocks x 256t = 192 waves = 0.15 fills on gfx1151's 1280 slots).
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif
#ifndef HC_CTRL_FINALIZE_KERNEL
#define HC_CTRL_FINALIZE_KERNEL hc_compute_control_vec4_finalize
#endif

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control(
    const float*  __restrict__ x_flat,   // [x_dim] F32 (DeepSeek V4 residual convention)
    const __half* __restrict__ w_fn,     // [n_ctrl, x_dim] F16 (HFQ-converted)
    const __half* __restrict__ base,     // [n_ctrl] F16
    float*        __restrict__ c,         // [n_ctrl] F32
    int n_ctrl,
    int x_dim
) {
    const int ctrl = blockIdx.x;
    if (ctrl >= n_ctrl) return;
    const int tid = threadIdx.x;

    // Compute BOTH dot product AND sum-of-squares in one pass.
    float acc_dot = 0.0f;
    float acc_sq  = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        float xv = x_flat[d];
        float wv = __half2float(w_fn[ctrl * x_dim + d]);
        acc_dot += xv * wv;
        acc_sq  += xv * xv;
    }

    // Wave-shuffle reduction (wave32 on gfx11xx).
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_dot += __shfl_down(acc_dot, offset);
        acc_sq  += __shfl_down(acc_sq,  offset);
    }

    __shared__ float partial_dot[THREADS_PER_BLOCK / 32];
    __shared__ float partial_sq [THREADS_PER_BLOCK / 32];
    if (tid % 32 == 0) {
        partial_dot[tid / 32] = acc_dot;
        partial_sq [tid / 32] = acc_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float total_dot = 0.0f;
        float total_sq  = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK / 32; ++i) {
            total_dot += partial_dot[i];
            total_sq  += partial_sq [i];
        }
        // DeepSeek V4 upstream: mixes = F.linear(x, hc_fn) * rsqrt(mean(x^2) + eps)
        float rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
        c[ctrl] = total_dot * rsqrt + __half2float(base[ctrl]);
    }
}

// gfx1151 decode experiment: vectorize the inner dimension four ways and keep
// four independent dot/square accumulators.  The ordinary scalar loop carries
// two 64-iteration dependency chains per lane for DeepSeek V4's x_dim=16384.
// This layout trades some registers for 4x fewer loop iterations, vector
// memory operations, and deeper within-wave ILP while retaining all 24
// control-row blocks.
__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control_vec4(
    const float*  __restrict__ x_flat,
    const __half* __restrict__ w_fn,
    const __half* __restrict__ base,
    float*        __restrict__ c,
    int n_ctrl,
    int x_dim
) {
    const int ctrl = blockIdx.x;
    if (ctrl >= n_ctrl) return;
    const int tid = threadIdx.x;

    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    const __half* w_row = w_fn + (size_t)ctrl * x_dim;
    for (int d = tid * 4; d < x_dim; d += THREADS_PER_BLOCK * 4) {
        float4 xv = *reinterpret_cast<const float4*>(x_flat + d);
        float2 w01 = __half22float2(
            *reinterpret_cast<const __half2*>(w_row + d));
        float2 w23 = __half22float2(
            *reinterpret_cast<const __half2*>(w_row + d + 2));
        dot0 += xv.x * w01.x;
        dot1 += xv.y * w01.y;
        dot2 += xv.z * w23.x;
        dot3 += xv.w * w23.y;
        sq0 += xv.x * xv.x;
        sq1 += xv.y * xv.y;
        sq2 += xv.z * xv.z;
        sq3 += xv.w * xv.w;
    }

    float acc_dot = (dot0 + dot1) + (dot2 + dot3);
    float acc_sq = (sq0 + sq1) + (sq2 + sq3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_dot += __shfl_down(acc_dot, offset);
        acc_sq += __shfl_down(acc_sq, offset);
    }

    __shared__ float partial_dot[THREADS_PER_BLOCK / 32];
    __shared__ float partial_sq[THREADS_PER_BLOCK / 32];
    if (tid % 32 == 0) {
        partial_dot[tid / 32] = acc_dot;
        partial_sq[tid / 32] = acc_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float total_dot = 0.0f;
        float total_sq = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK / 32; ++i) {
            total_dot += partial_dot[i];
            total_sq += partial_sq[i];
        }
        float rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
        c[ctrl] = total_dot * rsqrt + __half2float(base[ctrl]);
    }
}

// gfx1151 retained-decode fusion: the 24 control rows already execute as
// independent workgroups. Each completed row publishes its result and takes a
// ticket; the last workgroup performs the existing control finalizer before
// returning. This removes the separate finalizer dispatch without serializing
// the bandwidth-heavy row projections.
//
// c must contain one extra F32 slot at c[n_ctrl], used as a u32 ticket. The
// last workgroup resets it to zero before the kernel completes, so the buffer
// is reusable by a retained tape.
__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void HC_CTRL_FINALIZE_KERNEL(
    const float*  __restrict__ x_flat,
    const __half* __restrict__ w_fn,
    const __half* __restrict__ base,
    float*        __restrict__ c,
    const __half* __restrict__ alpha,
    int n_ctrl,
    int x_dim,
    float hc_eps,
    float post_scale,
    int sinkhorn_iters,
    int rsqrt_once
) {
    const int ctrl = blockIdx.x;
    if (ctrl >= n_ctrl) return;
    const int tid = threadIdx.x;

    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    const __half* w_row = w_fn + (size_t)ctrl * x_dim;
    for (int d = tid * 4; d < x_dim; d += THREADS_PER_BLOCK * 4) {
        float4 xv = *reinterpret_cast<const float4*>(x_flat + d);
        float2 w01 = __half22float2(
            *reinterpret_cast<const __half2*>(w_row + d));
        float2 w23 = __half22float2(
            *reinterpret_cast<const __half2*>(w_row + d + 2));
        dot0 += xv.x * w01.x;
        dot1 += xv.y * w01.y;
        dot2 += xv.z * w23.x;
        dot3 += xv.w * w23.y;
        if (!rsqrt_once || ctrl == 0) {
            sq0 += xv.x * xv.x;
            sq1 += xv.y * xv.y;
            sq2 += xv.z * xv.z;
            sq3 += xv.w * xv.w;
        }
    }

    float acc_dot = (dot0 + dot1) + (dot2 + dot3);
    float acc_sq = (sq0 + sq1) + (sq2 + sq3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_dot += __shfl_down(acc_dot, offset);
        acc_sq += __shfl_down(acc_sq, offset);
    }

    __shared__ float partial_dot[THREADS_PER_BLOCK / 32];
    __shared__ float partial_sq[THREADS_PER_BLOCK / 32];
    __shared__ int last_block;
    if (tid % 32 == 0) {
        partial_dot[tid / 32] = acc_dot;
        partial_sq[tid / 32] = acc_sq;
    }
    __syncthreads();

    if (tid == 0) {
        float total_dot = 0.0f;
        float total_sq = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK / 32; ++i) {
            total_dot += partial_dot[i];
            total_sq += partial_sq[i];
        }
        float rsqrt;
        unsigned int* ticket;
        if (rsqrt_once) {
            float* rsqrt_slot = c + n_ctrl;
            unsigned int* ready =
                reinterpret_cast<unsigned int*>(c + n_ctrl + 1);
            ticket = reinterpret_cast<unsigned int*>(c + n_ctrl + 2);
            if (ctrl == 0) {
                rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
                *rsqrt_slot = rsqrt;
                __threadfence();
                atomicExch(ready, 1u);
            } else {
                while (atomicAdd(ready, 0u) == 0u) {
                }
                __threadfence();
                rsqrt = *rsqrt_slot;
            }
        } else {
            rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
            ticket = reinterpret_cast<unsigned int*>(c + n_ctrl);
        }
        c[ctrl] = total_dot * rsqrt + __half2float(base[ctrl]);

        // Publish c[ctrl] before advertising this row as complete. The last
        // ticket holder may then safely consume all 24 rows.
        __threadfence();
        unsigned int prior = atomicAdd(ticket, 1u);
        last_block = prior == (unsigned int)(n_ctrl - 1);
    }
    __syncthreads();
    if (!last_block) return;

    if (tid < 24) {
        int seg;
        if      (tid < 4) seg = 0;
        else if (tid < 8) seg = 1;
        else              seg = 2;
        float a = __half2float(alpha[seg]);
        float b = __half2float(base[tid]);
        c[tid] = a * c[tid] + (1.0f - a) * b;
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
    __syncthreads();
    if (tid == 0) {
        unsigned int* ticket = reinterpret_cast<unsigned int*>(
            c + n_ctrl + (rsqrt_once ? 2 : 0));
        if (rsqrt_once) {
            atomicExch(
                reinterpret_cast<unsigned int*>(c + n_ctrl + 1),
                0u);
        }
        atomicExch(ticket, 0u);
    }
}
