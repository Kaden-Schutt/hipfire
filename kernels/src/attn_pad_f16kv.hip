// SPDX-License-Identifier: Apache-2.0
// hipfire — split/pad fused qkv for the f16-KV WMMA flash attention. See LICENSE / NOTICE.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Split a fused f32 qkv [N, 3*hidden] (hidden = num_heads*head_dim) into the
// separate, head-dim-padded q/k/v the f16-KV WMMA flash kernels want:
//   q_pad : f32   [N, num_heads, hdp]
//   k_pad : f16   [N, num_heads, hdp]
//   v_pad : f16   [N, num_heads, hdp]
// Dims [head_dim, hdp) are zero-filled (zeros contribute nothing to QKᵀ / PV).
// One thread per (token, head, padded-dim).
extern "C" __global__ void attn_split_pad_f16kv(
    const float* __restrict__ qkv,    // [N, 3*hidden]
    float* __restrict__ q_pad,        // [N*num_heads*hdp] f32
    __half* __restrict__ k_pad,       // [N*num_heads*hdp] f16
    __half* __restrict__ v_pad,       // [N*num_heads*hdp] f16
    int N, int hidden, int num_heads, int head_dim, int hdp,
    // Pre-scale applied to Q only. The downstream WMMA flash bakes a fixed
    // 1/sqrt(hdp) softmax scale; when the real head_dim != hdp (e.g. SigLIP's
    // 72 padded to 128), pass q_scale = sqrt(hdp/head_dim) so the effective
    // scale becomes the correct 1/sqrt(head_dim). Pass 1.0 when head_dim==hdp.
    float q_scale
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * num_heads * hdp;
    if (idx >= total) return;

    int d = (int)(idx % hdp);
    long t2 = idx / hdp;
    int head = (int)(t2 % num_heads);
    int t = (int)(t2 / num_heads);

    if (d < head_dim) {
        long stride = (long)3 * hidden;
        long base = (long)t * stride + (long)head * head_dim + d;
        float qv = qkv[base];
        float kv = qkv[base + hidden];
        float vv = qkv[base + 2 * hidden];
        q_pad[idx] = qv * q_scale;
        k_pad[idx] = __float2half(kv);
        v_pad[idx] = __float2half(vv);
    } else {
        q_pad[idx] = 0.0f;
        k_pad[idx] = __float2half(0.0f);
        v_pad[idx] = __float2half(0.0f);
    }
}

// Inverse of the padding on the attention output: [N, num_heads, hdp] f32 →
// the contiguous [N, num_heads, head_dim] = [N, hidden] the out-projection
// expects. One thread per (token, head, dim).
extern "C" __global__ void attn_unpad(
    const float* __restrict__ in,    // [N, num_heads, hdp]
    float* __restrict__ out,         // [N, num_heads, head_dim]
    int N, int num_heads, int head_dim, int hdp
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * num_heads * head_dim;
    if (idx >= total) return;

    int d = (int)(idx % head_dim);
    long t2 = idx / head_dim;
    int head = (int)(t2 % num_heads);
    int t = (int)(t2 / num_heads);
    out[idx] = in[((long)t * num_heads + head) * hdp + d];
}
