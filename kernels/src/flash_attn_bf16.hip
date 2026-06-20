// SPDX-License-Identifier: Apache-2.0
// hipfire — generic bidirectional flash attention, bf16 inputs. See LICENSE / NOTICE.

#include <hip/hip_runtime.h>

// Plain (non-causal) flash attention for a ViT-style vision encoder.
//
//   out[qi, head] = softmax( Q[qi]·Kᵀ / sqrt(head_dim) ) · V
//
// Fused interleaved qkv [N, 3*hidden] in bf16 (stored as u16 = top 16 bits of
// the f32); output [N, hidden] in f32. Online softmax with f32 accumulation —
// no full scores[N] materialisation (that was vit_attention_opt's ceiling), no
// causal mask (vision attention is bidirectional).
//
// "Generic first": one block per (head, query), a block-wide reduction per key.
// Correct and portable across RDNA archs; the WMMA-tiled fast path is a
// follow-up. Grid: [num_heads, N]. Block: [BLK] (>= head_dim). LDS: head_dim +
// BLK floats.

#define BLK 128

__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    union { unsigned int u; float f; } box;
    box.u = ((unsigned int)b) << 16;
    return box.f;
}

extern "C" __global__ void flash_attn_bf16(
    const unsigned short* __restrict__ qkv,  // [N, 3*hidden] bf16 (u16)
    float* __restrict__ out,                  // [N, hidden] f32
    int N, int hidden, int num_heads, int head_dim, float scale
) {
    const int head = blockIdx.x;
    const int qi = blockIdx.y;
    const int tid = threadIdx.x;
    if (head >= num_heads || qi >= N) return;

    const int stride = 3 * hidden;
    const int q_off = head * head_dim;
    const int k_off = hidden + head * head_dim;
    const int v_off = 2 * hidden + head * head_dim;

    extern __shared__ float smem[];
    float* q_sh = smem;            // [head_dim]
    float* red = smem + head_dim;  // [BLK] reduction workspace

    // Load Q (bf16 -> f32) into shared memory, visible to all threads.
    for (int d = tid; d < head_dim; d += BLK) {
        q_sh[d] = bf16_to_f32(qkv[(size_t)qi * stride + q_off + d]);
    }
    __syncthreads();

    // Online-softmax running state. Thread `tid` owns output dim `tid`
    // (when tid < head_dim) and its V accumulator.
    float acc = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    for (int j = 0; j < N; j++) {
        // Block-reduce the dot product Q·K[j].
        float partial = 0.0f;
        if (tid < head_dim) {
            partial = q_sh[tid] * bf16_to_f32(qkv[(size_t)j * stride + k_off + tid]);
        }
        red[tid] = partial;
        __syncthreads();
        for (int s = BLK / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();

        // Online softmax update (numerically stable).
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;
        if (tid < head_dim) {
            float vval = bf16_to_f32(qkv[(size_t)j * stride + v_off + tid]);
            acc = acc * corr + p * vval;
        }
        m = m_new;
    }

    if (tid < head_dim) {
        out[(size_t)qi * hidden + head * head_dim + tid] = acc / l;
    }
}
