// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Atomic-free batched indexed MoE down GEMV for QTIP3 — QTIP sibling of
// gemv_oq4g256_moe_down_k8_indexed_batched_expanded. Writes each (token, krank)
// result to expert_outputs[N × K_TOP × M]; reuse moe_down_combine_k8_batched to
// fold K_TOP outputs into the residual. 100 B/group [f32 scale | 96 B 3-bit
// trellis], computed 1MAD codebook. Grid (M, K_TOP, N), wave32, no atomics.

#define QTIP_MEAN     (-0.0071028476f)
#define QTIP_INV_STD  (0.9959345134f)
#define QTIP_STATE_MASK 0xFFFu

__device__ __forceinline__ float qtip_decode(unsigned int state) {
    unsigned int x = state & 0xFFFFFFFFu;
    x = x * 34038481u + 76625530u;
    unsigned int bs = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    float raw = ((float)bs - 510.0f) * (1.0f / 147.800537109375f);
    return (raw - QTIP_MEAN) * QTIP_INV_STD;
}

__device__ __forceinline__ float qtip_group_dot(
    const char* gptr, const float* x, int base_idx, int tid
) {
    const float scale = *reinterpret_cast<const float*>(gptr);
    const unsigned char* data = (const unsigned char*)(gptr + 4);
    const int boff = tid * 3;
    unsigned int pk = (unsigned int)data[boff]
                    | ((unsigned int)data[boff + 1] << 8)
                    | ((unsigned int)data[boff + 2] << 16);
    unsigned int s_m3 = 0u, s_m2 = 0u, s_m1 = 0u;
    if (tid > 0) {
        const int pboff = (tid - 1) * 3;
        unsigned int ppk = (unsigned int)data[pboff]
                         | ((unsigned int)data[pboff + 1] << 8)
                         | ((unsigned int)data[pboff + 2] << 16);
        s_m3 = (ppk >> 15) & 7u;
        s_m2 = (ppk >> 18) & 7u;
        s_m1 = (ppk >> 21) & 7u;
    }
    unsigned int q0 = pk & 7u, q1 = (pk >> 3) & 7u, q2 = (pk >> 6) & 7u, q3 = (pk >> 9) & 7u;
    unsigned int q4 = (pk >> 12) & 7u, q5 = (pk >> 15) & 7u, q6 = (pk >> 18) & 7u, q7 = (pk >> 21) & 7u;
    #define ST(a,b,c,d) ((((a)<<9)|((b)<<6)|((c)<<3)|(d)) & QTIP_STATE_MASK)
    float s = scale;
    return s * qtip_decode(ST(s_m3,s_m2,s_m1,q0)) * x[base_idx]
         + s * qtip_decode(ST(s_m2,s_m1,q0,q1))   * x[base_idx + 1]
         + s * qtip_decode(ST(s_m1,q0,q1,q2))      * x[base_idx + 2]
         + s * qtip_decode(ST(q0,q1,q2,q3))        * x[base_idx + 3]
         + s * qtip_decode(ST(q1,q2,q3,q4))        * x[base_idx + 4]
         + s * qtip_decode(ST(q2,q3,q4,q5))        * x[base_idx + 5]
         + s * qtip_decode(ST(q3,q4,q5,q6))        * x[base_idx + 6]
         + s * qtip_decode(ST(q4,q5,q6,q7))        * x[base_idx + 7];
    #undef ST
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_qtip3g256_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + ((long long)bid * K_TOP + krank) * K;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 100;

    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; g++) {
        acc += qtip_group_dot(row_ptr + g * 100, x, g * 256 + tid * 8, tid);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const long long out_offset = ((long long)bid * K_TOP + krank) * (long long)M + row;
        expert_outputs[out_offset] = acc;
    }
}
