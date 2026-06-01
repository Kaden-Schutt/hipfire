// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Conservative gfx1151 bring-up kernels for qwen35 MoE batched prefill
// families that do not yet have MQ4-grade grouped/MMQ implementations.
// They mirror the indexed/scalar contracts of the MQ4/MQ6 sisters:
// - shared gate/up: [total_m, N] scalar GEMM
// - shared down:    [M, N] sigmoid-scaled residual GEMV
// - routed gate/up: [M, K_TOP, N] indexed batched GEMV
// - routed down:    [M, K_TOP, N] indexed batched GEMV into expanded scratch

__device__ __forceinline__ float dot_hfq2_g256(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int K,
    int tid
) {
    float acc = 0.0f;
    const int groups_per_row = K / 256;
    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + g * 72;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* data = (const unsigned char*)(gp + 8);
        const int base = g * 256 + tid * 8;
        const int byte_off = tid * 2;
        const unsigned char b0 = data[byte_off];
        const unsigned char b1 = data[byte_off + 1];
        acc += (scale * (float)(b0 & 3u) + zero) * x[base]
            +  (scale * (float)((b0 >> 2) & 3u) + zero) * x[base + 1]
            +  (scale * (float)((b0 >> 4) & 3u) + zero) * x[base + 2]
            +  (scale * (float)(b0 >> 6) + zero) * x[base + 3]
            +  (scale * (float)(b1 & 3u) + zero) * x[base + 4]
            +  (scale * (float)((b1 >> 2) & 3u) + zero) * x[base + 5]
            +  (scale * (float)((b1 >> 4) & 3u) + zero) * x[base + 6]
            +  (scale * (float)(b1 >> 6) + zero) * x[base + 7];
    }
    return acc;
}

__device__ __forceinline__ float dot_hfq8_g256(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int K,
    int tid
) {
    float acc = 0.0f;
    const int groups_per_row = K / 256;
    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + g * 264;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* data = (const unsigned char*)(gp + 8);
        const int base = g * 256 + tid * 8;
        const int byte_off = tid * 8;
        acc += (scale * (float)data[byte_off] + zero) * x[base]
            +  (scale * (float)data[byte_off + 1] + zero) * x[base + 1]
            +  (scale * (float)data[byte_off + 2] + zero) * x[base + 2]
            +  (scale * (float)data[byte_off + 3] + zero) * x[base + 3]
            +  (scale * (float)data[byte_off + 4] + zero) * x[base + 4]
            +  (scale * (float)data[byte_off + 5] + zero) * x[base + 5]
            +  (scale * (float)data[byte_off + 6] + zero) * x[base + 6]
            +  (scale * (float)data[byte_off + 7] + zero) * x[base + 7];
    }
    return acc;
}

__device__ __forceinline__ float dot_mq2_lloyd_g256(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int K,
    int tid
) {
    float acc = 0.0f;
    const int groups_per_row = K / 256;
    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + g * 72;
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        const float cb0 = __half2float(cb_h[0]);
        const float cb1 = __half2float(cb_h[1]);
        const float cb2 = __half2float(cb_h[2]);
        const float cb3 = __half2float(cb_h[3]);
        const unsigned char* data = (const unsigned char*)(gp + 8);
        const int base = g * 256 + tid * 8;
        const int byte_off = tid * 2;
        const unsigned char b0 = data[byte_off];
        const unsigned char b1 = data[byte_off + 1];
        const unsigned int q0 = b0 & 3u;
        const unsigned int q1 = (b0 >> 2) & 3u;
        const unsigned int q2 = (b0 >> 4) & 3u;
        const unsigned int q3 = b0 >> 6;
        const unsigned int q4 = b1 & 3u;
        const unsigned int q5 = (b1 >> 2) & 3u;
        const unsigned int q6 = (b1 >> 4) & 3u;
        const unsigned int q7 = b1 >> 6;
        #define CB(q) ((q) == 0u ? cb0 : ((q) == 1u ? cb1 : ((q) == 2u ? cb2 : cb3)))
        acc += CB(q0) * x[base] + CB(q1) * x[base + 1]
            +  CB(q2) * x[base + 2] + CB(q3) * x[base + 3]
            +  CB(q4) * x[base + 4] + CB(q5) * x[base + 5]
            +  CB(q6) * x[base + 6] + CB(q7) * x[base + 7];
        #undef CB
    }
    return acc;
}

__device__ __forceinline__ float dot_mq3_lloyd_g256(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int K,
    int tid
) {
    float acc = 0.0f;
    const int groups_per_row = K / 256;
    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + g * 112;
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        float cb[8] = {
            __half2float(cb_h[0]), __half2float(cb_h[1]),
            __half2float(cb_h[2]), __half2float(cb_h[3]),
            __half2float(cb_h[4]), __half2float(cb_h[5]),
            __half2float(cb_h[6]), __half2float(cb_h[7]),
        };
        const unsigned char* data = (const unsigned char*)(gp + 16);
        const int base = g * 256 + tid * 8;
        const int byte_off = tid * 3;
        const unsigned int b0 = data[byte_off];
        const unsigned int b1 = data[byte_off + 1];
        const unsigned int b2 = data[byte_off + 2];
        acc += cb[b0 & 7u] * x[base]
            +  cb[(b0 >> 3) & 7u] * x[base + 1]
            +  cb[((b0 >> 6) | (b1 << 2)) & 7u] * x[base + 2]
            +  cb[(b1 >> 1) & 7u] * x[base + 3]
            +  cb[(b1 >> 4) & 7u] * x[base + 4]
            +  cb[((b1 >> 7) | (b2 << 1)) & 7u] * x[base + 5]
            +  cb[(b2 >> 2) & 7u] * x[base + 6]
            +  cb[(b2 >> 5) & 7u] * x[base + 7];
    }
    return acc;
}

#define REDUCE_ACC(acc) \
    for (int offset = 16; offset > 0; offset >>= 1) { \
        (acc) += __shfl_down((acc), offset); \
    }

#define DEFINE_MOE_FAMILY(TAG, DOTFN, STRIDE) \
extern "C" __global__ void gemm_gate_up_##TAG##_scalar_batched( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, \
    const float* __restrict__ X, float* __restrict__ Y_gate, float* __restrict__ Y_up, \
    int gate_m, int up_m, int K, int N) { \
    const int total_m = gate_m + up_m; \
    const int row = blockIdx.x; \
    const int bid = blockIdx.y; \
    const int tid = threadIdx.x; \
    if (row >= total_m || bid >= N) return; \
    const bool is_gate = row < gate_m; \
    const int local_row = is_gate ? row : row - gate_m; \
    const char* A = is_gate ? A_gate : A_up; \
    const char* row_ptr = A + (long long)local_row * (K / 256) * (STRIDE); \
    const float* x = X + (long long)bid * K; \
    float acc = DOTFN(row_ptr, x, K, tid); \
    REDUCE_ACC(acc); \
    if (tid == 0) { \
        if (is_gate) Y_gate[(long long)bid * gate_m + local_row] = acc; \
        else Y_up[(long long)bid * up_m + local_row] = acc; \
    } \
} \
extern "C" __global__ void gemv_##TAG##_residual_sigmoid_scaled_gpu_batched( \
    const char* __restrict__ A, const float* __restrict__ X, float* __restrict__ Y, \
    const float* __restrict__ C, int M, int K) { \
    const int row = blockIdx.x; \
    const int bid = blockIdx.y; \
    const int tid = threadIdx.x; \
    if (row >= M) return; \
    const char* row_ptr = A + (long long)row * (K / 256) * (STRIDE); \
    const float* x = X + (long long)bid * K; \
    float acc = DOTFN(row_ptr, x, K, tid); \
    REDUCE_ACC(acc); \
    if (tid == 0) { \
        const float gate = 1.0f / (1.0f + __expf(-C[bid])); \
        Y[(long long)bid * M + row] += gate * acc; \
    } \
} \
extern "C" __global__ void gemv_##TAG##_moe_gate_up_k8_indexed_batched( \
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices, \
    const float* __restrict__ X, float* __restrict__ Y_gate, float* __restrict__ Y_up, \
    int M, int K, int K_TOP) { \
    const int row = blockIdx.x; \
    const int krank = blockIdx.y; \
    const int bid = blockIdx.z; \
    const int tid = threadIdx.x; \
    if (row >= M) return; \
    const int expert_id = topk_indices[bid * K_TOP + krank]; \
    const char* A = reinterpret_cast<const char*>(expert_ptrs[expert_id]); \
    const char* row_ptr = A + (long long)row * (K / 256) * (STRIDE); \
    const float* x = X + (long long)bid * K; \
    float acc = DOTFN(row_ptr, x, K, tid); \
    REDUCE_ACC(acc); \
    if (tid == 0) { \
        const int mi = M >> 1; \
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi; \
        if (row < mi) Y_gate[base_out + row] = acc; \
        else Y_up[base_out + (row - mi)] = acc; \
    } \
} \
extern "C" __global__ void gemv_##TAG##_moe_down_k8_indexed_batched_expanded( \
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices, \
    const float* __restrict__ X, float* __restrict__ expert_outputs, \
    int M, int K, int K_TOP) { \
    const int row = blockIdx.x; \
    const int krank = blockIdx.y; \
    const int bid = blockIdx.z; \
    const int tid = threadIdx.x; \
    if (row >= M) return; \
    const int expert_id = topk_indices[bid * K_TOP + krank]; \
    const char* A = reinterpret_cast<const char*>(expert_ptrs[expert_id]); \
    const char* row_ptr = A + (long long)row * (K / 256) * (STRIDE); \
    const float* x = X + ((long long)bid * K_TOP + krank) * K; \
    float acc = DOTFN(row_ptr, x, K, tid); \
    REDUCE_ACC(acc); \
    if (tid == 0) expert_outputs[((long long)bid * K_TOP + krank) * M + row] = acc; \
}

DEFINE_MOE_FAMILY(hfq2g256, dot_hfq2_g256, 72)
DEFINE_MOE_FAMILY(hfq8g256, dot_hfq8_g256, 264)
DEFINE_MOE_FAMILY(mq2g256_lloyd, dot_mq2_lloyd_g256, 72)
DEFINE_MOE_FAMILY(mq3g256_lloyd, dot_mq3_lloyd_g256, 112)

