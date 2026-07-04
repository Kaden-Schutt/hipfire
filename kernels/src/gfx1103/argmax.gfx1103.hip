// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1103 (Phoenix) wave-reduced argmax family: plain, batched, and token-chain.
//
// Same ABI and math contract as kernels/src/argmax.hip,
// argmax_batched.hip, and argmax_token_chain.hip. The generic kernels run a
// blockDim.x LDS tournament over (value, index) pairs. gfx1103 is wave32: each
// wave reduces the pair with shuffles and only 2 words per wave touch LDS.
// Keeps the Phoenix UMA path off heavy LDS.
//
// Tie-break: on equal values the LOWEST index wins, matching the strict `>`
// tournament of the generic kernels (which keeps the earlier candidate).

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#if HIPFIRE_GFX1103
__device__ __forceinline__ void hipfire_wave_argmax32(float& v, int& idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_down(v, offset, 32);
        const int oi = __shfl_down(idx, offset, 32);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
}

// Block-wide argmax over up to 32 wave32 waves. Returns the winning index;
// only thread 0's return value is meaningful (it holds the block reduction).
// `scratch` needs 2*num_waves words: [0,num_waves) floats then num_waves ints.
__device__ __forceinline__ int hipfire_block_argmax(
    float v, int idx, float* scratch, int num_waves, int tid
) {
    const int lane = tid & 31;
    const int wave = tid >> 5;
    int* sidx = (int*)(scratch + num_waves);
    hipfire_wave_argmax32(v, idx);
    if (lane == 0) {
        scratch[wave] = v;
        sidx[wave] = idx;
    }
    __syncthreads();
    if (wave == 0) {
        float wv = (lane < num_waves) ? scratch[lane] : -1e30f;
        int wi = (lane < num_waves) ? sidx[lane] : 0x7fffffff;
        hipfire_wave_argmax32(wv, wi);
        if (lane == 0) {
            scratch[0] = wv;
            sidx[0] = wi;
        }
    }
    __syncthreads();
    return ((int*)(scratch + num_waves))[0];
}

__device__ __forceinline__ void hipfire_local_argmax(
    const float* __restrict__ row, int n, int tid, int stride, float& lmax, int& lidx
) {
    lmax = -1e30f;
    lidx = 0;
    for (int i = tid; i < n; i += stride) {
        const float v = row[i];
        if (v > lmax) {
            lmax = v;
            lidx = i;
        }
    }
}
#endif

extern "C" __launch_bounds__(256, 1)
__global__ void argmax_f32_gfx1103(
    const float* __restrict__ data, int* __restrict__ result, int n
) {
#if HIPFIRE_GFX1103
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;
    const int num_waves = (blockDim.x + 31) >> 5;
    float lmax;
    int lidx;
    hipfire_local_argmax(data, n, tid, blockDim.x, lmax, lidx);
    const int idx = hipfire_block_argmax(lmax, lidx, scratch, num_waves, tid);
    if (tid == 0) result[0] = idx;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) result[0] = (int)data[0];
#endif
}

extern "C" __launch_bounds__(256, 1)
__global__ void argmax_f32_batched_gfx1103(
    const float* __restrict__ data,  // [batch, n]
    int* __restrict__ result,         // [batch]
    int n
) {
#if HIPFIRE_GFX1103
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;
    const int num_waves = (blockDim.x + 31) >> 5;
    const float* row = data + (size_t)blockIdx.x * n;
    float lmax;
    int lidx;
    hipfire_local_argmax(row, n, tid, blockDim.x, lmax, lidx);
    const int idx = hipfire_block_argmax(lmax, lidx, scratch, num_waves, tid);
    if (tid == 0) result[blockIdx.x] = idx;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) result[0] = (int)data[0];
#endif
}

extern "C" __launch_bounds__(256, 1)
__global__ void argmax_token_chain_f32_gfx1103(
    const float* __restrict__ data,
    int* __restrict__ argmax_out,
    int* __restrict__ token_chain,
    const unsigned int* __restrict__ vocab_map,
    int n,
    int dst_slot,
    int use_vocab_map
) {
#if HIPFIRE_GFX1103
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;
    const int num_waves = (blockDim.x + 31) >> 5;
    float lmax;
    int lidx;
    hipfire_local_argmax(data, n, tid, blockDim.x, lmax, lidx);
    const int idx = hipfire_block_argmax(lmax, lidx, scratch, num_waves, tid);
    if (tid == 0) {
        argmax_out[0] = idx;
        token_chain[dst_slot] = use_vocab_map ? (int)vocab_map[idx] : idx;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) {
        argmax_out[0] = (int)data[0] + dst_slot + use_vocab_map + (int)token_chain[0]
            + (int)(vocab_map ? vocab_map[0] : 0u);
    }
#endif
}
