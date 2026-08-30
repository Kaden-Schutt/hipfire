// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.

// Candidate K-axis variants of the MQ4V2 MMQ residual GEMM.
// New TU — no production dispatch touches this file. Mirrors
// kernels/src/gemm_mq4g256v2_residual_mmq.hip (128x128 MMQ tile,
// 8 warps / 256 threads, LDS staging, i8 WMMA, pre-quantized Q8_1).
//
// Variants:
//   - _k2 : K-unroll x2 inside vec_dot (2 x QI8_1 per inner iter)
//   - _k4 : K-unroll x4 (all 4 QI8_1 slices of the 32-K half-block at once)
//   - _ksplit : K-split x4 (grid.z = 4, atomicAdd accumulation)
//
// All kernels keep the base 7-arg signature:
//   (const char* A, const block_q8_1_mmq* Xq, float* Y, int M, int K, int N, int add)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define HIPFIRE_RDNA3 1
#else
#define HIPFIRE_RDNA3 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32
#define QK8_1 32
#define QI8_1 8
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K 36
#define MMQ_TILE_X_K 76
#define K_SPLITS 4

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void quantize_q8_1_mmq_ds4(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
);

#if !HIPFIRE_RDNA3

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_k2(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_k2_full_add(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_k2_full_set(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_k4(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_k4_full_add(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_k4_full_set(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_ksplit(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_ksplit_full_add(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_ksplit_full_set(const char*, const block_q8_1_mmq*, float*, int,int,int,int) {}

#else

// ---------------------------------------------------------------------------
// Shared helpers — verbatim copies from the production MMQ TU so candidate
// correctness is isolated (no header dependency).
// ---------------------------------------------------------------------------

template <int nbytes>
static __device__ __forceinline__ void memcpy_aligned(void* __restrict__ dst, const void* __restrict__ src) {
    if constexpr (nbytes == 16) {
        ((int4*)dst)[0] = ((const int4*)src)[0];
    } else if constexpr (nbytes == 8) {
        ((int2*)dst)[0] = ((const int2*)src)[0];
    } else if constexpr (nbytes == 4) {
        ((int*)dst)[0] = ((const int*)src)[0];
    }
}

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
};

template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int) { return threadIdx.x % 16; }
    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 16) { return 2 * l + (threadIdx.x / 16); }
        else { return l; }
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int l) { return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l); }
    static __device__ __forceinline__ int get_j(const int l) { return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l); }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32 * 2;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int) { return threadIdx.x % 16; }
    static __device__ __forceinline__ int get_j(const int l) { return l; }
};

template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_16x8(tile<16, 8, T, dl>& t, const T* __restrict__ xs0, const int stride) {
    static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad layout");
    static_assert(sizeof(t.x) == 32, "bad mirrored tile size");
    memcpy_aligned<16>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
    memcpy_aligned<16>(t.x + 4, xs0 + t.get_i(0) * stride + 4);
}

template <data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma_i8(
    tile<16, 16, int, dl_d>& D,
    const tile<16, 8, int, dl_ab>& A,
    const tile<16, 8, int, dl_ab>& B
) {
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    int32x8_t* acc = (int32x8_t*)D.x;
    const int32x4_t* a_vec = (const int32x4_t*)A.x;
    const int32x4_t* b_vec = (const int32x4_t*)B.x;
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
}

template <bool FULL>
static __device__ __forceinline__ void load_mq4v2_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + threadIdx.x * 4);
        const unsigned int q0 = (qs0 >>  0) & 0xFu;
        const unsigned int q1 = (qs0 >>  4) & 0xFu;
        const unsigned int q2 = (qs0 >>  8) & 0xFu;
        const unsigned int q3 = (qs0 >> 12) & 0xFu;
        const unsigned int q4 = (qs0 >> 16) & 0xFu;
        const unsigned int q5 = (qs0 >> 20) & 0xFu;
        const unsigned int q6 = (qs0 >> 24) & 0xFu;
        const unsigned int q7 = (qs0 >> 28) & 0xFu;
        x_qs[i * MMQ_TILE_X_K + 2 * threadIdx.x + 0] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        x_qs[i * MMQ_TILE_X_K + 2 * threadIdx.x + 1] = q4 | (q5 << 8) | (q6 << 16) | (q7 << 24);
    }
    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
            const int ksc = threadIdx.x & 1;
            const unsigned int hs = *(const unsigned int*)(gp + 4 * ksc);
            const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
        }
    }
}

// ---------------------------------------------------------------------------
// Base vec_dot (kept for reference, also used by ksplit which reuses K=8 steps)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        tile_A A_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix_16x8(A_frag[n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0, MMQ_TILE_X_K);
        }
        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[n], B_frag);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0 / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K2 vec_dot: process two k01 slices per j0/B load pair.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma_k2(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    // Unroll factor 2: step 16 instead of 8; handle k01 and k01+8 together.
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 2 * QI8_1) {
        const int k0a = k00 + k01;
        const int k0b = k00 + k01 + QI8_1;
        tile_A A_frag_a[ntx];
        tile_A A_frag_b[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix_16x8(A_frag_a[n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0a, MMQ_TILE_X_K);
            load_ldmatrix_16x8(A_frag_b[n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0b, MMQ_TILE_X_K);
        }
        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
            // Slice A: load B for k01
            tile_B B_frag_a;
            load_ldmatrix_16x8(B_frag_a, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB_a = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag_a[n], B_frag_a);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0a / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB_a.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB_a.y;
                }
            }
            // Slice B: k01+8
            tile_B B_frag_b;
            load_ldmatrix_16x8(B_frag_b, y_qs + j0 * MMQ_TILE_Y_K + k01 + QI8_1, MMQ_TILE_Y_K);
            const float2 dsB_b = __half22float2(y_dm[j * MMQ_TILE_Y_K + (k01 + QI8_1) / QI8_1]);
            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag_b[n], B_frag_b);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0b / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB_b.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB_b.y;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K4 vec_dot: process all 4 k01 slices of the 32-K half-block together.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma_k4(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    // All 4 slices (k01=0,8,16,24) — single outer iteration, 8 mmas per j0 block.
    tile_A A_frag[4][2];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int k0 = k00 + k * QI8_1;
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix_16x8(A_frag[k][n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0, MMQ_TILE_X_K);
        }
    }
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * MMQ_TILE_Y_K + k * QI8_1, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k]);
            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[k][n], B_frag);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + (k00 + k * QI8_1) / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K2 kernels
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void gemm_mq4g256v2_mmq_k2_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        if constexpr (FULL) { load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row); }
        else { load_mq4v2_tile<false>(A, tile_x, row0, kb, M, groups_per_row); }
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        if constexpr (FULL) {
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by0[l];
            }
        } else {
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by0[l] : 0;
            }
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            if constexpr (FULL) {
                for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                    const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                    tile_y[l] = by1[l];
                }
            } else {
                for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                    const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                    const int j = l / MMQ_TILE_Y_K;
                    tile_y[l] = (col0 + j < N) ? by1[l] : 0;
                }
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if constexpr (FULL) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    *yp = sum[(j0 / tile_C::J + n) * tile_C::ne + l] + (0 ? 0 : 0);
                    // FULL kernels are dispatched via _add/_set wrappers that handle add separately
                } else {
                    if (row0 + i < M && col0 + j < N) {
                        float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                        // generic body contributes to shared sum; writeback handled by wrapper with add flag
                        // To keep single body, we store sum; wrapper does add vs set.
                        // This body always accumulates sum; caller wrapper writes it.
                        // For k2 generic we need add-aware writeback — defer to kernel entry.
                        (void)yp;
                    }
                }
            }
        }
    }
    // Real writeback is done by the entry wrappers below (avoid double code).
    // Keep sum in registers; wrappers will re-derive indices. Simpler: do full writeback here and let wrappers just return.
    // Instead, do per-wrapper writeback: duplicate minimal loop in each entry.
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_k2(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_mq4v2_tile<false>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    if (add) *yp += v; else *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_k2_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                *yp += sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_k2_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k2(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                *yp = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K4 kernels
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_k4(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_mq4v2_tile<false>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k4(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k4(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    if (add) *yp += v; else *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_k4_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k4(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k4(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                *yp += sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_k4_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_k4(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_k4(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                *yp = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K-split kernels — grid.z = K_SPLITS, each block sums its K slice then
// atomicAdds into Y. Y must contain the residual on entry for add=true;
// Y must be zero for set=false (microbench zeroes it). Both use atomicAdd
// so ordering is deterministic only up to FP associativity (rel-RMS <0.002
// gate still passes; document any larger spread).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_ksplit(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int ksplit = blockIdx.z;
    if (row0 >= M || col0 >= N) return;
    const int groups_per_row = K / 256;
    const int groups_per_split = (groups_per_row + K_SPLITS - 1) / K_SPLITS;
    const int g_start = ksplit * groups_per_split;
    int g_end = g_start + groups_per_split;
    if (g_end > groups_per_row) g_end = groups_per_row;
    if (g_start >= g_end) return;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = g_start; kb < g_end; ++kb) {
        load_mq4v2_tile<false>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
                    float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    atomicAdd(&Y[(long long)(col0 + j) * M + (row0 + i)], v);
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_ksplit_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int ksplit = blockIdx.z;
    const int groups_per_row = K / 256;
    const int groups_per_split = (groups_per_row + K_SPLITS - 1) / K_SPLITS;
    const int g_start = ksplit * groups_per_split;
    int g_end = g_start + groups_per_split;
    if (g_end > groups_per_row) g_end = groups_per_row;
    if (g_start >= g_end) return;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = g_start; kb < g_end; ++kb) {
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                atomicAdd(&Y[(long long)(col0 + j) * M + (row0 + i)], v);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_ksplit_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int ksplit = blockIdx.z;
    const int groups_per_row = K / 256;
    const int groups_per_split = (groups_per_row + K_SPLITS - 1) / K_SPLITS;
    const int g_start = ksplit * groups_per_split;
    int g_end = g_start + groups_per_split;
    if (g_end > groups_per_row) g_end = groups_per_row;
    if (g_start >= g_end) return;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    for (int kb = g_start; kb < g_end; ++kb) {
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                atomicAdd(&Y[(long long)(col0 + j) * M + (row0 + i)], v);
            }
        }
    }
}

#endif
