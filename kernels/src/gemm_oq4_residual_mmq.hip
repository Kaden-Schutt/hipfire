// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OQ4+ MMQ (int8-WMMA) batched GEMM for RDNA3/RDNA3.5 — the int8-dp4a-class
// backend that closes the OQ4+ prefill gap to mq4+. Ported from
// gemm_hfq4g256_residual_mmq.hip (mq4's MMQ): identical q8_1-activation +
// LDS-staged 128×128 i8-WMMA machinery, with ONE change — the weight-tile loader
// reads OQ4+'s SPLIT layout `[nibbles M*(K/2) | f32 scales M*ng]` and
// SIGN-EXTENDS the symmetric int4 weight to signed int8 (no zero-point, so the
// q8_1 zp-correction term vanishes). The activation is pre-quantized to
// block_q8_1_mmq by the shared quantize_q8_1_mmq_ds4 kernel (reused via
// ensure_q8_1_mmq_x — NOT redefined here, to avoid a duplicate symbol).
//
// int8-WMMA + LDS activation-reuse across 128 weight rows is ~1.5× the f16-WMMA
// path on gfx1103 (mq4 MMQ 1606 vs f16 1060 tok/s prefill). LDS use matches
// mq4's proven-on-gfx1103 footprint (no wedge).

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

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

#if !HIPFIRE_RDNA3

extern "C" __global__ void gemm_oq4_residual_mmq(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_oq4_residual_mmq_full_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_oq4_residual_mmq_full_add(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int) {}

#else

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
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
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

// OQ4+ weight tile load: split layout + sign-extended symmetric int4 → int8.
// Nibbles for (row, group kb) live at A + row*(K/2) + kb*128; the f32 per-group
// scale lives in the scales region right after M*(K/2) bytes. zp = 0 (symmetric).
template <bool FULL>
static __device__ __forceinline__ void load_oq4_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int K,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);

    const long long stride = (long long)K / 2;                 // nibble bytes per row
    const float* Ws = (const float*)(A + (long long)M * stride);

    constexpr int rows_per_warp = 1;
    const int txi = threadIdx.x;

    for (int i0 = 0; i0 < MMQ_Y; i0 += rows_per_warp * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const uint8_t* wrow = (const uint8_t*)A + (long long)row * stride + (long long)kb * 128;
        const unsigned int qs0 = *(const unsigned int*)(wrow + txi * 4); // 8 nibbles
        #define SX(v) ((int)(((int)(v) << 28) >> 28) & 0xFF)
        const int q0 = SX((qs0 >>  0) & 0xFu);
        const int q1 = SX((qs0 >>  4) & 0xFu);
        const int q2 = SX((qs0 >>  8) & 0xFu);
        const int q3 = SX((qs0 >> 12) & 0xFu);
        const int q4 = SX((qs0 >> 16) & 0xFu);
        const int q5 = SX((qs0 >> 20) & 0xFu);
        const int q6 = SX((qs0 >> 24) & 0xFu);
        const int q7 = SX((qs0 >> 28) & 0xFu);
        #undef SX
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] = q4 | (q5 << 8) | (q6 << 16) | (q7 << 24);
    }

    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const float sc = Ws[(long long)row * groups_per_row + kb];
            const int ksc = threadIdx.x & 1;
            const half2 dm = make_half2((_Float16)sc, (_Float16)0.0f); // zp = 0 (symmetric)
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
            }
        }
    }
}

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

// Generic bounds-checked OQ4+ MMQ GEMM. `add` (runtime): 0 → Y = W·x (SET, for
// qkv/gate_up), 1 → Y += W·x (residual, for wo/down). Block (32,8), grid
// [ceil(M/128), ceil(N/128)], shared_mem = (MMQ_X*36 + MMQ_Y*76)*4 bytes.
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_oq4_residual_mmq(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
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
        load_oq4_tile<false>(A, tile_x, row0, kb, M, K, groups_per_row);

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
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

// FULL variants: M%128==0 && N%128==0 — no min-clamped loads, no epilogue bounds
// checks (the mq4 full_set/full_add fast path; the real prefill speedup).
template <bool ADD>
static __device__ __forceinline__ void gemm_oq4_residual_mmq_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
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
        load_oq4_tile<true>(A, tile_x, row0, kb, M, K, groups_per_row);

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
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                if constexpr (ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_oq4_residual_mmq_full_set(
    const char* __restrict__ A, const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int
) {
    gemm_oq4_residual_mmq_full_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_oq4_residual_mmq_full_add(
    const char* __restrict__ A, const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int
) {
    gemm_oq4_residual_mmq_full_body<true>(A, Xq, Y, M, K, N);
}

#endif
