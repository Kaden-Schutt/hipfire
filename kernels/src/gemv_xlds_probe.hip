
#include <hip/hip_runtime.h>

// LDS-staged activation GEMV probe kernels. Each block stages the full
// activation vector in shared memory once, then processes ROWS_PER_BLOCK
// rows reading x from LDS instead of re-reading global/L1 per row.
// Per-row FMA order matches gemv_hfq4g256_residual.gfx1100.hip's quad-
// accumulator structure -> bit-exact vs shipping for identical inputs.

#define BODY(RPB)                                                        \
    extern __shared__ float xs[];                                        \
    const int groups_per_row = K >> 8;                                   \
    for (int i = threadIdx.x; i < K; i += 32) xs[i] = xg[i];             \
    __syncthreads();                                                     \
    for (int rr = 0; rr < (RPB); rr++) {                                 \
        const int row = blockIdx.x * (RPB) + rr;                         \
        if (row >= M) break;                                             \
        const char* row_ptr = A + (long long)row * (long long)groups_per_row * 136; \
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;        \
        for (int g = 0; g < groups_per_row; g++) {                       \
            const char* gp = row_ptr + g * 136;                          \
            const unsigned int pk = *(const unsigned int*)(gp + 8 + threadIdx.x * 4); \
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
            const float4 xv = *reinterpret_cast<const float4*>(xs + g * 256 + threadIdx.x * 8); \
            acc0 += (sc * (float)((pk) & 0xFu)         + zp) * xv.x     \
                  + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xv.y     \
                  + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xv.z     \
                  + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xv.w;    \
            acc1 += (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xv.x     \
                  + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xv.y     \
                  + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xv.z     \
                  + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xv.w;    \
        }                                                                \
        for (int offset = 16; offset > 0; offset >>= 1) {                \
            acc0 += __shfl_xor(acc0, offset);                            \
            acc1 += __shfl_xor(acc1, offset);                            \
        }                                                                \
        if (threadIdx.x == 0) y[row] += acc0 + acc1;                     \
    }

extern "C" __launch_bounds__(32, 4)
__global__ void gemv_xlds_r1(const char* A, const float* xg, float* y, int M, int K)
{ const int RPB = 1; BODY(1) }

extern "C" __launch_bounds__(32, 4)
__global__ void gemv_xlds_r2(const char* A, const float* xg, float* y, int M, int K)
{ const int RPB = 2; BODY(2) }

extern "C" __launch_bounds__(32, 2)
__global__ void gemv_xlds_r4(const char* A, const float* xg, float* y, int M, int K)
{ const int RPB = 4; BODY(4) }

extern "C" __launch_bounds__(32, 2)
__global__ void gemv_xlds_r8(const char* A, const float* xg, float* y, int M, int K)
{ const int RPB = 8; BODY(8) }
