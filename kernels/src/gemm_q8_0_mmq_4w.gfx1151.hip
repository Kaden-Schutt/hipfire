#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Dense i8-WMMA MMQ GEMM for Q8_0 weights — 4-warp 64x64 tile (gfx1151).
//
// i8 port of gemm_q8_0_wmma_4w: same 64 M-row × 64 N-col tile and LDS-staged
// activations, but runs i8 WMMA (i32_16x16x16_iu8, ~2x FP16 rate) on Q8_1
// activations and reads the int8 weights directly (no f16 dequant). The 16x16
// i8 kernel lost to this f16 64x64 kernel on large shapes purely on tile size;
// matching the tile + i8 rate is the real win.
//
//   W_q8 : Q8_0 packed [M, K/32*34], per 32-K block [2B fp16 scale | 32B int8].
//   X    : Q8_1 (block_q8_1_mmq) [K/128 × N], built by ensure_q8_1_mmq_x.
//   Y    : FP32 [N, M] row-major.
//
// Q8_0 has no zero-point: per 32-K block the partial dot is scaled by
// (w_scale[m] * d_x[n]) and summed into the float accumulator.
//
// Grid: [ceil(M/64), ceil(N/64)]. Block: [128]. LDS: 2KB int8 + 256B scales.

#if defined(__gfx1151__)

#define M_TILE 64
#define N_TILE 64
#define WARPS 4
#define M_PER_WARP 16
#define N_TILES_PER_WARP 4

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_q8_0_mmq_4w_gfx1151(
    const char*           __restrict__ W_q8,  // [M, K/32 * 34]
    const block_q8_1_mmq* __restrict__ X_q8,  // [K/128 × N]
    float*                __restrict__ Y,     // [N, M]
    int M, int K, int N
) {
    const int block_m = blockIdx.x * M_TILE;
    const int block_n = blockIdx.y * N_TILE;
    if (block_m >= M || block_n >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row = warp_m_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks32 = K / 32;
    const long long row_stride = (long long)blocks32 * 34;
    const char* row_base = W_q8 + (long long)safe_row * row_stride;

    // LDS: 64 tokens × 32 int8 for the current 32-K block + 64 d_x scales.
    __shared__ int8_t lds_q[N_TILE * 32];
    __shared__ float  lds_dx[N_TILE];

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

    const int blocks128 = K / 128;

    for (int kb128 = 0; kb128 < blocks128; kb128++) {
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            const int bi = kb128 * 4 + sb;   // 32-K block index

            // ── Cooperative stage: 64 tokens × 32 int8 + 64 d_x for this block.
            // 128 threads load 2048 int8 → 16 bytes each.
            {
                const int flat = tid * 16;          // 0..2032 step 16
                const int n_row = flat >> 5;        // 0..63
                const int k_off = flat & 31;        // 0 or 16
                const int gn = block_n + n_row;
                const int safe_gn = (gn < N) ? gn : (N - 1);
                const block_q8_1_mmq* xb = X_q8 + (long long)kb128 * N + safe_gn;
                int8x16_t v = *(const int8x16_t*)(xb->qs + sb * 32 + k_off);
                *(int8x16_t*)(lds_q + n_row * 32 + k_off) = v;
                if (k_off == 0) {
                    lds_dx[n_row] = (gn < N) ? __half22float2(xb->ds4[sb]).x : 0.0f;
                }
            }
            __syncthreads();

            // ── Weights for this lane's row at block bi.
            const char* bp = row_base + (long long)bi * 34;
            const __half* wsc = reinterpret_cast<const __half*>(bp);
            float w_scale_self = __half2float(*wsc);
            float w_scale_row[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                w_scale_row[j] = __shfl(w_scale_self, 2 * j + (lane >> 4));
            }
            const int8x16_t aw0 = *(const int8x16_t*)(bp + 2);
            const int8x16_t aw1 = *(const int8x16_t*)(bp + 2 + 16);
            int32x4_t ai0 = *(const int32x4_t*)&aw0;
            int32x4_t ai1 = *(const int32x4_t*)&aw1;

            #pragma unroll
            for (int n_tile = 0; n_tile < N_TILES_PER_WARP; n_tile++) {
                const int n_in_tile = n_tile * 16 + lane16;
                const int8_t* lq = lds_q + n_in_tile * 32;
                int8x16_t b0 = *(const int8x16_t*)(lq);
                int8x16_t b1 = *(const int8x16_t*)(lq + 16);
                int32x4_t bi0 = *(const int32x4_t*)&b0;
                int32x4_t bi1 = *(const int32x4_t*)&b1;

                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};
                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ai0, true, bi0, cacc, true);
                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ai1, true, bi1, cacc, true);

                const float d_x = lds_dx[n_in_tile];
                float8_t* acc = (n_tile == 0) ? &acc0 : (n_tile == 1) ? &acc1
                              : (n_tile == 2) ? &acc2 : &acc3;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    (*acc)[j] += w_scale_row[j] * d_x * (float)cacc[j];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; n_tile++) {
        const int out_col = block_n + n_tile * 16 + lane16;
        if (out_col >= N) continue;
        float8_t acc = (n_tile == 0) ? acc0 : (n_tile == 1) ? acc1
                     : (n_tile == 2) ? acc2 : acc3;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = warp_m_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_q8_0_mmq_4w_gfx1151(
    const char*, const void*, float*, int, int, int
) {}

#endif
