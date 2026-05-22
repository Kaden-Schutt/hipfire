// Shared body for HFQ3-G256 wave32 MMQ 3-way fused (Q + K + V) GEMM.
//
// HFQ3 qkv sibling of `gemm_hfq3g256_residual_mmq_body.cuh`. Same LDS-tiled
// X reuse + sdot4 inner loop; differs in:
//   - 3-way output routing per workgroup (workgroup picks one of Q/K/V
//     based on which row-tile band it falls into)
//   - Overwrite write-back (Y[col][row] = acc, no residual add)
//
// CALLER INVARIANT: q_m, k_m, v_m must each be a multiple of MMQ_Y=128.
// Qwen3.5 family satisfies this (e.g. 9B: q_m=4096, k_m=v_m=1024). If
// future shapes violate the invariant, a per-row routing variant would
// be needed (much more complex).
//
// Each tile-size instantiation defines MMQ_X (kernel-side batch tile) and
// MMQ_X_NAME (symbol suffix) before #including this header.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 32
#define QK8_1 32
#define X_STRIDE 40
#define Y_STRIDE 36

#ifndef MMQ_X
#error "MMQ_X must be defined before #including this body"
#endif
#ifndef KERNEL_NAME
#error "KERNEL_NAME must be defined before #including this body"
#endif

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

__launch_bounds__(128, 2)
extern "C" __global__ void KERNEL_NAME(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    // Workgroup row tile: which of the 3 matrices does this WG's 128
    // rows belong to? Routing assumes q_m, k_m, v_m each MMQ_Y-aligned.
    const int total_row = blockIdx.x * MMQ_Y;
    const int total_m = q_m + k_m + v_m;
    if (total_row >= total_m) return;

    const char* A;
    float* Y;
    int row0;     // row index within the routed matrix
    int out_m;    // M of the routed matrix (Y stride)
    if (total_row < q_m) {
        A = A_q; Y = Y_q; row0 = total_row;            out_m = q_m;
    } else if (total_row < q_m + k_m) {
        A = A_k; Y = Y_k; row0 = total_row - q_m;       out_m = k_m;
    } else {
        A = A_v; Y = Y_v; row0 = total_row - q_m - k_m; out_m = v_m;
    }

    const int col0 = blockIdx.y * MMQ_X;
    if (col0 >= N) return;

    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;

    extern __shared__ int smem[];
    int*    x_qs   = smem;
    float2* x_dm   = (float2*)(x_qs + MMQ_Y * X_STRIDE);
    int*    tile_y = (int*)(x_dm + MMQ_Y);

    float sum[(MMQ_X / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        #pragma unroll 1
        for (int window = 0; window < 2; ++window) {
            const int kb = 2 * kg + window;

            // ── Load Q8_1 Y tile (one 128-K block) ────────────────────────
            const int total_y_ints = MMQ_X * Y_STRIDE;
            for (int u = tid; u < total_y_ints; u += 128) {
                const int j = u / Y_STRIDE;
                const int slot = u % Y_STRIDE;
                const bool valid = (col0 + j) < N;
                const int col = valid ? (col0 + j) : (N - 1);
                const int* src = (const int*)(Xq + (long long)kb * N + col);
                tile_y[u] = valid ? src[slot] : 0;
            }

            // ── Load X tile (HFQ3 3-bit unpack → signed INT8 packed) ──────
            // out_m is the row stride of the routed weight matrix. Clamp
            // OOB rows to row out_m-1 (same as residual body).
            if (window == 0 && tid < 128) {
                const int i = tid;
                const int row = (row0 + i < out_m) ? (row0 + i) : (out_m - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 104;
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                x_dm[i] = make_float2(sc, zp + 4.0f * sc);
            }

            #pragma unroll
            for (int loop = 0; loop < 16; ++loop) {
                const int task_id = tid * 16 + loop;
                const int i = task_id / 16;
                const int chunk = task_id % 16;

                const int row = (row0 + i < out_m) ? (row0 + i) : (out_m - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 104;

                const unsigned char* d = (const unsigned char*)(gp + 8 + window * 48 + chunk * 3);
                const unsigned int pk = (unsigned int)d[0]
                                      | ((unsigned int)d[1] << 8)
                                      | ((unsigned int)d[2] << 16);

                const unsigned int n0 = (pk      ) & 7u;
                const unsigned int n1 = (pk >>  3) & 7u;
                const unsigned int n2 = (pk >>  6) & 7u;
                const unsigned int n3 = (pk >>  9) & 7u;
                const unsigned int n4 = (pk >> 12) & 7u;
                const unsigned int n5 = (pk >> 15) & 7u;
                const unsigned int n6 = (pk >> 18) & 7u;
                const unsigned int n7 = (pk >> 21) & 7u;
                const int int_a = (int)(((n0 - 4) & 0xFF) | (((n1 - 4) & 0xFF) << 8)
                                      | (((n2 - 4) & 0xFF) << 16) | (((n3 - 4) & 0xFF) << 24));
                const int int_b = (int)(((n4 - 4) & 0xFF) | (((n5 - 4) & 0xFF) << 8)
                                      | (((n6 - 4) & 0xFF) << 16) | (((n7 - 4) & 0xFF) << 24));

                x_qs[i * X_STRIDE + 2 * chunk + 0] = int_a;
                x_qs[i * X_STRIDE + 2 * chunk + 1] = int_b;
            }

            __syncthreads();

            // ── 4 sub-blocks back-to-back ──────────────────────────────────
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const int kx_start = sub * 8;
                const int ky_start = 4 + sub * 8;

                #pragma unroll
                for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_NWARPS) {
                    const int j = j0 + threadIdx.y;

                    const half2* y_ds_col = (const half2*)(tile_y + j * Y_STRIDE);
                    const half2 ds_j = y_ds_col[sub];

                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
                        const int i = i0 + threadIdx.x;

                        const int4 x_v0 = *(const int4*)&x_qs[i * X_STRIDE + kx_start + 0];
                        const int4 x_v1 = *(const int4*)&x_qs[i * X_STRIDE + kx_start + 4];
                        const int4 y_v0 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 0];
                        const int4 y_v1 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 4];

                        int sumi = 0;
                        sumi = __builtin_amdgcn_sdot4(x_v0.x, y_v0.x, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v0.y, y_v0.y, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v0.z, y_v0.z, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v0.w, y_v0.w, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.x, y_v1.x, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.y, y_v1.y, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.z, y_v1.z, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.w, y_v1.w, sumi, false);

                        const float2 dm_i = x_dm[i];
                        const float2 dsf = __half22float2(ds_j);
                        const float scale_w = dm_i.x;
                        const float zp_eff  = dm_i.y;
                        const float d_x     = dsf.x;
                        const float sum_x   = dsf.y;

                        const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
                        sum[idx] += scale_w * d_x * (float)sumi + zp_eff * sum_x;
                    }
                }
            }

            __syncthreads();
        }
    }

    // Write-back — OVERWRITE semantics (Y = acc, not Y += acc).
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col = col0 + j;
        if (col >= N) continue;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row = row0 + i;
            if (row >= out_m) continue;

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            const long long out_idx = (long long)col * out_m + row;
            Y[out_idx] = sum[idx];
        }
    }
}
