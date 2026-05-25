// Shared body for HFQ4-G256 wave32 MMQ 4-way fused (LA preamble) GEMM.
//
// Routes per workgroup to one of: wqkv / wz / w_beta / w_alpha. Used by
// the dense LinearAttention preamble in Qwen3.5 / Qwen3.5-VL on HFQ4.
//
// CALLER INVARIANT: qkv_m, z_m, beta_m, alpha_m must each be a multiple
// of MMQ_Y=128. Qwen3.5 family generally satisfies this except for
// beta_m/alpha_m on small LA configs — see the dispatcher's split
// routing (gemm_gate_up MMQ on qkv+z + dot2 on β+α) for those cases.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef MMQ_Y
#define MMQ_Y 128
#endif
#define MMQ_NWARPS 4
#define WAVE_SIZE 32
#define QK8_1 32
#define X_STRIDE 40
#define Y_STRIDE 36
#define X_LOADER_TASKS_PER_THREAD ((MMQ_Y * 16) / 128)

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
    const char* __restrict__ A_qkv,
    const char* __restrict__ A_z,
    const char* __restrict__ A_beta,
    const char* __restrict__ A_alpha,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_qkv,
    float* __restrict__ Y_z,
    float* __restrict__ Y_beta,
    float* __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_row = blockIdx.x * MMQ_Y;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (total_row >= total_m) return;

    const char* A;
    float* Y;
    int row0;
    int out_m;
    if (total_row < qkv_m) {
        A = A_qkv;   Y = Y_qkv;   row0 = total_row;                              out_m = qkv_m;
    } else if (total_row < qkv_m + z_m) {
        A = A_z;     Y = Y_z;     row0 = total_row - qkv_m;                       out_m = z_m;
    } else if (total_row < qkv_m + z_m + beta_m) {
        A = A_beta;  Y = Y_beta;  row0 = total_row - (qkv_m + z_m);                out_m = beta_m;
    } else {
        A = A_alpha; Y = Y_alpha; row0 = total_row - (qkv_m + z_m + beta_m);       out_m = alpha_m;
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

            const int total_y_ints = MMQ_X * Y_STRIDE;
            for (int u = tid; u < total_y_ints; u += 128) {
                const int j = u / Y_STRIDE;
                const int slot = u % Y_STRIDE;
                const bool valid = (col0 + j) < N;
                const int col = valid ? (col0 + j) : (N - 1);
                const int* src = (const int*)(Xq + (long long)kb * N + col);
                tile_y[u] = valid ? src[slot] : 0;
            }

            // HFQ4 X-tile metadata
            if (window == 0 && tid < 128) {
                const int i = tid;
                const int row = (row0 + i < out_m) ? (row0 + i) : (out_m - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                x_dm[i] = make_float2(sc, zp + 8.0f * sc);
            }

            #pragma unroll
            for (int loop = 0; loop < X_LOADER_TASKS_PER_THREAD; ++loop) {
                const int task_id = tid * X_LOADER_TASKS_PER_THREAD + loop;
                const int i = task_id / 16;
                const int chunk = task_id % 16;

                const int row = (row0 + i < out_m) ? (row0 + i) : (out_m - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
                const unsigned int qs0 = *(const unsigned int*)(gp + 8 + window * 64 + chunk * 4);

                const unsigned int n0 = (qs0      ) & 0xFu;
                const unsigned int n1 = (qs0 >>  4) & 0xFu;
                const unsigned int n2 = (qs0 >>  8) & 0xFu;
                const unsigned int n3 = (qs0 >> 12) & 0xFu;
                const unsigned int n4 = (qs0 >> 16) & 0xFu;
                const unsigned int n5 = (qs0 >> 20) & 0xFu;
                const unsigned int n6 = (qs0 >> 24) & 0xFu;
                const unsigned int n7 = (qs0 >> 28) & 0xFu;
                const int int_a = (int)(((n0 - 8) & 0xFF) | (((n1 - 8) & 0xFF) << 8)
                                      | (((n2 - 8) & 0xFF) << 16) | (((n3 - 8) & 0xFF) << 24));
                const int int_b = (int)(((n4 - 8) & 0xFF) | (((n5 - 8) & 0xFF) << 8)
                                      | (((n6 - 8) & 0xFF) << 16) | (((n7 - 8) & 0xFF) << 24));

                x_qs[i * X_STRIDE + 2 * chunk + 0] = int_a;
                x_qs[i * X_STRIDE + 2 * chunk + 1] = int_b;
            }

            __syncthreads();

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

    // OVERWRITE write-back.
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
