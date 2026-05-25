#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ4-Lloyd WMMA fused QKVZA GEMM (LA preamble: qkv + z + beta + alpha),
// 4× batch-tile fanout per workgroup. Phase D-B sibling.
//
// See gemm_mq4g256_lloyd_residual_wmma_mb4.hip for the structural template
// (16×64 output tile, 4 distinct named B-vars to defeat compiler register-
// recycling). The K-tile decode body is identical to the Phase B1 fused
// kernel; only the per-WG output shape, the 4-way fanout dispatch, and the
// per-sub-tile output-write loop differ.
//
// Grid: [ceil((qkv_m+z_m+beta_m+alpha_m)/16), ceil(N/64)]. Block: [32]. LDS: 512 B.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_mq4g256_lloyd_wmma_mb4(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 64;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_tile_row = tid & 15;
    const int my_row = row_start + my_tile_row;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < qkv_m) {
        A = A_qkv;   local_row = safe_row;
    } else if (safe_row < qkv_m + z_m) {
        A = A_z;     local_row = safe_row - qkv_m;
    } else if (safe_row < qkv_m + z_m + beta_m) {
        A = A_beta;  local_row = safe_row - (qkv_m + z_m);
    } else {
        A = A_alpha; local_row = safe_row - (qkv_m + z_m + beta_m);
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + local_row * row_stride;

    long long safe_batch_s[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int b_idx = batch_start + 16 * s + my_tile_row;
        safe_batch_s[s] = (b_idx < N) ? (long long)b_idx : 0LL;
    }

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 8;
    const int load_my_row = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < total_m) ? load_my_row : (total_m - 1);
    const char* load_A;
    int load_local_row;
    if (load_safe_row < qkv_m) {
        load_A = A_qkv;   load_local_row = load_safe_row;
    } else if (load_safe_row < qkv_m + z_m) {
        load_A = A_z;     load_local_row = load_safe_row - qkv_m;
    } else if (load_safe_row < qkv_m + z_m + beta_m) {
        load_A = A_beta;  load_local_row = load_safe_row - (qkv_m + z_m);
    } else {
        load_A = A_alpha; load_local_row = load_safe_row - (qkv_m + z_m + beta_m);
    }
    const char* load_row_base = load_A + (long long)load_local_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];

    float8_t acc[4] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const unsigned char* dp = (const unsigned char*)(gp + 32);
        const int cb_base = my_tile_row * 16;

        const _Float16* xg_s[4];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            xg_s[s] = X + safe_batch_s[s] * (long long)K + g * 256;
        }

        for (int kt = 0; kt < 16; kt += 2) {
            const unsigned char* dpa = dp + kt * 8;
            const unsigned char* dpb = dp + (kt + 1) * 8;
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            unsigned int pk0a = *(const unsigned int*)dpa;
            unsigned int pk1a = *(const unsigned int*)(dpa + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            half16_t b0a = *(const half16_t*)(xg_s[0] + k_off_a);
            half16_t b1a = *(const half16_t*)(xg_s[1] + k_off_a);
            half16_t b2a = *(const half16_t*)(xg_s[2] + k_off_a);
            half16_t b3a = *(const half16_t*)(xg_s[3] + k_off_a);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0a, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1a, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2a, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3a, acc[3]);

            unsigned int pk0b = *(const unsigned int*)dpb;
            unsigned int pk1b = *(const unsigned int*)(dpb + 4);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            half16_t b0b = *(const half16_t*)(xg_s[0] + k_off_b);
            half16_t b1b = *(const half16_t*)(xg_s[1] + k_off_b);
            half16_t b2b = *(const half16_t*)(xg_s[2] + k_off_b);
            half16_t b3b = *(const half16_t*)(xg_s[3] + k_off_b);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0b, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1b, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2b, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3b, acc[3]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int out_col = batch_start + 16 * s + (tid & 15);
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row < total_m) {
                    float* Y;
                    int proj_row;
                    int out_stride;
                    if (out_row < qkv_m) {
                        Y = Y_qkv;   proj_row = out_row;                          out_stride = qkv_m;
                    } else if (out_row < qkv_m + z_m) {
                        Y = Y_z;     proj_row = out_row - qkv_m;                  out_stride = z_m;
                    } else if (out_row < qkv_m + z_m + beta_m) {
                        Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);          out_stride = beta_m;
                    } else {
                        Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m); out_stride = alpha_m;
                    }
                    Y[(long long)out_col * out_stride + proj_row] = acc[s][j];
                }
            }
        }
    }
}
