#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ3 (uniform-MQ3) WMMA fused QKV GEMM, 4× batch-tile fanout.
// Sibling of gemm_hfq3g256_residual_wmma_mb4.hip; 3-way fan-out.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfq3g256_wmma_mb4(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_q,
    float*         __restrict__ Y_k,
    float*         __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m = q_m + k_m + v_m;
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
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
    }

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)local_row * groups_per_row * 104;

    long long safe_batch_s[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int b_idx = batch_start + 16 * s + my_tile_row;
        safe_batch_s[s] = (b_idx < N) ? (long long)b_idx : 0LL;
    }

    float8_t acc[4] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        const _Float16* xg_s[4];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            xg_s[s] = X + safe_batch_s[s] * (long long)K + g * 256;
        }

        for (int kt = 0; kt < 16; kt += 2) {
            const unsigned char* dpa = dp + kt * 6;
            const unsigned char* dpb = dp + (kt + 1) * 6;
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            unsigned int a0 = dpa[0], a1 = dpa[1], a2 = dpa[2];
            unsigned int a3 = dpa[3], a4 = dpa[4], a5 = dpa[5];

            unsigned int q0  = a0 & 7u;
            unsigned int q1  = (a0 >> 3) & 7u;
            unsigned int q2  = ((a0 >> 6) | (a1 << 2)) & 7u;
            unsigned int q3  = (a1 >> 1) & 7u;
            unsigned int q4  = (a1 >> 4) & 7u;
            unsigned int q5  = ((a1 >> 7) | (a2 << 1)) & 7u;
            unsigned int q6  = (a2 >> 2) & 7u;
            unsigned int q7  = (a2 >> 5) & 7u;
            unsigned int q8  = a3 & 7u;
            unsigned int q9  = (a3 >> 3) & 7u;
            unsigned int q10 = ((a3 >> 6) | (a4 << 2)) & 7u;
            unsigned int q11 = (a4 >> 1) & 7u;
            unsigned int q12 = (a4 >> 4) & 7u;
            unsigned int q13 = ((a4 >> 7) | (a5 << 1)) & 7u;
            unsigned int q14 = (a5 >> 2) & 7u;
            unsigned int q15 = (a5 >> 5) & 7u;

            half16_t a_reg;
            #define DQ(i, q) a_reg[i] = sc_h * (_Float16)(float)(q) + zp_h
            DQ(0,  q0);  DQ(1,  q1);  DQ(2,  q2);  DQ(3,  q3);
            DQ(4,  q4);  DQ(5,  q5);  DQ(6,  q6);  DQ(7,  q7);
            DQ(8,  q8);  DQ(9,  q9);  DQ(10, q10); DQ(11, q11);
            DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);

            half16_t b0a = *(const half16_t*)(xg_s[0] + k_off_a);
            half16_t b1a = *(const half16_t*)(xg_s[1] + k_off_a);
            half16_t b2a = *(const half16_t*)(xg_s[2] + k_off_a);
            half16_t b3a = *(const half16_t*)(xg_s[3] + k_off_a);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0a, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1a, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2a, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3a, acc[3]);

            unsigned int B0 = dpb[0], B1 = dpb[1], B2 = dpb[2];
            unsigned int B3 = dpb[3], B4 = dpb[4], B5 = dpb[5];

            unsigned int Q0  = B0 & 7u;
            unsigned int Q1  = (B0 >> 3) & 7u;
            unsigned int Q2  = ((B0 >> 6) | (B1 << 2)) & 7u;
            unsigned int Q3  = (B1 >> 1) & 7u;
            unsigned int Q4  = (B1 >> 4) & 7u;
            unsigned int Q5  = ((B1 >> 7) | (B2 << 1)) & 7u;
            unsigned int Q6  = (B2 >> 2) & 7u;
            unsigned int Q7  = (B2 >> 5) & 7u;
            unsigned int Q8  = B3 & 7u;
            unsigned int Q9  = (B3 >> 3) & 7u;
            unsigned int Q10 = ((B3 >> 6) | (B4 << 2)) & 7u;
            unsigned int Q11 = (B4 >> 1) & 7u;
            unsigned int Q12 = (B4 >> 4) & 7u;
            unsigned int Q13 = ((B4 >> 7) | (B5 << 1)) & 7u;
            unsigned int Q14 = (B5 >> 2) & 7u;
            unsigned int Q15 = (B5 >> 5) & 7u;

            DQ(0,  Q0);  DQ(1,  Q1);  DQ(2,  Q2);  DQ(3,  Q3);
            DQ(4,  Q4);  DQ(5,  Q5);  DQ(6,  Q6);  DQ(7,  Q7);
            DQ(8,  Q8);  DQ(9,  Q9);  DQ(10, Q10); DQ(11, Q11);
            DQ(12, Q12); DQ(13, Q13); DQ(14, Q14); DQ(15, Q15);
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
                    if (out_row < q_m) {
                        Y = Y_q;  proj_row = out_row;               out_stride = q_m;
                    } else if (out_row < q_m + k_m) {
                        Y = Y_k;  proj_row = out_row - q_m;          out_stride = k_m;
                    } else {
                        Y = Y_v;  proj_row = out_row - q_m - k_m;    out_stride = v_m;
                    }
                    Y[(long long)out_col * out_stride + proj_row] = acc[s][j];
                }
            }
        }
    }
}
