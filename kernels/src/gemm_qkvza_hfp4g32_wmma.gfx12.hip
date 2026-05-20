#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) sister of gemm_qkvza_hfp4g32_wmma. half8_t lane-split
// + K4 unroll. Same C-output mapping as gemm_qkvza_hfq4g256_wmma.gfx12.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_hfp4g32_wmma_gfx12(
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
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
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

    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)local_row * row_stride;
    const char* row_ptr = hdr + 16;

    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int groups_per_row = K / 256;

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_base = row_ptr + (long long)g * 136;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        for (int kt = 0; kt < 16; kt += 4) {
            const int blk_a = kt >> 1;
            const int blk_c = blk_a + 1;
            const char* bpa = group_base + blk_a * 17;
            const char* bpc = group_base + blk_c * 17;

            const unsigned int e_a = *(const unsigned char*)bpa;
            const unsigned int e_c = *(const unsigned char*)bpc;
            const float sc_block_a_f = __builtin_bit_cast(float, e_a << 23);
            const float sc_block_c_f = __builtin_bit_cast(float, e_c << 23);
            const _Float16 sc_h_ab = row_scale_h * (_Float16)sc_block_a_f;
            const _Float16 sc_h_cd = row_scale_h * (_Float16)sc_block_c_f;

            unsigned int pka = *(const unsigned int*)(bpa + 1 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(bpa + 9 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(bpc + 1 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(bpc + 9 + k_grp * 4);

            const int k_off_a = (kt + 0) * 16 + k_grp * 8;
            const int k_off_b = (kt + 1) * 16 + k_grp * 8;
            const int k_off_c = (kt + 2) * 16 + k_grp * 8;
            const int k_off_d = (kt + 3) * 16 + k_grp * 8;

            half8_t b_a = *(const half8_t*)(x_base + k_off_a);
            half8_t b_b = *(const half8_t*)(x_base + k_off_b);
            half8_t b_c = *(const half8_t*)(x_base + k_off_c);
            half8_t b_d = *(const half8_t*)(x_base + k_off_d);

            half8_t a_reg;
            #define DQ8(pk, sc) \
                a_reg[0] = (sc) * lut[((pk) >>  0) & 0xFu]; \
                a_reg[1] = (sc) * lut[((pk) >>  4) & 0xFu]; \
                a_reg[2] = (sc) * lut[((pk) >>  8) & 0xFu]; \
                a_reg[3] = (sc) * lut[((pk) >> 12) & 0xFu]; \
                a_reg[4] = (sc) * lut[((pk) >> 16) & 0xFu]; \
                a_reg[5] = (sc) * lut[((pk) >> 20) & 0xFu]; \
                a_reg[6] = (sc) * lut[((pk) >> 24) & 0xFu]; \
                a_reg[7] = (sc) * lut[((pk) >> 28) & 0xFu]

            DQ8(pka, sc_h_ab);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb, sc_h_ab);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc, sc_h_cd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd, sc_h_cd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);

            #undef DQ8
        }
    }

    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < qkv_m) {
                    Y = Y_qkv;   proj_row = out_row;                            out_stride = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                    out_stride = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);            out_stride = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m);   out_stride = alpha_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
