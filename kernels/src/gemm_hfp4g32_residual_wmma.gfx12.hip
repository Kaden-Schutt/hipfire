#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) sister of gemm_hfp4g32_residual_wmma. half8_t lane-split
// + K4 unroll. Same dequant + Y semantics; only WMMA shape differs.
//
// gfx12 wave32 WMMA C-mapping (rows contiguous):
//   acc[j] = C[8*(tid>>4) + j][tid & 15]
// Lane group 0 → output rows 0..7, group 1 → rows 8..15.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfp4g32_residual_wmma_gfx12(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

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

    const int safe_row   = (row_start   + m_lane < M)          ? (row_start   + m_lane) : (M - 1);
    const int safe_batch = (batch_start + m_lane < batch_size) ? (batch_start + m_lane) : 0;

    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)safe_row * row_stride;
    const char* row_ptr = hdr + 16;

    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);

    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const int groups_per_row = K / 256;

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_base = row_ptr + (long long)g * 136;
        const _Float16* xg = x_base + g * 256;

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

            half8_t b_a = *(const half8_t*)(xg + k_off_a);
            half8_t b_b = *(const half8_t*)(xg + k_off_b);
            half8_t b_c = *(const half8_t*)(xg + k_off_c);
            half8_t b_d = *(const half8_t*)(xg + k_off_d);

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

    const int out_col = batch_start + m_lane;  // batch index (N)
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;  // M index
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
