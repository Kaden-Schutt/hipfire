#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) variant of the MQ3 / HFQ3-G256 residual WMMA kernel
// (`Y += A · X`). See `gemm_qkvza_hfq3g256_wmma.gfx12.hip` for the gfx12
// K4-unroll + half8_t lane-split rationale; this is the simplest of the
// MQ3 family — single weight matrix, single output Y with += semantics.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq3g256_residual_wmma_gfx12(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int safe_batch = (batch_start + m_lane < batch_size) ? (batch_start + m_lane) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 104;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    #define UNPACK_CHUNK_DQ(b0, b1, b2)                          \
        do {                                                      \
            unsigned int _a0 = (b0), _a1 = (b1), _a2 = (b2);     \
            a_reg[0] = sc_h * (_Float16)(float)(_a0 & 7u) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)((_a0 >> 3) & 7u) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((_a0 >> 6) | (_a1 << 2)) & 7u) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)((_a1 >> 1) & 7u) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)((_a1 >> 4) & 7u) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((_a1 >> 7) | (_a2 << 1)) & 7u) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)((_a2 >> 2) & 7u) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)((_a2 >> 5) & 7u) + zp_h; \
        } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        for (int kt = 0; kt < 16; kt += 4) {
            const unsigned char* dpa = dp + (kt + 0) * 6 + k_grp * 3;
            const unsigned char* dpb = dp + (kt + 1) * 6 + k_grp * 3;
            const unsigned char* dpc = dp + (kt + 2) * 6 + k_grp * 3;
            const unsigned char* dpd = dp + (kt + 3) * 6 + k_grp * 3;

            half8_t b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);

            unsigned int a0 = dpa[0], a1 = dpa[1], a2 = dpa[2];
            unsigned int b0 = dpb[0], b1 = dpb[1], b2 = dpb[2];
            unsigned int c0 = dpc[0], c1 = dpc[1], c2 = dpc[2];
            unsigned int d0 = dpd[0], d1 = dpd[1], d2 = dpd[2];

            half8_t a_reg;
            UNPACK_CHUNK_DQ(a0, a1, a2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            UNPACK_CHUNK_DQ(b0, b1, b2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            UNPACK_CHUNK_DQ(c0, c1, c2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            UNPACK_CHUNK_DQ(d0, d1, d2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
    }
    #undef UNPACK_CHUNK_DQ

    // gfx12 wave32 WMMA C-mapping: contiguous rows per lane-group.
    const int out_col = batch_start + m_lane;
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
