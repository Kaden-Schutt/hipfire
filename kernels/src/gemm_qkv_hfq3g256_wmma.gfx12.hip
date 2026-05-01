#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) variant of the 3-way fused MQ3 / HFQ3-G256 q+k+v WMMA
// preamble (FA layer). See `gemm_qkvza_hfq3g256_wmma.gfx12.hip` for the
// gfx12 K4-unroll + half8_t lane-split rationale; only the output routing
// differs (3 weight matrices q/k/v vs 4 qkv/z/beta/alpha).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfq3g256_wmma_gfx12(
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
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
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
    const long long row_stride = (long long)groups_per_row * 104;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        for (int kt = 0; kt < 16; kt += 4) {
            const unsigned char* dpa = dp + (kt + 0) * 6 + k_grp * 3;
            const unsigned char* dpb = dp + (kt + 1) * 6 + k_grp * 3;
            const unsigned char* dpc = dp + (kt + 2) * 6 + k_grp * 3;
            const unsigned char* dpd = dp + (kt + 3) * 6 + k_grp * 3;

            half8_t b_a = *(const half8_t*)(x_base + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(x_base + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(x_base + (kt + 3) * 16 + k_grp * 8);

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

    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
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
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
