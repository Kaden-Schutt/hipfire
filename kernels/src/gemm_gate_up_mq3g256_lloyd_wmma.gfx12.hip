#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) variant of MQ3-Lloyd-G256 fused gate+up WMMA (FFN).
// Sister of gemm_gate_up_mq3g256_lloyd_wmma.hip and the residual gfx12.
// **Code-complete, runtime-unvalidated locally.**

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// C symbol kept unsuffixed (see residual gfx12 sibling for rationale).
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_mq3g256_lloyd_wmma(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
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
    if (safe_row < gate_m) {
        A = A_gate; local_row = safe_row;
    } else {
        A = A_up;   local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 4;
    const int load_my_row = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < total_m) ? load_my_row : (total_m - 1);
    const char* load_A;
    int load_local_row;
    if (load_safe_row < gate_m) {
        load_A = A_gate; load_local_row = load_safe_row;
    } else {
        load_A = A_up;   load_local_row = load_safe_row - gate_m;
    }
    const char* load_row_base = load_A + (long long)load_local_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 8];

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    #define UNPACK_CHUNK_DQ(b0, b1, b2)                                            \
        do {                                                                       \
            unsigned int _a0 = (b0), _a1 = (b1), _a2 = (b2);                       \
            a_reg[0] = cb_lds[cb_base + (_a0 & 7u)];                               \
            a_reg[1] = cb_lds[cb_base + ((_a0 >> 3) & 7u)];                        \
            a_reg[2] = cb_lds[cb_base + (((_a0 >> 6) | (_a1 << 2)) & 7u)];         \
            a_reg[3] = cb_lds[cb_base + ((_a1 >> 1) & 7u)];                        \
            a_reg[4] = cb_lds[cb_base + ((_a1 >> 4) & 7u)];                        \
            a_reg[5] = cb_lds[cb_base + (((_a1 >> 7) | (_a2 << 1)) & 7u)];         \
            a_reg[6] = cb_lds[cb_base + ((_a2 >> 2) & 7u)];                        \
            a_reg[7] = cb_lds[cb_base + ((_a2 >> 5) & 7u)];                        \
        } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 112);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cb_lds[load_tile_row * 8 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 112;
        const _Float16* xg = x_base + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 16);
        const int cb_base = m_lane * 8;

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
        __syncthreads();
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
                if (out_row < gate_m) {
                    Y = Y_gate; proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;   proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
