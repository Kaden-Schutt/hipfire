#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused HFQ4-G256 GEMM for the Qwen3.5
// DeltaNet LinearAttention preamble (qkv + z + beta + alpha projections).
//
// **gfx12 (RDNA4) variant** — sister of gemm_qkvza_hfq4g256_wmma.hip
// (gfx11 / RDNA3). Combines the gfx12 hfq4 recipe (validated commits
// 7a5a6b4 / 986a630) with the 4-output Q+K+V+Z+A routing.
//
// gfx12 differences from the gfx11 source:
//   1. _w32 → _w32_gfx12 builtin
//   2. half16_t → half8_t A/B operands; float8_t accumulator unchanged
//   3. K-split via tid >> 4 (each lane carries 8 of 16 K values)
//   4. C-output mapping: contiguous rows per lane-group:
//        acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// Grid: [ceil((qkv_m + z_m + beta_m + alpha_m)/16), ceil(N/16)].
// Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_hfq4g256_wmma_gfx12(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,         // [N x K], FP16
    float*         __restrict__ Y_qkv,      // [N x qkv_m]
    float*         __restrict__ Y_z,        // [N x z_m]
    float*         __restrict__ Y_beta,     // [N x beta_m]
    float*         __restrict__ Y_alpha,    // [N x alpha_m]
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's row.
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
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K4 unroll (issue #65 lever 2): 4 K-tiles per body, all loads
        // up-front. Measured 2× on gate_up at 9B pp=128.
        #define DQ8(pk) \
            a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

        for (int kt = 0; kt < 16; kt += 4) {
            const int k_off_a = (kt + 0) * 16;
            const int k_off_b = (kt + 1) * 16;
            const int k_off_c = (kt + 2) * 16;
            const int k_off_d = (kt + 3) * 16;

            unsigned int pka = *(const unsigned int*)(gp + 8 + k_off_a / 2 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(gp + 8 + k_off_b / 2 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(gp + 8 + k_off_c / 2 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(gp + 8 + k_off_d / 2 + k_grp * 4);

            half8_t b_a = *(const half8_t*)(x_base + k_off_a + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + k_off_b + k_grp * 8);
            half8_t b_c = *(const half8_t*)(x_base + k_off_c + k_grp * 8);
            half8_t b_d = *(const half8_t*)(x_base + k_off_d + k_grp * 8);

            half8_t a_reg;
            DQ8(pka);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ8
    }

    // gfx12 wave32 WMMA C-mapping: contiguous rows per lane-group.
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
