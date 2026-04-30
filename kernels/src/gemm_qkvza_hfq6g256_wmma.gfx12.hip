#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused HFQ6-G256 GEMM for the Qwen3.5
// DeltaNet LinearAttention preamble (qkv + z + beta + alpha projections).
//
// **gfx12 (RDNA4) variant** — sister of gemm_qkvza_hfq6g256_wmma.hip.
// Pure composition of two already-validated patterns:
//   - hfq6 dequant inner loop, validated in gemm_qkv_hfq6g256_wmma.gfx12.hip
//   - 4-output qkv/z/beta/alpha routing, validated in
//     gemm_qkvza_hfq4g256_wmma.gfx12.hip
//
// HFQ6-G256 group layout: 200 bytes / 256 elements
//   [0..4)   f32 scale
//   [4..8)   f32 zero
//   [8..200) 192B packed 6-bit values
//
// gfx12 differences from the gfx11 source:
//   1. _w32 → _w32_gfx12 builtin
//   2. half16_t → half8_t A/B operands; float8_t accumulator unchanged
//   3. K-split via tid >> 4 (each lane carries 8 of 16 K values, =6 bytes)
//   4. C-output mapping: contiguous rows per lane-group:
//        acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// Grid: [ceil((qkv_m + z_m + beta_m + alpha_m)/16), ceil(N/16)].
// Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_hfq6g256_wmma_gfx12(
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
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K4 unroll (issue #65 lever 2): 4 K-tiles per body, all loads
        // up-front. Same lift pattern as the hfq4 family.
        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_8(d0, d1) \
            DQ6(0, (d0), 0);  DQ6(1, (d0), 6);  DQ6(2, (d0), 12); DQ6(3, (d0), 18); \
            DQ6(4, (d1), 0);  DQ6(5, (d1), 6);  DQ6(6, (d1), 12); DQ6(7, (d1), 18)

        for (int kt = 0; kt < 16; kt += 4) {
            const int bb_a = (kt + 0) * 12 + k_grp * 6;
            const int bb_b = (kt + 1) * 12 + k_grp * 6;
            const int bb_c = (kt + 2) * 12 + k_grp * 6;
            const int bb_d = (kt + 3) * 12 + k_grp * 6;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + bb_a);
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + bb_b);
            const unsigned char* dp_c = (const unsigned char*)(gp + 8 + bb_c);
            const unsigned char* dp_d = (const unsigned char*)(gp + 8 + bb_d);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d0c = *(const unsigned int*)(dp_c);
            unsigned int d1c = *(const unsigned int*)(dp_c + 3);
            unsigned int d0d = *(const unsigned int*)(dp_d);
            unsigned int d1d = *(const unsigned int*)(dp_d + 3);

            half8_t b_a = *(const half8_t*)(x_base + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(x_base + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(x_base + (kt + 3) * 16 + k_grp * 8);

            half8_t a_reg;
            DQ6_8(d0a, d1a);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ6_8(d0b, d1b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ6_8(d0c, d1c);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ6_8(d0d, d1d);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ6_8
        #undef DQ6
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
