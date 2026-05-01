#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched HFQ3-G256 GEMM with fused residual add.
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Sister of `gemm_hfq4g256_residual_wmma_k2.hip` — same WMMA shape and
// lane decomposition; only the inner K-tile unpack differs (3-bit
// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
// instead of 136. K2 unroll with software pipelining (mirrors the
// production MQ4 _k2 variant; the base MQ4 _wmma variant has a
// documented output-mapping bug — `wmma — base WMMA, output-mapping
// bug — debug only` per dispatch.rs comment).
//
// Output convention (matches qkvza / gate_up): lane (tid & 15) covers
// column (batch) tid&15, with 8 rows per lane via
//   acc[j] = C[2*j + (tid>>4)][tid & 15].
// Y += acc[j] (residual add — caller must initialize Y with the
// residual stream before launching).
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq3g256_residual_wmma(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 104;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            const unsigned char* dpa = dp + kt * 6;

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            const unsigned char* dpb = dp + (kt + 1) * 6;

            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            unsigned int a0 = dpa[0], a1 = dpa[1], a2 = dpa[2];
            unsigned int a3 = dpa[3], a4 = dpa[4], a5 = dpa[5];

            unsigned int q0 = a0 & 7u;
            unsigned int q1 = (a0 >> 3) & 7u;
            unsigned int q2 = ((a0 >> 6) | (a1 << 2)) & 7u;
            unsigned int q3 = (a1 >> 1) & 7u;
            unsigned int q4 = (a1 >> 4) & 7u;
            unsigned int q5 = ((a1 >> 7) | (a2 << 1)) & 7u;
            unsigned int q6 = (a2 >> 2) & 7u;
            unsigned int q7 = (a2 >> 5) & 7u;
            unsigned int q8  = a3 & 7u;
            unsigned int q9  = (a3 >> 3) & 7u;
            unsigned int q10 = ((a3 >> 6) | (a4 << 2)) & 7u;
            unsigned int q11 = (a4 >> 1) & 7u;
            unsigned int q12 = (a4 >> 4) & 7u;
            unsigned int q13 = ((a4 >> 7) | (a5 << 1)) & 7u;
            unsigned int q14 = (a5 >> 2) & 7u;
            unsigned int q15 = (a5 >> 5) & 7u;

            half16_t a_a;
            #define DQ(i, q) a_a[i] = sc_h * (_Float16)(float)(q) + zp_h
            DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
            DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
            DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
            DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            unsigned int B0 = dpb[0], B1 = dpb[1], B2 = dpb[2];
            unsigned int B3 = dpb[3], B4 = dpb[4], B5 = dpb[5];

            unsigned int Q0 = B0 & 7u;
            unsigned int Q1 = (B0 >> 3) & 7u;
            unsigned int Q2 = ((B0 >> 6) | (B1 << 2)) & 7u;
            unsigned int Q3 = (B1 >> 1) & 7u;
            unsigned int Q4 = (B1 >> 4) & 7u;
            unsigned int Q5 = ((B1 >> 7) | (B2 << 1)) & 7u;
            unsigned int Q6 = (B2 >> 2) & 7u;
            unsigned int Q7 = (B2 >> 5) & 7u;
            unsigned int Q8  = B3 & 7u;
            unsigned int Q9  = (B3 >> 3) & 7u;
            unsigned int Q10 = ((B3 >> 6) | (B4 << 2)) & 7u;
            unsigned int Q11 = (B4 >> 1) & 7u;
            unsigned int Q12 = (B4 >> 4) & 7u;
            unsigned int Q13 = ((B4 >> 7) | (B5 << 1)) & 7u;
            unsigned int Q14 = (B5 >> 2) & 7u;
            unsigned int Q15 = (B5 >> 5) & 7u;

            DQ(0, Q0);   DQ(1, Q1);   DQ(2, Q2);   DQ(3, Q3);
            DQ(4, Q4);   DQ(5, Q5);   DQ(6, Q6);   DQ(7, Q7);
            DQ(8, Q8);   DQ(9, Q9);   DQ(10, Q10); DQ(11, Q11);
            DQ(12, Q12); DQ(13, Q13); DQ(14, Q14); DQ(15, Q15);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    const int out_col = batch_start + (tid & 15);
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
