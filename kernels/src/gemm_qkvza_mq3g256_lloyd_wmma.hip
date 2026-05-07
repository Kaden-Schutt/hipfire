#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused MQ3-Lloyd-G256 GEMM for the Qwen3.5/3.6
// DeltaNet LA preamble (qkv + z + beta + alpha projections).
// gfx1100+ only (RDNA3 wave32 WMMA). Phase B1 sibling of
// gemm_mq3g256_lloyd_residual_wmma.hip — same Lloyd codebook handling, same
// K-tile body; only the per-row Y routing differs (4-way fan-out).
//
// Group layout (112 B/group): 16 B fp16 codebook (8 entries) + 96 B 3-bit
// indices (cross-byte packed, identical to HFQ3). Each row has its own
// codebook → 16 codebooks per group staged in LDS (fp16, 256 B/workgroup).
// See plan §"What's new for Lloyd" and Phase A devlog (2026-05-07) for the
// LDS-storage rationale (fp16 won the Phase A bench by 7.15%).
//
// Output convention (matches HFQ3 fused QKVZA): lane (tid & 15) covers
// column (batch) tid&15, with 8 rows per lane via
//   acc[j] = C[2*j + (tid>>4)][tid & 15].
// Y[col][proj_row] = acc[j] (overwrite — caller initialized Y as needed).
//
// Grid: [ceil((qkv_m + z_m + beta_m + alpha_m)/16), ceil(N/16)]. Block: [32].
// LDS: 256 B (cb_lds).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_mq3g256_lloyd_wmma(
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

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_tile_row = tid & 15;
    const int my_row = row_start + my_tile_row;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route this thread's row to the correct A matrix.
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
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + my_tile_row < N)
                           ? (batch_start + my_tile_row) : 0;

    // Cooperative-load mapping for codebook staging. Independent of
    // my_tile_row — load_tile_row spans 0..15 over tid 0..31, with each
    // tile-row fetched from its own A matrix (since rows can straddle
    // projection boundaries).
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 4;    // 0 or 4
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

    __shared__ _Float16 cb_lds[16 * 8];

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load for group g.
        const __half* cb_h_load = (const __half*)(load_row_base + g * 112);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cb_lds[load_tile_row * 8 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 112;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 16);
        const int cb_base = my_tile_row * 8;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const unsigned char* dpa = dp + kt * 6;
            const int k_off_b = (kt + 1) * 16;
            const unsigned char* dpb = dp + (kt + 1) * 6;

            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

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

            half16_t a_reg;
            #define DQ(i, q) a_reg[i] = cb_lds[cb_base + (q)]
            DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
            DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
            DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
            DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

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

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
        __syncthreads();
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // 4-way fan-out routing per output row.
    const int out_col = batch_start + my_tile_row;
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
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
