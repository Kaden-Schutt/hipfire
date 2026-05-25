#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused MQ4-Lloyd-G256 GEMM for the Qwen3.5/3.6
// DeltaNet LA preamble (qkv + z + beta + alpha projections).
// gfx1100+ only (RDNA3 wave32 WMMA). Phase B1 sibling of
// gemm_mq4g256_lloyd_residual_wmma.hip — same Lloyd codebook handling, same
// K-tile body; only the per-row Y routing differs (4-way fan-out).
//
// Group layout (160 B/group): 16 B fp16 codebook (16 entries) + 96 B 3-bit
// indices (cross-byte packed, identical to HFQ4). Each row has its own
// codebook → 16 codebooks per group staged in LDS (fp16, 512 B/workgroup).
// See plan §"What's new for Lloyd" and Phase A devlog (2026-05-07) for the
// LDS-storage rationale (fp16 won the Phase A bench by 7.15%).
//
// Output convention (matches HFQ4 fused QKVZA): lane (tid & 15) covers
// column (batch) tid&15, with 8 rows per lane via
//   acc[j] = C[2*j + (tid>>4)][tid & 15].
// Y[col][proj_row] = acc[j] (overwrite — caller initialized Y as needed).
//
// Grid: [ceil((qkv_m + z_m + beta_m + alpha_m)/16), ceil(N/16)]. Block: [32].
// LDS: 512 B (cb_lds).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_mq4g256_lloyd_wmma(
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
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + my_tile_row < N)
                           ? (batch_start + my_tile_row) : 0;

    // Cooperative-load mapping for codebook staging. Independent of
    // my_tile_row — load_tile_row spans 0..15 over tid 0..31, with each
    // tile-row fetched from its own A matrix (since rows can straddle
    // projection boundaries).
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 8;    // 0 or 4
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

    __shared__ _Float16 cb_lds[16 * 16];

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load for group g.
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 32);
        const int cb_base = my_tile_row * 16;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const unsigned char* dpa = dp + kt * 8;
            const int k_off_b = (kt + 1) * 16;
            const unsigned char* dpb = dp + (kt + 1) * 8;

            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            unsigned int pk0a = *(const unsigned int*)dpa;
            unsigned int pk1a = *(const unsigned int*)(dpa + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            unsigned int pk0b = *(const unsigned int*)dpb;
            unsigned int pk1b = *(const unsigned int*)(dpb + 4);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
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
