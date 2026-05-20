#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused HFP4-G32 GEMM for the FP4
// FullAttention preamble (Q + K + V projections).
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Sister of gemm_qkv_hfq4g256_wmma. Same WMMA shape (16x16x16 f16),
// same lane decomposition, same C-output mapping. Differences:
//
//   1. Per-row layout — HFP4G32 has a 16-B row header (FP16 row_scale_a
//      first 2 bytes) followed by (K/32) * 17-byte blocks (1B UE8M0 +
//      16B nibbles). HFQ4-G256 has 136-byte groups (4B fp32 sc + 4B fp32
//      zp + 128B nibbles, one group per 256 K-elements).
//
//   2. Dequant arithmetic —
//        HFQ4: value = sc * (float)nibble + zp
//        HFP4: value = row_scale_a * 2^(block_e - 127) * E2M1_LUT[nibble]
//      The row_scale_a multiply hoists ONCE per row before the K loop.
//      The UE8M0 block scale is pure FP32 bit construction:
//      `__builtin_bit_cast(float, e_byte << 23)` materializes 2^(e-127)
//      directly (mirrors gemv_hfp4g32.hip:106).
//
//   3. K-tile loop — HFP4G32's 32-element block aligns with a pair of
//      WMMA 16-K tiles. Iterating 2 tiles per loop body keeps one block
//      load (block_e + 16B nibbles) shared across both tiles, saving
//      half the UE8M0 reads vs a per-tile fetch.
//
// Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)].
// Block: [32].  LDS: 32B (E2M1 LUT).
//
// Each 16-row tile may straddle projection boundaries.  Per-thread output
// routing resolves which Y pointer each row writes to.
// Overwrite semantics (y = acc, not +=).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfp4g32_wmma(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const _Float16* __restrict__ X,      // [N x K], FP16
    float*         __restrict__ Y_q,     // [N x q_m]
    float*         __restrict__ Y_k,     // [N x k_m]
    float*         __restrict__ Y_v,     // [N x v_m]
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m = q_m + k_m + v_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // 16-entry signed E2M1 magnitude LUT in LDS (32B). Lanes 0..15 init.
    // OCP E2M1: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Locked to spec.
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's row.
    const char* A;
    int local_row;
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
    }

    // Per row: 16-B header + (K/32) * 17-B blocks.
    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)local_row * row_stride;
    const char* row_ptr = hdr + 16;     // first block

    // Hoist FP16 row_scale_a — first 2 bytes of header.
    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);

    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Outer loop: each "group" covers 256 K-elements = 8 HFP4 blocks = 136 B.
    // This matches HFQ4G256's outer iteration shape and lets the inner
    // 2-tile-per-iter unroll consume one block per pair.
    const int groups_per_row = K / 256;

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_base = row_ptr + (long long)g * 136;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // 16 K-tiles per group, processed 2 at a time.
        // Each pair shares one HFP4 block (block_idx = kt >> 1):
        //   tile A = nibbles[0..8]   (low 16 K-elements of the block)
        //   tile B = nibbles[8..16]  (high 16 K-elements of the block)
        for (int kt = 0; kt < 16; kt += 2) {
            const int block_idx = kt >> 1;             // 0..7
            const char* bp = group_base + block_idx * 17;

            // UE8M0 block scale → FP32 bit-construct → FP16. One mul with
            // row_scale_h gives the per-tile dequant scale.
            const unsigned int e_byte = *(const unsigned char*)bp;
            const float sc_block_f = __builtin_bit_cast(float, e_byte << 23);
            const _Float16 sc_h = row_scale_h * (_Float16)sc_block_f;

            // 16 packed nibbles per tile (= 8 bytes = 2 uints).
            unsigned int pk0a = *(const unsigned int*)(bp + 1);
            unsigned int pk1a = *(const unsigned int*)(bp + 5);
            unsigned int pk0b = *(const unsigned int*)(bp + 9);
            unsigned int pk1b = *(const unsigned int*)(bp + 13);

            // Load both X tiles early — consecutive addresses, prefetcher-friendly.
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            // Dequant tile A.
            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * lut[((pk) >> (sh)) & 0xFu]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // Dequant tile B (same sc_h — same block).
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
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
