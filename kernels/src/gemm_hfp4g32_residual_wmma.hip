#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched HFP4-G32 GEMM with fused residual add.
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Sister of gemm_hfq4g256_residual_wmma_k2 for the FP4 (E2M1 + UE8M0
// g32 + FP16 row scale) family. Used for both wo (after attention) and
// w_down (after FFN) projections in the batched prefill path.
//
// HFP4G32 layout differences from HFQ4G256 (same as gemm_qkv_hfp4g32_wmma):
//   - Per row: 16-B header + (K/32) blocks * 17 B (1B UE8M0 + 16B nibbles)
//   - Outer iter on 256-K-element groups (= 8 blocks = 136 B)
//   - Inner 2-tile-per-iter shares one block's UE8M0 + nibbles
//   - Dequant: row_scale_h * 2^(block_e - 127) * E2M1_LUT[nibble]
//
// Output mapping: canonical RDNA3 wave32 WMMA — `acc[j] = C[2*j + (tid>>4)]
// [tid & 15]`. Mirrors the K2 residual variant; the base `_wmma` and `_wmma2`
// HFQ4 residual variants carry known mapping bugs and must NOT be mirrored.
//
// Y semantics: += (residual). Caller must initialize Y to the residual
// stream (e.g. embedding output) before this kernel.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 32B (E2M1 LUT).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfp4g32_residual_wmma(
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

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

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

        for (int kt = 0; kt < 16; kt += 2) {
            const int block_idx = kt >> 1;
            const char* bp = group_base + block_idx * 17;

            const unsigned int e_byte = *(const unsigned char*)bp;
            const float sc_block_f = __builtin_bit_cast(float, e_byte << 23);
            const _Float16 sc_h = row_scale_h * (_Float16)sc_block_f;

            unsigned int pk0a = *(const unsigned int*)(bp + 1);
            unsigned int pk1a = *(const unsigned int*)(bp + 5);
            unsigned int pk0b = *(const unsigned int*)(bp + 9);
            unsigned int pk1b = *(const unsigned int*)(bp + 13);

            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * lut[((pk) >> (sh)) & 0xFu]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // Canonical gfx11 wave32 WMMA mapping: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A = weight rows (M), B = batch X cols (N). Y[batch][m] = Y[batch * M + m].
    const int out_col = batch_start + (tid & 15);  // batch index (N)
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);  // M index
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
