#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MFMA-direct HFQ4G256 GEMM with fused residual add for gfx942 (MI300X CDNA3).
//
// Replaces the prefill path's `hfq4g256_dequantize_to_f16 → rocBLAS Tensile`
// combo with a single MFMA-direct kernel that:
//   * Reads HFQ4 weights, unpacks to FP16 in registers (no shadow buffer)
//   * Accumulates via V_MFMA_F32_16x16x16_F16
//   * Writes FP32 output (residual add via +=)
//
// Layouts (matching `gemm_hfq4g256_residual_fp16_wave64`):
//   A:  [M, K] HFQ4 weight (M = output features, K = input dim)
//   X:  [batch_size, K] FP16, row-major (pre-converted by ensure_fp16_x)
//   Y:  [batch_size, M] FP32, row-major (residual add via +=)
//
// Geometry:
//   block = [64, 1, 1]                     1 wave64 per WG, one 16×16 output tile
//   grid  = [(M+15)/16, (batch+15)/16, 1]  one WG per (output-row-tile, batch-tile)
//   LDS   = 0 (all in registers)
//
// MFMA F32_16x16x16_F16 layout (verified correct via mfma_test.cpp; channel-
// test against scalar HFQ4 reference passed at max_rel_err = 2e-5):
//   A operand: lane l holds a[r] = A[m = l%16, k = (l/16)*4 + r]
//   B operand: lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]
//   D output: lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]
//
// For this kernel, MFMA's "m" maps to hipfire output dim (M parameter), and
// MFMA's "n" maps to the token/batch dim (batch_size parameter).
//
// HFQ4G256 group format (136 bytes/group): 4B scale (f32) + 4B zp (f32) +
// 128B nibbles = 256 4-bit values (low nibble first within each byte).

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    vfloat4 __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(64)
void gemm_hfq4g256_residual_mfma_gfx942(
    const char*  __restrict__ A,     // [M, K] HFQ4 weight
    const half_t* __restrict__ X,    // [batch_size, K] FP16 activations, row-major
    float*       __restrict__ Y,     // [batch_size, M] FP32 output, row-major
    int M, int K, int batch_size
) {
    const int lane     = threadIdx.x;
    const int n_lane   = lane % 16;       // MFMA n-index within tile
    const int strip    = lane / 16;       // 0..3
    const int m_offset = blockIdx.x * 16; // output-row tile start
    const int b_offset = blockIdx.y * 16; // batch tile start

    // Each lane services output row (m_offset + n_lane)
    const int my_m = m_offset + n_lane;
    if (my_m >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)my_m * groups_per_row * 136;

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // The token this lane's B operand serves
    const int my_b = b_offset + n_lane;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        half_t sc_h = (half_t)sc;
        half_t zp_h = (half_t)zp;

        #pragma unroll
        for (int ki = 0; ki < 256; ki += 16) {
            // 4 K-elements per lane (strip selects which 4 of 16):
            // K-pos = ki + strip*4 + r, byte offset = (ki + strip*4) / 2.
            const unsigned short pk2 = *(const unsigned short*)(gp + 8 + ki/2 + strip*2);
            const unsigned int n0 = (pk2 >>  0) & 0xFu;
            const unsigned int n1 = (pk2 >>  4) & 0xFu;
            const unsigned int n2 = (pk2 >>  8) & 0xFu;
            const unsigned int n3 = (pk2 >> 12) & 0xFu;

            half4 a_reg;
            a_reg[0] = sc_h * (half_t)n0 + zp_h;
            a_reg[1] = sc_h * (half_t)n1 + zp_h;
            a_reg[2] = sc_h * (half_t)n2 + zp_h;
            a_reg[3] = sc_h * (half_t)n3 + zp_h;

            // X is [batch_size, K] row-major: X[my_b, k] = X[my_b * K + k]
            const int k_start = g * 256 + ki + strip * 4;
            half4 b_reg;
            if (my_b < batch_size) {
                b_reg[0] = X[(long long)my_b * K + (k_start + 0)];
                b_reg[1] = X[(long long)my_b * K + (k_start + 1)];
                b_reg[2] = X[(long long)my_b * K + (k_start + 2)];
                b_reg[3] = X[(long long)my_b * K + (k_start + 3)];
            } else {
                b_reg[0] = (half_t)0; b_reg[1] = (half_t)0;
                b_reg[2] = (half_t)0; b_reg[3] = (half_t)0;
            }

            c_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_acc, 0, 0, 0);
        }
    }

    // Output write: lane c[r] = D[m = strip*4 + r within tile, n = n_lane within tile]
    // Y is [batch_size, M] row-major: Y[batch, m] = Y[batch * M + m]
    // Lane (strip, n_lane): writes c_acc[r] to Y[b_offset + n_lane, m_offset + strip*4 + r]
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            Y[(long long)gb * M + gm] += c_acc[r];
        }
    }
}
