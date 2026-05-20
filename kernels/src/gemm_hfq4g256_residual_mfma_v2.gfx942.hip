#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MFMA-direct HFQ4G256 GEMM v2 for gfx942 (MI300X CDNA3) — 32x32 output
// tile per WG via 4 wave64s × 16x16 MFMA tile each.
//
// v1 (1 wave64/WG, 16x16 tile) shipped at 80% of rocBLAS perf. v2 changes:
//   * 4 wave64s/WG (block = [256,1,1])
//   * Each wave64 computes its own 16x16 sub-tile of a 32x32 WG output:
//     - wave 0: WG_out[ 0..15,  0..15]
//     - wave 1: WG_out[ 0..15, 16..31]
//     - wave 2: WG_out[16..31,  0..15]
//     - wave 3: WG_out[16..31, 16..31]
//   * Grid = [(M+31)/32, (batch+31)/32, 1] — 16x fewer WGs vs v1
//   * Higher wave-occupancy per CU (4 waves/WG → MFMA latency hides
//     across waves)
//
// Math byte-identical to v1 per wave (same MFMA F32_16x16x16_F16 chain).

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    vfloat4 __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(256)
void gemm_hfq4g256_residual_mfma_v2_gfx942(
    const char*  __restrict__ A,     // [M, K] HFQ4
    const half_t* __restrict__ X,    // [batch_size, K] FP16
    float*       __restrict__ Y,     // [batch_size, M] FP32
    int M, int K, int batch_size
) {
    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..3 — which wave64 in the WG
    const int lane_in_w = tid & 63;             // 0..63 within wave
    const int n_lane    = lane_in_w & 15;       // lane%16 — MFMA n/m index
    const int strip     = lane_in_w >> 4;       // lane/16, 0..3 — MFMA k-block / output-row-block

    // Per-wave 16x16 sub-tile offset within the 32x32 WG output:
    const int wave_m_off = (wave_id & 2) ? 16 : 0;   // wave 0,1: m=0..15; wave 2,3: m=16..31
    const int wave_n_off = (wave_id & 1) ? 16 : 0;   // wave 0,2: n=0..15; wave 1,3: n=16..31

    const int m_offset = blockIdx.x * 32 + wave_m_off;
    const int b_offset = blockIdx.y * 32 + wave_n_off;

    const int my_m = m_offset + n_lane;
    if (my_m >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)my_m * groups_per_row * 136;

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    const int my_b = b_offset + n_lane;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        half_t sc_h = (half_t)sc;
        half_t zp_h = (half_t)zp;

        #pragma unroll
        for (int ki = 0; ki < 256; ki += 16) {
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

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            Y[(long long)gb * M + gm] += c_acc[r];
        }
    }
}
