#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched HFQ4-G256 GEMM with fused residual add.
//
// **gfx12 (RDNA4) variant** — sister of gemm_hfq4g256_residual_wmma_k2.hip
// (gfx11 / RDNA3). Compile with `hipcc --offload-arch=gfx1200` (or 1201).
//
// Differences from the gfx11 K2 kernel — same four levers as the validated
// gfx12 QKV / QKVZA / gate_up ports (PR #56):
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//      (gfx11 uses _w32 without the _gfx12 suffix).
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t
//      (16 fp16 per lane). Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (kABKLane=2) instead of
//      replicated across all 32 lanes (gfx11 kABKLane=1):
//        lane (tid >> 4) = 0  -> carries K = [0..7]  of the 16-K tile
//        lane (tid >> 4) = 1  -> carries K = [8..15] of the 16-K tile
//   4. C-output mapping (rows contiguous, derived from CK kCM0/kCM1PerLane
//      swap; same pattern as the validated qkv / qkvza / gate_up gfx12 ports):
//        gfx11: acc[j] = C[2*j + (tid>>4)][tid & 15]   (rows interleaved)
//        gfx12: acc[j] = C[8*(tid>>4) + j][tid & 15]   (rows contiguous)
//
// Reference (ROCm 7.x):
//   /opt/rocm/include/ck/utility/amd_wmma.hpp
//   /opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp
//   .skills/hipfire-arch-port/wmma-matrix.md
//
// Source kernel pattern: gemm_hfq4g256_residual_wmma_k2.hip — chosen over
// the base `_wmma` and `_wmma2` variants which carry known output-mapping
// bugs (see dispatch.rs comments around `HIPFIRE_WO_WMMA_VARIANT`). The k2
// variant is the gfx11 dispatch default for M >= 8192 (the shape band where
// the residual GEMM dominates 9B prefill), so its mapping is exercised
// continuously by the speed-gate.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_residual_wmma_gfx12(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    // Lane decomposition (gfx12 wave32 WMMA):
    //   m_lane = tid & 15   -> M-row within tile (A) AND N-batch within tile (B)
    //   k_grp  = tid >> 4   -> selects which K half-tile this lane carries
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int safe_row   = (row_start   + m_lane < M)          ? (row_start   + m_lane) : (M - 1);
    const int safe_batch = (batch_start + m_lane < batch_size) ? (batch_start + m_lane) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

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

            half8_t b_a = *(const half8_t*)(xg + k_off_a + k_grp * 8);
            half8_t b_b = *(const half8_t*)(xg + k_off_b + k_grp * 8);
            half8_t b_c = *(const half8_t*)(xg + k_off_c + k_grp * 8);
            half8_t b_d = *(const half8_t*)(xg + k_off_d + k_grp * 8);

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

    // gfx12 wave32 WMMA output mapping (rows contiguous):
    //   acc[j] = C[8*(tid>>4) + j][tid & 15]
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // -> C row = 8*k_grp + j (M index), C col = m_lane (N=batch index).
    // Y is [N × M] row-major: Y[batch][m] = Y[batch*M + m]. Residual semantics
    // (overwrite-or-accumulate) handled by caller's pre-init of Y.
    const int out_col = batch_start + m_lane;  // batch index (N)
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;  // M index
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
