#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused HFQ4-G256 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections).
//
// **gfx12 (RDNA4) variant** — sister kernel to gemm_qkv_hfq4g256_wmma.hip
// (which targets gfx11 / RDNA3). Compile this file with
//   hipcc --offload-arch=gfx1200  (or gfx1201, ...).
//
// SCAFFOLD STATUS (2026-04-26):
//   This file is NOT yet wired into dispatch.rs. It compiles for gfx12
//   targets and is intended as the canonical pattern reference for the
//   five remaining gfx12 WMMA ports (qkvza-hfq4, gate_up-hfq4, plus
//   the three hfq6 variants). Runtime correctness must be validated on
//   real RDNA4 hardware (9070 XT / R9700) via test_kernels before any
//   dispatch.rs site is allowed to route here. The C-output mapping
//   below is a HYPOTHESIS derived from CK trait math and has NOT been
//   confirmed against silicon — see the channel-test note at the
//   write-back loop. See `.skills/hipfire-arch-port/` for the port
//   workflow.
//
// Differences from the gfx11 kernel:
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//      (gfx11 uses _w32 without the _gfx12 suffix).
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t
//      (16 fp16 per lane). Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (kABKLane=2) instead
//      of replicated across all 32 lanes (gfx11 kABKLane=1):
//        lane (tid >> 4) = 0  -> carries K = [0..7]  of the 16-K tile
//        lane (tid >> 4) = 1  -> carries K = [8..15] of the 16-K tile
//      Each lane therefore loads HALF as much per WMMA tile as gfx11.
//   4. C-output mapping (HYPOTHESIS — see below):
//        gfx11 (validated, commit b7ac66a):
//          acc[j] = C[2*j + (tid>>4)][tid & 15]
//        gfx12 (UNVERIFIED, derived from CK kCM0/kCM1PerLane swap):
//          acc[j] = C[8*(tid>>4) + j][tid & 15]
//
// Reference (ROCm 7.x):
//   /opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp
//   /opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp
//   /opt/rocm/include/ck/utility/amd_wmma.hpp
//
// Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)].
// Block: [32]. LDS: 0.
//
// Each 16-row tile may straddle projection boundaries. Per-thread output
// routing resolves which Y pointer each row writes to.
// Overwrite semantics (y = acc, not +=).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfq4g256_wmma_gfx12(
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

    // Lane decomposition (gfx12 wave32 WMMA):
    //   m_lane = tid & 15   (0..15)  -> row index within the 16-row tile
    //   k_grp  = tid >> 4   (0 or 1) -> which K half-tile this lane carries
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
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

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    // B operand indexes BATCH (N) dimension by m_lane (CK kBNLane=16).
    // In gfx11 each lane held the FULL 16-K of its batch row; in gfx12 each
    // lane holds only the 8-K slice corresponding to its k_grp.
    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K4 unroll (issue #65 lever 2): 4 K-tiles per loop body, all loads
        // up-front so the compiler can deep-pipeline the WMMA chain.
        // 4 iterations cover K=256. RDNA4 measured 2× on the gate_up kernel
        // at 9B pp=128.
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

    // --- Output ---
    //
    // gfx12 wave32 WMMA C-mapping HYPOTHESIS — DO NOT TRUST WITHOUT
    // CHANNEL-TEST ON REAL HARDWARE. Derived from the CK trait swap
    // (warp_gemm_attribute_wmma_impl_base_traits.hpp):
    //
    //   gfx11: kCMLane=2, kCM0PerLane=8, kCM1PerLane=1
    //          -> acc[j] = C[2*j + (tid>>4)][tid & 15]  (rows interleaved)
    //   gfx12: kCMLane=2, kCM0PerLane=1, kCM1PerLane=8
    //          -> acc[j] = C[8*(tid>>4) + j][tid & 15]  (rows contiguous)
    //
    // i.e. lane group 0 holds output rows 0..7, lane group 1 rows 8..15.
    // The gfx11 mapping was silently wrong for ~6 weeks before being fixed
    // in commit b7ac66a — assume the same risk here. Validation recipe
    // (from that fix): add `if (blockIdx.x == 0 && blockIdx.y == 0)
    // printf("tid=%d j=%d row=%d col=%d acc=%f\\n", tid, j, out_row,
    // out_col, acc[j]);` and compare against a CPU reference for a small
    // golden case. Adjust the (j, k_grp) -> out_row formula until it matches.
    const int out_col = batch_start + m_lane;  // batch index
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
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
