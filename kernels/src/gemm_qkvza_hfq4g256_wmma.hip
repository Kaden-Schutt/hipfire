#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused HFQ4-G256 GEMM for the Qwen3.5
// DeltaNet LA preamble (qkv + z + beta + alpha projections).
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// K2 variant: 2x K-tile unroll with software pipelining.
// Processes 2 K-tiles (32 K-elements) per loop body. Loads weights+X for
// tile B while WMMA on tile A is executing, enabling the compiler to
// overlap loads with compute (target: s_waitcnt vmcnt(2)).
//
// Same structure as gemm_qkvza_hfq4g256 but uses WMMA 16x16x16 tiles
// instead of scalar FMA.  X is pre-converted to FP16.
//
// Grid: [ceil((qkv_m + z_m + beta_m + alpha_m)/16), ceil(N/16)].
// Block: [32].  LDS: 0.
//
// Each 16-row tile may straddle projection boundaries.  Per-thread output
// routing resolves which Y pointer each row writes to.
// Overwrite semantics (y = acc, not +=).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// __restrict__ stripped: hipcc/clang (ROCm 7.2) miscompile on gfx1100 —
// leaving restrict on pointer params of batched-path kernels produces
// deterministic-wrong output at LA layer 0 and nondeterministic output
// at layers ≥ 1 in prefill (diagnosed via prefill_state_diff harness).
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_hfq4g256_wmma(
    const char*    A_qkv,
    const char*    A_z,
    const char*    A_beta,
    const char*    A_alpha,
    const _Float16* X,         // [N x K], FP16
    float*         Y_qkv,      // [N x qkv_m]
    float*         Y_z,        // [N x z_m]
    float*         Y_beta,     // [N x beta_m]
    float*         Y_alpha,    // [N x alpha_m]
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
    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's row.
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
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        // Keep scale/zero-point in FP32. Dequant done in FP32 and narrowed to
        // FP16 only at the WMMA input boundary. FP16 multiply-add over
        // K=4096 accumulates enough rounding error to flip LM-head argmax
        // into degenerate attractors — same fix PR #28 applied to the
        // sibling _wmma_k2 / qkv_wmma / gate_up_wmma kernels.
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            // Load both X tiles early (consecutive addresses — prefetcher-friendly)
            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            // Dequant tile A (FP32 arithmetic, narrow at last step)
            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = (_Float16)(sc * (float)(((pk) >> (sh)) & 0xFu) + zp)
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            // WMMA A — b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // Dequant tile B (reuse a_reg register)
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            // WMMA B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // --- Output ---
    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // So row = 2*j + (tid>>4), col (batch) = tid & 15.
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                // Route this row to the correct output matrix.
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
