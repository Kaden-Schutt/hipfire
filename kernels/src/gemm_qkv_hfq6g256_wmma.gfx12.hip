#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused HFQ6-G256 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections).
//
// **gfx12 (RDNA4) variant** — sister of gemm_qkv_hfq6g256_wmma.hip
// (gfx11 / RDNA3). Compile with `hipcc --offload-arch=gfx1200` (or 1201).
//
// HFQ6-G256 group layout (unchanged from gfx11): 200 bytes / 256 elements
//   bytes [0..4)   f32 scale
//   bytes [4..8)   f32 zero
//   bytes [8..200) 192B packed 6-bit values
//   Per K-tile (16 values): 12 bytes (16 × 6/8). Per K half-tile (8
//   values): 6 bytes.
//
// gfx12 differences from the gfx11 kernel — same recipe as
// gemm_qkv_hfq4g256_wmma.gfx12.hip (validated on R9700 / gfx1201):
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12.
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t.
//      Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (k_grp = tid >> 4):
//        k_grp = 0  -> K = [0..7]  of the 16-K tile
//        k_grp = 1  -> K = [8..15] of the 16-K tile
//      Each lane reads 6 bytes (8 × 6-bit) of weights per K-tile, vs the
//      12 bytes (16 × 6-bit) the gfx11 lane reads.
//   4. C-output mapping (validated via the QKV hfq4 scaffold's
//      channel-test on R9700):
//        acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// HFQ6 nibble-extract trick: 8 packed 6-bit values = 48 bits = 6 bytes.
// We do two unaligned uint reads at byte offsets {0, 3} (relative to the
// half-tile base) so each uint contains 4 cleanly bit-aligned 6-bit
// values at positions [0, 6, 12, 18]. The gfx11 kernel does the same
// thing four times (offsets {0, 3, 6, 9}); the gfx12 kernel only does
// the half corresponding to its k_grp:
//   k_grp = 0 -> base = byte_off + 0   (covers values 0..7)
//   k_grp = 1 -> base = byte_off + 6   (covers values 8..15)
//
// Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfq6g256_wmma_gfx12(
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
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K4 unroll (issue #65 lever 2): 4 K-tiles per body, all loads
        // up-front. Same lift pattern as the hfq4 family.
        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_8(d0, d1) \
            DQ6(0, (d0), 0);  DQ6(1, (d0), 6);  DQ6(2, (d0), 12); DQ6(3, (d0), 18); \
            DQ6(4, (d1), 0);  DQ6(5, (d1), 6);  DQ6(6, (d1), 12); DQ6(7, (d1), 18)

        for (int kt = 0; kt < 16; kt += 4) {
            // 4 weight half-tiles, each 6 bytes (8 × 6-bit).
            const int bb_a = (kt + 0) * 12 + k_grp * 6;
            const int bb_b = (kt + 1) * 12 + k_grp * 6;
            const int bb_c = (kt + 2) * 12 + k_grp * 6;
            const int bb_d = (kt + 3) * 12 + k_grp * 6;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + bb_a);
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + bb_b);
            const unsigned char* dp_c = (const unsigned char*)(gp + 8 + bb_c);
            const unsigned char* dp_d = (const unsigned char*)(gp + 8 + bb_d);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d0c = *(const unsigned int*)(dp_c);
            unsigned int d1c = *(const unsigned int*)(dp_c + 3);
            unsigned int d0d = *(const unsigned int*)(dp_d);
            unsigned int d1d = *(const unsigned int*)(dp_d + 3);

            half8_t b_a = *(const half8_t*)(x_base + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(x_base + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(x_base + (kt + 3) * 16 + k_grp * 8);

            half8_t a_reg;
            DQ6_8(d0a, d1a);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ6_8(d0b, d1b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ6_8(d0c, d1c);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ6_8(d0d, d1d);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ6_8
        #undef DQ6
    }

    // --- Output ---
    // gfx12 wave32 WMMA C-mapping: contiguous rows per lane-group
    // (validated on R9700 in the QKV hfq4 scaffold's channel-test).
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
