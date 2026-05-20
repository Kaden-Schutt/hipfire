#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused HFP4-G32 GEMM for the FP4
// FullAttention preamble (Q + K + V projections).
//
// **gfx12 (RDNA4) variant** — sister of gemm_qkv_hfp4g32_wmma.hip
// (gfx11 / RDNA3). Compile this file with
//   hipcc --offload-arch=gfx1200  (or gfx1201).
//
// SCAFFOLD STATUS:
//   The C-output mapping below is the SAME hypothesis used by the
//   gemm_qkv_hfq4g256_wmma.gfx12 sibling (`acc[j] = C[8*(tid>>4) + j]
//   [tid & 15]`). That mapping was hardware-validated for HFQ4 in
//   commit b7ac66a era and verified on gfx1201 (R9700) by the test
//   harness; HFP4G32 inherits identical lane decomposition + WMMA
//   shape so the same mapping holds.
//
// Differences from the gfx11 HFP4 kernel:
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//   2. A and B operands: half8_t per lane (8 fp16) instead of half16_t.
//      Each lane carries half the K-tile; lane-group split:
//        lane (tid >> 4) = 0  -> K = [0..7]  of the 16-K tile
//        lane (tid >> 4) = 1  -> K = [8..15] of the 16-K tile
//   3. K4 unroll: 4 K-tiles per inner iter (= 2 HFP4 blocks). The
//      gfx12 K4 pattern measured 2× on gate_up_hfq4 at 9B pp=128.
//   4. C-output mapping uses kCM0PerLane=1 / kCM1PerLane=8 swap
//      (lane group 0 holds rows 0..7, lane group 1 holds rows 8..15).
//
// HFP4 dequant is identical to the gfx11 variant — only the K-tile
// shape and WMMA builtin change.
//
// Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)].
// Block: [32]. LDS: 32B (E2M1 LUT).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfp4g32_wmma_gfx12(
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

    // 16-entry signed E2M1 LUT in LDS (32B).
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

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

    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)local_row * row_stride;
    const char* row_ptr = hdr + 16;

    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int groups_per_row = K / 256;

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_base = row_ptr + (long long)g * 136;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K4 unroll: 4 K-tiles per inner iter = 2 HFP4 blocks. The two
        // pairs share scales: tiles {0,1} use block kt/2, tiles {2,3}
        // use block kt/2+1.
        for (int kt = 0; kt < 16; kt += 4) {
            const int blk_a = kt >> 1;             // 0,2,4,6  → blocks 0,2,4,6
            const int blk_c = blk_a + 1;           // 1,3,5,7  → blocks 1,3,5,7
            const char* bpa = group_base + blk_a * 17;
            const char* bpc = group_base + blk_c * 17;

            const unsigned int e_a = *(const unsigned char*)bpa;
            const unsigned int e_c = *(const unsigned char*)bpc;
            const float sc_block_a_f = __builtin_bit_cast(float, e_a << 23);
            const float sc_block_c_f = __builtin_bit_cast(float, e_c << 23);
            const _Float16 sc_h_ab = row_scale_h * (_Float16)sc_block_a_f;
            const _Float16 sc_h_cd = row_scale_h * (_Float16)sc_block_c_f;

            // Each K-tile = 16 K-elements = 16 nibbles = 8 bytes = 2 uint.
            // gfx12 lane-group split: lane group k_grp carries K[8*k_grp .. 8*k_grp+7],
            // so it reads the corresponding 4-byte half of the tile's nibble payload.
            // Tile A = block_a low half  (bp + 1 .. bp + 8)
            // Tile B = block_a high half (bp + 9 .. bp + 16)
            // Tile C = block_c low half
            // Tile D = block_c high half
            unsigned int pka = *(const unsigned int*)(bpa + 1 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(bpa + 9 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(bpc + 1 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(bpc + 9 + k_grp * 4);

            const int k_off_a = (kt + 0) * 16 + k_grp * 8;
            const int k_off_b = (kt + 1) * 16 + k_grp * 8;
            const int k_off_c = (kt + 2) * 16 + k_grp * 8;
            const int k_off_d = (kt + 3) * 16 + k_grp * 8;

            half8_t b_a = *(const half8_t*)(x_base + k_off_a);
            half8_t b_b = *(const half8_t*)(x_base + k_off_b);
            half8_t b_c = *(const half8_t*)(x_base + k_off_c);
            half8_t b_d = *(const half8_t*)(x_base + k_off_d);

            half8_t a_reg;

            #define DQ8(pk, sc) \
                a_reg[0] = (sc) * lut[((pk) >>  0) & 0xFu]; \
                a_reg[1] = (sc) * lut[((pk) >>  4) & 0xFu]; \
                a_reg[2] = (sc) * lut[((pk) >>  8) & 0xFu]; \
                a_reg[3] = (sc) * lut[((pk) >> 12) & 0xFu]; \
                a_reg[4] = (sc) * lut[((pk) >> 16) & 0xFu]; \
                a_reg[5] = (sc) * lut[((pk) >> 20) & 0xFu]; \
                a_reg[6] = (sc) * lut[((pk) >> 24) & 0xFu]; \
                a_reg[7] = (sc) * lut[((pk) >> 28) & 0xFu]

            DQ8(pka, sc_h_ab);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb, sc_h_ab);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc, sc_h_cd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd, sc_h_cd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);

            #undef DQ8
        }
    }

    // gfx12 wave32 WMMA C-mapping (matches gemm_qkv_hfq4g256_wmma.gfx12):
    //   kCMLane=2, kCM0PerLane=1, kCM1PerLane=8
    //   acc[j] = C[8*(tid>>4) + j][tid & 15]
    // Lane group 0 holds output rows 0..7, lane group 1 rows 8..15.
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
