#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP8-WMMA-accelerated batched 3-way fused HFP4-G32 GEMM for the FP4
// FullAttention preamble (Q + K + V projections) — gfx12 (RDNA4) only.
//
// STATUS: opt-in research kernel, default-off (gate via HIPFIRE_FP8_WMMA=1
// in `gemm_qkv_hfp4g32` dispatch). Correctness PASS at NRMSE ~2.7% vs
// the FP16-WMMA sibling. Perf at K=2048, gfx1201:
//   N=1..64    FP8 is ~0.73-0.93x FP16 (consistently slower)
//   N=256-512  FP8 is ~0.86-0.94x FP16
//   N=2048     FP8 is ~0.82-1.26x FP16 (high variance from DPM noise,
//              mean ~1.05x — at-parity to marginal win)
//
// Architectural ceiling: the raw FP8 WMMA throughput on gfx1201 is
// 1.87x FP16-WMMA (microbenched). But HFP4G32's per-32-block UE8M0
// scale forces post-WMMA `acc[j] += partial[j] * scale[j]` chains on
// the F32 accumulator (FP16 path bakes scale into the half operand
// pre-WMMA so no post-mul is needed). The FMA chain + cross-lane
// shuffle to gather per-output-row scales costs roughly what the
// WMMA-throughput advantage delivers, netting close to parity.
//
// Lifting this would require either (a) baking row_scale * UE8M0 into
// the FP8 weight at dequant time (risks E4M3 saturation if UE8M0 is
// large) or (b) a new quant format with row-scale-only (no per-block
// UE8M0). Both are out of scope for the HFP4G32-storage-preserving v2
// effort that prompted this kernel.
//
// Sister kernel of gemm_qkv_hfp4g32_wmma.gfx12.hip. Same C-mapping,
// same lane-decomposition, same K4 unroll over 256-element groups.
// Differences from the FP16 sibling:
//   1. WMMA: __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12.
//   2. Weight LUT holds E2M1 values pre-converted to E4M3 bytes (no
//      scale baked).
//   3. Activation X is pre-packed by `pack_f32_to_fp8_gfx12` (F32 ->
//      E4M3 via cvt_pk_fp8_f32, unscaled — post-RMSNorm activations
//      are bounded well below E4M3 saturation). The in-kernel cvt
//      variant measured 0.7-0.86x FP16; moving cvt to the pre-pass
//      kernel recovered ~10pp. Both result + Y_q/k/v stay F32.
//   4. row_scale * UE8M0_block scale is applied AFTER each pair of
//      WMMAs sharing a UE8M0 block. Per-output-row scales gathered via
//      lane-shuffle (8 shfl per pair) since each lane's A-row is not
//      its output rows under WMMA C-mapping.
//
// Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)].
// Block: [32]. LDS: 16B (E4M3 LUT).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef int      __attribute__((ext_vector_type(2))) v2i_t;
typedef float    __attribute__((ext_vector_type(8))) v8f_t;

// E2M1 (FP4) value lookup as E4M3 (FP8) bytes.
// E4M3 byte layout: sign(1) | exp(4, bias 7) | mant(3).
// E2M1 values: +0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0 and signed negatives.
__device__ static constexpr unsigned char E2M1_AS_E4M3[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
};

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_hfp4g32_wmma_fp8_gfx12(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const unsigned char* __restrict__ X_fp8,  // [N x K], FP8 (E4M3) packed
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

    // 16-byte E4M3 LUT in LDS.
    __shared__ unsigned char lut[16];
    if (tid < 16) lut[tid] = E2M1_AS_E4M3[tid];
    __syncthreads();

    // Lane decomposition (gfx12 wave32 WMMA, identical to FP16 sibling).
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's A-row.
    const char* A;
    int local_row;
    if (safe_row < q_m) { A = A_q;  local_row = safe_row; }
    else if (safe_row < q_m + k_m) { A = A_k;  local_row = safe_row - q_m; }
    else { A = A_v;  local_row = safe_row - q_m - k_m; }

    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)local_row * row_stride;
    const char* row_ptr = hdr + 16;

    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);
    const float row_scale_self = (float)row_scale_h;

    // Gather the 8 per-output-row `row_scale` values for this lane's
    // output mapping (rows 8*k_grp .. 8*k_grp+7 under gfx12 C-mapping).
    // Source lane: m_lane = (8*k_grp + j) lives at lane index (8*k_grp + j)
    // in lane group 0 (lanes 0..15). Both lane groups read from lane group 0.
    float out_row_scales[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int src_lane = 8 * k_grp + j;
        out_row_scales[j] = __shfl(row_scale_self, src_lane);
    }

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    v8f_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int groups_per_row = K / 256;

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_base = row_ptr + (long long)g * 136;
        const unsigned char* x_base = X_fp8 + (long long)safe_batch * K + g * 256;

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 4) {
            const int blk_a = kt >> 1;
            const int blk_c = blk_a + 1;
            const char* bpa = group_base + blk_a * 17;
            const char* bpc = group_base + blk_c * 17;

            // Per-lane UE8M0 byte (= its own A-row's UE8M0 for current block).
            const unsigned int e_a_self = *(const unsigned char*)bpa;
            const unsigned int e_c_self = *(const unsigned char*)bpc;
            const float ue8m0_a_self = __builtin_bit_cast(float, e_a_self << 23);
            const float ue8m0_c_self = __builtin_bit_cast(float, e_c_self << 23);

            // Combined per-output-row scale gathered via shuffle.
            float scale_a[8], scale_c[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int src_lane = 8 * k_grp + j;
                scale_a[j] = out_row_scales[j] * __shfl(ue8m0_a_self, src_lane);
                scale_c[j] = out_row_scales[j] * __shfl(ue8m0_c_self, src_lane);
            }

            // Each tile = 16 K-elements = 16 nibbles = 8 bytes; lane group
            // k_grp carries K[8*k_grp .. 8*k_grp+7] = 4 bytes worth.
            unsigned int pka = *(const unsigned int*)(bpa + 1 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(bpa + 9 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(bpc + 1 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(bpc + 9 + k_grp * 4);

            const int k_off_a = (kt + 0) * 16 + k_grp * 8;
            const int k_off_b = (kt + 1) * 16 + k_grp * 8;
            const int k_off_c = (kt + 2) * 16 + k_grp * 8;
            const int k_off_d = (kt + 3) * 16 + k_grp * 8;

            // FP8 activation: 8 bytes per tile per lane (K[8*k_grp..8*k_grp+7]).
            // Pre-packed by `pack_fp16_to_fp8` so the inner loop has zero
            // conversion ALU. Loaded as v2i_t (2 uint32) directly.
            v2i_t x_a = *(const v2i_t*)(x_base + k_off_a);
            v2i_t x_b = *(const v2i_t*)(x_base + k_off_b);
            v2i_t x_c = *(const v2i_t*)(x_base + k_off_c);
            v2i_t x_d = *(const v2i_t*)(x_base + k_off_d);

            // Build FP8 weight v2i_t from nibble pack via LUT lookup.
            // pk holds 8 nibbles in 32 bits; each nibble -> 1 byte E4M3.
            #define BUILD_W_FP8(pk, out)                                                  \
                {                                                                          \
                    unsigned int lo = 0u, hi = 0u;                                         \
                    lo |= (unsigned int)lut[((pk) >>  0) & 0xFu] <<  0;                    \
                    lo |= (unsigned int)lut[((pk) >>  4) & 0xFu] <<  8;                    \
                    lo |= (unsigned int)lut[((pk) >>  8) & 0xFu] << 16;                    \
                    lo |= (unsigned int)lut[((pk) >> 12) & 0xFu] << 24;                    \
                    hi |= (unsigned int)lut[((pk) >> 16) & 0xFu] <<  0;                    \
                    hi |= (unsigned int)lut[((pk) >> 20) & 0xFu] <<  8;                    \
                    hi |= (unsigned int)lut[((pk) >> 24) & 0xFu] << 16;                    \
                    hi |= (unsigned int)lut[((pk) >> 28) & 0xFu] << 24;                    \
                    (out)[0] = (int)lo; (out)[1] = (int)hi;                                \
                }

            v2i_t a_a, a_b, a_c, a_d;
            BUILD_W_FP8(pka, a_a);
            BUILD_W_FP8(pkb, a_b);
            BUILD_W_FP8(pkc, a_c);
            BUILD_W_FP8(pkd, a_d);

            #undef BUILD_W_FP8

            // WMMA pair for block_a (tiles 0,1 share UE8M0 block_a).
            v8f_t part_ab = {0,0,0,0,0,0,0,0};
            part_ab = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_a, x_a, part_ab);
            part_ab = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_b, x_b, part_ab);
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[j] += part_ab[j] * scale_a[j];

            // WMMA pair for block_c.
            v8f_t part_cd = {0,0,0,0,0,0,0,0};
            part_cd = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_c, x_c, part_cd);
            part_cd = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_d, x_d, part_cd);
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[j] += part_cd[j] * scale_c[j];
        }
    }

    // gfx12 wave32 WMMA C-mapping (identical to FP16 sibling):
    //   acc[j] = C[8*(tid>>4) + j][tid & 15]
    const int out_col = batch_start + m_lane;
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
