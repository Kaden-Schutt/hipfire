#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM for MQ3-Lloyd weights on gfx1151 (Strix Halo).
//
// MQ3-Lloyd twin of `gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1151.hip`. The
// FP16-WMMA MQ3-Lloyd kernel (`gemm_mq3g256_lloyd_moe_grouped_wmma_k2`) is
// DEQUANT-bound: the 3D-bilinear Lloyd decode (8-entry codebook, ~10 ALU
// ops/weight) shares the VALU issue port with the WMMA on RDNA3.
//
// This variant kills both costs the same way the MQ2 sibling does:
//   1. Per group, quantize the 8-entry f16 Lloyd codebook to int8 + a single
//      f32 group scale ONCE (amortized over 256 weights/group), then decode
//      each 3-bit index via a byte-extract LUT (~3 ALU ops/weight).
//   2. Run i8 WMMA (`wmma_i32_16x16x16_iu8_w32`, ~2x FP16 rate) on Q8_1
//      activations, applying the group scale × activation scale AFTER the
//      int32 accumulate.
//
// MQ3-Lloyd has NO weight zero-point (the codebook carries signed values), so
// the scale FMA is `acc += sc * d_x * dot`.
//
// MQ3-Lloyd group layout (112 bytes / 256-K group; see the FP16 sibling):
//   gp+0  : cb0..cb7   (8 × f16 = 16 bytes)  — Lloyd codebook
//   gp+16 : 256 × 3-bit indices (= 96 bytes), LSB-first. Each K-tile (16
//           weights) = 6 bytes = two 24-bit packs (p0: weights 0..7, p1:
//           weights 8..15); idx_i = (pk >> (3*i)) & 7.
//
// X must be pre-quantized to block_q8_1_mmq via `ensure_q8_1_mmq_x`. Kernarg
// layout is identical to the MQ2 i8 sibling.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

#if defined(__gfx1151__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq3g256_lloyd_moe_grouped_mmq_gfx1151(
    const unsigned long long*  __restrict__ expert_weight_ptrs, // [E]
    const int*                 __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                 __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*      __restrict__ X_q8_src,           // [K/128 × x_src_rows]
    float*                     __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 112;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 112;

        // ── Quantize this lane's 8-entry Lloyd codebook to int8 + scale ──
        // Done ONCE per group (amortized over 256 weights). index q∈{0..7}
        // selects cb[q]; the 8 int8 levels pack into two u32 byte-extract LUTs.
        const __half* cbh = reinterpret_cast<const __half*>(gp);
        float cb[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) cb[i] = __half2float(cbh[i]);
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(cb[i]));
        float sc_self = amax * (1.0f / 127.0f);
        float inv_sc = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        unsigned int cb_lo = 0u, cb_hi = 0u;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cb_lo |= ((unsigned int)((int)rintf(cb[i] * inv_sc) & 0xFF)) << (i * 8);
            cb_hi |= ((unsigned int)((int)rintf(cb[i + 4] * inv_sc) & 0xFF)) << (i * 8);
        }

        // Broadcast each output row's group scale to the lanes that own it.
        float sc_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
        }

        // Index region: gp+16, 96 bytes (16 K-tiles × 6 B). Read per tile.
        const unsigned char* idx_base = (const unsigned char*)(gp + 16);

        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            const block_q8_1_mmq* xb = (x_row >= 0)
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : nullptr;

            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                float d_x_self = 0.0f;
                if (xb != nullptr) {
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    d_x_self = dm_f.x;   // Q8_1 per-sub-block activation scale
                }

                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    // K-offset within the 256-element group → tile index 0..15.
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;
                    const int tile = k_off / 16;
                    const unsigned char* w = idx_base + tile * 6;
                    // 6 B → two 24-bit packs (8 × 3-bit indices each), LSB-first.
                    unsigned int p0 = (unsigned int)w[0] | ((unsigned int)w[1] << 8)
                                    | ((unsigned int)w[2] << 16);
                    unsigned int p1 = (unsigned int)w[3] | ((unsigned int)w[4] << 8)
                                    | ((unsigned int)w[5] << 16);

                    // A operand: 3-bit index → int8 codebook via byte-extract LUT.
                    int8x16_t a_vec;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        unsigned int q = (p0 >> (3 * i)) & 0x7u;
                        unsigned int src = (q & 4u) ? cb_hi : cb_lo;
                        a_vec[i] = (int8_t)((src >> ((q & 3u) * 8)) & 0xFFu);
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        unsigned int q = (p1 >> (3 * i)) & 0x7u;
                        unsigned int src = (q & 4u) ? cb_hi : cb_lo;
                        a_vec[8 + i] = (int8_t)((src >> ((q & 3u) * 8)) & 0xFFu);
                    }

                    // B operand: Q8_1 int8 activations.
                    int8x16_t b_vec;
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                        b_vec = *(const int8x16_t*)qs_p;
                    } else {
                        b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                    }

                    int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                    int32x4_t b_i32 = *(const int32x4_t*)&b_vec;

                    // signed × signed (Lloyd levels can be negative).
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a_i32, true, b_i32, cacc, true
                    );
                } // kt

                // Scale FMA (no zero-point term for MQ3-Lloyd).
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[j] += sc_row[j] * d_x_self * (float)cacc[j];
                }
            } // sb
        } // kb_in_group
    } // g

    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_mq3g256_lloyd_moe_grouped_mmq_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
