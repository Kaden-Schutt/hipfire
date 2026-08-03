#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM for MQ2-Lloyd weights on gfx1151 (Strix Halo).
//
// Twin of `gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip`, adapted to the
// MQ2-Lloyd weight format. The FP16-WMMA MQ2-Lloyd kernel
// (`gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2`) is DEQUANT-bound: at the
// DeepSeek-V4 prefill shape it runs ~6.9 TFLOP/s vs ~15 TFLOP/s for the
// trivial-dequant Q8 path — the entire gap is the per-weight Lloyd bilinear
// decode (`cb0 + qlo*d01 + qhi*d02 + qlo*qhi*d_xor`, ~12 ALU ops/weight) which
// shares the VALU issue port with the WMMA on RDNA3.
//
// This variant kills both costs:
//   1. Per group, quantize the 4-entry f16 Lloyd codebook to int8 + a single
//      f32 group scale ONCE (amortized over 256 weights/group), then decode
//      each 2-bit index via a byte-extract LUT (~3 ALU ops/weight).
//   2. Run i8 WMMA (`wmma_i32_16x16x16_iu8_w32`, ~2x FP16 rate) on Q8_1
//      activations, applying the group scale × activation scale AFTER the
//      int32 accumulate.
//
// MQ2-Lloyd has NO weight zero-point (the codebook directly carries signed
// values), so the scale FMA is `acc += sc * d_x * dot` — simpler than the
// HFQ4 sibling (no zp*sum_x correction term).
//
// MQ2-Lloyd group layout (72 bytes / 256-K group):
//   gp+0 : cb0..cb3   (4 × f16 = 8 bytes)  — Lloyd codebook
//   gp+8 : 256 × 2-bit indices (= 64 bytes)
//
// X must be pre-quantized to block_q8_1_mmq via `ensure_q8_1_mmq_x` (the
// wrapper does this); kernarg layout is identical to the FP16 sibling plus the
// trailing `x_src_rows` (Q8_1 source row count).
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
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;

        // ── Quantize this lane's 4-entry Lloyd codebook to int8 + scale ──
        // Done ONCE per group (amortized over 256 weights). cb values are the
        // actual dequantized weight levels; index q∈{0,1,2,3} selects cb[q].
        const __half* cbh = reinterpret_cast<const __half*>(gp);
        float cb0 = __half2float(cbh[0]);
        float cb1 = __half2float(cbh[1]);
        float cb2 = __half2float(cbh[2]);
        float cb3 = __half2float(cbh[3]);
        float amax = fmaxf(fmaxf(fabsf(cb0), fabsf(cb1)), fmaxf(fabsf(cb2), fabsf(cb3)));
        float sc_self = amax * (1.0f / 127.0f);
        float inv_sc = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        // int8 codebook packed little-endian into a u32 for byte-extract LUT.
        int q0 = (int)rintf(cb0 * inv_sc);
        int q1 = (int)rintf(cb1 * inv_sc);
        int q2 = (int)rintf(cb2 * inv_sc);
        int q3 = (int)rintf(cb3 * inv_sc);
        unsigned int cbpack = ((unsigned int)(q0 & 0xFF))
                            | ((unsigned int)(q1 & 0xFF) << 8)
                            | ((unsigned int)(q2 & 0xFF) << 16)
                            | ((unsigned int)(q3 & 0xFF) << 24);

        // Broadcast each output row's group scale to the lanes that own it.
        // Output row r = row_start + 2*j + (tid>>4) ⇒ source lane = 2*j + (tid>>4).
        float sc_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
        }

        // Load the whole group's 64-byte 2-bit index region (16 K-tiles × one
        // u32 each) up front via four 128-bit loads, into registers. The inner
        // K-tile loop then reads from registers — the kernel is memory-unit
        // bound (93% MemUnitBusy), so collapsing 16 scalar u32 loads into 4
        // vector loads cuts load-instruction pressure 4x.
        unsigned int pk_all[16];
        {
            const int32x4_t* pkp = (const int32x4_t*)(gp + 8);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int32x4_t v = pkp[i];
                pk_all[i * 4 + 0] = (unsigned int)v[0];
                pk_all[i * 4 + 1] = (unsigned int)v[1];
                pk_all[i * 4 + 2] = (unsigned int)v[2];
                pk_all[i * 4 + 3] = (unsigned int)v[3];
            }
        }

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
                    // K-offset within the 256-element group.
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;
                    // 16 × 2-bit indices = 32 bits = pre-loaded u32 (k_off/16).
                    const unsigned int pk = pk_all[k_off / 16];

                    // A operand: 2-bit index → int8 codebook via byte-extract LUT.
                    int8x16_t a_vec;
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        unsigned int q = (pk >> (2 * i)) & 0x3u;
                        a_vec[i] = (int8_t)((cbpack >> (q * 8)) & 0xFFu);
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

                    // signed × signed: codebook int8 is signed (Lloyd levels can
                    // be negative); Q8_1 activations are signed int8.
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a_i32, true, b_i32, cacc, true
                    );
                } // kt

                // Scale FMA (no zero-point term for MQ2-Lloyd).
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

extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
