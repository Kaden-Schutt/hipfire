#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Q4_1 (signed INT4 in [-7,7], per-K=32 d/sum metadata) activation quantizer
// for gfx12 (RDNA4). Mirrors `quantize_q8_1_mmq_ds4` in
// gemm_hfq4g256_residual_mmq.hip but produces INT4 instead of INT8 to feed the
// iu4 K=32 wmma path (`gemm_hfq4g256_residual_iu4.gfx12.hip`).
//
// Output layout: `block_q4_1_mmq` per K=128 of one token, ordered by
// [K/128 block, batch column]:
//
//   struct block_q4_1_mmq {
//       half2 ds[4];       // (d, sum_q * d) per K=32 sub-block (4 sub-blocks
//                          //   = K=128). half2.x = d, half2.y = d * sum_q.
//       int8_t qs[64];     // 4 sub-blocks * 32 INT4 = 128 INT4 packed 2-per-byte.
//                          //   Within byte: low nibble = element 0 of pair,
//                          //   high nibble = element 1. INT4 stored as a signed
//                          //   value in [-7,7] sign-extended into the nibble
//                          //   per the iu4 wmma signed-input convention (lower
//                          //   3 bits = magnitude, bit 3 = sign for +/-).
//                          //   On read: `(int8_t)(byte << 4) >> 4` extracts
//                          //   low nibble; `(int8_t)(byte) >> 4` extracts high.
//   };
//
// sizeof = 8 + 64 = 72 B... wait, 4 half2 = 8 B + 64 INT4 bytes = 72 B. The
// spec said 80 B which assumed half2[4] takes 8 B and i8[64] takes 64 B.
// Re-checking: half2 = 4 B, half2[4] = 16 B. Plus 64 B = 80 B total. The
// spec is correct — half2 is FP16 + FP16 packed = 4 bytes, and `half2 ds[4]`
// is 4 * 4 = 16 bytes.
//
// Total per K=128 token-col = 16 + 64 = 80 bytes = 20 i32. The GEMM kernel
// uses MMQ_TILE_Y_K_Q4_1 = 20 to step through global / LDS cols.
//
// Quantization recipe (per K=32 sub-block, 32 elems):
//   amax = max_k |x[k]|
//   d = amax / 7              (INT4 signed range is [-7,7])
//   q[k] = clamp(round(x[k] / d), -7, 7)
//   sum_q = sum_k q[k]        (INT16 range — fits since 32 * 7 = 224)
//   ds[sub_block] = (FP16(d), FP16(d * sum_q))
//
// The inner-loop GEMM applies the algebraic correction:
//   C[m][n] += sum_k(a_int4 * w_int4) * d_a * scale_w
//            + (sum_k a_int4) * d_a * zero_w           (= ds.y * zero_w)
// — same structure as Q8_1 but with the INT4 range and per-K=32 rather than
// per-K=32 (Q8_1 group).
//
// Threading: 1 token per blockIdx.y, 1 char-quad (= 4 K-elements packed into
// one half-byte, NOPE — let me redo). Actually per i0 we handle 4 elements
// (1 char4 worth) like the Q8_1 reference, but for INT4 we need 8 elements
// per byte-write to keep the qs[] write coalesced. We'll do 8 elements per
// thread (= 1 i32 of activations -> 1 byte-quad of qs writes -> 4 INT4 pairs).
//
// Actually the cleaner design: each thread handles 4 K-elements (1 float4),
// writes 2 INT4 nibbles to one byte at offset (i0/2). __shfl_xor reduces
// amax/sum across the 8 threads owning one K=32 sub-block (4 elems per thread
// * 8 threads = 32 elems). Same shape as Q8_1's 4-elems-per-thread,
// 8-thread-reduce, just smaller K-per-thread sub-block.
//
// Wait — Q8_1 also does 4 elems per thread + 8-thread reduce per K=32 sub-block.
// So I keep the SAME thread layout, just change the q-encoding from INT8 to
// INT4 + the byte-pack stage.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#define WARP_SIZE 32
#define QK4_1 32
#define BLOCK_K 128

struct block_q4_1_mmq {
    half2 ds[4];        // (d, d*sum_q) per K=32 sub-block. 4 * 4 B = 16 B.
    int8_t qs[64];      // 4 * 32 INT4 packed 2-per-byte. 64 B.
};

static_assert(sizeof(block_q4_1_mmq) == 80, "bad block_q4_1_mmq size");

#if !HIPFIRE_GFX12

extern "C" __global__ void quantize_q4_1_mmq_ds4_gfx12(
    const float*,
    block_q4_1_mmq*,
    int,
    int
) {}

#else

// ---------------------------------------------------------------------------
// Quantize an [N x K] FP32 activation matrix into the block_q4_1_mmq layout.
//
// Grid: ((K+1023)/1024, N, 1)   block: (256, 1, 1)
// Each thread handles 4 contiguous K-elements (1 float4). 8 threads cover one
// K=32 sub-block; per-sub-block amax + sum reductions use __shfl_xor with
// xor masks 1,2,4 (covering the 8 threads of the sub-block — same as Q8_1).
//
// Output indexing (mirrors block_q8_1_mmq):
//   kblock = i0 / 128                          (which K=128 chunk of the row)
//   iqs    = i0 & 127                          (offset within the K=128 chunk)
//   out    = Y + kblock * N + token            (the K=128 chunk for this token)
//   sub    = iqs / 32                          (K=32 sub-block within the chunk)
//
// Each thread writes 4 INT4 values for 4 consecutive K-positions, packed as
// 2 bytes (low nibble = even K, high nibble = odd K) at byte offset iqs/2
// within out->qs[]. One thread per sub-block (lane 0 of the 8-thread group)
// writes the half2 ds entry.
// ---------------------------------------------------------------------------
extern "C" __global__ void quantize_q4_1_mmq_ds4_gfx12(
    const float* __restrict__ X,
    block_q4_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Per-sub-block reduce across the 8 threads owning the K=32 sub-block.
    // xor masks 1,2,4 → covers threads in the same (i0 / 32) group.
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    // INT4 signed range is [-7,7]. Use 7.0 (NOT 8.0) to stay strictly inside
    // the iu4 signed clamp (the wmma builtin clamp arg ensures saturation,
    // but rounding to ±7 keeps the sign-bit interpretation unambiguous).
    const float d = amax > 1.0e-20f ? amax / 7.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    // Quantize 4 floats → 4 INT4 values in [-7,7].
    int q0 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, xi.x * id)));
    int q1 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, xi.y * id)));
    int q2 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, xi.z * id)));
    int q3 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, xi.w * id)));

    // Sum the 4 quant values for this thread, then reduce across the 8
    // threads of the sub-block. Result is sum_q over K=32.
    int sum_q_local = q0 + q1 + q2 + q3;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum_q_local += __shfl_xor(sum_q_local, offset, WARP_SIZE);
    }
    const int sum_q = sum_q_local;

    // Pack 4 INT4 → 2 bytes. Low nibble of each byte = even-K element,
    // high nibble = odd-K element. INT4 sign-extends from bit 3 on read,
    // so we mask to 4 bits before OR-ing.
    const unsigned int q0_u = (unsigned)q0 & 0xFu;
    const unsigned int q1_u = (unsigned)q1 & 0xFu;
    const unsigned int q2_u = (unsigned)q2 & 0xFu;
    const unsigned int q3_u = (unsigned)q3 & 0xFu;
    const unsigned int byte0 = q0_u | (q1_u << 4);
    const unsigned int byte1 = q2_u | (q3_u << 4);

    const int kblock = i0 / BLOCK_K;
    const int iqs    = i0 & (BLOCK_K - 1);    // offset within K=128 chunk (0..127)
    const int sub    = iqs / QK4_1;           // K=32 sub-block index (0..3)
    block_q4_1_mmq* out = Y + (long long)kblock * N + token;

    // 4 K-positions → 2 bytes at byte offset iqs/2 within qs[].
    out->qs[iqs / 2 + 0] = (int8_t)byte0;
    out->qs[iqs / 2 + 1] = (int8_t)byte1;

    // Lane 0 of the 8-thread sub-block writes ds[sub] = (d, d*sum_q).
    if ((iqs & (QK4_1 - 1)) == 0) {
        out->ds[sub] = make_half2((_Float16)d, (_Float16)(d * (float)sum_q));
    }
}

#endif
