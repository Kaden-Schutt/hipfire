#include <hip/hip_runtime.h>

// HFP4G32 GEMV — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale).
// v1 correctness anchor: no WMMA, no FP8, no rotation. See docs/quant-formats/hfp4.md.
//
// Per-row layout: 16-B header (row_scale_a:f16, row_scale_b:f16, block_count:u16, flags:u8, ...)
//                 then (K/32) blocks × 17 B each (UE8M0:u8 + 16 B nibbles).
// Per element:    value = row_scale_a * 2^(block_e - 127) * E2M1_LUT[nibble]
//
// Geometry coincidence: 8 blocks × 17 B = 136 B per 256 K-elements.
// This matches HFQ4G256's per-group stride exactly, so the OUTER LOOP STRUCTURE
// (4-accumulator + tail-by-g%4) ports verbatim from gemv_hfq4g256.hip:33–124.
// Only the per-block decode differs:
//   - HFQ4: 4-byte fp32 scale + 4-byte fp32 zero + 128-byte nibbles → (sc * (float)n + zp) * x[i]
//   - HFP4: 16-byte row hdr + 8 × (1-byte UE8M0 + 16-byte nibbles) → (sc * lut[n]) * x[i]
//          where sc = row_scale_a * 2^(block_e - 127), per-block.
//
// Within a 256-K-element "group" of 8 blocks, each thread reads from ONE block:
//   block_in_iter = tid >> 2   (0..7) — which block of 8 this thread is in
//   nib_off       = (tid & 3) * 4  — byte offset into that block's 16-B nibble payload
//   K_in_block    = (tid & 3) * 8  — first element index within the block
//
// 4-accumulator interleaving + tail-by-g%4 + (acc0+acc1)+(acc2+acc3) finalize:
// the FP add ordering is byte-identical to gemv_hfq4g256 across archs, so the
// gfx1100 byte-exact gate carries over to default/RDNA1/RDNA2 fallbacks for free.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // 16-entry signed E2M1 magnitude LUT in LDS (32 B). Initialized once by lanes 0..15.
    // OCP E2M1 lattice: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Locked to spec.
    // Use built-in `_Float16` rather than `__half` (no `<hip/hip_fp16.h>` needed —
    // that header pulls in CUDA-shim complex_builtins which has a max/fmax declaration
    // bug in ROCm 7.2.2's clang headers).
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

    // Per-row layout: 16-B header + (K/32)*17 B blocks; equivalently (K/256) "groups" of 136 B.
    const int groups_per_row = K / 256;                 // each "group" = 8 HFP4 blocks = 256 K-elt
    const int row_block_bytes = (K / 32) * 17;          // = groups_per_row * 136
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                     // skip per-row header → first block

    // FP16 row scale lives in the first 2 bytes of the row header.
    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    // Thread slot within an 8-block group.
    const int block_in_iter = tid >> 2;     // 0..7
    const int nib_off = (tid & 3) * 4;      // 0/4/8/12
    const int boff = nib_off;               // alias to mirror HFQ4 naming

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // Per-group-thread byte offset to the thread's block within a 136-B group.
    // Block stride is 17 B; thread is in block_in_iter ∈ [0,7].
    const int thread_block_off = block_in_iter * 17;

    #define HFP4_DOG(pk, sc, b, a) \
        (a) += (sc * (float)lut[(pk)        & 0xFu]) * x[(b)    ] \
             + (sc * (float)lut[((pk) >>  4) & 0xFu]) * x[(b) + 1] \
             + (sc * (float)lut[((pk) >>  8) & 0xFu]) * x[(b) + 2] \
             + (sc * (float)lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + (sc * (float)lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + (sc * (float)lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + (sc * (float)lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + (sc * (float)lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    // Quad-unrolled main loop: 4 groups (= 32 blocks = 1024 K-elt) per quad iter.
    // Mirrors HFQ4G256's structure 1:1 — same group stride (136 B), same accumulator
    // interleaving, same tail-by-g%4 invariant.
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned char e0 = *(const unsigned char*)(gp0);
        const unsigned char e1 = *(const unsigned char*)(gp1);
        const unsigned char e2 = *(const unsigned char*)(gp2);
        const unsigned char e3 = *(const unsigned char*)(gp3);

        // UE8M0 dequant via direct FP32 bit construction: 2^(e-127) is exactly the FP32 number
        // with biased exponent = e, mantissa = 0, sign = 0. Pure bit-shift; portable across all
        // ROCm versions (avoids host-only ldexpf in ROCm ≥ 7.2.2 where it isn't device-callable).
        const float sc0 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e0) << 23);
        const float sc1 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e1) << 23);
        const float sc2 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e2) << 23);
        const float sc3 = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e3) << 23);

        const unsigned int pk0 = *(const unsigned int*)(gp0 + 1 + boff);
        const unsigned int pk1 = *(const unsigned int*)(gp1 + 1 + boff);
        const unsigned int pk2 = *(const unsigned int*)(gp2 + 1 + boff);
        const unsigned int pk3 = *(const unsigned int*)(gp3 + 1 + boff);

        // K_base for this thread's elements within group g:
        //   base = g * 256 + block_in_iter * 32 + (tid & 3) * 8
        // group has 8 blocks of 32 K-elts each; thread is in block block_in_iter,
        // covering 8 elements at offset (tid & 3) * 8 inside that block.
        const int base = g * 256 + tid * 8; // (tid>>2)*32 + (tid&3)*8 simplifies to tid*8

        HFP4_DOG(pk0, sc0, base,         acc0);
        HFP4_DOG(pk1, sc1, base + 256,   acc1);
        HFP4_DOG(pk2, sc2, base + 512,   acc2);
        HFP4_DOG(pk3, sc3, base + 768,   acc3);
    }
    #undef HFP4_DOG

    // Tail: groups_per_row may not be a multiple of 4. Each tail group must
    // accumulate into acc[g % 4] to preserve the 4-way interleaving the combine
    // step assumes. (See gemv_hfq4g256.hip for the bug writeup.)
    #define HFP4_TAIL_DOG(pk, sc, b, a) \
        (a) += (sc * (float)lut[(pk)        & 0xFu]) * x[(b)    ] \
             + (sc * (float)lut[((pk) >>  4) & 0xFu]) * x[(b) + 1] \
             + (sc * (float)lut[((pk) >>  8) & 0xFu]) * x[(b) + 2] \
             + (sc * (float)lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + (sc * (float)lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + (sc * (float)lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + (sc * (float)lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + (sc * (float)lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e) << 23);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8; // (tid>>2)*32 + (tid&3)*8 simplifies to tid*8
        HFP4_TAIL_DOG(pk, sc, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e) << 23);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8; // (tid>>2)*32 + (tid&3)*8 simplifies to tid*8
        HFP4_TAIL_DOG(pk, sc, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * __builtin_bit_cast(float, ((unsigned int)e) << 23);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8; // (tid>>2)*32 + (tid&3)*8 simplifies to tid*8
        HFP4_TAIL_DOG(pk, sc, base, acc2);
    }
    #undef HFP4_TAIL_DOG

    // Combine 4 accumulators in HFQ4-byte-exact order.
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
