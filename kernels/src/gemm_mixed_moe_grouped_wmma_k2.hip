#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

// Merged dtype-tag MoE batched-prefill grouped-WMMA kernel — gfx11 (RDNA3/3.5).
//
// Serves ALL four per-expert quantization formats in a single kernel via a
// block-uniform tag branch (zero warp divergence): each block's 16 slots all
// belong to the same expert, so tag = dtype_tags[expert_id] is the same for
// every thread in the block.
//
// Tag map (same as the decode merged kernel + load-time tag table):
//   0 = MQ6 / HFQ6-G256  — 200 B/group, 6-bit affine (sc*q+zp)
//   1 = MQ2-Lloyd-G256   —  72 B/group, 4-entry fp16 codebook (2-bit)
//   2 = MQ4 / HFQ4-G256  — 136 B/group, 4-bit affine (sc*q+zp)
//   3 = MQ3-Lloyd-G256   — 112 B/group, 8-entry fp16 codebook (3-bit)
//   4 = mfp4-E8          — 16 B row hdr + (K/32)*17 B; 136 B/group, E8 VQ
//
// CRITICAL (MQ2L/MQ3L tags 1 & 3): Per-lane REGISTER bilinear/trilinear
// decomposition — never LDS broadcast. Each lane handles a different M-row
// with its own per-group codebook, so LDS broadcast cross-contaminates lanes.
// The bilinear/trilinear FMA form also enables WMMA vectorization; the
// ternary-chain / LDS-indexed form suppresses it (verified 2026-05-22).
//
// The single kernel serves both gate_up and down dispatch:
//   gate_up: x_row_div = K_TOP  → x_row = flat / K_TOP
//   down:    x_row_div = 1      → x_row = flat
//
// Dequant lifted verbatim from committed per-dtype grouped kernels:
//   tag 0: gemm_hfq6g256_moe_grouped_wmma_k2.hip
//   tag 1: gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip
//   tag 2: gemm_hfq4g256_moe_grouped_wmma_k2.hip
//   tag 3: gemm_mq3g256_lloyd_moe_grouped_wmma_k2.hip
//
// WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (gfx11 _w32).
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// mfp4-E8 (tag 4) decode helpers — byte-identical to
// gemm_mfp4g32_e8_moe_grouped_wmma.gfx1151.hip.
//
// Tags 5 and 6 extend the E8 family to narrower nibble widths:
//   tag 5 = mfp3-E8: 3-bit nibbles, 13 B/block, 104 B/group, center 4, coset bit 23
//   tag 6 = mfp2-E8: 2-bit nibbles,  9 B/block,  72 B/group, center 2, coset bit 15
//
// QUANT_STEP constants MUST match crates/hipfire-quantize/src/e8.rs.
#define QUANT_STEP_MFP3 1.8f
#define QUANT_STEP_MFP2 3.8f

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f; // defensive: NaN code -> max finite (never emitted)
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}
#endif
__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}

// mfp3-E8 (tag 5): 3-bit nibbles, center 4, coset bit 23, e7_high 2b @ bit 21.
// High 8 bits of storing u32 are always zero — narrow 3-byte read is lossless.
// Center 4 (box [-4,3]) matches coord_bias_n(3)=4 in e8.rs (MSE-tuned recenter).
__device__ __forceinline__ void mfp3_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 23) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(3 * i)) & 0x7u; sl += e[i]; }
    const unsigned int e7_high = (idx >> 21) & 0x3u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 4); p[i] = coset ? c + 0.5f : c; }
}

// mfp2-E8 (tag 6): 2-bit nibbles, center 2, coset bit 15, e7_high 1b @ bit 14.
// High 16 bits of storing u32 are always zero — narrow 2-byte read is lossless.
// Center 2 (box [-2,1]) matches coord_bias_n(2)=2 in e8.rs (MSE-tuned recenter).
__device__ __forceinline__ void mfp2_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 15) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(2 * i)) & 0x3u; sl += e[i]; }
    const unsigned int e7_high = (idx >> 14) & 0x1u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 2); p[i] = coset ? c + 0.5f : c; }
}

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mixed_moe_grouped_wmma_k2(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E] u64
    const unsigned char*      __restrict__ dtype_tags,         // [E] u8 tag per expert
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    // Sentinel early-return: expert_tile_ids[tile_y] == -1 marks tiles
    // beyond the live m_total (the dispatcher may launch up to
    // m_total_max/16 tiles to skip the m_total dtoh sync).
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    // BLOCK-UNIFORM tag read — all 32 threads in the block share one expert,
    // so this branch has zero warp divergence.
    const unsigned char tag = dtype_tags[expert_id];
    // group_bytes MUST be computed before row_base (row_base depends on it).
    const int group_bytes = (tag == 0) ? 200 : (tag == 1) ? 72 : (tag == 2) ? 136 : 112;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather: each lane (tid & 15) gets its own routed slot, recovers
    // the source row via x_row_div (K_TOP for gate_up, 1 for down).
    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to last valid row (discarded by output mask).
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * group_bytes;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // -------------------------------------------------------------------------
    // Tag-dispatched inner loop (block-uniform → zero warp divergence).
    // Each branch is lifted verbatim from the matching per-dtype grouped kernel.
    // -------------------------------------------------------------------------

    if (tag == 0) {
        // ----- MQ6 / HFQ6-G256 (200 B/group) --------------------------------
        // Source: gemm_hfq6g256_moe_grouped_wmma_k2.hip, lines 84–136.
        // Layout: 4 B scale + 4 B zero + 192 B packed 6-bit (16 values per K-tile
        //         in 12 B, read as 4 overlapping uint32s at +0/+3/+6/+9).
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 200;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;

                const unsigned char* wa = (const unsigned char*)(gp + 8 + kt * 12);
                const unsigned char* wb = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
                // a3/c3 use only bits 0..23 (3 bytes); a `*(uint*)(w+9)` fetches a
                // 4th byte at w+12 that is 1 B past the group on the last tile of
                // the last group → GPU VM read fault on tightly-packed graded
                // buffers. Read exactly 3 bytes (bit-identical).
                unsigned int a0 = *(const unsigned int*)(wa);
                unsigned int a1 = *(const unsigned int*)(wa + 3);
                unsigned int a2 = *(const unsigned int*)(wa + 6);
                unsigned int a3 = (unsigned int)wa[9] | ((unsigned int)wa[10] << 8) | ((unsigned int)wa[11] << 16);
                unsigned int c0 = *(const unsigned int*)(wb);
                unsigned int c1 = *(const unsigned int*)(wb + 3);
                unsigned int c2 = *(const unsigned int*)(wb + 6);
                unsigned int c3 = (unsigned int)wb[9] | ((unsigned int)wb[10] << 8) | ((unsigned int)wb[11] << 16);

                half16_t b_a, b_b;
                if (xg != nullptr) {
                    b_a = *(const half16_t*)(xg + k_off_a);
                    b_b = *(const half16_t*)(xg + k_off_b);
                } else {
                    b_a = zero_b;
                    b_b = zero_b;
                }

                half16_t a_a;
                #define DQ6(reg, i, dw, sh) reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
                DQ6(a_a, 0,  a0, 0); DQ6(a_a, 1,  a0, 6); DQ6(a_a, 2,  a0, 12); DQ6(a_a, 3,  a0, 18);
                DQ6(a_a, 4,  a1, 0); DQ6(a_a, 5,  a1, 6); DQ6(a_a, 6,  a1, 12); DQ6(a_a, 7,  a1, 18);
                DQ6(a_a, 8,  a2, 0); DQ6(a_a, 9,  a2, 6); DQ6(a_a, 10, a2, 12); DQ6(a_a, 11, a2, 18);
                DQ6(a_a, 12, a3, 0); DQ6(a_a, 13, a3, 6); DQ6(a_a, 14, a3, 12); DQ6(a_a, 15, a3, 18);

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

                DQ6(a_a, 0,  c0, 0); DQ6(a_a, 1,  c0, 6); DQ6(a_a, 2,  c0, 12); DQ6(a_a, 3,  c0, 18);
                DQ6(a_a, 4,  c1, 0); DQ6(a_a, 5,  c1, 6); DQ6(a_a, 6,  c1, 12); DQ6(a_a, 7,  c1, 18);
                DQ6(a_a, 8,  c2, 0); DQ6(a_a, 9,  c2, 6); DQ6(a_a, 10, c2, 12); DQ6(a_a, 11, c2, 18);
                DQ6(a_a, 12, c3, 0); DQ6(a_a, 13, c3, 6); DQ6(a_a, 14, c3, 12); DQ6(a_a, 15, c3, 18);
                #undef DQ6

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
            }
        }

    } else if (tag == 1) {
        // ----- MQ2-Lloyd-G256 (72 B/group) -----------------------------------
        // Source: gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip, lines 85–146.
        // Layout: 8 B (4×fp16 codebook, ascending) + 64 B 2-bit packed.
        // Per-lane REGISTER bilinear decomposition — NOT LDS broadcast.
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 72;
            const __half* cb_h = reinterpret_cast<const __half*>(gp);
            _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
            _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
            _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
            _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
            // Bilinear decomposition (q = qhi*2 + qlo; avoids ternary-chain
            // which suppresses WMMA vectorization).
            const _Float16 d01   = cb1 - cb0;
            const _Float16 d02   = cb2 - cb0;
            const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;
            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;

                // 1 uint32 per K-tile (16 weights × 2 bits = 32 bits).
                unsigned int pa = *(const unsigned int*)(gp + 8 + k_off_a / 4);
                unsigned int pb = *(const unsigned int*)(gp + 8 + k_off_b / 4);

                half16_t b_a, b_b;
                if (xg != nullptr) {
                    b_a = *(const half16_t*)(xg + k_off_a);
                    b_b = *(const half16_t*)(xg + k_off_b);
                } else {
                    b_a = zero_b;
                    b_b = zero_b;
                }

                #define DQ(reg, i, pk, sh) do { \
                    _Float16 qlo = (_Float16)(int)(((pk) >> (sh))     & 0x1u); \
                    _Float16 qhi = (_Float16)(int)(((pk) >> ((sh)+1)) & 0x1u); \
                    reg[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor; \
                } while (0)

                half16_t a_a;
                DQ(a_a, 0,  pa,  0); DQ(a_a, 1,  pa,  2); DQ(a_a, 2,  pa,  4); DQ(a_a, 3,  pa,  6);
                DQ(a_a, 4,  pa,  8); DQ(a_a, 5,  pa, 10); DQ(a_a, 6,  pa, 12); DQ(a_a, 7,  pa, 14);
                DQ(a_a, 8,  pa, 16); DQ(a_a, 9,  pa, 18); DQ(a_a, 10, pa, 20); DQ(a_a, 11, pa, 22);
                DQ(a_a, 12, pa, 24); DQ(a_a, 13, pa, 26); DQ(a_a, 14, pa, 28); DQ(a_a, 15, pa, 30);

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

                DQ(a_a, 0,  pb,  0); DQ(a_a, 1,  pb,  2); DQ(a_a, 2,  pb,  4); DQ(a_a, 3,  pb,  6);
                DQ(a_a, 4,  pb,  8); DQ(a_a, 5,  pb, 10); DQ(a_a, 6,  pb, 12); DQ(a_a, 7,  pb, 14);
                DQ(a_a, 8,  pb, 16); DQ(a_a, 9,  pb, 18); DQ(a_a, 10, pb, 20); DQ(a_a, 11, pb, 22);
                DQ(a_a, 12, pb, 24); DQ(a_a, 13, pb, 26); DQ(a_a, 14, pb, 28); DQ(a_a, 15, pb, 30);
                #undef DQ

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
            }
        }

    } else if (tag == 2) {
        // ----- MQ4 / HFQ4-G256 (136 B/group) --------------------------------
        // Source: gemm_hfq4g256_moe_grouped_wmma_k2.hip, lines 93–139.
        // Layout: 4 B scale + 4 B zero + 128 B 4-bit packed (2 uint32s per K-tile).
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 136;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;

                unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
                unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
                unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
                unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

                half16_t b_a, b_b;
                if (xg != nullptr) {
                    b_a = *(const half16_t*)(xg + k_off_a);
                    b_b = *(const half16_t*)(xg + k_off_b);
                } else {
                    b_a = zero_b;
                    b_b = zero_b;
                }

                half16_t a_a;
                #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
                DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
                DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
                DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
                DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

                DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
                DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
                DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
                DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
                #undef DQ

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
            }
        }

    } else if (tag == 3) {
        // ----- MQ3-Lloyd-G256 (112 B/group) ---------------------------------
        // Source: gemm_mq3g256_lloyd_moe_grouped_wmma_k2.hip, lines 96–166.
        // Layout: 16 B (8×fp16 codebook) + 96 B 3-bit packed (6 B per K-tile).
        //
        // DIRECT-LUT REGISTER SELECT (replaces the 7-FMA multilinear decode).
        // The codebook (cb0..cb7) is already in registers; the 3-bit index just
        // selects one of them. A per-lane REGISTER select tree (v_cndmask, no
        // memory) is ~1 op vs the multilinear's 7 FMAs AND is strictly more
        // accurate (it returns cb[idx] exactly, with zero fp16 rounding — the
        // multilinear equals cb[idx] in REAL arithmetic but accumulates fp16
        // rounding over 7 FMAs). The select tree keeps each lane's value in a
        // register, so the half16_t WMMA A-fragment stays vectorized (an
        // LDS-indexed LUT would suppress WMMA — verified 2026-05-22).
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 112;
            const __half* cb_h = reinterpret_cast<const __half*>(gp);
            _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
            _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
            _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
            _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
            _Float16 cb4 = (_Float16)__half2float(cb_h[4]);
            _Float16 cb5 = (_Float16)__half2float(cb_h[5]);
            _Float16 cb6 = (_Float16)__half2float(cb_h[6]);
            _Float16 cb7 = (_Float16)__half2float(cb_h[7]);

            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;
                // Index payload at [16..112): each K-tile = 6 B (16 × 3-bit).
                const unsigned char* wa = (const unsigned char*)(gp + 16 + kt       * 6);
                const unsigned char* wb = (const unsigned char*)(gp + 16 + (kt + 1) * 6);
                // 6 B → two 24-bit packs (8 × 3-bit indices each), LSB-first.
                unsigned int p0a = (unsigned int)wa[0] | ((unsigned int)wa[1] << 8) | ((unsigned int)wa[2] << 16);
                unsigned int p1a = (unsigned int)wa[3] | ((unsigned int)wa[4] << 8) | ((unsigned int)wa[5] << 16);
                unsigned int p0b = (unsigned int)wb[0] | ((unsigned int)wb[1] << 8) | ((unsigned int)wb[2] << 16);
                unsigned int p1b = (unsigned int)wb[3] | ((unsigned int)wb[4] << 8) | ((unsigned int)wb[5] << 16);

                half16_t b_a, b_b;
                if (xg != nullptr) {
                    b_a = *(const half16_t*)(xg + k_off_a);
                    b_b = *(const half16_t*)(xg + k_off_b);
                } else {
                    b_a = zero_b;
                    b_b = zero_b;
                }

                // Direct LUT via register select tree → lowers to v_cndmask,
                // no scratch / no LDS, keeps the A-fragment vectorized.
                #define DQ3(reg, i, pk, sh) do { \
                    unsigned int b0 = ((pk) >> (sh))     & 1u; \
                    unsigned int b1 = ((pk) >> ((sh)+1)) & 1u; \
                    unsigned int b2 = ((pk) >> ((sh)+2)) & 1u; \
                    _Float16 s0 = b0 ? cb1 : cb0; \
                    _Float16 s1 = b0 ? cb3 : cb2; \
                    _Float16 s2 = b0 ? cb5 : cb4; \
                    _Float16 s3 = b0 ? cb7 : cb6; \
                    _Float16 t0 = b1 ? s1 : s0; \
                    _Float16 t1 = b1 ? s3 : s2; \
                    (reg)[i] = b2 ? t1 : t0; \
                } while (0)

                half16_t a_a;
                DQ3(a_a, 0,  p0a, 0); DQ3(a_a, 1,  p0a, 3); DQ3(a_a, 2,  p0a, 6);  DQ3(a_a, 3,  p0a, 9);
                DQ3(a_a, 4,  p0a, 12);DQ3(a_a, 5,  p0a, 15);DQ3(a_a, 6,  p0a, 18); DQ3(a_a, 7,  p0a, 21);
                DQ3(a_a, 8,  p1a, 0); DQ3(a_a, 9,  p1a, 3); DQ3(a_a, 10, p1a, 6);  DQ3(a_a, 11, p1a, 9);
                DQ3(a_a, 12, p1a, 12);DQ3(a_a, 13, p1a, 15);DQ3(a_a, 14, p1a, 18); DQ3(a_a, 15, p1a, 21);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

                DQ3(a_a, 0,  p0b, 0); DQ3(a_a, 1,  p0b, 3); DQ3(a_a, 2,  p0b, 6);  DQ3(a_a, 3,  p0b, 9);
                DQ3(a_a, 4,  p0b, 12);DQ3(a_a, 5,  p0b, 15);DQ3(a_a, 6,  p0b, 18); DQ3(a_a, 7,  p0b, 21);
                DQ3(a_a, 8,  p1b, 0); DQ3(a_a, 9,  p1b, 3); DQ3(a_a, 10, p1b, 6);  DQ3(a_a, 11, p1b, 9);
                DQ3(a_a, 12, p1b, 12);DQ3(a_a, 13, p1b, 15);DQ3(a_a, 14, p1b, 18); DQ3(a_a, 15, p1b, 21);
                #undef DQ3
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
            }
        }
    } else if (tag == 4) {
        // ----- mfp4-E8 (16 B row hdr + (K/32)*17 B; 136 B/group) ------------
        // Source: gemm_mfp4g32_e8_moe_grouped_wmma.gfx1151.hip (lines 101–131).
        // E8 carries a 16-byte row header (f16 row_scale) before the groups, so
        // it computes its own row/block pointers (shared row_base/group_bytes
        // assume a header-less group-major layout). half16: each 16-K WMMA tile
        // is two E8 codewords (cw0 → a[0..8), cw1 → a[8..16)).
        const int row_block_bytes = (K / 32) * 17;
        const char* e8_row_ptr =
            A + (long long)safe_row * (16 + (long long)row_block_bytes);
        const float row_scale_f =
            (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
        const char* blocks_ptr = e8_row_ptr + 16;
        for (int g = 0; g < groups_per_row; g++) {
            const char* group_ptr = blocks_ptr + (long long)g * 136;
            const _Float16* xg = (x_base != nullptr) ? (x_base + (long long)g * 256) : nullptr;
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                const int block_idx = k_off >> 5;            // 0..7
                const char* block_ptr = group_ptr + (long long)block_idx * 17;
                const float ssc = row_scale_f
                    * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * 0.88f;
                const int cw_base = (k_off & 31) >> 3;        // 0 or 2
                const unsigned int cw0 = *(const unsigned int*)(block_ptr + 1 + cw_base * 4);
                const unsigned int cw1 = *(const unsigned int*)(block_ptr + 1 + (cw_base + 1) * 4);
                float p0[8], p1[8];
                e8_decode_index(cw0, p0);
                e8_decode_index(cw1, p1);
                half16_t a_a;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    a_a[i]     = (_Float16)(p0[i] * ssc);
                    a_a[8 + i] = (_Float16)(p1[i] * ssc);
                }
                half16_t b = (xg != nullptr) ? *(const half16_t*)(xg + k_off) : zero_b;
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
            }
        }
    } else if (tag == 5) {
        // ----- mfp3-E8 (16 B row hdr + (K/32)*13 B; 104 B/group) ------------
        // Same structure as tag 4 but block_stride=13, group_stride=104,
        // codewords are 3 bytes each (narrow assembly read, no over-read).
        const int row_block_bytes = (K / 32) * 13;
        const char* e8_row_ptr =
            A + (long long)safe_row * (16 + (long long)row_block_bytes);
        const float row_scale_f =
            (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
        const char* blocks_ptr = e8_row_ptr + 16;
        for (int g = 0; g < groups_per_row; g++) {
            const char* group_ptr = blocks_ptr + (long long)g * 104;
            const _Float16* xg = (x_base != nullptr) ? (x_base + (long long)g * 256) : nullptr;
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                const int block_idx = k_off >> 5;            // 0..7
                const char* block_ptr = group_ptr + (long long)block_idx * 13;
                const float ssc = row_scale_f
                    * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * QUANT_STEP_MFP3;
                const int cw_base = (k_off & 31) >> 3;        // 0 or 2
                // 3-byte narrow reads — the last cw1 of the last block of the last
                // group would over-read by 1 B if done as *(uint*). Byte-assembly
                // reads exactly 3 bytes; result is identical, no page fault risk.
                const unsigned char* w0 = (const unsigned char*)(block_ptr + 1 + cw_base * 3);
                const unsigned char* w1 = (const unsigned char*)(block_ptr + 1 + (cw_base + 1) * 3);
                const unsigned int cw0 = (unsigned)w0[0] | ((unsigned)w0[1] << 8) | ((unsigned)w0[2] << 16);
                const unsigned int cw1 = (unsigned)w1[0] | ((unsigned)w1[1] << 8) | ((unsigned)w1[2] << 16);
                float p0[8], p1[8];
                mfp3_decode_index(cw0, p0);
                mfp3_decode_index(cw1, p1);
                half16_t a_a;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    a_a[i]     = (_Float16)(p0[i] * ssc);
                    a_a[8 + i] = (_Float16)(p1[i] * ssc);
                }
                half16_t b = (xg != nullptr) ? *(const half16_t*)(xg + k_off) : zero_b;
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
            }
        }
    } else if (tag == 6) {
        // ----- mfp2-E8 (16 B row hdr + (K/32)*9 B; 72 B/group) -------------
        // Same structure as tag 4 but block_stride=9, group_stride=72,
        // codewords are 2 bytes each (narrow assembly read, no over-read).
        const int row_block_bytes = (K / 32) * 9;
        const char* e8_row_ptr =
            A + (long long)safe_row * (16 + (long long)row_block_bytes);
        const float row_scale_f =
            (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
        const char* blocks_ptr = e8_row_ptr + 16;
        for (int g = 0; g < groups_per_row; g++) {
            const char* group_ptr = blocks_ptr + (long long)g * 72;
            const _Float16* xg = (x_base != nullptr) ? (x_base + (long long)g * 256) : nullptr;
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                const int block_idx = k_off >> 5;            // 0..7
                const char* block_ptr = group_ptr + (long long)block_idx * 9;
                const float ssc = row_scale_f
                    * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * QUANT_STEP_MFP2;
                const int cw_base = (k_off & 31) >> 3;        // 0 or 2
                // 2-byte narrow reads — last cw1 would over-read by 2 B as *(uint*).
                const unsigned char* w0 = (const unsigned char*)(block_ptr + 1 + cw_base * 2);
                const unsigned char* w1 = (const unsigned char*)(block_ptr + 1 + (cw_base + 1) * 2);
                const unsigned int cw0 = (unsigned)w0[0] | ((unsigned)w0[1] << 8);
                const unsigned int cw1 = (unsigned)w1[0] | ((unsigned)w1[1] << 8);
                float p0[8], p1[8];
                mfp2_decode_index(cw0, p0);
                mfp2_decode_index(cw1, p1);
                half16_t a_a;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    a_a[i]     = (_Float16)(p0[i] * ssc);
                    a_a[8 + i] = (_Float16)(p1[i] * ssc);
                }
                half16_t b = (xg != nullptr) ? *(const half16_t*)(xg + k_off) : zero_b;
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
            }
        }
    }

    // -------------------------------------------------------------------------
    // RDNA3 wave32 WMMA output mapping: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // Direct store (=) — Y_grouped is fresh per call.
    // Padding lanes (slot_idx >= m_total) are silently dropped.
    // -------------------------------------------------------------------------
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
