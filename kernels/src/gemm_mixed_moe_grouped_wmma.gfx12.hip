#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

// Merged dtype-tag MoE batched-prefill grouped-WMMA kernel — gfx12 (RDNA4).
//
// gfx12 sister of `gemm_mixed_moe_grouped_wmma_k2.hip`. Same 4-branch
// block-uniform tag dispatch; gfx12 WMMA differences:
//   - WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//   - Operands: half8_t (8 fp16/lane) instead of half16_t.
//   - K dim split across 2 lane-groups (k_grp = tid >> 4).
//   - C-output mapping: acc[j] = C[8*k_grp + j][m_lane].
//   - Inner loop unroll: K4 (4 K-tiles per body) instead of K2.
//
// Tag map (same as decode merged + gfx11 merged):
//   0 = MQ6 / HFQ6-G256  — 200 B/group, 6-bit affine (sc*q+zp)
//   1 = MQ2-Lloyd-G256   —  72 B/group, 4-entry fp16 codebook (2-bit)  [NET-NEW gfx12 port]
//   2 = MQ4 / HFQ4-G256  — 136 B/group, 4-bit affine (sc*q+zp)
//   3 = MQ3-Lloyd-G256   — 112 B/group, 8-entry fp16 codebook (3-bit)
//   4 = mfp4-E8          — 16 B row hdr + (K/32)*17 B; 136 B/group, E8 VQ
//
// CRITICAL (tags 1 & 3): per-lane REGISTER bilinear/trilinear decomposition
// — never LDS broadcast. Each m_lane handles a different M-row with its own
// per-group codebook. LDS broadcast cross-contaminates lanes.
//
// Dequant sources:
//   tag 0: gemm_hfq6g256_moe_grouped_wmma.gfx12.hip, lines 95–151 (lifted verbatim)
//   tag 1: net-new gfx12 MQ2-Lloyd port (no prior gfx12 source; authored inline)
//   tag 2: gemm_hfq4g256_moe_grouped_wmma.gfx12.hip, lines 78–126 (lifted verbatim)
//   tag 3: gemm_mq3g256_lloyd_moe_grouped_wmma.gfx12.hip, lines 92–164 (lifted verbatim)
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// mfp4-E8 (tag 4) decode helpers — byte-identical to
// gemm_mfp4g32_e8_moe_grouped_wmma.gfx12.hip.
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
extern "C" __global__ void gemm_mixed_moe_grouped_wmma_gfx12(
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

    // BLOCK-UNIFORM tag read — all 32 threads in the block share one expert.
    const unsigned char tag = dtype_tags[expert_id];
    // group_bytes MUST be computed before row_base (row_base depends on it).
    const int group_bytes = (tag == 0) ? 200 : (tag == 1) ? 72 : (tag == 2) ? 136 : 112;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather (per m_lane; both k_grp lanes for the same m_lane read the
    // same source row). -1 padding lanes contribute a zero B row.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to the last valid M-row (output mask
    // discards the result anyway).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * group_bytes;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    // -------------------------------------------------------------------------
    // Tag-dispatched inner loop (block-uniform → zero warp divergence).
    // -------------------------------------------------------------------------

    if (tag == 0) {
        // ----- MQ6 / HFQ6-G256 (200 B/group) --------------------------------
        // Source: gemm_hfq6g256_moe_grouped_wmma.gfx12.hip, lines 95–151.
        // Layout: 4 B scale + 4 B zero + 192 B packed 6-bit.
        // Per K-tile (16 values): 12 B. Per k_grp half: 6 B at kt*12 + k_grp*6.
        // Two overlapping uint32 reads (+0/+3) extract 8 × 6-bit values.

        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_8(d0, d1) \
            DQ6(0, (d0), 0);  DQ6(1, (d0), 6);  DQ6(2, (d0), 12); DQ6(3, (d0), 18); \
            DQ6(4, (d1), 0);  DQ6(5, (d1), 6);  DQ6(6, (d1), 12); DQ6(7, (d1), 18)

        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 200;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            for (int kt = 0; kt < 16; kt += 4) {
                const int bb_a = (kt + 0) * 12 + k_grp * 6;
                const int bb_b = (kt + 1) * 12 + k_grp * 6;
                const int bb_c = (kt + 2) * 12 + k_grp * 6;
                const int bb_d = (kt + 3) * 12 + k_grp * 6;
                const unsigned char* dp_a = (const unsigned char*)(gp + 8 + bb_a);
                const unsigned char* dp_b = (const unsigned char*)(gp + 8 + bb_b);
                const unsigned char* dp_c = (const unsigned char*)(gp + 8 + bb_c);
                const unsigned char* dp_d = (const unsigned char*)(gp + 8 + bb_d);
                // d1 reads only bytes [+3..+5] (DQ6_8 uses bits 0..23 = 3 bytes).
                // Read exactly 3 bytes — a `*(uint*)(dp+3)` would fetch a 4th byte
                // (dp+6) that is unused but reads 1 B past the group on the last
                // tile of the last group → GPU VM read fault on tightly-packed
                // graded expert buffers. Bit-identical result, no over-read.
                unsigned int d0a = *(const unsigned int*)(dp_a);
                unsigned int d1a = (unsigned int)dp_a[3] | ((unsigned int)dp_a[4] << 8) | ((unsigned int)dp_a[5] << 16);
                unsigned int d0b = *(const unsigned int*)(dp_b);
                unsigned int d1b = (unsigned int)dp_b[3] | ((unsigned int)dp_b[4] << 8) | ((unsigned int)dp_b[5] << 16);
                unsigned int d0c = *(const unsigned int*)(dp_c);
                unsigned int d1c = (unsigned int)dp_c[3] | ((unsigned int)dp_c[4] << 8) | ((unsigned int)dp_c[5] << 16);
                unsigned int d0d = *(const unsigned int*)(dp_d);
                unsigned int d1d = (unsigned int)dp_d[3] | ((unsigned int)dp_d[4] << 8) | ((unsigned int)dp_d[5] << 16);

                half8_t b_a, b_b, b_c, b_d;
                if (xg != nullptr) {
                    b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
                    b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
                    b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
                    b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);
                } else {
                    b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b;
                }

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
        }
        #undef DQ6_8
        #undef DQ6

    } else if (tag == 1) {
        // ----- MQ2-Lloyd-G256 (72 B/group) --- net-new gfx12 port -----------
        // No prior gfx12 source; authored by composing the gfx11 MQ2-Lloyd
        // bilinear form with the gfx12 K4-unroll / half8_t / k_grp layout.
        //
        // Layout: 8 B (4×fp16 codebook, ascending) + 64 B 2-bit packed.
        // Per K-tile (16 weights): 32 bits = 1 uint32 at gp+8+kt*4.
        //   (16 × 2 bits = 32 bits; gfx11: gp+8+k_off/4 where k_off=kt*16
        //    → gp+8+kt*4. Same offset for gfx12.)
        // Per k_grp half (8 weights): bits [k_grp*16 .. k_grp*16+16) of
        //   that uint32.  Shift the packed word right by k_grp*16, then
        //   the low 8 values live at shifts 0,2,4,...,14.
        //
        // Per-lane REGISTER bilinear decomposition (NOT LDS broadcast).

        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 72;
            const __half* cb_h = reinterpret_cast<const __half*>(gp);
            _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
            _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
            _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
            _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
            // Bilinear decomposition: q = qhi*2 + qlo; avoids ternary-chain.
            const _Float16 d01   = cb1 - cb0;
            const _Float16 d02   = cb2 - cb0;
            const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;
            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            // DQ2_8: decode 8 consecutive 2-bit weights from a shifted uint32.
            // `pk` is the raw uint32 already shifted right by k_grp*16 so
            // the low 8 values occupy bits 0..15.
            #define DQ2_8(d, pkraw) do { \
                unsigned int pk = (pkraw) >> (k_grp * 16); \
                _Float16 q0lo=(_Float16)(int)((pk>> 0)&1u),q0hi=(_Float16)(int)((pk>> 1)&1u); (d)[0]=cb0+q0lo*d01+q0hi*d02+(q0lo*q0hi)*d_xor; \
                _Float16 q1lo=(_Float16)(int)((pk>> 2)&1u),q1hi=(_Float16)(int)((pk>> 3)&1u); (d)[1]=cb0+q1lo*d01+q1hi*d02+(q1lo*q1hi)*d_xor; \
                _Float16 q2lo=(_Float16)(int)((pk>> 4)&1u),q2hi=(_Float16)(int)((pk>> 5)&1u); (d)[2]=cb0+q2lo*d01+q2hi*d02+(q2lo*q2hi)*d_xor; \
                _Float16 q3lo=(_Float16)(int)((pk>> 6)&1u),q3hi=(_Float16)(int)((pk>> 7)&1u); (d)[3]=cb0+q3lo*d01+q3hi*d02+(q3lo*q3hi)*d_xor; \
                _Float16 q4lo=(_Float16)(int)((pk>> 8)&1u),q4hi=(_Float16)(int)((pk>> 9)&1u); (d)[4]=cb0+q4lo*d01+q4hi*d02+(q4lo*q4hi)*d_xor; \
                _Float16 q5lo=(_Float16)(int)((pk>>10)&1u),q5hi=(_Float16)(int)((pk>>11)&1u); (d)[5]=cb0+q5lo*d01+q5hi*d02+(q5lo*q5hi)*d_xor; \
                _Float16 q6lo=(_Float16)(int)((pk>>12)&1u),q6hi=(_Float16)(int)((pk>>13)&1u); (d)[6]=cb0+q6lo*d01+q6hi*d02+(q6lo*q6hi)*d_xor; \
                _Float16 q7lo=(_Float16)(int)((pk>>14)&1u),q7hi=(_Float16)(int)((pk>>15)&1u); (d)[7]=cb0+q7lo*d01+q7hi*d02+(q7lo*q7hi)*d_xor; \
            } while (0)

            for (int kt = 0; kt < 16; kt += 4) {
                // 1 uint32 per K-tile. Byte offset: gp+8+kt*4.
                // (16×2 bits = 32 bits = 4 B per K-tile; matches gfx11 k_off/4.)
                unsigned int pka = *(const unsigned int*)(gp + 8 + (kt+0) * 4);
                unsigned int pkb = *(const unsigned int*)(gp + 8 + (kt+1) * 4);
                unsigned int pkc = *(const unsigned int*)(gp + 8 + (kt+2) * 4);
                unsigned int pkd = *(const unsigned int*)(gp + 8 + (kt+3) * 4);

                half8_t b_a, b_b, b_c, b_d;
                if (xg != nullptr) {
                    b_a = *(const half8_t*)(xg + (kt+0)*16 + k_grp*8);
                    b_b = *(const half8_t*)(xg + (kt+1)*16 + k_grp*8);
                    b_c = *(const half8_t*)(xg + (kt+2)*16 + k_grp*8);
                    b_d = *(const half8_t*)(xg + (kt+3)*16 + k_grp*8);
                } else { b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b; }

                half8_t a_reg;
                DQ2_8(a_reg, pka); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
                DQ2_8(a_reg, pkb); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
                DQ2_8(a_reg, pkc); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
                DQ2_8(a_reg, pkd); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
            }
            #undef DQ2_8
        }

    } else if (tag == 2) {
        // ----- MQ4 / HFQ4-G256 (136 B/group) --------------------------------
        // Source: gemm_hfq4g256_moe_grouped_wmma.gfx12.hip, lines 78–126.
        // Layout: 4 B scale + 4 B zero + 128 B 4-bit packed.
        // Per K-tile: 8 B. Per k_grp half: 1 uint32 at gp+8+k_off/2+k_grp*4.

        #define DQ8(pk) \
            a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_base + g * 136;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            for (int kt = 0; kt < 16; kt += 4) {
                const int k_off_a = (kt + 0) * 16;
                const int k_off_b = (kt + 1) * 16;
                const int k_off_c = (kt + 2) * 16;
                const int k_off_d = (kt + 3) * 16;

                unsigned int pka = *(const unsigned int*)(gp + 8 + k_off_a / 2 + k_grp * 4);
                unsigned int pkb = *(const unsigned int*)(gp + 8 + k_off_b / 2 + k_grp * 4);
                unsigned int pkc = *(const unsigned int*)(gp + 8 + k_off_c / 2 + k_grp * 4);
                unsigned int pkd = *(const unsigned int*)(gp + 8 + k_off_d / 2 + k_grp * 4);

                half8_t b_a, b_b, b_c, b_d;
                if (xg != nullptr) {
                    b_a = *(const half8_t*)(xg + k_off_a + k_grp * 8);
                    b_b = *(const half8_t*)(xg + k_off_b + k_grp * 8);
                    b_c = *(const half8_t*)(xg + k_off_c + k_grp * 8);
                    b_d = *(const half8_t*)(xg + k_off_d + k_grp * 8);
                } else {
                    b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b;
                }

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
        }
        #undef DQ8

    } else if (tag == 3) {
        // ----- MQ3-Lloyd-G256 (112 B/group) ---------------------------------
        // Source: gemm_mq3g256_lloyd_moe_grouped_wmma.gfx12.hip, lines 92–164.
        // Layout: 16 B (8×fp16 codebook) + 96 B 3-bit packed.
        // Per K-tile (16 weights): 6 B. Per k_grp half: 3 B at kt*6 + k_grp*3.
        // Per-lane REGISTER 3D bilinear decomposition (NOT LDS broadcast).

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

            // 3D bilinear coefficients (exact for any 8-entry table).
            const _Float16 coA = cb0;
            const _Float16 coB = cb1 - cb0;
            const _Float16 coC = cb2 - cb0;
            const _Float16 coD = cb4 - cb0;
            const _Float16 coE = cb3 - cb2 - cb1 + cb0;
            const _Float16 coF = cb5 - cb4 - cb1 + cb0;
            const _Float16 coG = cb6 - cb4 - cb2 + cb0;
            const _Float16 coH = cb7 - cb6 - cb5 + cb4 - cb3 + cb2 + cb1 - cb0;

            const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

            // K4 unroll: 4 K-tiles per inner body.
            // Each k_grp lane reads 3 B at gp+16+kt*6+k_grp*3.
            for (int kt = 0; kt < 16; kt += 4) {
                const unsigned char* dp_a = (const unsigned char*)(gp + 16 + (kt+0)*6 + k_grp*3);
                const unsigned char* dp_b = (const unsigned char*)(gp + 16 + (kt+1)*6 + k_grp*3);
                const unsigned char* dp_c = (const unsigned char*)(gp + 16 + (kt+2)*6 + k_grp*3);
                const unsigned char* dp_d = (const unsigned char*)(gp + 16 + (kt+3)*6 + k_grp*3);
                unsigned int pa = (unsigned int)dp_a[0]|((unsigned int)dp_a[1]<<8)|((unsigned int)dp_a[2]<<16);
                unsigned int pb = (unsigned int)dp_b[0]|((unsigned int)dp_b[1]<<8)|((unsigned int)dp_b[2]<<16);
                unsigned int pc = (unsigned int)dp_c[0]|((unsigned int)dp_c[1]<<8)|((unsigned int)dp_c[2]<<16);
                unsigned int pd = (unsigned int)dp_d[0]|((unsigned int)dp_d[1]<<8)|((unsigned int)dp_d[2]<<16);

                half8_t b_a, b_b, b_c, b_d;
                if (xg != nullptr) {
                    b_a = *(const half8_t*)(xg + (kt+0)*16 + k_grp*8);
                    b_b = *(const half8_t*)(xg + (kt+1)*16 + k_grp*8);
                    b_c = *(const half8_t*)(xg + (kt+2)*16 + k_grp*8);
                    b_d = *(const half8_t*)(xg + (kt+3)*16 + k_grp*8);
                } else { b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b; }

                #define DQ3(d, pk, sh) do { \
                    _Float16 b0 = (_Float16)(int)(((pk) >> (sh))     & 1u); \
                    _Float16 b1 = (_Float16)(int)(((pk) >> ((sh)+1)) & 1u); \
                    _Float16 b2 = (_Float16)(int)(((pk) >> ((sh)+2)) & 1u); \
                    (d) = coA + b0*coB + b1*coC + b2*coD + (b0*b1)*coE + (b0*b2)*coF + (b1*b2)*coG + (b0*b1*b2)*coH; \
                } while (0)

                #define DQ3_8(d, pk) \
                    DQ3((d)[0], (pk),  0); DQ3((d)[1], (pk),  3); \
                    DQ3((d)[2], (pk),  6); DQ3((d)[3], (pk),  9); \
                    DQ3((d)[4], (pk), 12); DQ3((d)[5], (pk), 15); \
                    DQ3((d)[6], (pk), 18); DQ3((d)[7], (pk), 21)

                half8_t a_reg;
                DQ3_8(a_reg, pa); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
                DQ3_8(a_reg, pb); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
                DQ3_8(a_reg, pc); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
                DQ3_8(a_reg, pd); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
                #undef DQ3_8
                #undef DQ3
            }
        }
    } else if (tag == 4) {
        // ----- mfp4-E8 (16 B row hdr + (K/32)*17 B; 136 B/group) ------------
        // Source: gemm_mfp4g32_e8_moe_grouped_wmma.gfx12.hip (lines 96–132).
        // E8 carries a 16-byte row header (f16 row_scale) before the groups,
        // so it computes its own row/block pointers — the shared row_base /
        // group_bytes assume a header-less group-major layout.
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
                const unsigned int cw =
                    *(const unsigned int*)(block_ptr + 1 + (cw_base + k_grp) * 4);
                float p[8];
                e8_decode_index(cw, p);
                half8_t a_reg;
                #pragma unroll
                for (int i = 0; i < 8; i++) a_reg[i] = (_Float16)(p[i] * ssc);
                half8_t b = (xg != nullptr)
                    ? *(const half8_t*)(xg + k_off + k_grp * 8) : zero_b;
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b, acc);
            }
        }
    } else if (tag == 5) {
        // ----- mfp3-E8 (16 B row hdr + (K/32)*13 B; 104 B/group) ------------
        // Same structure as tag 4 but block_stride=13, group_stride=104, 3-byte codewords.
        // k_grp selects which of the two 3-byte codewords to decode:
        //   offset = block_ptr + 1 + (cw_base + k_grp) * 3  (narrow 3-byte read).
        // The last cw (cw_base+1, k_grp=1) of the last block of the last group
        // sits at block_ptr+1+(2+1)*3=+10 within a 13-byte block → no over-read.
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
                // Narrow 3-byte read — safe for all positions including last block.
                const unsigned char* w = (const unsigned char*)(block_ptr + 1 + (cw_base + k_grp) * 3);
                const unsigned int cw = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
                float p[8];
                mfp3_decode_index(cw, p);
                half8_t a_reg;
                #pragma unroll
                for (int i = 0; i < 8; i++) a_reg[i] = (_Float16)(p[i] * ssc);
                half8_t b = (xg != nullptr)
                    ? *(const half8_t*)(xg + k_off + k_grp * 8) : zero_b;
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b, acc);
            }
        }
    } else if (tag == 6) {
        // ----- mfp2-E8 (16 B row hdr + (K/32)*9 B; 72 B/group) -------------
        // Same structure as tag 4 but block_stride=9, group_stride=72, 2-byte codewords.
        // k_grp selects which 2-byte codeword: offset = (cw_base + k_grp) * 2.
        // Last cw (cw_base=2, k_grp=1) sits at block_ptr+1+3*2=+7 within 9-byte block
        // and reads 2 bytes (+7,+8) → exactly within the block, no over-read.
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
                // Narrow 2-byte read — safe for all positions.
                const unsigned char* w = (const unsigned char*)(block_ptr + 1 + (cw_base + k_grp) * 2);
                const unsigned int cw = (unsigned)w[0] | ((unsigned)w[1] << 8);
                float p[8];
                mfp2_decode_index(cw, p);
                half8_t a_reg;
                #pragma unroll
                for (int i = 0; i < 8; i++) a_reg[i] = (_Float16)(p[i] * ssc);
                half8_t b = (xg != nullptr)
                    ? *(const half8_t*)(xg + k_off + k_grp * 8) : zero_b;
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b, acc);
            }
        }
    }

    // -------------------------------------------------------------------------
    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Direct store (=) — Y_grouped is fresh per call.
    // -------------------------------------------------------------------------
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
