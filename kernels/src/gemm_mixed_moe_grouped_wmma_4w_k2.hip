#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

// Merged dtype-tag MoE batched-prefill grouped-WMMA kernel — gfx11 (RDNA3/3.5),
// 4-warp LDS-TILED variant of `gemm_mixed_moe_grouped_wmma_k2`.
//
// LDS-TILING LEVER (load-bound prefill fix). The single-warp `_k2` kernel runs
// `LDS: 0`: every block re-reads its 16-slot X tile straight from L2, and for a
// fixed slot-tile that X tile is re-read once per output-row-tile (ceil(M/16)x —
// ~48x for gate_up, ~128x for down). That re-read is the bottleneck (the kernel
// sits at ~16% of fp16 WMMA peak; it is load-bound, not dequant-bound). This
// variant stages the 16-slot X tile into LDS ONCE per K-group and shares it
// across 4 cooperating warps (= 4 row-tiles), so each X load amortizes over 4x
// the output rows: ceil(M/16) -> ceil(M/64) X re-reads, i.e. 4x less B-fragment
// traffic per FLOP. Raises arithmetic intensity past the roofline ridge.
//
// Skeleton (4-warp LDS-X cooperative load + output map) lifted verbatim from
// `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2.hip` (Woolmer). Per-tag weight
// decode lifted verbatim from `gemm_mixed_moe_grouped_wmma_k2.hip`; the ONLY
// change to each decode is that the B (X) fragment is read from LDS
// (`lds_slot + k_off`) instead of global (`xg + k_off`). Invalid slots are
// zero-filled at the cooperative load, so the inner loops need no null checks.
//
// Tag map (block-uniform — all 16 slots in a block share one expert, so every
// thread reads the same tag; zero warp divergence):
//   0 = MQ6/HFQ6-G256  — 200 B/group, 6-bit affine
//   1 = MQ2-Lloyd-G256 —  72 B/group, 4-entry fp16 codebook
//   2 = MQ4/HFQ4-G256  — 136 B/group, 4-bit affine
//   3 = MQ3-Lloyd-G256 — 112 B/group, 8-entry fp16 codebook
//   4 = mfp4-E8        — 16 B row hdr + (K/32)*17 B; 136 B/group, E8 VQ
//   5 = mfp3-E8        — 16 B row hdr + (K/32)*13 B; 104 B/group, 3-bit E8 VQ
//   6 = mfp2-E8        — 16 B row hdr + (K/32)* 9 B;  72 B/group, 2-bit E8 VQ
//
// Per-block geometry: 128 threads (4 warps x 32 lanes), 64 M-rows x 16 slots.
//   warp_id w owns M-rows [block_m + w*16, +16); the 16 X-slots are LDS-shared.
// WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (gfx11 _w32).
// Grid: [ceil(M/64), m_total/16]. Block: [128]. LDS: ~8 KB (X stage) + 64 B.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define N_TILE   16   // slots per block (one expert tile); pinned
#define WARPS    4
#define M_TILE   64   // WARPS * 16
#define K_GROUP  256

// mfp4-E8 (tag 4) decode helpers — byte-identical to
// gemm_mixed_moe_grouped_wmma_k2.hip / ..._e8_..._gfx1151.hip.
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

// mfpN-E8 narrow-nibble decode + QUANT_STEP — byte-identical to
// gemm_mixed_moe_grouped_wmma_k2.hip. QUANT_STEP MUST match e8.rs.
#define QUANT_STEP_MFP3 1.8f
#define QUANT_STEP_MFP2 3.8f
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

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_mixed_moe_grouped_wmma_4w_k2(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E] u64
    const unsigned char*      __restrict__ dtype_tags,         // [E] u8 tag per expert
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows x K]
    float*                    __restrict__ Y_grouped,          // [m_total x M]
    int M, int K, int x_row_div, int m_total
) {
    const int block_m    = blockIdx.x * M_TILE;
    const int tile_y     = blockIdx.y;
    const int slot_start = tile_y * N_TILE;
    if (block_m >= M || slot_start >= m_total) return;

    // Sentinel early-return: expert_tile_ids[tile_y] == -1 marks padding tiles.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    // BLOCK-UNIFORM tag (all 128 threads share one expert) -> no divergence.
    const unsigned char tag = dtype_tags[expert_id];
    // group_bytes for tags 0-3 (the E8 family — tags 4/5/6 — computes its own
    // header-aware pointer below and never consumes this value, so the tag>3
    // fallthrough to 112 is harmless/unused for them).
    const int group_bytes = (tag == 0) ? 200 : (tag == 1) ? 72 : (tag == 2) ? 136 : 112;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3
    const int lane    = tid & 31;   // 0..31
    const int lane16  = lane & 15;  // 0..15 (WMMA fragment lane = slot within tile)

    // Each warp owns 16 M-rows.
    const int warp_m_start = block_m + warp_id * 16;
    const int my_row       = warp_m_start + lane16;
    const int safe_row     = (my_row < M) ? my_row : (M - 1);
    const int groups_per_row = K / K_GROUP;

    // Cache the 16 slot->x_row mappings in LDS (shared; no 8x redundant loads).
    __shared__ int slot_x_row[N_TILE];
    if (tid < N_TILE) {
        const int slot_idx = slot_start + tid;
        int x_row = -1;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
        slot_x_row[tid] = x_row;
    }
    // LDS X stage: [16 slots x 256 K] = 8 KB per K-group.
    __shared__ _Float16 lds_x[N_TILE * K_GROUP];

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Shared row_base for tags 0-3 (tag 4 ignores it).
    const char* row_base = A + (long long)safe_row * groups_per_row * group_bytes;

    __syncthreads();  // make slot_x_row visible to all warps

    for (int g = 0; g < groups_per_row; g++) {
        // -- Cooperative X load: 128 threads -> 16 slots x 256 K (coalesced).
        //    slot_in_block = warp_id*4 + lane/8; each thread loads 32 fp16.
        {
            const int slot_in_block = warp_id * 4 + (lane >> 3);
            const int k_off         = (lane & 7) * 32;
            const int x_row         = slot_x_row[slot_in_block];
            half16_t va, vb;
            if (x_row >= 0) {
                const long long x_off = (long long)x_row * K + (long long)g * K_GROUP + k_off;
                va = *(const half16_t*)(X_src + x_off);
                vb = *(const half16_t*)(X_src + x_off + 16);
            } else {
                va = zero_b;
                vb = zero_b;
            }
            _Float16* dst = lds_x + slot_in_block * K_GROUP + k_off;
            *(half16_t*)(dst)      = va;
            *(half16_t*)(dst + 16) = vb;
        }
        __syncthreads();

        // This lane's slot X (lane16 = slot within the tile), from LDS.
        const _Float16* lds_slot = lds_x + lane16 * K_GROUP;

        // -- Per-warp tag-dispatched decode (block-uniform -> no divergence).
        //    Each branch == the single-warp _k2 decode, B read from LDS.
        if (tag == 0) {
            // ----- MQ6 / HFQ6-G256 (200 B/group) -----
            const char* gp = row_base + g * 200;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;
                const unsigned char* wa = (const unsigned char*)(gp + 8 + kt * 12);
                const unsigned char* wb = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
                unsigned int a0 = *(const unsigned int*)(wa);
                unsigned int a1 = *(const unsigned int*)(wa + 3);
                unsigned int a2 = *(const unsigned int*)(wa + 6);
                unsigned int a3 = (unsigned int)wa[9] | ((unsigned int)wa[10] << 8) | ((unsigned int)wa[11] << 16);
                unsigned int c0 = *(const unsigned int*)(wb);
                unsigned int c1 = *(const unsigned int*)(wb + 3);
                unsigned int c2 = *(const unsigned int*)(wb + 6);
                unsigned int c3 = (unsigned int)wb[9] | ((unsigned int)wb[10] << 8) | ((unsigned int)wb[11] << 16);
                half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
                half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);
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

        } else if (tag == 1) {
            // ----- MQ2-Lloyd-G256 (72 B/group) -----
            const char* gp = row_base + g * 72;
            const __half* cb_h = reinterpret_cast<const __half*>(gp);
            _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
            _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
            _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
            _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
            const _Float16 d01   = cb1 - cb0;
            const _Float16 d02   = cb2 - cb0;
            const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;
            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;
                unsigned int pa = *(const unsigned int*)(gp + 8 + k_off_a / 4);
                unsigned int pb = *(const unsigned int*)(gp + 8 + k_off_b / 4);
                half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
                half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);
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

        } else if (tag == 2) {
            // ----- MQ4 / HFQ4-G256 (136 B/group) -----
            const char* gp = row_base + g * 136;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;
                unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
                unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
                unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
                unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);
                half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
                half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);
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

        } else if (tag == 3) {
            // ----- MQ3-Lloyd-G256 (112 B/group) -----
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
            const _Float16 coA = cb0;
            const _Float16 coB = cb1 - cb0;
            const _Float16 coC = cb2 - cb0;
            const _Float16 coD = cb4 - cb0;
            const _Float16 coE = cb3 - cb2 - cb1 + cb0;
            const _Float16 coF = cb5 - cb4 - cb1 + cb0;
            const _Float16 coG = cb6 - cb4 - cb2 + cb0;
            const _Float16 coH = cb7 - cb6 - cb5 + cb4 - cb3 + cb2 + cb1 - cb0;
            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                const int k_off_b = (kt + 1) * 16;
                const unsigned char* wa = (const unsigned char*)(gp + 16 + kt       * 6);
                const unsigned char* wb = (const unsigned char*)(gp + 16 + (kt + 1) * 6);
                unsigned int p0a = (unsigned int)wa[0] | ((unsigned int)wa[1] << 8) | ((unsigned int)wa[2] << 16);
                unsigned int p1a = (unsigned int)wa[3] | ((unsigned int)wa[4] << 8) | ((unsigned int)wa[5] << 16);
                unsigned int p0b = (unsigned int)wb[0] | ((unsigned int)wb[1] << 8) | ((unsigned int)wb[2] << 16);
                unsigned int p1b = (unsigned int)wb[3] | ((unsigned int)wb[4] << 8) | ((unsigned int)wb[5] << 16);
                half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
                half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);
                #define DQ3(reg, i, pk, sh) do { \
                    _Float16 b0 = (_Float16)(int)(((pk) >> (sh))     & 1u); \
                    _Float16 b1 = (_Float16)(int)(((pk) >> ((sh)+1)) & 1u); \
                    _Float16 b2 = (_Float16)(int)(((pk) >> ((sh)+2)) & 1u); \
                    (reg)[i] = coA + b0*coB + b1*coC + b2*coD + (b0*b1)*coE + (b0*b2)*coF + (b1*b2)*coG + (b0*b1*b2)*coH; \
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

        } else if (tag == 4) {
            // ----- mfp4-E8 (16 B row hdr + (K/32)*17 B; 136 B/group) -----
            // E8 carries a 16-byte row header before the groups, so it computes
            // its own header-aware pointers (shared row_base assumes a
            // header-less group-major layout). half16: each 16-K WMMA tile is
            // two E8 codewords (cw0 -> a[0..8), cw1 -> a[8..16)).
            const int row_block_bytes = (K / 32) * 17;
            const char* e8_row_ptr =
                A + (long long)safe_row * (16 + (long long)row_block_bytes);
            const float row_scale_f =
                (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
            const char* group_ptr = e8_row_ptr + 16 + (long long)g * 136;
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
                half16_t b = *(const half16_t*)(lds_slot + k_off);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
            }
        } else if (tag == 5) {
            // ----- mfp3-E8 (16 B row hdr + (K/32)*13 B; 104 B/group) -----
            // Same header-aware layout as tag 4; narrow 3-byte codeword reads
            // (last cw of last block reads bytes [10..12] within the 13-B block).
            const int row_block_bytes = (K / 32) * 13;
            const char* e8_row_ptr =
                A + (long long)safe_row * (16 + (long long)row_block_bytes);
            const float row_scale_f =
                (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
            const char* group_ptr = e8_row_ptr + 16 + (long long)g * 104;
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                const int block_idx = k_off >> 5;            // 0..7
                const char* block_ptr = group_ptr + (long long)block_idx * 13;
                const float ssc = row_scale_f
                    * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * QUANT_STEP_MFP3;
                const int cw_base = (k_off & 31) >> 3;        // 0 or 2
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
                half16_t b = *(const half16_t*)(lds_slot + k_off);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
            }
        } else if (tag == 6) {
            // ----- mfp2-E8 (16 B row hdr + (K/32)*9 B; 72 B/group) -----
            // Same header-aware layout as tag 4; narrow 2-byte codeword reads
            // (last cw of last block reads bytes [7..8] within the 9-B block).
            const int row_block_bytes = (K / 32) * 9;
            const char* e8_row_ptr =
                A + (long long)safe_row * (16 + (long long)row_block_bytes);
            const float row_scale_f =
                (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
            const char* group_ptr = e8_row_ptr + 16 + (long long)g * 72;
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                const int block_idx = k_off >> 5;            // 0..7
                const char* block_ptr = group_ptr + (long long)block_idx * 9;
                const float ssc = row_scale_f
                    * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * QUANT_STEP_MFP2;
                const int cw_base = (k_off & 31) >> 3;        // 0 or 2
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
                half16_t b = *(const half16_t*)(lds_slot + k_off);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
            }
        }
        __syncthreads();  // release LDS X stage before next K-group's load
    }

    // -- Output writes (per-warp). RDNA3 wave32 WMMA: acc[j] on lane L holds
    //    C[2*j + (L>>4)][L & 15]. Each warp owns M-rows [warp_m_start, +16).
    const int out_col = slot_start + lane16;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = warp_m_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
