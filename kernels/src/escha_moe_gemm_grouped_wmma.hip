// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Escha-W2 expert-grouped routed GEMM on the RDNA3 matrix cores.
//
// WHY THIS EXISTS
// ---------------
// The scalar grouped GEMM (`escha_moe_gemm_grouped.hip`) fixed prefill's
// WEIGHT traffic — each expert's trellis code is read once per (layer, batch)
// instead of once per (token, expert) slot, which took n=512 prefill from
// 4.525 to 2.657 ms/token. But that made weight traffic 4% of prefill time,
// and left the other 96% running scalar `acc += w * x` FMAs on the VALU:
//
//     escha  (scalar grouped) 1360 ms / 512 tok -> 1.75 TFLOP/s
//     ornith (WMMA grouped)    520 ms / 512 tok -> 4.59 TFLOP/s
//
// Same 2.6x as the throughput gap. Prefill is COMPUTE bound, so the fix is
// the matrix cores, not more bandwidth work.
//
// WHY THE TILE SHAPE IS A GIFT
// ----------------------------
// Escha codes a 16x16 weight tile as its unit. WMMA on gfx11 is 16x16x16.
// One decoded escha tile is therefore exactly one A fragment — no repacking,
// no padding, no partial-tile bookkeeping in the K direction.
//
// ORIENTATION
// -----------
// The GEMM is `y[slot, m] = sum_k x[slot, k] * W[k, m]`, and escha stores W
// IN-major: tile (kt, nt) holds rows `kt*16 + r` (contraction) by columns
// `nt*16 + c` (output). WMMA wants A as [m x k], so lane `l` takes COLUMN `l`
// of the decoded tile — i.e. `W[kt*16 + 0..15, nt*16 + l]`. Decoding into LDS
// first makes that a plain strided read; decoding straight to registers would
// need a cross-lane transpose for no benefit.
//
// NUMERICS
// --------
// WMMA accumulates in f32 over a different partition of the contraction than
// the scalar path, so this is NOT bit-identical to it, exactly as the scalar
// grouped path is not bit-identical to the slot-parallel one. The weight
// VALUES are the same decoded fp16 either way — only the summation order
// moves. Gated by measured bound, not by equality; see the gate.
//
// The codebook stays inline (no LUT): 65536 fp16 entries is 128 KB and
// gfx1151 has 64 KB of LDS.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// ── codec (transcribed from escha_decode_tiles.hip; see the deliberate
// duplication note there — the CPU reference is the gate for both) ─────────

__device__ __forceinline__ __half escha_w_cba(unsigned short state) {
    unsigned int r = ((unsigned int)state * 0xCBAC1FEDu) & 0x8FFF8FFFu;
    r ^= 0x3B603B60u;
    __half lo = __ushort_as_half((unsigned short)(r & 0xFFFFu));
    __half hi = __ushort_as_half((unsigned short)(r >> 16));
    return __hadd(lo, hi);
}

__device__ __forceinline__ unsigned int escha_w_word(
    const short* __restrict__ src, int i) {
    return ((unsigned int)(unsigned short)src[2 * i]) |
           (((unsigned int)(unsigned short)src[2 * i + 1]) << 16);
}

__device__ __forceinline__ void escha_w_dec8_k2(
    const short* __restrict__ src, int lane, unsigned short* out) {
    int t_off = lane * 8;
    int i1 = t_off >> 4;
    int i0 = (i1 + 15) & 15;
    unsigned long long merged =
        ((unsigned long long)escha_w_word(src, i0) << 32) | escha_w_word(src, i1);
    int shift = ((~t_off) & 8) << 1;
    unsigned int v = (unsigned int)((merged >> shift) & 0xFFFFFFFFull);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (unsigned short)(v >> (2 * (7 - j)));
}

__device__ __forceinline__ void escha_w_dec8_k3(
    const short* __restrict__ src, int lane, unsigned short* out) {
    const int BITS = 3;
    int t_off = lane * 8;
    int b1 = (t_off + 257) * BITS;
    int b0 = b1 - 16;
    int b2 = b1 + BITS * 7;
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    int s2 = ((i2 + 1) << 5) - b2;
    unsigned long long merged = ((unsigned long long)escha_w_word(src, i0 % 24) << 32) |
                                escha_w_word(src, i2 % 24);
    unsigned int w7 = (unsigned int)((merged >> s2) & 0xFFFFFFFFull);
    unsigned int w3 = (unsigned int)((merged >> (s2 + BITS * 4)) & 0xFFFFFFFFull);
    out[0] = (unsigned short)(w3 >> 9);
    out[1] = (unsigned short)(w3 >> 6);
    out[2] = (unsigned short)(w3 >> 3);
    out[3] = (unsigned short)(w3);
    out[4] = (unsigned short)(w7 >> 9);
    out[5] = (unsigned short)(w7 >> 6);
    out[6] = (unsigned short)(w7 >> 3);
    out[7] = (unsigned short)(w7);
}

/// Decode one 16x16 escha tile into `dst` (row-major, 16 rows of `stride`).
/// All 32 lanes of the wave participate: 32 lanes x 8 values = the 256 slots.
template <int TK>
__device__ __forceinline__ void escha_decode_tile_lds(
    const short* __restrict__ code_tile, _Float16* dst, int stride, int lane) {
    unsigned short st[8];
    if (TK == 2) escha_w_dec8_k2(code_tile, lane, st);
    else         escha_w_dec8_k3(code_tile, lane, st);

    const int l0 = lane & ~4;
    const int c_off = (lane >> 2) & 1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int fi = j >> 1;
        const int row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8;
        const int col = 2 * ((l0 >> 3) + (j >= 4 ? 4 : 0)) + c_off;
        dst[row * stride + col] = (_Float16)escha_w_cba(st[j]);
    }
}

// ── the GEMM ──────────────────────────────────────────────────────────────
//
// grid.x = output tile index (M / 16)
// grid.y = expert id
// block  = 32 lanes (one wave32)
//
// Each block owns one 16-wide output column tile for one expert, and walks
// that expert's slot group 16 slots at a time. `+1` on the LDS strides keeps
// the column reads off a single bank.

#define ESCHA_WMMA_GEMM(NAME, TK, NT)                                          \
    __launch_bounds__(32) extern "C" __global__ void NAME(                     \
        const unsigned long long* __restrict__ expert_ptrs,                    \
        const int* __restrict__ expert_offsets,                                \
        const int* __restrict__ sorted_slot_index,                             \
        const float* __restrict__ x_batch,                                     \
        float* __restrict__ y_batch,                                           \
        int M, int K) {                                                        \
        const int nt = blockIdx.x;                                             \
        const int e = blockIdx.y;                                              \
        const int lane = threadIdx.x;                                          \
        const int g_start = expert_offsets[e];                                 \
        const int g_end = expert_offsets[e + 1];                               \
        if (g_start >= g_end) return;                                          \
                                                                               \
        const short* code = (const short*)expert_ptrs[e];                      \
        const int ktiles = K / 16;                                             \
        const int ntiles = M / 16;                                             \
                                                                               \
        __shared__ _Float16 w_lds[16 * 17];                                    \
                                                                               \
        for (int base = g_start; base < g_end; base += 16 * NT) {              \
            float8_t acc[NT];                                                  \
            _Pragma("unroll") for (int t = 0; t < NT; ++t)                     \
                acc[t] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   \
                                                                               \
            for (int kt = 0; kt < ktiles; ++kt) {                              \
                /* A: decode escha tile (kt, nt); lane takes column `lane`. */ \
                const short* tile = code + (size_t)(kt * ntiles + nt) * 16 * TK;\
                escha_decode_tile_lds<TK>(tile, w_lds, 17, lane);              \
                                                                               \
                /* Stage the weight tile AND all NT activation tiles, then a  \
                   SINGLE barrier. An earlier version put a __syncthreads()    \
                   inside the NT loop against one shared x_lds; that serialised\
                   the staging and measured 2.4x SLOWER than NT=1 (13.06 vs    \
                   5.40 ms) even though the numerics were identical. */        \
                __syncthreads();                                               \
                const int m_lane = lane & 15;                                  \
                half16_t a_frag;                                               \
                _Pragma("unroll") for (int i = 0; i < 16; ++i)                 \
                    a_frag[i] = w_lds[i * 17 + m_lane]; /* column = out col */ \
                                                                               \
                /* B comes STRAIGHT FROM GLOBAL, not via LDS. Lane `l` needs   \
                   x[slot_l, kt*16 .. +16] — already contiguous, and each lane \
                   wants a different slot, so there is nothing to share and    \
                   nothing to transpose. Staging it through LDS was measured   \
                   to cost 0.52 G LDS instructions against the scalar kernel's \
                   0.05 G, which exactly cancelled the 2.2x VALU saving WMMA   \
                   bought (busy cycles 1.48 G vs 1.47 G). */                   \
                _Pragma("unroll") for (int t = 0; t < NT; ++t) {               \
                    const int p = base + t * 16 + m_lane;                      \
                    const int s = (p < g_end) ? sorted_slot_index[p] : -1;      \
                    half16_t b_frag;                                           \
                    if (s >= 0) {                                              \
                        const float* xr = x_batch + (size_t)s * K + kt * 16;   \
                        _Pragma("unroll") for (int i = 0; i < 16; ++i)         \
                            b_frag[i] = (_Float16)xr[i];                       \
                    } else {                                                   \
                        _Pragma("unroll") for (int i = 0; i < 16; ++i)         \
                            b_frag[i] = (_Float16)0.f;                         \
                    }                                                          \
                    acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                        a_frag, b_frag, acc[t]);                               \
                }                                                              \
                __syncthreads();                                               \
            }                                                                  \
                                                                               \
            /* gfx11 wave32 16x16 f32 accumulator layout (see the writeback  \
               in gemm_bq1g128_wmma.hip, which is the authoritative local      \
               example): the A-side index is `2*i + (lane>>4)` and the B-side  \
               index is `lane & 15`. Here A is the OUTPUT COLUMN and B is the  \
               SLOT — getting these the wrong way round still runs and still   \
               produces finite numbers, it just produces the wrong ones        \
               (measured max |diff| 2.2e2 against the scalar path). */         \
            const int slot_in_tile = lane & 15;                                \
            _Pragma("unroll") for (int t = 0; t < NT; ++t) {                   \
                const int p = base + t * 16 + slot_in_tile;                    \
                if (p < g_end) {                                               \
                    const int s = sorted_slot_index[p];                        \
                    if (s >= 0) {                                              \
                        _Pragma("unroll") for (int i = 0; i < 8; ++i) {        \
                            const int oc = nt * 16 + 2 * i + (lane >> 4);      \
                            y_batch[(size_t)s * M + oc] = acc[t][i];           \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
    }

ESCHA_WMMA_GEMM(escha_gemm_grouped_wmma_k2, 2, 1)
ESCHA_WMMA_GEMM(escha_gemm_grouped_wmma_k3, 3, 1)
