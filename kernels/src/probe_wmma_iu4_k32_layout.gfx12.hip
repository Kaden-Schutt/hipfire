#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx12 (RDNA4) iu4 K=32 WMMA — layout-discovery probe (issue #136 part B).
//
// The all-ones probe (probe_wmma_iu4_k32.gfx12.hip) only confirms the
// intrinsic dispatches and accumulates. This probe maps the per-lane
// element layout so we can write the production GEMM port.
//
// Three patterns, each running one wmma and dumping 32*8=256 i32:
//   pattern 0: A=ones, B=ones      → expected every entry = 32 (sanity)
//   pattern 1: A=row-id, B=ones    → confirms acc layout: lane l, slot j → C[8*(l>>4) + j][l & 0xF]
//                                    Expected acc[lane=l, slot=j] = 32 * (8*(l>>4) + j).
//                                    8-row-block layout — DIFFERENT from RDNA3 stride-2.
//                                    Lanes 0..15 own rows 0..7, lanes 16..31 own rows 8..15.
//   pattern 2: A=ones, B=col-id    → confirms col mapping: lane l → n = l & 0xF.
//                                    Expected acc[lane=l, slot=j] = 32 * (l & 0xF)
//                                    (same value across all 8 slots of any one lane).
//
// `out` is laid out [pattern0_dump | pattern1_dump | pattern2_dump], each
// 256 i32s. Host inspects to either confirm the assumed layout or extract
// the actual layout map.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void probe_wmma_iu4_k32_layout(int32_t*) {}

#else

extern "C" __global__ void probe_wmma_iu4_k32_layout(int32_t* __restrict__ out) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    const int lane = threadIdx.x;
    const int row_assumed = lane & 0xF;
    const int khalf = lane >> 4;
    (void)khalf; // unused under the K-symmetric patterns we run here.

    // Helper: pack a single 4-bit value v ∈ [0,15] into all 8 nibbles of an int32.
    auto pack_nibble = [](int v) -> int {
        return (v & 0xF) * 0x11111111;
    };

    // ---- Pattern 0: A=ones, B=ones (sanity, mirrors the all-ones probe).
    {
        int32x2_t a_v;
        a_v[0] = 0x11111111;
        a_v[1] = 0x11111111;
        int32x2_t b_v;
        b_v[0] = 0x11111111;
        b_v[1] = 0x11111111;
        int32x8_t acc = {0,0,0,0,0,0,0,0};
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a_v, false, b_v, acc, true);
        int* dst = out + 0 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }

    // ---- Pattern 1: A = row-id (each row m has every k-element = m), B = ones.
    // Under the assumed layout, lane l holds A row m = l & 0xF in both v2i halves.
    // Expected C[m][n] = sum_{k=0..31} m * 1 = 32 * m, independent of n.
    // Expected acc[lane=l, slot=j] = 32 * (2*j + (l >> 4)).
    {
        int32x2_t a_v;
        a_v[0] = pack_nibble(row_assumed);
        a_v[1] = pack_nibble(row_assumed);
        int32x2_t b_v;
        b_v[0] = 0x11111111;
        b_v[1] = 0x11111111;
        int32x8_t acc = {0,0,0,0,0,0,0,0};
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a_v, false, b_v, acc, true);
        int* dst = out + 1 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }

    // ---- Pattern 2: A = ones, B = col-id.
    // Under the assumed layout, lane l holds B col n = l & 0xF in both v2i halves.
    // Expected C[m][n] = sum_{k=0..31} 1 * n = 32 * n, independent of m.
    // Expected acc[lane=l, slot=j] = 32 * (l & 0xF) — same value across all 8 slots.
    {
        int32x2_t a_v;
        a_v[0] = 0x11111111;
        a_v[1] = 0x11111111;
        int32x2_t b_v;
        b_v[0] = pack_nibble(row_assumed);
        b_v[1] = pack_nibble(row_assumed);
        int32x8_t acc = {0,0,0,0,0,0,0,0};
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a_v, false, b_v, acc, true);
        int* dst = out + 2 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }
}

#endif
