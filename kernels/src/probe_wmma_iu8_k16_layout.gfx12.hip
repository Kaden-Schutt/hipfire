#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx12 (RDNA4) iu8 K=16 WMMA — layout-discovery probe (issue #136 part B).
//
// Probes the per-lane element layout for
// __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 on gfx1201. The iu4 K=32 layout
// probe (probe_wmma_iu4_k32_layout.gfx12.hip) discovered an 8-row-block acc
// layout — this probe checks whether iu8 K=16 follows the same wave32 layout
// or differs.
//
// Per-lane shape (correct, confirmed via CK header amd_wmma.hpp): the
// gfx12-suffixed builtin __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
// takes int8x8 = int32x2 per lane (NOT int32x4 — that was the RDNA3-shape
// builtin without _gfx12 suffix, which has no codegen on gfx12). 32 lanes ×
// 8 INT8 = 256 INT8 = 16 rows × 16 K, so the natural split is:
//   A input: lane l → row m = l & 0xF, k-half = l >> 4, 8 INT8 = half-row
//   B input: lane l → col n = l & 0xF, k-half = l >> 4
// First-guess acc layout: same 8-row-block as iu4 K=32:
//   acc[lane=l, slot=j] → C[8*(l>>4) + j][l & 0xF]
//
// Three patterns, each writing 256 i32 to its own region of `out`:
//   pattern 0: A=1, B=1                → expected every entry = 16 (sanity)
//   pattern 1: A[m][k]=m+1, B=1        → expected acc[l, j] = 16 * (8*(l>>4) + j + 1)
//   pattern 2: A=1, B[k][n]=n+1        → expected acc[l, j] = 16 * ((l & 0xF) + 1)

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void probe_wmma_iu8_k16_layout(int32_t*) {}

#else

extern "C" __global__ void probe_wmma_iu8_k16_layout(int32_t* __restrict__ out) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    const int lane = threadIdx.x;
    const int row_assumed = lane & 0xF;

    // Pack a single byte v ∈ [0,255] into all 4 lanes of an int32.
    auto pack_byte = [](int v) -> int {
        return (v & 0xFF) * 0x01010101;
    };

    // ---- Pattern 0: A=1, B=1 (sanity).
    {
        int32x2_t a_v;
        a_v[0] = pack_byte(1);
        a_v[1] = pack_byte(1);
        int32x2_t b_v;
        b_v[0] = pack_byte(1);
        b_v[1] = pack_byte(1);
        int32x8_t acc = {0,0,0,0,0,0,0,0};
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_v, false, b_v, acc, true);
        int* dst = out + 0 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }

    // ---- Pattern 1: A[m][k] = m+1, B=1.
    // Under assumed input layout: lane l holds A row m = l & 0xF (8 INT8 of K-half).
    // C[m][n] = sum_{k=0..15} (m+1) * 1 = 16 * (m+1), independent of n.
    // First-guess acc layout (8-row-block): acc[l, j] = 16 * (8*(l>>4) + j + 1).
    {
        int32x2_t a_v;
        a_v[0] = pack_byte(row_assumed + 1);
        a_v[1] = pack_byte(row_assumed + 1);
        int32x2_t b_v;
        b_v[0] = pack_byte(1);
        b_v[1] = pack_byte(1);
        int32x8_t acc = {0,0,0,0,0,0,0,0};
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_v, false, b_v, acc, true);
        int* dst = out + 1 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }

    // ---- Pattern 2: A=1, B[k][n] = n+1.
    // Under assumed input layout: lane l holds B col n = l & 0xF (8 INT8 of K-half).
    // C[m][n] = sum_{k=0..15} 1 * (n+1) = 16 * (n+1), independent of m.
    // Expected: every slot of a lane reads the same value = 16 * ((l & 0xF) + 1).
    {
        int32x2_t a_v;
        a_v[0] = pack_byte(1);
        a_v[1] = pack_byte(1);
        int32x2_t b_v;
        b_v[0] = pack_byte(row_assumed + 1);
        b_v[1] = pack_byte(row_assumed + 1);
        int32x8_t acc = {0,0,0,0,0,0,0,0};
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_v, false, b_v, acc, true);
        int* dst = out + 2 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }
}

#endif
