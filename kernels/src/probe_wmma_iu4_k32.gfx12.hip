#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx12 (RDNA4) iu4 K=32 WMMA probe — POC for issue #136 part B.
//
// Validates that __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12 dispatches
// and accumulates correctly on gfx1201 silicon. This is the K=32 form that
// matches HFQ4's native 4-bit weight packing — 1 int32 = 8 INT4 values.
//
// Probe-1 (this kernel): both A and B are all-ones in INT4. Expected output
// at every (m, n): C[m][n] = sum_{k=0..31} 1*1 = 32. Result is independent
// of the per-lane element layout, so we get a "does the intrinsic actually
// work on this hardware" yes/no without first discovering the layout. Layout
// discovery is a follow-up probe.
//
// Signature confirmed via trial-compile on ROCm 7.2.2 / clang 22:
//   v8i wmma_i32_16x16x32_iu4_w32_gfx12(bool, v2i, bool, v2i, v8i, bool)
// where v2i = 2 int32 per lane (= 16 INT4 = K=32 worth of A or B for one row),
//       v8i = 8 int32 per lane accumulator (32 lanes x 8 = 256 = 16x16 of C).

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void probe_wmma_iu4_k32_allones(int32_t*) {}

#else

extern "C" __global__ void probe_wmma_iu4_k32_allones(int32_t* __restrict__ acc_dump) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    // All-ones INT4 packed as int32. 0x11111111 = eight INT4 values, each = 1.
    // Each lane's two ints carry 16 INT4 ones for its slice of A (or B); the
    // intrinsic permutes the data into the matrix unit's expected layout
    // internally.
    int32x2_t a_v;
    a_v[0] = 0x11111111;
    a_v[1] = 0x11111111;
    int32x2_t b_v;
    b_v[0] = 0x11111111;
    b_v[1] = 0x11111111;

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    // Args: a_signed=false (treat as unsigned), a, b_signed=false, b, acc, clamp=true
    acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a_v, false, b_v, acc, true);

    // Dump 32 lanes x 8 acc values to global memory. Host inspects:
    // expected: every entry = 32 (sum of 32 ones-times-ones).
    const int lane = threadIdx.x;
    int* dst = acc_dump + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = acc[i];
}

#endif
