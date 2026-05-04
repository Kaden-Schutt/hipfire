#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx12 (RDNA4) FP8 (E4M3) WMMA — layout-discovery probe (issue #136 part B).
//
// Probes `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` on gfx1201 to
// confirm the wave32 layout before writing an FP8 GEMM port. The iu4 K=32
// and iu8 K=16 probes both discovered an 8-row-block acc layout — first
// guess is FP8 wmma uses the same layout (its operand vector size is also
// int32x2 per lane = 8 FP8 bytes per K-half).
//
// Per CK header `amd_wmma.hpp` confirmed signature:
//   wmma_f32_16x16x16_fp8_fp8_w32_gfx12(int32x2 a, int32x2 b, float8 acc) -> float8
// — no clamp/sign flags (unlike iu4/iu8). Acc is 8 FP32 per lane.
//
// Pattern set (one wave per launch, dumped to consecutive 256-FP32 regions):
//   0 — A=B=1.0                 → every entry = 16.0  (sanity)
//   1 — A[m][k]=(m+1).0, B=1.0  → acc[l, j] = 16 * (8*(l>>4) + j + 1)
//   2 — A=1.0, B[k][n]=(n+1).0  → acc[l, j] = 16 * ((l & 0xF) + 1)
//
// Why integer values 1..16: easy E4M3 representations (no rounding loss),
// distinct enough to bijectively map to (lane, slot) → (m, n).

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void probe_wmma_fp8_layout(float*) {}

#else

extern "C" __global__ void probe_wmma_fp8_layout(float* __restrict__ out) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using float32x8_t = __attribute__((__vector_size__(8 * sizeof(float)))) float;

    const int lane = threadIdx.x;
    const int row_assumed = lane & 0xF;

    // Pack a single FP32 value into all 8 FP8 slots of an int32x2 via 4 calls
    // to __builtin_amdgcn_cvt_pk_fp8_f32. Each call converts 2 FP32 to 2
    // packed FP8 bytes (E4M3) into a 16-bit half of the destination int32.
    // word_sel=false → low 16 bits, word_sel=true → high 16 bits.
    auto pack_fp8_v8 = [](float v) -> int32x2_t {
        int32x2_t r;
        unsigned int w0 = 0;
        w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, w0, false);
        w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, w0, true);
        unsigned int w1 = 0;
        w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, w1, false);
        w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, w1, true);
        r[0] = (int)w0;
        r[1] = (int)w1;
        return r;
    };

    // ---- Pattern 0: A=B=1.0
    {
        int32x2_t a = pack_fp8_v8(1.0f);
        int32x2_t b = pack_fp8_v8(1.0f);
        float32x8_t acc = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
        float* dst = out + 0 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }

    // ---- Pattern 1: A row-id+1, B all-ones.
    {
        int32x2_t a = pack_fp8_v8((float)(row_assumed + 1));
        int32x2_t b = pack_fp8_v8(1.0f);
        float32x8_t acc = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
        float* dst = out + 1 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }

    // ---- Pattern 2: A all-ones, B col-id+1.
    {
        int32x2_t a = pack_fp8_v8(1.0f);
        int32x2_t b = pack_fp8_v8((float)(row_assumed + 1));
        float32x8_t acc = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
        float* dst = out + 2 * (32 * 8) + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = acc[i];
    }
}

#endif
