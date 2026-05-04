#include <hip/hip_runtime.h>
#include <stdint.h>

// Dequant HFQ4-G256 packed weights to FP8 (E4M3), one-shot at model load
// time on gfx12 (RDNA4). Sister of `dequant_hfq4g256_to_f16` — same access
// pattern, different output dtype. Used to populate an "FP8 shadow" weight
// tensor that the FP8 wmma GEMM path consumes directly (eliminating the
// per-prefill in-kernel dequant cost).
//
// Storage delta: HFQ4-G256 is 0.531 B/w (136 B per 256 weights). FP8 is
// 1.0 B/w. Pre-converted FP8 shadow doubles VRAM cost; for 27B that's
// 27 GB total (vs HFQ4's 13.5 GB), which fits R9700's 32 GB but leaves
// thin headroom for KV cache. Caller decides when this trade is worth it.
//
// Input:  A_packed [M × groups_per_row × 136 bytes] (4-bit packed)
//   group layout: 4B FP32 scale | 4B FP32 zero | 128B nibbles (256 of them)
// Output: W_fp8   [M × K] FP8 (E4M3) bytes (row-major)
//
// Grid: [M, groups_per_row]. Block: [32].
// Each block dequants one 256-element group for one row. Thread t covers
// 8 nibbles (tid*8..tid*8+7) → 8 FP32 → 8 FP8 bytes packed as 2 int32 via
// `__builtin_amdgcn_cvt_pk_fp8_f32`.
//
// FP8 cast precision: HFQ4 weights have effective range scale*[0,15]+zero
// per group. After applying the per-group scale+zero, weight values land
// in whatever distribution the source model has. E4M3's representable
// range (±448) easily covers typical Qwen3.5 weight magnitudes (rarely
// exceed ~10 after scale/zero apply). Per-group scale+zero is BAKED INTO
// the FP8 representation here — the FP8 wmma GEMM consumer sees no extra
// scale/zero metadata.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void dequant_hfq4g256_to_fp8_gfx12(
    const char*, unsigned char*, int, int) {}

#else

extern "C" __launch_bounds__(32, 16)
__global__ void dequant_hfq4g256_to_fp8_gfx12(
    const char* __restrict__ A_packed,
    unsigned char* __restrict__ W_fp8,
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;

    const char* gp = A_packed + (long long)row * groups_per_row * 136 + group * 136;
    const float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));
    const float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

    const int boff = tid * 4;  // 8 nibbles = 4 bytes
    const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);

    // Dequant 8 nibbles → 8 FP32 in registers.
    const float w0 = scale * (float)((pk      ) & 0xFu) + zero;
    const float w1 = scale * (float)((pk >>  4) & 0xFu) + zero;
    const float w2 = scale * (float)((pk >>  8) & 0xFu) + zero;
    const float w3 = scale * (float)((pk >> 12) & 0xFu) + zero;
    const float w4 = scale * (float)((pk >> 16) & 0xFu) + zero;
    const float w5 = scale * (float)((pk >> 20) & 0xFu) + zero;
    const float w6 = scale * (float)((pk >> 24) & 0xFu) + zero;
    const float w7 = scale * (float)((pk >> 28) & 0xFu) + zero;

    // Pack 8 FP32 → 8 FP8 (E4M3) bytes as 2 int32. word_sel=false writes
    // the low 16 bits of `old`, word_sel=true writes the high 16 bits.
    // Each call converts 2 FP32 → 2 FP8 packed into a 16-bit half.
    unsigned int p0 = 0;
    p0 = __builtin_amdgcn_cvt_pk_fp8_f32(w0, w1, p0, false);
    p0 = __builtin_amdgcn_cvt_pk_fp8_f32(w2, w3, p0, true);
    unsigned int p1 = 0;
    p1 = __builtin_amdgcn_cvt_pk_fp8_f32(w4, w5, p1, false);
    p1 = __builtin_amdgcn_cvt_pk_fp8_f32(w6, w7, p1, true);

    // Output offset: row stride = K bytes (1 byte per FP8 element).
    // tid*8 byte offset within the group → tid*2 int32 offset.
    unsigned int* out_i32 = (unsigned int*)(
        W_fp8 + (long long)row * K + group * 256 + tid * 8);
    out_i32[0] = p0;
    out_i32[1] = p1;
}

#endif
