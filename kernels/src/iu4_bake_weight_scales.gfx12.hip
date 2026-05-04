#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// SmoothQuant weight-side bake for the gfx12 iu4 path:
//
//   per (row m, group g) of HFQ4-G256:
//     scale_baked = scale * s_group[g]
//     zero_baked  = zero  * s_group[g]
//
// HFQ4-G256 dequant: w = scale * q_w + zero. Multiplying the whole row by
// s_group means BOTH scale AND zero get multiplied:
//   w * s_group = (scale * s_group) * q_w + (zero * s_group)
//
// HFQ4-G256 layout per group (136 bytes):
//   [0..4)  fp32 scale (modified by this kernel)
//   [4..8)  fp32 zero  (modified by this kernel)
//   [8..136) 128 bytes of nibbles (unchanged)
//
// The kernel reads `s_group` as FP16 (per-K=256-group activation scale,
// computed at calibration time as the geometric or arithmetic mean of
// per-channel s_a values within each group). Per group → one FP32
// multiply on the scale field.
//
// Threading: one thread per (row, group) pair. Grid x = ceil(m/256), y =
// groups_per_row. Block = (256, 1, 1).

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#if !HIPFIRE_GFX12

extern "C" __global__ void iu4_bake_weight_scales_gfx12(
    char*,
    const __half*,
    int,
    int
) {}

#else

extern "C" __global__ void iu4_bake_weight_scales_gfx12(
    char* __restrict__ W,            // [m * groups_per_row * 136] HFQ4-G256
    const __half* __restrict__ s_group, // [groups_per_row]
    int m,
    int groups_per_row
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (row >= m || g >= groups_per_row) return;

    char* gp = W + ((long long)row * groups_per_row + g) * 136;

    // Read FP32 scale at byte offset 0 and FP32 zero at byte offset 4.
    float scale, zero;
    {
        unsigned int sc_bits = *(const unsigned int*)gp;
        unsigned int zp_bits = *(const unsigned int*)(gp + 4);
        scale = __builtin_bit_cast(float, sc_bits);
        zero  = __builtin_bit_cast(float, zp_bits);
    }

    // Multiply by FP16 group scale (cast to FP32 for the multiply).
    const float s = __half2float(s_group[g]);
    const float scale_baked = scale * s;
    const float zero_baked  = zero  * s;

    // Write back both scale (offset 0) and zero (offset 4).
    *(unsigned int*)gp       = __builtin_bit_cast(unsigned int, scale_baked);
    *(unsigned int*)(gp + 4) = __builtin_bit_cast(unsigned int, zero_baked);
}

#endif
