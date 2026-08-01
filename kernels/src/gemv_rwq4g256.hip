// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// RadioWave Quant RWQ4-G256 GEMV.
//
// x_rot is PRE-ROTATED by the caller (one FWHT-256 pass on x, not per-row), so
// the inner loop carries zero rotation overhead — identical contract to
// gemv_hfq4g256 / MQ4G256.
//
// Wire layout, 136 B per 256-weight group (byte-identical budget to MQ4G256):
//   [0..4)   f32 master scale
//   [4..8)   4 x u8 E4M3 sub-scale, one per 64 weights
//   [8..136) 128 B nibbles, 2 weights/byte, low nibble first
//   w[i] = RWQ4_CODEBOOK[nibble[i]] * master * e4m3(sub[i / 64])
//
// Structure deliberately mirrors gemv_hfq4g256.hip, which is what MQ4G256
// actually dispatches to. Three properties there are load-bearing for decode
// speed and were measured, not assumed:
//   * K4 unroll — four groups per iteration.
//   * Four independent accumulators, so the FMA chain is not serialized.
//   * ONE packed 32-bit weight load per group per lane, never four byte loads.
//     (A byte-load variant of this kernel measured up to +232% on gate_up.)
//
// RWQ4-specific: the 16 codebook entries live in LDS and hold RAW levels, with
// the group's effective scale (master * sub-scale) factored OUT of the dot
// product and applied once per 8 weights. That factoring is the decode win over
// MQ4's per-weight `scale * nibble + zero`.
//
// gfx10-compatible by construction: wave32, LDS, DPP row_xmask (GFX10+) and
// one __shfl_xor. No WMMA,
// no VOPD, no dp4a.

// Lloyd-Max optimum for a unit Gaussian, normalized so max|c| == 1. A universal
// constant, not calibration data — bit-identical to
// crates/hipfire-quantize/src/mqn.rs::CODEBOOK_4BIT.
// Baked as immediates on purpose: a probe measured __constant__ placement at
// +23% mean / +96% worst-case for this access pattern.
static const float RWQ4_CODEBOOK[16] = {
    -1.00000000f,
    -0.75716355f,
    -0.59212931f,
    -0.45972187f,
    -0.34485254f,
    -0.24034313f,
    -0.14200753f,
    -0.04698658f,
     0.04698658f,
     0.14200753f,
     0.24034313f,
     0.34485254f,
     0.45972187f,
     0.59212931f,
     0.75716355f,
     1.00000000f,
};

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_rwq4g256_prerotated(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,   // PRE-ROTATED x (FWHT already applied)
    float* __restrict__ y,
    int M, int K
) {
    __shared__ float lut[16];
    const int tid = threadIdx.x;
    if (tid < 16) lut[tid] = RWQ4_CODEBOOK[tid];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();   // blockDim == 32: one wave32, no s_barrier

    const int row = blockIdx.x;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
    // Each lane owns 8 consecutive weights, so 8 lanes share one 64-weight
    // sub-scale. Lane -> byte within the [4..8) sub-scale word.
    const int sub_shift = (tid >> 3) * 8;

    // Effective per-group scale: master * e4m3(sub-scale byte). The sub-scale
    // word is 4-byte aligned, so it is one dword load, not a byte gather.
    #define RWQ4_EFF(gp)                                                       \
        (__builtin_bit_cast(float, *(const unsigned int*)(gp))                 \
         * cvt_e4m3_scale_to_f32_dq(                                           \
             (unsigned char)((*(const unsigned int*)((gp) + 4) >> sub_shift)   \
                             & 0xFFu)))

    // Raw codebook levels summed first, effective scale applied ONCE.
    #define RWQ4_DOT(pk, eff, b, a)                                            \
        (a) += (lut[(pk) & 0xFu]          * x_rot[(b)]                         \
              + lut[((pk) >> 4)  & 0xFu]  * x_rot[(b) + 1]                     \
              + lut[((pk) >> 8)  & 0xFu]  * x_rot[(b) + 2]                     \
              + lut[((pk) >> 12) & 0xFu]  * x_rot[(b) + 3]                     \
              + lut[((pk) >> 16) & 0xFu]  * x_rot[(b) + 4]                     \
              + lut[((pk) >> 20) & 0xFu]  * x_rot[(b) + 5]                     \
              + lut[((pk) >> 24) & 0xFu]  * x_rot[(b) + 6]                     \
              + lut[((pk) >> 28) & 0xFu]  * x_rot[(b) + 7]) * (eff)

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float eff0 = RWQ4_EFF(gp0);
        float eff1 = RWQ4_EFF(gp1);
        float eff2 = RWQ4_EFF(gp2);
        float eff3 = RWQ4_EFF(gp3);

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        const int base = g * 256 + tid * 8;
        RWQ4_DOT(pk0, eff0, base, acc0);
        RWQ4_DOT(pk1, eff1, base + 256, acc1);
        RWQ4_DOT(pk2, eff2, base + 512, acc2);
        RWQ4_DOT(pk3, eff3, base + 768, acc3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        RWQ4_DOT(pk, RWQ4_EFF(gp), g * 256 + tid * 8, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        RWQ4_DOT(pk, RWQ4_EFF(gp), g * 256 + tid * 8, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        RWQ4_DOT(pk, RWQ4_EFF(gp), g * 256 + tid * 8, acc2);
    }

    #undef RWQ4_DOT
    #undef RWQ4_EFF

    // wave32_reduce_add (turbo_common.h) is bit-exact with the conventional
    // five-step __shfl_down loop but spends four of the five steps on the
    // VALU via DPP row_xmask instead of five serialized ds_bpermute.
    float acc = wave32_reduce_add((acc0 + acc1) + (acc2 + acc3));
    if (tid == 0) y[row] = acc;
}
