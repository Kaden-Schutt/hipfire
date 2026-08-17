// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Kill experiment: MQ4-G256-GL (qt=40) GEMV via int8 decode + sdot4.
// Same SoA blob as gemv_mq4g256gl; x is host-quantized int8 + one absmax scale.
//
// Numeric contract (from GL_CB4, never a hardcoded level table):
//   GL_I8_SCALE = 127 / 2.7326 = 46.475883
//   levels_i8[i] = round(GL_CB4[i] * GL_I8_SCALE)
//   acc_i32_g = sum_i levels_i8[code_i] * x_q8_i
//   y += (f32(scale_g) / GL_I8_SCALE) * x_scale * f32(acc_i32_g)
//
// MEASURED VERDICT FOR DECODE: **NO-GO.** Retained as a validated numeric
// starting point, not as a production path. Do not wire this into dispatch.
//
// gfx1201, m=4096 k=5120, device-side HIP events, 3 runs:
//   this kernel        40.401 / 40.560 / 40.641 us   ~263 GB/s
//   gemv_mq4g256gl     30.960 / 31.000 / 31.040 us   ~345 GB/s
// i.e. 31% SLOWER, while the intended mechanism worked exactly as predicted:
//   ds_load_b32   9 -> 1     (per-weight LDS codebook read eliminated)
//   spills        0 / 0
//
// Why it lost, which is the durable lesson: the 16-entry table lookup cannot be
// ELIMINATED, only RELOCATED. Moving it from LDS to ALU selects cost more than
// the LDS reads it saved --
//   v_cndmask_b32_e64        17 -> 23
//   max consecutive VMEM      3 -> 1     (clausing lost)
//   memory_clause_instructions 2 -> 1
//   vgpr_count               22 -> 37
//   static_instructions     250 -> 363
// This is the same irreducible cost that the branchless-LDS-codebook fix
// (e7b0fa56c) ran into from the other direction, and it is consistent with
// docs/perf-checkpoints/2026-07-13-gfx1201-qkvza-gateup-dot-screen.md finding
// v_dot4_i32_iu8 roughly neutral end-to-end on this architecture.
//
// The gate measured the CONSUMER ONLY -- x was host-quantized -- so a device
// activation-quant pass would add cost on top of an already 31%-slower consumer.
// The framing does not soften the verdict.
//
// The NUMERICS are fine and are the reason this file is kept: int8 levels
// {-127,-96,-75,-58,-44,-31,-18,-6, 6,18,31,44,58,75,96,127} reproduce exact-GL
// output to 0.21-0.27% peak-normalized deviation, and the kernel matches an
// int8-model f64 reference to 1.49e-7. Codec MSE penalty is +0.142% versus
// exact GL_CB4, against E4M3's +3.352% (locked in by e7978b3e2). So if a
// BATCHED path is ever built -- where wmma_i32_16x16x16_iu8 is fully utilized,
// unlike a batch-1 dot4 -- this is the validated numeric foundation for it.
// The decode NO-GO says nothing about that case.
//
// RDNA4 API note: __builtin_amdgcn_sdot4 does NOT compile on gfx1201, which
// lacks dot1-insts. The replacement used here is
//   __builtin_amdgcn_sudot4(true, w, true, x, acc, false)
// with both operands flagged signed, emitting v_dot4_i32_iu8.
//
// Single row / wave32. Stages int8 levels in VGPR-packed form (no per-weight
// LDS codebook load). Hoists the row's fp16 scales into LDS like the scalar path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// 127 / 2.7326 — matches the locked GL int8 contract.
static constexpr float GL_I8_SCALE = 46.475883f;

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

// Branchless extract of signed byte `q` from four packed int8 quads (16 levels).
__device__ __forceinline__ int lvl_i8(unsigned int q, int pk0, int pk1, int pk2, int pk3) {
    int pk = (q < 4) ? pk0 : (q < 8) ? pk1 : (q < 12) ? pk2 : pk3;
    unsigned int sh = (q & 3u) * 8u;
    return (int)(signed char)((pk >> sh) & 0xff);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256gl_int8(
    const char* __restrict__ A,
    const signed char* __restrict__ x_q8,
    float* __restrict__ y,
    float cb0,  float cb1,  float cb2,  float cb3,
    float cb4,  float cb5,  float cb6,  float cb7,
    float cb8,  float cb9,  float cb10, float cb11,
    float cb12, float cb13, float cb14, float cb15,
    int M, int K,
    float x_scale)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int gpr = K / 256;
    const char* row_ptr = A + (long long)row * gpr * 128;
    const __half* scales = (const __half*)(A + (long long)M * gpr * 128);
    const int boff = tid * 4;
    const int w_off = tid * 8;

    // Derive int8 levels from cb args (must match rdna_compute::GL_CB4 on host).
    // Packed into 4 VGPR ints so the nibble→level map is shift/cmov, not ds_load.
    const int pk0 = pack_i8x4(
        (int)rintf(cb0  * GL_I8_SCALE), (int)rintf(cb1  * GL_I8_SCALE),
        (int)rintf(cb2  * GL_I8_SCALE), (int)rintf(cb3  * GL_I8_SCALE));
    const int pk1 = pack_i8x4(
        (int)rintf(cb4  * GL_I8_SCALE), (int)rintf(cb5  * GL_I8_SCALE),
        (int)rintf(cb6  * GL_I8_SCALE), (int)rintf(cb7  * GL_I8_SCALE));
    const int pk2 = pack_i8x4(
        (int)rintf(cb8  * GL_I8_SCALE), (int)rintf(cb9  * GL_I8_SCALE),
        (int)rintf(cb10 * GL_I8_SCALE), (int)rintf(cb11 * GL_I8_SCALE));
    const int pk3 = pack_i8x4(
        (int)rintf(cb12 * GL_I8_SCALE), (int)rintf(cb13 * GL_I8_SCALE),
        (int)rintf(cb14 * GL_I8_SCALE), (int)rintf(cb15 * GL_I8_SCALE));

    // Hoist row scales into LDS (same trick as gemv_mq4g256gl).
    extern __shared__ float sc[];
    for (int i = tid; i < gpr; i += 32)
        sc[i] = __half2float(scales[(long long)row * gpr + i]);
    __syncthreads();

    float acc = 0.0f;
    const float inv_i8 = 1.0f / GL_I8_SCALE;
    const float xs = x_scale;

    for (int g = 0; g < gpr; g++) {
        const unsigned char* data = (const unsigned char*)(row_ptr + g * 128);

        unsigned int b0 = (unsigned int)data[boff];
        unsigned int b1 = (unsigned int)data[boff + 1];
        unsigned int b2 = (unsigned int)data[boff + 2];
        unsigned int b3 = (unsigned int)data[boff + 3];

        unsigned int q0 =  b0       & 0xFu;
        unsigned int q1 = (b0 >> 4) & 0xFu;
        unsigned int q2 =  b1       & 0xFu;
        unsigned int q3 = (b1 >> 4) & 0xFu;
        unsigned int q4 =  b2       & 0xFu;
        unsigned int q5 = (b2 >> 4) & 0xFu;
        unsigned int q6 =  b3       & 0xFu;
        unsigned int q7 = (b3 >> 4) & 0xFu;

        int wp0 = pack_i8x4(
            lvl_i8(q0, pk0, pk1, pk2, pk3),
            lvl_i8(q1, pk0, pk1, pk2, pk3),
            lvl_i8(q2, pk0, pk1, pk2, pk3),
            lvl_i8(q3, pk0, pk1, pk2, pk3));
        int wp1 = pack_i8x4(
            lvl_i8(q4, pk0, pk1, pk2, pk3),
            lvl_i8(q5, pk0, pk1, pk2, pk3),
            lvl_i8(q6, pk0, pk1, pk2, pk3),
            lvl_i8(q7, pk0, pk1, pk2, pk3));

        const signed char* xb = x_q8 + g * 256;
        int xp0 = *(const int*)(xb + w_off);
        int xp1 = *(const int*)(xb + w_off + 4);
        // gfx1201 has no dot1-insts / sdot4. Use sudot4 with both operands
        // signed (a_sign=true, b_sign=true) → v_dot4_i32_iu8 with signed×signed,
        // which is the RDNA3+/RDNA4 replacement for sdot4's v_dot4_i32_i8.
        // Clamp=false matches the mq8 sdot4 convention.
        int dot = __builtin_amdgcn_sudot4(true, wp0, true, xp0, 0, false);
        dot = __builtin_amdgcn_sudot4(true, wp1, true, xp1, dot, false);

        // One i32→f32 convert per group-slice; scales applied outside the dot.
        acc += sc[g] * inv_i8 * xs * (float)dot;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
