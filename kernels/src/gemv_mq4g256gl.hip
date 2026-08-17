// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// MagnumQuant 4-bit dense GEMV — MQ4-G256-GL (qt=40), global Lloyd codebook.
//
// Blob layout (SoA, TWO REGIONS, 130 B/group = 4.0625 bpw):
//   [0 .. M*gpr*128)            4-bit indices, 128 B/group, row-major (row, group)
//   [M*gpr*128 .. +M*gpr*2)     fp16 per-block scales, 2 B/group, row-major (row, group)
// gpr = K/256. Nibble packing is byte-identical to MQ4G256 / MQ4-Lloyd:
//   byte b holds weight 2b in the LOW nibble and weight 2b+1 in the HIGH nibble.
//
// Reconstruction (rotated domain): w = f32(scale) * GL_CB4[code].
// The 16-entry codebook arrives as scalar float kernel args (cb0..cb15) and
// MUST match rdna_compute::GL_CB4 bit-for-bit. A drifted codebook is a silent
// accuracy failure (plausible-but-wrong outputs), not a crash — never paste
// a literal copy of the 16 floats into HIP source.
//
// Donor: gemv_mq4g256_lloyd.hip. Five deltas:
//   1. row base is the INDEX region only (stride 128, not 160);
//   2. data pointer is gptr directly — no 32-byte inline codebook header;
//   3. scales live in a separate SoA region after all indices;
//   4. codebook is loop-invariant scalar args (no per-group __half loads);
//   5. scale folds once per group onto the 8-term sub-dot, not per weight.
//
// Per-thread work: 8 weights / thread × 32 threads = 256 weights / block.
// X must be FWHT-pre-rotated by the caller (same convention as every MQ* kernel).
// Grid: [M, 1, 1], block [32, 1, 1]. No LDS — cb lives in registers via CB_LOOKUP4.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256gl(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    float cb0,  float cb1,  float cb2,  float cb3,
    float cb4,  float cb5,  float cb6,  float cb7,
    float cb8,  float cb9,  float cb10, float cb11,
    float cb12, float cb13, float cb14, float cb15,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int gpr = K / 256;
    const char* row_ptr = A + (long long)row * gpr * 128;
    const __half* scales = (const __half*)(A + (long long)M * gpr * 128);
    const int boff = tid * 4;

    // Hoist the ENTIRE row's scale set before the K loop.
    //
    // The scales for row `row`, groups 0..gpr-1, are contiguous fp16 (gpr*2 B;
    // 40 B at K=5120), so one coalesced load by the first `gpr` lanes replaces
    // `gpr` narrow dependent loads. Read naively inside the loop, the compiler
    // emits one `global_load_d16_b16` per group -- the narrowest VMEM op there
    // is -- against a stream M*gpr*128 B away from the index stream, so it
    // cannot be claused with the index load and each iteration stalls on it.
    //
    // This is the one place MQ4-GL beats MQ4-Lloyd structurally rather than
    // just on size: our codebook is tensor-global and our scales are per-row
    // contiguous, so both amortize over the whole row. Lloyd's 32 B codebook is
    // per-GROUP data and must be re-fetched every iteration forever.
    // LDS layout: [16 f32 codebook][gpr f32 hoisted row scales].
    // Codebook is 64 B, staged once per workgroup and live for the whole K loop.
    extern __shared__ float lds[];
    float* cbt = lds;
    float* sc = lds + 16;
    if (tid < 16) {
        const float cbv[16] = {cb0, cb1, cb2,  cb3,  cb4,  cb5,  cb6,  cb7,
                               cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15};
        cbt[tid] = cbv[tid];
    }
    for (int i = tid; i < gpr; i += 32)
        sc[i] = __half2float(scales[(long long)row * gpr + i]);
    __syncthreads();

    float acc = 0.0f;

    for (int g = 0; g < gpr; g++) {
        const unsigned char* data = (const unsigned char*)(row_ptr + g * 128);
        const float scale = sc[g];

        // 8 × 4-bit nibble unpack (identical to uniform MQ4 / MQ4-Lloyd):
        //   data[boff + i] = idx[2i] (low) | idx[2i+1] (high)
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

        int base_idx = g * 256 + tid * 8;

        // BRANCHLESS 16-entry lookup via LDS.
        //
        // The MQ4-Lloyd donor uses a 15-deep `?:` chain and claims the compiler
        // folds it to conditional moves or a jump table. On gfx1201 that is
        // FALSE: hipcc lowers it to divergent control flow -- measured 105
        // `v_cmpx_lt_i32` + 115 `s_cbranch_execz` + 105 exec-mask save/restore
        // in the emitted object, against only 8 `v_fmac_f32`. With 32 lanes each
        // holding an independent 4-bit code the branch diverges up to 16 ways
        // and serializes, which is what actually bounds this kernel -- not
        // bandwidth (it reaches ~12% of peak) and not the scale stream.
        //
        // A `ds_read_b32` at a computed address is branchless. 32 lanes over 16
        // distinct addresses is a multicast pattern LDS handles well, and it is
        // unconditionally better than 16-way divergence.
        //
        // The donor kernel has the SAME defect (114 `s_cbranch_execz`); it is a
        // production kernel and worth fixing separately.
        float gacc = cbt[q0] * x[base_idx]
                   + cbt[q1] * x[base_idx + 1]
                   + cbt[q2] * x[base_idx + 2]
                   + cbt[q3] * x[base_idx + 3]
                   + cbt[q4] * x[base_idx + 4]
                   + cbt[q5] * x[base_idx + 5]
                   + cbt[q6] * x[base_idx + 6]
                   + cbt[q7] * x[base_idx + 7];
        acc += scale * gacc;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
