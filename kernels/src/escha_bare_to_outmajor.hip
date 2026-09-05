// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Escha-W2 load path, step 3 of 3 (Task 10). `escha_decode_tiles` writes the
// decoded weight BARE and IN-MAJOR — row-major `[in_features, out_features]`
// fp16 — because escha's code tile grid is in-major (`[in/16, out/16]`).
// hipfire's expert slots are OUT-major: `experts[X].gate_up` is
// `[2*moe_intermediate, hidden]` (see hipfire-arch-qwen35 weights.rs:69), and
// every hipfire GEMV walks K contiguously along a row. So the transpose
// happens HERE, folded into the store, and it is the single place in the
// port where a wrong orientation would still produce a full-rank,
// plausible-looking weight matrix. Do not "simplify" the indexing.
//
// Both entry points read `bare[i * oc + o]` (in-major) and write element
// `(o, i)` of an out-major `[oc, ic]` matrix.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Q8_0 target layout (hipfire's, see kernels/src/gemv_q8_0.hip): per output
// row `o`, `ic/32` consecutive blocks of 34 bytes — fp16 scale at +0, then 32
// signed int8. Row stride is therefore `(ic/32) * 34` bytes and there is no
// padding.
//
// One block per (o, blk) pair, 32 threads; thread `t` owns element
// `i = blk*32 + t`. The 32 reads are strided by `oc` fp16 in `bare` — that is
// inherent to transposing, and this runs ONCE PER EXPERT AT LOAD TIME, not
// per token, so it is not on any hot path.
//
// Scale rule matches llama.cpp/hipfire Q8_0: `d = amax/127` rounded to fp16,
// values `roundf(v/d)` clamped to [-127, 127]. The clamp is load-bearing:
// when `f16(amax/127) < amax/127` the extreme element rounds to +/-128, which
// would wrap in int8.
extern "C" __global__ void escha_bare_to_q8_0(
    const __half* __restrict__ bare,   // [ic, oc] in-major
    unsigned char* __restrict__ out,   // [oc][ic/32][34]
    int ic, int oc) {
    const int blocks_per_row = ic >> 5;
    const int o = blockIdx.x / blocks_per_row;
    const int blk = blockIdx.x - o * blocks_per_row;
    const int t = threadIdx.x;
    const int i = (blk << 5) + t;

    const float v = __half2float(bare[(size_t)i * oc + o]);

    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    }

    const __half dh = __float2half(amax / 127.0f);
    const float d = __half2float(dh);
    int q = 0;
    if (d != 0.0f) {
        float r = roundf(v / d);
        q = (int)fminf(fmaxf(r, -127.0f), 127.0f);
    }

    unsigned char* blkp = out + ((size_t)o * blocks_per_row + blk) * 34;
    if (t == 0) {
        unsigned short sb = __half_as_ushort(dh);
        blkp[0] = (unsigned char)(sb & 0xFF);
        blkp[1] = (unsigned char)(sb >> 8);
    }
    blkp[2 + t] = (unsigned char)(signed char)q;
}

// F32 out-major transpose — the bit-exact control arm. `escha_bare_to_q8_0`
// re-quantises the decoded weight to 8 bits, which costs real accuracy (see
// the G4 gate); this entry stores the decoded fp16 value widened to f32 with
// no loss at all, so a caller can measure the re-quantisation cost against a
// weight-exact baseline instead of guessing at it. 4 bytes/weight, so it is a
// diagnostic/small-layer path, not a way to run the 35B model.
extern "C" __global__ void escha_bare_to_f32(
    const __half* __restrict__ bare,   // [ic, oc] in-major
    float* __restrict__ out,           // [oc, ic] out-major
    int ic, int oc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y;
    if (i >= ic) return;
    out[(size_t)o * ic + i] = __half2float(bare[(size_t)i * oc + o]);
}

// F16 out-major transpose — the weight-exact arm that fits a whole model.
//
// This is PURE DATA MOVEMENT: `escha_decode_tiles` already produced fp16, so
// the stored value is bit-identical to `escha_ref::reconstruct`'s output. F32
// (above) is exactly as exact and twice the bytes, which is why it is only a
// small-layer diagnostic; this one is what a model-scale weight-exact
// reference arm has to use.
//
// The 35B's A3B shapes make the difference stark, and it is an allocator
// effect rather than an arithmetic one. Per expert the two projections are
// gate_up `[1024, 2048]` and down `[2048, 512]`:
//
//   store  gate_up      down        per-expert   x10240 experts
//   Q8_0   2.125 MiB    1.0625 MiB  3.1875 MiB   31.9 GiB  (34.2 GB)
//   F16    4 MiB        2 MiB       6 MiB        60 GiB    (64.4 GB)
//   F32    8 MiB        4 MiB       12 MiB       120 GiB   (129 GB)
//
// but each buffer is a separate allocation that the HIP allocator rounds up
// to a 2 MiB granule, so the Q8_0 arm's 2.125/1.0625 MiB buffers occupy 4 and
// 2 MiB anyway — 60 GiB resident, the SAME as F16. Measured on gfx1151:
// 67.9 GB of GTT for the Q8_0 model against 64.4 GB of expert granules plus
// ~3.4 GB of everything else. So F16 is weight-exact at no resident cost over
// production, whereas F32 would be 2x and would not fit.
extern "C" __global__ void escha_bare_to_f16(
    const __half* __restrict__ bare,   // [ic, oc] in-major
    __half* __restrict__ out,          // [oc, ic] out-major
    int ic, int oc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y;
    if (i >= ic) return;
    out[(size_t)o * ic + i] = bare[(size_t)i * oc + o];
}
