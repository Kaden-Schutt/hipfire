// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — batched mfp4-E8 SoA decode-GEMV for small-batch (speculative
// verify) forwards on gfx1151.
//
// One wave owns one output row, reads that weight row EXACTLY ONCE, and
// applies it to `E8_BATCHED_B` activation vectors. Layout, scale conversion
// and lattice decode are copied verbatim from `gemv_mfp4g32_e8_soa_u4`, so
// B=1 is bit-identical to the decode path.
//
// Why this exists. The batched/prefill path dispatches dense projections
// through `gemm_mfp4g32_e8_soa_wmma_gfx1151`, which tiles the TOKEN axis at 16
// ("each wave computes one 16-row x 16-token output tile"). Below B=16 it
// computes a full 16-token tile regardless of how many tokens are real, and it
// launches only M/16 waves. Measured on the ds4 gfx1151 route, the batched
// forward costs `104 + 12.3*B` ms — the 104 ms constant being exactly that
// tile — so a B=1 batched forward takes 115.7 ms where the decode GEMV does
// the identical token in 36.1 ms.
//
// The alternative already in tree is worse: falling back to per-token GEMV
// re-reads the whole dense tier B times (6 x 19.5 ms at B=6).
//
// This kernel takes neither penalty. It keeps the decode GEMV's M-wave
// occupancy (M waves, not M/16) and its single weight-row read, and pays B x
// the arithmetic — which is close to free on a path measured at ~178 GB/s
// (70% of peak) and low single-digit percent of FLOP peak. Activations are
// tiny (B*K*4 = 98 KB at B=6, K=4096) and shared by every row, so they stay
// cache-resident.

#include <hip/hip_runtime.h>

#ifndef E8_BATCHED_B
#define E8_BATCHED_B 6
#endif
#ifndef E8_BATCHED_KERNEL
#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_gfx1151
#endif
#ifndef E8_BATCHED_PAIR
#define E8_BATCHED_PAIR 0
#endif

__device__ __forceinline__ float cvt_e4m3_scale_batched(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index_batched(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sl += e[i];
    }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) {
        float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

// A: E8-SoA [M, K].  X: f32 [B, K] (row-major per token).  Y: f32 [B, M].
__launch_bounds__(32)
#if E8_BATCHED_PAIR
extern "C" __global__ void E8_BATCHED_KERNEL(
    const char* __restrict__ A0,
    const char* __restrict__ A1,
    const float* __restrict__ X,
    float* __restrict__ Y0,
    float* __restrict__ Y1,
    int M,
    int K
) {
    const int matrix = blockIdx.x >= M;
    const int row = blockIdx.x - matrix * M;
    const char* __restrict__ A = matrix ? A1 : A0;
    float* __restrict__ Y = matrix ? Y1 : Y0;
#else
extern "C" __global__ void E8_BATCHED_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K
) {
    const int row = blockIdx.x;
#endif
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;

    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)row * row_total;
    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const char* scale_base = hdr + 16;
    const char* cw_base = hdr + 16 + scale_padded;
    const int block_in_group = tid >> 2;
    const int cw_slot = tid & 3;

    // Four accumulators PER TOKEN, strided by four groups, mirroring
    // `gemv_mfp4g32_e8_soa_u4` operation-for-operation: group g -> acc0,
    // g+1 -> acc1, g+2 -> acc2, g+3 -> acc3, tail groups -> acc0, final
    // combine ((acc0+acc1)+acc2)+acc3. A single sequential accumulator per
    // token is numerically equivalent but NOT bit-identical (measured:
    // 2037/4096 rows differing, max_rel 4.6e-5) because float addition does
    // not reassociate — and this kernel feeds speculative verify, whose logits
    // decide token acceptance, so byte-identity is the gate.
    //
    // B is a compile-time constant so these stay in registers; a runtime bound
    // would force a dynamically indexed array into scratch.
    float acc0[E8_BATCHED_B], acc1[E8_BATCHED_B];
    float acc2[E8_BATCHED_B], acc3[E8_BATCHED_B];
#pragma unroll
    for (int b = 0; b < E8_BATCHED_B; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
        acc2[b] = 0.0f;
        acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int blk0 = g * 8 + block_in_group;
        const int blk1 = blk0 + 8, blk2 = blk0 + 16, blk3 = blk0 + 24;

        const float ssc0 = row_scale_f
            * cvt_e4m3_scale_batched(*(const unsigned char*)(scale_base + blk0)) * 0.88f;
        const float ssc1 = row_scale_f
            * cvt_e4m3_scale_batched(*(const unsigned char*)(scale_base + blk1)) * 0.88f;
        const float ssc2 = row_scale_f
            * cvt_e4m3_scale_batched(*(const unsigned char*)(scale_base + blk2)) * 0.88f;
        const float ssc3 = row_scale_f
            * cvt_e4m3_scale_batched(*(const unsigned char*)(scale_base + blk3)) * 0.88f;

        const unsigned int ew0 =
            *(const unsigned int*)(cw_base + (long long)blk0 * 16 + cw_slot * 4);
        const unsigned int ew1 =
            *(const unsigned int*)(cw_base + (long long)blk1 * 16 + cw_slot * 4);
        const unsigned int ew2 =
            *(const unsigned int*)(cw_base + (long long)blk2 * 16 + cw_slot * 4);
        const unsigned int ew3 =
            *(const unsigned int*)(cw_base + (long long)blk3 * 16 + cw_slot * 4);

        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256, base2 = base0 + 512, base3 = base0 + 768;

        // ONE reused decode buffer, not four. Splitting into pe0..pe3 costs
        // 32 VGPRs and was measured perf-null on the decode kernel (LLVM
        // scalar-replaces the array, so the apparent write-after-read hazard
        // never existed). At B=6 those 32 registers are the difference between
        // 9 and 16 waves/SIMD. Numerics are unchanged: each group's accumulate
        // consumes pe immediately after its own decode.
        float pe[8];

        e8_decode_index_batched(ew0, pe);
#pragma unroll
        for (int b = 0; b < E8_BATCHED_B; b++) {
            const float* xb = X + (long long)b * K;
            acc0[b] += ssc0*pe[0]*xb[base0]+ssc0*pe[1]*xb[base0+1]+ssc0*pe[2]*xb[base0+2]+ssc0*pe[3]*xb[base0+3]
                     + ssc0*pe[4]*xb[base0+4]+ssc0*pe[5]*xb[base0+5]+ssc0*pe[6]*xb[base0+6]+ssc0*pe[7]*xb[base0+7];
        }
        e8_decode_index_batched(ew1, pe);
#pragma unroll
        for (int b = 0; b < E8_BATCHED_B; b++) {
            const float* xb = X + (long long)b * K;
            acc1[b] += ssc1*pe[0]*xb[base1]+ssc1*pe[1]*xb[base1+1]+ssc1*pe[2]*xb[base1+2]+ssc1*pe[3]*xb[base1+3]
                     + ssc1*pe[4]*xb[base1+4]+ssc1*pe[5]*xb[base1+5]+ssc1*pe[6]*xb[base1+6]+ssc1*pe[7]*xb[base1+7];
        }
        e8_decode_index_batched(ew2, pe);
#pragma unroll
        for (int b = 0; b < E8_BATCHED_B; b++) {
            const float* xb = X + (long long)b * K;
            acc2[b] += ssc2*pe[0]*xb[base2]+ssc2*pe[1]*xb[base2+1]+ssc2*pe[2]*xb[base2+2]+ssc2*pe[3]*xb[base2+3]
                     + ssc2*pe[4]*xb[base2+4]+ssc2*pe[5]*xb[base2+5]+ssc2*pe[6]*xb[base2+6]+ssc2*pe[7]*xb[base2+7];
        }
        e8_decode_index_batched(ew3, pe);
#pragma unroll
        for (int b = 0; b < E8_BATCHED_B; b++) {
            const float* xb = X + (long long)b * K;
            acc3[b] += ssc3*pe[0]*xb[base3]+ssc3*pe[1]*xb[base3+1]+ssc3*pe[2]*xb[base3+2]+ssc3*pe[3]*xb[base3+3]
                     + ssc3*pe[4]*xb[base3+4]+ssc3*pe[5]*xb[base3+5]+ssc3*pe[6]*xb[base3+6]+ssc3*pe[7]*xb[base3+7];
        }
    }

    // Tail: remaining 0..3 groups, one at a time into acc0 (matches u4).
    for (int gg = quads << 2; gg < groups_per_row; gg++) {
        const int blk = gg * 8 + block_in_group;
        const float ssc = row_scale_f
            * cvt_e4m3_scale_batched(*(const unsigned char*)(scale_base + blk)) * 0.88f;
        const unsigned int ew =
            *(const unsigned int*)(cw_base + (long long)blk * 16 + cw_slot * 4);
        const int base = gg * 256 + tid * 8;
        float pe[8];
        e8_decode_index_batched(ew, pe);
#pragma unroll
        for (int b = 0; b < E8_BATCHED_B; b++) {
            const float* xb = X + (long long)b * K;
            acc0[b] += ssc*pe[0]*xb[base]+ssc*pe[1]*xb[base+1]+ssc*pe[2]*xb[base+2]+ssc*pe[3]*xb[base+3]
                     + ssc*pe[4]*xb[base+4]+ssc*pe[5]*xb[base+5]+ssc*pe[6]*xb[base+6]+ssc*pe[7]*xb[base+7];
        }
    }

#pragma unroll
    for (int b = 0; b < E8_BATCHED_B; b++) {
        float a = acc0[b] + acc1[b] + acc2[b] + acc3[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down(a, offset);
        }
        if (tid == 0) {
            Y[(long long)b * M + row] = a;
        }
    }
}
