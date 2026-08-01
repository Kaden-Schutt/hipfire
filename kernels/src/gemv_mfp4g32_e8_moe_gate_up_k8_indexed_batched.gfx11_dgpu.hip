// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 MoE gate_up decode GEMV, gfx1100 dGPU (RDNA3 discrete).
//
// gfx1100 (RX 7900 XTX) twin of the gfx1151 kernel.
// DO NOT DISPATCH on gfx1151 (Strix Halo iGPU) — use the .gfx1151.hip sibling.
// Guard: arch_caps.is_rdna3_dgpu() in gemv.rs.
//
// Strategy (vs gfx1151 sibling):
//   gfx1151 uses __launch_bounds__(32, 7) + 2-way unroll → 62 VGPR / 16 waves.
//   gfx1100 has ~4x BW (GDDR6 ~960 GB/s + 96MB ∞-cache) vs gfx1151 LPDDR5.
//   BW is higher so pipeline fill requires MORE outstanding loads per wave, not
//   fewer. Strategy: 4-way unroll (matching hfq4g256 quads at 150 tok/s) with
//   4 independent accumulators + all loads issued before first decode/MAC.
//   The pe[8] stack array that the gfx1151 shared helper uses is hoisted into 8
//   scalar temporaries via an inline decode macro, which lets the compiler
//   eliminate the array alloc and reach ~80 VGPR (12 waves).
//
// Math is BYTE-IDENTICAL to the gfx1151 kernel.
// 0 register spills (private_segment_fixed_size == 0 confirmed at gfx1100).
//
// HIPFIRE_E8_DGPU_TWIN: default OFF (gated but preserved). Measured on gfx1100
// 2026-06-13: delta ~0% (101.3 vs 101.7 tok/s, within noise). Root cause: 96MB
// Infinity Cache keeps expert weights hot, so occupancy drop 16→10 waves offsets
// the ILP gain. Set =1 to activate for experiments (batched / spill-to-GDDR6).

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f; // defensive: NaN code -> max finite (never emitted)
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}
#endif

// Expand one 32-bit E8 lattice codeword into 8 float coordinates without
// allocating a pe[8] array — each coordinate is a named scalar so the compiler
// can assign them to individual VGPRs without spilling an intermediate array.
// Same arithmetic as e8_decode_index; output is 8 scalars p0..p7.
#define E8_DECODE(idx, p0, p1, p2, p3, p4, p5, p6, p7) do { \
    const int _coset = (int)(((idx) >> 31) & 1u); \
    const unsigned int _e0 = ((idx) >> 0)  & 0xFu; \
    const unsigned int _e1 = ((idx) >> 4)  & 0xFu; \
    const unsigned int _e2 = ((idx) >> 8)  & 0xFu; \
    const unsigned int _e3 = ((idx) >> 12) & 0xFu; \
    const unsigned int _e4 = ((idx) >> 16) & 0xFu; \
    const unsigned int _e5 = ((idx) >> 20) & 0xFu; \
    const unsigned int _e6 = ((idx) >> 24) & 0xFu; \
    const unsigned int _sl = _e0+_e1+_e2+_e3+_e4+_e5+_e6; \
    const unsigned int _e7h = ((idx) >> 28) & 0x7u; \
    const unsigned int _p7u = _e7h << 1; \
    const unsigned int _lsb = (_sl + _p7u) & 1u; \
    const unsigned int _e7 = _p7u | _lsb; \
    const float _h = _coset ? 0.5f : 0.0f; \
    (p0) = (float)((int)_e0 - 7) + _h; \
    (p1) = (float)((int)_e1 - 7) + _h; \
    (p2) = (float)((int)_e2 - 7) + _h; \
    (p3) = (float)((int)_e3 - 7) + _h; \
    (p4) = (float)((int)_e4 - 7) + _h; \
    (p5) = (float)((int)_e5 - 7) + _h; \
    (p6) = (float)((int)_e6 - 7) + _h; \
    (p7) = (float)((int)_e7 - 7) + _h; \
} while (0)

// One output row of an E8 expert weight dot-producted against x_row.
// 4-way group unroll: 4 groups × 256 weights = 1024 weights per loop iteration.
// All 4 group loads are issued before any decode/MAC — fills the gfx1100 L∞ cache.
__device__ __forceinline__ float e8_moe_row_acc_dgpu(
    const char* __restrict__ A, const float* __restrict__ x, int row, int K, int tid
) {
    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = tid >> 2;
    const int cw_off = (tid & 3) * 4;
    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g       * 136 + thread_block_off;
        const char* gp1 = row_ptr + (long long)(g + 1) * 136 + thread_block_off;
        const char* gp2 = row_ptr + (long long)(g + 2) * 136 + thread_block_off;
        const char* gp3 = row_ptr + (long long)(g + 3) * 136 + thread_block_off;

        // All loads before first decode — 4 independent streams in flight.
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;
        const float ssc2 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp2) * 0.88f;
        const float ssc3 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp3) * 0.88f;

        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);
        const unsigned int ew2 = *(const unsigned int*)(gp2 + 1 + cw_off);
        const unsigned int ew3 = *(const unsigned int*)(gp3 + 1 + cw_off);

        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;
        const int base2 = base0 + 512;
        const int base3 = base0 + 768;

        // Decode group 0 into scalar p0..p7, accumulate into acc0.
        {
            float p0,p1,p2,p3,p4,p5,p6,p7;
            E8_DECODE(ew0, p0,p1,p2,p3,p4,p5,p6,p7);
            acc0 += ssc0*p0*x[base0  ] + ssc0*p1*x[base0+1] + ssc0*p2*x[base0+2] + ssc0*p3*x[base0+3]
                  + ssc0*p4*x[base0+4] + ssc0*p5*x[base0+5] + ssc0*p6*x[base0+6] + ssc0*p7*x[base0+7];
        }
        // Decode group 1 → acc1.
        {
            float p0,p1,p2,p3,p4,p5,p6,p7;
            E8_DECODE(ew1, p0,p1,p2,p3,p4,p5,p6,p7);
            acc1 += ssc1*p0*x[base1  ] + ssc1*p1*x[base1+1] + ssc1*p2*x[base1+2] + ssc1*p3*x[base1+3]
                  + ssc1*p4*x[base1+4] + ssc1*p5*x[base1+5] + ssc1*p6*x[base1+6] + ssc1*p7*x[base1+7];
        }
        // Decode group 2 → acc2.
        {
            float p0,p1,p2,p3,p4,p5,p6,p7;
            E8_DECODE(ew2, p0,p1,p2,p3,p4,p5,p6,p7);
            acc2 += ssc2*p0*x[base2  ] + ssc2*p1*x[base2+1] + ssc2*p2*x[base2+2] + ssc2*p3*x[base2+3]
                  + ssc2*p4*x[base2+4] + ssc2*p5*x[base2+5] + ssc2*p6*x[base2+6] + ssc2*p7*x[base2+7];
        }
        // Decode group 3 → acc3.
        {
            float p0,p1,p2,p3,p4,p5,p6,p7;
            E8_DECODE(ew3, p0,p1,p2,p3,p4,p5,p6,p7);
            acc3 += ssc3*p0*x[base3  ] + ssc3*p1*x[base3+1] + ssc3*p2*x[base3+2] + ssc3*p3*x[base3+3]
                  + ssc3*p4*x[base3+4] + ssc3*p5*x[base3+5] + ssc3*p6*x[base3+6] + ssc3*p7*x[base3+7];
        }
    }

    // Tail: 0..3 remaining groups.
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float p0,p1,p2,p3,p4,p5,p6,p7;
        E8_DECODE(ew, p0,p1,p2,p3,p4,p5,p6,p7);
        acc0 += ssc*p0*x[base  ] + ssc*p1*x[base+1] + ssc*p2*x[base+2] + ssc*p3*x[base+3]
              + ssc*p4*x[base+4] + ssc*p5*x[base+5] + ssc*p6*x[base+6] + ssc*p7*x[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float p0,p1,p2,p3,p4,p5,p6,p7;
        E8_DECODE(ew, p0,p1,p2,p3,p4,p5,p6,p7);
        acc1 += ssc*p0*x[base  ] + ssc*p1*x[base+1] + ssc*p2*x[base+2] + ssc*p3*x[base+3]
              + ssc*p4*x[base+4] + ssc*p5*x[base+5] + ssc*p6*x[base+6] + ssc*p7*x[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float p0,p1,p2,p3,p4,p5,p6,p7;
        E8_DECODE(ew, p0,p1,p2,p3,p4,p5,p6,p7);
        acc2 += ssc*p0*x[base  ] + ssc*p1*x[base+1] + ssc*p2*x[base+2] + ssc*p3*x[base+3]
              + ssc*p4*x[base+4] + ssc*p5*x[base+5] + ssc*p6*x[base+6] + ssc*p7*x[base+7];
    }

    // Fold 4 partials and reduce across wave32.
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    return acc;
}

// __launch_bounds__(32, 4): allows ~112 VGPR budget before compiler must spill.
// Actual VGPR target ~80 (12 waves/SIMD), same as hfq4g256 150 tok/s reference.
// Do NOT use (32, 16): caps VGPR <=64, negating the 4-way unroll benefit.
__launch_bounds__(32, 4)
extern "C" __global__ void gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched_dgpu(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]  (FWHT-rotated)
    float*       __restrict__ y_gate,                    // [N × K_TOP × MI]
    float*       __restrict__ y_up,                      // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const float acc = e8_moe_row_acc_dgpu(A, x_row, row, K, tid);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) y_gate[base_out + row] = acc;
        else          y_up[base_out + (row - mi)] = acc;
    }
}
