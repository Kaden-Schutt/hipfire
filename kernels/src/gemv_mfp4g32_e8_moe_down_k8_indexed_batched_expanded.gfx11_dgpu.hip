// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 grouped MoE down decode GEMV (atomic-free expanded), gfx11_dgpu.
//
// RDNA3 dGPU sister of gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded.gfx1151.hip.
// Target: gfx1100 / RX 7900 XTX (96 CU, GDDR6 ~960 GB/s, 96 MB Infinity Cache L3).
//
// Same design rationale as the gate_up twin: gfx1151's (32,7) → 62 VGPR → 16 waves/SIMD
// is MAX but zero latency-hiding headroom. gfx1100 dGPU version uses a 4-way group unroll
// with 4 independent accumulators + all loads issued before decodes → latency-hiding for
// the memory-bound GEMV. __launch_bounds__(32, 4) → ~94 VGPR → 10 waves/SIMD on gfx1100.
//
// Kernel contract: identical to gfx1151 down kernel.
//   - Grid: (M, K_TOP, N)
//   - Block: (32, 1, 1)
//   - expert_outputs[bid * K_TOP * M + krank * M + row] = dot(A_expert[row], x_token_expert)
//   - 0 atomicAdd. Caller folds via moe_down_combine_k8_batched.
//
// Per-row math BYTE-IDENTICAL to gfx1151 — only load schedule / unroll depth differs.
// 0 register spills (private_segment_fixed_size must be 0).

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

__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}

// 4-way unroll row accumulator. Same contract as the gate_up twin.
// K must be a multiple of 256 (groups_per_row integer); A3B has K=2048 → 8 groups → quads=2.
__device__ __forceinline__ float e8_moe_row_acc_dgpu(
    const char* __restrict__ A, const float* __restrict__ x, int row, int K, int tid
) {
    const int groups_per_row  = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off   = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr    = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter   = tid >> 2;
    const int cw_off          = (tid & 3) * 4;
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

        float pe[8];
        e8_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*x[base0  ] + ssc0*pe[1]*x[base0+1] + ssc0*pe[2]*x[base0+2] + ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4] + ssc0*pe[5]*x[base0+5] + ssc0*pe[6]*x[base0+6] + ssc0*pe[7]*x[base0+7];

        e8_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1  ] + ssc1*pe[1]*x[base1+1] + ssc1*pe[2]*x[base1+2] + ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4] + ssc1*pe[5]*x[base1+5] + ssc1*pe[6]*x[base1+6] + ssc1*pe[7]*x[base1+7];

        e8_decode_index(ew2, pe);
        acc2 += ssc2*pe[0]*x[base2  ] + ssc2*pe[1]*x[base2+1] + ssc2*pe[2]*x[base2+2] + ssc2*pe[3]*x[base2+3]
              + ssc2*pe[4]*x[base2+4] + ssc2*pe[5]*x[base2+5] + ssc2*pe[6]*x[base2+6] + ssc2*pe[7]*x[base2+7];

        e8_decode_index(ew3, pe);
        acc3 += ssc3*pe[0]*x[base3  ] + ssc3*pe[1]*x[base3+1] + ssc3*pe[2]*x[base3+2] + ssc3*pe[3]*x[base3+3]
              + ssc3*pe[4]*x[base3+4] + ssc3*pe[5]*x[base3+5] + ssc3*pe[6]*x[base3+6] + ssc3*pe[7]*x[base3+7];
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc1 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc2 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    return acc;
}

__launch_bounds__(32, 4)
extern "C" __global__ void gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded_dgpu(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M] (pure store)
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid   = blockIdx.z;
    const int tid   = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id    = topk_indices[routing_base + krank];
    const char*  __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + ((long long)bid * K_TOP + krank) * K;

    const float acc = e8_moe_row_acc_dgpu(A, x, row, K, tid);

    if (tid == 0) {
        const long long out_offset = ((long long)bid * K_TOP + krank) * (long long)M + row;
        expert_outputs[out_offset] = acc;
    }
}
