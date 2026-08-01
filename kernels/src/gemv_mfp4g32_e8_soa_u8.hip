// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 SoA decode-GEMV, 8-WAY UNROLL variant (bench experiment).
//
// Same SoA layout + math as gemv_mfp4g32_e8_soa, 8 groups/iter, 8 accumulators,
// all loads before decode. Probes whether MLP keeps climbing toward the dGPU
// cache roofline past 4-way, or saturates.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32_soa(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32_soa(unsigned char b) {
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

__device__ __forceinline__ void e8_decode_index_soa(unsigned int idx, float p[8]) {
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

__device__ __forceinline__ float e8_row_dot(float ssc, const float pe[8], const float* x, int base) {
    return ssc*pe[0]*x[base]+ssc*pe[1]*x[base+1]+ssc*pe[2]*x[base+2]+ssc*pe[3]*x[base+3]
         + ssc*pe[4]*x[base+4]+ssc*pe[5]*x[base+5]+ssc*pe[6]*x[base+6]+ssc*pe[7]*x[base+7];
}

__launch_bounds__(32)
extern "C" __global__ void gemv_mfp4g32_e8_soa_u8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int n_blocks       = K / 32;
    const int groups_per_row = K / 256;

    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)row * row_total;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const char* scale_base = hdr + 16;
    const char* cw_base    = hdr + 16 + scale_padded;

    const int block_in_group = tid >> 2;
    const int cw_slot        = tid & 3;

    float acc[8] = {0,0,0,0,0,0,0,0};
    const int octs = groups_per_row >> 3;
    const int done = octs << 3;

    for (int o = 0; o < octs; o++) {
        const int g = o << 3;
        float ssc[8];
        unsigned int ew[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int b = (g + i) * 8 + block_in_group;
            ssc[i] = row_scale_f * cvt_e4m3_scale_to_f32_soa(*(const unsigned char*)(scale_base + b)) * 0.88f;
            ew[i]  = *(const unsigned int*)(cw_base + (long long)b * 16 + cw_slot * 4);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float pe[8];
            e8_decode_index_soa(ew[i], pe);
            acc[i] += e8_row_dot(ssc[i], pe, x, (g + i) * 256 + tid * 8);
        }
    }

    // Tail: remaining 0..7 groups into acc[0].
    for (int gg = done; gg < groups_per_row; gg++) {
        const int b = gg * 8 + block_in_group;
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32_soa(*(const unsigned char*)(scale_base + b)) * 0.88f;
        const unsigned int ew0 = *(const unsigned int*)(cw_base + (long long)b * 16 + cw_slot * 4);
        float pe[8];
        e8_decode_index_soa(ew0, pe);
        acc[0] += e8_row_dot(ssc0, pe, x, gg * 256 + tid * 8);
    }

    float a = acc[0]+acc[1]+acc[2]+acc[3]+acc[4]+acc[5]+acc[6]+acc[7];
    for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down(a, offset);
    if (tid == 0) y[row] = a;
}
