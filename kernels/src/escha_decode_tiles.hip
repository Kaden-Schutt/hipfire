// Escha-W2 one-shot tile decode -> Q8_0 resident weights (Phase 1).
//
// Reads the verbatim int16 code stream, decodes 16x16 trellis tiles, and
// writes hipfire Q8_0 (34 bytes per 32 elements: f16 scale + 32 int8).
//
// The codebook is computed inline. A 65536-entry fp16 LUT would be 128 KB and
// gfx1151 has 64 KB LDS, so there is no table anywhere in this file.
//
// Escha's tile grid is in-major [in/16, out/16]; hipfire stores weights
// out-major [out, in]. This kernel transposes on the way out.
//
// DELIBERATE DUPLICATION: hipfire-quantize/src/escha_ref.rs implements this
// same lane maths in Rust. That is the G2 gate — this kernel is asserted
// bit-exact against it. Generating either from the other would make G2
// circular. Do not deduplicate.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ __half escha_cba(unsigned short state) {
    unsigned int r = ((unsigned int)state * 0xCBAC1FEDu) & 0x8FFF8FFFu;
    r ^= 0x3B603B60u;
    __half lo = __ushort_as_half((unsigned short)(r & 0xFFFFu));
    __half hi = __ushort_as_half((unsigned short)(r >> 16));
    return __hadd(lo, hi);  // fp16 RNE add — matches the reference exactly
}

__device__ __forceinline__ void escha_decode8_k2(
    const unsigned int* w, int lane, unsigned short* out) {
    int t_off = lane * 8;
    int i1 = t_off >> 4;
    int i0 = (i1 + 15) & 15;
    unsigned long long merged = ((unsigned long long)w[i0] << 32) | w[i1];
    int shift = ((~t_off) & 8) << 1;   // 16 for even lanes, 0 for odd
    unsigned int v = (unsigned int)((merged >> shift) & 0xFFFFFFFFull);
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (unsigned short)(v >> (2 * (7 - j)));
}

__device__ __forceinline__ void escha_decode8_k3(
    const unsigned int* w, int lane, unsigned short* out) {
    const int BITS = 3;
    int t_off = lane * 8;
    int b1 = (t_off + 257) * BITS;
    int b0 = b1 - 16;
    int b2 = b1 + BITS * 7;
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    int s2 = ((i2 + 1) << 5) - b2;
    unsigned long long merged =
        ((unsigned long long)w[i0 % 24] << 32) | w[i2 % 24];
    unsigned int w7 = (unsigned int)((merged >> s2) & 0xFFFFFFFFull);
    unsigned int w3 = (unsigned int)((merged >> (s2 + BITS * 4)) & 0xFFFFFFFFull);
    out[0] = (unsigned short)(w3 >> 9); out[1] = (unsigned short)(w3 >> 6);
    out[2] = (unsigned short)(w3 >> 3); out[3] = (unsigned short)(w3);
    out[4] = (unsigned short)(w7 >> 9); out[5] = (unsigned short)(w7 >> 6);
    out[6] = (unsigned short)(w7 >> 3); out[7] = (unsigned short)(w7);
}

// One block per tile. 32 lanes, 8 values each = the 256 tile slots.
extern "C" __global__ void escha_decode_tiles(
    const short* __restrict__ code,   // [in/16, out/16, 16K]
    __half* __restrict__ bare,        // [in, out] fp16 scratch
    int in_features, int out_features, int K) {
    int tile = blockIdx.x;
    int tn = out_features / 16;
    int kt = tile / tn, nt = tile % tn;
    int lane = threadIdx.x;
    if (lane >= 32) return;

    unsigned int words[24];
    const short* src = code + (size_t)tile * 16 * K;
    for (int i = 0; i < 8 * K; ++i)
        words[i] = ((unsigned int)(unsigned short)src[2 * i]) |
                   (((unsigned int)(unsigned short)src[2 * i + 1]) << 16);

    unsigned short st[8];
    if (K == 2) escha_decode8_k2(words, lane, st);
    else        escha_decode8_k3(words, lane, st);

    int l0 = lane & ~4;
    int c_off = (lane >> 2) & 1;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int fi = j >> 1;
        int row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8;
        int col = 2 * ((l0 >> 3) + (j >= 4 ? 4 : 0)) + c_off;
        bare[(size_t)(kt * 16 + row) * out_features + (nt * 16 + col)] = escha_cba(st[j]);
    }
}
