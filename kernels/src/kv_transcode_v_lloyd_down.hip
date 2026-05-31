// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// V transcode: lloyd_hi → lloyd_lo, within the 256-LUT family. Adaptive-KV
// downshift (lloyd4→3, lloyd3→2, and lloyd4→2 for the direct path), all
// positions of one FA layer, in place. NO FWHT — the values are already in
// rotated space; we only re-quantize to a coarser LUT.
//
// Per (head, pos), per thread (owns 8 rotated dims = tid*8 .. tid*8+7):
//   1. read src cnorm + the 8 packed src-bit indices for this thread.
//   2. reconstruct the rotated value  rv = cnorm_hi * TURBO_C{src}_256[idx_hi].
//   3. re-quantize: idx_lo = turbo_quantize_{dst}bit_256(rv).
//   4. recompute cnorm_lo = orig_norm / recon_norm over the reconstructed
//      rotated vector (orig_norm = ||rv||, recon_norm = ||LUT_dst[idx_lo]||),
//      and repack at the dst bit width.
//
// This is equivalent to re-running the lloyd write on the already-rotated
// reconstructed values (the FWHT-write path normalizes, quantizes, and stores
// cnorm = orig/recon). Because the cnorm scale differs between tiers, we do NOT
// just remap the index — we reconstruct the value and recompute cnorm.
//
// ALIASING: src and dst are SEPARATE buffers. The orchestrator copies the live
// layer into a 1-layer scratch (src) and writes to the layer (dst). This is
// crash-safe and overlap-safe. (True in-place dst==src would also be safe on
// these downshifts since each thread reads its src indices into registers before
// writing AND the dst record is a prefix-compaction — but we use a scratch for
// uniformity with the q8→lloyd4 path, which is NOT in-place safe.)
//
// Grid: [n_kv_heads, n_positions]. Block: [32]. LDS: (head_dim + 32) * 4 bytes
// (only the +32 scratch tail is used; sized to match the V-write template).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Read 8 packed indices for thread `tid` from a lloyd record at `bits` width.
// rec points at the record base ([4B cnorm][32 threads × bits bytes]).
static __device__ inline void unpack8(const unsigned char* rec, int tid, int bits, int idx[8]) {
    const int bytes = bits;  // 2,3,4 bytes/thread for 2,3,4-bit (8 idx * bits / 8)
    const unsigned char* base = rec + 4 + tid * bytes;
    unsigned int packed = (unsigned int)base[0];
    if (bytes >= 2) packed |= ((unsigned int)base[1] << 8);
    if (bytes >= 3) packed |= ((unsigned int)base[2] << 16);
    if (bytes >= 4) packed |= ((unsigned int)base[3] << 24);
    const int mask = (1 << bits) - 1;
    for (int i = 0; i < 8; i++) idx[i] = (int)((packed >> (i * bits)) & (unsigned int)mask);
}

static __device__ inline float lut_lookup(int bits, int idx) {
    if (bits == 2) return TURBO_C2_256[idx];
    if (bits == 3) return TURBO_C3_256[idx];
    return TURBO_C4_256[idx];
}

static __device__ inline int quantize_dst(int bits, float x) {
    if (bits == 2) return turbo_quantize_2bit_256(x);
    if (bits == 3) return turbo_quantize_3bit_256(x);
    return turbo_quantize_4bit_256(x);
}

extern "C" __global__ void kv_transcode_v_lloyd_down(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    int n_kv_heads, int head_dim, int n_positions, int src_bits, int dst_bits
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int pos = blockIdx.y;
    if (pos >= n_positions) return;
    const int tid = threadIdx.x;

    const int src_bph = 4 + (head_dim * src_bits) / 8;
    const int src_bpp = n_kv_heads * src_bph;
    const int dst_bph = 4 + (head_dim * dst_bits) / 8;
    const int dst_bpp = n_kv_heads * dst_bph;

    const unsigned char* src_rec = src + (size_t)pos * src_bpp + (size_t)h * src_bph;
    unsigned char* out = dst + (size_t)pos * dst_bpp + (size_t)h * dst_bph;

    extern __shared__ float smem[];
    float* scratch = smem + head_dim;

    // Read src cnorm + indices into registers BEFORE any write (in-place safety).
    float cnorm_hi = *(const float*)src_rec;
    int sidx[8];
    unpack8(src_rec, tid, src_bits, sidx);

    // Reconstruct the rotated values for this thread's 8 dims.
    float rv[8];
    for (int i = 0; i < 8; i++) rv[i] = cnorm_hi * lut_lookup(src_bits, sidx[i]);

    // orig_norm = ||reconstructed rotated vector|| (reduce local sq across threads).
    float local_osq = 0.0f;
    for (int i = 0; i < 8; i++) local_osq += rv[i] * rv[i];
    scratch[tid] = local_osq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);

    // Re-quantize at dst tier. The dst LUT is fit for unit-norm rotated values,
    // so quantize the unit-normalized reconstruction (matches the write kernels,
    // which normalize before quantizing).
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    int didx[8];
    for (int i = 0; i < 8; i++) didx[i] = quantize_dst(dst_bits, rv[i] * inv_norm);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) {
        float c = lut_lookup(dst_bits, didx[i]);
        local_rsq += c * c;
    }
    __syncthreads();
    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm_lo = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm_lo;

    // Repack at dst bit width.
    unsigned int packed = 0;
    const unsigned int mask = (1u << dst_bits) - 1u;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(didx[i]) & mask) << (i * dst_bits);
    const int dbytes = dst_bits;  // 2,3,4 bytes/thread
    unsigned char* base = out + 4 + tid * dbytes;
    base[0] = (unsigned char)(packed & 0xFF);
    if (dbytes >= 2) base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    if (dbytes >= 3) base[2] = (unsigned char)((packed >> 16) & 0xFF);
    if (dbytes >= 4) base[3] = (unsigned char)((packed >> 24) & 0xFF);
}
