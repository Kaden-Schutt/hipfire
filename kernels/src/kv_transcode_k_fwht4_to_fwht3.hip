// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// K transcode: fwht4 → fwht3, RE-ROTATION (128-wide → 256-wide width change).
// Adaptive-KV K downshift, the ONLY re-rotation case (advanced selector,
// k_floor=fwht3). All positions of one FA layer, src→dst (separate buffers).
//
// Unlike the same-width fwht4→fwht2 remap, fwht4 (TWO 128-wide FWHT halves)
// and fwht3 (ONE 256-wide FWHT) DO NOT share a rotation basis, so the stored
// rotated coefficients are NOT reusable. We must round-trip through normal
// space:
//   1. Read the fwht4 record → dequant each dim  cnorm4 * TURBO_C4[idx4]  →
//      the 128-wide-rotated coefficients for each of the two halves.
//   2. INVERSE-rotate each 128-half back to normal space (fwht_shfl_inverse,
//      128-wide) using signs1/signs2[0..127]. Thread tid recovers normal-space
//      dims  half*128 + tid*4 + {0,1,2,3}. Write them to LDS at that global
//      normal-space dim so the forward-256 pass can pick them up by tid*8.
//   3. FORWARD-rotate the full 256-dim normal-space vector (fwht_shfl_forward_256)
//      using signs1/signs2[0..255]. Thread tid owns normal dims tid*8..tid*8+7.
//   4. Quantize to TURBO_C3_256 (turbo_quantize_3bit_256), recompute cnorm,
//      write the fwht3 record — packing byte-identical to
//      kv_cache_write_asym_k_fwht3 (8 dims/thread, base = out+4+tid*3,
//      3 bits/dim packed little-endian into 24 bits).
//
// cnorm semantics (matches the writers): stored cnorm = orig_norm / recon_norm
// where orig_norm = ||normal-space K reconstructed from fwht4|| and recon_norm =
// ||quantized-3bit reconstruction in the rotated space||. The FWHT is
// orthonormal, so ||rotated|| == ||normal||; we therefore reduce orig_norm over
// the normal-space reconstruction (post inverse, pre re-normalization) and feed
// the UNIT-normalized normal vector to the forward-256, exactly as the fwht3
// write kernel normalizes its input before rotating.
//
// SIGN WIDTH: the inverse-128 reads signs[0..127]; the forward-256 reads
// signs[0..255]. A 256-wide sign table satisfies BOTH (the LCG prefix makes the
// first 128 entries byte-identical to a 128-wide table, so the inverse matches
// what fwht4's forward used). The orchestrator (set_adaptive_floor_alloc with
// k_floor=fwht3) upgrades the cache signs to 256-wide before this runs.
//
// ALIASING: src and dst are SEPARATE buffers (orchestrator uses a 1-layer
// scratch; do NOT alias). dst stride (fwht3, 100B@256) < src stride (fwht4,
// 132B@256).
//
// Grid: [n_kv_heads, n_positions]. Block: [32]. LDS: (head_dim + 32) * 4 bytes
// (head_dim floats hold the normal-space K between the inverse and forward
// passes; the +32 tail is the norm-reduce scratch).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_transcode_k_fwht4_to_fwht3(
    unsigned char* __restrict__ dst,        // fwht3 layout (100 B/head/pos @256)
    const unsigned char* __restrict__ src,  // fwht4 layout (132 B/head/pos @256)
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads, int head_dim, int n_positions
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int pos = blockIdx.y;
    if (pos >= n_positions) return;
    const int tid = threadIdx.x;

    const int n_halves = head_dim / 128;  // 2 @256

    const int src_bph = 4 + head_dim / 2;        // fwht4: 132 @256
    const int src_bpp = n_kv_heads * src_bph;
    const int dst_bph = 4 + (head_dim * 3) / 8;  // fwht3: 100 @256
    const int dst_bpp = n_kv_heads * dst_bph;

    const unsigned char* src_rec = src + (size_t)pos * src_bpp + (size_t)h * src_bph;
    unsigned char* out = dst + (size_t)pos * dst_bpp + (size_t)h * dst_bph;

    extern __shared__ float smem[];
    float* x = smem;                 // head_dim normal-space K (between passes)
    float* scratch = smem + head_dim;

    const float cnorm4 = *(const float*)src_rec;

    // --- Step 1+2: per 128-half, dequant fwht4 coeffs then inverse-rotate to
    // normal space; deposit into LDS at the global normal-space dim index. ---
    for (int half = 0; half < n_halves; half++) {
        // fwht4: 2 bytes/thread/half at src_rec[4 + half*64 + tid*2].
        const unsigned char* b = src_rec + 4 + half * 64 + tid * 2;
        int i0 = (int)(b[0] & 0xF);
        int i1 = (int)((b[0] >> 4) & 0xF);
        int i2 = (int)(b[1] & 0xF);
        int i3 = (int)((b[1] >> 4) & 0xF);
        // Reconstructed 128-wide-rotated coefficients (dequant, scale by cnorm4).
        float fa = cnorm4 * TURBO_C4[i0];
        float fb = cnorm4 * TURBO_C4[i1];
        float fc = cnorm4 * TURBO_C4[i2];
        float fd = cnorm4 * TURBO_C4[i3];
        // Inverse 128-wide FWHT: (fa,fb,fc,fd) -> normal-space dims
        // tid*4..tid*4+3 within this half. signs[0..127] (first 128 entries).
        fwht_shfl_inverse(fa, fb, fc, fd, signs1, signs2, tid);
        // Deposit at the global normal-space dim: half*128 + tid*4 + {0..3}.
        int g = half * 128 + tid * 4;
        x[g + 0] = fa; x[g + 1] = fb; x[g + 2] = fc; x[g + 3] = fd;
    }
    __syncthreads();

    // --- orig_norm over the WHOLE head (256 normal-space dims). The FWHT is
    // orthonormal so ||normal|| == ||rotated||; we reduce in normal space. ---
    float local_osq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) local_osq += x[d] * x[d];
    scratch[tid] = local_osq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    // --- Step 3: load this thread's 8 normal-space dims (tid*8..tid*8+7),
    // unit-normalize, forward-rotate 256-wide. ---
    const int d0 = tid * 8;
    float v0 = x[d0 + 0] * inv_norm, v1 = x[d0 + 1] * inv_norm;
    float v2 = x[d0 + 2] * inv_norm, v3 = x[d0 + 3] * inv_norm;
    float v4 = x[d0 + 4] * inv_norm, v5 = x[d0 + 5] * inv_norm;
    float v6 = x[d0 + 6] * inv_norm, v7 = x[d0 + 7] * inv_norm;
    __syncthreads();  // LDS x[] fully read before any reuse (scratch reuse below)

    fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

    // --- Step 4: quantize to 3-bit (256-family), recompute cnorm, pack. ---
    int idx[8];
    idx[0] = turbo_quantize_3bit_256(v0);
    idx[1] = turbo_quantize_3bit_256(v1);
    idx[2] = turbo_quantize_3bit_256(v2);
    idx[3] = turbo_quantize_3bit_256(v3);
    idx[4] = turbo_quantize_3bit_256(v4);
    idx[5] = turbo_quantize_3bit_256(v5);
    idx[6] = turbo_quantize_3bit_256(v6);
    idx[7] = turbo_quantize_3bit_256(v7);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) local_rsq += TURBO_C3_256[idx[i]] * TURBO_C3_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    // Pack 8×3-bit indices into 24 bits = 3 bytes per thread, at the EXACT slot
    // kv_cache_write_asym_k_fwht3 uses (out[4 + tid*3]).
    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 7)) << (i * 3);
    unsigned char* base = out + 4 + tid * 3;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
}
