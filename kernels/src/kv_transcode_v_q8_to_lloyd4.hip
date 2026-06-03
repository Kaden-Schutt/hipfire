// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// V transcode: q8_0 → lloyd4 (FWHT-256). Adaptive-KV downshift, all positions of
// one FA layer, in place. This is the ONLY V transcode that performs an FWHT —
// q8 stores V in normal (un-rotated) space, lloyd stores rotated indices.
//
// Per (head, pos):
//   1. dequant the 272 B/head q8 record to a 256-dim normal-space vector
//      (8 blocks × [f16 scale (2B)][int8 × 32 (32B)]; val = int8[e]*scale[b]).
//   2. unit-normalize, FWHT-forward-256 (signs seeds 42/1042), quantize to
//      TURBO_C4_256 with per-(pos,head) cnorm = orig_norm / recon_norm.
//   3. write the 132 B/head lloyd4 record: [4B cnorm][32 threads × 4 bytes].
//
// This is exactly kv_cache_write_fwht256_4bit with the LDS load replaced by a
// q8 dequant. See kernels/src/kv_cache_write_fwht256_4bit.hip (write side) and
// the q8 V-read in attention_flash_q8_0_tile.hip (read side).
//
// ALIASING: src and dst are SEPARATE buffers. The orchestrator (transcode_v_step)
// copies the live layer into a 1-layer scratch and passes scratch as `src`, the
// layer buffer as `dst`. This is crash-safe (a HIP error never half-writes the
// live buffer) AND overlap-safe. NOTE: dst==src (true in-place) is NOT safe for
// q8→lloyd4: dst stride 132B < src stride 272B, but dst_base(p+1)=(p+1)*132*nkv
// overlaps src_read(p)=[p*272*nkv, (p+1)*272*nkv) for small p (e.g. p=0 vs p=1:
// write 528..1056 lands inside read 0..1088), and HIP gives no cross-block
// ordering — so a later position's block can clobber an earlier position's
// still-unread q8 record. Always use a scratch for this kernel.
//
// Grid: [n_kv_heads, n_positions]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_transcode_v_q8_to_lloyd4(
    unsigned char* __restrict__ dst,        // lloyd4 layout (132 B/head/pos)
    const unsigned char* __restrict__ src,  // q8 layout (272 B/head/pos)
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads, int head_dim, int n_positions
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int pos = blockIdx.y;
    if (pos >= n_positions) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // q8 source geometry (mirrors attention_flash_q8_0_tile.hip).
    const int blocks_per_head = head_dim / 32;
    const int q8_bph = blocks_per_head * 34;           // 272 @256
    const int q8_bpp = n_kv_heads * q8_bph;
    const unsigned char* src_head = src + (size_t)pos * q8_bpp + (size_t)h * q8_bph;

    // lloyd4 destination geometry (mirrors kv_cache_write_fwht256_4bit.hip).
    const int l4_bph = 4 + (head_dim * 4) / 8;         // 132 @256
    const int l4_bpp = n_kv_heads * l4_bph;
    unsigned char* out = dst + (size_t)pos * l4_bpp + (size_t)h * l4_bph;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    // --- Dequant q8 head into LDS (normal space) ---
    for (int d = tid; d < head_dim; d += nthreads) {
        const int b = d / 32;       // block index
        const int e = d % 32;       // element within block
        const unsigned char* blk = src_head + (size_t)b * 34;
        float scale = (float)*((const _Float16*)blk);
        signed char q = (signed char)blk[2 + e];
        x[d] = scale * (float)q;
    }
    __syncthreads();

    // --- Unit-normalize ---
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
    __syncthreads();

    // Each thread handles 8 consecutive dims = tid*8 .. tid*8+7.
    const int d0 = tid * 8;
    float v0 = x[d0 + 0], v1 = x[d0 + 1];
    float v2 = x[d0 + 2], v3 = x[d0 + 3];
    float v4 = x[d0 + 4], v5 = x[d0 + 5];
    float v6 = x[d0 + 6], v7 = x[d0 + 7];

    // Single-pass signed-FWHT over the full 256 dims.
    fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

    int idx[8];
    idx[0] = turbo_quantize_4bit_256(v0);
    idx[1] = turbo_quantize_4bit_256(v1);
    idx[2] = turbo_quantize_4bit_256(v2);
    idx[3] = turbo_quantize_4bit_256(v3);
    idx[4] = turbo_quantize_4bit_256(v4);
    idx[5] = turbo_quantize_4bit_256(v5);
    idx[6] = turbo_quantize_4bit_256(v6);
    idx[7] = turbo_quantize_4bit_256(v7);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) local_rsq += TURBO_C4_256[idx[i]] * TURBO_C4_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 15)) << (i * 4);
    unsigned char* base = out + 4 + tid * 4;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
    base[3] = (unsigned char)((packed >> 24) & 0xFF);
}
