// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// K transcode: fwht4 → fwht2, SAME-WIDTH remap (both 128-wide rotation).
// Adaptive-KV K downshift (the only K step the balanced/aggressive presets
// take). All positions of one FA layer, src→dst (separate buffers).
//
// NO FWHT. fwht4 and fwht2 rotate 128-wide with the IDENTICAL signed-FWHT
// basis (same signs1/signs2), so the rotated coefficients already stored in
// the fwht4 record are exactly the rotated coefficients fwht2 would store —
// we only re-quantize each rotated dim from the 4-bit 128-family LUT
// (TURBO_C4) to the 2-bit 128-family LUT (TURBO_C2) and recompute cnorm.
// This is the K-side twin of kv_transcode_v_lloyd_down.hip; the differences
// are (1) the 128-dim LUT family (NOT the _256 family) and the matching
// turbo_quantize_2bit (NOT _256), and (2) the per-thread dim layout +
// bit-packing, which mirror kv_cache_write_asym_k_fwht4 / _fwht2 EXACTLY so
// a subsequent fwht2 attention read sees a correct record.
//
// LAYOUT PRESERVATION (must round-trip the writers):
//   Both K writers use 32 threads, 4 dims/thread per 128-wide half,
//   n_halves = head_dim/128. Thread `tid` in half `half` owns the post-FWHT
//   coefficients with index slots  base = half*4 + {0,1,2,3}.
//   fwht4 record: [4B cnorm][per half: 64 bytes = 32 thr × 2 bytes]
//     out[4 + half*64 + tid*2]     = (idx[base+1] << 4) | (idx[base+0] & 0xF)
//     out[4 + half*64 + tid*2 + 1] = (idx[base+3] << 4) | (idx[base+2] & 0xF)
//   fwht2 record: [4B cnorm][per half: 32 bytes = 32 thr × 1 byte]
//     out[4 + half*32 + tid]       = idx0 | idx1<<2 | idx2<<4 | idx3<<6
//   We read the 4 fwht4 nibble indices for (half, tid), dequant via
//   cnorm4 * TURBO_C4[idx], and re-pack the corresponding 4 fwht2 indices
//   into the fwht2 slot for (half, tid) — preserving the writer's exact
//   slot→dim correspondence. cnorm is ONE scalar per head spanning both
//   halves (the writers normalize over the whole head_dim and reduce
//   local_rsq across both halves), so we reduce across all 32 threads ONCE.
//
// cnorm semantics (matches the writers): stored cnorm = orig_norm /
// recon_norm where orig_norm = ||reconstructed unit-normalized rotated head||
// and recon_norm = ||quantized-LUT reconstruction||. Reconstructing from the
// fwht4 record already yields the unit-normalized rotated vector scaled by
// cnorm4 (cnorm4 ≈ orig_norm4/recon_norm4 ≈ 1 since fwht4's recon is close to
// unit). We renormalize that reconstruction to unit before re-quantizing at
// 2-bit, exactly as the lloyd-down V transcode does.
//
// ALIASING: src and dst are SEPARATE buffers (orchestrator uses a 1-layer
// scratch; do NOT alias). dst stride (fwht2, 68B@256) < src stride (fwht4,
// 132B@256), so even an in-place ascending pass would be safe, but we do not
// rely on it.
//
// Grid: [n_kv_heads, n_positions]. Block: [32]. LDS: (head_dim + 32) * 4 bytes
// (only the +32 scratch tail is used; sized to match the K-write template).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_transcode_k_fwht4_to_fwht2(
    unsigned char* __restrict__ dst,        // fwht2 layout (68 B/head/pos @256)
    const unsigned char* __restrict__ src,  // fwht4 layout (132 B/head/pos @256)
    int n_kv_heads, int head_dim, int n_positions
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int pos = blockIdx.y;
    if (pos >= n_positions) return;
    const int tid = threadIdx.x;

    const int n_halves = head_dim / 128;

    const int src_bph = 4 + head_dim / 2;   // fwht4: 132 @256
    const int src_bpp = n_kv_heads * src_bph;
    const int dst_bph = 4 + head_dim / 4;   // fwht2: 68 @256
    const int dst_bpp = n_kv_heads * dst_bph;

    const unsigned char* src_rec = src + (size_t)pos * src_bpp + (size_t)h * src_bph;
    unsigned char* out = dst + (size_t)pos * dst_bpp + (size_t)h * dst_bph;

    extern __shared__ float smem[];
    float* scratch = smem + head_dim;

    // Read the fwht4 cnorm + this thread's nibble indices for every half into
    // registers BEFORE any write (overlap-safe even if aliased).
    float cnorm4 = *(const float*)src_rec;

    // Reconstruct the rotated values for this thread's 4*n_halves dims.
    // rv[half*4 + i] = cnorm4 * TURBO_C4[idx]. (4 dims per half.)
    float rv[8];  // max n_halves=2 → 8 dims/thread
    for (int half = 0; half < n_halves; half++) {
        // fwht4: 2 bytes/thread/half at out[4 + half*64 + tid*2].
        const unsigned char* b = src_rec + 4 + half * 64 + tid * 2;
        int i0 = (int)(b[0] & 0xF);
        int i1 = (int)((b[0] >> 4) & 0xF);
        int i2 = (int)(b[1] & 0xF);
        int i3 = (int)((b[1] >> 4) & 0xF);
        int base = half * 4;
        rv[base + 0] = cnorm4 * TURBO_C4[i0];
        rv[base + 1] = cnorm4 * TURBO_C4[i1];
        rv[base + 2] = cnorm4 * TURBO_C4[i2];
        rv[base + 3] = cnorm4 * TURBO_C4[i3];
    }

    // orig_norm = ||reconstructed rotated vector|| over the WHOLE head
    // (both halves, all 32 threads) — one reduce, matching the writers.
    float local_osq = 0.0f;
    for (int i = 0; i < n_halves * 4; i++) local_osq += rv[i] * rv[i];
    scratch[tid] = local_osq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);

    // Re-quantize the UNIT-normalized reconstruction at 2-bit (matches the
    // write kernels, which normalize before quantizing).
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    int didx[8];
    float local_rsq = 0.0f;
    for (int i = 0; i < n_halves * 4; i++) {
        didx[i] = turbo_quantize_2bit(rv[i] * inv_norm);
        float c = TURBO_C2[didx[i]];
        local_rsq += c * c;
    }

    __syncthreads();
    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm2 = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm2;

    // Pack fwht2: 4 values × 2 bits = 1 byte per thread per half, at the
    // EXACT slot kv_cache_write_asym_k_fwht2 uses (out[4 + half*32 + tid]).
    for (int half = 0; half < n_halves; half++) {
        int base = half * 4;
        int write_off = 4 + half * 32 + tid;
        out[write_off] = (unsigned char)(
            (didx[base + 0] & 0x3) |
            ((didx[base + 1] & 0x3) << 2) |
            ((didx[base + 2] & 0x3) << 4) |
            ((didx[base + 3] & 0x3) << 6)
        );
    }
}
