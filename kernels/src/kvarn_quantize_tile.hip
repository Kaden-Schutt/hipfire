// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// KVarN tile quantizer (Phase D1 component #2 core): variance-normalize
// (log-domain Sinkhorn) + per-row 4-bit affine quantize + pack to the on-device
// KVarN record. One BLOCK per [r_dim × c_dim] tile. The tile stays in GLOBAL
// memory (re-read each Sinkhorn pass); only the small per-row/per-col scale
// arrays live in LDS — this avoids the gfx1103 large-LDS hang hazard while still
// being a register/wave-friendly reduction. Mirrors kvarn.rs::quantize_tile +
// pack_kvarn_tile (the CPU golden oracle); GPU uses f32 where CPU uses f64, so
// validate by dequant cos-sim, not bit-exact.
//
// Record layout (== pack_kvarn_tile_bits), `bits` per code, 8/bits codes/byte:
//   [ (r*c)/(8/bits) ]  packed q codes (LSB-first, row-major)
//   [ r_dim*2  ]   scale_abs[r] fp16
//   [ r_dim*2  ]   zp_abs[r]    fp16
//   [ c_dim*2  ]   s_col[c]     fp16
// Dequant: (q*scale_abs[r] + zp_abs[r]) * s_col[c].
//
// `bits` in {2,4,8}: codes/byte = 8/bits; qmax = 2^bits - 1. bits=4 is the
// legacy nibble layout. Constraints: c_dim*bits % 8 == 0 (each row owns whole
// bytes → the one-thread-per-row pack is race-free); r_dim,c_dim <= 256.
// Grid:[n_tiles], block:[256]. SINKHORN_ITERS matches kvarn.rs.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define KVARN_SINKHORN_ITERS 10
#define KVARN_RMAX 256

__device__ __forceinline__ float kvarn_log_clamp(float x) {
    return fminf(fmaxf(x, -0.3f), 10.0f);
}

// One block per tile. Tile pointer = T + tile_idx * (r_dim*c_dim).
__launch_bounds__(256)
extern "C" __global__ void kvarn_quantize_tile(
    const float*  __restrict__ T,      // [n_tiles, r_dim*c_dim] f32 row-major tiles
    uint8_t*      __restrict__ REC,     // [n_tiles, record_bytes]
    int n_tiles, int r_dim, int c_dim, int record_bytes,
    int bits                            // per-code bits in {2,4,8}; 8/bits codes/byte
) {
    const float qmax = (float)((1 << bits) - 1);
    const int cpb = 8 / bits;           // codes per byte
    const int codemask = (1 << bits) - 1;
    const int t = blockIdx.x;
    if (t >= n_tiles) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const float* tile = T + (long long)t * r_dim * c_dim;
    uint8_t* rec = REC + (long long)t * record_bytes;

    __shared__ float lc[KVARN_RMAX];      // log_s_col[c]
    __shared__ float lr[KVARN_RMAX];      // log_s_row[r]
    __shared__ float blc[KVARN_RMAX];     // best log_s_col
    __shared__ float blr[KVARN_RMAX];     // best log_s_row
    __shared__ float red[256];            // cross-thread reduction scratch
    __shared__ float best_imb_s;

    for (int c = tid; c < c_dim; c += nth) { lc[c] = 0.0f; blc[c] = 0.0f; }
    for (int r = tid; r < r_dim; r += nth) { lr[r] = 0.0f; blr[r] = 0.0f; }
    if (tid == 0) best_imb_s = 3.4e38f;
    __syncthreads();

    for (int it = 0; it < KVARN_SINKHORN_ITERS; ++it) {
        // ── Column pass: one thread per column, std down rows of cur[r,c].
        for (int c = tid; c < c_dim; c += nth) {
            float sum = 0.f, sq = 0.f;
            float ec = __expf(-lc[c]);
            for (int r = 0; r < r_dim; ++r) {
                float v = tile[r * c_dim + c] * __expf(-lr[r]) * ec;
                sum += v; sq += v * v;
            }
            float n = (float)r_dim;
            float var = fmaxf(sq / n - (sum / n) * (sum / n), 0.f);
            float sd = fminf(fmaxf(sqrtf(var), 1e-3f), 1e3f);
            lc[c] = kvarn_log_clamp(lc[c] + __logf(sd));
        }
        __syncthreads();
        // ── Row pass: one thread per row, std across columns of cur[r,c].
        for (int r = tid; r < r_dim; r += nth) {
            float sum = 0.f, sq = 0.f;
            float er = __expf(-lr[r]);
            for (int c = 0; c < c_dim; ++c) {
                float v = tile[r * c_dim + c] * er * __expf(-lc[c]);
                sum += v; sq += v * v;
            }
            float n = (float)c_dim;
            float var = fmaxf(sq / n - (sum / n) * (sum / n), 0.f);
            float sd = fminf(fmaxf(sqrtf(var), 1e-3f), 1e3f);
            lr[r] = kvarn_log_clamp(lr[r] + __logf(sd));
        }
        __syncthreads();
        // ── Imbalance of candidate = col_max/col_min + row_max/row_min.
        // Col stds → reduce min/max.
        float cmin = 3.4e38f, cmax = 0.f;
        for (int c = tid; c < c_dim; c += nth) {
            float sum = 0.f, sq = 0.f; float ec = __expf(-lc[c]);
            for (int r = 0; r < r_dim; ++r) {
                float v = tile[r * c_dim + c] * __expf(-lr[r]) * ec;
                sum += v; sq += v * v;
            }
            float n = (float)r_dim;
            float sd = sqrtf(fmaxf(sq / n - (sum / n) * (sum / n), 0.f));
            cmin = fminf(cmin, sd); cmax = fmaxf(cmax, sd);
        }
        red[tid] = cmin; __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fminf(red[tid], red[tid + s]); __syncthreads(); }
        float col_min = red[0]; __syncthreads();
        red[tid] = cmax; __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
        float col_max = red[0]; __syncthreads();
        float rmin = 3.4e38f, rmax = 0.f;
        for (int r = tid; r < r_dim; r += nth) {
            float sum = 0.f, sq = 0.f; float er = __expf(-lr[r]);
            for (int c = 0; c < c_dim; ++c) {
                float v = tile[r * c_dim + c] * er * __expf(-lc[c]);
                sum += v; sq += v * v;
            }
            float n = (float)c_dim;
            float sd = sqrtf(fmaxf(sq / n - (sum / n) * (sum / n), 0.f));
            rmin = fminf(rmin, sd); rmax = fmaxf(rmax, sd);
        }
        red[tid] = rmin; __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fminf(red[tid], red[tid + s]); __syncthreads(); }
        float row_min = red[0]; __syncthreads();
        red[tid] = rmax; __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
        float row_max = red[0]; __syncthreads();
        float imb = col_max / fmaxf(col_min, 1e-8f) + row_max / fmaxf(row_min, 1e-8f);
        if (imb < best_imb_s) {
            if (tid == 0) best_imb_s = imb;
            for (int c = tid; c < c_dim; c += nth) blc[c] = lc[c];
            for (int r = tid; r < r_dim; r += nth) blr[r] = lr[r];
        }
        __syncthreads();
    }

    // ── Quantize using best scales: per-row `bits`-bit min/max of balanced tile.
    const int n = r_dim * c_dim;
    const int qbytes = (n + cpb - 1) / cpb;
    const int off_scale = qbytes;
    const int off_zp = off_scale + r_dim * 2;
    const int off_scol = off_zp + r_dim * 2;
    for (int r = tid; r < r_dim; r += nth) {
        float er = __expf(-blr[r]);
        float lo = 3.4e38f, hi = -3.4e38f;
        for (int c = 0; c < c_dim; ++c) {
            float v = tile[r * c_dim + c] * er * __expf(-blc[c]);
            lo = fminf(lo, v); hi = fmaxf(hi, v);
        }
        float scale = fmaxf((hi - lo) / qmax, 1e-8f);
        float s_row = __expf(blr[r]);
        // q codes for this row: c_dim*bits % 8 == 0 → each row owns whole bytes,
        // so this single-thread-per-row pack is race-free. LSB-first within byte.
        for (int c = 0; c < c_dim; ++c) {
            float v = tile[r * c_dim + c] * er * __expf(-blc[c]);
            int code = (int)(fminf(fmaxf(roundf((v - lo) / scale), 0.f), qmax));
            int gi = r * c_dim + c;
            int sub = gi % cpb;
            uint8_t shifted = (uint8_t)((code & codemask) << (sub * bits));
            if (sub == 0) rec[gi / cpb] = shifted;   // first code in byte clears it
            else rec[gi / cpb] |= shifted;
        }
        __half sa = __float2half(scale * s_row);
        __half za = __float2half(lo * s_row);
        *(__half*)(rec + off_scale + r * 2) = sa;
        *(__half*)(rec + off_zp + r * 2) = za;
    }
    for (int c = tid; c < c_dim; c += nth) {
        *(__half*)(rec + off_scol + c * 2) = __float2half(__expf(blc[c]));
    }
}
