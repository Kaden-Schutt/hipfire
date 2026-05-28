// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 indexer score — WMMA-accelerated BATCHED variant.
// Phase C1 of the deepseek4 prefill catch-up plan (vs antirez/ds4).
//
// Replaces `indexer_relu_score_batched_f32` (one F32 scalar thread per
// head, ~64 redundant K-row reads per (b, n) cell) with a WMMA tile
// formulation:
//
//   scores[b, n] = sum_h relu(Q[b, h, :] · K[n, :]) * weights[b, h]
//
// Each block handles 16 consecutive compressed slots (block_n) of one
// batch position (b). The 64 heads are split across 4 warps; each warp
// computes one [16 heads × 16 slots] WMMA tile of Q·K^T (D=128 → 8
// 16-step iters per tile). After the K-loop, per-element ReLU + weights
// multiply runs on the tile registers, results land in LDS, and a 16-
// thread reduction sums across heads for each of the 16 slots.
//
// F32 → F16 conversion happens inline before WMMA frag load (matches
// antirez/ds4 ds4_cuda.cu line 4609 approach). Keeps Q and K_cache in
// F32 on the engine side — no extra storage.
//
// Grid:  [batch_size, ceil(N_max / 16), 1]
// Block: [128, 1, 1]            = 4 warps × 32 threads
// LDS:   64 × 16 × 4 = 4096 B   (per-(head, slot_in_tile) partial)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define WMMA_TILE 16
#define HEADS_PER_WARP 16
#define WARPS_PER_BLOCK 4
#define MAX_HEADS (HEADS_PER_WARP * WARPS_PER_BLOCK) // 64

__launch_bounds__(128, 2)
extern "C" __global__ void indexer_relu_score_wmma_batched_f32(
    const float* __restrict__ q,         // [B, H, D]
    const float* __restrict__ k_cache,   // [N_max, D] shared across batch
    const float* __restrict__ weights,   // [B, H]
    const int*   __restrict__ n_per_batch, // [B] per-batch valid prefix
    float*       __restrict__ scores,    // [B, N_max]
    int H,
    int D,
    int N_max,
    int batch_size
) {
    const int b       = blockIdx.x;
    const int n_block = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int lane16  = tid & 15; // wave32 WMMA: lanes 0..15 and 16..31 duplicate

    if (b >= batch_size) return;
    const int n_start = n_block * WMMA_TILE;
    if (n_start >= N_max) return;

    const int n_valid_b = n_per_batch[b];

    // Fast path: whole tile is past the causal cutoff — write -inf and return.
    if (n_start >= n_valid_b) {
        if (tid < WMMA_TILE) {
            const int slot = n_start + tid;
            if (slot < N_max) {
                scores[(size_t)b * N_max + slot] = -1e30f;
            }
        }
        return;
    }

    // Per-warp head range and per-thread (head, slot) for fragment loads.
    const int h_start = warp_id * HEADS_PER_WARP;
    const int my_h    = h_start + lane16;
    const int my_n    = n_start + lane16;

    // Clamp to valid range so the dummy reads don't go out of bounds. We
    // mask the results post-hoc (masked lanes contribute 0 after ReLU+w
    // gating below).
    const int safe_h = (my_h < H) ? my_h : (H - 1);
    const int safe_n = (my_n < n_valid_b) ? my_n : (n_valid_b - 1);

    const float* qh = q + ((size_t)b * H + safe_h) * D;
    const float* kn = k_cache + (size_t)safe_n * D;

    // WMMA accumulator — 8 floats per thread, one [16h × 16n] tile per warp.
    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // K-loop. 16-elements per step; inline F32 → F16 conversion before
    // WMMA frag load. D=128 → 8 iters.
    for (int k = 0; k < D; k += WMMA_TILE) {
        half16_t q_frag;
        half16_t k_frag;
        #pragma unroll
        for (int i = 0; i < WMMA_TILE; ++i) {
            q_frag[i] = (_Float16)qh[k + i];
            k_frag[i] = (_Float16)kn[k + i];
        }
        // C[h, n] += Q[h, k..k+15] · K[n, k..k+15]   (Q·K^T tile)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag, k_frag, acc);
    }

    // Tile result layout (RDNA3 wave32 WMMA): acc[j] = C[2*j + (tid >> 4)][lane16].
    // → tile_row (=head_in_tile) = 2*j + (lane >> 4 within warp), tile_col (=slot_in_tile) = lane16.
    // Translate to absolute (h, slot).

    __shared__ float lds[MAX_HEADS][WMMA_TILE];

    // Initialize the warp's slice of LDS to zero so masked-out lanes
    // don't contribute. Only 16 of 32 lanes per warp are unique writers;
    // the other 16 are duplicates per WMMA spec.
    if (lane < WMMA_TILE) {
        #pragma unroll
        for (int hh = 0; hh < HEADS_PER_WARP; ++hh) {
            lds[h_start + hh][lane] = 0.0f;
        }
    }
    __syncthreads();

    // Per-element ReLU + weights multiply, then deposit into LDS.
    // Per RDNA3 wave32 WMMA spec, acc[j] on lane L holds C[2*j+(tid>>4)][lane16].
    // → lanes 0..15 carry the even tile rows (0,2,4,...,14);
    //   lanes 16..31 carry the odd  tile rows (1,3,5,...,15).
    // ALL 32 lanes per warp must deposit (each owns 8 unique rows) — gating
    // on `lane16 < WMMA_TILE` is redundant since lane16 ∈ [0,16) always.
    const int slot_col = my_n; // tile_col = lane16; my_n = n_start + lane16
    if (slot_col < n_valid_b) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int tile_row = 2 * j + (tid >> 4); // 0..15 within tile
            const int head     = h_start + tile_row;
            if (head < H) {
                const float qk = acc[j];
                const float w  = weights[(size_t)b * H + head];
                const float v  = (qk > 0.0f ? qk : 0.0f) * w;
                lds[head][lane16] = v;
            }
        }
    }
    __syncthreads();

    // Per-slot reduction across H heads. 16 threads each sum one column
    // of lds[H][16] in fixed order → deterministic.
    if (tid < WMMA_TILE) {
        const int slot = n_start + tid;
        if (slot < n_valid_b) {
            float sum = 0.0f;
            #pragma unroll
            for (int h = 0; h < MAX_HEADS; ++h) {
                if (h < H) sum += lds[h][tid];
            }
            scores[(size_t)b * N_max + slot] = sum;
        } else if (slot < N_max) {
            scores[(size_t)b * N_max + slot] = -1e30f;
        }
    }
}
