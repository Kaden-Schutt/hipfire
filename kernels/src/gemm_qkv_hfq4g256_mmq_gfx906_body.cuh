// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ═══ HFQ4-G256 × Q8_1 dp4a MMQ 3-way fused (Q+K+V) — gfx906 ═══
//
// Sibling to gemm_hfq4g256_residual_mmq_gfx906_body.cuh. Same LDS-tiled
// Option C+pad streaming X reuse + sdot4 inner loop; differs in:
//   - 3-way output routing per workgroup (Q/K/V band selection at WG-
//     prologue, before any LDS staging)
//   - Overwrite write-back (Y[col][row] = acc, no residual add)
//   - Wave64 / 256-thread block (vs PR #315's wave32 / 128-thread gfx1030
//     sibling at kernels/src/gemm_qkv_hfq4g256_mmq_body.cuh)
//
// Why fuse for gfx906?  Today the dispatcher (dispatch.rs:6346-6355)
// emits 3 separate `gemm_hfq4g256_mmq_set_gfx906` launches per QKV call,
// each one independently loading the Q8_1 batch tile `Xq` from L2.
// `Xq` for B=256 / k=4096 is 1.18 MiB — comfortably in gfx906's ~4 MiB
// L2. Fusing 3-way interleaves the 3 outputs on the same CUs so each
// Xq line is loaded by exactly 1 launch instead of 3, halving L2 read
// pressure. Per-row weight reads (A_q/A_k/A_v) do not amortize — each
// row-band still owns exactly 1 weight tile, no savings there.
//
// Probe attribution (2026-05-23, /local/hipfire/qwen3.5-9b.mq4, B=256):
//   mmq_set_gfx906 = 53.6% of all prefill wall, 4.25 calls/layer.
// Floor for this kernel's contribution: 5% prefill-wide; ceiling 8-12%
// once gate_up_2way and qkvza-head_2way also land (since the 2-way
// pattern fires far more frequently than 3-way QKV on Qwen3.5 9B).
// See experiments/gfx906-fused-mmq/probe-results.md.
//
// ═══ KERNEL INVARIANTS — caller MUST satisfy ═══
//
//   1. K must be a multiple of 256 (= HFQ4 group_size). Body assumes
//      groups_per_row = K / 256; not asserted at runtime.
//
//   2. q_m, k_m, v_m must each be a multiple of MMQ_Y=128. Qwen3.5
//      family satisfies this (9B FullAttn: q_m=4096, k_m=v_m=1024;
//      4B FullAttn: q_m=2048, k_m=v_m=512). The bounds-checked
//      `_x{N}` variant tolerates ragged M but is only emitted as a
//      defense-in-depth fallback; `_full_*_x{N}` variants drop the
//      row-clamp for ~5% perf and require strict alignment.
//
//   3. N (batch_size) alignment depends on the kernel variant:
//        `_x{N}`        → ANY N ≥ 1 is safe (need_check=true skips
//                         OOB cols).
//        `_full_*_x{N}` → CALLER MUST GUARANTEE batch_size % mmq_x == 0.
//
//   4. Q/K/V outputs are written as 3 SEPARATE column-major tensors
//      Y_q[col][row_in_q], Y_k[col][row_in_k], Y_v[col][row_in_v] —
//      NOT a concatenated Y_qkv. This matches the existing
//      `gemm_qkv_hfq4g256_wave64_dp4a` ABI on gfx906 and the gfx1030
//      sibling at gemm_qkv_hfq4g256_mmq_body.cuh.

#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32
#define QK8_1 32
#define QI8_1 8

// X_STRIDE selection mirrors the gfx906 residual sibling exactly. See
// gemm_hfq4g256_residual_mmq_gfx906_body.cuh:82-100 for the b128 cliff
// rationale and PMC validation. Keep in lockstep with that file.
template <int mmq_x>
constexpr int x_stride_for() { return mmq_x >= 32 ? 40 : 33; }

#define Y_STRIDE 36

// LDS layout invariant — KEEP IN SYNC WITH dispatch.rs's qkv_mmq_gfx906 tile:
//   [x_qs:   i32    × MMQ_Y * x_stride       ]  = 128 * x_stride * 4
//   [x_dm:   float2 × MMQ_Y                  ]  = 128 * 8     =  1,024 B
//   [tile_y: i32    × mmq_x * Y_STRIDE       ]  = mmq_x * 144 B
// Total budget identical to the residual sibling (≤ 32 KiB/WG so 2 WG/CU).

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// ─── Tile loaders (verbatim from residual sibling) ────────────────────────

template <int x_stride>
static __device__ __forceinline__ void load_hfq4_tile_streaming(
    const char* __restrict__ A,
    int* __restrict__ x_qs,
    float2* __restrict__ x_dm,
    int row0, int kg, int window, int M, int groups_per_row
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255

    if (window == 0) {
        // tid < MMQ_Y guard — at MMQ_Y=128 (current default) this is
        // tid<128. Kept symbolic so the eventual MMQ_Y=64/96 sweep
        // (plan §4.2) cannot re-introduce PR #315 commit 4e5fefc7's
        // LDS-OOB bug on gfx906.
        if (tid < MMQ_Y) {
            const int i = tid;
            const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            x_dm[i] = make_float2(sc, zp + 8.0f * sc);
        }
    }

    #pragma unroll
    for (int loop = 0; loop < 8; ++loop) {
        const int task_id = tid * 8 + loop;
        const int i = task_id / 16;
        const int chunk = task_id % 16;

        const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kg) * 136;

        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + window * 64 + chunk * 4);

        const unsigned int n0 = (qs0      ) & 0xFu;
        const unsigned int n1 = (qs0 >>  4) & 0xFu;
        const unsigned int n2 = (qs0 >>  8) & 0xFu;
        const unsigned int n3 = (qs0 >> 12) & 0xFu;
        const unsigned int n4 = (qs0 >> 16) & 0xFu;
        const unsigned int n5 = (qs0 >> 20) & 0xFu;
        const unsigned int n6 = (qs0 >> 24) & 0xFu;
        const unsigned int n7 = (qs0 >> 28) & 0xFu;

        const int int_a = (int)(((n0 - 8) & 0xFF) | (((n1 - 8) & 0xFF) << 8)
                              | (((n2 - 8) & 0xFF) << 16) | (((n3 - 8) & 0xFF) << 24));
        const int int_b = (int)(((n4 - 8) & 0xFF) | (((n5 - 8) & 0xFF) << 8)
                              | (((n6 - 8) & 0xFF) << 16) | (((n7 - 8) & 0xFF) << 24));

        x_qs[i * x_stride + 2 * chunk + 0] = int_a;
        x_qs[i * x_stride + 2 * chunk + 1] = int_b;
    }
}

template <int mmq_x>
static __device__ __forceinline__ void load_q8_1_tile_coalesced(
    const block_q8_1_mmq* __restrict__ Xq,
    int* __restrict__ tile_y,
    int col0, int kb, int N
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255
    const int total_ints = mmq_x * Y_STRIDE;

    #pragma unroll
    for (int u = tid; u < total_ints; u += 256) {
        const int j = u / Y_STRIDE;
        const int slot = u % Y_STRIDE;
        const bool valid = (col0 + j) < N;
        const int col = valid ? (col0 + j) : (N - 1);

        const int* src = (const int*)(Xq + (long long)kb * N + col);
        tile_y[u] = valid ? src[slot] : 0;
    }
}

// ─── dp4a compute (verbatim from residual sibling) ────────────────────────

template <int mmq_x>
static __device__ __forceinline__ void vec_dot_dp4a_streaming(
    const int* __restrict__ x_qs,
    const float2* __restrict__ x_dm,
    const int* __restrict__ tile_y,
    float* __restrict__ sum,
    int sub_block
) {
    constexpr int vdr = 8;
    constexpr int x_stride = x_stride_for<mmq_x>();
    const int kx_start = sub_block * 8;
    const int ky_start = 4 + sub_block * 8;

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;

        const half2* y_ds_col = (const half2*)(tile_y + j * Y_STRIDE);
        const half2 ds_j = y_ds_col[sub_block];

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
            const int i = i0 + threadIdx.x;

            int sumi = 0;
            if constexpr (mmq_x >= 32) {
                const int4 x_v0 = *(const int4*)&x_qs[i * x_stride + kx_start + 0];
                const int4 x_v1 = *(const int4*)&x_qs[i * x_stride + kx_start + 4];
                const int4 y_v0 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 0];
                const int4 y_v1 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 4];
                sumi = __builtin_amdgcn_sdot4(x_v0.x, y_v0.x, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v0.y, y_v0.y, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v0.z, y_v0.z, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v0.w, y_v0.w, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.x, y_v1.x, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.y, y_v1.y, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.z, y_v1.z, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.w, y_v1.w, sumi, false);
            } else {
                #pragma unroll
                for (int v = 0; v < vdr; ++v) {
                    const int x_int = x_qs[i * x_stride + kx_start + v];
                    const int y_int = tile_y[j * Y_STRIDE + ky_start + v];
                    sumi = __builtin_amdgcn_sdot4(x_int, y_int, sumi, false);
                }
            }

            const float2 dm_i = x_dm[i];
            const float2 dsf = __half22float2(ds_j);
            const float scale_w = dm_i.x;
            const float zp_eff  = dm_i.y;
            const float d_x     = dsf.x;
            const float sum_x   = dsf.y;

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            sum[idx] += scale_w * d_x * (float)sumi + zp_eff * sum_x;
        }
    }
}

// ─── Write-back: SET-mode (overwrite). No `add` variant for qkv-fused. ────

template <int mmq_x, bool need_check>
static __device__ __forceinline__ void write_back_set_templated(
    float* __restrict__ Y,
    const float* __restrict__ sum,
    int row0, int col0, int M, int N
) {
    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col = col0 + j;
        if (need_check && col >= N) continue;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row = row0 + i;
            if (need_check && row >= M) continue;

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            const long long out_idx = (long long)col * M + row;
            Y[out_idx] = sum[idx];
        }
    }
}

// ─── Shared kernel body ───────────────────────────────────────────────────
//
// Workgroup geometry (gfx906 wave64-native, same as residual sibling):
//   blockDim = (64, 4, 1) = 256 threads = 4 wave64s
//   gridDim  = (
//       ceil((q_m + k_m + v_m) / MMQ_Y),    // total_row tiles, 3-way routed
//       ceil(N / mmq_x),                    // batch column tiles
//       1
//   )
//
// Per-WG row-band routing (mirrors gfx1030 sibling lines 53-67):
//   total_row = blockIdx.x * MMQ_Y
//   if      total_row < q_m            : A=A_q, Y=Y_q, out_m=q_m
//   else if total_row < q_m+k_m        : A=A_k, Y=Y_k, out_m=k_m,  row0 -= q_m
//   else                               : A=A_v, Y=Y_v, out_m=v_m,  row0 -= q_m+k_m
//
// One LDS staging path (x_qs / x_dm) per WG — only the weight for the
// selected output is loaded. The Q8_1 batch tile (tile_y) is identical
// across all 3 outputs, so 3 WGs targeting Q+K+V at the same col-band
// will all read the same Xq lines — L2 hits across them are the
// fusion win on gfx906 (rather than X-reload elimination).

template <int mmq_x, bool need_check>
static __device__ __forceinline__ void qkv_mmq_body_templated(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_row = blockIdx.x * MMQ_Y;
    const int total_m = q_m + k_m + v_m;
    if (need_check && total_row >= total_m) return;

    // 3-way row-band routing. Note: `total_m` upper-bound check above
    // already gates the v branch; the else clause is reachable only
    // when total_row >= q_m + k_m && total_row < total_m.
    const char* A;
    float* Y;
    int row0;
    int out_m;
    if (total_row < q_m) {
        A = A_q; Y = Y_q; row0 = total_row;            out_m = q_m;
    } else if (total_row < q_m + k_m) {
        A = A_k; Y = Y_k; row0 = total_row - q_m;       out_m = k_m;
    } else {
        A = A_v; Y = Y_v; row0 = total_row - q_m - k_m; out_m = v_m;
    }

    const int col0 = blockIdx.y * mmq_x;
    if (need_check && col0 >= N) return;

    const int groups_per_row = K / 256;
    constexpr int x_stride = x_stride_for<mmq_x>();

    extern __shared__ int smem[];
    int*    x_qs   = smem;
    float2* x_dm   = (float2*)(x_qs + MMQ_Y * x_stride);
    int*    tile_y = (int*)(x_dm + MMQ_Y);

    float sum[(mmq_x / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        // Option C: 2 windows × 4 sub-blocks each, 4 syncs/group total.
        for (int window = 0; window < 2; ++window) {
            load_q8_1_tile_coalesced<mmq_x>(Xq, tile_y, col0, 2*kg + window, N);
            load_hfq4_tile_streaming<x_stride>(A, x_qs, x_dm, row0, kg, window, out_m, groups_per_row);
            __syncthreads();
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                vec_dot_dp4a_streaming<mmq_x>(x_qs, x_dm, tile_y, sum, sub);
            }
            __syncthreads();
        }
    }

    write_back_set_templated<mmq_x, need_check>(Y, sum, row0, col0, out_m, N);
}
