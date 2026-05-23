// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ═══ HFQ4-G256 × Q8_1 dp4a MMQ 2-way fused — gfx906 ═══
//
// Sibling to gemm_qkv_hfq4g256_mmq_gfx906_body.cuh. Same LDS-tiled
// Option C+pad streaming X reuse + sdot4 inner loop; differs in:
//   - 2-way output routing per workgroup (out_a / out_b band selection)
//   - Reused for THREE dispatch sites on gfx906:
//       * FFN gate_up:        out_a=gate,  out_b=up   (32/32 layers)
//       * LA QKVZA-head:      out_a=qkv,   out_b=z    (24/32 layers)
//       * (potentially more) — the kernel is shape-agnostic, just
//         needs both Ms multiples of MMQ_Y=128.
//
// Why fuse for gfx906?  Today the dispatcher emits 2 separate
// `gemm_hfq4g256_mmq_set_gfx906` launches per gate_up call (and per
// LA QKVZA-head call). Fusing 2-way halves launch overhead and lets
// L2 hits on the Q8_1 batch tile `Xq` amortize across the two outputs.
// Fires far more often than the 3-way QKV: every layer of gate_up
// + every LA layer of QKVZA-head = 32 + 24 = 56 invocations/forward
// on Qwen3.5 9B, vs the 3-way's 8 (FA only). This is the kernel
// that should drive the bulk of the gfx906 fusion win.
//
// Probe attribution (2026-05-23, B=256, 9B MQ4):
//   mmq_set_gfx906 = 53.6% of all prefill wall, 4.25 calls/layer.
// After Phase 2 (qkv 3-way) shipped, 8 of 136 mmq_set calls were
// absorbed into 8 qkv-fused calls. The remaining 112 mmq_set calls
// are: 32 layers × 2 gate_up + 24 LA × 2 qkvza-head = 64 + 48 = 112.
// Phase 3 collapses both groups: gate_up 32 → 32 fused, QKVZA-head
// 48 → 24 fused; net mmq_set → 0, total 2-way fused calls = 32 + 24
// = 56.
//
// ═══ KERNEL INVARIANTS — caller MUST satisfy ═══
//
//   1. K must be a multiple of 256 (= HFQ4 group_size).
//
//   2. Both M values (m_a, m_b) must be multiples of MMQ_Y=128.
//      Qwen3.5 satisfies:
//        gate_up:    9B gate_m=up_m=4864, 4B 2560, 27B 6976
//        qkvza-head: 9B qkv_m=2304 z_m=2304, 4B 1152/1152, 27B 3328/3328
//      (all multiples of 128 ✓)
//
//   3. N (batch_size) alignment depends on the kernel variant:
//        `_x{N}`        → ANY N ≥ 1 is safe.
//        `_full_*_x{N}` → CALLER MUST GUARANTEE batch_size % mmq_x == 0.
//
//   4. Outputs are 2 SEPARATE column-major tensors Y_a[col][row_in_a],
//      Y_b[col][row_in_b]. Matches the existing gate_up_wave64_dp4a
//      ABI on gfx906.

// MMQ_Y is the row-tile height per workgroup. Default 128 (the
// established sweet spot from the residual sibling); a Y=64 variant
// exists to probe the higher-occupancy regime (smaller LDS X-tile,
// halved accumulator register footprint, 2× more WGs per grid →
// better utilization on gfx906's 60 CUs at modest grid sizes).
//
// IMPORTANT INVARIANT: MMQ_Y must be a multiple of 16. The loader
// distributes (MMQ_Y * 16) tasks across 256 threads as
// `loops_per_thread = MMQ_Y / 16`. Y values that aren't multiples of
// 16 would either over-fetch (OOB LDS writes, same shape as PR #315
// commit 4e5fefc7's RDNA2 NaN bug) or leave threads idle.
#ifndef MMQ_Y
#define MMQ_Y 128
#endif
static_assert(MMQ_Y % 16 == 0, "MMQ_Y must be a multiple of 16 (loader invariant)");
static_assert(MMQ_Y == 64 || MMQ_Y == 128, "Only Y=64 and Y=128 are currently supported (Y=32/96 untested)");

#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32
#define QK8_1 32
#define QI8_1 8

// X_STRIDE selection mirrors the gfx906 residual sibling. See
// gemm_hfq4g256_residual_mmq_gfx906_body.cuh:82-100 for rationale.
template <int mmq_x>
constexpr int x_stride_for() { return mmq_x >= 32 ? 40 : 33; }

#define Y_STRIDE 36

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// ─── Tile loaders (verbatim from qkv sibling) ─────────────────────────────

template <int x_stride>
static __device__ __forceinline__ void load_hfq4_tile_streaming(
    const char* __restrict__ A,
    int* __restrict__ x_qs,
    float2* __restrict__ x_dm,
    int row0, int kg, int window, int M, int groups_per_row
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;

    if (window == 0) {
        if (tid < MMQ_Y) {
            const int i = tid;
            const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            x_dm[i] = make_float2(sc, zp + 8.0f * sc);
        }
    }

    // Loader fans MMQ_Y rows × 16 chunks-per-row across 256 threads.
    // loops_per_thread = MMQ_Y * 16 / 256 = MMQ_Y / 16.
    // Y=128 → 8 loops, Y=64 → 4 loops. Each thread handles consecutive
    // task_ids so the stride `tid * loops_per_thread` keeps the per-row
    // chunk distribution unchanged from the Y=128 baseline.
    constexpr int loops_per_thread = MMQ_Y / 16;
    #pragma unroll
    for (int loop = 0; loop < loops_per_thread; ++loop) {
        const int task_id = tid * loops_per_thread + loop;
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
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;
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

// ─── dp4a compute (verbatim from qkv sibling) ─────────────────────────────

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

// ─── Write-back: SET-mode only (overwrite). ────────────────────────────

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
// Workgroup geometry (gfx906 wave64-native):
//   blockDim = (64, 4, 1) = 256 threads = 4 wave64s
//   gridDim  = (
//       ceil((m_a + m_b) / MMQ_Y),    // total_row tiles, 2-way routed
//       ceil(N / mmq_x),              // batch column tiles
//       1
//   )
//
// Per-WG row-band routing:
//   total_row = blockIdx.x * MMQ_Y
//   if total_row < m_a : A=A_a, Y=Y_a, out_m=m_a
//   else               : A=A_b, Y=Y_b, out_m=m_b, row0 -= m_a
//
// The naming uses generic `a`/`b` so the same kernel serves both
// gate_up (a=gate, b=up) and qkvza-head (a=qkv, b=z) dispatch sites.

template <int mmq_x, bool need_check>
static __device__ __forceinline__ void gate_up_mmq_body_templated(
    const char* __restrict__ A_a,
    const char* __restrict__ A_b,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_a,
    float* __restrict__ Y_b,
    int m_a, int m_b,
    int K, int N
) {
    const int total_row = blockIdx.x * MMQ_Y;
    const int total_m = m_a + m_b;
    if (need_check && total_row >= total_m) return;

    const char* A;
    float* Y;
    int row0;
    int out_m;
    if (total_row < m_a) {
        A = A_a; Y = Y_a; row0 = total_row;       out_m = m_a;
    } else {
        A = A_b; Y = Y_b; row0 = total_row - m_a; out_m = m_b;
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
