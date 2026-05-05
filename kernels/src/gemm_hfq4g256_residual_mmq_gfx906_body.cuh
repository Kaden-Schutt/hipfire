#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ═══ HFQ4-G256 × Q8_1 dp4a MMQ kernel body for gfx906 ═══
//
// Topology (wave-native gfx906):
//   block dim (64, 4, 1) = 256 threads = 4 wave64s
//   mmq_y = 128
//   mmq_x = templated {8, 16, 24, 32, 40, 48, 56, 64}
//
// Option C+pad (Window Streaming, supersedes Option B):
//   HFQ4 group = 256 K-elements = 2 Q8_1 blocks = 2 windows.
//   Each window covers 128 K-elements (= one Q8_1 block, = 4 sub-blocks
//   of 32 K-elements each).
//
// Per-window pipeline (= 4 syncs/group total, vs Option B's 16):
//   1. Load 128-K of x_qs (32 data ints + 1 pad int per row) + Q8_1
//      block of tile_y
//   2. __syncthreads
//   3. 4 sub-blocks computed back-to-back (no syncs between)
//   4. __syncthreads
//
// LDS layout: per-mmq_x X_STRIDE (see x_stride_for<>() below).
// Trade-off between ds_read_b128 alignment and bank-conflict pattern;
// the optimum stride differs at small vs large mmq_x. PMC validation
// is in docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md
// §5.

#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32 // 32 K-elements per streaming sub-iter
#define QK8_1 32
#define QI8_1 8

// X_STRIDE chosen per-mmq_x to balance two LDS effects:
//   - mmq_x >= 64: stride 40 (32 data ints + 8 pad). 40 × 4 = 160 B
//     is 16-B aligned every row → 100% ds_read_b128. 40 % 32 = 8 →
//     4-way bank conflict, but the b128 issue rate dominates.
//   - mmq_x < 64: stride 33 (32 data ints + 1 pad). 33 × 4 = 132 B
//     is 16-B aligned every 4th row only. 33 % 32 = 1 → 0-way bank
//     conflict. Smaller-mmq_x kernels (used at small batch sizes,
//     e.g. _full_add_x16 in attn-out residual) regressed under
//     stride-40 due to the bank conflict cost outweighing the b128
//     win when j0 has few iterations. PMC-validated.
template <int mmq_x>
constexpr int x_stride_for() { return mmq_x >= 64 ? 40 : 33; }

#define Y_STRIDE 36

// LDS layout invariant — KEEP IN SYNC WITH dispatch.rs:
//   [x_qs:   i32    × MMQ_Y * x_stride       ]  = 128 * x_stride * 4
//   [x_dm:   float2 × MMQ_Y                  ]  = 128 * 8     =  1,024 B
//   [tile_y: i32    × mmq_x * Y_STRIDE       ]  = mmq_x * 144 B
// At mmq_x=64 (stride 40):  20,480 + 1,024 + 9,216 = 30,720 B per WG.
// At mmq_x=8  (stride 33):  16,896 + 1,024 + 1,152 = 19,072 B per WG.
// Budget: ≤ 32 KiB/WG so 2 WGs/CU fit in 64 KiB cap. Verified ✅.

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// ─── Tile loaders ─────────────────────────────────────────────────────────

// Load 128 K-elements of X (unpacked) for one window.
// window ∈ [0, 1]. Each window loads 64 bytes of nibbles per row
// = 128 K-elements = 32 ints in x_qs (4 sub-blocks × 8 ints each).
template <int x_stride>
static __device__ __forceinline__ void load_hfq4_tile_streaming(
    const char* __restrict__ A,
    int* __restrict__ x_qs,
    float2* __restrict__ x_dm,
    int row0, int kg, int window, int M, int groups_per_row
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255

    if (window == 0) {
        if (tid < 128) {
            const int i = tid;
            const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            x_dm[i] = make_float2(sc, zp + 8.0f * sc);
        }
    }

    // X-qs window load: 128 rows × 16 uints/row = 2048 uint reads,
    // distributed across 256 threads → 8 uints/thread. Chunk-major layout.
    #pragma unroll
    for (int loop = 0; loop < 8; ++loop) {
        const int task_id = tid * 8 + loop; // 0..2047
        const int i = task_id / 16;         // row, 0..127
        const int chunk = task_id % 16;     // which uint in row, 0..15

        const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kg) * 136;

        // Offset: 8 (header) + window * 64 (block A vs B) + chunk * 4.
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + window * 64 + chunk * 4);
        
        // Decompose 8 nibbles into 2 ints.
        const unsigned int n0 = (qs0 >>  0) & 0xFu;
        const unsigned int n1 = (qs0 >>  4) & 0xFu;
        const unsigned int n2 = (qs0 >>  8) & 0xFu;
        const unsigned int n3 = (qs0 >> 12) & 0xFu;
        const unsigned int n4 = (qs0 >> 16) & 0xFu;
        const unsigned int n5 = (qs0 >> 20) & 0xFu;
        const unsigned int n6 = (qs0 >> 24) & 0xFu;
        const unsigned int n7 = (qs0 >> 28) & 0xFu;

        const int int_a = (int)(((n0 - 8) & 0xFF) | (((n1 - 8) & 0xFF) << 8) | (((n2 - 8) & 0xFF) << 16) | (((n3 - 8) & 0xFF) << 24));
        const int int_b = (int)(((n4 - 8) & 0xFF) | (((n5 - 8) & 0xFF) << 8) | (((n6 - 8) & 0xFF) << 16) | (((n7 - 8) & 0xFF) << 24));

        x_qs[i * x_stride + 2 * chunk + 0] = int_a;
        x_qs[i * x_stride + 2 * chunk + 1] = int_b;
    }
}

// Load one Q8_1 block (128 elements) for all mmq_x columns.
template <int mmq_x>
static __device__ __forceinline__ void load_q8_1_tile_coalesced(
    const block_q8_1_mmq* __restrict__ Xq,
    int* __restrict__ tile_y,
    int col0, int kb, int N
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255
    const int total_ints = mmq_x * Y_STRIDE;
    
    // Each thread loads total_ints / 256 ints.
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

// ─── dp4a compute ─────────────────────────────────────────────────────────

// Compute one 32-K sub-block's contribution into `sum`.
//
// Invariants (caller-enforced):
//   - x_qs: holds the current 128-K window (4 sub-blocks resident,
//     8 ints per sub-block per row). Reads stripe at
//     `kx_start = sub_block * 8` within each row of stride x_stride.
//   - tile_y: holds ONE Q8_1 block (128 K-elements = 4 sub-blocks).
//     Loaded by the caller once per window. `sub_block` (0..3) selects
//     which 32-K sub-block to consume from both x_qs and tile_y.
//
// tile_y layout (Y_STRIDE=36 ints/col):
//   slots 0..3 : 4 half2 ds values (one per 32-K sub-block)
//   slots 4..35: 32 int8 qs (= 8 ints/sub-block × 4 sub-blocks)
template <int mmq_x>
static __device__ __forceinline__ void vec_dot_dp4a_streaming(
    const int* __restrict__ x_qs,
    const float2* __restrict__ x_dm,
    const int* __restrict__ tile_y,
    float* __restrict__ sum,
    int sub_block
) {
    constexpr int vdr = 8; // 8 ints cover 32 K-elements.
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
            if constexpr (mmq_x >= 64) {
                // b128 path: issue 8 ints per operand as 2× int4 (b128)
                // reads. With X_STRIDE=40 (160 B/row, 16-B aligned every
                // row) the compiler emits 100% ds_read_b128 — no b32-quad
                // fallback. Threshold mmq_x≥64 empirically determined
                // (see plan v2.16): at smaller mmq_x the int4-unpack
                // overhead exceeds the issue-rate win.
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
                // Scalar (b32) path for small mmq_x where b128 unpack
                // overhead exceeds the issue-rate win. Empirically:
                // mmq_x=8/16/24 regress under b128, mmq_x≥32 wins.
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

// ─── Write-back ───────────────────────────────────────────────────────────

template <int mmq_x, bool need_check>
static __device__ __forceinline__ void write_back_residual_templated(
    float* __restrict__ Y,
    const float* __restrict__ sum,
    int row0, int col0, int M, int N, int add
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
            if (add) {
                Y[out_idx] += sum[idx];
            } else {
                Y[out_idx]  = sum[idx];
            }
        }
    }
}

// ─── Shared kernel body ───────────────────────────────────────────────────

template <int mmq_x, bool need_check, int add_mode>
static __device__ __forceinline__ void mmq_body_templated(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add_param
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * mmq_x;
    if (need_check && (row0 >= M || col0 >= N)) return;

    const int groups_per_row = K / 256;
    const int add = (add_mode == -1) ? add_param : add_mode;

    // LDS layout (must match dispatch.rs shared_mem calc):
    //   x_qs   : MMQ_Y * x_stride ints
    //   x_dm   : MMQ_Y float2
    //   tile_y : mmq_x * Y_STRIDE ints
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
            load_hfq4_tile_streaming<x_stride>(A, x_qs, x_dm, row0, kg, window, M, groups_per_row);
            __syncthreads();
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                vec_dot_dp4a_streaming<mmq_x>(x_qs, x_dm, tile_y, sum, sub);
            }
            __syncthreads();
        }
    }

    write_back_residual_templated<mmq_x, need_check>(Y, sum, row0, col0, M, N, add);
}
