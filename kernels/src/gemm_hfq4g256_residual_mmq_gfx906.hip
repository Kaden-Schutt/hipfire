#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ═══ HFQ4-G256 × Q8_1 dp4a MMQ residual GEMM for gfx906 (MI50/MI60) ═══
//
// Variant override: replaces the empty stubs in
// gemm_hfq4g256_residual_mmq.hip when compiled for --offload-arch=gfx906
// (see scripts/compile-kernels.sh chip-specific variant lookup).
//
// Phase 1 implementation per plans/p1.2_dp4a_mmq_design.md.
// Reference: llama.cpp-gfx906 mmq.cuh:3429-3462 (process_tile structure)
// and vecdotq.cuh:254-278 (dp4a + Q8_1 zp correction).
//
// Topology (wave-native gfx906):
//   block dim (64, 2, 1) = 128 threads = 2 wave64s
//   grid (ceil(M/128), ceil(N/64), 1)  — one 128×64 tile per WG
//
// Per outer kg iter (one HFQ4 group = 256 K-elements):
//   1. Load X tile once (full 256-K span, unpacked nibbles → int8)
//   2. Load Y first half (32 K-elements), compute, sync
//   3. Load Y second half (32 K-elements), compute, sync
//
// LDS budget at (mmq_y=128, mmq_x=64):
//   x_qs:  128 × 65 ints   = 33,280 B  (64 ints data + 1 int padding/row)
//   x_dm:  128 half2       =    512 B
//   tile_y: 64 × 36 ints   =  9,216 B  (4 ints inlined ds + 32 qs/col)
//   TOTAL                  = 43,008 B  (42.0 KiB / 64 KiB cap)
//
// L2 prefetch: NOT included in this commit. To be added in a follow-up.
//
// HFQ4 layout (per group, 136 B): scale (f32) + zp (f32) + 128 B nibbles.
//   Nibble 2n = K-element 2n in low 4 bits of byte n.
//   Nibble 2n+1 = K-element 2n+1 in high 4 bits of byte n.
//   256 nibbles/group cover 256 K-elements.
//
// Q8_1 block (block_q8_1_mmq, 144 B): 4×half2 ds + 128 int8 qs.
//   One block covers 128 K-elements (= half an HFQ4 group).
//
// dp4a math: HFQ4 nibbles are 0..15 (positive int8 range), safe for sdot4
// without biasing. Q8_1 is symmetric (no zp on activations) — only one
// correction term needed: scale_w * d_x * sumi + zp_w * sum_x.

#define MMQ_X 64
#define MMQ_Y 128
#define MMQ_NWARPS 2
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32
#define QK8_1 32
#define QI8_1 8

// X-tile per-row stride: 64 ints of unpacked weights + 1 int padding for
// LDS bank conflict avoidance. Matches llama.cpp dp4a formula
// "K % 2 == 1 for dp4a" (mmq.cuh:175).
#define X_STRIDE 65

// Y-tile per-col stride: 4 ints of inlined ds (4 half2 values for the 4
// 32-K-element sub-blocks within a Q8_1 block) + 32 ints of qs. We use
// the first 4 ints as the ds region; loaded by all threads at the head
// of the col slot.
#define Y_STRIDE 36

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// ─── Tile loaders ─────────────────────────────────────────────────────────

// Load one HFQ4 group (256 K-elements) into x_qs and x_dm.
// Each thread reads 4 bytes of HFQ4 (= 8 nibbles) and produces 2 int32s
// in the LDS tile. Thread layout:
//   tid (0..127) → row = tid % 128 (one row per thread for one chunk),
//                  chunk = tid / 128 (0..15? — we need 16 chunks per row).
// Actually with 128 threads and 64 ints/row, we need 32 chunks per row
// (each chunk = 2 ints from one 4-byte HFQ4 read). 32 × 128 = 4096 ints
// total; 128 threads × 32 ints each. So loop unroll 32 within each thread.
//
// Simpler tiling: tid = (chunk_id) for chunks 0..127, where each thread
// handles all 32 chunks for its assigned row pair... no, that doesn't
// scale either. Let's just iterate.
static __device__ __forceinline__ void load_hfq4_tile_dp4a(
    const char* __restrict__ A,
    int* __restrict__ x_qs,
    half2* __restrict__ x_dm,
    int row0, int kg, int M, int groups_per_row
) {
    // Block has 128 threads; X tile has 128 rows × 64 ints to fill.
    // Use 16 threads per row (each loads 4 ints = 8 nibbles = 2 uint reads
    // worth, OR 4 ints from 2 separate uint reads).
    //
    // Simplest pattern: tid → which (row, ints_pos) does this thread own?
    // 128 threads × 64 ints/row = 8192 ints total to write.
    // 128 rows / nwarps_collective groups... Actually let's just loop.
    //
    // Each thread reads ONE uint (4 bytes = 8 nibbles) and writes TWO ints
    // to x_qs (low-nibble int and high-nibble int — but in K-sequential
    // order: int_lo holds K{4n,4n+1,4n+2,4n+3} or similar?).
    //
    // Actually per the gemv reference: 4 bytes packs 8 K-sequential
    // nibbles. Decompose: byte b (b=0..3) holds K-elements {2b, 2b+1}.
    // For dp4a we want 4 K-sequential int8s per int32. So:
    //   int_a = nibble 0 (K0), nibble 1 (K1), nibble 2 (K2), nibble 3 (K3)
    //   int_b = nibble 4 (K4), nibble 5 (K5), nibble 6 (K6), nibble 7 (K7)
    // i.e. int_a packs K-elements 0..3 in bytes 0..3 (low byte = K0).
    //      int_b packs K-elements 4..7 in bytes 0..3.
    //
    // From a single uint qs0 = bytes [b0, b1, b2, b3] little-endian:
    //   K0 = b0 & 0xF      → byte 0 of int_a
    //   K1 = (b0 >> 4) & 0xF → byte 1 of int_a
    //   K2 = b1 & 0xF      → byte 2 of int_a
    //   K3 = (b1 >> 4) & 0xF → byte 3 of int_a
    //   K4 = b2 & 0xF      → byte 0 of int_b
    //   K5 = (b2 >> 4) & 0xF → byte 1 of int_b
    //   K6 = b3 & 0xF      → byte 2 of int_b
    //   K7 = (b3 >> 4) & 0xF → byte 3 of int_b
    //
    // So:
    //   int_a = (qs0 & 0x0F) | ((qs0 & 0xF0) << 4) | ((qs0 & 0x0F00) << 8) | ((qs0 & 0xF000) << 12);
    //   int_b = ((qs0 >> 16) & 0x0F) | ((qs0 >> 16) & 0xF0) << 4 | ...
    // Hmm, tricky. Let me factor:
    //   int_a is in K-elements 0..3 covering byte offset boff=0 of HFQ4
    //   int_b is in K-elements 4..7 covering byte offset boff=2 of HFQ4
    // So they come from different uint reads.
    //
    // Cleaner: each thread reads ONE uint (4 contiguous bytes from HFQ4
    // payload) which holds 8 K-sequential nibbles, and produces TWO
    // sequential x_qs ints. That's exactly what we need for sequential K
    // staging.
    //
    // Per row: 64 x_qs ints = 32 uint reads from HFQ4 payload (32 × 2 = 64).
    // Hmm wait: 32 uint reads × 2 ints/uint = 64 ints per row. ✓
    //
    // Distribute: 128 threads × 32 chunks/row × 128 rows / 128 threads
    //   = 32 chunks per thread for its 1 assigned row.
    // i.e. 1 thread per row, each loops 32 times.
    //
    // That's a lot of dependent iterations per thread but only ~16 KB of
    // HBM read per thread (32 × 4 bytes × 1 row), which the L2 should
    // absorb. Acceptable for a first pass.

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..127
    const int i = tid;                                        // row in tile, 0..127
    const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
    const char* gp = A + ((long long)row * groups_per_row + kg) * 136;

    // Load scale and zp once per row (broadcast across all 256 K-elements).
    const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
    const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
    x_dm[i] = make_half2((_Float16)sc, (_Float16)zp);

    // Load 32 chunks of 4 bytes each = 128 bytes of nibbles per row.
    // Each chunk produces 2 sequential x_qs ints.
    #pragma unroll
    for (int chunk = 0; chunk < 32; ++chunk) {
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + chunk * 4);

        // Decompose the 8 nibbles in qs0 into 2 K-sequential ints.
        // Each output int has K-elements {4c, 4c+1, 4c+2, 4c+3} for c=2*chunk
        // (int_a) and c=2*chunk+1 (int_b).
        //
        // int_a layout (little-endian bytes):
        //   byte 0 = K{4c+0} = qs0 & 0x0F           = (qs0 >>  0) & 0x0F
        //   byte 1 = K{4c+1} = (qs0 >> 4) & 0x0F     = (qs0 >>  4) & 0x0F
        //   byte 2 = K{4c+2} = (qs0 >> 8) & 0x0F     = (qs0 >>  8) & 0x0F
        //   byte 3 = K{4c+3} = (qs0 >> 12) & 0x0F    = (qs0 >> 12) & 0x0F
        //
        // int_b layout:
        //   byte 0 = K{4c+4} = (qs0 >> 16) & 0x0F
        //   byte 1 = K{4c+5} = (qs0 >> 20) & 0x0F
        //   byte 2 = K{4c+6} = (qs0 >> 24) & 0x0F
        //   byte 3 = K{4c+7} = (qs0 >> 28) & 0x0F
        const unsigned int n0 = (qs0 >>  0) & 0xFu;
        const unsigned int n1 = (qs0 >>  4) & 0xFu;
        const unsigned int n2 = (qs0 >>  8) & 0xFu;
        const unsigned int n3 = (qs0 >> 12) & 0xFu;
        const unsigned int n4 = (qs0 >> 16) & 0xFu;
        const unsigned int n5 = (qs0 >> 20) & 0xFu;
        const unsigned int n6 = (qs0 >> 24) & 0xFu;
        const unsigned int n7 = (qs0 >> 28) & 0xFu;

        const int int_a = (int)(n0 | (n1 << 8) | (n2 << 16) | (n3 << 24));
        const int int_b = (int)(n4 | (n5 << 8) | (n6 << 16) | (n7 << 24));

        x_qs[i * X_STRIDE + 2 * chunk + 0] = int_a;
        x_qs[i * X_STRIDE + 2 * chunk + 1] = int_b;
    }
}

// Load one Q8_1 K-block (128 K-elements = 32 ints/col) into tile_y.
// Layout per col in tile_y:
//   slot[0..3]   = 4 half2 ds (8 ints worth of half2 = 4 ints) — packed as
//                  (d0, sum0), (d1, sum1), (d2, sum2), (d3, sum3) for
//                  the 4 sub-32-K sub-blocks in this 128-K block.
//   slot[4..35]  = 32 ints of int8 qs (128 int8s)
// Y_STRIDE = 36 (4 ds + 32 qs).
//
// Source: Xq[kb * N + token] where kb is the Q8_1 block index = 2*kg + half.
// One block_q8_1_mmq is 144 B = 36 ints, exactly matching Y_STRIDE.
// So we can just memcpy 36 ints per col.
static __device__ __forceinline__ void load_q8_1_tile(
    const block_q8_1_mmq* __restrict__ Xq,
    int* __restrict__ tile_y,
    int col0, int kb, int N
) {
    // 64 cols × 36 ints/col = 2304 ints. 128 threads → 18 ints/thread.
    // Distribute: tid → flat index in tile_y (col-major).
    //   col = tid / 36, slot = tid % 36 — but 128 threads × 18 ints isn't
    //   uniformly aligned to col boundaries. Easier: 128 threads load 64
    //   cols × 36 ints = 2304 ints via flat-strided iteration.
    //
    // Each col is 36 ints = exactly one block_q8_1_mmq cast as int*.
    // Load by col: each col handled by 36/k threads if k threads per col.
    //   With 128 threads / 64 cols = 2 threads per col → each loads 18 ints.
    // tid = col_pair * 2 + thread_in_col. col_pair = tid / 2, thread = tid % 2.
    // thread 0 of pair loads ints [0..17], thread 1 loads ints [18..35].

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..127
    const int j = tid / 2;                  // col index, 0..63
    const int half = tid & 1;               // 0 or 1: which 18-int half of the col
    const bool valid = (col0 + j) < N;
    const int col = valid ? (col0 + j) : (N - 1);

    // Q8_1 block layout: Xq[kb * N + col] is one block (36 ints).
    const int* src = (const int*)(Xq + (long long)kb * N + col);

    // Zero OOB columns (rather than duplicating last valid col's data).
    // Matches the Strix Halo MMQ kernel pattern (gemm_hfq4g256_residual_mmq.hip:367).
    // Without this, vec_dot's j-loop reads junk for OOB cols and accumulates
    // garbage into per-thread sum[] cells that, while never written, can
    // create issues if compiler aliasing or similar register effects occur.
    #pragma unroll
    for (int u = 0; u < 18; ++u) {
        const int slot = half * 18 + u;     // 0..35
        tile_y[j * Y_STRIDE + slot] = valid ? src[slot] : 0;
    }
}

// ─── dp4a compute ─────────────────────────────────────────────────────────

// Compute partial dot products for one Q8_1 K-block half (32 K-elements).
// k_offset = 0 for the first half, MMQ_TILE_NE_K=32 for the second half.
// (The X tile spans 64 ints/row covering 256 K-elements, so we read with
// k_offset = 0 or 32 to select which half.)
static __device__ __forceinline__ void vec_dot_dp4a(
    const int* __restrict__ x_qs,
    const half2* __restrict__ x_dm,
    const int* __restrict__ tile_y,
    float* __restrict__ sum,
    int k_offset
) {
    // tile_y col layout: ds[0..3] (as 4 ints holding 4 half2s) then qs[4..35].
    const int* y_qs_base = tile_y + 4;            // start of qs per col
    const half2* y_ds_base = (const half2*)tile_y;  // ds at tile_y[0..3] = 2 half2s? wait.

    // Actually: tile_y[0..3] holds 4 half2s = 4 × 4 bytes = 16 bytes = 4 ints. ✓
    // Each half2 is (d, sum) for one 32-K sub-block within this 128-K Q8_1 block.
    // For the *currently loaded* Q8_1 block (one half of the HFQ4 group),
    // there are 4 half2s. We compute 32 K-elements at a time, so we use
    // half2 index k01/QI8_1 ∈ {0, 1, 2, 3} as k01 ranges 0..MMQ_TILE_NE_K-1
    // by step QI8_1=8.
    //
    // Note: k_offset is unused for indexing into tile_y because tile_y
    // holds the currently-loaded half, with ds at the beginning. k_offset
    // is only used to index into x_qs (offset in K within the X tile).

    constexpr int vdr = 8;  // dp4a calls per (i, j, k01) — covers 8 K-elements

    #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += vdr) {  // 4 iters
        const int kx = k01 + k_offset;  // offset into x_qs row
        const int ky = k01;             // offset into y_qs col

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_NWARPS) {  // 32 iters when MMQ_X=64
            const int j = j0 + threadIdx.y;

            // Read the (d, sum) pair for this 32-K sub-block of column j.
            // y_ds_base for col j is at tile_y + j * Y_STRIDE, viewed as half2*.
            // The 4 half2s occupy tile_y[j*Y_STRIDE..j*Y_STRIDE+3] (4 ints)
            // = 8 half2 slots if interpreted contiguously. We want the
            // (k01/QI8_1)-th half2.
            const half2* y_ds_col = (const half2*)(tile_y + j * Y_STRIDE);
            const half2 ds_j = y_ds_col[k01 / QI8_1];  // (d, sum) for this 8-K-element subgroup

            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {  // 2 iters when MMQ_Y=128
                const int i = i0 + threadIdx.x;

                int sumi = 0;
                #pragma unroll
                for (int v = 0; v < vdr; ++v) {
                    const int x_int = x_qs[i * X_STRIDE + kx + v];
                    const int y_int = y_qs_base[j * Y_STRIDE + ky + v];
                    sumi = __builtin_amdgcn_sdot4(x_int, y_int, sumi, false);
                }

                // Scale/zp correction (HFQ4 affine × Q8_1 symmetric → 2 terms).
                // Bias divisor (QI8_1 / vdr) = 8 / 8 = 1 → omitted.
                const half2 dm_i = x_dm[i];
                const float2 dmf = __half22float2(dm_i);
                const float2 dsf = __half22float2(ds_j);
                const float scale_w = dmf.x;
                const float zp_w    = dmf.y;
                const float d_x     = dsf.x;
                const float sum_x   = dsf.y;

                const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
                sum[idx] += scale_w * d_x * (float)sumi + zp_w * sum_x;
            }
        }
    }
}

// ─── Write-back ───────────────────────────────────────────────────────────

// Write the per-thread sum[] tile back to Y[N × M].
// sum[idx] for idx = (j0/NWARPS) * (MMQ_Y/WAVE) + (i0/WAVE)
// covers (j0..j0+1) × (i0..i0+WAVE) for j0 ∈ [0, MMQ_X) step NWARPS, i0 ∈ [0, MMQ_Y) step WAVE.
// The thread's actual (j, i) = (j0 + threadIdx.y, i0 + threadIdx.x).
//
// add=1 means accumulate (Y += result), add=0 means overwrite (Y = result).
static __device__ __forceinline__ void write_back_residual(
    float* __restrict__ Y,
    const float* __restrict__ sum,
    int row0, int col0, int M, int N, int add
) {
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col = col0 + j;
        if (col >= N) continue;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row = row0 + i;
            if (row >= M) continue;

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            // Y is laid out as [N, M] row-major: Y[col * M + row].
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

// Body factored out so the three exported __global__ entry points can share
// it without one __global__ calling another (HIP doesn't allow that).
static __device__ __forceinline__ void mmq_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;

    // LDS layout (43 KiB total):
    //   x_qs: 128 × X_STRIDE (= 65) ints           = 33,280 B
    //   x_dm: 128 half2                             =    512 B
    //   tile_y: MMQ_X × Y_STRIDE (= 64 × 36) ints  =  9,216 B
    extern __shared__ int smem[];
    int*  x_qs   = smem;
    half2* x_dm  = (half2*)(x_qs + MMQ_Y * X_STRIDE);
    int*  tile_y = (int*)(x_dm + MMQ_Y);

    // Per-thread accumulators: 64 floats = 256 B = 64 VGPRs.
    float sum[(MMQ_X / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        // ── X once: full HFQ4 group covers 256 K-elements ──
        load_hfq4_tile_dp4a(A, x_qs, x_dm, row0, kg, M, groups_per_row);

        // ── Y first half: Q8_1 K-block at kb = 2*kg ──
        load_q8_1_tile(Xq, tile_y, col0, /*kb=*/ 2*kg, N);
        __syncthreads();
        vec_dot_dp4a(x_qs, x_dm, tile_y, sum, /*k_offset=*/ 0);
        __syncthreads();

        // ── Y second half: Q8_1 K-block at kb = 2*kg + 1 ──
        load_q8_1_tile(Xq, tile_y, col0, /*kb=*/ 2*kg + 1, N);
        __syncthreads();
        vec_dot_dp4a(x_qs, x_dm, tile_y, sum, /*k_offset=*/ MMQ_TILE_NE_K);
        __syncthreads();
    }

    write_back_residual(Y, sum, row0, col0, M, N, add);
}

// ─── Kernel entry points ──────────────────────────────────────────────────
//
// Symbol names are gfx906-suffixed so the runtime kernel cache and
// dispatch routing don't collide with the RDNA3 MMQ kernel symbols
// produced by gemm_hfq4g256_residual_mmq.hip. The runtime compiles this
// file as its own module via GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_SRC.

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WAVE_SIZE, 2)
void gemm_hfq4g256_residual_mmq_gfx906(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    mmq_body(A, Xq, Y, M, K, N, add);
}

// _full_add / _full_set variants for the M%MMQ_Y==0 && N%MMQ_X==0 hot path.
// Currently same body with fixed `add`; P2 will specialize to drop bounds checks.
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WAVE_SIZE, 2)
void gemm_hfq4g256_residual_mmq_gfx906_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int /*add*/
) {
    mmq_body(A, Xq, Y, M, K, N, /*add=*/ 1);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WAVE_SIZE, 2)
void gemm_hfq4g256_residual_mmq_gfx906_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int /*add*/
) {
    mmq_body(A, Xq, Y, M, K, N, /*add=*/ 0);
}
