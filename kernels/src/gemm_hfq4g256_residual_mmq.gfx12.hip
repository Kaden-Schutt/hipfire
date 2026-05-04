#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Experimental HFQ4-G256 MMQ residual GEMM for RDNA4 (gfx1200/gfx1201).
//
// Sister of `gemm_hfq4g256_residual_mmq.hip` (RDNA3/3.5). Same outer recipe:
//   - pre-quantize the activation matrix into block_q8_1_mmq DS4 layout
//   - process 128 output rows x 128 batch columns per workgroup
//   - stage both HFQ4 weights and Q8_1 activations in LDS
//   - use i8 WMMA for the dot product and apply HFQ4 scale/zero via Q8_1
//     scale/sum correction
//
// Differences from the RDNA3 reference (validated via the iu8 K=16 layout
// probe in probe_wmma_iu8_k16_layout.gfx12.hip — confirmed 8-row-block acc
// layout matches iu4 K=32):
//
//   1. WMMA builtin: __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
//      (the unsuffixed builtin compiles but does not lower on gfx12).
//
//   2. Operand shape per lane: int32x2 (8 INT8 = K-half of one row),
//      NOT the RDNA3 int32x4 (16 INT8 = full K=16). Gfx12 splits the
//      K=16 across the two lane-groups (lane>>4):
//        A: lane l → row m = l & 0xF, k-half = 8*(l>>4) .. 8*(l>>4)+7
//        B: lane l → col n = l & 0xF, k-half = 8*(l>>4) .. 8*(l>>4)+7
//      One WMMA call consumes K=16 worth — so for each Q8_1 sub-block of
//      K=32 we issue 2 WMMA calls (one per K=16 sub-tile).
//
//   3. Acc layout: 8-row-block (NOT RDNA3's stride-2 interleave):
//        lane l, slot j ∈ [0,7] → C[8*(l>>4) + j][l & 0xF]
//      In the J_MAJOR sense for tile_C, this maps to:
//        get_i(l) = 8*(threadIdx.x>>4) + l
//        get_j(l) = threadIdx.x & 0xF
//      8 slots per lane × 32 lanes = 256 = full 16×16 C tile.
//
// Skips the `_full_add` and `_full_set` boundary-elided variants — only the
// basic `_residual_mmq` with a runtime `add` flag. Those can be added later
// once the basic variant is validated.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32
#define QK8_1 32
#define QI8_1 8
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K 36
#define MMQ_TILE_X_K 76

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

#if !HIPFIRE_GFX12

extern "C" __global__ void quantize_q8_1_mmq_ds4_gfx12(
    const float*,
    block_q8_1_mmq*,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_gfx12(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

#else

// ---------------------------------------------------------------------------
// Activation Q8_1 quantization. Arch-agnostic body — uses __shfl_xor and FP16
// conv only. Identical to the RDNA3 `quantize_q8_1_mmq_ds4` (lines 184-224 of
// the RDNA3 file); the function is renamed here so both kernels can coexist
// in the same compiled module if a future caller wants to dispatch by arch
// from a single binary.
// ---------------------------------------------------------------------------
extern "C" __global__ void quantize_q8_1_mmq_ds4_gfx12(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 31) == 0) {
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
    }
}

// ---------------------------------------------------------------------------
// HFQ4 weight tile loader. Same byte layout as RDNA3 — 136 B per 256-elem
// group: 4 B FP32 scale + 4 B FP32 zero + 128 B of nibbles. The only thing
// that matters for gfx12 is the LDS layout (`x_qs[i*MMQ_TILE_X_K + k]`,
// `x_dm[i*MMQ_TILE_X_K + ...]` with the same i32 / half2 strides), and that
// is identical to RDNA3 since the dot consumer rewrites the per-lane stride
// only inside the MMA (no shared-memory bank-conflict patterns differ in a
// way that the staging code needs to know about).
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void load_hfq4_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int rows_per_warp = 1;
    const int txi = threadIdx.x;

    for (int i0 = 0; i0 < MMQ_Y; i0 += rows_per_warp * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + txi * 4);

        const unsigned int q0 = (qs0 >>  0) & 0xFu;
        const unsigned int q1 = (qs0 >>  4) & 0xFu;
        const unsigned int q2 = (qs0 >>  8) & 0xFu;
        const unsigned int q3 = (qs0 >> 12) & 0xFu;
        const unsigned int q4 = (qs0 >> 16) & 0xFu;
        const unsigned int q5 = (qs0 >> 20) & 0xFu;
        const unsigned int q6 = (qs0 >> 24) & 0xFu;
        const unsigned int q7 = (qs0 >> 28) & 0xFu;
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] = q4 | (q5 << 8) | (q6 << 16) | (q7 << 24);
    }

    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const int ksc = threadIdx.x & 1;
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// gfx12 i8 WMMA dot product.
//
// Produces partial sums for a 16-row × 16-col block of the C output. Each
// warp covers 32 rows (rows_per_warp = 32 = 2 × tile_C::I) and iterates over
// MMQ_X / (ntx * tile_C::J) = 128 / (2 * 16) = 4 col-blocks of 16 cols each.
// Total per-warp accumulator footprint: 32 rows × 128 cols / 32 lanes = 128
// floats per lane (= 4 col-blocks × 2 m-tiles × 16 slots per warp / 32 lanes
// × 8 slots per lane wait let me recount).
//
// Footprint reconciliation: total work per workgroup = MMQ_X * MMQ_Y =
// 128*128 = 16384 acc floats. NWARPS * WARP_SIZE = 8 * 32 = 256 lanes.
// 16384 / 256 = 64 floats per lane. The host-allocated `sum[]` buffer must
// hold this — see the kernel body below.
//
// Per inner-k iteration (k01 step of QI8_1 = 8 i32 = K=32 of activations =
// one Q8_1 sub-block), we issue 2 WMMA calls — one per K=16 sub-tile. The
// scale/sum correction (`dmA.x * dsB.x * acc + dmA.y * dsB.y`) applies once
// per Q8_1 sub-block, so we accumulate the int32 from both K=16 MMAs into
// one running int and apply the correction at the end of the k01 iteration.
// That keeps the residual semantics aligned with RDNA3.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma_gfx12(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    constexpr int tile_C_I = 16;
    constexpr int tile_C_J = 16;
    constexpr int tile_C_ne = 8;  // 8 acc slots per lane (8-row-block layout)
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C_I;  // = 2 m-tiles per warp

    y += (threadIdx.y % ntx) * (tile_C_J * MMQ_TILE_Y_K);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    // Per-lane decomposition (8-row-block):
    //   m_lane = tid & 0xF   -> row within 16-row tile (also col for B)
    //   k_grp  = tid >> 4    -> selects K-half within each K=16 MMA tile
    const int tid    = (int)threadIdx.x;
    const int m_lane = tid & 0xF;
    const int k_grp  = tid >> 4;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;  // i32 stride into x_qs / y_qs (K=4 INT8 per i32)

        // Per-m-tile activation pre-load (independent of n-tile loop).
        // tile_A_int[n][sub] holds 2 i32 (= 8 INT8 = K=8) per K=16 sub-tile.
        // n in [0, ntx), sub in [0, 2) selecting K=0..15 vs K=16..31 of the
        // QI8_1 stride.
        int32x2_t a_frag[ntx][2];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            const int row_in_tile = n * tile_C_I + m_lane;
            const int* row_ptr = x_qs + (i0 + row_in_tile) * MMQ_TILE_X_K + k0;
            // sub-tile 0: K=0..15 of this k01 step, lane k-half offset = 2*k_grp i32
            // sub-tile 1: K=16..31, base offset = 4 i32 (= K=16 in i32 stride)
            a_frag[n][0][0] = row_ptr[2 * k_grp + 0];
            a_frag[n][0][1] = row_ptr[2 * k_grp + 1];
            a_frag[n][1][0] = row_ptr[4 + 2 * k_grp + 0];
            a_frag[n][1][1] = row_ptr[4 + 2 * k_grp + 1];
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J) {
            // B fragment for this n-block (16 cols = 1 J=16 tile per sub-tile).
            // get_i for B = m_lane (col within 16-col block); K positions per
            // K=16 sub-tile = 2 i32 per lane at offset 2*k_grp i32 inside the
            // sub-tile's i32 base.
            const int col_in_tile = m_lane;  // J_MAJOR-equivalent: lane idx maps to col
            const int* col_ptr = y_qs + (j0 + col_in_tile) * MMQ_TILE_Y_K + k01;
            int32x2_t b0_v;
            int32x2_t b1_v;
            b0_v[0] = col_ptr[2 * k_grp + 0];
            b0_v[1] = col_ptr[2 * k_grp + 1];
            b1_v[0] = col_ptr[4 + 2 * k_grp + 0];
            b1_v[1] = col_ptr[4 + 2 * k_grp + 1];

            // dsB applies to the whole K=32 Q8_1 sub-block (one per k01 step).
            // We pull the d/s scalars per col, indexed by lane (the
            // 8-row-block acc layout means lane l, slot j outputs to
            // C[8*(l>>4)+j][l&0xF], so the col index is m_lane = l & 0xF —
            // which matches col_in_tile above).
            const int j = j0 + col_in_tile;
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                int32x8_t acc = {0,0,0,0,0,0,0,0};
                // Two K=16 WMMA calls accumulate into the same int acc; the
                // scale-correction is applied once after both.
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    /*a_signed*/ true, a_frag[n][0],
                    /*b_signed*/ true, b0_v,
                    acc, /*clamp*/ true);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, a_frag[n][1],
                    true, b1_v,
                    acc, true);

                // Apply HFQ4 scale + Q8_1 zero-correction. Acc layout:
                //   slot j ∈ [0,7] → C-row = 8*k_grp + j (within the n*16
                //   m-tile), C-col = m_lane (within the j0 n-tile).
                // The host `sum[]` buffer is laid out by (n-block-id,
                // m-tile-id-n, slot) so the index is identical to the RDNA3
                // store path with tile_C::ne = 8 instead of 4.
                #pragma unroll
                for (int l = 0; l < tile_C_ne; ++l) {
                    const int i_in_tile = n * tile_C_I + 8 * k_grp + l;
                    const int i = i0 + i_in_tile;
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0 / QI8_1]);
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l] += dmA.x * dsB.x * (float)acc[l];
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main MMQ residual GEMM kernel for gfx12.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_gfx12(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

    // Per-lane sum buffer:
    //   total elements = MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)
    //                  = 128 * 128 / 256 = 64
    // Layout: 4 n-blocks × 2 m-tiles × 8 slots per lane = 64 floats per lane.
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_hfq4_tile<false>(A, tile_x, row0, kb, M, groups_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_gfx12(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_gfx12(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    // C-store. 8-row-block acc layout:
    //   lane l, slot j → C[8*(l>>4) + j][l & 0xF]  (within each 16×16 tile)
    constexpr int tile_C_I = 16;
    constexpr int tile_C_J = 16;
    constexpr int tile_C_ne = 8;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C_I;

    const int tid    = (int)threadIdx.x;
    const int m_lane = tid & 0xF;
    const int k_grp  = tid >> 4;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    const int j_offset_in_warp = (threadIdx.y % ntx) * tile_C_J;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C_ne; ++l) {
                // i within 16-row tile = 8*k_grp + l (8-row-block).
                // j within 16-col tile = m_lane.
                const int i = i0 + n * tile_C_I + 8 * k_grp + l;
                const int j = j0 + j_offset_in_warp + m_lane;
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C_J + n) * tile_C_ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

#endif
