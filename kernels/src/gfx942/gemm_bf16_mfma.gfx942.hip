#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// BF16 × BF16 → FP32 MFMA-direct GEMM for gfx942 (MI300X CDNA3).
//
// Foundation kernel for the Tier 1 hipfire-native BF16 calibration path
// (replaces Tier 2 llama.cpp `llama-imatrix` + PyTorch `collect_hessian.py`
// subprocess wrappers). Both operands are BF16 in global memory — no
// dequantization, no FP16 shadow buffer — feeding straight into
// `__builtin_amdgcn_mfma_f32_16x16x16bf16`.
//
// Layout copies the v3 sweet-spot from `gemm_hfq4g256_residual_mfma.gfx942.hip`
// (commit 496edfbc, 97.6% of rocBLAS Tensile on HFQ4 prefill):
//
//   A:  [M, K] BF16, row-major  (M = output features, K = input dim)
//   B:  [batch_size, K] BF16, row-major (calibration activations / token tiles)
//   D:  [batch_size, M] FP32, row-major
//
// Geometry:
//   block = [256, 1, 1]                4 wave64/WG, 32×32 output tile per WG
//   grid  = [(M+31)/32, (batch+31)/32] one WG per (output-row-tile, batch-tile)
//   LDS   = 32 × K_CHUNK × 2 B         B-tile cooperative load (4 KB at K_CHUNK=128)
//
// Intrinsic: `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`.
// The `_1k` suffix is the gfx90a/gfx942 inherited name for the 16-K-wide
// BF16 MFMA (4 BF16 lanes × 16 m-rows × 16 n-cols per call). The operand
// vector type is `short __attribute__((ext_vector_type(4)))` — clang
// represents BF16 lane bits as i16 here, so we bit-cast a `__bf16` x4
// vector into the i16 x4 expected by the intrinsic. No data movement;
// pure type punning.
//
// MFMA F32_16x16x16_BF16 lane layout — identical to the F32_16x16x16_F16
// variant verified via `docs/investigations/2026-05-19-mfma-hfq4/mfma_test.cpp`
// (max_err 8.9e-08 on F16; the BF16 intrinsic shares the same lane mapping,
// only the operand dtype differs):
//
//   A operand:  lane l holds a[r] = A[m = l%16,        k = (l/16)*4 + r]
//   B operand:  lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]
//   D output:   lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]   ← strip-major
//
// The output strip-major layout (lane l holds D[m=(l/16)*4+r, n=l%16])
// differs from the A-operand m-major-by-lane input layout. Easy bug to
// introduce when swapping intrinsics. The v3 HFQ4 kernel has the same
// trap and resolves it the same way.
//
// Wave-tile arrangement within a WG (4 wave64 → 2×2 wave grid over the
// 32×32 output tile):
//
//   wave_id 0 → output sub-tile (m_off=0,  n_off=0)
//   wave_id 1 → output sub-tile (m_off=0,  n_off=16)
//   wave_id 2 → output sub-tile (m_off=16, n_off=0)
//   wave_id 3 → output sub-tile (m_off=16, n_off=16)
//
// All 4 waves cooperatively load one B-chunk [32 cols × K_CHUNK k] into LDS
// per outer-k step, then each wave's 16×16 MFMA reads B from LDS. v3's
// HFQ4 measurement showed this LDS B-tile is the lever that closes the
// gap to rocBLAS — same expectation applies here (B is now actual BF16
// activations in HBM, the cooperative load amortizes HBM bandwidth across
// 4 MFMA iterations sharing the chunk).

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef short  s16x4  __attribute__((ext_vector_type(4)));
typedef float  vfloat4 __attribute__((ext_vector_type(4)));

// MFMA F32_16x16x16_BF16 intrinsic — operands are bit-cast __bf16x4 → short x4.
// LLVM's amdgcn intrinsic models the BF16 lane bits as i16; bit_cast is the
// no-op punning the compiler folds away.
__device__ static inline vfloat4 mfma_bf16(bf16x4 a, bf16x4 b, vfloat4 c) {
    s16x4 ai = __builtin_bit_cast(s16x4, a);
    s16x4 bi = __builtin_bit_cast(s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}

#define K_CHUNK 128
#define N_TILE  32

extern "C" __global__ __launch_bounds__(256)
void gemm_bf16_mfma_gfx942(
    const bf16_t* __restrict__ A,    // [M, K] BF16 row-major
    const bf16_t* __restrict__ B,    // [batch_size, K] BF16 row-major
    float*        __restrict__ D,    // [batch_size, M] FP32 row-major (overwritten)
    int M, int K, int batch_size
) {
    __shared__ bf16_t lds_b[N_TILE * K_CHUNK];   // 32 × 128 = 4096 bf16 = 8 KB

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..3
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;       // 0..15  (A.m, B.n, D.n within tile)
    const int strip     = lane_in_w >> 4;       // 0..3   (k-strip / D row-strip)

    const int wave_m_off = (wave_id & 2) ? 16 : 0;
    const int wave_n_off = (wave_id & 1) ? 16 : 0;
    const int m_offset = blockIdx.x * 32 + wave_m_off;          // global output-row start
    const int b_offset_global = blockIdx.y * 32;                // global batch start (per-WG)
    const int b_offset = b_offset_global + wave_n_off;          // per-wave batch start

    const int my_m = m_offset + n_lane;                         // this lane's output row

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // K outer loop in chunks of K_CHUNK (= 8 MFMA-K=16 iterations per chunk).
    for (int kc = 0; kc < K; kc += K_CHUNK) {
        // ─── Cooperative LDS load of B chunk [32 cols × K_CHUNK k] ───
        // 256 threads load 32 × 128 = 4096 bf16 = 16 per thread.
        __syncthreads();
        #pragma unroll
        for (int li = 0; li < 16; li++) {
            int idx = tid + li * 256;
            int b_col = idx / K_CHUNK;             // 0..31
            int b_k   = idx % K_CHUNK;             // 0..K_CHUNK-1
            int gb = b_offset_global + b_col;
            int gk = kc + b_k;
            bf16_t v = (gb < batch_size && gk < K)
                ? B[(long long)gb * K + gk]
                : (bf16_t)0;
            lds_b[b_col * K_CHUNK + b_k] = v;
        }
        __syncthreads();

        // ─── MFMA loop over chunk in 16-element K steps ───
        #pragma unroll
        for (int kk = 0; kk < K_CHUNK; kk += 16) {
            int gk_start = kc + kk;

            // A operand: each lane reads 4 contiguous K-elements
            //   k = gk_start + strip*4 + r for r=0..3
            // Out-of-bounds rows / k's get zero (clean boundary handling).
            bf16x4 a_reg;
            int a_k_base = gk_start + strip * 4;
            if (my_m < M) {
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int gk = a_k_base + r;
                    a_reg[r] = (gk < K)
                        ? A[(long long)my_m * K + gk]
                        : (bf16_t)0;
                }
            } else {
                a_reg[0] = (bf16_t)0; a_reg[1] = (bf16_t)0;
                a_reg[2] = (bf16_t)0; a_reg[3] = (bf16_t)0;
            }

            // B operand: read 4 contiguous K-elements from LDS
            //   col_in_tile = wave_n_off + n_lane  (0..31, this wave's n half)
            //   k_in_chunk_base = kk + strip*4
            int b_col_in_tile   = wave_n_off + n_lane;
            int k_in_chunk_base = kk + strip * 4;
            bf16x4 b_reg;
            b_reg[0] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 0];
            b_reg[1] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 1];
            b_reg[2] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 2];
            b_reg[3] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 3];

            c_acc = mfma_bf16(a_reg, b_reg, c_acc);
        }
    }

    // ─── Output write: strip-major over m ─────────────────────────────
    // Lane (strip, n_lane) holds c_acc[r] = D[m = m_offset + strip*4 + r,
    //                                          n = b_offset + n_lane]
    // D is [batch_size, M] row-major: D[b, m] = D[b * M + m]
    //
    // Plain GEMM: overwrite (no residual add). The Tier 1 calibration
    // path uses this kernel as a forward primitive; the Σx² / outer-
    // product reductions sit on top and accumulate themselves.
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            D[(long long)gb * M + gm] = c_acc[r];
        }
    }
}
