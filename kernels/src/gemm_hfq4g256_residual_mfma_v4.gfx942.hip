#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MFMA-direct HFQ4G256 GEMM v4 for gfx942 — 1×4 col wave arrangement +
// shared LDS A-tile (dequanted weights).
//
// v3 trajectory (with K_CHUNK=128 LDS B-tile): 1284 / 1315 = 97.6%.
// v4 changes:
//   * Waves arranged 1×4 in COLS: all 4 waves cover the same 16-row slab,
//     but 4 different 16-col output slabs. This means all 4 waves READ
//     THE SAME HFQ4 WEIGHTS per K-iter → cooperative LDS A-tile.
//   * Output tile: 16 rows × 64 cols per WG (vs v3's 32 × 32).
//   * LDS holds dequanted FP16 A-tile (16 × K_CHUNK) + B-tile (64 × K_CHUNK).
//     For K_CHUNK=128: A=4KB, B=16KB = 20KB total per WG.
//
// Expected win: ~4× less HFQ4 weight HBM traffic per WG (16 rows × K
// once vs 4 separate row-groups × K each in v3's 2×2 arrangement).
// Combined with LDS A-tile sharing, should beat rocBLAS for this shape.
//
// Math: byte-identical to v3 per MFMA call (same `__builtin_amdgcn_mfma_f32_16x16x16f16`).

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    vfloat4 __attribute__((ext_vector_type(4)));

#ifndef K_CHUNK
#define K_CHUNK 128
#endif
#define M_TILE 16
#define N_TILE 64    // 4 waves × 16 cols each

extern "C" __global__ __launch_bounds__(256)
void gemm_hfq4g256_residual_mfma_v4_gfx942(
    const char*  __restrict__ A,
    const half_t* __restrict__ X,
    float*       __restrict__ Y,
    int M, int K, int batch_size
) {
    __shared__ half_t lds_a[M_TILE * K_CHUNK];   // 16 × 128 = 2048 fp16 = 4 KB
    __shared__ half_t lds_b[N_TILE * K_CHUNK];   // 64 × 128 = 8192 fp16 = 16 KB

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..3
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;
    const int strip     = lane_in_w >> 4;

    // 1×4 col arrangement: all waves share rows 0..15, different col ranges.
    const int wave_n_off = wave_id * 16;        // 0, 16, 32, 48
    const int m_offset = blockIdx.x * M_TILE;
    const int b_offset_global = blockIdx.y * N_TILE;
    const int b_offset = b_offset_global + wave_n_off;

    const int groups_per_row = K / 256;

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kc = 0; kc < K; kc += K_CHUNK) {
        // ─── Cooperative LDS load of A-tile (16 rows × K_CHUNK, dequanted) ───
        // 256 threads load 16 × K_CHUNK = 2048 fp16. Each thread loads 8 fp16.
        // Map: thread tid handles 8 contiguous K-positions for row (tid / (K_CHUNK/8))
        // Actually simpler: 16 rows × K_CHUNK / 256 threads = K_CHUNK / 16 elements/thread.
        // For K_CHUNK=128: 8 elements per thread (one nibble byte = 2 elements).
        __syncthreads();

        // Each thread covers row = tid / (K_CHUNK / 8) and (K_CHUNK / 8) consecutive
        // K-positions starting at base. We use byte-level traversal:
        //   nibble_per_byte = 2 → 4 bytes covers 8 K-elements
        //   per thread = 4 bytes ⇔ 8 K-elements
        {
            const int elems_per_thread = (M_TILE * K_CHUNK) / 256;   // 8 for K_CHUNK=128
            const int my_row_in_tile = tid / (K_CHUNK / elems_per_thread);
            const int my_k_chunk_pos = (tid % (K_CHUNK / elems_per_thread)) * elems_per_thread;
            const int global_row = m_offset + my_row_in_tile;
            if (global_row < M) {
                // Load elems_per_thread K-elements starting at kc + my_k_chunk_pos
                const int gk_base = kc + my_k_chunk_pos;
                #pragma unroll
                for (int ee = 0; ee < elems_per_thread; ee += 8) {
                    int gk = gk_base + ee;
                    int g = gk / 256;
                    int ki = gk % 256;
                    const char* gp = A + (long long)global_row * groups_per_row * 136 + g * 136;
                    float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                    float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                    half_t sc_h = (half_t)sc, zp_h = (half_t)zp;
                    // Read 4 bytes = 8 nibbles
                    const unsigned int pk = *(const unsigned int*)(gp + 8 + ki/2);
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        unsigned int nib = (pk >> (b*4)) & 0xFu;
                        lds_a[my_row_in_tile * K_CHUNK + my_k_chunk_pos + ee + b] =
                            sc_h * (half_t)nib + zp_h;
                    }
                }
            }
        }

        // ─── Cooperative LDS load of B-tile (64 cols × K_CHUNK) ───
        {
            const int elems_per_thread = (N_TILE * K_CHUNK) / 256;   // 32 for K_CHUNK=128
            const int my_col = tid / (K_CHUNK / elems_per_thread);   // tid / 4
            const int my_k_base = (tid % (K_CHUNK / elems_per_thread)) * elems_per_thread;
            int gb = b_offset_global + my_col;
            #pragma unroll
            for (int e = 0; e < elems_per_thread; e++) {
                int gk = kc + my_k_base + e;
                half_t v = (gb < batch_size && gk < K) ? X[(long long)gb * K + gk] : (half_t)0;
                lds_b[my_col * K_CHUNK + my_k_base + e] = v;
            }
        }
        __syncthreads();

        // ─── MFMA loop over chunk (K_CHUNK / 16 MFMA-K steps) ───
        #pragma unroll
        for (int kk = 0; kk < K_CHUNK; kk += 16) {
            // Read A from LDS: lane (n_lane, strip) → A[n_lane, kk + strip*4 + r]
            half4 a_reg;
            int a_row_in_tile = n_lane;
            int a_k_base = kk + strip * 4;
            a_reg[0] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 0];
            a_reg[1] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 1];
            a_reg[2] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 2];
            a_reg[3] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 3];

            // Read B from LDS: lane (n_lane, strip) → B[kk+strip*4+r, wave_n_off+n_lane]
            int b_col_in_tile = wave_n_off + n_lane;
            int b_k_base = kk + strip * 4;
            half4 b_reg;
            b_reg[0] = lds_b[b_col_in_tile * K_CHUNK + b_k_base + 0];
            b_reg[1] = lds_b[b_col_in_tile * K_CHUNK + b_k_base + 1];
            b_reg[2] = lds_b[b_col_in_tile * K_CHUNK + b_k_base + 2];
            b_reg[3] = lds_b[b_col_in_tile * K_CHUNK + b_k_base + 3];

            c_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_acc, 0, 0, 0);
        }
    }

    // Output: lane c[r] = C[m = strip*4 + r within tile, n = n_lane within tile]
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            Y[(long long)gb * M + gm] += c_acc[r];
        }
    }
}
