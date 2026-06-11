#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MFMA-direct HFQ4G256 GEMM v3 for gfx942 (MI300X CDNA3) — adds the LDS
// B-tile sharing that v2 was missing.
//
// v1 (1 wave64/WG, 16x16 tile): 1046 tok/s — 80% of rocBLAS
// v2 (4 wave64/WG, 32x32 tile, NO LDS): 1009 tok/s — 77% of rocBLAS
// v3 (4 wave64/WG, 32x32 tile, LDS B-chunk): targets 1300+ tok/s
//
// Critical fix: v2 had 4 waves each independently loading B from HBM
// (4x redundancy on the 32-col tile). v3 cooperatively loads a 32-col x
// K_CHUNK=64 B-tile into LDS once per chunk, then all 4 waves' MFMA
// calls in that chunk read from LDS (which has ~10x bandwidth advantage
// over HBM and zero latency).
//
// Per WG memory cost:
//   v2: B HBM reads = 4 waves × 64 lanes × 4 fp16 = 2 KB per MFMA-K=16 step
//   v3: B HBM reads = 32 cols × 64 K × 2 bytes / (K_CHUNK/16) = 1 KB per step
//   ↳ 2x HBM B-traffic reduction (cooperative load amortizes across 4
//     MFMA iter that share the chunk)
//
// LDS budget: 32 × 64 × 2 = 4096 bytes per WG — trivial (gfx942 has 64 KB).

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    vfloat4 __attribute__((ext_vector_type(4)));

#define K_CHUNK 128
#define N_TILE  32

extern "C" __global__ __launch_bounds__(256)
void gemm_hfq4g256_residual_mfma_v3_gfx942(
    const char*  __restrict__ A,
    const half_t* __restrict__ X,
    float*       __restrict__ Y,
    int M, int K, int batch_size
) {
    __shared__ half_t lds_b[N_TILE * K_CHUNK];   // 32 × 64 = 2048 fp16 = 4 KB

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..3
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;
    const int strip     = lane_in_w >> 4;

    const int wave_m_off = (wave_id & 2) ? 16 : 0;
    const int wave_n_off = (wave_id & 1) ? 16 : 0;
    const int m_offset = blockIdx.x * 32 + wave_m_off;
    const int b_offset_global = blockIdx.y * 32;
    const int b_offset = b_offset_global + wave_n_off;

    const int my_m = m_offset + n_lane;
    if (my_m >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)my_m * groups_per_row * 136;

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // K outer loop in chunks of K_CHUNK=64 (= 4 MFMA-K=16 iterations).
    for (int kc = 0; kc < K; kc += K_CHUNK) {
        // ─── Cooperative LDS load of B chunk [32 cols × 64 K] ───
        // 256 threads load 2048 fp16 = 8 elements per thread.
        __syncthreads();
        #pragma unroll
        for (int li = 0; li < 8; li++) {
            int idx = tid + li * 256;
            int b_col = idx / K_CHUNK;             // 0..31
            int b_k   = idx % K_CHUNK;             // 0..63
            int gb = b_offset_global + b_col;
            int gk = kc + b_k;
            half_t v = (gb < batch_size && gk < K)
                ? X[(long long)gb * K + gk]
                : (half_t)0;
            lds_b[b_col * K_CHUNK + b_k] = v;
        }
        __syncthreads();

        // ─── MFMA loop over chunk in 16-element K steps ───
        #pragma unroll
        for (int kk = 0; kk < K_CHUNK; kk += 16) {
            int gk_start = kc + kk;
            int g  = gk_start / 256;
            int ki = gk_start % 256;

            const char* gp = row_ptr + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            half_t sc_h = (half_t)sc, zp_h = (half_t)zp;

            const unsigned short pk2 = *(const unsigned short*)(gp + 8 + ki/2 + strip*2);
            const unsigned int n0 = (pk2 >>  0) & 0xFu;
            const unsigned int n1 = (pk2 >>  4) & 0xFu;
            const unsigned int n2 = (pk2 >>  8) & 0xFu;
            const unsigned int n3 = (pk2 >> 12) & 0xFu;

            half4 a_reg;
            a_reg[0] = sc_h * (half_t)n0 + zp_h;
            a_reg[1] = sc_h * (half_t)n1 + zp_h;
            a_reg[2] = sc_h * (half_t)n2 + zp_h;
            a_reg[3] = sc_h * (half_t)n3 + zp_h;

            // Read B from LDS:
            //   col_in_tile = wave_n_off + n_lane (0..31)
            //   k_in_chunk = kk + strip*4 + r (r = 0..3)
            int b_col_in_tile = wave_n_off + n_lane;
            int k_in_chunk_base = kk + strip * 4;
            half4 b_reg;
            b_reg[0] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 0];
            b_reg[1] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 1];
            b_reg[2] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 2];
            b_reg[3] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 3];

            c_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_acc, 0, 0, 0);
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            Y[(long long)gb * M + gm] += c_acc[r];
        }
    }
}
