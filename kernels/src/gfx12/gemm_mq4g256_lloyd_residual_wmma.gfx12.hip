#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) variant of MQ4-Lloyd-G256 residual WMMA (Y += A·X).
// Sister of gemm_mq4g256_lloyd_residual_wmma.hip; same per-row LDS-staged
// codebook (512 B fp16) + same group stride (160 B). Mirrors gfx12's
// half8_t lane-split / K4-unroll / contiguous-rows output mapping from
// gemm_mq3g256_lloyd_residual_wmma.gfx12.hip and gemm_hfq4g256_residual_wmma.gfx12.hip.
//
// **K4 vs gfx11's K2 — why the difference**: gfx12 splits its 32-thread wave
// into a (m_lane=tid&15, k_grp=tid>>4) lane-group layout so each lane only
// reads half the K-tile per iteration. K4 unroll spreads the same total
// K-coverage across the same wall time as gfx11's K2 (which uses the full
// tile per lane). The K-unroll is hardware-shape driven, not a perf knob —
// porting K4 to gfx11 wouldn't help because gfx11's full-tile-per-lane
// mapping doesn't have the lane-group split to amortize over.
//
// **gfx12 variant: code-complete, runtime-unvalidated locally** — the bench
// host has gfx1100/gfx1151 only. RDNA4 reviewers please run the parity
// test on this kernel before relying on it (see plan §"Risks D2").

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
// C symbol kept unsuffixed so the dispatch arm's unsuffixed
// ensure_kernel lookup resolves on both archs. Per-arch cache
// keys are still distinct via the *_for_arch module suffix.
extern "C" __global__ void gemm_mq4g256_lloyd_residual_wmma(
    const char*    __restrict__ A,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int safe_batch = (batch_start + m_lane < batch_size) ? (batch_start + m_lane) : 0;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + safe_row * row_stride;
    const _Float16* x_base = X + (long long)safe_batch * K;

    // Cooperative-load mapping: tid 0..31 → 8 fp16 reads each
    // = 256 entries (16 row-codebooks × 16 entries each).
    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 8;
    const int safe_load_row = (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);
    const char* load_row_base = A + (long long)safe_load_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];  // 512 B

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    #define UNPACK_CHUNK_DQ(pk)                                                   \
        do {                                                                      \
            unsigned int _p = (pk);                                               \
            a_reg[0] = cb_lds[cb_base + ((_p >>  0) & 0xFu)];                     \
            a_reg[1] = cb_lds[cb_base + ((_p >>  4) & 0xFu)];                     \
            a_reg[2] = cb_lds[cb_base + ((_p >>  8) & 0xFu)];                     \
            a_reg[3] = cb_lds[cb_base + ((_p >> 12) & 0xFu)];                     \
            a_reg[4] = cb_lds[cb_base + ((_p >> 16) & 0xFu)];                     \
            a_reg[5] = cb_lds[cb_base + ((_p >> 20) & 0xFu)];                     \
            a_reg[6] = cb_lds[cb_base + ((_p >> 24) & 0xFu)];                     \
            a_reg[7] = cb_lds[cb_base + ((_p >> 28) & 0xFu)];                     \
        } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load for group g.
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const _Float16* xg = x_base + g * 256;
        const char* dp = gp + 32;  // skip 32 B codebook header
        const int cb_base = m_lane * 16;

        // K4 unroll: 4 WMMAs per kt iteration. Each lane-group reads
        // 4 bytes (8 nibbles) per K-tile via uint32; lane-split by k_grp.
        for (int kt = 0; kt < 16; kt += 4) {
            unsigned int pka = *(const unsigned int*)(dp + (kt + 0) * 8 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(dp + (kt + 1) * 8 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(dp + (kt + 2) * 8 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(dp + (kt + 3) * 8 + k_grp * 4);

            half8_t b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);

            half8_t a_reg;
            UNPACK_CHUNK_DQ(pka);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            UNPACK_CHUNK_DQ(pkb);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            UNPACK_CHUNK_DQ(pkc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            UNPACK_CHUNK_DQ(pkd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        __syncthreads();
    }
    #undef UNPACK_CHUNK_DQ

    // gfx12 wave32 WMMA C-mapping: contiguous rows per lane-group.
    const int out_col = batch_start + m_lane;
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
