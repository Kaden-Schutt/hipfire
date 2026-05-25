#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ4-Lloyd WMMA residual GEMM with 2× batch-tile fanout per WG.
// Phase D experiment — see docs/plans/mq4-lloyd-batch-fanout-phase-d.md
// and gemm_mq4g256_lloyd_residual_wmma_mb4.hip for the structural template.
//
// Motivation: Phase D-A's mb4 (16×64 output tile) hits 23-30 GiB/s on
// large-M projections (gate_up M=28672) but lags at 16-19 GiB/s on the
// small-M residual cases (M=4096 model-dim wo + mlp.down). Diagnosis
// (post-D profile, devlog 2026-05-09): mb4 grid (M/16, batch/64) at
// M=4096 batch=256 produces only 1024 WGs across 40 CUs, giving 25
// WGs/CU. Combined with VGPR=106 forcing __launch_bounds__(32, 2) to
// 2 waves/CU, SIMDs are starved for latency hiding.
//
// mb2 trades half the per-WG weight reuse (2× vs mb4's 4×) for 2× the
// WG count and ~21 fewer VGPRs (estimated ~85 vs mb4's 106). At small
// M, the better occupancy-vs-reuse balance should win; at large M
// (gate_up), mb4 still dominates because there's already plenty of
// parallelism to mask occupancy and the 4× reuse pays off more.
//
// Grid: [ceil(M/16), ceil(batch/32)]. Block: [32]. LDS: 512 B.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256_lloyd_residual_wmma_mb2(
    const char*    __restrict__ A,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 32;  // 2 sub-tiles × 16 = 32

    if (row_start >= M || batch_start >= batch_size) return;

    const int my_tile_row = tid & 15;
    const int safe_row = (row_start + my_tile_row < M) ? (row_start + my_tile_row) : (M - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + safe_row * row_stride;

    long long safe_batch_s[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int b_idx = batch_start + 16 * s + my_tile_row;
        safe_batch_s[s] = (b_idx < batch_size) ? (long long)b_idx : 0LL;
    }

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 8;
    const int safe_load_row = (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);
    const char* load_row_base = A + (long long)safe_load_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];

    float8_t acc[2] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const char* dp = gp + 32;
        const int cb_base = my_tile_row * 16;

        const _Float16* xg_s[2];
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            xg_s[s] = X + safe_batch_s[s] * (long long)K + g * 256;
        }

        for (int kt = 0; kt < 16; kt += 2) {
            unsigned int pk0a = *(const unsigned int*)(dp + kt * 8);
            unsigned int pk1a = *(const unsigned int*)(dp + kt * 8 + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            const int k_off_a = kt * 16;
            half16_t b0a = *(const half16_t*)(xg_s[0] + k_off_a);
            half16_t b1a = *(const half16_t*)(xg_s[1] + k_off_a);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0a, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1a, acc[1]);

            unsigned int pk0b = *(const unsigned int*)(dp + (kt + 1) * 8);
            unsigned int pk1b = *(const unsigned int*)(dp + (kt + 1) * 8 + 4);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            const int k_off_b = (kt + 1) * 16;
            half16_t b0b = *(const half16_t*)(xg_s[0] + k_off_b);
            half16_t b1b = *(const half16_t*)(xg_s[1] + k_off_b);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0b, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1b, acc[1]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int out_col = batch_start + 16 * s + (tid & 15);
        if (out_col < batch_size) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] += acc[s][j];
                }
            }
        }
    }
}
