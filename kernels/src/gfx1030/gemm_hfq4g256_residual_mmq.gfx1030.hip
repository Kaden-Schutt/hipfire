#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// HFQ4-G256 × Q8_1 dp4a MMQ residual kernel — wave32 port for RDNA2.
//
// Direct port of gemm_hfq4g256_residual_mmq_gfx906_body.cuh's Option C
// streaming design, trimmed for minimal-probe scope:
//   - mmq_x = 32 only (single tile size, drops the 8-way template family)
//   - need_check always true (no _full_* M%128 alignment fast path)
//   - b128 LDS read path only (mmq_x=32 hits the cliff)
//   - residual add mode only (Y += A·X)
//
// Topology (wave32 native, gfx1030+):
//   block_dim = (32, 4, 1) = 128 threads = 4 wave32 warps
//   MMQ_Y = 128 rows per workgroup
//   mmq_x = 32 batch elements per workgroup
//   per-thread accumulators: (MMQ_Y/32) × (mmq_x/4) = 4 × 8 = 32 floats
//
// Per-window pipeline (4 syncs/group, matches gfx906 Option C):
//   1. Load 128-K of x_qs (unpacked nibbles → INT8) + one Q8_1 tile_y block
//   2. __syncthreads
//   3. 4 sub-blocks computed back-to-back (no syncs between)
//   4. __syncthreads
//
// LDS layout (must match dispatcher's shared_mem calculation):
//   x_qs   : 128 rows × 40 ints  = 20480 B
//   x_dm   : 128 float2          =  1024 B
//   tile_y :  32 cols × 36 ints  =  4608 B
//   total                        = 26112 B (≤ 64 KiB cap, 2 WGs/CU)
//
// Phase 3 minimal probe (docs/plans/gfx10_mq3_prefill.md). Goal: test
// whether the gfx906 MMQ design transfers to RDNA2 wave32 for HFQ4.

#define MMQ_Y 128
#define MMQ_X 32
#define MMQ_NWARPS 4
#define WAVE_SIZE 32
#define QK8_1 32
#define X_STRIDE 40  // mmq_x ≥ 32 → b128 path requires stride 40 (16-B aligned)
#define Y_STRIDE 36

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_hfq4g256_residual_mmq_rdna2(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;  // 0..127

    extern __shared__ int smem[];
    int*    x_qs   = smem;
    float2* x_dm   = (float2*)(x_qs + MMQ_Y * X_STRIDE);
    int*    tile_y = (int*)(x_dm + MMQ_Y);

    // 32 output cells per thread: (mmq_x/4) × (MMQ_Y/WAVE_SIZE) = 8 × 4
    float sum[(MMQ_X / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        // Two windows × 4 sub-blocks each.
        #pragma unroll 1
        for (int window = 0; window < 2; ++window) {
            // ── Load Q8_1 Y tile (one 128-K block) ────────────────────────
            const int kb = 2 * kg + window;
            const int total_y_ints = MMQ_X * Y_STRIDE;
            #pragma unroll
            for (int u = tid; u < total_y_ints; u += 128) {
                const int j = u / Y_STRIDE;
                const int slot = u % Y_STRIDE;
                const bool valid = (col0 + j) < N;
                const int col = valid ? (col0 + j) : (N - 1);
                const int* src = (const int*)(Xq + (long long)kb * N + col);
                tile_y[u] = valid ? src[slot] : 0;
            }

            // ── Load X tile (unpack nibbles → INT8 packed) ────────────────
            if (window == 0 && tid < 128) {
                const int i = tid;
                const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                x_dm[i] = make_float2(sc, zp + 8.0f * sc);
            }

            // 128 rows × 16 uints/row = 2048 uint reads / 128 threads = 16 uints/thread.
            #pragma unroll
            for (int loop = 0; loop < 16; ++loop) {
                const int task_id = tid * 16 + loop;  // 0..2047
                const int i = task_id / 16;            // row 0..127
                const int chunk = task_id % 16;        // uint within row 0..15

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

                x_qs[i * X_STRIDE + 2 * chunk + 0] = int_a;
                x_qs[i * X_STRIDE + 2 * chunk + 1] = int_b;
            }

            __syncthreads();

            // ── 4 sub-blocks back-to-back ──────────────────────────────────
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const int kx_start = sub * 8;
                const int ky_start = 4 + sub * 8;

                #pragma unroll
                for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_NWARPS) {
                    const int j = j0 + threadIdx.y;

                    const half2* y_ds_col = (const half2*)(tile_y + j * Y_STRIDE);
                    const half2 ds_j = y_ds_col[sub];

                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
                        const int i = i0 + threadIdx.x;

                        // b128 path: 8 ints per operand as 2× int4.
                        const int4 x_v0 = *(const int4*)&x_qs[i * X_STRIDE + kx_start + 0];
                        const int4 x_v1 = *(const int4*)&x_qs[i * X_STRIDE + kx_start + 4];
                        const int4 y_v0 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 0];
                        const int4 y_v1 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 4];

                        int sumi = 0;
                        sumi = __builtin_amdgcn_sdot4(x_v0.x, y_v0.x, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v0.y, y_v0.y, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v0.z, y_v0.z, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v0.w, y_v0.w, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.x, y_v1.x, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.y, y_v1.y, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.z, y_v1.z, sumi, false);
                        sumi = __builtin_amdgcn_sdot4(x_v1.w, y_v1.w, sumi, false);

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

            __syncthreads();
        }
    }

    // ── Write-back with residual add ──────────────────────────────────────
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
            const long long out_idx = (long long)col * M + row;
            Y[out_idx] += sum[idx];
        }
    }
}
