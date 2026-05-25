// Shared body for HFQ4-G256 wave32 MMQ residual kernels.
//
// Direct sibling of `gemm_hfq3g256_residual_mmq_body.cuh`. Same topology
// (LDS-tiled X reuse + sdot4 inner loop + Q8_1 X quantization); differs
// only in the HFQ4-specific unpack:
//
//   group stride = 136 B  (vs 104 B for HFQ3)
//   body         = 128 B  (4-bit nibbles in 4-byte uints, vs 3-bit trits in 3-byte words)
//   zp eff       = zp + 8*sc  (4-bit center = 8, vs 4 for 3-bit)
//   window stride = window * 64 + chunk * 4  (vs window * 48 + chunk * 3)
//   nibble extract = (qs0 >> (i*4)) & 0xF   (vs 3-bit shifts)
//   signed bias = n - 8   (vs n - 4)
//
// Each tile-size instantiation defines MMQ_X (kernel-side batch tile)
// and KERNEL_NAME before #including this header. MMQ_Y defaults to 128;
// override with `#define MMQ_Y N` BEFORE the include to cut LDS budget
// and raise CU occupancy (MQ3 phase-2 finding: y=64 wins for narrow
// output kernels, y=32 regresses).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef MMQ_Y
#define MMQ_Y 128
#endif
#define MMQ_NWARPS 4
#define WAVE_SIZE 32
#define QK8_1 32
#define X_STRIDE 40
#define Y_STRIDE 36
#define X_LOADER_TASKS_PER_THREAD ((MMQ_Y * 16) / 128)

#ifndef MMQ_X
#error "MMQ_X must be defined before #including this body"
#endif
#ifndef KERNEL_NAME
#error "KERNEL_NAME must be defined before #including this body"
#endif

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

__launch_bounds__(128, 2)
extern "C" __global__ void KERNEL_NAME(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;

    extern __shared__ int smem[];
    int*    x_qs   = smem;
    float2* x_dm   = (float2*)(x_qs + MMQ_Y * X_STRIDE);
    int*    tile_y = (int*)(x_dm + MMQ_Y);

    float sum[(MMQ_X / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        #pragma unroll 1
        for (int window = 0; window < 2; ++window) {
            const int kb = 2 * kg + window;

            // ── Load Q8_1 Y tile ──────────────────────────────────────────
            const int total_y_ints = MMQ_X * Y_STRIDE;
            for (int u = tid; u < total_y_ints; u += 128) {
                const int j = u / Y_STRIDE;
                const int slot = u % Y_STRIDE;
                const bool valid = (col0 + j) < N;
                const int col = valid ? (col0 + j) : (N - 1);
                const int* src = (const int*)(Xq + (long long)kb * N + col);
                tile_y[u] = valid ? src[slot] : 0;
            }

            // ── Load X tile metadata (HFQ4 header) ────────────────────────
            // One thread per output row writes one float2 to x_dm[i] (sized
            // MMQ_Y). Gating on `tid < MMQ_Y` (NOT the workgroup width 128)
            // is required: when MMQ_Y=64, threads 64..127 must NOT write,
            // since `x_dm[64..127]` overlaps the adjacent `tile_y` LDS
            // region (see allocator at line 64). Without this gate, the
            // y64 variant corrupts the Q8_1 Y tile and emits NaN.
            if (window == 0 && tid < MMQ_Y) {
                const int i = tid;
                const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 136;
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                x_dm[i] = make_float2(sc, zp + 8.0f * sc);
            }

            // X-tile loader (HFQ4 unpack: 4-bit nibbles → INT8 packed).
            // Scales with MMQ_Y; at MMQ_Y=128 → 16 loops/thread,
            // MMQ_Y=64 → 8 loops/thread.
            #pragma unroll
            for (int loop = 0; loop < X_LOADER_TASKS_PER_THREAD; ++loop) {
                const int task_id = tid * X_LOADER_TASKS_PER_THREAD + loop;
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
