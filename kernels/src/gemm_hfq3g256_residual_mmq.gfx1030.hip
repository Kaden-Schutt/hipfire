#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// HFQ3-G256 × Q8_1 dp4a MMQ residual kernel — wave32 RDNA2.
//
// HFQ3 sibling of gemm_hfq4g256_residual_mmq.gfx1030.hip. Same MMQ_Y/mmq_x
// topology and LDS streaming pipeline; only the X tile loader differs:
//   - 104 B/group (vs 136 for HFQ4)
//   - 96 B body (vs 128 for HFQ4) — 8 trits per 3-byte group per thread
//   - Signed mapping `trit - 4` → [-4, 3] for sdot4
//   - zp_eff = zp + 4*sc (vs zp + 8*sc for HFQ4)
//   - uint24 byte-combine (OOB-safe) for trit reads
//
// Phase 3 minimal probe (docs/plans/gfx10_mq3_prefill.md). Co-test with
// HFQ4 MMQ sibling: if HFQ4 MMQ beats HFQ4 fp16 but HFQ3 MMQ loses to
// HFQ3 dot2, the 3-bit unpack overhead is the smoking gun. If both win
// or both lose, the LDS-tiling pattern is the dominant factor.

#define MMQ_Y 128
#define MMQ_X 32
#define MMQ_NWARPS 4
#define WAVE_SIZE 32
#define QK8_1 32
#define X_STRIDE 40
#define Y_STRIDE 36

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_hfq3g256_residual_mmq(
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
        // HFQ3 group spans 256 K-elements = 2 Q8_1 kblocks of 128 K each.
        // Each kg iter processes ONE 104-B HFQ3 group split into 2 windows.
        // Window 0 → kblock 2*kg+0 (first 128 K) → bytes [0..47] of HFQ3 body
        // Window 1 → kblock 2*kg+1 (second 128 K) → bytes [48..95] of HFQ3 body
        #pragma unroll 1
        for (int window = 0; window < 2; ++window) {
            const int kb = 2 * kg + window;

            // ── Load Q8_1 Y tile (one 128-K block) ────────────────────────
            const int total_y_ints = MMQ_X * Y_STRIDE;
            for (int u = tid; u < total_y_ints; u += 128) {
                const int j = u / Y_STRIDE;
                const int slot = u % Y_STRIDE;
                const bool valid = (col0 + j) < N;
                const int col = valid ? (col0 + j) : (N - 1);
                const int* src = (const int*)(Xq + (long long)kb * N + col);
                tile_y[u] = valid ? src[slot] : 0;
            }

            // ── Load X tile (unpack 3-bit trits → signed INT8 packed) ─────
            if (window == 0 && tid < 128) {
                const int i = tid;
                const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 104;
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                // Signed mapping: w_unsigned = sc*trit + zp = sc*(trit_signed+4) + zp
                //                            = sc*trit_signed + (zp + 4*sc)
                x_dm[i] = make_float2(sc, zp + 4.0f * sc);
            }

            // 128 rows × 16 ints/row in LDS = 2048 ints. But each window
            // covers 128 K-elements = 16 ints worth of LDS slots per row.
            // Each row's HFQ3 body for THIS window is 48 bytes (window 0
            // = bytes [0..47], window 1 = bytes [48..95]). 16 thread-loads
            // per row × 3 bytes per load = 48 bytes per row. Matches.
            //
            // 128 rows × 16 loads/row = 2048 task IDs / 128 threads = 16 each.
            #pragma unroll
            for (int loop = 0; loop < 16; ++loop) {
                const int task_id = tid * 16 + loop;
                const int i = task_id / 16;       // row 0..127
                const int chunk = task_id % 16;   // 0..15 within row

                const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
                const char* gp = A + ((long long)row * groups_per_row + kg) * 104;

                // chunk maps to 8 trits within the window. Each chunk loads 3
                // bytes (= 8 trits = 1 uint24). The window's body byte offset:
                // window 0: byte 8 + chunk*3       (chunks 0..15 → bytes 8..53)
                // window 1: byte 8 + 48 + chunk*3  (chunks 0..15 → bytes 56..101)
                // Body ends at byte 103 (= 8 + 96). Last chunk in window 1
                // would read bytes 101..103, which is in-bounds.
                const unsigned char* d = (const unsigned char*)(gp + 8 + window * 48 + chunk * 3);
                const unsigned int pk = (unsigned int)d[0]
                                      | ((unsigned int)d[1] << 8)
                                      | ((unsigned int)d[2] << 16);

                const unsigned int n0 = (pk      ) & 7u;
                const unsigned int n1 = (pk >>  3) & 7u;
                const unsigned int n2 = (pk >>  6) & 7u;
                const unsigned int n3 = (pk >>  9) & 7u;
                const unsigned int n4 = (pk >> 12) & 7u;
                const unsigned int n5 = (pk >> 15) & 7u;
                const unsigned int n6 = (pk >> 18) & 7u;
                const unsigned int n7 = (pk >> 21) & 7u;
                const int int_a = (int)(((n0 - 4) & 0xFF) | (((n1 - 4) & 0xFF) << 8)
                                      | (((n2 - 4) & 0xFF) << 16) | (((n3 - 4) & 0xFF) << 24));
                const int int_b = (int)(((n4 - 4) & 0xFF) | (((n5 - 4) & 0xFF) << 8)
                                      | (((n6 - 4) & 0xFF) << 16) | (((n7 - 4) & 0xFF) << 24));

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

    // Write-back with residual add.
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
