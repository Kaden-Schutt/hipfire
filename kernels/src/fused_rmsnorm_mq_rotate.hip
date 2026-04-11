#include <hip/hip_runtime.h>

// Fused RMSNorm + FWHT rotation for MagnumQuant inputs.
// Replaces the rmsnorm_f32 + mq_rotate_x kernel pair with a single launch.
//
// Layout:
// - 1 workgroup of 256 threads = 8 waves of 32 threads *per batch element*.
// - Grid.x is the batch dimension — grid=[1,1,1] for decode, grid=[N,1,1]
//   for batched prefill. Each block picks row blockIdx.x out of the [N × K]
//   x and x_rot tensors. LDS is per-block, so batches don't interfere.
// - The whole K-length input vector is processed within that workgroup.
// - LDS holds `x_shared[K] + reduce[256]` (K<=12288 fits in 64 KB).
//
// Phase 1 — RMSNorm pre-pass:
//   Each thread strides through x, caching values into x_shared[] and
//   accumulating local sum-of-squares. A block reduction in `reduce[]` gives
//   the global sum; rsqrt gives rms. Each thread then multiplies its cached
//   values by weight[i] * rms in place inside x_shared[].
//
// Phase 2 — FWHT rotation per 256-element group:
//   Each of the 8 waves handles `groups_per_warp` groups serially. Within a
//   wave, lane_id selects 8 elements per group, multiplied by signs1, then
//   the local 3-stage butterfly (strides 1/2/4) plus the wave-level
//   ds_swizzle butterfly (strides 1-16). Result scaled by 1/16 and signs2,
//   then written to x_rot[].
//
// Correctness vs the split version:
//   Byte-exact for normalization + rotation modulo float reordering (the
//   reduction order for sum-of-squares is identical to rmsnorm_f32, and the
//   butterfly order matches mq_rotate_x exactly).

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float* x_shared = smem;            // [K]
    float* reduce   = smem + K;        // [256]

    const int tid = threadIdx.x;
    const int block_size = blockDim.x; // 256
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;      // 0..7
    const int num_warps = block_size >> 5;

    // Batch row: this block processes x[blockIdx.x * K .. (blockIdx.x+1) * K].
    // For decode (grid.x == 1), offset is 0 and the behaviour is identical
    // to the pre-batching version. weight/signs1/signs2 are shared across
    // the batch (same per-feature scalars for every token).
    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b     = x     + batch_off;
          float* x_rot_b = x_rot + batch_off;

    // Phase 1a: gather → cache to LDS and accumulate sum-of-squares.
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float v = x_b[i];
        x_shared[i] = v;
        local_sum += v * v;
    }
    reduce[tid] = local_sum;
    __syncthreads();

    // Phase 1b: block reduction → RMS.
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(reduce[0] / (float)K + eps);

    // Phase 1c: normalize in place. x_shared now holds rmsnormalized values.
    for (int i = tid; i < K; i += block_size) {
        x_shared[i] = x_shared[i] * weight[i] * rms;
    }
    __syncthreads();

    // Phase 2: FWHT per group, 256 elements at a time. Each wave owns
    // `groups_per_warp` groups and processes them serially. Register usage per
    // thread: 8 floats for v0..v7 + a handful of scalars, well under budget.
    const int groups_total = K / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;

    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int d0 = lane_id * 8;
        const int base = group * 256 + d0;

        float v0 = x_shared[base]     * signs1[d0];
        float v1 = x_shared[base + 1] * signs1[d0 + 1];
        float v2 = x_shared[base + 2] * signs1[d0 + 2];
        float v3 = x_shared[base + 3] * signs1[d0 + 3];
        float v4 = x_shared[base + 4] * signs1[d0 + 4];
        float v5 = x_shared[base + 5] * signs1[d0 + 5];
        float v6 = x_shared[base + 6] * signs1[d0 + 6];
        float v7 = x_shared[base + 7] * signs1[d0 + 7];

        // Local butterfly: strides 1, 2, 4 (within each thread's 8 registers).
        float t;
        t = v0; v0 = v0 + v1; v1 = t - v1;
        t = v2; v2 = v2 + v3; v3 = t - v3;
        t = v4; v4 = v4 + v5; v5 = t - v5;
        t = v6; v6 = v6 + v7; v7 = t - v7;

        t = v0; v0 = v0 + v2; v2 = t - v2;
        t = v1; v1 = v1 + v3; v3 = t - v3;
        t = v4; v4 = v4 + v6; v6 = t - v6;
        t = v5; v5 = v5 + v7; v7 = t - v7;

        t = v0; v0 = v0 + v4; v4 = t - v4;
        t = v1; v1 = v1 + v5; v5 = t - v5;
        t = v2; v2 = v2 + v6; v6 = t - v6;
        t = v3; v3 = v3 + v7; v7 = t - v7;

        // Wave butterfly via ds_swizzle. ds_swizzle is wave-local, so each of
        // the 8 warps applies its own butterfly independently within its 32
        // lanes. `lane_id & (str)` selects which side of the butterfly each
        // thread writes.
        #define HBFLY(v, pat, str) do { \
            float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
            if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
        } while (0)
        #define HBFLY8(pat, str) \
            HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
            HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

        HBFLY8(0x041F, 1);
        HBFLY8(0x081F, 2);
        HBFLY8(0x101F, 4);
        HBFLY8(0x201F, 8);
        HBFLY8(0x401F, 16);
        #undef HBFLY8
        #undef HBFLY

        const float s = 0.0625f;
        x_rot_b[base]     = v0 * s * signs2[d0];
        x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
        x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
        x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
        x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
        x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
        x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
        x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
    }
}
