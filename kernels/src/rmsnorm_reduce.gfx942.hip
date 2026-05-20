#include <hip/hip_runtime.h>

// Kernel A of the two-kernel fused_rmsnorm_mq_rotate split for gfx942.
//
// Computes rms = rsqrt(sum(x²) / K + eps) per batch element.
// Pairs with rotate_with_rms_gfx942 (Kernel B) which consumes rms_out
// and emits the normalized + FWHT-rotated output.
//
// Geometry:
//   block = [1024, 1, 1]   = 16 wave64s (max occupancy for reduce)
//   grid  = [batch, 1, 1]
//   LDS   = 1024 floats reduce buffer (4 KB)
//
// Why split: the single-kernel fused version's grid was [batch, 1, 1]
// (one WG per batch element). At decode batch=1 that's 4 wave64s on a
// 304-CU GPU. The expensive Phase 2 (FWHT per group, K/256 groups per
// batch) is trivially parallelizable across groups; splitting into A
// (reduce) + B (per-group rotate) unlocks K/256 × batch WGs in B.
//
// Math: byte-identical to fused_rmsnorm_mq_rotate's Phase 1a-1b
// (same lane-stride sum-of-squares, same binary-tree reduction order).

extern "C" __global__ __launch_bounds__(1024)
void rmsnorm_reduce_gfx942(
    const float* __restrict__ x,     // [batch, K]
    float* __restrict__ rms_out,     // [batch] — one f32 per batch element
    int K,
    float eps
) {
    __shared__ float lds[1024];

    const int tid = threadIdx.x;
    const int batch = blockIdx.x;
    const float* x_b = x + (long long)batch * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += 1024) {
        float v = x_b[i];
        sum += v * v;
    }

    lds[tid] = sum;
    __syncthreads();

    // Binary-tree reduction in LDS. 1024 → 512 → ... → 1 in 10 stages.
    for (int s = 512; s > 0; s >>= 1) {
        if (tid < s) lds[tid] += lds[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        rms_out[batch] = rsqrtf(lds[0] / (float)K + eps);
    }
}
