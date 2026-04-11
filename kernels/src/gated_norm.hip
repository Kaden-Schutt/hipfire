#include <hip/hip_runtime.h>
// Grid: [n_heads, batch, 1]. For decode grid.y=1 and the offset below is 0
// (byte-identical to pre-batching). For prefill grid.y=N; x/z/out are laid
// out as [N × (n_heads * head_dim)] row-major.
extern "C" __global__ void gated_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n_heads, int head_dim, float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    const float* xh = x + batch_off + h * head_dim;
    const float* zh = z + batch_off + h * head_dim;
    float* oh = out + batch_off + h * head_dim;

    float sq_sum = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += 32) {
        float normed = xh[i] * inv_rms * weight[i];
        float z_val = zh[i];
        float silu_z = z_val / (1.0f + expf(-z_val));
        oh[i] = normed * silu_z;
    }
}
