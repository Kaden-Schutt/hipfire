// Non-causal cross-attention kernel for DFlash draft forward.
//
// Each query position attends to ALL key/value positions (non-causal,
// bidirectional). Supports GQA where n_heads > n_kv_heads. Q/K/V are
// separate buffers — K can be longer than Q. Grid: [n_heads, B, 1],
// one block per (head, query-position). Block size is chosen to cover
// max(L, head_dim). Shared mem: L scores + block_size workspace.
//
// Layout (row-major, flat):
//   q    : [B  * n_heads    * head_dim]
//   k    : [L  * n_kv_heads * head_dim]
//   v    : [L  * n_kv_heads * head_dim]
//   out  : [B  * n_heads    * head_dim]
//
// rep = n_heads / n_kv_heads (must divide evenly).

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_dflash_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int qi = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (head >= n_heads || qi >= B) return;

    int rep = n_heads / n_kv_heads;
    int kv_head = head / rep;
    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    float* scores = sdata;               // [L]
    float* ws = sdata + L;               // [blockDim.x]

    const float* q_ptr = q + qi * q_stride + head * head_dim;

    // Phase 1: scores[j] = dot(Q[qi,head], K[j,kv_head]) * scale, tracking max.
    float local_max = -1e30f;
    for (int j = tid; j < L; j += nthreads) {
        const float* k_ptr = k + j * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        float s = dot * scale;
        scores[j] = s;
        if (s > local_max) local_max = s;
    }

    // Block-wide max reduce.
    ws[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = ws[tid];
            float b = ws[tid + s];
            ws[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float max_val = ws[0];
    __syncthreads();

    // Phase 2: exp(score - max), tracking sum.
    float local_sum = 0.0f;
    for (int j = tid; j < L; j += nthreads) {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    ws[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] += ws[tid + s];
        __syncthreads();
    }
    float sum_val = ws[0];
    __syncthreads();

    // Normalize.
    float inv_sum = 1.0f / sum_val;
    for (int j = tid; j < L; j += nthreads) {
        scores[j] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: out[qi, head, :] = sum_j scores[j] * V[j, kv_head, :]
    float* out_ptr = out + qi * q_stride + head * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int j = 0; j < L; ++j) {
            val += scores[j] * v[j * kv_stride + kv_head * head_dim + d];
        }
        out_ptr[d] = val;
    }
}
