// ALU-only Phase 1.0 spike: scalar wave32 FA on PRE-DEQUANTIZED fp16 K/V.
//
// Compared to attention_flash_asym2_tile_batched, this kernel strips:
//   - asym2 4-byte cnorm + 2-bit nibble dequant on K
//   - Q8_0 fp16 scale + int8 dequant on V
//   - Givens-2x2 rotation on Q
// Everything else (BLOCK_K = head_dim/4 per thread, tile-local FA-2 softmax,
// partial-write contract) matches the production tile kernel one-to-one so
// the spike's A/B against fa_wmma_fp16 is an apples-to-apples test of WMMA
// vs scalar ALU. Dequant overhead is excluded from BOTH baselines.
//
// Grid: [n_heads, max_tiles, batch].  Block: [32, 1, 1].
// LDS: tile_size floats for scores.
// Contract: writes per-(bid, head, kv_tile) partial tile of (2 + head_dim) floats
// — same layout consumed by attention_flash_asym_reduce_batched.
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void fa_scalar_fp16(
    const float* __restrict__ q,                  // [batch × n_heads × head_dim] fp32
    const _Float16* __restrict__ k_cache,         // [seq × n_kv_heads × head_dim] fp16, contiguous
    const _Float16* __restrict__ v_cache,         // [seq × n_kv_heads × head_dim] fp16, contiguous
    float* __restrict__ partials,                 // [batch × n_heads × max_tiles × (2+head_dim)]
    const int* __restrict__ positions,            // [batch]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int seq_len = positions[local_bid] + 1;
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    // Each thread owns 4 dims for hd=128 (or 8 for hd=256). For the spike,
    // BLOCK_K = head_dim / 32 dims per thread.
    const int dims_per_thread = head_dim / 32;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // ── Load Q for this (bid, head) into registers ──
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    float mq[8] = {0};   // up to 8 dims per thread for hd=256
    #pragma unroll
    for (int i = 0; i < dims_per_thread; i++) {
        mq[i] = q_head[tid * dims_per_thread + i];
    }

    // ═══ Phase A: Q·K dot products ═══
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const _Float16* kb = k_cache + (size_t)t * kv_dim + kv_h * head_dim;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < dims_per_thread; i++) {
            partial += mq[i] * (float) kb[tid * dims_per_thread + i];
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // ═══ Phase B: tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    // ═══ Phase D: V-weighted accumulation (NORMAL space fp16 V) ═══
    float out_vec[8] = {0};
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const _Float16* vb = v_cache + (size_t)t * kv_dim + kv_h * head_dim;
        #pragma unroll
        for (int i = 0; i < dims_per_thread; i++) {
            out_vec[i] += w * (float) vb[tid * dims_per_thread + i];
        }
    }

    // ═══ Write tile partials ═══
    const int stride = 2 + head_dim;
    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride
                        + tile_id * stride;
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < dims_per_thread; i++) {
        p[2 + tid * dims_per_thread + i] = out_vec[i];
    }
}
