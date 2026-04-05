#include <hip/hip_runtime.h>

// Asymmetric attention: Q8 K cache + HF4 V cache, head_dim=256.
// HF4-V dequant: min + nibble * scale (1 FMA per element, zero LUT, zero FWHT)
// Same cache access pattern as Q8: contiguous header + data per head.
//
// Grid: [n_heads]. Block: [32]. Shared: scores[seq_len].
extern "C" __launch_bounds__(32, 10)
__global__ void attention_q8k_hf4v_256(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_q8,
    const unsigned char* __restrict__ v_hf4,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int seq_len_hint,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn
) {
    const int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_h = h / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int d0 = tid * 8;

    // Q8 K layout
    const int q8_bph = (head_dim / 32) * 34;
    const int q8_bpp = n_kv_heads * q8_bph;
    const int blk_byte_off = kv_h * q8_bph + (d0 / 32) * 34;
    const int blk_off = d0 % 32;

    // HF4 V layout: [f32 norm (4B)] [f16 scale (2B)] [f16 min (2B)] [128 nibbles] = 136B/head
    const int hf4_bph = 8 + head_dim / 2;  // 136
    const int hf4_bpp = n_kv_heads * hf4_bph;
    const int hf4_head_off = kv_h * hf4_bph;

    float* scores = sdata;
    const float* qh = q + h * head_dim + d0;

    // ── Phase 1: Q @ K^T — Q8 dequant (identical to asym kernel) ──
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* blk = k_q8 + (size_t)t * q8_bpp + blk_byte_off;
        float bs = (float)(*((const _Float16*)blk));
        const signed char* kv = (const signed char*)(blk + 2) + blk_off;
        float p = qh[0]*(bs*(float)kv[0]) + qh[1]*(bs*(float)kv[1])
                + qh[2]*(bs*(float)kv[2]) + qh[3]*(bs*(float)kv[3])
                + qh[4]*(bs*(float)kv[4]) + qh[5]*(bs*(float)kv[5])
                + qh[6]*(bs*(float)kv[6]) + qh[7]*(bs*(float)kv[7]);
        for (int off = 16; off > 0; off >>= 1) p += __shfl_xor(p, off);
        if (tid == 0) scores[t] = p * scale_attn;
    }
    __syncthreads();

    // ── Softmax ──
    float lmx = -1e30f;
    for (int t = tid; t < seq_len; t += 32) lmx = fmaxf(lmx, scores[t]);
    for (int off = 16; off > 0; off >>= 1) lmx = fmaxf(lmx, __shfl_xor(lmx, off));
    float lsm = 0.0f;
    for (int t = tid; t < seq_len; t += 32) {
        float e = expf(scores[t] - lmx); scores[t] = e; lsm += e;
    }
    for (int off = 16; off > 0; off >>= 1) lsm += __shfl_xor(lsm, off);
    __syncthreads();
    float is = 1.0f / lsm;
    for (int t = tid; t < seq_len; t += 32) scores[t] *= is;
    __syncthreads();

    // ── Phase 2: V weighted sum — HF4 dequant (1 FMA per element) ──
    // dequant: norm * (min + nibble * scale) = norm*min + nibble * norm*scale
    // Precompute nothing — just fma in the loop.
    float m0=0, m1=0, m2=0, m3=0, m4=0, m5=0, m6=0, m7=0;
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;  // sparse V skip
        const unsigned char* vb = v_hf4 + (size_t)t * hf4_bpp + hf4_head_off;

        // Header: [f32 norm] [f16 scale] [f16 min]
        float vnorm = *(const float*)vb;
        float vscale = (float)(*(const _Float16*)(vb + 4));
        float vmin = (float)(*(const _Float16*)(vb + 6));

        // Precompute: w * norm * scale, w * norm * min
        float wns = w * vnorm * vscale;
        float wnm = w * vnorm * vmin;

        // Load 4 packed nibble bytes (contiguous, cache-friendly)
        unsigned int pk = *(const unsigned int*)(vb + 8 + tid * 4);

        // Dequant + weighted accumulate: mi += w*norm*(min + nibble*scale)
        //                                   = wnm + nibble * wns
        m0 += fmaf((float)((pk)       & 0xF), wns, wnm);
        m1 += fmaf((float)((pk >> 4)  & 0xF), wns, wnm);
        m2 += fmaf((float)((pk >> 8)  & 0xF), wns, wnm);
        m3 += fmaf((float)((pk >> 12) & 0xF), wns, wnm);
        m4 += fmaf((float)((pk >> 16) & 0xF), wns, wnm);
        m5 += fmaf((float)((pk >> 20) & 0xF), wns, wnm);
        m6 += fmaf((float)((pk >> 24) & 0xF), wns, wnm);
        m7 += fmaf((float)((pk >> 28) & 0xF), wns, wnm);
    }

    // No FWHT inverse needed — HF4 stores raw values, not rotated
    float* oh = out + h * head_dim + d0;
    oh[0]=m0; oh[1]=m1; oh[2]=m2; oh[3]=m3;
    oh[4]=m4; oh[5]=m5; oh[6]=m6; oh[7]=m7;
}
