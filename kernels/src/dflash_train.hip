// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// DFlash trainer fp32 forward primitives (spike): a clean, self-consistent
// [tokens, features] row-major dense linear, and a naive GQA block-attention
// forward with target-context KV-injection. Both are deliberately simple
// (correctness-first, small block sizes) so the hand-coded backward
// (step 4) is a direct transpose/recompute of these — no fused flash kernel.
//
// Conventions (held EVERYWHERE in the trainer to avoid transpose bugs):
//   - activations X: [M tokens, K features] row-major  -> X[m*K + k]
//   - weight     W: [N out, K in] row-major (PyTorch nn.Linear.weight)
//   - linear out Y: [M tokens, N out] row-major  -> Y[m*N + n]   (NOT transposed)

#include <hip/hip_runtime.h>

// Y[M,N] = X[M,K] · W[N,K]^T   (one warp per output element, K-reduction).
extern "C" __global__ void linear_fwd_f32(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y,
    int M, int K, int N)
{
    int m = blockIdx.x;
    int n = blockIdx.y;
    if (m >= M || n >= N) return;
    int lane = threadIdx.x;            // 0..31
    const float* xr = X + (size_t)m * K;
    const float* wr = W + (size_t)n * K;
    float s = 0.0f;
    for (int k = lane; k < K; k += 32) s += xr[k] * wr[k];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down(s, off);
    if (lane == 0) Y[(size_t)m * N + n] = s;
}

// GQA block attention forward with injected context.
//   Q: [B, nH, hd]     queries (block positions)
//   K: [L, nKV, hd]    keys   = concat([ctx_K (n_ctx) ; block_K (B)]) along axis 0
//   V: [L, nKV, hd]    values = concat([ctx_V ; block_V])
//   O: [B, nH, hd]     output  (same layout as Q)
//   L = n_ctx + B. Full (bidirectional) attention over all L keys — no causal
//   mask: the diffusion block attends within itself and to all context.
//   RoPE + per-head qk-RMSNorm are applied to Q/K BEFORE this kernel.
// One warp (32 lanes) per (query i, head h); online (flash-style) softmax.
extern "C" __global__ void attn_block_ctx_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int L, int nH, int nKV, int hd, float scale)
{
    int qh = blockIdx.x;
    if (qh >= B * nH) return;
    int i = qh / nH;                  // query index
    int h = qh % nH;                  // attention head
    int group = nH / nKV;
    int g = h / group;                // kv head
    int lane = threadIdx.x;           // 0..31
    // hd <= 128  => up to 4 head-dim elements per lane.
    float q[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) { int d = lane + e * 32; q[e] = (d < hd) ? Q[((size_t)(i * nH + h)) * hd + d] : 0.0f; }
    float acc[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) acc[e] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < L; j++) {
        const float* kb = K + ((size_t)(j * nKV + g)) * hd;
        float partial = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; e++) { int d = lane + e * 32; if (d < hd) partial += q[e] * kb[d]; }
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);  // all-lane sum
        float score = partial * scale;
        float new_m = fmaxf(m, score);
        float corr = __expf(m - new_m);
        float p = __expf(score - new_m);
        l = l * corr + p;
        const float* vb = V + ((size_t)(j * nKV + g)) * hd;
        #pragma unroll
        for (int e = 0; e < 4; e++) { int d = lane + e * 32; acc[e] = acc[e] * corr + (d < hd ? p * vb[d] : 0.0f); }
        m = new_m;
    }
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; e++) { int d = lane + e * 32; if (d < hd) O[((size_t)(i * nH + h)) * hd + d] = acc[e] * inv; }
}

// ===================================================================
// DFlash trainer fp32 BACKWARD primitives (step 4).
// Each is the exact reverse-mode adjoint of the matching forward op.
// Conventions identical to the forward section above.
// ===================================================================

// linear backward — dX = dY · W.   dX[m,k] = Σ_n dY[m,n]·W[n,k]
//   dY [M,N], W [N,K], dX [M,K].  Warp per (m,k), reduce over N.
extern "C" __global__ void linear_bwd_dx_f32(
    const float* __restrict__ dY,
    const float* __restrict__ W,
    float* __restrict__ dX,
    int M, int K, int N)
{
    int m = blockIdx.x, k = blockIdx.y;
    if (m >= M || k >= K) return;
    int lane = threadIdx.x;
    float s = 0.0f;
    for (int n = lane; n < N; n += 32) s += dY[(size_t)m * N + n] * W[(size_t)n * K + k];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
    if (lane == 0) dX[(size_t)m * K + k] = s;
}

// linear backward — dW = dY^T · X.   dW[n,k] = Σ_m dY[m,n]·X[m,k]
//   dY [M,N], X [M,K], dW [N,K].  Warp per (n,k), reduce over M. Overwrites dW.
extern "C" __global__ void linear_bwd_dw_f32(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dW,
    int M, int K, int N)
{
    int n = blockIdx.x, k = blockIdx.y;
    if (n >= N || k >= K) return;
    int lane = threadIdx.x;
    float s = 0.0f;
    for (int m = lane; m < M; m += 32) s += dY[(size_t)m * N + n] * X[(size_t)m * K + k];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
    if (lane == 0) dW[(size_t)n * K + k] = s;
}

// RMSNorm backward (per row).  Forward: y_i = g_i x_i / rms, rms=sqrt(mean(x^2)+eps).
//   dx_k = r[ g_k dy_k - (r^2 x_k / n) · c ],  c = Σ_i dy_i g_i x_i,  r = 1/rms
//   dg_k += dy_k x_k r   (accumulated over rows via atomics; dG pre-zeroed)
extern "C" __global__ void rmsnorm_bwd_f32(
    const float* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ dY,
    float* __restrict__ dX,
    float* __restrict__ dG,
    int rows, int n, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* x = X + (size_t)row * n;
    const float* dy = dY + (size_t)row * n;
    __shared__ float red[256];
    int t = threadIdx.x, bs = blockDim.x;
    float lq = 0.0f, lc = 0.0f;
    for (int i = t; i < n; i += bs) { float xv = x[i]; lq += xv * xv; lc += dy[i] * G[i] * xv; }
    red[t] = lq; __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float q = red[0]; __syncthreads();
    red[t] = lc; __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float c = red[0]; __syncthreads();
    float rms = sqrtf(q / (float)n + eps);
    float r = 1.0f / rms;
    float r2 = r * r;
    for (int i = t; i < n; i += bs) {
        float xv = x[i];
        float dx = r * (G[i] * dy[i] - (r2 * xv / (float)n) * c);
        dX[(size_t)row * n + i] = dx;
        atomicAdd(&dG[i], dy[i] * xv * r);
    }
}

// SwiGLU backward.  out = silu(g)·u ;  silu(g)=g·σ(g)
//   dU = dOut·silu(g) ;  dG = dOut·u·silu'(g),  silu'(g)=σ(g)(1+g(1-σ(g)))
extern "C" __global__ void silu_mul_bwd_f32(
    const float* __restrict__ G,
    const float* __restrict__ U,
    const float* __restrict__ dOut,
    float* __restrict__ dG,
    float* __restrict__ dU,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = G[i], u = U[i], d = dOut[i];
    float sg = 1.0f / (1.0f + __expf(-g));
    float silu = g * sg;
    float dsilu = sg * (1.0f + g * (1.0f - sg));
    dU[i] = d * silu;
    dG[i] = d * u * dsilu;
}

// RoPE backward (rotate_half, in place on the grad). Forward is an orthogonal
// rotation R(θ) per (i, i+half) pair; backward is R(θ)^T:
//   din_i = dout_i·c + dout_{i+half}·s ;  din_{i+half} = -dout_i·s + dout_{i+half}·c
extern "C" __global__ void rope_rows_bwd_f32(
    float* __restrict__ grad,
    const int* __restrict__ positions,
    int n_heads, int head_dim, float freq_base, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = head_dim / 2;
    if (i >= half || b >= rows) return;
    int pos = positions[b];
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float c = cosf(val), s = sinf(val);
    int stride = n_heads * head_dim;
    for (int h = 0; h < n_heads; h++) {
        int base = b * stride + h * head_dim;
        float g0 = grad[base + i], g1 = grad[base + i + half];
        grad[base + i]        = g0 * c + g1 * s;
        grad[base + i + half] = -g0 * s + g1 * c;
    }
}

// GQA block attention backward (grad-checkpointing: recomputes the forward
// softmax). One warp per (query i, head h). dQ written directly (each (i,h)
// owns its row); dK/dV accumulated via atomicAdd (must be PRE-ZEROED).
//   p_j = softmax_j(scale·Q_i·K_j) ;  O_i = Σ_j p_j V_j
//   dV_j += p_j dO_i ;  dp_j = dO_i·V_j ;  D = Σ_j p_j dp_j
//   ds_j = p_j(dp_j - D) ;  dQ_i += scale Σ_j ds_j K_j ;  dK_j += scale ds_j Q_i
extern "C" __global__ void attn_block_ctx_bwd_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int B, int L, int nH, int nKV, int hd, float scale)
{
    int qh = blockIdx.x;
    if (qh >= B * nH) return;
    int i = qh / nH, h = qh % nH;
    int group = nH / nKV, g = h / group;
    int lane = threadIdx.x;
    float q[4], dout[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        int d = lane + e * 32;
        q[e]    = (d < hd) ? Q[((size_t)(i * nH + h)) * hd + d]  : 0.0f;
        dout[e] = (d < hd) ? dO[((size_t)(i * nH + h)) * hd + d] : 0.0f;
    }
    // pass 1: softmax stats (m, l)
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < L; j++) {
        const float* kb = K + ((size_t)(j * nKV + g)) * hd;
        float p = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; e++) { int d = lane + e * 32; if (d < hd) p += q[e] * kb[d]; }
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor(p, o);
        float s = p * scale, nm = fmaxf(m, s), corr = __expf(m - nm), pp = __expf(s - nm);
        l = l * corr + pp; m = nm;
    }
    float invl = (l > 0.0f) ? 1.0f / l : 0.0f;
    // pass 2: D = Σ_j p_j dp_j
    float D = 0.0f;
    for (int j = 0; j < L; j++) {
        const float* kb = K + ((size_t)(j * nKV + g)) * hd;
        const float* vb = V + ((size_t)(j * nKV + g)) * hd;
        float pdot = 0.0f, dp = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; e++) { int d = lane + e * 32; if (d < hd) { pdot += q[e] * kb[d]; dp += dout[e] * vb[d]; } }
        for (int o = 16; o > 0; o >>= 1) { pdot += __shfl_xor(pdot, o); dp += __shfl_xor(dp, o); }
        float pj = __expf(pdot * scale - m) * invl;
        D += pj * dp;
    }
    // pass 3: grads
    float dq[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) dq[e] = 0.0f;
    for (int j = 0; j < L; j++) {
        const float* kb = K + ((size_t)(j * nKV + g)) * hd;
        const float* vb = V + ((size_t)(j * nKV + g)) * hd;
        float pdot = 0.0f, dp = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; e++) { int d = lane + e * 32; if (d < hd) { pdot += q[e] * kb[d]; dp += dout[e] * vb[d]; } }
        for (int o = 16; o > 0; o >>= 1) { pdot += __shfl_xor(pdot, o); dp += __shfl_xor(dp, o); }
        float pj = __expf(pdot * scale - m) * invl;
        float ds = pj * (dp - D);
        float dsc = ds * scale;
        float* dkb = dK + ((size_t)(j * nKV + g)) * hd;
        float* dvb = dV + ((size_t)(j * nKV + g)) * hd;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int d = lane + e * 32;
            if (d < hd) {
                dq[e] += dsc * kb[d];
                atomicAdd(&dkb[d], dsc * q[e]);
                atomicAdd(&dvb[d], pj * dout[e]);
            }
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; e++) { int d = lane + e * 32; if (d < hd) dQ[((size_t)(i * nH + h)) * hd + d] = dq[e]; }
}

// ===================================================================
// DFlash trainer: loss backward + optimizer (step 5).
// ===================================================================

// Softmax cross-entropy backward + per-row loss, with a per-row weight.
//   p = softmax(logits_row) ;  loss_b = w_b · (-log p[target_b])
//   dlogits_b = w_b · (p - onehot(target_b))
// One block per row, block-reduced max + sumexp. Non-positive weight (revealed
// / non-masked positions) -> zero loss + zero grad contribution naturally.
extern "C" __global__ void ce_loss_bwd_f32(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ weights,
    float* __restrict__ dlogits,
    float* __restrict__ loss,
    int B, int V)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int t = threadIdx.x, bs = blockDim.x;
    const float* lg = logits + (size_t)b * V;
    float w = weights[b];
    int tgt = targets[b];
    __shared__ float red[256];
    // max
    float lm = -INFINITY;
    for (int i = t; i < V; i += bs) lm = fmaxf(lm, lg[i]);
    red[t] = lm; __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    // sumexp
    float ls = 0.0f;
    for (int i = t; i < V; i += bs) ls += __expf(lg[i] - mx);
    red[t] = ls; __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float z = red[0]; __syncthreads();
    float invz = 1.0f / z;
    for (int i = t; i < V; i += bs) {
        float p = __expf(lg[i] - mx) * invz;
        dlogits[(size_t)b * V + i] = w * (p - (i == tgt ? 1.0f : 0.0f));
    }
    if (t == 0) loss[b] = w * (logf(z) + mx - lg[tgt]);
}

// AdamW step. Bias-correction factors bc1 = 1/(1-b1^t), bc2 = 1/(1-b2^t) are
// passed precomputed from the host. Decoupled weight decay `wd`.
//   m = b1 m + (1-b1) g ;  v = b2 v + (1-b2) g^2
//   w -= lr · ( (m·bc1) / (sqrt(v·bc2) + eps) + wd · w )
extern "C" __global__ void adam_step_f32(
    float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    float mh = mi * bc1;
    float vh = vi * bc2;
    w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
}
