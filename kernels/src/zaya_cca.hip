// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// ZAYA1 CCA + EDA/MoD custom elementwise kernels. All are register-only
// (NO __shared__/LDS — gfx1103 wedge hazard): one thread per output element,
// global reads only. Faithful to crates/hipfire-arch-zaya/src/cpu.rs.

#include <hip/hip_runtime.h>

// Embedding gather: out[t, i] = embed[ids[t]*hidden + i]. n = s*hidden.
extern "C" __global__ void zaya_embed_gather_f32(
    float* __restrict__ out, const float* __restrict__ embed,
    const int* __restrict__ ids, int hidden, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = i / hidden;
    int c = i % hidden;
    out[i] = embed[ids[t] * hidden + c];
}

// Broadcast bias add: x[i] += bias[i % d]. n = s*d.
extern "C" __global__ void zaya_bias_add_f32(
    float* __restrict__ x, const float* __restrict__ bias, int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += bias[i % d];
}

// Global input residual affine on the embeddings: out[i] = (x[i]+bias[c])*scale[c],
// c = i % d. n = seq*d.
extern "C" __global__ void zaya_affine_input_f32(
    float* __restrict__ out, const float* __restrict__ x,
    const float* __restrict__ scale, const float* __restrict__ bias,
    int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % d;
    out[i] = (x[i] + bias[c]) * scale[c];
}

// Learned residual merge: out[i] = (h[i]+hb[c])*hs[c] + (res[i]+rb[c])*rs[c].
extern "C" __global__ void zaya_affine_residual_f32(
    float* __restrict__ out, const float* __restrict__ h, const float* __restrict__ res,
    const float* __restrict__ hs, const float* __restrict__ hb,
    const float* __restrict__ rs, const float* __restrict__ rb,
    int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % d;
    out[i] = (h[i] + hb[c]) * hs[c] + (res[i] + rb[c]) * rs[c];
}

// Scatter q[s,q_dim] and k[s,k_dim] into a left-padded channel-major stream
// [conv_ch, pad+s] (pad-region pre-zeroed by the caller). conv_ch = q_dim+k_dim.
// One thread per (t, channel) over conv_ch*s.
extern "C" __global__ void zaya_qk_stream_f32(
    float* __restrict__ stream, const float* __restrict__ q, const float* __restrict__ k,
    int s, int q_dim, int k_dim, int pad
) {
    int conv_ch = q_dim + k_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_ch * s) return;
    int c = idx / s;
    int t = idx % s;
    int plen = pad + s;
    float v = (c < q_dim) ? q[t * q_dim + c] : k[t * k_dim + (c - q_dim)];
    stream[c * plen + pad + t] = v;
}

// Generic VALID causal grouped conv1d, channel-major. in: [channels, in_len];
// out: [channels, out_len] with out_len = in_len - kernel + 1. weight layout
// [out_channels, in_per_group, kernel] row-major. in==out==channels here so
// in_per_group == out_per_group == channels/groups. Depthwise == groups==channels.
extern "C" __global__ void zaya_conv1d_valid_f32(
    float* __restrict__ out, const float* __restrict__ in,
    const float* __restrict__ weight, const float* __restrict__ bias,
    int channels, int groups, int kernel, int in_len, int out_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * out_len) return;
    int c = idx / out_len;
    int t = idx % out_len;
    int per_group = channels / groups;          // in_per_group == out_per_group
    int base = (c / per_group) * per_group;      // first input channel of this group
    float acc = bias[c];
    for (int j = 0; j < per_group; ++j) {
        const float* wr = &weight[(c * per_group + j) * kernel];
        const float* ir = &in[(base + j) * in_len + t];
        for (int kk = 0; kk < kernel; ++kk) {
            acc += wr[kk] * ir[kk];
        }
    }
    out[c * out_len + t] = acc;
}

// q/k residual paths. query_residual[t,head,d] = (q[t,head,d] + k[t,head/groups,d]) * 0.5;
// key_residual[t,kh,d] = mean over the `groups` q-heads of query_residual.
// Launch over s*nq*hd (writes query_residual) then s*nkv*hd (writes key_residual)
// via two calls distinguished by `mode` (0 = query, 1 = key).
extern "C" __global__ void zaya_qk_residual_f32(
    float* __restrict__ query_res, float* __restrict__ key_res,
    const float* __restrict__ q, const float* __restrict__ k,
    int s, int nq, int nkv, int hd, int mode
) {
    int groups = nq / nkv;
    if (mode == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= s * nq * hd) return;
        int d = idx % hd;
        int head = (idx / hd) % nq;
        int t = idx / (hd * nq);
        int kh = head / groups;
        float qv = q[t * (nq * hd) + head * hd + d];
        float kv = k[t * (nkv * hd) + kh * hd + d];
        query_res[(t * nq + head) * hd + d] = (qv + kv) * 0.5f;
    } else {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= s * nkv * hd) return;
        int d = idx % hd;
        int kh = (idx / hd) % nkv;
        int t = idx / (hd * nkv);
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            acc += query_res[(t * nq + kh * groups + g) * hd + d];
        }
        key_res[(t * nkv + kh) * hd + d] = acc / (float)groups;
    }
}

// Add conv output (channel-major [conv_ch, s]) back to the residual paths and
// emit head-major q [s,nq,hd] and k [s,nkv,hd]. mode 0 = query, 1 = key.
extern "C" __global__ void zaya_add_conv_residual_f32(
    float* __restrict__ out, const float* __restrict__ conv, const float* __restrict__ res,
    int s, int heads, int hd, int q_dim, int mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * heads * hd) return;
    int d = idx % hd;
    int head = (idx / hd) % heads;
    int t = idx / (hd * heads);
    int conv_ch = (mode == 0) ? head * hd + d : q_dim + head * hd + d;
    int conv_seq = s; // conv stored [conv_ch, s]
    out[(t * heads + head) * hd + d] = conv[conv_ch * conv_seq + t] + res[(t * heads + head) * hd + d];
}

// Compositional value: head 0 = current-token v, head 1 = previous-token delayed v.
// value [s, nkv=2, hd]; v_cur/v_del [s, hd].
extern "C" __global__ void zaya_value_compose_f32(
    float* __restrict__ value, const float* __restrict__ v_cur, const float* __restrict__ v_del,
    int s, int nkv, int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * hd) return;
    int d = idx % hd;
    int t = idx / hd;
    value[(t * nkv + 0) * hd + d] = v_cur[t * hd + d];
    value[(t * nkv + 1) * hd + d] = (t == 0) ? 0.0f : v_del[(t - 1) * hd + d];
}

// L2-normalize each [head, hd] row to `scale` (= sqrt(head_dim)); if `temp` is
// non-null, multiply each head by temp[head] (key path). No LDS: each thread
// owns one (t, head) row and loops hd.
extern "C" __global__ void zaya_qk_l2norm_temp_f32(
    float* __restrict__ x, const float* __restrict__ temp,
    int s, int heads, int hd, float scale, float eps, int has_temp
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= s * heads) return;
    float* v = &x[row * hd];
    float nrm = 0.0f;
    for (int d = 0; d < hd; ++d) nrm += v[d] * v[d];
    float inv = scale / fmaxf(sqrtf(nrm), eps);
    float tmul = 1.0f;
    if (has_temp != 0) {
        int head = row % heads;
        tmul = temp[head];
    }
    for (int d = 0; d < hd; ++d) v[d] = v[d] * inv * tmul;
}

// Exact erf-GELU (A&S 7.1.26), in place. n elements.
extern "C" __global__ void zaya_gelu_exact_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float a = v * 0.70710678118654752440f; // x/sqrt(2)
    float sign = (a < 0.0f) ? -1.0f : 1.0f;
    float ax = fabsf(a);
    float tt = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt + 0.254829592f) * tt * expf(-ax * ax);
    float erf = sign * y;
    x[i] = 0.5f * v * (1.0f + erf);
}

// Partial rotary (half-split), position = pos_base + token index. x [s,heads,hd];
// only first n_rot dims rotate. One thread per (token, head). pos_base=0 for
// prefill; pos_base=current position for single-token decode.
extern "C" __global__ void zaya_rope_partial_f32(
    float* __restrict__ x, int s, int heads, int hd, int n_rot, float theta, int pos_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * heads) return;
    int head = idx % heads;
    int t = (idx / heads) + pos_base;
    (void)head;
    float* v = &x[idx * hd];
    int half = n_rot / 2;
    for (int d = 0; d < half; ++d) {
        float inv_freq = powf(theta, -(2.0f * (float)d) / (float)n_rot);
        float ang = (float)t * inv_freq;
        float c = cosf(ang), sn = sinf(ang);
        float a = v[d];
        float b = v[d + half];
        v[d] = a * c - b * sn;
        v[d + half] = b * c + a * sn;
    }
}

// Simple GQA causal attention (bring-up; O(s^2) per row, NO LDS). q [s,nq,hd],
// k/v [s,nkv,hd] head-major; out [s,nq,hd]. scaling applied to scores. One
// thread per (query position i, head). For short prompts; flash path later.
extern "C" __global__ void zaya_gqa_attn_f32(
    float* __restrict__ out, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v,
    int s, int nq, int nkv, int hd, float scaling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * nq) return;
    int head = idx % nq;
    int i = idx / nq;
    int groups = nq / nkv;
    int kh = head / groups;
    const float* qv = &q[(i * nq + head) * hd];
    // online softmax over j = 0..i
    float maxv = -1e30f;
    for (int j = 0; j <= i; ++j) {
        const float* kv = &k[(j * nkv + kh) * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        dot *= scaling;
        if (dot > maxv) maxv = dot;
    }
    float denom = 0.0f;
    float* o = &out[(i * nq + head) * hd];
    for (int d = 0; d < hd; ++d) o[d] = 0.0f;
    for (int j = 0; j <= i; ++j) {
        const float* kv = &k[(j * nkv + kh) * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        float p = expf(dot * scaling - maxv);
        denom += p;
        const float* vv = &v[(j * nkv + kh) * hd];
        for (int d = 0; d < hd; ++d) o[d] += p * vv[d];
    }
    float inv = 1.0f / denom;
    for (int d = 0; d < hd; ++d) o[d] *= inv;
}

// Strided row copy: dst[r*dst_stride + j] = src[r*src_stride + src_off + j],
// j in 0..len, r in 0..rows. (Used to seed the decode conv ring from the prefill
// stream's last `pad` columns.) One thread per (row, j).
extern "C" __global__ void zaya_strided_copy_f32(
    float* __restrict__ dst, const float* __restrict__ src,
    int rows, int dst_stride, int src_stride, int src_off, int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * len) return;
    int r = idx / len;
    int j = idx % len;
    dst[r * dst_stride + j] = src[r * src_stride + src_off + j];
}

// Decode conv window + ring update. Build window [conv_ch, pad+1] = [ring | cur]
// (channel-major) and advance the ring (drop oldest, push cur). One thread/channel.
extern "C" __global__ void zaya_conv_window_f32(
    float* __restrict__ window, float* __restrict__ ring, const float* __restrict__ cur,
    int conv_ch, int pad
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_ch) return;
    int wlen = pad + 1;
    for (int j = 0; j < pad; ++j) window[c * wlen + j] = ring[c * pad + j];
    window[c * wlen + pad] = cur[c];
    for (int j = 0; j + 1 < pad; ++j) ring[c * pad + j] = ring[c * pad + j + 1];
    ring[c * pad + (pad - 1)] = cur[c];
}

// Copy src[0..n] into dst[offset..offset+n] (KV-cache append). One thread/elem.
extern "C" __global__ void zaya_write_at_f32(
    float* __restrict__ dst, const float* __restrict__ src, int offset, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[offset + i] = src[i];
}

// Single-token GQA decode attention over the KV cache. q [nq,hd]; k_cache/v_cache
// [max_seq, nkv, hd]; out [nq,hd]; attends positions 0..=pos. One thread/(head).
extern "C" __global__ void zaya_gqa_decode_f32(
    float* __restrict__ out, const float* __restrict__ q,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    int pos, int nq, int nkv, int hd, float scaling
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= nq) return;
    int groups = nq / nkv;
    int kh = head / groups;
    const float* qv = &q[head * hd];
    int kvdim = nkv * hd;
    float maxv = -1e30f;
    for (int j = 0; j <= pos; ++j) {
        const float* kv = &k_cache[j * kvdim + kh * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        dot *= scaling;
        if (dot > maxv) maxv = dot;
    }
    float denom = 0.0f;
    float* o = &out[head * hd];
    for (int d = 0; d < hd; ++d) o[d] = 0.0f;
    for (int j = 0; j <= pos; ++j) {
        const float* kv = &k_cache[j * kvdim + kh * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        float p = expf(dot * scaling - maxv);
        denom += p;
        const float* vv = &v_cache[j * kvdim + kh * hd];
        for (int d = 0; d < hd; ++d) o[d] += p * vv[d];
    }
    float inv = 1.0f / denom;
    for (int d = 0; d < hd; ++d) o[d] *= inv;
}

// EDA cross-layer state add: router_hidden[t,i] += prev[t,i] * scale[i]. n = s*rh.
extern "C" __global__ void zaya_eda_add_f32(
    float* __restrict__ router_hidden, const float* __restrict__ prev,
    const float* __restrict__ scale, int rh, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % rh;
    router_hidden[i] += prev[i] * scale[c];
}
