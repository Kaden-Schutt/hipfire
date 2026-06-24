// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! nemotron_h GQA attention (`*`) block — **NoPE** decode + KV cache.
//!
//! Confirmed from the checkpoint: nemotron_h attention has NO positional
//! embedding (no rope keys at all) and NO bias (`attention_bias=false`). So the
//! decode is the qwen2 GQA path *minus* the bias-add and RoPE steps:
//! ```text
//!   q = q_proj @ x ; k = k_proj @ x ; v = v_proj @ x      # gemv, no bias
//!   kv_cache_write(k,v @ pos)                              # F32 cache
//!   a = attention_flash(q, k_cache, v_cache, seq=pos+1)    # causal GQA, NoPE
//!   out = o_proj @ a
//! ```
//! KV cache layout matches qwen2: `[max_seq × n_kv_heads × head_dim]` flat per
//! block, written at `pos*kv_dim`. f32 / decode-only.

use crate::AttnConfig;
use hip_bridge::{DeviceBuffer, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

/// CPU reference: causal GQA attention (NoPE) for one query against the full
/// `[seq_len]` k/v history. `q` is `[n_heads*head_dim]`, `k_hist`/`v_hist` are
/// `seq_len` vectors each `[n_kv_heads*head_dim]` (head-major). Returns the
/// attention output `[n_heads*head_dim]` (pre o_proj).
pub fn gqa_attention(
    q: &[f32],
    k_hist: &[Vec<f32>],
    v_hist: &[Vec<f32>],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let seq = k_hist.len();
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..h * head_dim + head_dim];
        // scores
        let mut scores = vec![0.0f32; seq];
        let mut mx = f32::NEG_INFINITY;
        for (t, kt) in k_hist.iter().enumerate() {
            let kk = &kt[kvh * head_dim..kvh * head_dim + head_dim];
            let dot: f32 = qh.iter().zip(kk).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
            mx = mx.max(scores[t]);
        }
        // softmax
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        // weighted sum of V
        let oh = &mut out[h * head_dim..h * head_dim + head_dim];
        for (t, vt) in v_hist.iter().enumerate() {
            let w = scores[t] / sum;
            let vv = &vt[kvh * head_dim..kvh * head_dim + head_dim];
            for d in 0..head_dim {
                oh[d] += w * vv[d];
            }
        }
    }
    out
}

/// GPU-resident GQA attention block (q/k/v/o weights + KV cache + scratch).
pub struct NemotronAttnGpu {
    cfg: AttnConfig,
    hidden: usize,
    max_seq: usize,
    // weights (bias-free)
    q_proj: GpuTensor,
    k_proj: GpuTensor,
    v_proj: GpuTensor,
    o_proj: GpuTensor,
    // KV cache + position
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    pos_buf: DeviceBuffer,
    // scratch
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attn_out: GpuTensor,
    attn_partials: GpuTensor,
    out: GpuTensor,
}

impl NemotronAttnGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &mut Gpu,
        cfg: AttnConfig,
        hidden: usize,
        max_seq: usize,
        q_proj: &[f32],
        k_proj: &[f32],
        v_proj: &[f32],
        o_proj: &[f32],
    ) -> HipResult<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let n_chunks_max = max_seq.div_ceil(128);
        Ok(Self {
            cfg,
            hidden,
            max_seq,
            q_proj: gpu.upload_f32(q_proj, &[q_dim, hidden])?,
            k_proj: gpu.upload_f32(k_proj, &[kv_dim, hidden])?,
            v_proj: gpu.upload_f32(v_proj, &[kv_dim, hidden])?,
            o_proj: gpu.upload_f32(o_proj, &[hidden, q_dim])?,
            k_cache: gpu.zeros(&[max_seq * kv_dim], DType::F32)?,
            v_cache: gpu.zeros(&[max_seq * kv_dim], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
            q: gpu.zeros(&[q_dim], DType::F32)?,
            k: gpu.zeros(&[kv_dim], DType::F32)?,
            v: gpu.zeros(&[kv_dim], DType::F32)?,
            attn_out: gpu.zeros(&[q_dim], DType::F32)?,
            attn_partials: gpu.zeros(
                &[cfg.num_heads * n_chunks_max * (2 + cfg.head_dim)],
                DType::F32,
            )?,
            out: gpu.zeros(&[hidden], DType::F32)?,
        })
    }

    /// One decode step at absolute position `pos` (0-based). Reads `x`
    /// `[hidden]`, writes k/v into the cache, returns the `[hidden]` output.
    pub fn forward(&mut self, gpu: &mut Gpu, x: &GpuTensor, pos: usize) -> HipResult<&GpuTensor> {
        let kv_dim = self.cfg.num_kv_heads * self.cfg.head_dim;
        gpu.hip
            .memcpy_htod(&self.pos_buf, &(pos as i32).to_ne_bytes())?;

        // q/k/v projections (no bias, no rope)
        gpu.gemv_f32(&self.q_proj, x, &self.q)?;
        gpu.gemv_f32(&self.k_proj, x, &self.k)?;
        gpu.gemv_f32(&self.v_proj, x, &self.v)?;

        // KV cache write at pos
        gpu.kv_cache_write(&self.k_cache, &self.k, &self.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&self.v_cache, &self.v, &self.pos_buf, kv_dim)?;

        // causal GQA flash decode over seq_len = pos+1
        gpu.attention_flash(
            &self.q,
            &self.k_cache,
            &self.v_cache,
            &self.attn_out,
            &self.attn_partials,
            pos + 1,
            self.cfg.num_heads,
            self.cfg.num_kv_heads,
            self.cfg.head_dim,
            self.max_seq,
        )?;

        // output projection
        gpu.gemv_f32(&self.o_proj, &self.attn_out, &self.out)?;
        Ok(&self.out)
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Free all GPU tensors + the pos buffer (consumes the block).
    pub fn free(self, gpu: &mut Gpu) {
        let _ = gpu.hip.free(self.pos_buf);
        for t in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.k_cache,
            self.v_cache,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.attn_partials,
            self.out,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_key_attention_is_value() {
        // seq_len=1 → softmax is 1.0 → attention output == v (per head).
        let n_heads = 2;
        let n_kv = 1;
        let hd = 2;
        let q = vec![0.5, -0.3, 1.0, 0.2];
        let k = vec![0.1, 0.4];
        let v = vec![7.0, -2.0];
        let out = gqa_attention(&q, &[k], &[v.clone()], n_heads, n_kv, hd);
        // both query heads share kv head 0 → each gets v
        assert!((out[0] - v[0]).abs() < 1e-5 && (out[1] - v[1]).abs() < 1e-5);
        assert!((out[2] - v[0]).abs() < 1e-5 && (out[3] - v[1]).abs() < 1e-5);
    }
}
