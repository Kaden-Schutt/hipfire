// SPDX-License-Identifier: Apache-2.0
//! KV-compression sim-noise for recovery-FT probes (KVarN-4bit + CASK merge).
//!
//! Injects, as a forward-only perturbation on post-RoPE K and on V, the two
//! noise sources hipfire's hierarchical KV path introduces:
//!   1. **KVarN-4bit** — per-(token,kv-head) symmetric 4-bit quant round-trip.
//!      (Per the kv-compression study this is ~lossless even at 2-bit; included
//!      for completeness, not the dominant cost.)
//!   2. **CASK cold-token merge** — keep the last `hot` tokens exact; replace
//!      each older ("cold") token's K/V by the per-channel mean of its `fold`-
//!      sized position group. This is the RoPE-phase blur that the study isolates
//!      as the whole merge cost (the ~+3 PPL "inherent lossy-merge floor").
//!
//! Used inside `block_forward` between RoPE and GQA. The perturbed K/V are stored
//! in `BlockActivations`, so `gqa_backward` stays self-consistent and the quant/
//! merge acts as a straight-through estimator (gradient flows to q/v projections
//! + norms as identity) — exactly QAT-with-STE on the KV path.
//!
//! All gated by env so the default forward is byte-identical:
//!   HIPFIRE_KVNOISE=1     enable
//!   HIPFIRE_KVNOISE_HOT   exact recent window (default 4)
//!   HIPFIRE_KVNOISE_FOLD  cold merge group size (default 4)
//!   HIPFIRE_KVNOISE_BITS  KVarN quant bits (default 4; 0 = skip quant)

use rdna_compute::{Gpu, GpuTensor, HipResult};

#[derive(Clone, Copy)]
pub struct KvNoiseCfg {
    pub hot: usize,
    pub fold: usize,
    pub bits: u32,
}

/// Read the env config; returns `None` (no-op) unless HIPFIRE_KVNOISE=1.
pub fn cfg_from_env() -> Option<KvNoiseCfg> {
    if std::env::var("HIPFIRE_KVNOISE").ok().as_deref() != Some("1") {
        return None;
    }
    let u = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    Some(KvNoiseCfg {
        hot: u("HIPFIRE_KVNOISE_HOT", 4),
        fold: u("HIPFIRE_KVNOISE_FOLD", 4).max(1),
        bits: u("HIPFIRE_KVNOISE_BITS", 4) as u32,
    })
}

/// Symmetric n-bit quant round-trip of one vector slice, in place.
fn quant_vec(v: &mut [f32], bits: u32) {
    if bits == 0 {
        return;
    }
    let qmax = ((1i32 << (bits - 1)) - 1) as f32; // e.g. bits=4 → 7
    let amax = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if amax <= 0.0 {
        return;
    }
    let scale = amax / qmax;
    let inv = 1.0 / scale;
    for x in v.iter_mut() {
        *x = (*x * inv).round().clamp(-qmax, qmax) * scale;
    }
}

/// CASK merge + KVarN quant on a host `[seq, dim]` token-major buffer, where each
/// token vector is `n_head` contiguous `head_dim` sub-vectors. Merge groups cold
/// tokens by adjacent position (RoPE-phase blur); quant is per (token, head).
fn compress_host(buf: &mut [f32], seq: usize, dim: usize, head_dim: usize, cfg: KvNoiseCfg) {
    let n_head = dim / head_dim;
    let cold_end = seq.saturating_sub(cfg.hot); // tokens [0, cold_end) are cold
    // CASK merge: replace each cold token's vector by its fold-group mean.
    let mut g = 0;
    while g < cold_end {
        let end = (g + cfg.fold).min(cold_end);
        let n = end - g;
        if n > 1 {
            for c in 0..dim {
                let mut s = 0.0f32;
                for t in g..end {
                    s += buf[t * dim + c];
                }
                let m = s / n as f32;
                for t in g..end {
                    buf[t * dim + c] = m;
                }
            }
        }
        g = end;
    }
    // KVarN quant: per (token, head) symmetric n-bit round-trip.
    if cfg.bits > 0 {
        for t in 0..seq {
            for hh in 0..n_head {
                let off = t * dim + hh * head_dim;
                quant_vec(&mut buf[off..off + head_dim], cfg.bits);
            }
        }
    }
}

/// Apply KV-noise to post-RoPE K and to V, returning fresh tensors (originals
/// freed). No-op pass-through when disabled. `kvd = n_kv * head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn maybe_compress_kv(
    gpu: &mut Gpu,
    k_r: GpuTensor,
    v: GpuTensor,
    cfg: Option<KvNoiseCfg>,
    seq: usize,
    kvd: usize,
    head_dim: usize,
) -> HipResult<(GpuTensor, GpuTensor)> {
    let Some(cfg) = cfg else {
        return Ok((k_r, v));
    };
    let mut kh = gpu.download_f32(&k_r)?;
    let mut vh = gpu.download_f32(&v)?;
    compress_host(&mut kh, seq, kvd, head_dim, cfg);
    compress_host(&mut vh, seq, kvd, head_dim, cfg);
    let k_new = gpu.upload_f32(&kh, &[seq * kvd])?;
    let v_new = gpu.upload_f32(&vh, &[seq * kvd])?;
    gpu.free_tensor(k_r)?;
    gpu.free_tensor(v)?;
    Ok((k_new, v_new))
}
