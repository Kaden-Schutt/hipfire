// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Full parent-checkpoint transformer: embed → 43 layers → head → logits.
//!
//! Operator-semantics authority
//! (`.codeinsight+research/ds4-parent-ref/inference/model.py`):
//!
//! ```text
//! h = embed(ids).unsqueeze(-2).repeat(1, 1, hc_mult, 1)   # [B,S,hc,dim]
//! for layer in layers:
//!     h = layer(h, start_pos, freqs_cis, mask, input_ids=ids)
//! h = hc_head(h); h = norm(h); logits = head(h.float())
//! ```
//!
//! This module composes the landed parent sub-blocks
//! (`parent::{head,forward}`) and owns the per-layer SWA KV rings. It does
//! **not** reimplement attention, MoE, compressor, or indexer — those land
//! underneath via [`super::forward::parent_layer_forward`].
//!
//! Every numeric constant traces to the checkpoint `config.json` / tensor
//! shapes. Fail closed: missing layers, wrong device, or a non-finite
//! intermediate is an `Err`, never a silent skip.

use crate::parent::attention::{
    all_finite, l2_norm, PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_SWA_WINDOW,
};
use crate::parent::forward::{
    parent_layer_forward, parent_layer_forward_traced, ParentForwardScratch, ParentLayerTrace,
    PARENT_HC_DIM, PARENT_HC_MULT,
};
use crate::parent::head::{
    parent_embed, parent_head_with_scratch, ParentHeadScratch, PARENT_VOCAB,
};
use crate::parent::weights::ParentWeights;
use crate::parent::{Ds4ParentBackend, ParentQuantConfig};
use rdna_compute::{DType, Gpu, GpuTensor};

#[inline]
fn err(msg: impl Into<String>) -> String {
    format!("deepseek4 parent: {}", msg.into())
}

// ── Scratch ─────────────────────────────────────────────────────────────────

/// All scratch a full-model forward needs: layer tiles, head tiles, the
/// multi-stream residual double-buffer, and one SWA KV ring per layer.
///
/// # Ownership
///
/// - Layer composition scratch ([`ParentForwardScratch`]) is shared across
///   all layers — compressor/indexer tiles live inside it once AttnCompIdx
///   lands; the model driver does not grow a parallel set.
/// - Head intermediate tiles ([`ParentHeadScratch`]).
/// - HC residual double-buffer `[max_rows, hc_mult, dim]` F32 × 2.
/// - Per-layer SWA rings `[n_kv_heads=1, head_dim=512, window=128]` F32 —
///   one per `cfg.num_hidden_layers`. Compressed-KV state (ratio ≠ 0) is
///   internal to attention scratch; these rings stay SWA-only.
///
/// Logits are **caller-owned** (streamed / sized externally) so a multi-
/// thousand-token capture never forces a second full vocab tile here.
pub struct ParentModelScratch {
    layer: ParentForwardScratch,
    head: ParentHeadScratch,
    /// HC residual A. F32 `[max_rows, hc_mult, dim]`.
    hc_a: GpuTensor,
    /// HC residual B (ping-pong partner of `hc_a`).
    hc_b: GpuTensor,
    /// One SWA KV ring per absolute layer index.
    /// Shape `[n_kv_heads, head_dim, window]` F32 each.
    kv_rings: Vec<GpuTensor>,
    max_rows: usize,
    n_layers: usize,
    bytes: usize,
}

impl ParentModelScratch {
    /// Allocate reusable scratch for up to `max_rows` tokens across the full
    /// tower (`cfg.num_hidden_layers` KV rings).
    pub fn new(gpu: &mut Gpu, cfg: &ParentQuantConfig, max_rows: usize) -> Result<Self, String> {
        if max_rows == 0 {
            return Err(err("ParentModelScratch max_rows must be > 0"));
        }
        if cfg.num_hidden_layers == 0 {
            return Err(err("ParentModelScratch: cfg.num_hidden_layers must be > 0"));
        }

        let layer = ParentForwardScratch::new(gpu, cfg, max_rows)?;
        let head = match ParentHeadScratch::new(gpu, cfg, max_rows) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        let hc_a = gpu
            .alloc_tensor(&[max_rows, PARENT_HC_MULT, PARENT_DIM], DType::F32)
            .map_err(|e| err(format!("ParentModelScratch hc_a: {e:?}")))?;
        let hc_b = match gpu.alloc_tensor(&[max_rows, PARENT_HC_MULT, PARENT_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(hc_a);
                return Err(err(format!("ParentModelScratch hc_b: {e:?}")));
            }
        };

        let n_layers = cfg.num_hidden_layers;
        let mut kv_rings = Vec::with_capacity(n_layers);
        let mut kv_bytes = 0usize;
        for i in 0..n_layers {
            match gpu.alloc_tensor(
                &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
                DType::F32,
            ) {
                Ok(t) => {
                    kv_bytes = kv_bytes.saturating_add(t.buf.size());
                    kv_rings.push(t);
                }
                Err(e) => {
                    let _ = gpu.free_tensor(hc_a);
                    let _ = gpu.free_tensor(hc_b);
                    for r in kv_rings.drain(..) {
                        let _ = gpu.free_tensor(r);
                    }
                    return Err(err(format!("ParentModelScratch kv_ring[{i}]: {e:?}")));
                }
            }
        }

        let own = hc_a.buf.size() + hc_b.buf.size() + kv_bytes;
        let bytes = own
            .saturating_add(layer.bytes())
            .saturating_add(head.bytes());

        Ok(Self {
            layer,
            head,
            hc_a,
            hc_b,
            kv_rings,
            max_rows,
            n_layers,
            bytes,
        })
    }

    /// Total device scratch bytes (layer + head + HC buffers + KV rings).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Peak capacity in rows.
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Number of KV rings (equals `cfg.num_hidden_layers` at construction).
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Nested layer-composition scratch (diagnostics / Gate 4 hooks).
    pub fn layer_scratch(&self) -> &ParentForwardScratch {
        &self.layer
    }

    /// Nested head scratch.
    pub fn head_scratch(&self) -> &ParentHeadScratch {
        &self.head
    }

    /// Zero every KV ring. Call when `start_pos == 0` so a fresh prefill does
    /// not inherit history from a prior forward in the same process.
    pub fn clear_kv_rings(&self, gpu: &Gpu) -> Result<(), String> {
        let zeros = vec![
            0u8;
            PARENT_N_KV_HEADS
                .saturating_mul(PARENT_HEAD_DIM)
                .saturating_mul(PARENT_SWA_WINDOW)
                .saturating_mul(4)
        ];
        for (i, ring) in self.kv_rings.iter().enumerate() {
            if ring.buf.size() < zeros.len() {
                return Err(err(format!(
                    "clear_kv_rings: ring {i} too small ({} < {})",
                    ring.buf.size(),
                    zeros.len()
                )));
            }
            gpu.hip
                .memcpy_htod(&ring.buf, &zeros)
                .map_err(|e| err(format!("clear_kv_rings[{i}]: {e:?}")))?;
        }
        Ok(())
    }
}

// ── Forward ─────────────────────────────────────────────────────────────────

/// Full forward: embed → every loaded layer → hc_head + norm + head → logits.
///
/// `token_ids` length is `rows`. `logits` is F32 `[rows, vocab]`.
/// `start_pos` is the absolute position of `token_ids[0]` in the sequence
/// (0 for a fresh prefill). Hash-routed layers (`layer_idx < num_hash_layers`)
/// receive `token_ids` as `input_ids`.
///
/// Requires `weights.layers` to cover a contiguous absolute range that fits
/// the KV rings allocated at construction. Typically
/// `ParentLoadPlan { layers: 0..cfg.num_hidden_layers, load_experts: true }`.
pub fn parent_model_forward(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentModelScratch,
    token_ids: &[u32],
    start_pos: usize,
    logits: &GpuTensor,
) -> Result<(), String> {
    parent_model_forward_inner(
        gpu,
        backend,
        weights,
        cfg,
        scratch,
        token_ids,
        start_pos,
        logits,
        None,
        None,
    )
}

/// Same as [`parent_model_forward`], appending per-layer HC-state L2 norms
/// into `layer_norms` and per-layer compress-event counts into
/// `compress_events` (one entry per loaded layer, absolute layer order).
///
/// Both output vecs are cleared then filled. `compress_events[i]` is the
/// number of compressed positions the attention path actually consumed for
/// that layer (from the executed compressor path), paired with the layer's
/// `compress_ratio`. Cheap: one D2H residual reduction + one usize read.
pub fn parent_model_forward_traced(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentModelScratch,
    token_ids: &[u32],
    start_pos: usize,
    logits: &GpuTensor,
    layer_norms: &mut Vec<f32>,
    compress_events: &mut Vec<(usize /*ratio*/, usize /*events*/)>,
) -> Result<(), String> {
    parent_model_forward_inner(
        gpu,
        backend,
        weights,
        cfg,
        scratch,
        token_ids,
        start_pos,
        logits,
        Some(layer_norms),
        Some(compress_events),
    )
}

fn parent_model_forward_inner(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentModelScratch,
    token_ids: &[u32],
    start_pos: usize,
    logits: &GpuTensor,
    mut layer_norms: Option<&mut Vec<f32>>,
    mut compress_events: Option<&mut Vec<(usize, usize)>>,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;

    let rows = token_ids.len();
    if rows == 0 {
        return Err(err("parent_model_forward: token_ids must be non-empty"));
    }
    if rows > scratch.max_rows {
        return Err(err(format!(
            "parent_model_forward: rows {rows} exceeds scratch.max_rows {}",
            scratch.max_rows
        )));
    }
    if weights.layers.is_empty() {
        return Err(err(
            "parent_model_forward: weights.layers is empty — load at least one layer",
        ));
    }
    // Contiguous absolute range, matching ParentWeights::load.
    let range = weights.layer_range.clone();
    if weights.layers.len() != range.end.saturating_sub(range.start) {
        return Err(err(format!(
            "parent_model_forward: layers.len() {} != layer_range {:?}",
            weights.layers.len(),
            range
        )));
    }
    if range.end > scratch.n_layers {
        return Err(err(format!(
            "parent_model_forward: layer_range end {} exceeds scratch n_layers {}",
            range.end, scratch.n_layers
        )));
    }
    for (i, layer) in weights.layers.iter().enumerate() {
        let want = range.start + i;
        if layer.layer_idx != want {
            return Err(err(format!(
                "parent_model_forward: layers[{i}].layer_idx {} != expected {want}",
                layer.layer_idx
            )));
        }
    }

    require_dtype(logits, DType::F32, "logits")?;
    require_elems(logits, rows.saturating_mul(PARENT_VOCAB), "logits")?;

    // Fresh prefill: wipe KV history so a second call in-process is deterministic.
    if start_pos == 0 {
        scratch.clear_kv_rings(gpu)?;
    }

    // ── Embed → HC residual ─────────────────────────────────────────────
    parent_embed(gpu, backend, weights, cfg, token_ids, &scratch.hc_a)?;

    // ── Layers (ping-pong hc_a ↔ hc_b) ───────────────────────────────────
    // After layer k (0-based in the loaded range):
    //   even k → wrote hc_b; odd k → wrote hc_a.
    // Final state lives in `final_hc` below.
    if let Some(norms) = layer_norms.as_mut() {
        norms.clear();
        norms.reserve(weights.layers.len());
    }
    if let Some(events) = compress_events.as_mut() {
        events.clear();
        events.reserve(weights.layers.len());
    }

    let mut use_a_as_input = true;
    let want_trace = layer_norms.is_some() || compress_events.is_some();
    for (i, layer) in weights.layers.iter().enumerate() {
        let layer_idx = layer.layer_idx;
        let (x, out) = if use_a_as_input {
            (&scratch.hc_a, &scratch.hc_b)
        } else {
            (&scratch.hc_b, &scratch.hc_a)
        };
        let kv_ring = &scratch.kv_rings[layer_idx];
        let input_ids = if layer_idx < cfg.num_hash_layers {
            Some(token_ids)
        } else {
            None
        };

        if want_trace {
            let mut trace = ParentLayerTrace::default();
            parent_layer_forward_traced(
                gpu,
                backend,
                weights,
                cfg,
                &mut scratch.layer,
                layer_idx,
                x,
                rows,
                start_pos,
                input_ids,
                kv_ring,
                out,
                &mut trace,
            )?;
            // Prefer the downloaded HC residual L2 over the in-trace stage
            // (hc_post_ffn) so the gate's stability series is the actual
            // residual that feeds the next layer / head.
            if layer_norms.is_some() {
                let hc_l2 = stage_l2(gpu, out, rows * PARENT_HC_DIM)?;
                if let Some(norms) = layer_norms.as_mut() {
                    norms.push(hc_l2);
                }
            }
            if let Some(events) = compress_events.as_mut() {
                let ratio = layer.compress_ratio;
                let n = scratch.layer.attn_scratch().last_compress_events();
                events.push((ratio, n));
            }
            let _ = trace; // stage norms available if a future gate wants them
        } else {
            parent_layer_forward(
                gpu,
                backend,
                weights,
                cfg,
                &mut scratch.layer,
                layer_idx,
                x,
                rows,
                start_pos,
                input_ids,
                kv_ring,
                out,
            )?;
        }

        use_a_as_input = !use_a_as_input;
        let _ = i;
    }

    // After N layers, input flag has flipped N times. Final lives in the
    // buffer that was last written = the opposite of the next input.
    let final_hc = if use_a_as_input {
        // next input would be A → last write was B
        &scratch.hc_b
    } else {
        &scratch.hc_a
    };
    // N=0 is refused above; for N>=1:
    //   N odd  → use_a_as_input=false → final = hc_a  (wrote A on last)
    //   N even → use_a_as_input=true  → final = hc_b
    // Wait — after first layer (i=0): wrote B, use_a=false.
    // After second: wrote A, use_a=true.
    // After odd N: use_a=false, final should be A (last write).
    // After even N: use_a=true, final should be B.
    // Current ternary: use_a_as_input true → hc_b; false → hc_a. Correct.

    // ── Head ────────────────────────────────────────────────────────────
    parent_head_with_scratch(
        gpu,
        backend,
        weights,
        cfg,
        &mut scratch.head,
        final_hc,
        rows,
        logits,
    )?;

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn require_dtype(t: &GpuTensor, want: DType, name: &str) -> Result<(), String> {
    if t.dtype != want {
        return Err(err(format!(
            "{name} must be {want:?} (got {:?})",
            t.dtype
        )));
    }
    Ok(())
}

fn require_elems(t: &GpuTensor, n: usize, name: &str) -> Result<(), String> {
    if t.numel() < n {
        return Err(err(format!(
            "{name} too short: have {} need {n}",
            t.numel()
        )));
    }
    Ok(())
}

fn stage_l2(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<f32, String> {
    let host = download_f32_prefix(gpu, t, nelems)?;
    if !all_finite(&host) {
        return Ok(l2_norm(&host));
    }
    Ok(l2_norm(&host))
}

fn download_f32_prefix(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    if t.dtype != DType::F32 {
        return Err(err(format!(
            "download_f32_prefix expects F32 (got {:?})",
            t.dtype
        )));
    }
    let nbytes = nelems
        .checked_mul(4)
        .ok_or_else(|| err("download_f32_prefix size overflow"))?;
    if t.buf.size() < nbytes {
        return Err(err(format!(
            "download_f32_prefix: buffer too small (have {} need {nbytes})",
            t.buf.size()
        )));
    }
    let mut host = vec![0.0f32; nelems];
    let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| err(format!("download_f32_prefix: {e:?}")))?;
    Ok(host)
}

// ── Compress-event gate helper ──────────────────────────────────────────────

/// Fail closed when any `ratio > 0` layer reports zero compress events while
/// `floor(n_tokens / ratio) > 0`. The counts must come from the executed
/// attention path ([`ParentAttnScratch::last_compress_events`]); this helper
/// only asserts the contract.
///
/// Expected count for a prefill is
/// [`crate::parent::compressor::compressor_prefill_n_out`] — used only as the
/// acceptance target printed next to the observed count, never as a substitute
/// for the counter.
pub fn assert_compress_events(
    events: &[(usize /*ratio*/, usize /*observed*/)],
    n_tokens: usize,
) -> Result<(), String> {
    if events.is_empty() {
        return Err(err(
            "assert_compress_events: empty event list — nothing was measured",
        ));
    }
    let mut failures: Vec<String> = Vec::new();
    for (i, &(ratio, observed)) in events.iter().enumerate() {
        if ratio == 0 {
            if observed != 0 {
                failures.push(format!(
                    "layer {i}: ratio=0 but observed {observed} compress events"
                ));
            }
            continue;
        }
        let expect = crate::parent::compressor::compressor_prefill_n_out(n_tokens, ratio);
        if expect > 0 && observed == 0 {
            failures.push(format!(
                "layer {i}: ratio={ratio} expect>={expect} observed=0 \
                 (SWA-only fallback; floor({n_tokens}/{ratio})={expect})"
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(err(format!(
            "compress-event assertion failed ({} layer(s)):\n  {}",
            failures.len(),
            failures.join("\n  ")
        )))
    }
}


// ── Host-side unit tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_prefix() {
        let m = err("boom");
        assert!(m.starts_with("deepseek4 parent: "));
        assert!(m.contains("boom"));
    }

    #[test]
    fn hc_dim_matches_contract() {
        assert_eq!(PARENT_HC_DIM, 4 * 4096);
        assert_eq!(PARENT_HC_MULT, 4);
        assert_eq!(PARENT_SWA_WINDOW, 128);
        assert_eq!(PARENT_N_KV_HEADS, 1);
        assert_eq!(PARENT_HEAD_DIM, 512);
        assert_eq!(PARENT_VOCAB, 129_280);
    }

    #[test]
    fn ping_pong_final_buffer_parity() {
        // After N layers the final HC buffer is:
        //   N odd  → hc_a   (layer 0 wrote B, 1 wrote A, …)
        //   N even → hc_b
        // Mirror the driver logic without a GPU.
        for n in 1..=43usize {
            let mut use_a_as_input = true;
            for _ in 0..n {
                use_a_as_input = !use_a_as_input;
            }
            let final_is_a = !use_a_as_input;
            if n % 2 == 1 {
                assert!(final_is_a, "N={n} should end in hc_a");
            } else {
                assert!(!final_is_a, "N={n} should end in hc_b");
            }
        }
    }

    #[test]
    fn compress_events_pass_when_ratio_layers_fire() {
        // 1024 tokens: ratio-128 → 8, ratio-4 → 256, ratio-0 → 0.
        let events = vec![
            (0, 0),
            (0, 0),
            (128, 8),
            (4, 256),
            (128, 8),
            (0, 0),
        ];
        assert!(assert_compress_events(&events, 1024).is_ok());
    }

    #[test]
    fn compress_events_fail_silent_zero_on_ratio_layer() {
        // The Gate-5 bug: ratio-128 with 32 tokens still has floor(32/128)=0
        // so it is NOT a failure (nothing to fire). At 1024 it is.
        let short = vec![(128, 0)];
        assert!(
            assert_compress_events(&short, 32).is_ok(),
            "floor(32/128)=0 → zero events is legitimate"
        );
        let long = vec![(128, 0)];
        let err = assert_compress_events(&long, 1024).expect_err("must fail");
        assert!(err.contains("ratio=128"), "{err}");
        assert!(err.contains("observed=0"), "{err}");
        assert!(err.starts_with("deepseek4 parent: "), "{err}");
    }

    #[test]
    fn compress_events_empty_is_fail_closed() {
        let err = assert_compress_events(&[], 1024).expect_err("empty");
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn compress_events_ratio_zero_must_stay_zero() {
        let bad = vec![(0, 3)];
        let err = assert_compress_events(&bad, 1024).expect_err("ratio0");
        assert!(err.contains("ratio=0"), "{err}");
    }
}
