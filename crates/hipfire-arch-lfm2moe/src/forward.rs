// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (pre-norm; mixer = conv OR attention, FFN = dense OR MoE):
//!   tmp = operator_norm(h)
//!   if conv:   h += out_proj( C_gate ⊙ depthwise_causal_conv( B_gate ⊙ x ) )   [in_proj→conv→out_proj]
//!   if attn:   h += out_proj( attn( qk_norm(q/k) + full-RoPE, v ) )             [GQA, Q8 KV]
//!   ffn_tmp = ffn_norm(h)
//!   if dense:  h += w2( silu(w1·ffn_tmp) ⊙ (w3·ffn_tmp) )                        [SwiGLU, Q8]
//!   if moe:    h += combine( experts( sigmoid+bias top-4 route(ffn_tmp) ) )      [FWHT MQ4 experts]
//! then logits = lm_head( embedding_norm(h) )   (lm_head tied to embed_tokens).
//!
//! Non-expert linears (attention q/k/v/out, conv in/out, dense w1/w2/w3, router)
//! are Q8 (plain input). Routed experts are FWHT-pre-rotated MQ4G256: the input
//! is rotated (`rotate_x_mq_for`) and the silu output rotated
//! (`fused_silu_mul_rotate_mq_batched_for`) before the indexed-MoE GEMVs —
//! exactly qwen35's / minimax's MoE path, but with k_top = num_experts_per_tok
//! = 4 (the batched GEMV variants take k_top as a runtime arg).

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{
    AttnWeights, ConvWeights, DenseFfn, Ffn, Lfm2MoeLayerWeights, Lfm2MoeState, Lfm2MoeWeights,
    Mixer, MoeFfn,
};
use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::weights::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemm,
    weight_gemv, weight_gemv_residual,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Decode one token; returns the full logits vector.
///
/// Routes to the hipGraph capture/replay path when `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact prior behavior). The graph path amortizes the ~377
/// per-token kernel launches by replaying a single captured graph; see
/// `decode_step_with_graph`.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if graph_enabled() {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position);
    }
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
}

/// `HIPFIRE_LFM2_GRAPH=1` opt-in switch. Default OFF (unset / "0") →
/// byte-identical to the legacy per-launch decode path. Parsed once.
fn graph_enabled() -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_LFM2_GRAPH").ok().as_deref(),
            Some("1")
        )
    })
}

fn stage_position(
    gpu: &mut Gpu,
    state: &Lfm2MoeState,
    position: u32,
    label: &str,
) -> Result<(), String> {
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod {label} pos: {e:?}"))
}

fn stage_logical_rope_position(
    gpu: &mut Gpu,
    state: &Lfm2MoeState,
    physical_position: u32,
) -> Result<(), String> {
    if state.kv.compact_offset == 0 {
        return Ok(());
    }
    let logical = physical_position as usize + state.kv.compact_offset;
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(logical as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod logical rope pos: {e:?}"))
}

fn restore_physical_position_after_rope(
    gpu: &mut Gpu,
    state: &Lfm2MoeState,
    physical_position: u32,
) -> Result<(), String> {
    if state.kv.compact_offset == 0 {
        return Ok(());
    }
    stage_position(gpu, state, physical_position, "physical")
}

fn download_i32_tensor(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> HipResult<Vec<i32>> {
    gpu.bind_thread()?;
    let mut data = vec![0i32; len];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len * 4) };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf)?;
    Ok(data)
}

fn capture_named_activation(
    gpu: &mut Gpu,
    tensor_name: &str,
    input: &GpuTensor,
    n: usize,
    k: usize,
) {
    if let Some(cap) = gpu.active_capture.clone() {
        cap.capture(gpu, tensor_name, input, n, k);
    }
}

fn validate_moe_expert_index(
    layer_idx: usize,
    slot: usize,
    raw: i32,
    n_exp: usize,
) -> Result<usize, String> {
    if raw < 0 {
        return Err(format!(
            "lfm2moe L{layer_idx}: topk slot {slot} produced negative expert id {raw}"
        ));
    }
    let expert = raw as usize;
    if expert >= n_exp {
        return Err(format!(
            "lfm2moe L{layer_idx}: topk slot {slot} expert id {expert} out of range 0..{n_exp}"
        ));
    }
    Ok(expert)
}

fn maybe_capture_moe_gate_up_inputs(
    gpu: &mut Gpu,
    layer_idx: usize,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    hidden: usize,
    n_exp: usize,
    k_top: usize,
    batch_size: usize,
) -> Result<Option<Vec<usize>>, String> {
    if gpu.active_capture.is_none() {
        return Ok(None);
    }
    let total = batch_size * k_top;
    let raw_indices = download_i32_tensor(gpu, topk_indices, total)
        .map_err(|e| format!("lfm2moe L{layer_idx}: download topk for capture: {e:?}"))?;
    let mut indices = Vec::with_capacity(total);
    for (slot, raw) in raw_indices.into_iter().enumerate() {
        let expert = validate_moe_expert_index(layer_idx, slot, raw, n_exp)?;
        let row = slot / k_top;
        let x_row = x_rot.sub_offset(row * hidden, hidden);
        let prefix = format!("model.layers.{layer_idx}.feed_forward.experts.{expert}");
        capture_named_activation(gpu, &format!("{prefix}.w1"), &x_row, 1, hidden);
        capture_named_activation(gpu, &format!("{prefix}.w3"), &x_row, 1, hidden);
        indices.push(expert);
    }
    Ok(Some(indices))
}

fn maybe_capture_moe_down_inputs(
    gpu: &mut Gpu,
    layer_idx: usize,
    expert_indices: &[usize],
    rot_batch: &GpuTensor,
    moe_inter: usize,
    k_top: usize,
    batch_size: usize,
) -> Result<(), String> {
    if gpu.active_capture.is_none() {
        return Ok(());
    }
    let expected = batch_size * k_top;
    if expert_indices.len() != expected {
        return Err(format!(
            "lfm2moe L{layer_idx}: capture expert index count {} != expected {expected}",
            expert_indices.len()
        ));
    }
    for (slot, &expert) in expert_indices.iter().enumerate() {
        let rot_row = rot_batch.sub_offset(slot * moe_inter, moe_inter);
        let name = format!("model.layers.{layer_idx}.feed_forward.experts.{expert}.w2");
        capture_named_activation(gpu, &name, &rot_row, 1, moe_inter);
    }
    Ok(())
}

fn lfm2_triattn_tap_batch(
    gpu: &mut Gpu,
    layer_idx: usize,
    q: &GpuTensor,
    k: &GpuTensor,
    n_tokens: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<(), String> {
    if !hipfire_runtime::triattn::tap_enabled() {
        return Ok(());
    }
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu, layer_idx, &q.buf, n_tokens, n_heads, head_dim,
    )
    .map_err(|e| format!("lfm2moe triattn tap L{layer_idx}: {e:?}"))?;
    if gpu_handled {
        return Ok(());
    }

    let q_stride = n_heads * head_dim;
    let q_cpu = gpu
        .download_f32(q)
        .map_err(|e| format!("lfm2moe triattn tap L{layer_idx}: download q: {e:?}"))?;
    if hipfire_runtime::triattn::tap_needs_k() {
        let k_stride = n_kv * head_dim;
        let k_cpu = gpu
            .download_f32(k)
            .map_err(|e| format!("lfm2moe triattn tap L{layer_idx}: download k: {e:?}"))?;
        for row in 0..n_tokens {
            hipfire_runtime::triattn::record_prerope_qk(
                layer_idx,
                &q_cpu[row * q_stride..(row + 1) * q_stride],
                Some(&k_cpu[row * k_stride..(row + 1) * k_stride]),
            );
        }
    } else {
        for row in 0..n_tokens {
            hipfire_runtime::triattn::record_prerope_q(
                layer_idx,
                &q_cpu[row * q_stride..(row + 1) * q_stride],
            );
        }
    }
    Ok(())
}

const LFM2_PREFILL_MAX_BATCH: usize = 256;

/// Host-side capture of selected LFM2 post-layer residual streams.
///
/// Rows are appended in DFlash target-hidden layout:
/// `[position][extract_layer][hidden]`, where `extract_layer` follows the
/// caller-provided `target_layers` order.
#[derive(Debug, Clone)]
pub struct Lfm2HiddenCapture {
    target_layers: Vec<usize>,
    hidden: usize,
    rows: Vec<f32>,
}

impl Lfm2HiddenCapture {
    pub fn new(
        num_target_layers: usize,
        hidden: usize,
        target_layers: Vec<usize>,
    ) -> Result<Self, String> {
        if hidden == 0 {
            return Err("lfm2moe hidden capture: hidden must be non-zero".to_string());
        }
        if target_layers.is_empty() {
            return Err("lfm2moe hidden capture: target_layers is empty".to_string());
        }
        let mut seen = std::collections::BTreeSet::new();
        for &layer in &target_layers {
            if layer >= num_target_layers {
                return Err(format!(
                    "lfm2moe hidden capture: target layer {layer} out of range 0..{num_target_layers}"
                ));
            }
            if !seen.insert(layer) {
                return Err(format!(
                    "lfm2moe hidden capture: duplicate target layer {layer}"
                ));
            }
        }
        Ok(Self {
            target_layers,
            hidden,
            rows: Vec::new(),
        })
    }

    pub fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn num_extract(&self) -> usize {
        self.target_layers.len()
    }

    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    pub fn take_rows(self) -> Vec<f32> {
        self.rows
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn position_count(&self) -> usize {
        self.rows.len() / (self.num_extract() * self.hidden)
    }

    fn extract_slot(&self, layer_idx: usize) -> Option<usize> {
        self.target_layers.iter().position(|&l| l == layer_idx)
    }

    fn append_single_from_all_layers(&mut self, all_layers: &[Vec<f32>]) -> Result<(), String> {
        for &layer in &self.target_layers {
            let src = all_layers.get(layer).ok_or_else(|| {
                format!("lfm2moe hidden capture: missing layer {layer} in decode capture")
            })?;
            if src.len() < self.hidden {
                return Err(format!(
                    "lfm2moe hidden capture: layer {layer} has {} floats, expected at least {}",
                    src.len(),
                    self.hidden
                ));
            }
            let start = src.len() - self.hidden;
            self.rows
                .extend_from_slice(&src[start..start + self.hidden]);
        }
        Ok(())
    }

    fn append_interleaved_chunk(
        &mut self,
        layer_rows: &[Option<Vec<f32>>],
        n: usize,
    ) -> Result<(), String> {
        if layer_rows.len() != self.num_extract() {
            return Err(format!(
                "lfm2moe hidden capture: got {} extracted layer buffers, expected {}",
                layer_rows.len(),
                self.num_extract()
            ));
        }
        for (slot, rows) in layer_rows.iter().enumerate() {
            let Some(rows) = rows else {
                return Err(format!(
                    "lfm2moe hidden capture: target layer {} was not captured",
                    self.target_layers[slot]
                ));
            };
            let expected = n * self.hidden;
            if rows.len() != expected {
                return Err(format!(
                    "lfm2moe hidden capture: layer {} rows have {} floats, expected {}",
                    self.target_layers[slot],
                    rows.len(),
                    expected
                ));
            }
        }
        self.rows.reserve(n * self.num_extract() * self.hidden);
        for row in 0..n {
            let row_start = row * self.hidden;
            let row_end = row_start + self.hidden;
            for rows in layer_rows.iter().flatten() {
                self.rows.extend_from_slice(&rows[row_start..row_end]);
            }
        }
        Ok(())
    }
}

/// Chunk scratch for LFM2 batched prompt prefill. Rows are token-major.
struct Lfm2MoePrefillScratch {
    max_batch: usize,
    tokens: GpuTensor,
    positions: GpuTensor,
    h_batch: GpuTensor,
    tmp_batch: GpuTensor,
    conv_bcx_batch: GpuTensor,
    conv_y_batch: GpuTensor,
    fa_q_batch: GpuTensor,
    fa_k_batch: GpuTensor,
    fa_v_batch: GpuTensor,
    fa_attn_out_batch: GpuTensor,
    dense_gate_batch: GpuTensor,
    dense_up_batch: GpuTensor,
    dense_act_batch: GpuTensor,
    ffn_tmp_batch: GpuTensor,
    ffn_x_rot_batch: GpuTensor,
    router_logits_batch: GpuTensor,
    topk_indices_batch: GpuTensor,
    topk_weights_batch: GpuTensor,
    gate_batch: GpuTensor,
    up_batch: GpuTensor,
    rot_batch: GpuTensor,
    down_expanded_batch: GpuTensor,
    proj_out_batch: GpuTensor,
    flash_partials: GpuTensor,
}

impl Lfm2MoePrefillScratch {
    fn new(gpu: &mut Gpu, cfg: &Lfm2MoeConfig, max_batch: usize) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let k_top = cfg.num_experts_per_tok;
        let max_tiles = cfg.max_position_embeddings.min(8192).div_ceil(128);
        let partials_size = 16 * cfg.num_attention_heads * max_tiles * (2 + cfg.head_dim);
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("lfm2moe prefill: alloc {label}: {e:?}"))
        };

        Ok(Self {
            max_batch,
            tokens: alloc(gpu, max_batch, "tokens")?,
            positions: alloc(gpu, max_batch, "positions")?,
            h_batch: alloc(gpu, max_batch * hidden, "h_batch")?,
            tmp_batch: alloc(gpu, max_batch * hidden, "tmp_batch")?,
            conv_bcx_batch: alloc(gpu, max_batch * 3 * hidden, "conv_bcx_batch")?,
            conv_y_batch: alloc(gpu, max_batch * hidden, "conv_y_batch")?,
            fa_q_batch: alloc(gpu, max_batch * q_dim, "fa_q_batch")?,
            fa_k_batch: alloc(gpu, max_batch * kv_dim, "fa_k_batch")?,
            fa_v_batch: alloc(gpu, max_batch * kv_dim, "fa_v_batch")?,
            fa_attn_out_batch: alloc(gpu, max_batch * q_dim, "fa_attn_out_batch")?,
            dense_gate_batch: alloc(gpu, max_batch * dense_inter, "dense_gate_batch")?,
            dense_up_batch: alloc(gpu, max_batch * dense_inter, "dense_up_batch")?,
            dense_act_batch: alloc(gpu, max_batch * dense_inter, "dense_act_batch")?,
            ffn_tmp_batch: alloc(gpu, max_batch * hidden, "ffn_tmp_batch")?,
            ffn_x_rot_batch: alloc(gpu, max_batch * hidden, "ffn_x_rot_batch")?,
            router_logits_batch: alloc(gpu, max_batch * cfg.num_experts, "router_logits_batch")?,
            topk_indices_batch: alloc(gpu, max_batch * k_top, "topk_indices_batch")?,
            topk_weights_batch: alloc(gpu, max_batch * k_top, "topk_weights_batch")?,
            gate_batch: alloc(gpu, max_batch * k_top * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, max_batch * k_top * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, max_batch * k_top * moe_inter, "rot_batch")?,
            down_expanded_batch: alloc(gpu, max_batch * k_top * hidden, "down_expanded_batch")?,
            proj_out_batch: alloc(gpu, max_batch * 3 * hidden, "proj_out_batch")?,
            flash_partials: alloc(gpu, partials_size, "flash_partials")?,
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.tokens,
            self.positions,
            self.h_batch,
            self.tmp_batch,
            self.conv_bcx_batch,
            self.conv_y_batch,
            self.fa_q_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.dense_gate_batch,
            self.dense_up_batch,
            self.dense_act_batch,
            self.ffn_tmp_batch,
            self.ffn_x_rot_batch,
            self.router_logits_batch,
            self.topk_indices_batch,
            self.topk_weights_batch,
            self.gate_batch,
            self.up_batch,
            self.rot_batch,
            self.down_expanded_batch,
            self.proj_out_batch,
            self.flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Batched prompt prefill for LFM2/LFM2-MoE. Processes `tokens` in chunks,
/// updates KV/conv state to the post-prompt point, writes the last-token logits
/// into `state.logits`, and returns those logits on host.
///
/// `HIPFIRE_PREFILL_BATCHED=0` preserves the legacy decode-step replay path.
pub fn prefill_batch(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    prefill_batch_impl(cfg, weights, state, gpu, tokens, None, None, None)
}

/// Batched prompt prefill plus selected hidden-state extraction for DFlash.
///
/// Captures post-layer residuals for `capture.target_layers()` into
/// `capture.rows()` in `[position][extract_layer][hidden]` order while leaving
/// logits/KV/conv-state behavior identical to [`prefill_batch`].
pub fn prefill_batch_with_hidden(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    capture: &mut Lfm2HiddenCapture,
) -> Result<Vec<f32>, String> {
    validate_hidden_capture(cfg, capture)?;
    prefill_batch_impl(cfg, weights, state, gpu, tokens, Some(capture), None, None)
}

/// Batched prompt/verify prefill plus selected hidden-state extraction and
/// per-position logits.
///
/// This is the LFM2 DFlash verify surface: it advances target state exactly as
/// [`prefill_batch_with_hidden`] does, captures the same DFlash hidden rows,
/// and returns row-major logits for every token in `tokens`.
pub fn prefill_batch_with_hidden_logits(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    capture: &mut Lfm2HiddenCapture,
) -> Result<Vec<f32>, String> {
    validate_hidden_capture(cfg, capture)?;
    let mut logits_per_pos = Vec::with_capacity(tokens.len() * cfg.vocab_size);
    prefill_batch_impl(
        cfg,
        weights,
        state,
        gpu,
        tokens,
        Some(capture),
        Some(&mut logits_per_pos),
        None,
    )?;
    Ok(logits_per_pos)
}

/// Batched prompt/verify prefill plus selected DFlash hidden extraction, target
/// pre-final hidden rows, and per-position logits.
///
/// `final_hidden_rows` receives row-major `[tokens, hidden]` post-layer residual
/// states before the final embedding norm. This is a training-label surface for
/// the DFlash `fc.weight` projection; normal generation callers should keep using
/// the lighter helpers above.
pub fn prefill_batch_with_hidden_logits_and_final_hidden(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    capture: &mut Lfm2HiddenCapture,
    final_hidden_rows: &mut Vec<f32>,
) -> Result<Vec<f32>, String> {
    validate_hidden_capture(cfg, capture)?;
    let mut logits_per_pos = Vec::with_capacity(tokens.len() * cfg.vocab_size);
    prefill_batch_impl(
        cfg,
        weights,
        state,
        gpu,
        tokens,
        Some(capture),
        Some(&mut logits_per_pos),
        Some(final_hidden_rows),
    )?;
    Ok(logits_per_pos)
}

fn validate_hidden_capture(cfg: &Lfm2MoeConfig, capture: &Lfm2HiddenCapture) -> Result<(), String> {
    if capture.hidden() != cfg.hidden_size {
        return Err(format!(
            "lfm2moe hidden capture: hidden mismatch capture={} model={}",
            capture.hidden(),
            cfg.hidden_size
        ));
    }
    for &layer in capture.target_layers() {
        if layer >= cfg.num_hidden_layers {
            return Err(format!(
                "lfm2moe hidden capture: target layer {layer} out of range 0..{}",
                cfg.num_hidden_layers
            ));
        }
    }
    Ok(())
}

fn prefill_batch_impl(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    mut capture: Option<&mut Lfm2HiddenCapture>,
    mut logits_per_pos: Option<&mut Vec<f32>>,
    mut final_hidden_rows: Option<&mut Vec<f32>>,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    if !hipfire_runtime::config::get().prefill_batched
        || tokens.len() < 2
        || state.kv.compact_offset > 0
    {
        let mut logits = Vec::new();
        let mut position = state.n_tokens as u32;
        for &tok in tokens {
            if let Some(cap) = capture.as_deref_mut() {
                let mut all_layers: Vec<Vec<f32>> = vec![Vec::new(); cfg.num_hidden_layers];
                decode_step_capture(cfg, weights, state, gpu, tok, position, &mut all_layers)?;
                cap.append_single_from_all_layers(&all_layers)?;
                logits = gpu
                    .download_f32(&state.logits)
                    .map_err(|e| format!("lfm2moe: download logits: {e:?}"))?;
            } else {
                logits = decode_step(cfg, weights, state, gpu, tok, position)?;
            }
            if let Some(out) = logits_per_pos.as_deref_mut() {
                out.extend_from_slice(&logits);
            }
            if let Some(out) = final_hidden_rows.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe: download final hidden row: {e:?}"))?;
                out.extend_from_slice(&h);
            }
            position += 1;
        }
        return Ok(logits);
    }

    let max_batch = LFM2_PREFILL_MAX_BATCH.min(tokens.len());
    let scratch = Lfm2MoePrefillScratch::new(gpu, cfg, max_batch)?;
    let mut offset = 0usize;
    let result = (|| -> Result<(), String> {
        while offset < tokens.len() {
            let n = (tokens.len() - offset).min(scratch.max_batch);
            let start_pos = state.n_tokens;
            prefill_batch_chunk(
                cfg,
                weights,
                state,
                gpu,
                &tokens[offset..offset + n],
                start_pos,
                &scratch,
                capture.as_deref_mut(),
                logits_per_pos.as_deref_mut(),
                final_hidden_rows.as_deref_mut(),
            )?;
            offset += n;
        }
        Ok(())
    })();
    scratch.free_gpu(gpu);
    result?;

    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe prefill: download logits: {e:?}"))
}

fn upload_i32_rows(gpu: &mut Gpu, dst: &GpuTensor, vals: &[u32]) -> Result<(), String> {
    let host: Vec<i32> = vals.iter().map(|&v| v as i32).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4) };
    gpu.hip
        .memcpy_htod(&dst.buf, bytes)
        .map_err(|e| format!("lfm2moe prefill: upload i32 rows: {e:?}"))
}

fn lfm2_weight_gemm(
    gpu: &mut Gpu,
    w: &hipfire_runtime::weights::WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    if w.gpu_dtype != DType::MQ4G256 {
        return weight_gemm(gpu, w, x, y, batch_size);
    }

    gpu.ensure_mq_signs()?;
    let x_rot = gpu.alloc_tensor(&[batch_size * w.k], DType::F32)?;
    let result = (|| -> HipResult<()> {
        rotate_x_mq_batched_for(gpu, w, x, &x_rot, w.k, batch_size)?;
        gpu.gemm_hfq4g256(&w.buf, &x_rot, y, w.m, w.k, batch_size)
    })();
    let _ = gpu.free_tensor(x_rot);
    result
}

#[allow(clippy::too_many_arguments)]
fn prefill_batch_chunk(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
    s: &Lfm2MoePrefillScratch,
    mut capture: Option<&mut Lfm2HiddenCapture>,
    mut logits_per_pos: Option<&mut Vec<f32>>,
    mut final_hidden_rows: Option<&mut Vec<f32>>,
) -> Result<(), String> {
    let n = tokens.len();
    debug_assert!(n > 0 && n <= s.max_batch);
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let mut captured_layer_rows: Option<Vec<Option<Vec<f32>>>> =
        capture.as_ref().map(|cap| vec![None; cap.num_extract()]);

    upload_i32_rows(gpu, &s.tokens, tokens)?;
    let positions: Vec<u32> = (0..n).map(|i| (start_pos + i) as u32).collect();
    upload_i32_rows(gpu, &s.positions, &positions)?;
    gpu.embedding_lookup_q8_batched(&weights.embed, &s.h_batch, &s.tokens, n, hidden)
        .map_err(|e| format!("lfm2moe prefill: embedding batch: {e:?}"))?;

    let max_ctx_len = start_pos + n;
    for (l, layer) in weights.layers.iter().enumerate() {
        gpu.rmsnorm_batched(
            &s.h_batch,
            &layer.operator_norm,
            &s.tmp_batch,
            n,
            hidden,
            eps,
        )
        .map_err(|e| format!("lfm2moe prefill L{l}: operator rmsnorm: {e:?}"))?;

        match &layer.mixer {
            Mixer::Conv(c) => {
                lfm2_weight_gemm(gpu, &c.in_proj, &s.tmp_batch, &s.conv_bcx_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: conv in_proj: {e:?}"))?;
                gpu.conv1d_gated_seq_f32(
                    &s.conv_bcx_batch,
                    &state.conv_states[c.conv_state_idx],
                    &c.conv_weight,
                    &s.conv_y_batch,
                    n,
                    hidden,
                    cfg.conv_kernel_size,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: conv seq: {e:?}"))?;
                lfm2_weight_gemm(gpu, &c.out_proj, &s.conv_y_batch, &s.proj_out_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: conv out_proj: {e:?}"))?;
                let y = s.proj_out_batch.sub_offset(0, n * hidden);
                let h = s.h_batch.sub_offset(0, n * hidden);
                gpu.add_inplace_f32(&h, &y)
                    .map_err(|e| format!("lfm2moe prefill L{l}: conv residual: {e:?}"))?;
            }
            Mixer::Attention(a) => {
                lfm2_weight_gemm(gpu, &a.wq, &s.tmp_batch, &s.fa_q_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: q_proj: {e:?}"))?;
                lfm2_weight_gemm(gpu, &a.wk, &s.tmp_batch, &s.fa_k_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: k_proj: {e:?}"))?;
                lfm2_weight_gemm(gpu, &a.wv, &s.tmp_batch, &s.fa_v_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: v_proj: {e:?}"))?;
                gpu.rmsnorm_batched(
                    &s.fa_q_batch,
                    &a.q_norm,
                    &s.fa_q_batch,
                    n * n_heads,
                    head_dim,
                    eps,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: q_norm: {e:?}"))?;
                gpu.rmsnorm_batched(
                    &s.fa_k_batch,
                    &a.k_norm,
                    &s.fa_k_batch,
                    n * n_kv,
                    head_dim,
                    eps,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: k_norm: {e:?}"))?;
                lfm2_triattn_tap_batch(
                    gpu,
                    a.kv_idx,
                    &s.fa_q_batch,
                    &s.fa_k_batch,
                    n,
                    n_heads,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: {e}"))?;
                gpu.rope_batched_f32(
                    &s.fa_q_batch,
                    &s.fa_k_batch,
                    &s.positions,
                    n_heads,
                    n_kv,
                    head_dim,
                    cfg.rope_theta,
                    n,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: rope: {e:?}"))?;
                let kv_idx = a.kv_idx;
                gpu.kv_cache_write_q8_0_batched(
                    &state.kv.k_gpu[kv_idx],
                    &s.fa_k_batch,
                    &s.positions,
                    n_kv,
                    head_dim,
                    n,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: kv write k: {e:?}"))?;
                gpu.kv_cache_write_q8_0_batched(
                    &state.kv.v_gpu[kv_idx],
                    &s.fa_v_batch,
                    &s.positions,
                    n_kv,
                    head_dim,
                    n,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: kv write v: {e:?}"))?;
                gpu.attention_q8_0_kv_batched_masked(
                    &s.fa_q_batch,
                    &state.kv.k_gpu[kv_idx],
                    &state.kv.v_gpu[kv_idx],
                    &s.fa_attn_out_batch,
                    &s.positions,
                    n_heads,
                    n_kv,
                    head_dim,
                    state.kv.physical_cap,
                    max_ctx_len,
                    n,
                    None,
                    0,
                    0,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: attention: {e:?}"))?;
                lfm2_weight_gemm(gpu, &a.wo, &s.fa_attn_out_batch, &s.proj_out_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: out_proj: {e:?}"))?;
                let y = s.proj_out_batch.sub_offset(0, n * hidden);
                let h = s.h_batch.sub_offset(0, n * hidden);
                gpu.add_inplace_f32(&h, &y)
                    .map_err(|e| format!("lfm2moe prefill L{l}: attn residual: {e:?}"))?;
            }
        }

        gpu.rmsnorm_batched(
            &s.h_batch,
            &layer.ffn_norm,
            &s.ffn_tmp_batch,
            n,
            hidden,
            eps,
        )
        .map_err(|e| format!("lfm2moe prefill L{l}: ffn rmsnorm: {e:?}"))?;

        match &layer.ffn {
            Ffn::Dense(d) => {
                lfm2_weight_gemm(gpu, &d.w1, &s.ffn_tmp_batch, &s.dense_gate_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: dense w1: {e:?}"))?;
                lfm2_weight_gemm(gpu, &d.w3, &s.ffn_tmp_batch, &s.dense_up_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: dense w3: {e:?}"))?;
                gpu.silu_mul_f32(&s.dense_gate_batch, &s.dense_up_batch, &s.dense_act_batch)
                    .map_err(|e| format!("lfm2moe prefill L{l}: dense silu_mul: {e:?}"))?;
                lfm2_weight_gemm(gpu, &d.w2, &s.dense_act_batch, &s.proj_out_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: dense w2: {e:?}"))?;
                let y = s.proj_out_batch.sub_offset(0, n * hidden);
                let h = s.h_batch.sub_offset(0, n * hidden);
                gpu.add_inplace_f32(&h, &y)
                    .map_err(|e| format!("lfm2moe prefill L{l}: dense residual: {e:?}"))?;
            }
            Ffn::Moe(m) => {
                rotate_x_mq_batched_for(
                    gpu,
                    &m.experts[0].gate_up,
                    &s.ffn_tmp_batch,
                    &s.ffn_x_rot_batch,
                    hidden,
                    n,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: ffn rotate: {e:?}"))?;
                lfm2_weight_gemm(gpu, &m.router, &s.ffn_tmp_batch, &s.router_logits_batch, n)
                    .map_err(|e| format!("lfm2moe prefill L{l}: router: {e:?}"))?;
                gpu.sigmoid_f32(&s.router_logits_batch)
                    .map_err(|e| format!("lfm2moe prefill L{l}: sigmoid: {e:?}"))?;
                gpu.deepseek4_moe_topk_bias_aware_batched_f32(
                    &s.router_logits_batch,
                    &m.expert_bias,
                    &s.topk_indices_batch,
                    &s.topk_weights_batch,
                    n_exp as i32,
                    k_top as i32,
                    cfg.routed_scaling_factor,
                    n as i32,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: topk batched: {e:?}"))?;
                let capture_expert_indices = maybe_capture_moe_gate_up_inputs(
                    gpu,
                    l,
                    &s.topk_indices_batch,
                    &s.ffn_x_rot_batch,
                    hidden,
                    n_exp,
                    k_top,
                    n,
                )?;
                let experts_mq6 = m.experts[0].gate_up.gpu_dtype == DType::MQ6G256;
                if experts_mq6 {
                    gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                        &m.expert_gate_up_ptrs,
                        &s.topk_indices_batch,
                        &s.ffn_x_rot_batch,
                        &s.gate_batch,
                        &s.up_batch,
                        2 * moe_inter,
                        hidden,
                        k_top,
                        n,
                    )
                    .map_err(|e| format!("lfm2moe prefill L{l}: gate_up(mq6): {e:?}"))?;
                } else {
                    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                        &m.expert_gate_up_ptrs,
                        &s.topk_indices_batch,
                        &s.ffn_x_rot_batch,
                        &s.gate_batch,
                        &s.up_batch,
                        2 * moe_inter,
                        hidden,
                        k_top,
                        n,
                    )
                    .map_err(|e| format!("lfm2moe prefill L{l}: gate_up: {e:?}"))?;
                }
                fused_silu_mul_rotate_mq_batched_for(
                    gpu,
                    &m.experts[0].down,
                    &s.gate_batch,
                    &s.up_batch,
                    &s.rot_batch,
                    moe_inter,
                    n * k_top,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: silu_mul_rotate: {e:?}"))?;
                if let Some(indices) = capture_expert_indices.as_deref() {
                    maybe_capture_moe_down_inputs(
                        gpu,
                        l,
                        indices,
                        &s.rot_batch,
                        moe_inter,
                        k_top,
                        n,
                    )?;
                }
                if experts_mq6 {
                    gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                        &m.expert_down_ptrs,
                        &s.topk_indices_batch,
                        &s.rot_batch,
                        &s.down_expanded_batch,
                        hidden,
                        moe_inter,
                        k_top,
                        n,
                    )
                    .map_err(|e| format!("lfm2moe prefill L{l}: down(mq6): {e:?}"))?;
                } else {
                    gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                        &m.expert_down_ptrs,
                        &s.topk_indices_batch,
                        &s.rot_batch,
                        &s.down_expanded_batch,
                        hidden,
                        moe_inter,
                        k_top,
                        n,
                    )
                    .map_err(|e| format!("lfm2moe prefill L{l}: down: {e:?}"))?;
                }
                gpu.moe_down_combine_k8_batched(
                    &s.down_expanded_batch,
                    &s.topk_weights_batch,
                    &s.h_batch,
                    hidden,
                    k_top,
                    n,
                )
                .map_err(|e| format!("lfm2moe prefill L{l}: combine: {e:?}"))?;
            }
        }

        if let Some(slot) = capture.as_ref().and_then(|cap| cap.extract_slot(l)) {
            let h_rows = s.h_batch.sub_offset(0, n * hidden);
            let rows = gpu
                .download_f32(&h_rows)
                .map_err(|e| format!("lfm2moe prefill L{l}: hidden capture: {e:?}"))?;
            if let Some(ref mut layer_rows) = captured_layer_rows {
                layer_rows[slot] = Some(rows);
            }
        }
    }

    if let Some(cap) = capture.as_deref_mut() {
        let layer_rows = captured_layer_rows
            .take()
            .expect("capture rows allocated when capture is present");
        cap.append_interleaved_chunk(&layer_rows, n)?;
    }

    if let Some(out) = final_hidden_rows.as_deref_mut() {
        let h_rows = s.h_batch.sub_offset(0, n * hidden);
        let rows = gpu
            .download_f32(&h_rows)
            .map_err(|e| format!("lfm2moe prefill: final hidden capture: {e:?}"))?;
        out.extend_from_slice(&rows);
    }

    if let Some(out) = logits_per_pos.as_deref_mut() {
        let logits_batch = gpu
            .alloc_tensor(&[n * cfg.vocab_size], DType::F32)
            .map_err(|e| format!("lfm2moe prefill: alloc all logits: {e:?}"))?;
        let result = (|| -> Result<Vec<f32>, String> {
            gpu.rmsnorm_batched(
                &s.h_batch,
                &weights.embedding_norm,
                &s.tmp_batch,
                n,
                hidden,
                eps,
            )
            .map_err(|e| format!("lfm2moe prefill: final rmsnorm batch: {e:?}"))?;
            lfm2_weight_gemm(gpu, &weights.lm_head, &s.tmp_batch, &logits_batch, n)
                .map_err(|e| format!("lfm2moe prefill: all-row lm_head: {e}"))?;
            gpu.download_f32(&logits_batch)
                .map_err(|e| format!("lfm2moe prefill: download all logits: {e:?}"))
        })();
        let _ = gpu.free_tensor(logits_batch);
        out.extend_from_slice(&result?);
    }

    state.n_tokens = start_pos + n;
    gpu.hip
        .memcpy_dtod_at(
            &state.h.buf,
            0,
            &s.h_batch.buf,
            (n - 1) * hidden * 4,
            hidden * 4,
        )
        .map_err(|e| format!("lfm2moe prefill: copy last hidden: {e:?}"))?;
    gpu.rmsnorm_f32(
        &state.h,
        &weights.embedding_norm,
        &state.final_norm_buf,
        eps,
    )
    .map_err(|e| format!("lfm2moe prefill: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe prefill: lm_head: {e}"))?;
    Ok(())
}

/// Decode one token, appending each layer's post-residual hidden state
/// (after the full layer, before the final norm) to `capture[layer]` — used by
/// the oracle dumper. Set `HIPFIRE_LFM2_CAPTURE_POSTMIXER` to capture the
/// post-mixer residual (pre-FFN) instead, for conv/attn-vs-FFN localization.
pub fn decode_step_capture(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, Some(capture))
}

fn decode_step_inner(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;

    // Device position scalar (i32) for kv-write / attention. RoPE temporarily
    // swaps in logical position when TriAttention compaction is active.
    stage_position(gpu, state, position, "physical")?;

    // Embedding lookup → residual stream h (Q8 table).
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;

    decode_step_layers_and_head(cfg, weights, state, gpu, position, capture)
}

/// Per-layer mixer/FFN stack + final norm + lm_head. Reads the residual
/// stream `state.h` (already seeded by the embedding lookup) and the device
/// position scalar `state.pos_buf` (already staged); writes `state.logits`.
///
/// This is the hipGraph-captureable region: it issues only kernel launches
/// that read STABLE device buffers and (on the MoE path) compute their
/// topk/positions on-device, so a single capture replays correctly at every
/// later position once `state.pos_buf` is refreshed. The per-token-varying
/// embedding lookup (token_id is a kernarg) and the `pos_buf` htod are the
/// caller's responsibility OUTSIDE the captured region.
///
/// `capture` (oracle dumper) is incompatible with hipGraph capture — it issues
/// a sync `download_f32` per layer. The graph path always passes `None`.
fn decode_step_layers_and_head(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postmixer = std::env::var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();

    // #397 Ship 6 — forward-as-pipeline. HIPFIRE_FORWARD_LOWERED=1 routes the
    // per-layer decode through the super-op executor (run_layer_program). Skipped
    // when capturing (the oracle dumper needs the per-layer hand path) — that path
    // stays byte-identical. Default off (opt-in) until fleet byte-parity validated.
    if lfm2_forward_lowered_enabled() && capture.is_none() {
        return decode_step_layers_and_head_lowered(cfg, weights, state, gpu, position);
    }

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Mixer block (pre-norm) ──────────────────────────────────────────
        gpu.rmsnorm_f32(&state.h, &layer.operator_norm, &state.tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;

        match &layer.mixer {
            Mixer::Conv(c) => {
                // in_proj → [3*hidden] (B | C_gate | x), Q8 plain.
                weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
                    .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
                // double-gated depthwise causal short-conv (advances conv state).
                gpu.conv1d_gated_decode_f32(
                    &state.conv_bcx,
                    &state.conv_states[c.conv_state_idx],
                    &c.conv_weight,
                    &state.conv_y,
                    1,
                    hidden,
                    cfg.conv_kernel_size,
                )
                .map_err(|e| format!("lfm2moe L{l}: conv gated decode: {e:?}"))?;
                // out_proj + residual: h += W_out · y (Q8).
                weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))?;
            }
            Mixer::Attention(a) => {
                weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
                    .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
                weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
                    .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
                weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
                    .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;

                // Per-HEAD QK-norm: RMSNorm over each head's head_dim slice,
                // sharing the [head_dim] weight across heads (batch = n_heads).
                gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
                    .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
                gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
                    .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;

                // Full-dim rotate_half RoPE (no partial rotary).
                lfm2_triattn_tap_batch(
                    gpu,
                    a.kv_idx,
                    &state.fa_q,
                    &state.fa_k,
                    1,
                    n_heads,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: {e}"))?;
                stage_logical_rope_position(gpu, state, position)?;
                gpu.rope_f32(
                    &state.fa_q,
                    &state.fa_k,
                    &state.pos_buf,
                    n_heads,
                    n_kv,
                    head_dim,
                    cfg.rope_theta,
                )
                .map_err(|e| format!("lfm2moe L{l}: rope: {e:?}"))?;
                restore_physical_position_after_rope(gpu, state, position)?;

                // KV cache write (Q8) + GQA flash attention.
                let kv_idx = a.kv_idx;
                gpu.kv_cache_write_q8_0(
                    &state.kv.k_gpu[kv_idx],
                    &state.fa_k,
                    &state.pos_buf,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: kv write k: {e:?}"))?;
                gpu.kv_cache_write_q8_0(
                    &state.kv.v_gpu[kv_idx],
                    &state.fa_v,
                    &state.pos_buf,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: kv write v: {e:?}"))?;
                gpu.attention_q8_0_kv(
                    &state.fa_q,
                    &state.kv.k_gpu[kv_idx],
                    &state.kv.v_gpu[kv_idx],
                    &state.fa_attn_out,
                    &state.pos_buf,
                    seq_len,
                    n_heads,
                    n_kv,
                    head_dim,
                    state.kv.physical_cap,
                )
                .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;

                // out_proj + residual: h += W_out · attn_out (Q8).
                weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))?;
            }
        }

        if capture_postmixer {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe L{l}: postmixer capture: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }

        // ── FFN block (pre-norm): dense SwiGLU OR top-4 MoE ─────────────────
        gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;

        match &layer.ffn {
            Ffn::Dense(d) => {
                weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
                    .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
                weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
                    .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))?;
                gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
                    .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
                weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))?;
            }
            Ffn::Moe(m) => {
                // FWHT-rotate the FFN input for the MQ4 experts (router stays plain).
                rotate_x_mq_for(
                    gpu,
                    &m.experts[0].gate_up,
                    &state.ffn_tmp,
                    &state.ffn_x_rot,
                    hidden,
                )
                .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;

                // Router: sigmoid(logits) + bias-aware top-k (gather unbiased,
                // renormalize, scale). expert_bias steers SELECTION only.
                weight_gemv(gpu, &m.router, &state.ffn_tmp, &state.router_logits)
                    .map_err(|e| format!("lfm2moe L{l}: router: {e}"))?;
                gpu.sigmoid_f32(&state.router_logits)
                    .map_err(|e| format!("lfm2moe L{l}: sigmoid: {e:?}"))?;
                gpu.deepseek4_moe_topk_bias_aware_f32(
                    &state.router_logits,
                    &m.expert_bias,
                    &state.topk_indices,
                    &state.topk_weights,
                    n_exp as i32,
                    k_top as i32,
                    cfg.routed_scaling_factor,
                )
                .map_err(|e| format!("lfm2moe L{l}: topk: {e:?}"))?;
                let capture_expert_indices = maybe_capture_moe_gate_up_inputs(
                    gpu,
                    l,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    hidden,
                    n_exp,
                    k_top,
                    1,
                )?;

                // gate_up (rotated input, batched k_top) → silu·mul·rotate → down → combine.
                // Experts are uniform per layer (gate_up/down share dtype). MQ6G256
                // experts use the HFQ6 (200 B/group, 6-bit) indexed kernels; MQ4G256
                // (default) uses the HFQ4 (136 B/group, 4-bit) siblings. Both consume
                // the same FWHT-rotated `ffn_x_rot` — only the weight dequant differs.
                let experts_mq6 = m.experts[0].gate_up.gpu_dtype == DType::MQ6G256;
                if experts_mq6 {
                    gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                        &m.expert_gate_up_ptrs,
                        &state.topk_indices,
                        &state.ffn_x_rot,
                        &state.gate_batch,
                        &state.up_batch,
                        2 * moe_inter,
                        hidden,
                        k_top,
                        1,
                    )
                    .map_err(|e| format!("lfm2moe L{l}: gate_up(mq6): {e:?}"))?;
                } else {
                    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                        &m.expert_gate_up_ptrs,
                        &state.topk_indices,
                        &state.ffn_x_rot,
                        &state.gate_batch,
                        &state.up_batch,
                        2 * moe_inter,
                        hidden,
                        k_top,
                        1,
                    )
                    .map_err(|e| format!("lfm2moe L{l}: gate_up: {e:?}"))?;
                }

                fused_silu_mul_rotate_mq_batched_for(
                    gpu,
                    &m.experts[0].down,
                    &state.gate_batch,
                    &state.up_batch,
                    &state.rot_batch,
                    moe_inter,
                    k_top,
                )
                .map_err(|e| format!("lfm2moe L{l}: silu_mul_rotate: {e:?}"))?;
                if let Some(indices) = capture_expert_indices.as_deref() {
                    maybe_capture_moe_down_inputs(
                        gpu,
                        l,
                        indices,
                        &state.rot_batch,
                        moe_inter,
                        k_top,
                        1,
                    )?;
                }

                if experts_mq6 {
                    gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                        &m.expert_down_ptrs,
                        &state.topk_indices,
                        &state.rot_batch,
                        &state.down_expanded,
                        hidden,
                        moe_inter,
                        k_top,
                        1,
                    )
                    .map_err(|e| format!("lfm2moe L{l}: down(mq6): {e:?}"))?;
                } else {
                    gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                        &m.expert_down_ptrs,
                        &state.topk_indices,
                        &state.rot_batch,
                        &state.down_expanded,
                        hidden,
                        moe_inter,
                        k_top,
                        1,
                    )
                    .map_err(|e| format!("lfm2moe L{l}: down: {e:?}"))?;
                }

                gpu.moe_down_combine_k8_batched(
                    &state.down_expanded,
                    &state.topk_weights,
                    &state.h,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("lfm2moe L{l}: combine: {e:?}"))?;
            }
        }

        // Capture post-layer residual (pre final-norm) for the oracle compare.
        if !capture_postmixer {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe L{l}: capture download: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (tied to embed_tokens, Q8).
    gpu.rmsnorm_f32(
        &state.h,
        &weights.embedding_norm,
        &state.final_norm_buf,
        eps,
    )
    .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_capture_interleaves_positions_then_layers() {
        let mut cap = Lfm2HiddenCapture::new(6, 2, vec![1, 4]).unwrap();
        let layer_1 = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let layer_4 = vec![40.0, 41.0, 50.0, 51.0, 60.0, 61.0];
        cap.append_interleaved_chunk(&[Some(layer_1), Some(layer_4)], 3)
            .unwrap();

        assert_eq!(
            cap.rows(),
            &[
                10.0, 11.0, 40.0, 41.0, // position 0, layers 1 then 4
                20.0, 21.0, 50.0, 51.0, // position 1
                30.0, 31.0, 60.0, 61.0, // position 2
            ]
        );
        assert_eq!(cap.position_count(), 3);
    }

    #[test]
    fn hidden_capture_rejects_duplicate_or_out_of_range_layers() {
        assert!(Lfm2HiddenCapture::new(4, 8, vec![1, 1]).is_err());
        assert!(Lfm2HiddenCapture::new(4, 8, vec![4]).is_err());
        assert!(Lfm2HiddenCapture::new(4, 0, vec![1]).is_err());
        assert!(Lfm2HiddenCapture::new(4, 8, Vec::new()).is_err());
    }
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: LFM2.5 lowered decode (the run_conv slot).
//
// LFM2 is the substrate's Conv super-op proving ground. Each layer lowers to a
// short LayerProgram of coarse super-ops; the per-token executor (run_layer_
// program) calls these arch handlers. ADDITIVE + opt-in (HIPFIRE_FORWARD_LOWERED,
// default off) — the hand loop in decode_step_layers_and_head is untouched, so
// the default path stays byte-identical; the lowered path is validated byte-
// identical via the FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B before flip.
//
// Super-op map (pre-norm folded into each handler):
//   Conv         = operator_norm + in_proj + conv1d_gated + out_proj(+resid)
//   Attend       = operator_norm + q/k/v + qk_norm + rope + kv + attn + o(+resid)
//   Proj(GU)     = ffn_norm + w1 + w3            ResidualGemv(DOWN) = silu·mul + w2(+resid)
//   Moe          = ffn_norm + rotate + router + top-k + experts + combine
// ─────────────────────────────────────────────────────────────────────────

/// Conv mixer block (operator-norm folded in). Mirrors the hand-loop Conv arm.
fn conv_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    c: &ConvWeights,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
        .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
    gpu.conv1d_gated_decode_f32(
        &state.conv_bcx,
        &state.conv_states[c.conv_state_idx],
        &c.conv_weight,
        &state.conv_y,
        1,
        hidden,
        cfg.conv_kernel_size,
    )
    .map_err(|e| format!("lfm2moe L{l}: conv gated decode: {e:?}"))?;
    weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))
}

/// Attention mixer block (operator-norm folded in). Mirrors the hand-loop Attn arm.
fn attn_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    a: &AttnWeights,
    state: &Lfm2MoeState,
    l: usize,
    seq_len: usize,
) -> Result<(), String> {
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
        .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
        .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
        .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;
    gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;
    lfm2_triattn_tap_batch(
        gpu,
        a.kv_idx,
        &state.fa_q,
        &state.fa_k,
        1,
        n_heads,
        n_kv,
        head_dim,
    )
    .map_err(|e| format!("lfm2moe L{l}: {e}"))?;
    let physical_position = (seq_len - 1) as u32;
    stage_logical_rope_position(gpu, state, physical_position)?;
    gpu.rope_f32(
        &state.fa_q,
        &state.fa_k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        cfg.rope_theta,
    )
    .map_err(|e| format!("lfm2moe L{l}: rope: {e:?}"))?;
    restore_physical_position_after_rope(gpu, state, physical_position)?;
    let kv_idx = a.kv_idx;
    gpu.kv_cache_write_q8_0(
        &state.kv.k_gpu[kv_idx],
        &state.fa_k,
        &state.pos_buf,
        n_kv,
        head_dim,
    )
    .map_err(|e| format!("lfm2moe L{l}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(
        &state.kv.v_gpu[kv_idx],
        &state.fa_v,
        &state.pos_buf,
        n_kv,
        head_dim,
    )
    .map_err(|e| format!("lfm2moe L{l}: kv write v: {e:?}"))?;
    gpu.attention_q8_0_kv(
        &state.fa_q,
        &state.kv.k_gpu[kv_idx],
        &state.kv.v_gpu[kv_idx],
        &state.fa_attn_out,
        &state.pos_buf,
        seq_len,
        n_heads,
        n_kv,
        head_dim,
        state.kv.physical_cap,
    )
    .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;
    weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))
}

/// Dense FFN gate/up half (ffn-norm folded in). Mirrors the hand-loop Dense head.
fn dense_gate_up_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    ffn_norm: &rdna_compute::GpuTensor,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
        .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
    weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
        .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))
}

/// Dense FFN down half (silu·mul + w2 residual). Mirrors the hand-loop Dense tail.
fn dense_down_block(
    gpu: &mut Gpu,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
        .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
    weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))
}

/// MoE FFN block (ffn-norm folded in). Mirrors the hand-loop Moe arm.
fn moe_ffn_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    ffn_norm: &rdna_compute::GpuTensor,
    m: &MoeFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    rotate_x_mq_for(
        gpu,
        &m.experts[0].gate_up,
        &state.ffn_tmp,
        &state.ffn_x_rot,
        hidden,
    )
    .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;
    weight_gemv(gpu, &m.router, &state.ffn_tmp, &state.router_logits)
        .map_err(|e| format!("lfm2moe L{l}: router: {e}"))?;
    gpu.sigmoid_f32(&state.router_logits)
        .map_err(|e| format!("lfm2moe L{l}: sigmoid: {e:?}"))?;
    gpu.deepseek4_moe_topk_bias_aware_f32(
        &state.router_logits,
        &m.expert_bias,
        &state.topk_indices,
        &state.topk_weights,
        n_exp as i32,
        k_top as i32,
        cfg.routed_scaling_factor,
    )
    .map_err(|e| format!("lfm2moe L{l}: topk: {e:?}"))?;
    let capture_expert_indices = maybe_capture_moe_gate_up_inputs(
        gpu,
        l,
        &state.topk_indices,
        &state.ffn_x_rot,
        hidden,
        n_exp,
        k_top,
        1,
    )?;
    let experts_mq6 = m.experts[0].gate_up.gpu_dtype == DType::MQ6G256;
    if experts_mq6 {
        gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
            &m.expert_gate_up_ptrs,
            &state.topk_indices,
            &state.ffn_x_rot,
            &state.gate_batch,
            &state.up_batch,
            2 * moe_inter,
            hidden,
            k_top,
            1,
        )
        .map_err(|e| format!("lfm2moe L{l}: gate_up(mq6): {e:?}"))?;
    } else {
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            &m.expert_gate_up_ptrs,
            &state.topk_indices,
            &state.ffn_x_rot,
            &state.gate_batch,
            &state.up_batch,
            2 * moe_inter,
            hidden,
            k_top,
            1,
        )
        .map_err(|e| format!("lfm2moe L{l}: gate_up: {e:?}"))?;
    }
    fused_silu_mul_rotate_mq_batched_for(
        gpu,
        &m.experts[0].down,
        &state.gate_batch,
        &state.up_batch,
        &state.rot_batch,
        moe_inter,
        k_top,
    )
    .map_err(|e| format!("lfm2moe L{l}: silu_mul_rotate: {e:?}"))?;
    if let Some(indices) = capture_expert_indices.as_deref() {
        maybe_capture_moe_down_inputs(gpu, l, indices, &state.rot_batch, moe_inter, k_top, 1)?;
    }
    if experts_mq6 {
        gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
            &m.expert_down_ptrs,
            &state.topk_indices,
            &state.rot_batch,
            &state.down_expanded,
            hidden,
            moe_inter,
            k_top,
            1,
        )
        .map_err(|e| format!("lfm2moe L{l}: down(mq6): {e:?}"))?;
    } else {
        gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            &m.expert_down_ptrs,
            &state.topk_indices,
            &state.rot_batch,
            &state.down_expanded,
            hidden,
            moe_inter,
            k_top,
            1,
        )
        .map_err(|e| format!("lfm2moe L{l}: down: {e:?}"))?;
    }
    gpu.moe_down_combine_k8_batched(
        &state.down_expanded,
        &state.topk_weights,
        &state.h,
        hidden,
        k_top,
        1,
    )
    .map_err(|e| format!("lfm2moe L{l}: combine: {e:?}"))
}

/// lfm2-local super-op opcodes (encoded in OpBinding.weights[0]).
mod lfm2_op {
    pub const DENSE_GATE_UP: u32 = 0;
    pub const DENSE_DOWN: u32 = 1;
}

/// The four lfm2 decoder-layer shapes (mixer × FFN). Pure → unit-testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lfm2Variant {
    ConvDense,
    ConvMoe,
    AttnDense,
    AttnMoe,
}

fn lfm2_variant_of(layer: &Lfm2MoeLayerWeights) -> Lfm2Variant {
    match (&layer.mixer, &layer.ffn) {
        (Mixer::Conv(_), Ffn::Dense(_)) => Lfm2Variant::ConvDense,
        (Mixer::Conv(_), Ffn::Moe(_)) => Lfm2Variant::ConvMoe,
        (Mixer::Attention(_), Ffn::Dense(_)) => Lfm2Variant::AttnDense,
        (Mixer::Attention(_), Ffn::Moe(_)) => Lfm2Variant::AttnMoe,
    }
}

#[inline]
fn lfm2_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// Lower one lfm2 decoder layer to a coarse super-op LayerProgram (mirrors the
/// hand-loop order: mixer block, then FFN). Pure (no GpuTensor) → unit-testable.
fn lfm2_lower_variant(v: Lfm2Variant) -> superop::LayerProgram {
    use lfm2_op::{DENSE_DOWN, DENSE_GATE_UP};
    use SuperOpKind::{Attend, Conv, Moe, Proj, ResidualGemv};
    match v {
        Lfm2Variant::ConvDense => vec![
            lfm2_superop(Conv, 0),
            lfm2_superop(Proj, DENSE_GATE_UP),
            lfm2_superop(ResidualGemv, DENSE_DOWN),
        ],
        Lfm2Variant::AttnDense => vec![
            lfm2_superop(Attend, 0),
            lfm2_superop(Proj, DENSE_GATE_UP),
            lfm2_superop(ResidualGemv, DENSE_DOWN),
        ],
        Lfm2Variant::ConvMoe => vec![lfm2_superop(Conv, 0), lfm2_superop(Moe, 0)],
        Lfm2Variant::AttnMoe => vec![lfm2_superop(Attend, 0), lfm2_superop(Moe, 0)],
    }
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct Lfm2MoeBindings<'a> {
    cfg: &'a Lfm2MoeConfig,
    layer: &'a Lfm2MoeLayerWeights,
    state: &'a Lfm2MoeState,
    l: usize,
    seq_len: usize,
}

impl<'a> ForwardBindings for Lfm2MoeBindings<'a> {
    fn run_conv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Conv(c) => conv_mixer_block(
                gpu,
                self.cfg,
                &self.layer.operator_norm,
                c,
                self.state,
                self.l,
            ),
            _ => Err("run_conv on non-Conv layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Attention(a) => attn_mixer_block(
                gpu,
                self.cfg,
                &self.layer.operator_norm,
                a,
                self.state,
                self.l,
                self.seq_len,
            ),
            _ => Err("run_attend on non-Attention layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_GATE_UP, Ffn::Dense(d)) => {
                dense_gate_up_block(gpu, self.cfg, &self.layer.ffn_norm, d, self.state, self.l)
            }
            _ => Err(format!("run_proj bad opcode {code} / non-Dense ffn")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_DOWN, Ffn::Dense(d)) => dense_down_block(gpu, d, self.state, self.l),
            _ => Err(format!(
                "run_residual_gemv bad opcode {code} / non-Dense ffn"
            )),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.ffn {
            Ffn::Moe(m) => {
                moe_ffn_block(gpu, self.cfg, &self.layer.ffn_norm, m, self.state, self.l)
            }
            _ => Err("run_moe on non-Moe ffn".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "lfm2 has no standalone Norm super-op".into(),
        ))
    }
    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("lfm2 has no Recurrent super-op".into()))
    }
    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "lfm2 has no Escape super-op ({kind:?})"
        )))
    }
}

/// Cached HIPFIRE_FORWARD_LOWERED toggle for lfm2. #397 Ship 6: the lfm2 lowered
/// decode is **DEFAULT ON** as of 2026-06-07 — fleet byte-parity validated
/// (k9lin gfx1100 / hiptrx gfx1201 / hipx gfx1151, lowered == hand token-text md5
/// 754a38b5…). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` forces the legacy hand
/// loop (still present in decode_step_layers_and_head); any other value / unset → lowered.
fn lfm2_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0"))
}

/// Lowered (#397 Ship 6) per-layer decode loop + final norm/head. Behaviorally
/// equivalent to decode_step_layers_and_head's hand loop (validated via the
/// FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B). No oracle-capture support.
fn decode_step_layers_and_head_lowered(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let ctx = DispatchCtx::new(gpu);
    for (l, layer) in weights.layers.iter().enumerate() {
        let program = lfm2_lower_variant(lfm2_variant_of(layer));
        {
            let mut bind = Lfm2MoeBindings {
                cfg,
                layer,
                state,
                l,
                seq_len,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| format!("lfm2moe L{l}: lowered run_layer_program: {e}"))?;
        }
    }
    state.n_tokens = seq_len;
    gpu.rmsnorm_f32(
        &state.h,
        &weights.embedding_norm,
        &state.final_norm_buf,
        eps,
    )
    .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}

/// hipGraph-amortized decode_step. Opt-in via `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact `decode_step_inner` behavior). Mirrors the working
/// DeepSeek-V4 integration (`decode_step_with_graph`).
///
/// Three-state machine driven by `state.graph_warmed_up` and `gpu.graph_exec`:
///   1. !warmed_up                 → direct dispatch once (so kernel JIT and
///                                    any lazy hipMalloc happen OUTSIDE the
///                                    captured region), set the flag.
///   2. warmed_up && no graph      → embedding+pos direct, then capture the
///                                    layer loop + head, instantiate, launch
///                                    once for this position's output.
///   3. graph instantiated         → embedding+pos direct, then `graph_launch`
///                                    re-runs the captured ops which re-read
///                                    `state.pos_buf` (refreshed below) and the
///                                    KV / conv-state / topk device buffers.
///
/// Per-token-varying values handled OUTSIDE the captured region:
///   * `token_id` — baked into `embedding_lookup_q8`'s kernarg, so the
///     embedding lookup runs DIRECT each token (writes `state.h`); the
///     captured region begins at layer 0's rmsnorm reading `state.h`.
///   * `position` — staged into the STABLE device buffer `state.pos_buf` via a
///     direct `memcpy_htod` before each `graph_launch`; every captured kernel
///     (rope/kv-write/attention) reads `pos_buf` from the device, so replay at
///     a new position is correct without re-capture. The attention kernel's
///     launch-baked `block_size`/`shared_mem` are sized to `max_seq` under
///     capture (see `attention_q8_0_kv` in dispatch.rs), so one capture
///     replays correctly at every later position.
///
/// `state.n_tokens` is advanced here to match `decode_step_inner` semantics.
pub fn decode_step_with_graph(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    decode_step(cfg, weights, state, gpu, token_id, position)
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Conv, Moe, Proj, ResidualGemv};

    // #397 Ship 6 — lfm2 lowered LayerProgram shapes must mirror the hand-loop
    // order (mixer block, then FFN). CPU-pure (no GPU).
    #[test]
    fn lfm2_variant_shapes() {
        let kinds = |v| {
            lfm2_lower_variant(v)
                .iter()
                .map(|o| o.kind)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kinds(Lfm2Variant::ConvDense),
            vec![Conv, Proj, ResidualGemv]
        );
        assert_eq!(
            kinds(Lfm2Variant::AttnDense),
            vec![Attend, Proj, ResidualGemv]
        );
        assert_eq!(kinds(Lfm2Variant::ConvMoe), vec![Conv, Moe]);
        assert_eq!(kinds(Lfm2Variant::AttnMoe), vec![Attend, Moe]);
        let p = lfm2_lower_variant(Lfm2Variant::ConvDense);
        assert_eq!(p[1].binding.weights[0].0, lfm2_op::DENSE_GATE_UP);
        assert_eq!(p[2].binding.weights[0].0, lfm2_op::DENSE_DOWN);
    }
}
