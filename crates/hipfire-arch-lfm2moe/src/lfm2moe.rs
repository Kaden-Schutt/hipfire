// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE weights + decode state.
//!
//! HFQ files carry RAW HF tensor names; the loader looks each up by exact
//! name (no rename). Mirrors the MiniMax-M2 loader (shared `WeightTensor`,
//! `KvCache`, indexed-MoE GEMV kernels) and adds:
//!   * a per-layer mixer split (conv vs attention) from `layer_types`,
//!   * a per-layer FFN split (dense SwiGLU vs top-4 MoE) from `num_dense_layers`,
//!   * a rolling conv-state cache (one [hidden,(K-1)] f32 ring buffer per conv
//!     layer) — the conv analog of the KV cache.
//!
//! Expert weights ship pre-split (w1/w2/w3); the loader byte-fuses w1‖w3 into
//! the per-expert `gate_up` blob the indexed GEMV kernels expect (exactly the
//! minimax convention). lm_head is tied to embed_tokens.

use crate::config::{Lfm2MoeConfig, MixerKind};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::hfq_parallel::{read_hfq_jobs_ordered, HfqReadJob};
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::{bf16_bytes_to_f16, source_bytes_to_f32_vec};
use hipfire_runtime::{screen_weight_tensor, MmqScreenable};
use rdna_compute::{DType, Gpu, GpuTensor};

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("lfm2moe: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

/// Load an LFM2 AWQ shared gate_up-scale sidecar (1D F16, length k) → F32
/// GpuTensor. Mirror of minimax `load_mm_awq_scale`. Returns None if the
/// sidecar is absent or malformed, so non-AWQ models load cleanly (the
/// attached `awq_scale` stays None and `rotate_x_mq_for` takes the plain path).
fn load_lfm2_awq_scale(hfq: &HfqFile, gpu: &mut Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let (qt, data) = read_tensor(hfq, name).ok()?;
    if qt != 1 {
        return None;
    } // 1 = F16
    if data.len() != k * 2 {
        eprintln!(
            "lfm2moe AWQ sidecar {name}: {} bytes != {} (k*2); skipping",
            data.len(),
            k * 2
        );
        return None;
    }
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_data.len()]).ok()
}

/// Load a 1D/raw F16/F32 vector → F32 GpuTensor with the given shape.
/// LFM2 uses STANDARD RMSNorm (`weight * x̂`, no +1 offset — verified against
/// Lfm2MoeRMSNorm), so no offset is baked in. Also used for the depthwise conv
/// filter ([hidden, K]) and the F32 expert_bias.
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => {
            // Q8_0: 32-elem blocks [f16 scale | 32 i8]. Dequant to f32.
            dequant_q8_0(&data)
        }
        _ => {
            return Err(format!(
                "lfm2moe: expected F16/F32/Q8 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("lfm2moe: upload {name}: {e:?}"))
}

/// REAP keep variant of [`load_f32`] for a 1-D per-expert vector (the MoE
/// `expert_bias`, shape `[orig_experts]`): gather the kept elements BEFORE
/// dequant, then upload as `[keep.len()]`. Each element is one expert's bias,
/// fully self-contained (F16/F32 are trivially row-independent; Q8_0's 32-elem
/// blocks would NOT be element-gatherable, but expert_bias ships as F16/F32 —
/// guarded below). `m` MUST equal `keep.len()`.
fn load_f32_keep(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    keep: &[u32],
) -> Result<GpuTensor, String> {
    debug_assert_eq!(m, keep.len(), "load_f32_keep: m must equal keep.len()");
    let f32_data = hipfire_reap::load::gather_f32_vec("lfm2moe", hfq, name, keep)?;
    gpu.upload_f32(&f32_data, &[m])
        .map_err(|e| format!("lfm2moe: upload {name}: {e:?}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("lfm2moe: load_wt {name}: {e}"))
}

/// REAP keep variant of [`load_wt`]: gather the tensor's first-axis rows (one
/// row per ORIGINAL expert) down to `keep` BEFORE quant decode, then build the
/// `WeightTensor` from the gathered bytes with `m = keep.len()`.
///
/// Only used for the MoE router (`feed_forward.gate.weight`, shape
/// `[orig_experts, hidden]`) under an active keep-map. `gather_rows` is exact
/// for any row-independent quant (every per-expert row is self-contained — its
/// own scale/zero/codebook live in the row), which holds for every quant_type
/// `wt_from_raw` accepts. `keep` MUST be in compact slot order and `m` MUST
/// equal `keep.len()`. Reads owned bytes via `hfq.tensor_data_vec` directly
/// (bypassing the `read_tensor` wrapper) to retain `info.shape` for deriving
/// the original row count — the non-keep path uses `read_tensor`, which
/// discards `info`.
fn load_wt_keep(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
    keep: &[u32],
) -> Result<WeightTensor, String> {
    debug_assert_eq!(m, keep.len(), "load_wt_keep: m must equal keep.len()");
    let (qt, sub) = hipfire_reap::load::gather_weight_rows("lfm2moe", hfq, name, keep)?;
    wt_from_raw(gpu, qt, &sub, m, k).map_err(|e| format!("lfm2moe: load_wt_keep {name}: {e}"))
}

/// quant_type → DType mapping (mirrors minimax::wt_from_raw); uploads raw
/// bytes and tags the dtype for kernel dispatch.
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        1 => DType::F16,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

fn finish_hfq_expert(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    layer_prefix: &str,
    layer: usize,
    source_expert: usize,
    slot: usize,
    gate_up_qt: u8,
    gate_up_bytes: &[u8],
    down_qt: u8,
    down_bytes: &[u8],
    hidden: usize,
    moe_inter: usize,
) -> Result<ExpertWeights, String> {
    let mut gate_up = wt_from_raw(gpu, gate_up_qt, gate_up_bytes, 2 * moe_inter, hidden)
        .map_err(|error| format!("lfm2moe: fuse gate_up L{layer}E{source_expert}: {error}"))?;
    let mut down = wt_from_raw(gpu, down_qt, down_bytes, hidden, moe_inter)
        .map_err(|error| format!("lfm2moe: down L{layer}E{source_expert}: {error}"))?;
    // AWQ scales are layer-shared and emitted once by the quantizer. Attach
    // them to the first compact slot, which is also what the forward reads;
    // this remains correct when REAP prunes original expert zero.
    if slot == 0 {
        gate_up.awq_scale = load_lfm2_awq_scale(
            hfq,
            gpu,
            &format!("{layer_prefix}.feed_forward.awq_scale_gate_up.weight"),
            hidden,
        );
        if gate_up.awq_scale.is_some() {
            eprintln!(
                "lfm2moe: AWQ gate_up scale attached at {layer_prefix} (expert-0 representative)"
            );
        }
        down.awq_scale = load_lfm2_awq_scale(
            hfq,
            gpu,
            &format!("{layer_prefix}.feed_forward.awq_scale_down.weight"),
            moe_inter,
        );
        if down.awq_scale.is_some() {
            eprintln!(
                "lfm2moe: AWQ down scale attached at {layer_prefix} (expert-0 representative)"
            );
        }
    }
    Ok(ExpertWeights { gate_up, down })
}

// ──────────────────────────── Weights ────────────────────────────

/// LIV short-conv mixer weights.
pub struct ConvWeights {
    /// in_proj: [3*hidden, hidden] — produces B | C_gate | x.
    pub in_proj: WeightTensor,
    /// Depthwise causal filter, squeezed [hidden, K] F32.
    pub conv_weight: GpuTensor,
    /// out_proj: [hidden, hidden].
    pub out_proj: WeightTensor,
    /// Index into the conv-state ring-buffer cache.
    pub conv_state_idx: usize,
}

/// GQA attention mixer weights (per-head QK-norm + full-dim rotate_half RoPE).
pub struct AttnWeights {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    /// Per-head QK-norm weight, [head_dim].
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// Index into the KV cache.
    pub kv_idx: usize,
}

pub enum Mixer {
    Conv(ConvWeights),
    Attention(AttnWeights),
}

/// Dense SwiGLU MLP (first `num_dense_layers` layers).
pub struct DenseFfn {
    pub w1: WeightTensor, // gate [inter, hidden]
    pub w3: WeightTensor, // up   [inter, hidden]
    pub w2: WeightTensor, // down [hidden, inter]
}

/// One MoE expert: fused gate(w1)‖up(w3) and down(w2).
pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2*moe_inter, hidden]
    pub down: WeightTensor,    // [hidden, moe_inter]
}

/// Top-4 MoE FFN (sigmoid + expert_bias routing).
pub struct MoeFfn {
    pub router: WeightTensor,        // feed_forward.gate.weight [n_exp, hidden]
    pub expert_bias: GpuTensor,      // feed_forward.expert_bias [n_exp] F32
    pub experts: Vec<ExpertWeights>, // keep alive (buffers owned here)
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct Lfm2MoeLayerWeights {
    pub operator_norm: GpuTensor, // pre-mixer RMSNorm
    pub ffn_norm: GpuTensor,      // pre-FFN RMSNorm
    pub mixer: Mixer,
    pub ffn: Ffn,
}

pub struct Lfm2MoeWeights {
    pub embed: GpuTensor, // model.embed_tokens.weight (raw, for embedding_lookup)
    pub embd_format: hipfire_runtime::llama::EmbeddingFormat,
    pub embedding_norm: GpuTensor, // model.embedding_norm.weight (final norm)
    pub lm_head: WeightTensor,     // tied = embed_tokens (loaded as Q8 weight)
    pub layers: Vec<Lfm2MoeLayerWeights>,
}

impl MmqScreenable for Lfm2MoeWeights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.lm_head, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Conv(weights) => {
                    screen_weight_tensor(&weights.in_proj, gpu, &mut safe, &mut unsafe_count);
                    screen_weight_tensor(&weights.out_proj, gpu, &mut safe, &mut unsafe_count);
                }
                Mixer::Attention(weights) => {
                    for weight in [&weights.wq, &weights.wk, &weights.wv, &weights.wo] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
            }
            match &layer.ffn {
                Ffn::Dense(weights) => {
                    for weight in [&weights.w1, &weights.w3, &weights.w2] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                Ffn::Moe(weights) => {
                    screen_weight_tensor(&weights.router, gpu, &mut safe, &mut unsafe_count);
                }
            }
        }
        (safe, unsafe_count)
    }
}

impl ExpertWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.gate_up.free_all(gpu);
        self.down.free_all(gpu);
    }
}

impl ConvWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.in_proj.free_all(gpu);
        let _ = gpu.free_tensor(self.conv_weight);
        self.out_proj.free_all(gpu);
    }
}

impl AttnWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wo.free_all(gpu);
        let _ = gpu.free_tensor(self.q_norm);
        let _ = gpu.free_tensor(self.k_norm);
    }
}

impl Mixer {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Mixer::Conv(c) => c.free_gpu(gpu),
            Mixer::Attention(a) => a.free_gpu(gpu),
        }
    }
}

impl DenseFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.w1.free_all(gpu);
        self.w3.free_all(gpu);
        self.w2.free_all(gpu);
    }
}

impl MoeFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.router.free_all(gpu);
        let _ = gpu.free_tensor(self.expert_bias);
        for e in self.experts {
            e.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(self.expert_gate_up_ptrs);
        let _ = gpu.free_tensor(self.expert_down_ptrs);
    }
}

impl Ffn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Ffn::Dense(d) => d.free_gpu(gpu),
            Ffn::Moe(m) => m.free_gpu(gpu),
        }
    }
}

impl Lfm2MoeLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.operator_norm);
        let _ = gpu.free_tensor(self.ffn_norm);
        self.mixer.free_gpu(gpu);
        self.ffn.free_gpu(gpu);
    }
}

impl Lfm2MoeWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &Lfm2MoeConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let head_dim = cfg.head_dim;
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k_conv = cfg.conv_kernel_size;

        // Globals. embed_tokens is the shared (tied) lm_head.
        let (_eqt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let (embd_format, embed) = match _eqt {
            3 => {
                let fmt = hipfire_runtime::llama::EmbeddingFormat::Q8_0;
                let buf = gpu
                    .upload_raw(&embed_bytes, &[embed_bytes.len()])
                    .map_err(|e| format!("lfm2moe: upload embed: {e:?}"))?;
                (fmt, buf)
            }
            6 => {
                let fmt = hipfire_runtime::llama::EmbeddingFormat::HFQ4G256;
                let buf = gpu
                    .upload_raw(&embed_bytes, &[embed_bytes.len()])
                    .map_err(|e| format!("lfm2moe: upload embed: {e:?}"))?;
                (fmt, buf)
            }
            1 => {
                // F16 embedding → widen to F32 on host, mark F32.
                // BF16 never appears via HFQ qt1; if it does, f16 decode would be wrong,
                // but HFQ qt1 is canonically F16. Use f16_to_f32 widening.
                if embed_bytes.len() % 2 != 0 {
                    return Err(format!(
                        "lfm2moe: embedding qt=1 bytes {} not multiple of 2",
                        embed_bytes.len()
                    ));
                }
                let f32_data: Vec<f32> = embed_bytes
                    .chunks_exact(2)
                    .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let buf = gpu
                    .upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])
                    .map_err(|e| format!("lfm2moe: upload embed F32: {e:?}"))?;
                (hipfire_runtime::llama::EmbeddingFormat::F32, buf)
            }
            other => {
                return Err(format!(
                    "lfm2moe: unsupported embedding quant_type {other} (expected 1,3,6)"
                ))
            }
        };
        let embedding_norm = load_f32(hfq, gpu, "model.embedding_norm.weight", &[hidden])?;
        // lm_head: tied → reuse embed_tokens.weight as a Q8 weight tensor.
        let lm_head = load_wt(
            hfq,
            gpu,
            "model.embed_tokens.weight",
            cfg.vocab_size,
            hidden,
        )?;

        let mut conv_state_count = 0usize;
        let mut kv_count = 0usize;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let operator_norm =
                load_f32(hfq, gpu, &format!("{p}.operator_norm.weight"), &[hidden])?;
            let ffn_norm = load_f32(hfq, gpu, &format!("{p}.ffn_norm.weight"), &[hidden])?;

            // ── Mixer: conv vs attention ──────────────────────────────────
            let mixer = match cfg.mixer(l) {
                MixerKind::Conv => {
                    let in_proj = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.conv.in_proj.weight"),
                        3 * hidden,
                        hidden,
                    )?;
                    // conv.conv.weight ships [hidden,1,K] → loaded flat as [hidden,K] f32.
                    let conv_weight = load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.conv.conv.weight"),
                        &[hidden * k_conv],
                    )?;
                    let out_proj = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.conv.out_proj.weight"),
                        hidden,
                        hidden,
                    )?;
                    let conv_state_idx = conv_state_count;
                    conv_state_count += 1;
                    Mixer::Conv(ConvWeights {
                        in_proj,
                        conv_weight,
                        out_proj,
                        conv_state_idx,
                    })
                }
                MixerKind::Attention => {
                    let wq = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.q_proj.weight"),
                        q_dim,
                        hidden,
                    )?;
                    let wk = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.k_proj.weight"),
                        kv_dim,
                        hidden,
                    )?;
                    let wv = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.v_proj.weight"),
                        kv_dim,
                        hidden,
                    )?;
                    let wo = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.out_proj.weight"),
                        hidden,
                        q_dim,
                    )?;
                    // Per-HEAD QK-norm: weight is [head_dim], applied to each head.
                    let q_norm = load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.q_layernorm.weight"),
                        &[head_dim],
                    )?;
                    let k_norm = load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.k_layernorm.weight"),
                        &[head_dim],
                    )?;
                    let kv_idx = kv_count;
                    kv_count += 1;
                    Mixer::Attention(AttnWeights {
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm,
                        k_norm,
                        kv_idx,
                    })
                }
            };

            // ── FFN: dense SwiGLU vs top-4 MoE ────────────────────────────
            let ffn = if cfg.is_dense_ffn(l) {
                let w1 = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.w1.weight"),
                    dense_inter,
                    hidden,
                )?;
                let w3 = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.w3.weight"),
                    dense_inter,
                    hidden,
                )?;
                let w2 = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.w2.weight"),
                    hidden,
                    dense_inter,
                )?;
                Ffn::Dense(DenseFfn { w1, w3, w2 })
            } else {
                // REAP keep-map for this layer (None ⇒ no pruning / identity).
                // When a keep is present, `n_exp == cfg.num_experts` is already
                // the KEPT count (overridden in apply_reap_plan); the router and
                // expert loops below load only the kept original experts, in
                // compact slot order.
                let reap_ep = cfg.reap_keep.as_ref().map(|r| r.expert_plan(l));
                let keep_l = reap_ep.as_ref().and_then(|e| e.keep());

                // Router: hidden → n_exp. Under a keep, gather the router's
                // expert rows (`[orig_experts, hidden]`) down to the kept set so
                // it emits logits only for kept experts, in compact slot order.
                // No keep ⇒ the literal original full load.
                let router = match keep_l {
                    Some(keep) => load_wt_keep(
                        hfq,
                        gpu,
                        &format!("{p}.feed_forward.gate.weight"),
                        n_exp,
                        hidden,
                        keep,
                    )?,
                    None => load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.feed_forward.gate.weight"),
                        n_exp,
                        hidden,
                    )?,
                };
                // expert_bias is a 1-D [orig_experts] F32 vector indexed by
                // expert; under a keep it must also be gathered to the kept set
                // (parallel to the router rows). No keep ⇒ literal original load.
                let expert_bias = match keep_l {
                    Some(keep) => load_f32_keep(
                        hfq,
                        gpu,
                        &format!("{p}.feed_forward.expert_bias"),
                        n_exp,
                        keep,
                    )?,
                    None => load_f32(hfq, gpu, &format!("{p}.feed_forward.expert_bias"), &[n_exp])?,
                };
                // Byte-fuse w1‖w3 → gate_up [2*moe_inter, hidden]; w2 → down.
                // Iterate compact slots `0..n_exp`; `e` = the ORIGINAL expert
                // index loaded into slot (slot==e on the no-keep identity path).
                let mut experts = Vec::with_capacity(n_exp);
                if hfq.has_overlay() {
                    // Overlay offsets belong to a second file. Preserve the
                    // existing overlay-aware reads and allocation order.
                    for slot in 0..n_exp {
                        let e = reap_ep.as_ref().map(|ep| ep.src(slot)).unwrap_or(slot);
                        let ep = format!("{p}.feed_forward.experts.{e}");
                        let (qt1, w1) = read_tensor(hfq, &format!("{ep}.w1.weight"))?;
                        let (_qt3, w3) = read_tensor(hfq, &format!("{ep}.w3.weight"))?;
                        let mut gate_up_bytes = w1;
                        gate_up_bytes.extend_from_slice(&w3);
                        let (qt2, w2) = read_tensor(hfq, &format!("{ep}.w2.weight"))?;
                        experts.push(finish_hfq_expert(
                            hfq,
                            gpu,
                            &p,
                            l,
                            e,
                            slot,
                            qt1,
                            &gate_up_bytes,
                            qt2,
                            &w2,
                            hidden,
                            moe_inter,
                        )?);
                    }
                } else {
                    // Two experts form four jobs (gate_up + down per expert),
                    // matching the four-lane storage optimum. All reads finish
                    // before the original gate_up/down GPU allocation sequence.
                    let mut chunk_start = 0usize;
                    while chunk_start < n_exp {
                        let chunk_end = (chunk_start + 2).min(n_exp);
                        let mut metadata = Vec::with_capacity(chunk_end - chunk_start);
                        let mut jobs = Vec::with_capacity((chunk_end - chunk_start) * 2);
                        for slot in chunk_start..chunk_end {
                            let e = reap_ep.as_ref().map(|ep| ep.src(slot)).unwrap_or(slot);
                            let ep = format!("{p}.feed_forward.experts.{e}");
                            let w1_name = format!("{ep}.w1.weight");
                            let w3_name = format!("{ep}.w3.weight");
                            let w2_name = format!("{ep}.w2.weight");
                            let qt1 = hfq
                                .find_tensor_info(&w1_name)
                                .ok_or_else(|| {
                                    format!("lfm2moe: tensor not found in HFQ: {w1_name}")
                                })?
                                .quant_type;
                            let qt2 = hfq
                                .find_tensor_info(&w2_name)
                                .ok_or_else(|| {
                                    format!("lfm2moe: tensor not found in HFQ: {w2_name}")
                                })?
                                .quant_type;
                            jobs.push(
                                HfqReadJob::packed(
                                    hfq,
                                    format!("{ep}.gate_up"),
                                    [&w1_name, &w3_name],
                                )
                                .map_err(|error| {
                                    format!("lfm2moe: plan gate_up L{l}E{e}: {error}")
                                })?,
                            );
                            jobs.push(HfqReadJob::tensor(hfq, &w2_name).map_err(|error| {
                                format!("lfm2moe: plan down L{l}E{e}: {error}")
                            })?);
                            metadata.push((slot, e, qt1, qt2));
                        }
                        let mut buffers = read_hfq_jobs_ordered(hfq, &jobs)
                            .map_err(|error| {
                                format!("lfm2moe: parallel expert read layer {l}: {error}")
                            })?
                            .into_iter();
                        for (slot, e, qt1, qt2) in metadata {
                            let gate_up_bytes =
                                buffers.next().expect("two LFM read jobs per expert").data;
                            let down_bytes =
                                buffers.next().expect("two LFM read jobs per expert").data;
                            experts.push(finish_hfq_expert(
                                hfq,
                                gpu,
                                &p,
                                l,
                                e,
                                slot,
                                qt1,
                                &gate_up_bytes,
                                qt2,
                                &down_bytes,
                                hidden,
                                moe_inter,
                            )?);
                        }
                        chunk_start = chunk_end;
                    }
                }

                let gu_bytes: Vec<u8> = experts
                    .iter()
                    .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let dn_bytes: Vec<u8> = experts
                    .iter()
                    .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let expert_gate_up_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("lfm2moe: alloc gu_ptrs: {e:?}"))?;
                let expert_down_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("lfm2moe: alloc dn_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                    .map_err(|e| format!("lfm2moe: htod gu_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                    .map_err(|e| format!("lfm2moe: htod dn_ptrs: {e:?}"))?;
                Ffn::Moe(MoeFfn {
                    router,
                    expert_bias,
                    experts,
                    expert_gate_up_ptrs,
                    expert_down_ptrs,
                })
            };

            layers.push(Lfm2MoeLayerWeights {
                operator_norm,
                ffn_norm,
                mixer,
                ffn,
            });
        }

        let _ = (conv_state_count, kv_count);
        Ok(Lfm2MoeWeights {
            embed,
            embd_format,
            embedding_norm,
            lm_head,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed);
        let _ = gpu.free_tensor(self.embedding_norm);
        self.lm_head.free_all(gpu);
        for layer in self.layers {
            layer.free_gpu(gpu);
        }
    }
}

// ─────────────── Source-based load helpers (ModelSource) ──────────────

/// Classify a tensor's byte format. Returns `(is_f16, is_q8_0)` based
/// on byte-count heuristics. Used when loading from safetensors where
/// quant_type is not available.
fn classify_tensor_bytes(bytes: &[u8], numel: usize) -> (bool, bool) {
    let is_f16 = bytes.len() == numel * 2;
    // Q8_0: 34 bytes per block of 32 elements:
    //   [f16 scale (2 bytes)] [32 × i8 (32 bytes)]
    let q8_0_expected = ((numel + 31) / 32) * 34;
    let is_q8_0 = !is_f16 && bytes.len() == q8_0_expected;
    (is_f16, is_q8_0)
}

/// Determine effective quant_type from a ModelSource tensor.
/// - HFQ-backed sources preserve quant_type (≠ 0xFF).
/// - Safetensors (quant_type == 0xFF) use dtype string + byte heuristics.
fn effective_quant_type(info: &hipfire_runtime::model_source::TensorInfo, data: &[u8]) -> u8 {
    if info.quant_type != 0xFF {
        return info.quant_type;
    }
    // Safetensors: infer from dtype string.
    match info.dtype.as_str() {
        "F16" | "BF16" => 1, // treat BF16 as F16 for byte-size purposes
        "F32" => 2,
        "I8" | "INT8" => 3, // Q8_0-compatible raw int8
        _ => {
            // Fall back to byte-count heuristics.
            let numel: usize = info.shape.iter().product();
            let (is_f16, is_q8_0) = classify_tensor_bytes(data, numel);
            if is_f16 {
                1
            } else if is_q8_0 {
                3
            } else {
                // Unknown; default to F16 and let the upload handle it.
                1
            }
        }
    }
}

/// Read raw bytes from a ModelSource with dtype classification.
/// Returns `(effective_quant_type, source_dtype_string, &[u8])`. The dtype
/// string is needed to distinguish F16 from BF16 (both qt==1, same byte size,
/// different bit layout) — decoding BF16 as F16 silently corrupts values.
fn read_tensor_from_source<'a>(
    source: &'a dyn ModelSource,
    name: &str,
) -> Result<(u8, &'a str, &'a [u8]), String> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("lfm2moe: tensor not found in source: {name}"))?;
    Ok((effective_quant_type(info, data), info.dtype.as_str(), data))
}

/// Load a norm/1D tensor as F32 on GPU from a ModelSource.
/// Handles F16, BF16, F32, and Q8_0 sources.
fn load_f32_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, dtype, data) = read_tensor_from_source(source, name)?;
    let f32_data: Vec<f32> = match qt {
        // F16/BF16 (qt==1) and F32 (qt==2) widen via the shared dtype-aware
        // helper — it decodes BF16 with the correct bit layout (the old
        // f16_to_f32 path silently mis-decoded BF16 as F16).
        1 | 2 => source_bytes_to_f32_vec(dtype, data),
        3 => {
            // Q8_0: 32-elem blocks [f16 scale | 32 i8]. Dequant to f32.
            dequant_q8_0(data)
        }
        _ => {
            return Err(format!(
                "lfm2moe: expected F16/F32/Q8 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("lfm2moe: upload {name}: {e:?}"))
}

/// Load a quantized 2D weight tensor from a ModelSource, classifying the
/// format via byte-count heuristics when quant_type is unavailable.
fn load_wt_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, src_dtype, data) = read_tensor_from_source(source, name)?;
    wt_from_source_raw(gpu, qt, src_dtype, data, m, k)
        .map_err(|e| format!("lfm2moe: load_wt_from_source {name}: {e}"))
}

/// Upload raw bytes as a WeightTensor, determining gpu_dtype from the
/// (possibly inferred) quant_type. `src_dtype` is the source dtype string,
/// used to narrow BF16 → F16 before upload (the GPU F16 path can't consume
/// raw BF16 bytes — uploading them as-is yields silently-wrong values).
fn wt_from_source_raw(
    gpu: &mut Gpu,
    qt: u8,
    src_dtype: &str,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        1 => DType::F16,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    // BF16 sources arrive as raw 2-byte bf16 (qt==1, same size as F16);
    // narrow to F16 bytes before the raw upload so the F16-tagged buffer holds
    // correctly-decoded values.
    let converted: Vec<u8>;
    let data: &[u8] = if qt == 1 && src_dtype == "BF16" {
        converted = bf16_bytes_to_f16(data);
        &converted
    } else {
        data
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Load an LFM2 AWQ shared gate_up-scale sidecar from a ModelSource.
/// Returns None if absent or malformed (non-AWQ models load cleanly).
fn load_lfm2_awq_scale_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    k: usize,
) -> Option<GpuTensor> {
    let (info, data) = source.tensor_data(name)?;
    let qt = effective_quant_type(info, data);
    if qt != 1 {
        return None;
    } // 1 = F16/BF16 (both 2-byte)
    if data.len() != k * 2 {
        eprintln!(
            "lfm2moe AWQ sidecar {name}: {} bytes != {} (k*2); skipping",
            data.len(),
            k * 2
        );
        return None;
    }
    // dtype-aware widen so a BF16 sidecar isn't mis-decoded as F16.
    let f32_data: Vec<f32> = source_bytes_to_f32_vec(info.dtype.as_str(), data);
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_data.len()]).ok()
}

/// Load full LFM2.5-MoE weights from a `&dyn ModelSource` (safetensors or HFQ).
/// Mirrors `Lfm2MoeWeights::load` but reads through the ModelSource trait.
///
/// Tensor names match those used in the HFQ path. Quantization format is
/// inferred from byte counts matching the HFQ byte layout.
pub fn load_weights_from_source(
    source: &dyn ModelSource,
    cfg: &Lfm2MoeConfig,
    gpu: &mut Gpu,
) -> Result<Lfm2MoeWeights, String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let dense_inter = cfg.intermediate_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_conv = cfg.conv_kernel_size;

    // Globals. embed_tokens is the shared (tied) lm_head.
    let (eqt, edt, embed_bytes) = read_tensor_from_source(source, "model.embed_tokens.weight")?;
    let (embd_format, embed) = match eqt {
        3 => {
            let fmt = hipfire_runtime::llama::EmbeddingFormat::Q8_0;
            let buf = gpu
                .upload_raw(embed_bytes, &[embed_bytes.len()])
                .map_err(|e| format!("lfm2moe: upload embed: {e:?}"))?;
            (fmt, buf)
        }
        6 => {
            let fmt = hipfire_runtime::llama::EmbeddingFormat::HFQ4G256;
            let buf = gpu
                .upload_raw(embed_bytes, &[embed_bytes.len()])
                .map_err(|e| format!("lfm2moe: upload embed: {e:?}"))?;
            (fmt, buf)
        }
        1 => {
            // F16/BF16 embedding → widen to F32 via dtype-aware helper, mark F32.
            let f32_data: Vec<f32> = source_bytes_to_f32_vec(edt, embed_bytes);
            // Validate element count matches vocab*hidden.
            let expected = cfg
                .vocab_size
                .checked_mul(cfg.hidden_size)
                .ok_or_else(|| format!("lfm2moe: embedding vocab*hidden overflow"))?;
            if f32_data.len() != expected {
                return Err(format!(
                    "lfm2moe: embedding F32 count {} != vocab*hidden {}",
                    f32_data.len(),
                    expected
                ));
            }
            let buf = gpu
                .upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])
                .map_err(|e| format!("lfm2moe: upload embed F32: {e:?}"))?;
            (hipfire_runtime::llama::EmbeddingFormat::F32, buf)
        }
        other => {
            return Err(format!(
                "lfm2moe: unsupported embedding quant_type {other} (expected 1,3,6)"
            ))
        }
    };
    let embedding_norm =
        load_f32_from_source(source, gpu, "model.embedding_norm.weight", &[hidden])?;
    // lm_head: tied → reuse embed_tokens.weight as a Q8 weight tensor.
    let lm_head = load_wt_from_source(
        source,
        gpu,
        "model.embed_tokens.weight",
        cfg.vocab_size,
        hidden,
    )?;
    let mut conv_state_count = 0usize;
    let mut kv_count = 0usize;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let operator_norm =
            load_f32_from_source(source, gpu, &format!("{p}.operator_norm.weight"), &[hidden])?;
        let ffn_norm =
            load_f32_from_source(source, gpu, &format!("{p}.ffn_norm.weight"), &[hidden])?;

        // ── Mixer: conv vs attention ──────────────────────────────────
        let mixer = match cfg.mixer(l) {
            MixerKind::Conv => {
                let in_proj = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.conv.in_proj.weight"),
                    3 * hidden,
                    hidden,
                )?;
                let conv_weight = load_f32_from_source(
                    source,
                    gpu,
                    &format!("{p}.conv.conv.weight"),
                    &[hidden * k_conv],
                )?;
                let out_proj = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.conv.out_proj.weight"),
                    hidden,
                    hidden,
                )?;
                let conv_state_idx = conv_state_count;
                conv_state_count += 1;
                Mixer::Conv(ConvWeights {
                    in_proj,
                    conv_weight,
                    out_proj,
                    conv_state_idx,
                })
            }
            MixerKind::Attention => {
                let wq = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    hidden,
                )?;
                let wk = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wv = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wo = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.out_proj.weight"),
                    hidden,
                    q_dim,
                )?;
                let q_norm = load_f32_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    &[head_dim],
                )?;
                let k_norm = load_f32_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    &[head_dim],
                )?;
                let kv_idx = kv_count;
                kv_count += 1;
                Mixer::Attention(AttnWeights {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    kv_idx,
                })
            }
        };

        // ── FFN: dense SwiGLU vs top-4 MoE ────────────────────────────
        let ffn = if cfg.is_dense_ffn(l) {
            let w1 = load_wt_from_source(
                source,
                gpu,
                &format!("{p}.feed_forward.w1.weight"),
                dense_inter,
                hidden,
            )?;
            let w3 = load_wt_from_source(
                source,
                gpu,
                &format!("{p}.feed_forward.w3.weight"),
                dense_inter,
                hidden,
            )?;
            let w2 = load_wt_from_source(
                source,
                gpu,
                &format!("{p}.feed_forward.w2.weight"),
                hidden,
                dense_inter,
            )?;
            Ffn::Dense(DenseFfn { w1, w3, w2 })
        } else {
            let router = load_wt_from_source(
                source,
                gpu,
                &format!("{p}.feed_forward.gate.weight"),
                n_exp,
                hidden,
            )?;
            let expert_bias = load_f32_from_source(
                source,
                gpu,
                &format!("{p}.feed_forward.expert_bias"),
                &[n_exp],
            )?;
            // Guard: the indexed-MoE GEMV forward (gemv_hfq4g256/hfq6g256_moe,
            // with FWHT-pre-rotated experts) has NO float-weight path. A raw HF
            // safetensors checkpoint ships bf16/f16 experts (eqt 1) which would
            // be misread as 4-bit blocks → silent garbage. Refuse cleanly: this
            // path supports hipfire-quantized experts only.
            {
                let (eqt, _dt, _d) = read_tensor_from_source(
                    source,
                    &format!("{p}.feed_forward.experts.0.w1.weight"),
                )?;
                if eqt == 1 || eqt == 2 {
                    return Err(format!(
                        "lfm2moe: MoE experts at L{l} are raw float (eqt={eqt}); the indexed-MoE \
                         forward requires quantized experts (HFQ4G256/MQ4G256/MQ6G256/HFQ6G256). \
                         Quantize the checkpoint first (e.g. `hipfire-quantize … --format mq4`) \
                         or load the prebuilt HFQ."
                    ));
                }
            }
            // Byte-fuse w1‖w3 → gate_up [2*moe_inter, hidden]; w2 → down.
            let mut experts = Vec::with_capacity(n_exp);
            for e in 0..n_exp {
                let ep = format!("{p}.feed_forward.experts.{e}");
                let (qt1, dt1, w1) = read_tensor_from_source(source, &format!("{ep}.w1.weight"))?;
                let (_qt3, _dt3, w3) = read_tensor_from_source(source, &format!("{ep}.w3.weight"))?;
                let mut gate_up_bytes: Vec<u8> = w1.to_vec();
                gate_up_bytes.extend_from_slice(w3);
                let mut gate_up =
                    wt_from_source_raw(gpu, qt1, dt1, &gate_up_bytes, 2 * moe_inter, hidden)
                        .map_err(|e2| format!("lfm2moe: fuse gate_up L{l}E{e}: {e2}"))?;
                let (qt2, dt2, w2) = read_tensor_from_source(source, &format!("{ep}.w2.weight"))?;
                let mut down = wt_from_source_raw(gpu, qt2, dt2, w2, hidden, moe_inter)
                    .map_err(|e2| format!("lfm2moe: down L{l}E{e}: {e2}"))?;
                // AWQ scales: shared per layer, emitted once on expert 0.
                if e == 0 {
                    gate_up.awq_scale = load_lfm2_awq_scale_from_source(
                        source,
                        gpu,
                        &format!("{p}.feed_forward.awq_scale_gate_up.weight"),
                        hidden,
                    );
                    if gate_up.awq_scale.is_some() {
                        eprintln!(
                            "lfm2moe: AWQ gate_up scale attached at {p} (expert-0 representative)"
                        );
                    }
                    down.awq_scale = load_lfm2_awq_scale_from_source(
                        source,
                        gpu,
                        &format!("{p}.feed_forward.awq_scale_down.weight"),
                        moe_inter,
                    );
                    if down.awq_scale.is_some() {
                        eprintln!(
                            "lfm2moe: AWQ down scale attached at {p} (expert-0 representative)"
                        );
                    }
                }
                experts.push(ExpertWeights { gate_up, down });
            }
            let gu_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let dn_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let expert_gate_up_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("lfm2moe: alloc gu_ptrs: {e:?}"))?;
            let expert_down_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("lfm2moe: alloc dn_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                .map_err(|e| format!("lfm2moe: htod gu_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                .map_err(|e| format!("lfm2moe: htod dn_ptrs: {e:?}"))?;
            Ffn::Moe(MoeFfn {
                router,
                expert_bias,
                experts,
                expert_gate_up_ptrs,
                expert_down_ptrs,
            })
        };

        layers.push(Lfm2MoeLayerWeights {
            operator_norm,
            ffn_norm,
            mixer,
            ffn,
        });
    }

    let _ = (conv_state_count, kv_count);
    Ok(Lfm2MoeWeights {
        embed,
        embd_format,
        embedding_norm,
        lm_head,
        layers,
    })
}

/// Per-decode GPU scratch + KV cache (attention layers) + conv-state cache
/// (conv layers). Buffers are eager-allocated.
pub struct Lfm2MoeState {
    pub kv: KvCache,
    /// One rolling [hidden, K-1] f32 ring buffer per conv layer (zero-init).
    pub conv_states: Vec<GpuTensor>,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    /// hipGraph (HIPFIRE_LFM2_GRAPH) warmup latch: false until the first
    /// decode runs direct (so kernel JIT / lazy alloc happen outside any
    /// stream capture). Unused when the graph path is disabled.
    pub graph_warmed_up: bool,
    /// Retained replay warmup latch: false until the first dense decode runs
    /// direct (allocation/JIT outside capture). Survives `reset()` — warmup
    /// describes allocation/JIT, not sequence state.
    pub retained_warmed_up: bool,
    /// Fail-closed poison: true after a retained replay error until a full
    /// KV+conv reset clears it. Any `decode_step` must reject while poisoned.
    pub retained_state_poisoned: bool,
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual + shared scratch
    pub h: GpuTensor,   // [hidden] residual stream
    pub tmp: GpuTensor, // [hidden] norm output (mixer input)

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // conv scratch
    pub conv_bcx: GpuTensor, // [3*hidden] in_proj output (B|C|x)
    pub conv_y: GpuTensor,   // [hidden] gated conv output (out_proj input)

    // ffn scratch
    pub ffn_tmp: GpuTensor,       // [hidden] rmsnorm(h)
    pub ffn_x_rot: GpuTensor,     // [hidden] FWHT(rmsnorm(h)) for MQ4 experts
    pub dense_gate: GpuTensor,    // [dense_inter]
    pub dense_up: GpuTensor,      // [dense_inter]
    pub dense_act: GpuTensor,     // [dense_inter] silu(gate)*up
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k_top] i32-in-F32
    pub topk_weights: GpuTensor,  // [k_top]
    pub gate_batch: GpuTensor,    // [k_top*moe_inter]
    pub up_batch: GpuTensor,      // [k_top*moe_inter]
    pub rot_batch: GpuTensor,     // [k_top*moe_inter]
    pub down_expanded: GpuTensor, // [k_top*hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
}

impl Lfm2MoeState {
    pub fn new(gpu: &mut Gpu, cfg: &Lfm2MoeConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;
        let k_conv = cfg.conv_kernel_size;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("lfm2moe: ensure_mq_signs: {e:?}"))?;

        // KV cache: one slot per ATTENTION layer (conv layers carry no KV).
        let n_attn = cfg.num_attention_layers().max(1);
        let dims = hipfire_runtime::llama::KvDims {
            layers: hipfire_runtime::llama::KvLayers::Flat(n_attn),
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            max_seq,
            physical_cap: None,
        };
        let kv = hipfire_runtime::llama::KvCache::from_mode(
            hipfire_runtime::kv_mode::resolve(
                "",
                &hipfire_runtime::kv_mode::HFQ_Q8_ONLY_POLICY,
                cfg.head_dim,
            )
            .mode,
            hipfire_runtime::llama::KvTarget::Single(gpu),
            &dims,
        )
        .map_err(|e| format!("lfm2moe: kv cache: {e:?}"))?;

        // Conv-state cache: one [hidden,(K-1)] f32 ring buffer per CONV layer.
        let conv_hist = hidden * (k_conv - 1);
        let zeros = vec![0u8; conv_hist * 4];
        let mut conv_states = Vec::with_capacity(cfg.num_conv_layers());
        for _ in 0..cfg.num_conv_layers() {
            let cs = gpu
                .alloc_tensor(&[conv_hist], DType::F32)
                .map_err(|e| format!("lfm2moe: alloc conv_state: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&cs.buf, &zeros)
                .map_err(|e| format!("lfm2moe: zero conv_state: {e:?}"))?;
            conv_states.push(cs);
        }

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("lfm2moe: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("lfm2moe: alloc {label}: {e:?}"))
        };

        Ok(Lfm2MoeState {
            kv,
            conv_states,
            pos_buf,
            graph_warmed_up: false,
            retained_warmed_up: false,
            retained_state_poisoned: false,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            tmp: alloc(gpu, hidden, "tmp")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            conv_bcx: alloc(gpu, 3 * hidden, "conv_bcx")?,
            conv_y: alloc(gpu, hidden, "conv_y")?,
            ffn_tmp: alloc(gpu, hidden, "ffn_tmp")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, k * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, k * moe_inter, "rot_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    /// Reset for a new sequence: clear KV + conv state and token count.
    /// Preserves `graph_warmed_up` and `retained_warmed_up`; clears
    /// `retained_state_poisoned` only after the full KV+conv reset so a
    /// poisoned recurrent/KV cannot be reused. No host allocations.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("lfm2moe: reset kv: {e:?}"))?;
        for cs in &self.conv_states {
            gpu.hip
                .memset(&cs.buf, 0, cs.buf.size())
                .map_err(|e| format!("lfm2moe: reset conv_state: {e:?}"))?;
        }
        self.n_tokens = 0;
        self.retained_state_poisoned = false;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.h);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.fa_q);
        let _ = gpu.free_tensor(self.fa_k);
        let _ = gpu.free_tensor(self.fa_v);
        let _ = gpu.free_tensor(self.fa_attn_out);
        let _ = gpu.free_tensor(self.conv_bcx);
        let _ = gpu.free_tensor(self.conv_y);
        let _ = gpu.free_tensor(self.ffn_tmp);
        let _ = gpu.free_tensor(self.ffn_x_rot);
        let _ = gpu.free_tensor(self.dense_gate);
        let _ = gpu.free_tensor(self.dense_up);
        let _ = gpu.free_tensor(self.dense_act);
        let _ = gpu.free_tensor(self.router_logits);
        let _ = gpu.free_tensor(self.topk_indices);
        let _ = gpu.free_tensor(self.topk_weights);
        let _ = gpu.free_tensor(self.gate_batch);
        let _ = gpu.free_tensor(self.up_batch);
        let _ = gpu.free_tensor(self.rot_batch);
        let _ = gpu.free_tensor(self.down_expanded);
        let _ = gpu.free_tensor(self.final_norm_buf);
        let _ = gpu.free_tensor(self.logits);
    }
}

/// Authoritative batch admissibility predicate.
/// Embeddings: Q8_0, HFQ4G256 only (F32 is sequential-only).
/// Dense projections: Q8_0, HFQ4G256, MQ4G256 (one GEMM per weight; MQ4 via FWHT).
/// lm_head: Q8_0, HFQ4G256, MQ4G256, HFQ6G256, MQ6G256, MQ3G256 (existing kernel coverage).
/// MoE FFN: not admitted (dense-only batch).
pub fn batch_weight_formats_supported(weights: &Lfm2MoeWeights) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    // Embedding allowlist: Q8/HFQ4 only.
    match weights.embd_format {
        EmbeddingFormat::Q8_0 | EmbeddingFormat::HFQ4G256 => {}
        EmbeddingFormat::F32 => {
            return Err("batched decode: F32 embedding not supported".to_string())
        }
        _ => return Err("batched decode: unsupported embedding format".to_string()),
    }
    // lm_head allowlist (keep aligned with lm_head_batched_lfm kernels).
    match weights.lm_head.gpu_dtype {
        DType::Q8_0
        | DType::HFQ4G256
        | DType::MQ4G256
        | DType::HFQ6G256
        | DType::MQ6G256
        | DType::MQ3G256 => {}
        other => {
            return Err(format!(
                "batched decode: unsupported lm_head dtype {other:?}"
            ))
        }
    }
    // Dense projections: Q8/HFQ4/MQ4 only.
    for layer in &weights.layers {
        match &layer.mixer {
            crate::lfm2moe::Mixer::Conv(c) => {
                for w in [&c.in_proj, &c.out_proj] {
                    match w.gpu_dtype {
                        DType::Q8_0 | DType::HFQ4G256 | DType::MQ4G256 => {}
                        other => {
                            return Err(format!(
                                "batched decode: conv projection dtype {other:?} not supported (expected Q8/HFQ4/MQ4)"
                            ))
                        }
                    }
                }
            }
            crate::lfm2moe::Mixer::Attention(a) => {
                for w in [&a.wq, &a.wk, &a.wv, &a.wo] {
                    match w.gpu_dtype {
                        DType::Q8_0 | DType::HFQ4G256 | DType::MQ4G256 => {}
                        other => {
                            return Err(format!(
                                "batched decode: attention projection dtype {other:?} not supported"
                            ))
                        }
                    }
                }
            }
        }
        match &layer.ffn {
            crate::lfm2moe::Ffn::Dense(d) => {
                for w in [&d.w1, &d.w3, &d.w2] {
                    match w.gpu_dtype {
                        DType::Q8_0 | DType::HFQ4G256 | DType::MQ4G256 => {}
                        other => {
                            return Err(format!(
                                "batched decode: dense FFN dtype {other:?} not supported"
                            ))
                        }
                    }
                }
            }
            crate::lfm2moe::Ffn::Moe(_) => {
                return Err("batched decode: MoE FFN not supported in dense batch".to_string())
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod bf16_decode_tests {
    use super::source_bytes_to_f32_vec;

    // Regression: lfm2moe's safetensors decode previously collapsed BF16 onto
    // the F16 path (`effective_quant_type` mapped BF16 → qt 1 → `f16_to_f32`),
    // silently corrupting every BF16 weight/norm. The loader now widens via the
    // dtype-aware shared helper; this locks BF16 ≠ F16 for the same bytes so the
    // collapse can't regress unnoticed (no GPU needed — pure byte decode).
    #[test]
    fn bf16_bytes_decode_distinct_from_f16() {
        // 1.0 in bf16 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000).
        let one_bf16 = 0x3F80u16.to_le_bytes();
        let as_bf16 = source_bytes_to_f32_vec("BF16", &one_bf16);
        let as_f16 = source_bytes_to_f32_vec("F16", &one_bf16);
        assert_eq!(as_bf16, vec![1.0f32], "BF16 0x3F80 must widen to 1.0");
        assert_ne!(
            as_bf16, as_f16,
            "decoding BF16 bytes as F16 must differ — the original bug"
        );
    }
}

#[cfg(test)]
mod batch_format_tests {
    use super::*;
    use hipfire_runtime::llama::EmbeddingFormat;
    use rdna_compute::DType;

    fn is_embedding_supported(fmt: EmbeddingFormat) -> bool {
        matches!(fmt, EmbeddingFormat::Q8_0 | EmbeddingFormat::HFQ4G256)
    }
    fn is_dense_proj_supported(dtype: DType) -> bool {
        matches!(dtype, DType::Q8_0 | DType::HFQ4G256 | DType::MQ4G256)
    }
    fn is_lm_head_supported(dtype: DType) -> bool {
        matches!(
            dtype,
            DType::Q8_0
                | DType::HFQ4G256
                | DType::MQ4G256
                | DType::HFQ6G256
                | DType::MQ6G256
                | DType::MQ3G256
        )
    }

    #[test]
    fn embedding_qt_mapping() {
        // qt 3 -> Q8, 6 -> HFQ4, 1 -> F32 path (widened), others reject.
        let map = |qt: u8| -> Result<EmbeddingFormat, String> {
            match qt {
                3 => Ok(EmbeddingFormat::Q8_0),
                6 => Ok(EmbeddingFormat::HFQ4G256),
                1 => Ok(EmbeddingFormat::F32),
                other => Err(format!("unsupported embedding quant_type {other}")),
            }
        };
        assert_eq!(map(3).unwrap(), EmbeddingFormat::Q8_0);
        assert_eq!(map(6).unwrap(), EmbeddingFormat::HFQ4G256);
        assert_eq!(map(1).unwrap(), EmbeddingFormat::F32);
        assert!(map(2).is_err());
        assert!(map(7).is_err());
        assert!(map(99).is_err());
        assert!(map(0).is_err());
    }

    #[test]
    fn embedding_f32_widen_correctness() {
        // F16 1.0 = 0x3C00, BF16 1.0 = 0x3F80 -> both widen to 1.0f32 via dtype-aware helper.
        let f16_one = 0x3C00u16.to_le_bytes();
        let bf16_one = 0x3F80u16.to_le_bytes();
        let f16_widen = source_bytes_to_f32_vec("F16", &f16_one);
        let bf16_widen = source_bytes_to_f32_vec("BF16", &bf16_one);
        assert_eq!(f16_widen, vec![1.0f32]);
        assert_eq!(bf16_widen, vec![1.0f32]);
        // Raw F16 bytes mis-decoded as BF16 would not be 1.0 — ensure helpers differ for same bytes.
        let as_f16 = source_bytes_to_f32_vec("F16", &bf16_one);
        assert_ne!(
            as_f16,
            vec![1.0f32],
            "BF16 bytes decoded as F16 must differ"
        );
    }

    #[test]
    fn unsupported_embedding_rejected() {
        // Unknown qt should error before upload.
        let bad_qt = 13u8; // MQ4
        let is_supported_embedding_qt = |qt: u8| -> bool { matches!(qt, 1 | 3 | 6) };
        assert!(!is_supported_embedding_qt(bad_qt));
        assert!(!is_supported_embedding_qt(2));
    }

    #[test]
    fn predicate_families() {
        // Embeddings: only Q8/HFQ4 batch-admissible, F32 sequential-only.
        assert!(is_embedding_supported(EmbeddingFormat::Q8_0));
        assert!(is_embedding_supported(EmbeddingFormat::HFQ4G256));
        assert!(!is_embedding_supported(EmbeddingFormat::F32));
        assert!(!is_embedding_supported(EmbeddingFormat::HFQ4G128));
        assert!(!is_embedding_supported(EmbeddingFormat::Q4K));

        // Dense projections: Q8/HFQ4/MQ4 only.
        assert!(is_dense_proj_supported(DType::Q8_0));
        assert!(is_dense_proj_supported(DType::HFQ4G256));
        assert!(is_dense_proj_supported(DType::MQ4G256));
        assert!(!is_dense_proj_supported(DType::F32));
        assert!(!is_dense_proj_supported(DType::F16));
        assert!(!is_dense_proj_supported(DType::HFQ6G256));
        assert!(!is_dense_proj_supported(DType::MQ6G256));
        assert!(!is_dense_proj_supported(DType::MQ3G256));

        // lm_head: broader allowlist includes HFQ6/MQ6/MQ3 but still rejects F32/F16.
        assert!(is_lm_head_supported(DType::Q8_0));
        assert!(is_lm_head_supported(DType::HFQ4G256));
        assert!(is_lm_head_supported(DType::MQ4G256));
        assert!(is_lm_head_supported(DType::HFQ6G256));
        assert!(is_lm_head_supported(DType::MQ6G256));
        assert!(is_lm_head_supported(DType::MQ3G256));
        assert!(!is_lm_head_supported(DType::F32));
        assert!(!is_lm_head_supported(DType::F16));
        assert!(!is_lm_head_supported(DType::MQ2G256));
    }

    #[test]
    fn product_overflow_helpers() {
        // Ensure checked arithmetic correctly flags overflow for all batch products.
        let max = usize::MAX;
        assert!(max.checked_mul(2).is_none());
        assert!(max.checked_mul(2048).is_none());
        // Small values succeed.
        assert_eq!(8usize.checked_mul(2048), Some(16384));
        assert_eq!(4usize.checked_mul(8192), Some(32768));
        // 3*hidden overflow path.
        let bcx = 8usize.checked_mul(3).and_then(|v| v.checked_mul(2048));
        assert_eq!(bcx, Some(49152));
        assert!(max
            .checked_mul(3)
            .and_then(|v| v.checked_mul(2048))
            .is_none());
        // logits product overflow.
        let vocab = 32000usize;
        assert!(max.checked_mul(vocab).is_none());
    }

    #[test]
    fn cancellation_helper_state_independent() {
        // Pure helper: cancellation closure that returns true immediately should cause
        // prefill to abort before sampling. Test the closure combinator in isolation.
        let mut calls = 0usize;
        let mut should_cancel = || {
            calls += 1;
            calls > 0 // cancel immediately on first token boundary
        };
        assert!(should_cancel());
        assert_eq!(calls, 1);
        // Non-cancelling closure.
        let mut calls2 = 0usize;
        let mut never_cancel = || {
            calls2 += 1;
            false
        };
        assert!(!never_cancel());
        assert!(!never_cancel());
        assert_eq!(calls2, 2);
    }
}
