// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Weights + state: loading, KV caches, layer mapping.
//!
//! Tensors live under `model.language_model.*` except `lm_head.weight`
//! which is a separate (untied) tensor, NOT an alias of embed_tokens
//! (see `lib.rs`).

use crate::config::GlimmerConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Upper bound on the DFlash speculation block, used to size `logits_batch`
/// once at init. The Glimmer assistant's trained `block_size` is 16.
pub const GLIMMER_MAX_SPEC_BLOCK: usize = 32;

// ───────────────────────── HFQ load helpers ─────────────────────────

fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer: tensor not found: {name}"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if expected_n != 0 && n != expected_n {
        return Err(format!(
            "glimmer: shape mismatch for {name}: expected {expected_n}, got {n}"
        ));
    }
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        qt => {
            return Err(format!(
                "glimmer: unexpected quant_type {qt} for f32 vec {name}"
            ))
        }
    };
    Ok(f32_data)
}

/// Load an RMSNorm weight.
///
/// Muse Glimmer uses **two different norm conventions**, and mixing them up is a
/// silent-wrongness trap. Per HF `modeling_muse_glimmer.py`:
///
/// | tensor | class | convention |
/// |---|---|---|
/// | `input_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | `post_attention_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | `pre_feedforward_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | `post_feedforward_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | final `norm` | `MuseGlimmerRMSNorm` | plain `x_norm * w` |
/// | `qk_norm`, `embed_norm` | `MuseGlimmerRMSNorm(with_scale=False)` | scale-less |
///
/// `centered` bakes `1 + w` at load so the hot path stays on the ordinary
/// `rmsnorm_f32` kernel — no new kernel, no per-call cost.
///
/// The centered classification is corroborated by the checkpoint itself: the
/// post-norms store NEGATIVE weights (`post_attention_layernorm` −0.523, −0.480,
/// −0.237; `post_feedforward_layernorm` −0.357, −0.371, −0.192), impossible under
/// plain `x * w` since they would flip the residual's sign. Under `1 + w` they
/// become sensible scales of ~0.48–0.89. The final `norm` stores ±3.x, which is
/// only sensible under the PLAIN convention — centering it was a real bug.
///
/// Opt out of centering with `HIPFIRE_GLIMMER_NO_CENTERED_NORM=1` for A/B.
fn load_norm_with(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
    centered: bool,
) -> Result<GpuTensor, String> {
    let mut f32_data = load_f32_vec(hfq, name, dim)?;
    if centered
        && std::env::var("HIPFIRE_GLIMMER_NO_CENTERED_NORM")
            .ok()
            .as_deref()
            != Some("1")
    {
        for v in f32_data.iter_mut() {
            *v += 1.0;
        }
    }
    gpu.upload_f32(&f32_data, &[dim])
        .map_err(|e| format!("glimmer: upload norm {name}: {e:?}"))
}

/// Centered (`1 + w`) norm — the four per-decoder-layer norms.
fn load_norm(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
) -> Result<GpuTensor, String> {
    load_norm_with(hfq, gpu, name, dim, true)
}

/// Plain (`w`) norm — the final `model.language_model.norm`.
fn load_norm_plain(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
) -> Result<GpuTensor, String> {
    load_norm_with(hfq, gpu, name, dim, false)
}


/// quant_type → DType mapping for projection weights.
/// F16 is dequantized to F32 on upload.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer: tensor not found: {name}"))?;
    if info.quant_type == 1 {
        let f32_data: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let buf = gpu
            .upload_f32(&f32_data, &[m, k])
            .map_err(|e| format!("glimmer: upload F32 {name}: {e:?}"))?;
        return Ok(WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        });
    }
    let dtype = match info.quant_type {
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m, k])
                .map_err(|e| format!("glimmer: upload F32 {name}: {e:?}"))?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            });
        }
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        11 => DType::HFQ3G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        19 => DType::MQ4G256,
        qt => return Err(format!("glimmer: unsupported quant_type {qt} for {name}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("glimmer: upload {name}: {e:?}"))?;
    let awq_scale = if dtype.supports_awq_sidecar() {
        hipfire_runtime::hfq::load_awq_scale(hfq, gpu, name, k)
    } else {
        None
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale,
    })
}

// ──────────────────────────── Weights ────────────────────────────

/// Per-layer weights. Glimmer uses uniform head_dim 128, so a single struct
/// covers both sliding and full layers (unlike Gemma4's Sliding/Full split).
pub struct GlimmerLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    /// Gated attention gate: `self_attn.gate_proj` — NOT the MLP gate.
    pub attn_gate_proj: WeightTensor,
    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub gate_proj: WeightTensor, // mlp.gate_proj
    pub up_proj: WeightTensor,   // mlp.up_proj
    pub down_proj: WeightTensor, // mlp.down_proj
}

pub struct GlimmerWeights {
    /// Token embedding [vocab, dim]
    pub embed_tokens: GpuTensor,
    pub embd_format: EmbeddingFormat,
    /// LM head — separate allocation (untied). NOT an alias of embed_tokens.
    pub lm_head: WeightTensor,
    pub final_norm: GpuTensor,
    pub layers: Vec<GlimmerLayerWeights>,
}

impl GlimmerWeights {
    pub fn load(hfq: &HfqFile, cfg: &GlimmerConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let dim = cfg.dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hidden_dim = cfg.hidden_dim;

        // ── Embedding ──────────────────────────────────────────────────
        let embed_name = "model.language_model.embed_tokens.weight";
        let (embed_info, embed_data) = hfq
            .tensor_data(embed_name)
            .ok_or_else(|| "glimmer: embed_tokens not found in HFQ".to_string())?;
        let (embed_tokens, embd_format) = match embed_info.quant_type {
            3 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("glimmer: upload embed: {e:?}"))?,
                EmbeddingFormat::Q8_0,
            ),
            6 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("glimmer: upload embed: {e:?}"))?,
                EmbeddingFormat::HFQ4G256,
            ),
            7 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("glimmer: upload embed: {e:?}"))?,
                EmbeddingFormat::HFQ4G128,
            ),
            1 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, dim])
                        .map_err(|e| format!("glimmer: upload embed f32: {e:?}"))?,
                    EmbeddingFormat::F32,
                )
            }
            2 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, dim])
                        .map_err(|e| format!("glimmer: upload embed f32: {e:?}"))?,
                    EmbeddingFormat::F32,
                )
            }
            qt => return Err(format!("glimmer: unsupported embed quant_type {qt}")),
        };

        // ── Untied LM head ─────────────────────────────────────────────
        // Separate tensor "lm_head.weight" [vocab, dim], not alias of embed.
        let lm_head = load_wt(hfq, gpu, "lm_head.weight", cfg.vocab_size, dim)?;

        // PLAIN norm: HF builds the final `norm` as MuseGlimmerRMSNorm
        // (with_scale=True), NOT MuseGlimmerTextCenteredRMSNorm. Centering it
        // shifts every logit and was a real bring-up bug.
        let final_norm = load_norm_plain(
            hfq,
            gpu,
            "model.language_model.norm.weight",
            dim,
        )?;

        // ── Layers ─────────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.language_model.layers.{i}");
            layers.push(GlimmerLayerWeights {
                input_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    dim,
                )?,
                post_attention_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    dim,
                )?,
                pre_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    dim,
                )?,
                post_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    dim,
                )?,
                attn_gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    q_dim,
                    dim,
                )?,
                q_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    dim,
                )?,
                k_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    dim,
                )?,
                v_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    dim,
                )?,
                o_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    dim,
                    q_dim,
                )?,
                gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    hidden_dim,
                    dim,
                )?,
                up_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    hidden_dim,
                    dim,
                )?,
                down_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    dim,
                    hidden_dim,
                )?,
            });
        }

        Ok(GlimmerWeights {
            embed_tokens,
            embd_format,
            lm_head,
            final_norm,
            layers,
        })
    }

    /// Return all GPU weight buffers to the pool. Consumes self.
    /// lm_head is a separate allocation (untied) and IS freed here,
    /// unlike Gemma4's tied alias.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        self.lm_head.free_all(gpu);
        let _ = gpu.free_tensor(self.final_norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            let _ = gpu.free_tensor(l.pre_feedforward_layernorm);
            let _ = gpu.free_tensor(l.post_feedforward_layernorm);
            l.attn_gate_proj.free_all(gpu);
            l.q_proj.free_all(gpu);
            l.k_proj.free_all(gpu);
            l.v_proj.free_all(gpu);
            l.o_proj.free_all(gpu);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV caches.
///
/// # KV cache topology
/// Head_dim is uniform 128 (`lib.rs` — unlike Gemma4's 256/512 split), so
/// a single cache shape would suffice physically. We retain the dual-cache
/// topology (sliding + full) for logical isolation: sliding layers attend
/// with a 2048 window while full layers attend full-causal (window=0). The
/// two caches share identical geometry `(head_dim=128, n_kv=2)` and differ
/// only in layer count (39 vs 13) and window at attention dispatch. A single
/// unified cache would also be valid and save one bookkeeping vector; we
/// keep the split to mirror Gemma4's proven eviction/windowing discipline
/// without paying a shape-mismatch cost.
pub struct GlimmerState {
    /// Sliding-window KV cache (head_dim 128), one slot per sliding layer.
    pub kv_sliding: KvCache,
    /// Full-attention KV cache (head_dim 128), one slot per full layer.
    pub kv_full: KvCache,
    /// Per-layer slot index into the matching per-type cache.
    pub kv_slot_for_layer: Vec<usize>,

    pub pos_buf: hip_bridge::DeviceBuffer,
    /// Stable host source for the device position scalar (hipGraph-safe if ever
    /// captured: the captured memcpy re-reads this Box on replay).
    pub pos_host: Box<[i32]>,
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual stream + scratch
    pub x: GpuTensor,        // [dim]
    pub residual: GpuTensor, // [dim]
    pub tmp: GpuTensor,      // [dim] norm scratch
    pub x_rot: GpuTensor,    // [dim] FWHT scratch for shared rotation (MQ4)

    // attention scratch (uniform dims)
    pub q: GpuTensor,        // [q_dim]
    pub k: GpuTensor,        // [kv_dim]
    pub v: GpuTensor,        // [kv_dim]
    pub attn_out: GpuTensor, // [q_dim]
    pub attn_gate: GpuTensor, // [q_dim] sigmoid gate

    /// Ones-filled weight for scale-less QK-norm (head_dim ones).
    pub qk_norm_ones: GpuTensor, // [head_dim]
    /// Ones-filled weight for the scale-less embedding norm (hidden_size ones).
    ///
    /// HF wraps the embedding table in `MuseGlimmerTextNormedEmbedding`, whose
    /// `forward` is `embed_norm(Embedding::forward(ids))` with
    /// `MuseGlimmerRMSNorm(eps=rms_norm_eps, with_scale=False)`. The norm is
    /// deliberately NOT folded into the embedding matrix upstream ("Dflash
    /// implem needs to embed without the norm"), so it must run per lookup.
    pub embed_norm_ones: GpuTensor, // [hidden_size]

    // FFN scratch
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // head
    pub logits: GpuTensor, // [vocab]
    /// Persistent [block_max * vocab] logits buffer for the batched lm_head.
    ///
    /// Allocated ONCE. The batched lm_head previously did `alloc_tensor` +
    /// `free_tensor` of this ~12.9 MB buffer on every call, and a cold
    /// `hipMalloc` is both slow and synchronizing — which is what made the
    /// FIRST batched lm_head of each window (the draft's) cost 69 ms while the
    /// SECOND (verify's, reusing the block the first had just freed) cost 7.6 ms
    /// for the same weight through the same kernel.
    pub logits_batch: GpuTensor, // [GLIMMER_MAX_SPEC_BLOCK * vocab]
    /// Batched flash-attention partials for prefill over-window recovery.
    /// Lazily allocated on first over-window sliding chunk: n_heads *
    /// ceil(max_seq/128) * (2+head_dim) * 64 floats (~65 MiB at max_seq=8192).
    /// Factor-64 precedent: crates/hipfire-arch-cohere2moe/src/cohere2moe.rs:496-511.
    pub prefill_flash_partials: Option<GpuTensor>,
    /// Single-element i32 position tensor for the flash decode path. `pos_buf`
    /// is a raw DeviceBuffer, but the batched flash kernel takes a GpuTensor of
    /// positions; at batch_size=1 it holds the same value. Lazily allocated.
    pub decode_pos: Option<GpuTensor>,
}

impl GlimmerState {
    pub fn new(gpu: &mut Gpu, cfg: &GlimmerConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let dim = cfg.dim;

        // Two Q8 KV caches: one slot per layer of the matching type.
        // Both have identical head_dim=128 geometry; split is logical.
        let kv_sliding = KvCache::new_gpu_q8(
            gpu,
            cfg.n_sliding_layers(),
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("glimmer: sliding kv cache: {e:?}"))?;
        let kv_full = KvCache::new_gpu_q8(
            gpu,
            cfg.n_full_layers(),
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("glimmer: full kv cache: {e:?}"))?;

        // Per-layer slot mapping: sequential count within each type.
        let mut kv_slot_for_layer = Vec::with_capacity(cfg.n_layers);
        let mut s = 0usize;
        let mut f = 0usize;
        for &lt in cfg.layer_types.iter() {
            match lt {
                crate::config::GlimmerLayerType::Sliding => {
                    kv_slot_for_layer.push(s);
                    s += 1;
                }
                crate::config::GlimmerLayerType::Full => {
                    kv_slot_for_layer.push(f);
                    f += 1;
                }
            }
        }

        // FWHT sign LUT must exist before any fused_rmsnorm_rotate_mq
        // launch (the shared-rotation path). Mirrors gemma4's ensure at state init.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("glimmer: ensure_mq_signs: {e:?}"))?;

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("glimmer: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("glimmer: alloc {label}: {e:?}"))
        };

        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hd = cfg.head_dim;

        // Ones for scale-less QK-norm.
        let qk_norm_ones = alloc(gpu, hd, "qk_norm_ones")?;
        {
            let ones: Vec<f32> = vec![1.0; hd];
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
            gpu.hip
                .memcpy_htod(&qk_norm_ones.buf, bytes)
                .map_err(|e| format!("glimmer: init qk_norm_ones: {e:?}"))?;
        }

        let embed_norm_ones = alloc(gpu, dim, "embed_norm_ones")?;
        {
            let ones: Vec<f32> = vec![1.0; dim];
            let bytes =
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
            gpu.hip.memcpy_htod(&embed_norm_ones.buf, bytes)
                .map_err(|e| format!("glimmer: upload embed_norm_ones: {e:?}"))?;
        }

        Ok(GlimmerState {
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            x: alloc(gpu, dim, "x")?,
            residual: alloc(gpu, dim, "residual")?,
            tmp: alloc(gpu, dim, "tmp")?,
            x_rot: alloc(gpu, dim, "x_rot")?,
            q: alloc(gpu, q_dim, "q")?,
            k: alloc(gpu, kv_dim, "k")?,
            v: alloc(gpu, kv_dim, "v")?,
            attn_out: alloc(gpu, q_dim, "attn_out")?,
            attn_gate: alloc(gpu, q_dim, "attn_gate")?,
            qk_norm_ones,
            embed_norm_ones,
            gate_ffn: alloc(gpu, cfg.hidden_dim, "gate_ffn")?,
            up_ffn: alloc(gpu, cfg.hidden_dim, "up_ffn")?,
            ffn_hidden: alloc(gpu, cfg.hidden_dim, "ffn_hidden")?,
            ffn_out: alloc(gpu, dim, "ffn_out")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            logits_batch: alloc(
                gpu,
                GLIMMER_MAX_SPEC_BLOCK * cfg.vocab_size,
                "logits_batch",
            )?,
            prefill_flash_partials: None,
            decode_pos: None,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
    }

    /// Return all GPU state buffers to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv_sliding.free_gpu(gpu);
        let _ = self.kv_full.free_gpu(gpu);
        let _ = gpu.hip.free(self.pos_buf);
        for t in [
            self.x,
            self.residual,
            self.tmp,
            self.x_rot,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.attn_gate,
            self.qk_norm_ones,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.logits_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.decode_pos {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.prefill_flash_partials {
            let _ = gpu.free_tensor(t);
        }
    }
}
