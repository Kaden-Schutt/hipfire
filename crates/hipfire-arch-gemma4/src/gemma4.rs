// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense (text-only) weights + per-decode state.
//!
//! Ported from the old branch's `crate::engine::gemma4` loader, dropping the
//! MoE branch (`enable_moe_block=false` on 12B), the E-series per-layer
//! embedding / KV-sharing / double-wide-MLP machinery, and vision. Tensor names
//! carry the `model.language_model.*` prefix. lm_head is TIED to embed_tokens
//! (alias buffer). Two KV caches are held: sliding (head_dim 256) and full
//! (head_dim 512), each sized to the count of its layer type.

use crate::config::{Gemma4Config, LayerType};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

// ───────────────────────── HFQ load helpers ─────────────────────────

/// Decode a shape-[n] F16/F32 tensor into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4: tensor not found: {name}"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if expected_n != 0 && n != expected_n {
        return Err(format!(
            "gemma4: shape mismatch for {name}: expected {expected_n}, got {n}"
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
        qt => return Err(format!("gemma4: expected F16/F32 for {name}, got qt={qt}")),
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — plain `x * w` form (NO +1 shift), unless the
/// `norm_plus_one` config toggle is set (`HIPFIRE_GEMMA4_NORM_PLUS_ONE=1`), in
/// which case the Gemma-2/3 `x * (1 + w)` convention is baked at load time.
fn load_norm(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
    plus_one: bool,
) -> Result<GpuTensor, String> {
    let mut f32_data = load_f32_vec(hfq, name, dim)?;
    if plus_one {
        for v in f32_data.iter_mut() {
            *v += 1.0;
        }
    }
    gpu.upload_f32(&f32_data, &[dim])
        .map_err(|e| format!("gemma4: upload norm {name}: {e:?}"))
}

/// Load the learned per-layer `layer_scalar` (shape-[1]); returns the host f32
/// value so decode can call `gpu.scale_f32(x, scalar)` with no D2H round-trip.
/// Returns `1.0` (no-op) when the tensor is absent — the 12B may not ship it.
fn load_layer_scalar(hfq: &HfqFile, name: &str) -> f32 {
    match load_f32_vec(hfq, name, 1) {
        Ok(v) => v[0],
        Err(_) => 1.0,
    }
}

/// quant_type → DType mapping for Gemma 4 projection weights. Mirrors the old
/// branch's `load_gemma4_weight`. F16 is dequantized to F32 on upload.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4: tensor not found: {name}"))?;
    if info.quant_type == 1 {
        // F16 → upload as F32.
        let f32_data: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
        let buf = gpu
            .upload_raw(bytes, &[m, k])
            .map_err(|e| format!("gemma4: upload F32 {name}: {e:?}"))?;
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
            let buf = gpu
                .upload_raw(data, &[m, k])
                .map_err(|e| format!("gemma4: upload F32 {name}: {e:?}"))?;
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
        // MG4G256 (qt=19) reads back as MQ4G256: identical binary layout.
        19 => DType::MQ4G256,
        qt => return Err(format!("gemma4: unsupported quant_type {qt} for {name}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("gemma4: upload {name}: {e:?}"))?;
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

/// Per-layer weights for a SLIDING layer (head_dim 256, full RoPE, own v_proj).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar_host: f32,

    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim]
    pub k_norm: GpuTensor, // [head_dim]

    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

/// Per-layer weights for a FULL layer (head_dim 512, partial RoPE).
///
/// `attention_k_eq_v == true` (12B): no `v_proj`; V is the PRE-k_norm output of
/// k_proj, renormed by a weight-less (ones) RMSNorm. When false (other Gemma 4
/// variants), a separate `v_proj` is loaded.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar_host: f32,

    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: Option<WeightTensor>, // None when attention_k_eq_v
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim]
    pub k_norm: GpuTensor, // [head_dim]
    // no v_norm weight — v is no-scale (ones buffer passed at decode time)

    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub struct Gemma4Weights {
    /// Token embedding [vocab, dim]; aliased as lm_head when tied.
    pub embed_tokens: GpuTensor,
    pub embd_format: EmbeddingFormat,
    /// LM head (shares bytes with `embed_tokens` when tied).
    pub lm_head: WeightTensor,
    pub final_norm: GpuTensor,
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn load(hfq: &HfqFile, cfg: &Gemma4Config, gpu: &mut Gpu) -> Result<Self, String> {
        let dim = cfg.dim;
        let plus_one = cfg.norm_plus_one;

        // ── Embedding ──────────────────────────────────────────────────────
        let embed_name = "model.language_model.embed_tokens.weight";
        let (embed_info, embed_data) = hfq
            .tensor_data(embed_name)
            .ok_or_else(|| "gemma4: embed_tokens not found in HFQ".to_string())?;
        let (embed_tokens, embd_format) = match embed_info.quant_type {
            3 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("gemma4: upload embed: {e:?}"))?,
                EmbeddingFormat::Q8_0,
            ),
            6 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("gemma4: upload embed: {e:?}"))?,
                EmbeddingFormat::HFQ4G256,
            ),
            7 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("gemma4: upload embed: {e:?}"))?,
                EmbeddingFormat::HFQ4G128,
            ),
            1 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, dim])
                        .map_err(|e| format!("gemma4: upload embed f32: {e:?}"))?,
                    EmbeddingFormat::F32,
                )
            }
            qt => return Err(format!("gemma4: unsupported embed quant_type {qt}")),
        };

        // Tied lm_head: WeightTensor aliasing the embed allocation. free path
        // skips it (embed owns the bytes).
        let lm_head = {
            let alias_buf = unsafe { embed_tokens.buf.alias() };
            let dtype = match embd_format {
                EmbeddingFormat::Q8_0 => DType::Q8_0,
                EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
                EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
                EmbeddingFormat::F32 => DType::F32,
                EmbeddingFormat::Q4K => DType::Q4K,
            };
            let alias_tensor = GpuTensor {
                buf: alias_buf,
                shape: embed_tokens.shape.clone(),
                dtype,
            };
            WeightTensor {
                buf: alias_tensor,
                gpu_dtype: dtype,
                m: cfg.vocab_size,
                k: dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        };

        let final_norm = load_norm(
            hfq,
            gpu,
            "model.language_model.norm.weight",
            dim,
            plus_one,
        )?;

        // ── Layers ─────────────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.language_model.layers.{i}");
            let layer_scalar_host = load_layer_scalar(hfq, &format!("{p}.layer_scalar"));
            let ffn_hd = cfg.hidden_dim;
            match cfg.layer_types[i] {
                LayerType::Sliding => {
                    let hd = cfg.sliding_head_dim;
                    let kv_dim = cfg.sliding_n_kv_heads * hd;
                    let q_dim = cfg.n_heads * hd;
                    layers.push(LayerWeights::Sliding(SlidingLayerWeights {
                        input_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.input_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        post_attention_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.post_attention_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        pre_feedforward_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.pre_feedforward_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        post_feedforward_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.post_feedforward_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        layer_scalar_host,
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
                        q_norm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.q_norm.weight"),
                            hd,
                            plus_one,
                        )?,
                        k_norm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.k_norm.weight"),
                            hd,
                            plus_one,
                        )?,
                        gate_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.gate_proj.weight"),
                            ffn_hd,
                            dim,
                        )?,
                        up_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.up_proj.weight"),
                            ffn_hd,
                            dim,
                        )?,
                        down_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.down_proj.weight"),
                            dim,
                            ffn_hd,
                        )?,
                    }));
                }
                LayerType::Full => {
                    let hd = cfg.full_head_dim;
                    let kv_dim = cfg.full_n_kv_heads * hd;
                    let q_dim = cfg.n_heads * hd;
                    let v_proj = if cfg.attention_k_eq_v {
                        None
                    } else {
                        Some(load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.v_proj.weight"),
                            kv_dim,
                            dim,
                        )?)
                    };
                    layers.push(LayerWeights::Full(FullLayerWeights {
                        input_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.input_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        post_attention_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.post_attention_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        pre_feedforward_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.pre_feedforward_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        post_feedforward_layernorm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.post_feedforward_layernorm.weight"),
                            dim,
                            plus_one,
                        )?,
                        layer_scalar_host,
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
                        v_proj,
                        o_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.o_proj.weight"),
                            dim,
                            q_dim,
                        )?,
                        q_norm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.q_norm.weight"),
                            hd,
                            plus_one,
                        )?,
                        k_norm: load_norm(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.k_norm.weight"),
                            hd,
                            plus_one,
                        )?,
                        gate_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.gate_proj.weight"),
                            ffn_hd,
                            dim,
                        )?,
                        up_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.up_proj.weight"),
                            ffn_hd,
                            dim,
                        )?,
                        down_proj: load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.down_proj.weight"),
                            dim,
                            ffn_hd,
                        )?,
                    }));
                }
            }
        }

        Ok(Gemma4Weights {
            embed_tokens,
            embd_format,
            lm_head,
            final_norm,
            layers,
        })
    }

    /// Return all GPU weight buffers to the pool (drained on unload by the
    /// daemon's `unload_model`). Consumes self.
    ///
    /// `lm_head` is NOT freed here: it is always an alias of `embed_tokens`'
    /// allocation (tied embeddings — see the alias construction in `load`).
    /// `embed_tokens` owns the bytes and is freed exactly once below; dropping
    /// the alias is a no-op (`DeviceBuffer` has no `Drop`).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        for l in self.layers {
            match l {
                LayerWeights::Sliding(l) => {
                    let _ = gpu.free_tensor(l.input_layernorm);
                    let _ = gpu.free_tensor(l.post_attention_layernorm);
                    let _ = gpu.free_tensor(l.pre_feedforward_layernorm);
                    let _ = gpu.free_tensor(l.post_feedforward_layernorm);
                    l.q_proj.free_all(gpu);
                    l.k_proj.free_all(gpu);
                    l.v_proj.free_all(gpu);
                    l.o_proj.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    l.gate_proj.free_all(gpu);
                    l.up_proj.free_all(gpu);
                    l.down_proj.free_all(gpu);
                }
                LayerWeights::Full(l) => {
                    let _ = gpu.free_tensor(l.input_layernorm);
                    let _ = gpu.free_tensor(l.post_attention_layernorm);
                    let _ = gpu.free_tensor(l.pre_feedforward_layernorm);
                    let _ = gpu.free_tensor(l.post_feedforward_layernorm);
                    l.q_proj.free_all(gpu);
                    l.k_proj.free_all(gpu);
                    if let Some(v) = l.v_proj {
                        v.free_all(gpu);
                    }
                    l.o_proj.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    l.gate_proj.free_all(gpu);
                    l.up_proj.free_all(gpu);
                    l.down_proj.free_all(gpu);
                }
            }
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + the two KV caches (sliding + full).
///
/// Scratch is sized once against the MAX of sliding/full attention dims so a
/// single buffer set works across both layer types. `kv_slot_for_layer` maps
/// each layer ordinal to its slot within the per-type cache.
pub struct Gemma4State {
    /// Sliding-window KV cache (head_dim 256), one slot per sliding layer.
    pub kv_sliding: KvCache,
    /// Full-attention KV cache (head_dim 512), one slot per full layer.
    pub kv_full: KvCache,
    /// Per-layer slot index into the matching per-type cache.
    pub kv_slot_for_layer: Vec<usize>,

    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    /// Stable host source for the device position scalar. The hipGraph decode
    /// path captures a `memcpy_htod_auto` from these bytes; the captured node
    /// re-reads this heap-stable `Box` on every replay (see
    /// `decode_step_with_graph`). Updated host-side before each `graph_launch`.
    pub pos_host: Box<[i32]>,
    pub max_seq: usize,
    pub n_tokens: usize,
    /// hipGraph warmup gate: the first decode after a fresh load runs eager
    /// (no capture) to JIT-compile kernels + settle DPM, then the next call
    /// captures. Survives turn resets (the graph stays valid for the same model
    /// — only weight pointers + device buffers are baked, and those are stable).
    pub ar_warmed_up: bool,

    // residual stream + scratch
    pub x: GpuTensor,        // [dim]
    pub residual: GpuTensor, // [dim]
    pub tmp: GpuTensor,      // [dim] norm / o_proj scratch
    /// FWHT-rotated rmsnorm output for the fused MQ4 FFN path
    /// (`fused_rmsnorm_rotate_mq` → `fused_gate_up_hfq4g256`). [dim]
    pub tmp_rot: GpuTensor,

    // attention scratch (sized to max over layer types)
    pub q: GpuTensor,        // [max_q_dim]
    pub k: GpuTensor,        // [max_kv_dim]
    pub v: GpuTensor,        // [max_kv_dim]
    pub attn_out: GpuTensor, // [max_q_dim]

    /// Ones-filled weight buffer for the weight-less V RMSNorm on full layers.
    pub v_norm_ones: GpuTensor, // [max_head_dim]

    // FFN scratch
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // head
    pub logits: GpuTensor, // [vocab]
}

impl Gemma4State {
    pub fn new(gpu: &mut Gpu, cfg: &Gemma4Config) -> Result<Self, String> {
        // Cap the KV cache so the 262144-ctx config doesn't OOM.
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Gemma4Config,
        max_seq: usize,
    ) -> Result<Self, String> {
        let dim = cfg.dim;

        // FWHT sign LUT must exist before any fused_rmsnorm_rotate_mq /
        // fused_gate_up_hfq4g256 launch (the MQ4 fused FFN path).
        gpu.ensure_mq_signs()
            .map_err(|e| format!("gemma4: ensure_mq_signs: {e:?}"))?;

        // Two Q8 KV caches: one slot per layer of the matching type.
        let kv_sliding = KvCache::new_gpu_q8(
            gpu,
            cfg.n_sliding_layers(),
            cfg.sliding_n_kv_heads,
            cfg.sliding_head_dim,
            max_seq,
        )
        .map_err(|e| format!("gemma4: sliding kv cache: {e:?}"))?;
        let kv_full = KvCache::new_gpu_q8(
            gpu,
            cfg.n_full_layers(),
            cfg.full_n_kv_heads,
            cfg.full_head_dim,
            max_seq,
        )
        .map_err(|e| format!("gemma4: full kv cache: {e:?}"))?;

        // Per-layer slot mapping: sequential count within each type.
        let mut kv_slot_for_layer = Vec::with_capacity(cfg.n_layers);
        let mut s = 0usize;
        let mut f = 0usize;
        for &lt in cfg.layer_types.iter() {
            match lt {
                LayerType::Sliding => {
                    kv_slot_for_layer.push(s);
                    s += 1;
                }
                LayerType::Full => {
                    kv_slot_for_layer.push(f);
                    f += 1;
                }
            }
        }

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("gemma4: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("gemma4: alloc {label}: {e:?}"))
        };

        let max_q = cfg.max_q_dim();
        let max_kv = cfg.max_kv_dim();
        let max_hd = cfg.max_head_dim();

        // Ones-filled weight buffer for the weight-less V RMSNorm.
        let v_norm_ones = alloc(gpu, max_hd, "v_norm_ones")?;
        {
            let ones: Vec<f32> = vec![1.0; max_hd];
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
            gpu.hip
                .memcpy_htod(&v_norm_ones.buf, bytes)
                .map_err(|e| format!("gemma4: init v_norm_ones: {e:?}"))?;
        }

        Ok(Gemma4State {
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            ar_warmed_up: false,
            x: alloc(gpu, dim, "x")?,
            residual: alloc(gpu, dim, "residual")?,
            tmp: alloc(gpu, dim, "tmp")?,
            tmp_rot: alloc(gpu, dim, "tmp_rot")?,
            q: alloc(gpu, max_q, "q")?,
            k: alloc(gpu, max_kv, "k")?,
            v: alloc(gpu, max_kv, "v")?,
            attn_out: alloc(gpu, max_q, "attn_out")?,
            v_norm_ones,
            gate_ffn: alloc(gpu, cfg.hidden_dim, "gate_ffn")?,
            up_ffn: alloc(gpu, cfg.hidden_dim, "up_ffn")?,
            ffn_hidden: alloc(gpu, cfg.hidden_dim, "ffn_hidden")?,
            ffn_out: alloc(gpu, dim, "ffn_out")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    /// Like [] but replaces the full-layer KV cache with
    /// FWHT-512 3-bit K + Q8_0 V (fwht3).  The sliding-layer KV cache stays
    /// Q8_0 (sliding layers use head_dim=256, which has the same byte layout).
    pub fn new_with_fwht3_max_seq(
        gpu: &mut Gpu,
        cfg: &Gemma4Config,
        max_seq: usize,
    ) -> Result<Self, String> {
        let dim = cfg.dim;
        gpu.ensure_mq_signs()
            .map_err(|e| format!("gemma4-fwht3: ensure_mq_signs: {e:?}"))?;

        // Sliding layers: Q8 (hd=256, ring-capped to sliding_window).
        let kv_sliding = KvCache::new_gpu_q8(
            gpu,
            cfg.n_sliding_layers(),
            cfg.sliding_n_kv_heads,
            cfg.sliding_head_dim,
            max_seq,
        )
        .map_err(|e| format!("gemma4-fwht3: sliding kv cache: {e:?}"))?;

        // Full layers: fwht3 hd=512 (K = FWHT-512 3-bit, V = Q8_0).
        let all_true: Vec<bool> = vec![true; cfg.n_full_layers()];
        let kv_full = KvCache::new_gpu_fwht3_capped_filtered_gemma4(
            gpu,
            &all_true,
            cfg.full_n_kv_heads,
            cfg.full_head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("gemma4-fwht3: full kv cache: {e:?}"))?;

        // Per-layer slot mapping (identical to new_with_max_seq).
        let mut kv_slot_for_layer = Vec::with_capacity(cfg.n_layers);
        let mut s = 0usize;
        let mut f = 0usize;
        for &lt in cfg.layer_types.iter() {
            match lt {
                LayerType::Sliding => { kv_slot_for_layer.push(s); s += 1; }
                LayerType::Full    => { kv_slot_for_layer.push(f); f += 1; }
            }
        }

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("gemma4-fwht3: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("gemma4-fwht3: alloc {label}: {e:?}"))
        };

        let max_q = cfg.max_q_dim();
        let max_kv = cfg.max_kv_dim();
        let max_hd = cfg.max_head_dim();

        let v_norm_ones = alloc(gpu, max_hd, "v_norm_ones")?;
        {
            let ones: Vec<f32> = vec![1.0; max_hd];
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
            gpu.hip
                .memcpy_htod(&v_norm_ones.buf, bytes)
                .map_err(|e| format!("gemma4-fwht3: init v_norm_ones: {e:?}"))?;
        }

        Ok(Gemma4State {
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            ar_warmed_up: false,
            x: alloc(gpu, dim, "x")?,
            residual: alloc(gpu, dim, "residual")?,
            tmp: alloc(gpu, dim, "tmp")?,
            tmp_rot: alloc(gpu, dim, "tmp_rot")?,
            q: alloc(gpu, max_q, "q")?,
            k: alloc(gpu, max_kv, "k")?,
            v: alloc(gpu, max_kv, "v")?,
            attn_out: alloc(gpu, max_q, "attn_out")?,
            v_norm_ones,
            gate_ffn: alloc(gpu, cfg.hidden_dim, "gate_ffn")?,
            up_ffn: alloc(gpu, cfg.hidden_dim, "up_ffn")?,
            ffn_hidden: alloc(gpu, cfg.hidden_dim, "ffn_hidden")?,
            ffn_out: alloc(gpu, dim, "ffn_out")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
    }

    /// Return all GPU state buffers (both KV caches, the device position
    /// scalar, and the per-decode scratch tensors) to the pool. Consumes
    /// self. Caller follows with `gpu.drain_pool()` (the daemon's
    /// `unload_model` already does).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.kv_sliding.free_gpu(gpu);
        self.kv_full.free_gpu(gpu);
        let _ = gpu.hip.free(self.pos_buf);
        for t in [
            self.x,
            self.residual,
            self.tmp,
            self.tmp_rot,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.v_norm_ones,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
