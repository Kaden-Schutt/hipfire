// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 EAGLE / MTP drafter (`google/gemma-4-12B-it-assistant`,
//! `model_type = gemma4_unified_assistant`, arch_id = 22).
//!
//! A single-block speculative-decode draft head for the arch-13 Gemma 4 target.
//! It is NOT a standalone LM: every draft step reads the TARGET's per-step
//! embedding table + the TARGET's shared KV (last sliding-layer slot + last
//! full-layer slot) and a TARGET-provided hidden half. Architecture
//! (code-traced against transformers-main
//! `models/gemma4_assistant/modeling_gemma4_assistant.py` +
//! `generation/candidate_generator.py::SinglePositionMultiTokenCandidateGenerator`):
//!
//!   inputs_embeds = concat[ target_embed(prev_token)·√backbone_hidden  ‖  hidden_backbone ]   # 2·3840 = 7680
//!   x = pre_projection · inputs_embeds                                                          # 7680 → 1024, no bias
//!   for each drafter layer (4: [sliding,sliding,sliding,full]):
//!     # standard Gemma-4 sandwich-norm block, but the layer is a KV-SHARED
//!     # layer (num_kv_shared_layers = num_hidden_layers): it has q_proj /
//!     # q_norm / o_proj ONLY — no k/v projections. K,V come VERBATIM from the
//!     # target's shared cache (already k_norm'd + RoPE'd + v_norm'd).
//!     residual = x
//!     n1 = input_layernorm(x); q = q_proj(n1); per-head q_norm; q·√head_dim
//!     RoPE the drafter Q only (sliding: θ=1e4 full rotate-half over hd=256;
//!       full: θ=1e6 proportional-0.25 over hd=512) at the CONSTANT query_pos
//!     attend Q against the target's shared K/V (sliding layers → last sliding
//!       slot, SWA window 1024; full layer → last full slot, full ctx; MQA);
//!       scale 1.0 (q pre-scaled → kernel 1/√hd cancels)
//!     attn = o_proj; post_attention_layernorm; x = residual + attn
//!     residual = x; n2 = pre_feedforward_layernorm(x)
//!     ffn = down_proj( gelu_tanh(gate_proj(n2)) * up_proj(n2) ); post_ffn norm
//!     x = residual + ffn; x ·= layer_scalar
//!   n = model.norm(x)                                                                          # 1024
//!   logits = n · embed_tokens.T   (tied lm_head, full vocab, NO final softcap) → argmax
//!   post_projection_out = post_projection · n                                                  # 1024 → 3840
//!
//! No drafter KV cache, no cache writes: the whole step is a single-token
//! forward at a FIXED position. `use_ordered_embeddings`/centroid masking is
//! disabled (the shipped head ties lm_head directly).
//!
//! REUSE: this introduces NO new kernels. The pre/post projections and lm_head
//! are plain `weight_gemv`; attention is the existing `attention_q8_0_kv_swa`
//! (sliding) / `attention_q8_0_kv` (full) called WITHOUT the kv-write step
//! (drafter does not own the cache); norms/rope/gelu/rmsnorm are the same
//! helpers `crate::forward` uses for the target. The target's `decode_step`,
//! `forward_batch`, weights and state are READ-ONLY here and byte-unchanged.

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_runtime::hfq::{HfqFile, load_awq_scale};
use hipfire_runtime::llama::{f16_to_f32, weight_gemv, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

pub const DRAFTER_ARCH_ID: u32 = 22;

// ─── Config ─────────────────────────────────────────────────────────────

/// Typed Gemma 4 drafter (`gemma4_unified_assistant`) shape constants. Parsed
/// from the same `metadata.config` envelope as the target: `text_config.*` for
/// the per-block dims plus top-level `backbone_hidden_size` / `num_centroids` /
/// `centroid_intermediate_top_k`.
#[derive(Debug, Clone)]
pub struct Gemma4DrafterConfig {
    pub hidden: usize,    // text_config.hidden_size  (1024 on the 12B drafter)
    pub n_layers: usize,  // text_config.num_hidden_layers (4)
    pub vocab_size: usize,
    pub norm_eps: f32,

    pub n_heads: usize,            // num_attention_heads (16)
    pub sliding_head_dim: usize,   // head_dim (256)
    pub sliding_n_kv_heads: usize, // num_key_value_heads (8)
    pub sliding_rope_theta: f32,   // 1e4
    pub sliding_window: usize,     // 1024

    pub full_head_dim: usize,     // global_head_dim (512)
    pub full_n_kv_heads: usize,   // num_global_key_value_heads (1, MQA)
    pub full_rope_theta: f32,     // 1e6
    pub full_rope_type: RopeType, // Proportional
    pub full_partial_rotary_factor: f32, // 0.25

    pub hidden_dim: usize, // intermediate_size (8192)

    /// Target hidden width: the hidden half + the projection in/out widths.
    pub backbone_hidden: usize, // backbone_hidden_size (3840)
    pub final_logit_softcapping: f32, // null on the drafter → 0.0 (NO softcap)

    /// Clustered-head params (stored, UNUSED — head is plain tied lm_head).
    pub num_centroids: usize,
    pub centroid_top_k: usize,
    pub use_ordered_embeddings: bool,

    pub layer_types: Vec<LayerType>,
    pub norm_plus_one: bool,
}

impl Gemma4DrafterConfig {
    /// Parse from the HFQ metadata envelope (arch_id 22). Mirrors
    /// `Gemma4Config::from_hfq`; the drafter config nests the per-block dims
    /// under `text_config` and keeps `backbone_hidden_size` / `num_centroids` /
    /// `centroid_intermediate_top_k` / `use_ordered_embeddings` at top level.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("gemma4-drafter: metadata_json not valid JSON: {e}"))?;
        let config = meta
            .get("config")
            .ok_or_else(|| "gemma4-drafter: metadata_json missing `config` wrapper".to_string())?;
        let tc = config.get("text_config").unwrap_or(config);

        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let getb = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_bool());

        let hidden = getu(tc, "hidden_size").ok_or("gemma4-drafter: missing hidden_size")? as usize;
        let n_layers = getu(tc, "num_hidden_layers")
            .ok_or("gemma4-drafter: missing num_hidden_layers")? as usize;
        let vocab_size =
            getu(tc, "vocab_size").ok_or("gemma4-drafter: missing vocab_size")? as usize;
        let norm_eps = getf(tc, "rms_norm_eps").unwrap_or(1e-6) as f32;

        let n_heads = getu(tc, "num_attention_heads")
            .ok_or("gemma4-drafter: missing num_attention_heads")? as usize;

        let sliding_head_dim = getu(tc, "head_dim")
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let sliding_n_kv_heads =
            getu(tc, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
        let sliding_window = getu(tc, "sliding_window").unwrap_or(1024) as usize;

        let full_head_dim = getu(tc, "global_head_dim")
            .map(|v| v as usize)
            .unwrap_or(sliding_head_dim);
        let full_n_kv_heads =
            getu(tc, "num_global_key_value_heads").unwrap_or(sliding_n_kv_heads as u64) as usize;

        let rope_params = tc.get("rope_parameters");
        let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
        let full_rope = rope_params.and_then(|r| r.get("full_attention"));
        let sliding_rope_theta = sliding_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(10_000.0) as f32;
        let full_rope_theta = full_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(1_000_000.0) as f32;
        let full_rope_type = match full_rope
            .and_then(|r| r.get("rope_type"))
            .and_then(|v| v.as_str())
        {
            Some("proportional") => RopeType::Proportional,
            _ => RopeType::Default,
        };
        let full_partial_rotary_factor = full_rope
            .and_then(|r| getf(r, "partial_rotary_factor"))
            .unwrap_or(1.0) as f32;

        let hidden_dim = getu(tc, "intermediate_size")
            .ok_or("gemma4-drafter: missing intermediate_size")? as usize;

        // text_config softcap is null on the drafter; default 0.0 = NO softcap.
        let final_logit_softcapping = getf(tc, "final_logit_softcapping").unwrap_or(0.0) as f32;

        // Top-level drafter-specific fields.
        let backbone_hidden = getu(config, "backbone_hidden_size")
            .map(|v| v as usize)
            .ok_or("gemma4-drafter: missing backbone_hidden_size")?;
        let num_centroids = getu(config, "num_centroids").unwrap_or(2048) as usize;
        let centroid_top_k = getu(config, "centroid_intermediate_top_k").unwrap_or(32) as usize;
        let use_ordered_embeddings = getb(config, "use_ordered_embeddings").unwrap_or(false);

        let layer_types: Vec<LayerType> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                        "full_attention" => LayerType::Full,
                        _ => LayerType::Sliding,
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                // Drafter default: 3 sliding + 1 full.
                let mut v = vec![LayerType::Sliding; n_layers];
                if let Some(last) = v.last_mut() {
                    *last = LayerType::Full;
                }
                v
            });

        let norm_plus_one =
            std::env::var("HIPFIRE_GEMMA4_NORM_PLUS_ONE").ok().as_deref() == Some("1");

        Ok(Gemma4DrafterConfig {
            hidden,
            n_layers,
            vocab_size,
            norm_eps,
            n_heads,
            sliding_head_dim,
            sliding_n_kv_heads,
            sliding_rope_theta,
            sliding_window,
            full_head_dim,
            full_n_kv_heads,
            full_rope_theta,
            full_rope_type,
            full_partial_rotary_factor,
            hidden_dim,
            backbone_hidden,
            final_logit_softcapping,
            num_centroids,
            centroid_top_k,
            use_ordered_embeddings,
            layer_types,
            norm_plus_one,
        })
    }

    /// Concatenated pre-projection input width (2 · backbone_hidden = 7680).
    pub fn pre_proj_in(&self) -> usize {
        2 * self.backbone_hidden
    }

    /// Max q projection width across layer types (scratch sizing).
    pub fn max_q_dim(&self) -> usize {
        (self.n_heads * self.sliding_head_dim).max(self.n_heads * self.full_head_dim)
    }

    /// Max head_dim across the two attention flavours (scratch sizing).
    pub fn max_head_dim(&self) -> usize {
        self.sliding_head_dim.max(self.full_head_dim)
    }
}

// ─── HFQ load helpers (drafter-scoped; flat `model.*` names) ──────────────

fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4-drafter: tensor not found: {name}"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if expected_n != 0 && n != expected_n {
        return Err(format!(
            "gemma4-drafter: shape mismatch for {name}: expected {expected_n}, got {n}"
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
        16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        qt => return Err(format!("gemma4-drafter: expected F16/F32 for {name}, got qt={qt}")),
    };
    Ok(f32_data)
}

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
        .map_err(|e| format!("gemma4-drafter: upload norm {name}: {e:?}"))
}

fn load_layer_scalar(hfq: &HfqFile, name: &str) -> f32 {
    match load_f32_vec(hfq, name, 1) {
        Ok(v) => v[0],
        Err(_) => 1.0,
    }
}

/// Projection weight loader — mirrors `gemma4::load_wt` (F16/F32/Q8/MQ* paths).
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4-drafter: tensor not found: {name}"))?;
    if info.quant_type == 1 || info.quant_type == 16 {
        if info.quant_type == 16 && rdna_compute::calib_force_bf16() {
            // Native BF16 teacher — keep raw BF16 for consistency (drafter decode is GEMV, not batched MFMA).
            let buf = gpu
                .upload_raw(data, &[m, k])
                .map_err(|e| format!("gemma4-drafter: upload BF16 {name}: {e:?}"))?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::BF16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            });
        }
        let f32_data: Vec<f32> = if info.quant_type == 16 {
            data.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        } else {
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu
            .upload_raw(bytes, &[m, k])
            .map_err(|e| format!("gemma4-drafter: upload F32 {name}: {e:?}"))?;
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
                .map_err(|e| format!("gemma4-drafter: upload F32 {name}: {e:?}"))?;
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
        19 => DType::MQ4G256,
        qt => return Err(format!("gemma4-drafter: unsupported quant_type {qt} for {name}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("gemma4-drafter: upload {name}: {e:?}"))?;
    let awq_scale = if dtype.supports_awq_sidecar() {
        load_awq_scale(hfq, gpu, name, k)
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

// ─── Weights ──────────────────────────────────────────────────────────────

/// One drafter layer. KV-shared (num_kv_shared_layers == n_layers) → no
/// k/v projections, no k/v norms; K/V supplied by the target cache.
pub struct DrafterLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar_host: f32,

    pub q_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim]

    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub struct Gemma4DrafterWeights {
    /// Own token embedding [vocab, hidden=1024]; aliased as the drafter lm_head.
    pub lm_head: WeightTensor, // tied to embed_tokens, dim = hidden
    pub embed_tokens: GpuTensor,
    pub final_norm: GpuTensor, // model.norm [hidden]

    pub pre_projection: WeightTensor,  // [hidden, 2·backbone_hidden]
    pub post_projection: WeightTensor, // [backbone_hidden, hidden]

    pub layers: Vec<DrafterLayerWeights>,
    embd_q8: bool,
}

impl Gemma4DrafterWeights {
    /// Load the 48 drafter tensors (FLAT `model.*` names + two top-level
    /// projections). embd format follows the quant_type of `model.embed_tokens`.
    pub fn load(
        hfq: &HfqFile,
        cfg: &Gemma4DrafterConfig,
        gpu: &mut Gpu,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden;
        let plus_one = cfg.norm_plus_one;

        // Embedding (tied lm_head).
        let embed_name = "model.embed_tokens.weight";
        let (embed_info, embed_data) = hfq
            .tensor_data(embed_name)
            .ok_or_else(|| "gemma4-drafter: embed_tokens not found".to_string())?;
        let (embed_tokens, embd_dtype, embd_q8) = match embed_info.quant_type {
            3 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("gemma4-drafter: upload embed: {e:?}"))?,
                DType::Q8_0,
                true,
            ),
            1 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, hidden])
                        .map_err(|e| format!("gemma4-drafter: upload embed f32: {e:?}"))?,
                    DType::F32,
                    false,
                )
            }
            16 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, hidden])
                        .map_err(|e| format!("gemma4-drafter: upload embed f32: {e:?}"))?,
                    DType::F32,
                    false,
                )
            }
            qt => return Err(format!("gemma4-drafter: unsupported embed quant_type {qt}")),
        };
        let lm_head = {
            let alias_buf = unsafe { embed_tokens.buf.alias() };
            let alias_tensor = GpuTensor {
                buf: alias_buf,
                shape: embed_tokens.shape.clone(),
                dtype: embd_dtype,
            };
            WeightTensor {
                buf: alias_tensor,
                gpu_dtype: embd_dtype,
                m: cfg.vocab_size,
                k: hidden,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        };

        let final_norm = load_norm(hfq, gpu, "model.norm.weight", hidden, plus_one)?;

        // Two top-level projections (no bias).
        let pre_projection = load_wt(
            hfq,
            gpu,
            "pre_projection.weight",
            hidden,
            cfg.pre_proj_in(),
        )?;
        let post_projection =
            load_wt(hfq, gpu, "post_projection.weight", cfg.backbone_hidden, hidden)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            let hd = match cfg.layer_types[i] {
                LayerType::Sliding => cfg.sliding_head_dim,
                LayerType::Full => cfg.full_head_dim,
            };
            let q_dim = cfg.n_heads * hd;
            layers.push(DrafterLayerWeights {
                input_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                post_attention_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                pre_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                post_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                layer_scalar_host: load_layer_scalar(hfq, &format!("{p}.layer_scalar")),
                q_proj: load_wt(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, hidden)?,
                o_proj: load_wt(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), hidden, q_dim)?,
                q_norm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    hd,
                    plus_one,
                )?,
                gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    cfg.hidden_dim,
                    hidden,
                )?,
                up_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    cfg.hidden_dim,
                    hidden,
                )?,
                down_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    cfg.hidden_dim,
                )?,
            });
        }

        Ok(Gemma4DrafterWeights {
            lm_head,
            embed_tokens,
            final_norm,
            pre_projection,
            post_projection,
            layers,
            embd_q8,
        })
    }

    /// Return all GPU weight buffers to the pool (drained by the daemon's
    /// `unload_model`). Consumes self.
    ///
    /// `lm_head` is NOT freed here: it is always an alias of `embed_tokens`'
    /// allocation (tied embedding — see the alias construction in `load`).
    /// `embed_tokens` owns the bytes and is freed exactly once below; dropping
    /// the alias is a no-op (`DeviceBuffer` has no `Drop`). Same rule as
    /// `Gemma4Weights::free_gpu` on the target.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        self.pre_projection.free_all(gpu);
        self.post_projection.free_all(gpu);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            let _ = gpu.free_tensor(l.pre_feedforward_layernorm);
            let _ = gpu.free_tensor(l.post_feedforward_layernorm);
            l.q_proj.free_all(gpu);
            l.o_proj.free_all(gpu);
            let _ = gpu.free_tensor(l.q_norm);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

// ─── Scratch ───────────────────────────────────────────────────────────────

/// Per-draft-step GPU scratch. No KV cache (the drafter reads the target's).
pub struct Gemma4DrafterScratch {
    pub concat: GpuTensor,     // [2·backbone_hidden] = 7680 pre-proj input
    pub embed_half: GpuTensor, // [backbone_hidden] = 3840 target-embed row scratch
    pub x: GpuTensor,          // [hidden] = 1024 residual stream
    pub residual: GpuTensor,   // [hidden]
    pub tmp: GpuTensor,        // [hidden] norm / o_proj scratch
    pub q: GpuTensor,          // [max_q_dim]
    pub attn_out: GpuTensor,   // [max_q_dim]
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [hidden]
    pub normed: GpuTensor,     // [hidden] final-norm output (drives lm_head + post_proj)
    pub post_proj: GpuTensor,  // [backbone_hidden] post_projection output
    pub logits: GpuTensor,     // [vocab]
    /// device i32 query position scalar (constant across a round).
    pub pos_buf: hip_bridge::DeviceBuffer,
}

impl Gemma4DrafterScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Gemma4DrafterConfig) -> Result<Self, String> {
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("gemma4-drafter: alloc {label}: {e:?}"))
        };
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("gemma4-drafter: pos_buf malloc: {e:?}"))?;
        Ok(Gemma4DrafterScratch {
            concat: alloc(gpu, cfg.pre_proj_in(), "concat")?,
            embed_half: alloc(gpu, cfg.backbone_hidden, "embed_half")?,
            x: alloc(gpu, cfg.hidden, "x")?,
            residual: alloc(gpu, cfg.hidden, "residual")?,
            tmp: alloc(gpu, cfg.hidden, "tmp")?,
            q: alloc(gpu, cfg.max_q_dim(), "q")?,
            attn_out: alloc(gpu, cfg.max_q_dim(), "attn_out")?,
            gate_ffn: alloc(gpu, cfg.hidden_dim, "gate_ffn")?,
            up_ffn: alloc(gpu, cfg.hidden_dim, "up_ffn")?,
            ffn_hidden: alloc(gpu, cfg.hidden_dim, "ffn_hidden")?,
            ffn_out: alloc(gpu, cfg.hidden, "ffn_out")?,
            normed: alloc(gpu, cfg.hidden, "normed")?,
            post_proj: alloc(gpu, cfg.backbone_hidden, "post_proj")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            pos_buf,
        })
    }

    /// Return all scratch buffers to the pool. Consumes self. The raw
    /// `pos_buf` DeviceBuffer is freed directly (it never went through the
    /// tensor pool).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.concat);
        let _ = gpu.free_tensor(self.embed_half);
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up_ffn);
        let _ = gpu.free_tensor(self.ffn_hidden);
        let _ = gpu.free_tensor(self.ffn_out);
        let _ = gpu.free_tensor(self.normed);
        let _ = gpu.free_tensor(self.post_proj);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── One draft step ─────────────────────────────────────────────────────────

/// Result of one drafter step: the argmax draft token + its post-projected
/// hidden half (3840), which becomes the `hidden_3840` half of the NEXT step's
/// concat. Also exposes the final normed hidden + full logits for validation.
pub struct DrafterStepOut {
    pub argmax: u32,
    /// post_projection(normed) — [backbone_hidden]; the next step's hidden half.
    pub post_proj_hidden: Vec<f32>,
    /// final-norm output [hidden] (drives lm_head + post_proj) — for oracle.
    pub normed_hidden: Vec<f32>,
    /// full logits [vocab] — for oracle cosine.
    pub logits: Vec<f32>,
}

/// One drafter step (single-token forward at a FIXED `query_pos`).
///
/// * `dw` / `dcfg` / `ds` — drafter weights / config / scratch.
/// * `target_weights` — for the target embed table (3840-dim) + √backbone scale.
/// * `target_state`   — for the target's last sliding / last full KV slot.
/// * `prev_token`     — token whose target-embed forms the concat's [0:3840] half.
/// * `hidden_backbone`— the [3840] hidden half (target's NORMED final hidden on
///                      step 0, else the previous step's `post_proj_hidden`).
/// * `query_pos`      — CONSTANT across the round = committed seq_len − 1.
///
/// Reads `target_weights` + `target_state` ONLY; writes nothing back.
#[allow(clippy::too_many_arguments)]
pub fn drafter_step(
    gpu: &mut Gpu,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    target_weights: &Gemma4Weights,
    target_state: &Gemma4State,
    target_cfg: &Gemma4Config,
    prev_token: u32,
    hidden_backbone: &GpuTensor,
    query_pos: usize,
) -> Result<DrafterStepOut, String> {
    let bb = dcfg.backbone_hidden;

    // Set the device position scalar = query_pos (drives RoPE + attention range).
    {
        let pos_host = [query_pos as i32];
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&ds.pos_buf, pos_bytes)
            .map_err(|e| format!("gemma4-drafter: htod pos: {e:?}"))?;
    }

    // ── (a) concat = [ target_embed(prev)·√bb  ‖  hidden_backbone ] (7680) ──
    // target_embed half: look up the TARGET embed row (bb-dim), scale by √bb.
    // (The target's ScaledWordEmbedding bakes ·√bb into its forward, which the
    // candidate generator invokes; we replicate that scale here.)
    use hipfire_runtime::llama::EmbeddingFormat;
    match target_weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&target_weights.embed_tokens, &ds.embed_half, prev_token, bb)
            .map_err(|e| format!("gemma4-drafter: target embed hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&target_weights.embed_tokens, &ds.embed_half, prev_token, bb)
            .map_err(|e| format!("gemma4-drafter: target embed hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&target_weights.embed_tokens, &ds.embed_half, prev_token, bb)
            .map_err(|e| format!("gemma4-drafter: target embed q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&target_weights.embed_tokens, &ds.embed_half, prev_token, bb)
            .map_err(|e| format!("gemma4-drafter: target embed f32: {e:?}"))?,
        EmbeddingFormat::Q4K => {
            return Err("gemma4-drafter: Q4K target embed unsupported".to_string())
        }
    }
    let _ = target_cfg; // embed scale uses bb (= target hidden), not target_cfg directly
    gpu.scale_f32(&ds.embed_half, (bb as f32).sqrt())
        .map_err(|e| format!("gemma4-drafter: embed scale: {e:?}"))?;
    // concat[0:bb] = embed_half;  concat[bb:2bb] = hidden_backbone (no scale).
    gpu.memcpy_dtod_at_auto(&ds.concat.buf, 0, &ds.embed_half.buf, 0, bb * 4)
        .map_err(|e| format!("gemma4-drafter: concat embed half: {e:?}"))?;
    gpu.memcpy_dtod_at_auto(&ds.concat.buf, bb * 4, &hidden_backbone.buf, 0, bb * 4)
        .map_err(|e| format!("gemma4-drafter: concat hidden half: {e:?}"))?;

    // ── (b)-(f) pre_projection → layers → norm → lm_head/post_proj ──
    // pos_buf is already staged above; concat is built into ds.concat.
    drafter_step_from_concat_inner(gpu, dw, dcfg, ds, target_state, target_cfg)
}

/// Run (b)-(f) of a draft step from a PRE-BUILT concat in `ds.concat` and a
/// PRE-STAGED `ds.pos_buf`. Used by `drafter_step` (after it builds the concat)
/// and by the validation harness (which injects an HF-exact concat to isolate
/// the backbone + heads from the embed/feedback path).
#[allow(clippy::too_many_arguments)]
pub fn drafter_step_from_concat(
    gpu: &mut Gpu,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    target_state: &Gemma4State,
    target_cfg: &Gemma4Config,
    concat: &[f32],
    query_pos: usize,
) -> Result<DrafterStepOut, String> {
    {
        let pos_host = [query_pos as i32];
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&ds.pos_buf, pos_bytes)
            .map_err(|e| format!("gemma4-drafter: htod pos: {e:?}"))?;
    }
    if concat.len() != dcfg.pre_proj_in() {
        return Err(format!(
            "gemma4-drafter: concat len {} != {}",
            concat.len(),
            dcfg.pre_proj_in()
        ));
    }
    gpu.hip
        .memcpy_htod(&ds.concat.buf, unsafe {
            std::slice::from_raw_parts(concat.as_ptr() as *const u8, concat.len() * 4)
        })
        .map_err(|e| format!("gemma4-drafter: upload concat: {e:?}"))?;
    drafter_step_from_concat_inner(gpu, dw, dcfg, ds, target_state, target_cfg)
}

fn drafter_step_from_concat_inner(
    gpu: &mut Gpu,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    target_state: &Gemma4State,
    target_cfg: &Gemma4Config,
) -> Result<DrafterStepOut, String> {
    let eps = dcfg.norm_eps;

    // ── (b) x = pre_projection · concat  (7680 → 1024, no bias) ──
    weight_gemv(gpu, &dw.pre_projection, &ds.concat, &ds.x)
        .map_err(|e| format!("gemma4-drafter: pre_projection: {e}"))?;

    // ── (c) drafter layers ──
    for layer_idx in 0..dcfg.n_layers {
        drafter_layer(
            gpu,
            dcfg,
            target_cfg,
            &dw.layers[layer_idx],
            target_state,
            ds,
            dcfg.layer_types[layer_idx],
        )?;
    }

    // ── (d) n = model.norm(x)  (1024) ──
    gpu.rmsnorm_f32(&ds.x, &dw.final_norm, &ds.normed, eps)
        .map_err(|e| format!("gemma4-drafter: final rmsnorm: {e:?}"))?;

    // ── (e) logits = n · embed_tokens.T  (tied, full vocab, NO softcap) → argmax ──
    weight_gemv(gpu, &dw.lm_head, &ds.normed, &ds.logits)
        .map_err(|e| format!("gemma4-drafter: lm_head: {e}"))?;
    // (final_logit_softcapping is null on the drafter — do NOT apply.)
    let logits = gpu
        .download_f32(&ds.logits)
        .map_err(|e| format!("gemma4-drafter: download logits: {e:?}"))?;
    let argmax = argmax_f32(&logits);

    // ── (f) post_proj = post_projection · n  (1024 → 3840) ──
    weight_gemv(gpu, &dw.post_projection, &ds.normed, &ds.post_proj)
        .map_err(|e| format!("gemma4-drafter: post_projection: {e}"))?;
    let post_proj_hidden = gpu
        .download_f32(&ds.post_proj)
        .map_err(|e| format!("gemma4-drafter: download post_proj: {e:?}"))?;
    let normed_hidden = gpu
        .download_f32(&ds.normed)
        .map_err(|e| format!("gemma4-drafter: download normed: {e:?}"))?;

    let _ = dw.embd_q8;
    Ok(DrafterStepOut {
        argmax,
        post_proj_hidden,
        normed_hidden,
        logits,
    })
}

/// One drafter layer: sandwich-norm block over `ds.x`, KV-shared attention
/// against the target's last-of-type slot. Mirrors the target's
/// sliding/full `*_layer_decode` attention half + the shared FFN tail, but
/// with NO k/v projection and NO cache write.
#[allow(clippy::too_many_arguments)]
fn drafter_layer(
    gpu: &mut Gpu,
    dcfg: &Gemma4DrafterConfig,
    target_cfg: &Gemma4Config,
    lw: &DrafterLayerWeights,
    target_state: &Gemma4State,
    ds: &mut Gemma4DrafterScratch,
    lt: LayerType,
) -> Result<(), String> {
    let hidden = dcfg.hidden;
    let eps = dcfg.norm_eps;
    let hidden_bytes = hidden * 4;
    let n_heads = dcfg.n_heads;

    let (head_dim, n_kv, window, rope) = match lt {
        LayerType::Sliding => (
            dcfg.sliding_head_dim,
            dcfg.sliding_n_kv_heads,
            dcfg.sliding_window,
            RopeFlavour::Full(dcfg.sliding_rope_theta),
        ),
        LayerType::Full => {
            let n_rot_pairs = match dcfg.full_rope_type {
                RopeType::Proportional => {
                    ((dcfg.full_head_dim as f32) * dcfg.full_partial_rotary_factor * 0.5) as usize
                }
                RopeType::Default => dcfg.full_head_dim / 2,
            };
            (
                dcfg.full_head_dim,
                dcfg.full_n_kv_heads,
                0usize,
                RopeFlavour::PartialHalved(dcfg.full_rope_theta, n_rot_pairs),
            )
        }
    };

    // The shared KV slot in the TARGET state: the LAST slot of the matching
    // per-type cache (the target's last sliding / last full layer).
    let (kv_cache, kv_slot) = match lt {
        LayerType::Sliding => {
            let slot = target_cfg.n_sliding_layers().checked_sub(1).ok_or(
                "gemma4-drafter: target has no sliding layers to share KV from",
            )?;
            (&target_state.kv_sliding, slot)
        }
        LayerType::Full => {
            let slot = target_cfg
                .n_full_layers()
                .checked_sub(1)
                .ok_or("gemma4-drafter: target has no full layers to share KV from")?;
            (&target_state.kv_full, slot)
        }
    };

    // residual = x
    gpu.memcpy_dtod_auto(&ds.residual.buf, &ds.x.buf, hidden_bytes)
        .map_err(|e| format!("gemma4-drafter: save residual: {e:?}"))?;

    // n1 = input_layernorm(x) → tmp
    gpu.rmsnorm_f32(&ds.x, &lw.input_layernorm, &ds.tmp, eps)
        .map_err(|e| format!("gemma4-drafter: input rmsnorm: {e:?}"))?;

    // q = q_proj(n1)
    weight_gemv(gpu, &lw.q_proj, &ds.tmp, &ds.q)
        .map_err(|e| format!("gemma4-drafter: q_proj: {e}"))?;
    // per-head q_norm over head_dim
    gpu.rmsnorm_batched(&ds.q, &lw.q_norm, &ds.q, n_heads, head_dim, eps)
        .map_err(|e| format!("gemma4-drafter: q_norm: {e:?}"))?;
    // q ·= √head_dim (cancels the kernel's 1/√head_dim → effective scale 1.0).
    gpu.scale_f32(&ds.q, (head_dim as f32).sqrt())
        .map_err(|e| format!("gemma4-drafter: q scale: {e:?}"))?;

    // RoPE the drafter Q ONLY (n_heads_k = 0 ⇒ the kernel skips K untouched).
    match rope {
        RopeFlavour::Full(theta) => {
            gpu.rope_f32(&ds.q, &ds.q, &ds.pos_buf, n_heads, 0, head_dim, theta)
                .map_err(|e| format!("gemma4-drafter: rope: {e:?}"))?;
        }
        RopeFlavour::PartialHalved(theta, n_rot_pairs) => {
            gpu.rope_partial_halved_f32(
                &ds.q,
                &ds.q,
                &ds.pos_buf,
                n_heads,
                0,
                head_dim,
                n_rot_pairs,
                theta,
            )
            .map_err(|e| format!("gemma4-drafter: rope partial: {e:?}"))?;
        }
    }

    // Attention over the target's shared K/V (NO write). pos_buf = query_pos so
    // the kernel reads seq_len = query_pos+1 from pos_buf[0]+1.
    let max_seq = target_state.max_seq;
    let phys_cap = kv_cache.physical_cap;
    let k_cache = &kv_cache.k_gpu[kv_slot];
    let v_cache = &kv_cache.v_gpu[kv_slot];
    if window > 0 {
        gpu.attention_q8_0_kv_swa(
            &ds.q, k_cache, v_cache, &ds.attn_out, &ds.pos_buf, max_seq, n_heads, n_kv,
            head_dim, phys_cap, window,
        )
        .map_err(|e| format!("gemma4-drafter: attention swa: {e:?}"))?;
    } else {
        gpu.attention_q8_0_kv(
            &ds.q, k_cache, v_cache, &ds.attn_out, &ds.pos_buf, max_seq, n_heads, n_kv,
            head_dim, phys_cap,
        )
        .map_err(|e| format!("gemma4-drafter: attention full: {e:?}"))?;
    }

    // o_proj(attn_out) → tmp; post_attention_layernorm; x = residual + tmp.
    weight_gemv(gpu, &lw.o_proj, &ds.attn_out, &ds.tmp)
        .map_err(|e| format!("gemma4-drafter: o_proj: {e}"))?;
    gpu.rmsnorm_f32(&ds.tmp, &lw.post_attention_layernorm, &ds.tmp, eps)
        .map_err(|e| format!("gemma4-drafter: post_attn rmsnorm: {e:?}"))?;
    gpu.memcpy_dtod_auto(&ds.x.buf, &ds.residual.buf, hidden_bytes)
        .map_err(|e| format!("gemma4-drafter: reset x: {e:?}"))?;
    gpu.add_inplace_f32(&ds.x, &ds.tmp)
        .map_err(|e| format!("gemma4-drafter: attn residual add: {e:?}"))?;

    // residual = x; FFN: GeGLU(gelu_tanh) → down_proj; post_ffn norm; x += ffn.
    gpu.memcpy_dtod_auto(&ds.residual.buf, &ds.x.buf, hidden_bytes)
        .map_err(|e| format!("gemma4-drafter: save ffn residual: {e:?}"))?;
    gpu.rmsnorm_f32(&ds.x, &lw.pre_feedforward_layernorm, &ds.tmp, eps)
        .map_err(|e| format!("gemma4-drafter: pre_ffn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &lw.gate_proj, &ds.tmp, &ds.gate_ffn)
        .map_err(|e| format!("gemma4-drafter: gate_proj: {e}"))?;
    weight_gemv(gpu, &lw.up_proj, &ds.tmp, &ds.up_ffn)
        .map_err(|e| format!("gemma4-drafter: up_proj: {e}"))?;
    gpu.gelu_tanh_f32(&ds.gate_ffn, &ds.ffn_hidden, dcfg.hidden_dim)
        .map_err(|e| format!("gemma4-drafter: gelu_tanh: {e:?}"))?;
    gpu.mul_f32(&ds.ffn_hidden, &ds.up_ffn, &ds.ffn_hidden)
        .map_err(|e| format!("gemma4-drafter: geglu mul: {e:?}"))?;
    weight_gemv(gpu, &lw.down_proj, &ds.ffn_hidden, &ds.ffn_out)
        .map_err(|e| format!("gemma4-drafter: down_proj: {e}"))?;
    gpu.rmsnorm_f32(&ds.ffn_out, &lw.post_feedforward_layernorm, &ds.tmp, eps)
        .map_err(|e| format!("gemma4-drafter: post_ffn rmsnorm: {e:?}"))?;
    gpu.memcpy_dtod_auto(&ds.x.buf, &ds.residual.buf, hidden_bytes)
        .map_err(|e| format!("gemma4-drafter: reset x (ffn): {e:?}"))?;
    gpu.add_inplace_f32(&ds.x, &ds.tmp)
        .map_err(|e| format!("gemma4-drafter: ffn residual add: {e:?}"))?;

    // learned per-layer scalar.
    if lw.layer_scalar_host != 1.0 {
        gpu.scale_f32(&ds.x, lw.layer_scalar_host)
            .map_err(|e| format!("gemma4-drafter: layer_scalar: {e:?}"))?;
    }
    Ok(())
}

enum RopeFlavour {
    /// Full rotate-half over the whole head_dim (sliding). θ.
    Full(f32),
    /// Partial proportional rotate-half (full layer). (θ, n_rot_pairs).
    PartialHalved(f32, usize),
}

fn argmax_f32(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}
