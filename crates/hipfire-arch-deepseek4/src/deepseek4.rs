// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Config / Weights / State types for DeepSeek V4 Flash.
//!
//! `DeepseekV4Config` mirrors the fields in the upstream
//! `config.json`. Defaults come from the released
//! `deepseek-ai/DeepSeek-V4-Flash` checkpoint.

use hipfire_runtime::hfq::HfqFile;
use serde::{Deserialize, Serialize};

/// Per-layer compression mode for the indexer / KV path.
///
/// `compress_ratios` in `config.json` is a per-layer array. The
/// observed pattern on the released DeepSeek V4 is `[0, 0, 4, 128, 4, 128,
/// ..., 4, 128, 4, 0]` — i.e. the first two and the last layer use
/// `0` (no compression / full attention), and the middle layers
/// alternate `4` / `128`. We carry the raw `u32` per layer rather
/// than collapsing to an enum so future fine-tunes that pick
/// different ratios still round-trip cleanly.
pub type CompressRatio = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV4Config {
    // ── transformer-shape basics ────────────────────────────────
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,

    // ── DeepSeek-specific attention ─────────────────────────────
    /// Q-projection LoRA rank. Q goes through a `hidden × q_lora_rank`
    /// then `q_lora_rank × (n_heads · head_dim)` factorisation rather
    /// than full `hidden × (n_heads · head_dim)`.
    pub q_lora_rank: usize,
    /// O-projection LoRA rank. Same shape pattern as Q but on the
    /// output side.
    pub o_lora_rank: usize,
    /// Number of tail dimensions per head that carry RoPE. The
    /// remaining `head_dim - qk_rope_head_dim` dims are straight Q·K.
    pub qk_rope_head_dim: usize,
    /// O-projection grouping for the LoRA-bottlenecked output.
    pub o_groups: usize,

    // ── MoE ─────────────────────────────────────────────────────
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    /// `noaux_tc` etc. — only `noaux_tc` is supported initially.
    pub topk_method: String,
    /// `sqrtsoftplus` — the routing-score function.
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub swiglu_limit: f32,

    // ── Hyper-Connections ───────────────────────────────────────
    /// Number of residual streams (typically 4).
    pub hc_mult: usize,
    /// Sinkhorn iteration count for the residual gating matrix.
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // ── Compressed-KV indexer ───────────────────────────────────
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// Per-layer compression ratio array (length =
    /// `num_hidden_layers + num_nextn_predict_layers`). `0` = no
    /// compression, otherwise the indexer stride for that layer.
    pub compress_ratios: Vec<CompressRatio>,
    /// Compressed-KV path uses its own rope_theta.
    pub compress_rope_theta: f32,

    // ── RoPE / sliding window ───────────────────────────────────
    pub rope_theta: f32,
    /// YaRN scaling factor (typically 16 for DeepSeek V4's 1M context).
    pub rope_scaling_factor: f32,
    pub rope_scaling_original_max_position_embeddings: usize,
    pub rope_scaling_beta_fast: usize,
    pub rope_scaling_beta_slow: usize,
    /// SWA window length for the main attention path (DeepSeek V4: 128).
    pub sliding_window: usize,

    // ── Multi-Token-Prediction (MTP) ────────────────────────────
    /// Number of next-token prediction layers appended after the
    /// main stack. DeepSeek V4 ships `1` (one MTP head).
    pub num_nextn_predict_layers: usize,

    // ── hash-routing (DeepSeek V4-only) ─────────────────────────────────
    pub num_hash_layers: usize,
}

/// Raw upstream JSON shape — only the fields we read. Used to drive
/// `from_hfq`. We deliberately mirror the upstream key names with
/// `#[serde(rename)]` rather than renaming on the HFQ side so the
/// metadata is the byte-for-byte same JSON the converter sees.
#[derive(Debug, Deserialize)]
struct RawDeepseekV4Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,

    q_lora_rank: usize,
    o_lora_rank: usize,
    qk_rope_head_dim: usize,
    o_groups: usize,

    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    routed_scaling_factor: f32,
    topk_method: String,
    scoring_func: String,
    norm_topk_prob: bool,
    swiglu_limit: f32,

    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,

    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    compress_ratios: Vec<u32>,
    compress_rope_theta: f32,

    rope_theta: f32,
    rope_scaling: RawYarnScaling,
    sliding_window: usize,

    num_nextn_predict_layers: usize,
    num_hash_layers: usize,
}

#[derive(Debug, Deserialize)]
struct RawYarnScaling {
    factor: f32,
    original_max_position_embeddings: usize,
    beta_fast: usize,
    beta_slow: usize,
    #[serde(rename = "type")]
    _kind: String,
}

impl DeepseekV4Config {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        // The quantizer wraps the DeepSeek V4 config inside an outer
        // `{"architecture":..., "config":{...}, "tokenizer":...,
        // "tokenizer_config":...}` envelope (matches the Qwen3.5
        // pattern; see crates/hipfire-quantize/src/main.rs around
        // line ~3805). Unwrap the inner `config` slice before parsing.
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("deepseek4: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "deepseek4: metadata_json missing `config` wrapper".to_string())?;
        let raw: RawDeepseekV4Config = serde_json::from_value(inner.clone())
            .map_err(|e| format!("deepseek4: parsing inner config failed: {e}"))?;
        Ok(DeepseekV4Config {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            q_lora_rank: raw.q_lora_rank,
            o_lora_rank: raw.o_lora_rank,
            qk_rope_head_dim: raw.qk_rope_head_dim,
            o_groups: raw.o_groups,
            n_routed_experts: raw.n_routed_experts,
            n_shared_experts: raw.n_shared_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            moe_intermediate_size: raw.moe_intermediate_size,
            routed_scaling_factor: raw.routed_scaling_factor,
            topk_method: raw.topk_method,
            scoring_func: raw.scoring_func,
            norm_topk_prob: raw.norm_topk_prob,
            swiglu_limit: raw.swiglu_limit,
            hc_mult: raw.hc_mult,
            hc_sinkhorn_iters: raw.hc_sinkhorn_iters,
            hc_eps: raw.hc_eps,
            index_n_heads: raw.index_n_heads,
            index_head_dim: raw.index_head_dim,
            index_topk: raw.index_topk,
            compress_ratios: raw.compress_ratios,
            compress_rope_theta: raw.compress_rope_theta,
            rope_theta: raw.rope_theta,
            rope_scaling_factor: raw.rope_scaling.factor,
            rope_scaling_original_max_position_embeddings: raw
                .rope_scaling
                .original_max_position_embeddings,
            rope_scaling_beta_fast: raw.rope_scaling.beta_fast,
            rope_scaling_beta_slow: raw.rope_scaling.beta_slow,
            sliding_window: raw.sliding_window,
            num_nextn_predict_layers: raw.num_nextn_predict_layers,
            num_hash_layers: raw.num_hash_layers,
        })
    }
}

/// Per-layer GPU-resident weights. Slots match DeepSeek V4 shipped tensor
/// inventory; each is `Option<GpuTensor>` so partial-upload paths
/// (host walk only / minimal upload / full upload) can populate
/// progressively. Forward bring-up asserts all relevant slots are
/// Some before dispatching.
pub struct DeepseekV4LayerWeights {
    pub compress_ratio: u32, // 0 = no indexer; otherwise stride

    // Norms (F16 vectors).
    pub attn_norm: Option<rdna_compute::GpuTensor>,
    pub ffn_norm: Option<rdna_compute::GpuTensor>,
    pub q_norm: Option<rdna_compute::GpuTensor>,
    pub kv_norm: Option<rdna_compute::GpuTensor>,
    pub attn_sink: Option<rdna_compute::GpuTensor>, // [n_heads]

    // Attention LoRA + KV joint (MQ-family quantized).
    pub wq_a: Option<rdna_compute::GpuTensor>,
    pub wq_b: Option<rdna_compute::GpuTensor>,
    pub wkv: Option<rdna_compute::GpuTensor>,
    pub wo_a: Option<rdna_compute::GpuTensor>,
    pub wo_b: Option<rdna_compute::GpuTensor>,

    // Main-attention compressor (compress_ratio > 0). Stores compressed
    // KV at slot pos//ratio for later main-attention gather. Distinct
    // from indexer's own compressor below.
    pub compressor_wkv: Option<rdna_compute::GpuTensor>,
    pub compressor_wgate: Option<rdna_compute::GpuTensor>,
    pub compressor_norm: Option<rdna_compute::GpuTensor>,
    pub compressor_ape: Option<rdna_compute::GpuTensor>, // [ratio, coff*head_dim]
    /// F16-native copies of the compressor projections for the WMMA
    /// GEMM path. Same data as `compressor_w{kv,gate}` but stored as
    /// F16 bytes directly (no F32 decode). Populated when
    /// `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` is enabled at load time.
    pub compressor_wkv_f16: Option<rdna_compute::GpuTensor>,
    pub compressor_wgate_f16: Option<rdna_compute::GpuTensor>,

    // Indexer sub-module — only on layers with compress_ratio == 4.
    // Selects top-k positions for sparse attention beyond SWA window.
    pub indexer_wq_b: Option<rdna_compute::GpuTensor>, // [idx_n_heads * idx_head_dim, q_lora_rank]
    pub indexer_weights_proj: Option<rdna_compute::GpuTensor>, // [idx_n_heads, hidden]
    pub indexer_compressor_wkv: Option<rdna_compute::GpuTensor>, // [coff*idx_head_dim, hidden]
    pub indexer_compressor_wgate: Option<rdna_compute::GpuTensor>,
    /// F16-native copies of the indexer compressor projections for WMMA.
    pub indexer_compressor_wkv_f16: Option<rdna_compute::GpuTensor>,
    pub indexer_compressor_wgate_f16: Option<rdna_compute::GpuTensor>,
    pub indexer_compressor_norm: Option<rdna_compute::GpuTensor>, // [idx_head_dim]
    pub indexer_compressor_ape: Option<rdna_compute::GpuTensor>,  // [ratio, coff*idx_head_dim]

    // MTP-specific fields (only populated for the MTP "layer" in
    // `DeepseekV4Weights.mtp_layer`; always None for the normal
    // `layers[0..n_hidden]` bundles). These implement the
    // DeepSeek V3-style next-token-prediction head:
    //   x_in   = e_proj(enorm(embed_{n+1})) + h_proj(hnorm(h_n))
    //   x_attn = attention(attn_norm(x_in))   + x_in
    //   x_ffn  = ffn(ffn_norm(x_attn))        + x_attn
    //   h_n+1  = mtp_final_norm(x_ffn)        → shared head → logits_{n+2}
    // `e_proj`/`h_proj` are FP8-source → Q8F16 on device (under the
    // deepseek4-source-precision quant). `enorm`/`hnorm`/`mtp_final_norm`
    // are BF16-source → F16.
    pub mtp_enorm: Option<rdna_compute::GpuTensor>, // [hidden]
    pub mtp_hnorm: Option<rdna_compute::GpuTensor>, // [hidden]
    pub mtp_e_proj: Option<rdna_compute::GpuTensor>, // [hidden, hidden]
    pub mtp_h_proj: Option<rdna_compute::GpuTensor>, // [hidden, hidden]
    pub mtp_final_norm: Option<rdna_compute::GpuTensor>, // [hidden]
    /// MTP-specific head-HC matrices. DeepSeek V4 ships these alongside the per-MTP
    /// HC matrices (hc_attn_*, hc_ffn_*). Their presence in the safetensors
    /// means the MTP layer was trained WITH head-HC mixing on its lm_head
    /// path — taking stream 0 alone (as a naive port of V3's transformer
    /// formulation does) drops the cross-stream mixing that MTP expects
    /// for its prediction, measured as ~50% lower acceptance vs head-HC
    /// mixed. Shapes match main globals: hc_head_fn [hc_mult, hc_mult*hidden],
    /// hc_head_base [hc_mult], hc_head_scale [1] (scalar host-side after load).
    pub mtp_hc_head_fn: Option<rdna_compute::GpuTensor>, // [hc_mult, hc_mult*hidden]
    pub mtp_hc_head_base: Option<rdna_compute::GpuTensor>, // [hc_mult]
    pub mtp_hc_head_scale: f32,                     // scalar (loaded from [1] F16)

    // Hyper-Connections (F16 small matrices).
    pub hc_attn_base: Option<rdna_compute::GpuTensor>,
    pub hc_attn_fn: Option<rdna_compute::GpuTensor>,
    pub hc_attn_scale: Option<rdna_compute::GpuTensor>,
    pub hc_ffn_base: Option<rdna_compute::GpuTensor>,
    pub hc_ffn_fn: Option<rdna_compute::GpuTensor>,
    pub hc_ffn_scale: Option<rdna_compute::GpuTensor>,

    // FFN router. `gate.bias` is None for hash-routed layers (first
    // `num_hash_layers`).
    pub gate_weight: Option<rdna_compute::GpuTensor>,
    pub gate_bias: Option<rdna_compute::GpuTensor>,
    /// Host-cached gate_bias for CPU-side topk-with-bias logic. DeepSeek V4
    /// adds bias to the routing-selection scores (but not the routing
    /// weights), so we need fast CPU access during ffn_routed. Length
    /// = n_routed_experts; empty for hash-routed layers.
    pub gate_bias_host: Vec<f32>,
    /// Hash-routing lookup table `tid2eid[vocab_size, n_activated_
    /// experts]` for layers 0..num_hash_layers. Each token_id maps to
    /// a static set of K expert IDs. Empty for non-hash-routed layers.
    /// Stored as flat row-major Vec<u32> length = vocab_size * k.
    pub tid2eid_host: Vec<u32>,
    /// Device-resident twin of `tid2eid_host` for the GPU hash-router
    /// path (eliminates per-step d2h+h2d of scores/weights). Allocated
    /// at load time alongside `tid2eid_host`. `None` for non-hash
    /// layers or when the HFQ shipped without the table.
    pub tid2eid_dev: Option<rdna_compute::GpuTensor>,

    // Shared expert (one per layer, w1/w2/w3, MQ-family quantized).
    pub shared_w1: Option<rdna_compute::GpuTensor>,
    pub shared_w2: Option<rdna_compute::GpuTensor>,
    pub shared_w3: Option<rdna_compute::GpuTensor>,

    // Routed experts. To avoid 256 × 43 × 3 = 33K separate hipMalloc
    // calls (drives load time to 3+ minutes), all 256 experts for each
    // (layer, projection) are uploaded as ONE contiguous blob. The
    // indexed MoE GEMV kernels consume a device-side pointer table.
    //
    // Layout per blob: `[n_routed_experts × bytes_per_expert]` raw bytes.
    // Pointer table: F32 GpuTensor of length `2 * n_routed_experts`
    //   (two F32 slots per u64 pointer, matching qwen35 convention).
    pub expert_w1_blob: Option<rdna_compute::GpuTensor>,
    pub expert_w2_blob: Option<rdna_compute::GpuTensor>,
    pub expert_w3_blob: Option<rdna_compute::GpuTensor>,
    pub expert_w1_ptrs: Option<rdna_compute::GpuTensor>,
    pub expert_w2_ptrs: Option<rdna_compute::GpuTensor>,
    pub expert_w3_ptrs: Option<rdna_compute::GpuTensor>,
    /// Bytes per expert (uniform across all experts in a layer). Used
    /// for sub_offset math when forward needs a per-expert view (rarely).
    pub expert_w1_stride: usize,
    pub expert_w2_stride: usize,
    pub expert_w3_stride: usize,

    /// Phase 1 perf: fused MoE indexed GEMV dispatch.
    /// `expert_gate_up_blob` is a single contiguous device buffer
    /// `[n_routed_experts × (stride_w1 + stride_w3)]` where each
    /// expert's region holds the gate rows immediately followed by the
    /// up rows. The `_indexed` MQ2-Lloyd MoE kernel reads from this as
    /// a [2*intermediate, hidden] weight and splits the output by row.
    /// `expert_gate_up_ptrs` is the per-expert pointer table.
    pub expert_gate_up_blob: Option<rdna_compute::GpuTensor>,
    pub expert_gate_up_ptrs: Option<rdna_compute::GpuTensor>,
    pub expert_gate_up_stride: usize,
}

impl DeepseekV4LayerWeights {
    pub fn new_empty(compress_ratio: u32) -> Self {
        DeepseekV4LayerWeights {
            compress_ratio,
            attn_norm: None,
            ffn_norm: None,
            q_norm: None,
            kv_norm: None,
            attn_sink: None,
            wq_a: None,
            wq_b: None,
            wkv: None,
            wo_a: None,
            wo_b: None,
            compressor_wkv: None,
            compressor_wgate: None,
            compressor_norm: None,
            compressor_ape: None,
            compressor_wkv_f16: None,
            compressor_wgate_f16: None,
            indexer_wq_b: None,
            indexer_weights_proj: None,
            indexer_compressor_wkv: None,
            indexer_compressor_wgate: None,
            indexer_compressor_wkv_f16: None,
            indexer_compressor_wgate_f16: None,
            indexer_compressor_norm: None,
            indexer_compressor_ape: None,
            mtp_enorm: None,
            mtp_hnorm: None,
            mtp_e_proj: None,
            mtp_h_proj: None,
            mtp_final_norm: None,
            mtp_hc_head_fn: None,
            mtp_hc_head_base: None,
            mtp_hc_head_scale: 0.0,
            hc_attn_base: None,
            hc_attn_fn: None,
            hc_attn_scale: None,
            hc_ffn_base: None,
            hc_ffn_fn: None,
            hc_ffn_scale: None,
            gate_weight: None,
            gate_bias: None,
            gate_bias_host: Vec::new(),
            tid2eid_host: Vec::new(),
            tid2eid_dev: None,
            shared_w1: None,
            shared_w2: None,
            shared_w3: None,
            expert_w1_blob: None,
            expert_w2_blob: None,
            expert_w3_blob: None,
            expert_w1_ptrs: None,
            expert_w2_ptrs: None,
            expert_w3_ptrs: None,
            expert_w1_stride: 0,
            expert_w2_stride: 0,
            expert_w3_stride: 0,
            expert_gate_up_blob: None,
            expert_gate_up_ptrs: None,
            expert_gate_up_stride: 0,
        }
    }

    /// Release every GPU buffer this layer owns back to the pool.
    /// Used by `DeepseekV4Weights::free_gpu` to walk all 43 main layers
    /// plus the optional MTP layer.
    pub fn free_gpu(mut self, gpu: &mut rdna_compute::Gpu) {
        fn free_opt(gpu: &mut rdna_compute::Gpu, t: &mut Option<rdna_compute::GpuTensor>) {
            if let Some(t) = t.take() {
                let _ = gpu.free_tensor(t);
            }
        }
        free_opt(gpu, &mut self.attn_norm);
        free_opt(gpu, &mut self.ffn_norm);
        free_opt(gpu, &mut self.q_norm);
        free_opt(gpu, &mut self.kv_norm);
        free_opt(gpu, &mut self.attn_sink);
        free_opt(gpu, &mut self.wq_a);
        free_opt(gpu, &mut self.wq_b);
        free_opt(gpu, &mut self.wkv);
        free_opt(gpu, &mut self.wo_a);
        free_opt(gpu, &mut self.wo_b);
        free_opt(gpu, &mut self.compressor_wkv);
        free_opt(gpu, &mut self.compressor_wgate);
        free_opt(gpu, &mut self.compressor_norm);
        free_opt(gpu, &mut self.compressor_ape);
        free_opt(gpu, &mut self.compressor_wkv_f16);
        free_opt(gpu, &mut self.compressor_wgate_f16);
        free_opt(gpu, &mut self.indexer_wq_b);
        free_opt(gpu, &mut self.indexer_weights_proj);
        free_opt(gpu, &mut self.indexer_compressor_wkv);
        free_opt(gpu, &mut self.indexer_compressor_wgate);
        free_opt(gpu, &mut self.indexer_compressor_wkv_f16);
        free_opt(gpu, &mut self.indexer_compressor_wgate_f16);
        free_opt(gpu, &mut self.indexer_compressor_norm);
        free_opt(gpu, &mut self.indexer_compressor_ape);
        free_opt(gpu, &mut self.mtp_enorm);
        free_opt(gpu, &mut self.mtp_hnorm);
        free_opt(gpu, &mut self.mtp_e_proj);
        free_opt(gpu, &mut self.mtp_h_proj);
        free_opt(gpu, &mut self.mtp_final_norm);
        free_opt(gpu, &mut self.mtp_hc_head_fn);
        free_opt(gpu, &mut self.mtp_hc_head_base);
        free_opt(gpu, &mut self.hc_attn_base);
        free_opt(gpu, &mut self.hc_attn_fn);
        free_opt(gpu, &mut self.hc_attn_scale);
        free_opt(gpu, &mut self.hc_ffn_base);
        free_opt(gpu, &mut self.hc_ffn_fn);
        free_opt(gpu, &mut self.hc_ffn_scale);
        free_opt(gpu, &mut self.gate_weight);
        free_opt(gpu, &mut self.gate_bias);
        free_opt(gpu, &mut self.tid2eid_dev);
        free_opt(gpu, &mut self.shared_w1);
        free_opt(gpu, &mut self.shared_w2);
        free_opt(gpu, &mut self.shared_w3);
        free_opt(gpu, &mut self.expert_w1_blob);
        free_opt(gpu, &mut self.expert_w2_blob);
        free_opt(gpu, &mut self.expert_w3_blob);
        free_opt(gpu, &mut self.expert_w1_ptrs);
        free_opt(gpu, &mut self.expert_w2_ptrs);
        free_opt(gpu, &mut self.expert_w3_ptrs);
        free_opt(gpu, &mut self.expert_gate_up_blob);
        free_opt(gpu, &mut self.expert_gate_up_ptrs);
    }
}

/// DeepSeek V4 weights — fully populated: global embeddings + norms, per-layer
/// LoRAs (q_lora, o_lora), compressor + indexer projections, attn KV,
/// Hyper-Connections gates, routed-expert blobs, and the optional MTP
/// addon layer (`mtp.0.*`).
///
/// `mtp_layer` is `Some` after Phase 5 lands (when the
/// `mtp.` prefix-skip in `hipfire-quantize` is lifted and MTP
/// tensors are quantized alongside main layers).
pub struct DeepseekV4Weights {
    /// Token embedding table. Stored as raw Q8F16 bytes on GPU
    /// (matches the `embed.weight` quant_type from Phase 1 ingest).
    pub token_embd: Option<rdna_compute::GpuTensor>,
    /// Final output norm (RMSNorm scale, F32 — converted from F16 at load time).
    pub output_norm: Option<rdna_compute::GpuTensor>,
    /// LM head weight (Q8_0 in the canonical build; quant_type follows the
    /// source HFQ — `upload_quant_or_f16` routes by dtype). Shape
    /// `[vocab_size, hidden]`.
    pub head: Option<rdna_compute::GpuTensor>,
    /// Head HC mix: `hc_head_fn` [hc_mult, hc_mult * hidden] F16 raw on GPU.
    pub hc_head_fn: Option<rdna_compute::GpuTensor>,
    /// `hc_head_base` [hc_mult] F16 raw on GPU.
    pub hc_head_base: Option<rdna_compute::GpuTensor>,
    /// `hc_head_scale` is shape [1] F16 on disk — cached as host f32 scalar.
    pub hc_head_scale: f32,
    /// One bundle per `num_hidden_layers` (43 on DeepSeek V4).
    pub layers: Vec<DeepseekV4LayerWeights>,
    /// MTP head — structurally identical to a main layer, plus an
    /// `input_proj` conditioning on the base model's hidden state.
    /// `None` at scaffold stage; populated when Phase 5 ships.
    pub mtp_layer: Option<DeepseekV4LayerWeights>,
    pub _scaffold: (),
}

impl DeepseekV4Weights {
    /// Look up the layer-shaped weight bundle by index. Resolves to
    /// `layers[idx]` for the main `0..num_hidden_layers` range and to
    /// `mtp_layer` for `idx == layers.len()`. Used by the per-layer
    /// helpers so they can run against either a normal layer or the
    /// MTP head with no signature change.
    ///
    /// Panics if `idx == layers.len()` but `mtp_layer.is_none()`, or
    /// if `idx > layers.len()`.
    /// Release every GPU buffer these weights own back to the pool.
    /// Consumes self. Walked by `unload_model` on idle eviction or
    /// explicit unload so VRAM is actually returned to the system (not
    /// just released back into the daemon's pool — `drain_pool` calls
    /// `hipFree` after this).
    pub fn free_gpu(mut self, gpu: &mut rdna_compute::Gpu) {
        fn free_opt(gpu: &mut rdna_compute::Gpu, t: &mut Option<rdna_compute::GpuTensor>) {
            if let Some(t) = t.take() {
                let _ = gpu.free_tensor(t);
            }
        }
        free_opt(gpu, &mut self.token_embd);
        free_opt(gpu, &mut self.output_norm);
        free_opt(gpu, &mut self.head);
        free_opt(gpu, &mut self.hc_head_fn);
        free_opt(gpu, &mut self.hc_head_base);
        for l in self.layers.drain(..) {
            l.free_gpu(gpu);
        }
        if let Some(mtp) = self.mtp_layer.take() {
            mtp.free_gpu(gpu);
        }
    }

    pub fn resolve_layer(&self, idx: usize) -> &DeepseekV4LayerWeights {
        if idx < self.layers.len() {
            &self.layers[idx]
        } else if idx == self.layers.len() {
            self.mtp_layer.as_ref().unwrap_or_else(|| {
                panic!(
                    "DeepseekV4Weights::resolve_layer({idx}) — \
                     idx points at the MTP slot but mtp_layer is None. \
                     Either set HIPFIRE_DEEPSEEK4_LOAD_MTP=1 and use a model \
                     quantized with MTP, or stop calling mtp_forward."
                )
            })
        } else {
            panic!(
                "DeepseekV4Weights::resolve_layer({idx}) out of range \
                 (have {} main layers + {} MTP)",
                self.layers.len(),
                self.mtp_layer.is_some() as usize,
            );
        }
    }
}

/// Per-layer state for the compressed-KV indexer (Phase 2, Lever 3).
///
/// Active only on layers with `compress_ratios[l] > 0`. Each layer
/// holds:
/// - a sparse compressed-K cache at stride `compress_ratios[l]`
/// - scratch for the current-step top-k position indices
///
/// See `docs/plans/deepseek4-phase2-indexer.md` for the full kernel
/// design and forward sequence.
pub struct IndexerLayerState {
    /// `compress_ratios[layer]` — stride of the compressed cache.
    /// `0` means this layer doesn't use the indexer (full SWA only).
    pub compress_ratio: u32,

    // ── Main-attention compressor state (ratio > 0) ────────────────
    /// Compressed KV cache `[max_compressed_pos, head_dim]` F32. Holds
    /// gated-pooled compressed values at slot pos//ratio. Used by main
    /// attention's gather step to extend SWA window.
    pub main_kv_cache: Option<rdna_compute::GpuTensor>,
    /// Per-position kv state buffer `[coff*ratio, coff*head_dim]` F32.
    /// Holds raw kv values within the current and (for overlap=true)
    /// previous compress window.
    pub main_kv_state: Option<rdna_compute::GpuTensor>,
    /// Per-position score buffer `[coff*ratio, coff*head_dim]` F32 with
    /// hc_*.ape positional bias added. Pooled via softmax to compress kv.
    pub main_score_state: Option<rdna_compute::GpuTensor>,

    // ── Indexer state (ratio == 4 only) ────────────────────────────
    /// Indexer-specific compressed KV cache `[max_compressed_pos, idx_head_dim]`
    /// F32. Built by indexer's separate compressor. Used by Q · K_idx
    /// scoring step.
    pub indexer_kv_cache: Option<rdna_compute::GpuTensor>,
    pub indexer_kv_state: Option<rdna_compute::GpuTensor>,
    pub indexer_score_state: Option<rdna_compute::GpuTensor>,
    /// Per-step indexer scratch:
    ///   q_idx [n_idx_heads, idx_head_dim] = [64, 128]
    ///   weights [n_idx_heads] = [64]
    ///   index_score [n_compressed] (per current step)
    ///   topk_indices [index_topk = 512]
    pub q_idx: Option<rdna_compute::GpuTensor>,
    pub idx_weights: Option<rdna_compute::GpuTensor>,
    pub index_score: Option<rdna_compute::GpuTensor>,
    pub topk_idx_indices: Option<rdna_compute::GpuTensor>,

    // Compressor per-step scratch (re-used main and indexer; sized for
    // the LARGER of the two — main has coff*head_dim = 1024 for ratio=4,
    // indexer has 256). Lazy-alloc by compressor_forward.
    /// Per-step kv = wkv @ x   [proj_dim = coff*head_dim] F32.
    pub comp_kv_buf: Option<rdna_compute::GpuTensor>,
    /// Per-step score = wgate @ x + ape   [proj_dim] F32.
    pub comp_score_buf: Option<rdna_compute::GpuTensor>,
    /// Concat scratch for overlap-pool   [2*ratio, head_dim] F32.
    pub comp_concat_kv: Option<rdna_compute::GpuTensor>,
    pub comp_concat_score: Option<rdna_compute::GpuTensor>,
}

/// Per-layer scratch for the main attention path's gathered K/V rows.
///
/// The main attention attends to `sliding_window + index_topk` total
/// positions per step: a bounded ring of the last 128 raw KV rows
/// (SWA window) plus 512 rows gathered from the indexer's top-k.
pub struct MainAttentionLayerState {
    /// SWA ring K cache `[n_kv_heads, head_dim, sliding_window]` F32.
    /// `None` until `decode_step` allocates on first call.
    pub swa_k: Option<rdna_compute::GpuTensor>,
    /// SWA ring V cache. DeepSeek V4 has tied K=V so this is a copy of swa_k.
    pub swa_v: Option<rdna_compute::GpuTensor>,

    /// Full positional K/V cache for indexer-gathered attention. Layout:
    /// `[max_ctx, n_kv_heads * head_dim]` F32. Written at each decode
    /// step (after tail-RoPE). Used by the modified main attention when
    /// the indexer's top-K points to positions outside the SWA window.
    ///
    /// DeepSeek V4 has tied K=V so we keep one buffer; `full_v_cache` is None
    /// in practice and we re-use `full_k_cache` for both. Field kept for
    /// future-proofing models with untied K/V.
    pub full_k_cache: Option<rdna_compute::GpuTensor>,
    pub full_v_cache: Option<rdna_compute::GpuTensor>,

    /// Gather scratch — concat of SWA-window K/V + indexer-gathered K/V
    /// for the modified attention pass. `[n_kv_heads, head_dim,
    /// sliding_window + index_topk]` F32, lazy-alloc.
    pub gathered_k: Option<rdna_compute::GpuTensor>,
    pub gathered_v: Option<rdna_compute::GpuTensor>,
}

/// DeepSeek V4 per-decode state. Held on the daemon's per-session struct,
/// reused across decode steps. Allocated once via `new_state`.
pub struct DeepseekV4State {
    /// Per-layer (43 + 1 MTP = 44). Layers with `compress_ratio == 0`
    /// skip the indexer.
    pub _indexer: Vec<IndexerLayerState>,
    pub _attention: Vec<MainAttentionLayerState>,

    /// Hyper-Connections residual streams `[hc_mult = 4, hidden = 4096]`.
    /// Stored as F32 to match hipfire's standard residual convention
    /// (llama / qwen35 use f32 residuals + f32 RMSNorm). Quantized
    /// kernels handle the f32 input directly.
    /// `None` until `decode_step` allocates on first call.
    pub residual_streams: Option<rdna_compute::GpuTensor>,

    /// Single-row embedding scratch `[hidden]` for the current decode
    /// step's token lookup. F32 to match residual_streams convention.
    pub embed_scratch: Option<rdna_compute::GpuTensor>,

    /// Per-step scratch `[hidden]` F32 — used for RMSNorm output,
    /// FWHT-rotated input to first GEMV, etc. Reused across layers.
    pub tmp: Option<rdna_compute::GpuTensor>,

    /// Plain RMSNorm'd attention-side input `[hidden]` F32 — no FWHT.
    /// Mirrors `tmp` but skips the rotation step. Consumed by F32 (F16-
    /// source) non-expert GEMVs (`--non-expert-f16` antirez recipe) since
    /// `gemv_f32` expects un-rotated input. Computed once per layer in
    /// `q_lora` alongside `tmp`.
    pub tmp_plain: Option<rdna_compute::GpuTensor>,

    /// MTP pre-block scratch — RMSNorm output of the embed input
    /// `[hidden]` F32. Holds `mtp_enorm(embed_lookup(next_token))`
    /// across the two-GEMV `mtp_e_proj @ ... + mtp_h_proj @ ...` fusion.
    /// Only allocated when `mtp_forward` is called.
    pub mtp_e_norm_scratch: Option<rdna_compute::GpuTensor>,

    /// MTP pre-block scratch — RMSNorm output of the hidden input
    /// `[hc_mult, hidden]` F32. Holds `mtp_hnorm(h_n)` applied per HC row
    /// (rmsnorm_batched). Same lazy-allocation pattern as
    /// `mtp_e_norm_scratch`. Reallocated if hc_mult ever changes.
    pub mtp_h_norm_scratch: Option<rdna_compute::GpuTensor>,

    /// Post-layer-block residual stream `[hc_mult, hidden]` F32 from the
    /// most-recent `decode_step` or `mtp_forward` call. Per antirez/ds4
    /// reference, DeepSeek V4 MTP consumes the FULL HC stream (not just stream 0)
    /// of the previous position as its `h_n` input — capturing only stream
    /// 0 discards 75% of the HC signal and empirically pins K=2 acceptance
    /// at ~50%. Populated by both `final_norm_and_head` (decode path) and
    /// `mtp_forward` step 7 so `speculative_decode_step` can chain K MTP
    /// iterations.
    pub mtp_last_hidden: Option<rdna_compute::GpuTensor>,

    /// Q-LoRA bottleneck `[q_lora_rank = 1024]` F32. Output of
    /// `wq_a @ x`, input to `wq_b`. Reused across layers.
    pub q_lat: Option<rdna_compute::GpuTensor>,

    /// Q-LoRA bottleneck rotated `[q_lora_rank]` F32. FWHT-rotated
    /// view of q_lat, input to the MQ4 GEMV against wq_b.
    pub q_lat_rot: Option<rdna_compute::GpuTensor>,

    /// Full Q `[n_heads * head_dim = 64 * 512 = 32768]` F32. Output
    /// of `wq_b @ q_lat_rot`. Tail-only RoPE applied in place.
    pub q: Option<rdna_compute::GpuTensor>,

    /// Joint KV stream `[n_kv_heads * head_dim = 1 * 512 = 512]` F32.
    /// Output of `wkv @ x`. DeepSeek V4 uses tied K=V via this single vector
    /// (MQA with V tied to K — see project memory for the layout
    /// open question; revisit during numerical-correctness gate).
    /// Tail-only RoPE applied to last `qk_rope_head_dim = 64` dims.
    pub kv: Option<rdna_compute::GpuTensor>,

    /// Position counter for RoPE. Stored as a 1-element F32 GpuTensor
    /// where we write the i32 position bits via memcpy_htod (the
    /// rope_tail kernel reinterprets the bytes as int via cast).
    ///
    /// HIP-graphs note: in the graph-capture path (`HIPFIRE_DEEPSEEK4_GRAPH=1`)
    /// this buffer is a sub_offset slice of `pos_array_device`, written
    /// ONCE per token at decode_step entry from `pos_array_host`. In the
    /// legacy direct-dispatch path it's a standalone [1] F32 buffer that
    /// gets overwritten per layer.
    pub pos_buf: Option<rdna_compute::GpuTensor>,

    /// Separate position buffer for the indexer compressor's tail-RoPE
    /// step. Distinct from `pos_buf` because the compressor uses a
    /// start-of-window position `(pos / ratio) * ratio`, while the main
    /// attention's inverse-rope (called after the compressor) needs the
    /// current `position`. Sharing one buffer would clobber the value
    /// the main-attn inverse rope reads.
    pub comp_pos_buf: Option<rdna_compute::GpuTensor>,

    /// HIP-graphs prerequisite: per-layer pre-computed position array
    /// `[(num_hidden_layers + 1) * 3]` i32 (stored as F32 bits — kernels
    /// reinterpret). Layout per layer: `[qk_pos, main_comp_rope_pos,
    /// indexer_comp_rope_pos]`. Filled ONCE per decode_step from
    /// `pos_array_host`, then sliced for per-layer kernel reads. Lets us
    /// lift the ~130 per-token `memcpy_htod` pos_buf writes out of the
    /// captured region so a single graph replay covers an entire decode.
    pub pos_array_device: Option<rdna_compute::GpuTensor>,

    /// Stable-pointer host source for `pos_array_device`. Heap-allocated
    /// `Box<[i32]>` so the underlying address stays valid across graph
    /// replays — captured memcpy nodes re-read this pointer on each replay
    /// and find the values we wrote for the current position.
    pub pos_array_host: Option<Box<[i32]>>,

    /// First-call gate for the HIP graph capture path. The first
    /// `decode_step_with_graph` call after `HIPFIRE_DEEPSEEK4_GRAPH=1` runs
    /// direct so kernel JIT and lazy scratch allocations (rope buffers,
    /// indexer scratch, FFN scratch, MoE expert pointers, etc.) all
    /// happen OUTSIDE any captured region. The second call captures the
    /// fully warm forward; the third and later calls replay it. Without
    /// this flag, the first capture would hit
    /// `hipMalloc not permitted under stream capture` and fail.
    pub ar_forward_warmed_up: bool,

    /// Single-i32 device buffer holding the current step's `token_id`.
    /// Read by `hash_router_normalize_f32_buf` so the captured graph
    /// re-reads it on every replay (mirrors the `pos_array_*` pattern).
    /// Lazy-allocated by the first hash-routed layer that needs it.
    pub token_id_buf: Option<rdna_compute::GpuTensor>,

    /// Stable host-side source for `token_id_buf`. The captured htod
    /// node re-reads this pointer on every graph_launch — must be a
    /// heap allocation so the address survives across replays.
    pub token_id_host: Option<Box<[i32; 1]>>,

    /// Ten-slot device buffer for SWA + compressor runtime state.
    /// Layout (all i32 stored as F32 bits):
    ///   [0] swa_slot          = pos % sliding_window
    ///   [1] n_valid_swa       = min(pos + 1, sliding_window)
    ///   [2] n_compressed_4    = (pos + 1) / 4    (ratio=4 layers)
    ///   [3] n_compressed_128  = (pos + 1) / 128  (ratio=128 layers)
    ///   [4] k_active_4        = min(index_topk, n_compressed_4)
    ///   [5] k_active_128      = min(topk_window, n_compressed_128)
    ///   [6] ring_slot_4       = ring write slot for ratio=4 state
    ///                            buffer (overlap path: 4 + pos%4)
    ///   [7] commit_slot_4     = pos/4 if (pos+1)%4 == 0 else -1
    ///   [8] ring_slot_128     = pos % 128 (ratio=128 state ring slot)
    ///   [9] commit_slot_128   = pos/128 if (pos+1)%128 == 0 else -1
    ///
    /// All values derived from `state.n_tokens` at decode_step entry.
    /// The `_buf` variants of SWA / topk-gather / topk-attention /
    /// compressor kernels read the relevant slots so captured HIP graphs
    /// pick up new positions on each replay without re-capture. Slots
    /// 7 and 9 store -1 (sentinel) on non-commit positions so the
    /// commit kernels can early-return without writing.
    pub attn_state_buf: Option<rdna_compute::GpuTensor>,
    /// Stable-pointer host source for `attn_state_buf`. Same rationale
    /// as `pos_array_host`: captured memcpy nodes re-read this pointer
    /// on each graph replay and find the values written for the current
    /// position.
    pub attn_state_host: Option<Box<[i32; 10]>>,

    /// Per-token attention output `[hidden]` F32, fed to HC attn mix
    /// as the `transform_out` arg. Currently a stub: holds a sliced
    /// view of `q` until real attention + O-LoRA lands.
    pub attn_out: Option<rdna_compute::GpuTensor>,

    /// Per-token FFN output `[hidden]` F32, fed to HC FFN mix as
    /// `transform_out`. Currently = shared expert output (real),
    /// routed experts pending.
    pub ffn_out: Option<rdna_compute::GpuTensor>,

    /// FFN normalised input `[hidden]` F32. RMSNorm(stream0, ffn_norm)
    /// then FWHT-rotated for the shared-expert MQ4 GEMVs.
    pub ffn_x_rot: Option<rdna_compute::GpuTensor>,

    /// Plain RMSNorm'd FFN-side input `[hidden]` F32 — no FWHT. Mirror of
    /// `ffn_x_rot` for F32 (F16-source) non-expert GEMVs (antirez recipe).
    pub ffn_x_plain: Option<rdna_compute::GpuTensor>,

    /// Shared expert SwiGLU gate scratch `[moe_intermediate=2048]` F32.
    pub ffn_gate: Option<rdna_compute::GpuTensor>,
    /// Shared expert SwiGLU up scratch `[moe_intermediate]` F32.
    pub ffn_up: Option<rdna_compute::GpuTensor>,
    /// FWHT-rotated silu(gate)*up for the down GEMV.
    pub ffn_silu_rot: Option<rdna_compute::GpuTensor>,

    /// Final pre-lm_head normalized residual `[hidden]` F32. Output
    /// of the global RMSNorm against `output_norm`.
    pub final_norm: Option<rdna_compute::GpuTensor>,

    /// LM head output logits `[vocab_size = 129280]` F32. Output of
    /// `head_weight @ final_norm`.
    pub logits: Option<rdna_compute::GpuTensor>,

    /// FWHT-rotated `final_norm` for the MQ4 head GEMV. Shape `[hidden]`.
    pub final_norm_rot: Option<rdna_compute::GpuTensor>,

    /// Input-mapping output: `x_in = A · X`. Fed to the transform (attn
    /// or FFN) as its [hidden] input.
    pub hc_x_in: Option<rdna_compute::GpuTensor>,

    /// mHC control vector `[24]` F32, set by `hc_compute_control` and
    /// consumed by `hc_mix_4stream`. Allocated once per session.
    /// Layout: c[0..4]=Ã, c[4..20]=B̃, c[20..24]=C̃.
    pub hc_c: Option<rdna_compute::GpuTensor>,

    /// MoE router scores `[n_routed_experts = 256]` F32, set by the
    /// router step (gate.weight @ ffn_input + bias → sqrt_softplus).
    pub router_scores: Option<rdna_compute::GpuTensor>,
    /// Top-K expert indices, allocated as F32 view but interpreted
    /// as i32. Shape `[num_experts_per_tok = 6]`.
    pub topk_indices: Option<rdna_compute::GpuTensor>,
    /// Per-routed-expert output scratch `[hidden]` F32. Reused for
    /// each of the K=6 selected experts; weighted-accumulated into
    /// `ffn_out` via `scaled_add_inplace_cpu_scalar_f32`. Legacy
    /// fallback-path scratch (no longer reachable from forward.rs).
    pub routed_expert_out: Option<rdna_compute::GpuTensor>,

    /// Phase 1 perf: fused MoE dispatch scratch.
    /// `moe_topk_indices` [k_top] i32, `moe_topk_weights` [k_top] f32
    /// (pre-multiplied by route_scale_override). Filled per-token from
    /// CPU top-K result.
    pub moe_topk_indices: Option<rdna_compute::GpuTensor>,
    pub moe_topk_weights: Option<rdna_compute::GpuTensor>,
    /// Per-expert SwiGLU intermediate buffers `[k_top × intermediate]`.
    pub moe_gate_batch: Option<rdna_compute::GpuTensor>,
    pub moe_up_batch: Option<rdna_compute::GpuTensor>,
    pub moe_rot_batch: Option<rdna_compute::GpuTensor>,
    /// `[k_top × hidden]` per-expert down outputs for the deterministic MoE combine.
    pub moe_down_expert_outputs: Option<rdna_compute::GpuTensor>,

    /// Buffer of all-ones, length `head_dim`, used as the weight arg
    /// to the per-head Q RMSNorm (upstream DeepSeek V4 has NO learnable scale
    /// on the post-wq_b Q-norm, just rsqrt(mean(sq)+eps)). Allocated
    /// once on first attention layer.
    pub q_head_ones: Option<rdna_compute::GpuTensor>,

    /// Raw attention output `[n_heads, head_dim]` F32 = 32768 elems.
    /// Fed into the O-LoRA projection (wo_a + wo_b → state.attn_out).
    pub attn_out_raw: Option<rdna_compute::GpuTensor>,
    /// FWHT-rotated `attn_out_raw` for wo_a GEMV input.
    pub attn_out_raw_rot: Option<rdna_compute::GpuTensor>,
    /// wo_a output `[n_groups * o_lora_rank]` F32 = 8192 elems.
    pub wo_a_out: Option<rdna_compute::GpuTensor>,
    /// FWHT-rotated wo_a_out for the wo_b GEMV input.
    pub wo_a_out_rot: Option<rdna_compute::GpuTensor>,

    /// Head HC pre-weights `[hc_mult=4]` F32 from hc_head_compute_pre.
    pub head_hc_pre: Option<rdna_compute::GpuTensor>,
    /// Head HC combined-streams output `[hidden]` F32 → output_norm → lm_head.
    pub head_hc_out: Option<rdna_compute::GpuTensor>,

    /// Batched verify lm_head scratch (spec-decode). The lm_head weight is
    /// `[vocab, hidden]` (~565 MB Q8) and the GEMV is pure weight-BW-bound;
    /// the old per-position loop re-read it K times per window. These cache
    /// the staging buffers so the verifier reads it ONCE per window:
    ///   `head_norm_batch`   — per-position pre-lm_head normed acts `[K, hidden]` F32
    ///   `head_x_f16`        — F16-staged GEMM input `[K*hidden]`
    ///   `head_logits_batch` — `[K, vocab]` logits from the single batched GEMV
    pub head_norm_batch: Option<rdna_compute::GpuTensor>,
    pub head_x_f16: Option<rdna_compute::GpuTensor>,
    pub head_logits_batch: Option<rdna_compute::GpuTensor>,

    /// Monotonic position counter — how many tokens this session has
    /// processed. Used to compute the SWA cache slot (`pos % window`)
    /// and number of valid cached positions.
    pub n_tokens: u64,

    pub _scaffold: (),
}

impl DeepseekV4State {
    pub fn new(cfg: &DeepseekV4Config) -> Result<Self, String> {
        let n_layers_total = cfg.num_hidden_layers + cfg.num_nextn_predict_layers;
        let mut indexer = Vec::with_capacity(n_layers_total);
        let mut attention = Vec::with_capacity(n_layers_total);
        for layer in 0..n_layers_total {
            let ratio = *cfg.compress_ratios.get(layer).unwrap_or(&0);
            indexer.push(IndexerLayerState {
                compress_ratio: ratio,
                main_kv_cache: None,
                main_kv_state: None,
                main_score_state: None,
                indexer_kv_cache: None,
                indexer_kv_state: None,
                indexer_score_state: None,
                q_idx: None,
                idx_weights: None,
                index_score: None,
                topk_idx_indices: None,
                comp_kv_buf: None,
                comp_score_buf: None,
                comp_concat_kv: None,
                comp_concat_score: None,
            });
            attention.push(MainAttentionLayerState {
                swa_k: None,
                swa_v: None,
                full_k_cache: None,
                full_v_cache: None,
                gathered_k: None,
                gathered_v: None,
            });
        }
        Ok(DeepseekV4State {
            _indexer: indexer,
            _attention: attention,
            residual_streams: None, // allocated on first `decode_step` (needs Gpu).
            embed_scratch: None,
            tmp: None,
            tmp_plain: None,
            mtp_e_norm_scratch: None,
            mtp_h_norm_scratch: None,
            mtp_last_hidden: None,
            q_lat: None,
            q_lat_rot: None,
            q: None,
            kv: None,
            pos_buf: None,
            comp_pos_buf: None,
            pos_array_device: None,
            pos_array_host: None,
            ar_forward_warmed_up: false,
            token_id_buf: None,
            token_id_host: None,
            attn_state_buf: None,
            attn_state_host: None,
            attn_out: None,
            ffn_out: None,
            ffn_x_rot: None,
            ffn_x_plain: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_silu_rot: None,
            final_norm: None,
            logits: None,
            final_norm_rot: None,
            hc_x_in: None,
            hc_c: None,
            router_scores: None,
            topk_indices: None,
            routed_expert_out: None,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_gate_batch: None,
            moe_up_batch: None,
            moe_rot_batch: None,
            moe_down_expert_outputs: None,
            q_head_ones: None,
            attn_out_raw: None,
            attn_out_raw_rot: None,
            wo_a_out: None,
            wo_a_out_rot: None,
            head_hc_pre: None,
            head_hc_out: None,
            head_norm_batch: None,
            head_x_f16: None,
            head_logits_batch: None,
            n_tokens: 0,
            _scaffold: (),
        })
    }

    /// Reset the per-conversation position cursor so the next prefill
    /// starts at slot 0 of the SWA cache and slot 0 of the compressed-KV
    /// rings. Mirrors `Qwen2State::reset` — we keep every allocated GPU
    /// tensor alive (avoiding the realloc churn of `drop + ::new`) and
    /// rely on the position-derived slot computations
    /// (`pos % sliding_window`, `pos / ratio`) plus the
    /// `n_valid = min(pos+1, sliding_window)` clamp in attention to
    /// ensure stale data beyond the new `n_tokens` is never attended
    /// to. Future writes overwrite the slots as prefill progresses.
    ///
    /// Why this matters: the daemon's stateless OpenAI-API contract
    /// is "every request carries the full conversation; daemon serves
    /// it from scratch." Without this reset, the V4F arm's
    /// `state.n_tokens` accumulated across requests (Qwen-style arches
    /// already reset their equivalent counters in
    /// `daemon.rs::"reset"`), so `forward_prefill_batch_chunked` ran
    /// with `start_pos = sum_of_prior_prefill_lengths` and wrote the
    /// new conversation AFTER the prior one's KV slots. The model then
    /// saw two stacked conversations through compressed-KV attention,
    /// which matches the multi-turn path-recall corruption reported
    /// in pi-coding-agent sessions (`/home/nick/CLion/tembrane`,
    /// `/home/n/Downloads`) — the indexer was top-K-selecting
    /// compressed slots from BOTH the current turn and an earlier
    /// turn that had a similar-but-distinct embedding.
    ///
    /// If/when we add real prefix caching (LCP detection between the
    /// new prompt and the existing KV-resident tokens), the daemon
    /// should grow a separate "continue" command that skips this
    /// reset and uses `start_pos = lcp_len` instead.
    pub fn reset(&mut self) {
        self.n_tokens = 0;
        // mtp_last_hidden carries the prior decode's full HC residual
        // stream and is only consumed by `mtp_forward` (spec decode).
        // Leaving it populated would let the first MTP step of a new
        // turn read stale data; drop the handle so the next request's
        // first spec step takes the alloc-then-fill path again.
        self.mtp_last_hidden = None;
        // Force the next decode loop to retrace the warmup path
        // (`decode_step` direct dispatch) before re-entering capture.
        // Pairs with `gpu.invalidate_graph_state()` in the daemon's
        // reset handler: graph_exec is dropped, ar_forward_warmed_up
        // is dropped, so the next session's first decode call runs
        // plain `decode_step` (allocs land out of the captured region),
        // the second call captures fresh, and replay starts on the
        // third call against state shaped for THIS session.
        //
        // Why this matters: the captured graph bakes the device-buffer
        // pointers visited at capture time. Across reset() the buffers
        // themselves are stable, but state-derived host scalars
        // (compressor rope_pos default differs between
        // `precompute_positions` "mid" and `update_pos_array_host`
        // "start") and slot-derived lazy allocs (e.g.
        // `state._attention[l].gathered_k` only allocates when a
        // mixed-attention layer first fires) can land INSIDE the
        // captured region of session-2 if session-1 didn't hit them.
        // Forcing warmup → capture per session sidesteps the whole
        // class of capture-time/replay-time staleness — and the cost
        // is one extra direct-dispatch decode per turn, which is
        // dwarfed by the prefill it follows.
        self.ar_forward_warmed_up = false;
        // Other transient scratch tensors (`tmp`, `tmp_plain`, residual_streams,
        // attn_state_host, …) are pure per-step working memory: each
        // forward step writes them before reading, so stale contents
        // are overwritten on first use and don't need explicit zeroing.
    }

    /// Zero the persistent per-layer decode caches (SWA ring, full + compressed
    /// KV, indexer/compressor scratch) so a fresh conversation starts from the
    /// same clean state as a freshly-launched daemon.
    ///
    /// `reset()` only rewinds `n_tokens` and leaves these position-indexed
    /// caches intact; a short new conversation doesn't overwrite every slot the
    /// forward reads, so the prior turn's residue bleeds in and drifts greedy
    /// decode.
    ///
    /// Must be called where `gpu` is available (the daemon's lcp==0 fresh-
    /// conversation handler), not from `reset()` (no gpu there).
    pub fn zero_decode_caches(&mut self, gpu: &mut rdna_compute::Gpu) {
        fn z(gpu: &mut rdna_compute::Gpu, t: &Option<rdna_compute::GpuTensor>) {
            if let Some(t) = t {
                let _ = gpu.hip.memset(&t.buf, 0, t.byte_size());
            }
        }
        // The compressor `score_state` ring must reset to -inf, NOT 0 (matches
        // the reference `torch.full(-inf)`): a fresh conversation's first
        // compressed block has no overlap prev-window, and those unfilled slots
        // must get zero softmax weight in the pooling. Reset to 0 would instead
        // pool the prior turn's stale window (or dilute block 0 with zeros).
        fn zinf(gpu: &mut rdna_compute::Gpu, t: &Option<rdna_compute::GpuTensor>) {
            if let Some(t) = t {
                let _ = gpu.fill_f32(t, f32::NEG_INFINITY);
            }
        }
        for l in &self._indexer {
            z(gpu, &l.main_kv_cache);
            z(gpu, &l.main_kv_state);
            zinf(gpu, &l.main_score_state);
            z(gpu, &l.indexer_kv_cache);
            z(gpu, &l.indexer_kv_state);
            zinf(gpu, &l.indexer_score_state);
            z(gpu, &l.comp_kv_buf);
            z(gpu, &l.comp_score_buf);
            z(gpu, &l.comp_concat_kv);
            z(gpu, &l.comp_concat_score);
        }
        for l in &self._attention {
            z(gpu, &l.swa_k);
            z(gpu, &l.swa_v);
            z(gpu, &l.full_k_cache);
            z(gpu, &l.full_v_cache);
            z(gpu, &l.gathered_k);
            z(gpu, &l.gathered_v);
        }
    }

    /// Release every GPU buffer this state owns back to the pool.
    /// Consumes self. Mirrors `Qwen2State::free_gpu`.
    ///
    /// Without this, `unload_model` for an `arch_id=9` LoadedModel would
    /// drop `DeepseekV4State` via plain Rust Drop, which doesn't return
    /// the pool-managed GpuTensors — every load/unload cycle leaks the
    /// per-session scratch + per-layer SWA/indexer/compressor caches
    /// until the daemon exits. Idle eviction can't reclaim VRAM in that
    /// regime, defeating its purpose.
    pub fn free_gpu(mut self, gpu: &mut rdna_compute::Gpu) {
        fn free_opt(gpu: &mut rdna_compute::Gpu, t: &mut Option<rdna_compute::GpuTensor>) {
            if let Some(t) = t.take() {
                let _ = gpu.free_tensor(t);
            }
        }
        // Per-layer indexer + main-attention caches.
        for mut l in self._indexer.drain(..) {
            free_opt(gpu, &mut l.main_kv_cache);
            free_opt(gpu, &mut l.main_kv_state);
            free_opt(gpu, &mut l.main_score_state);
            free_opt(gpu, &mut l.indexer_kv_cache);
            free_opt(gpu, &mut l.indexer_kv_state);
            free_opt(gpu, &mut l.indexer_score_state);
            free_opt(gpu, &mut l.q_idx);
            free_opt(gpu, &mut l.idx_weights);
            free_opt(gpu, &mut l.index_score);
            free_opt(gpu, &mut l.topk_idx_indices);
            free_opt(gpu, &mut l.comp_kv_buf);
            free_opt(gpu, &mut l.comp_score_buf);
            free_opt(gpu, &mut l.comp_concat_kv);
            free_opt(gpu, &mut l.comp_concat_score);
        }
        for mut l in self._attention.drain(..) {
            free_opt(gpu, &mut l.swa_k);
            free_opt(gpu, &mut l.swa_v);
            free_opt(gpu, &mut l.full_k_cache);
            free_opt(gpu, &mut l.full_v_cache);
            free_opt(gpu, &mut l.gathered_k);
            free_opt(gpu, &mut l.gathered_v);
        }
        // Top-level scratch.
        free_opt(gpu, &mut self.residual_streams);
        free_opt(gpu, &mut self.embed_scratch);
        free_opt(gpu, &mut self.tmp);
        free_opt(gpu, &mut self.tmp_plain);
        free_opt(gpu, &mut self.mtp_e_norm_scratch);
        free_opt(gpu, &mut self.mtp_h_norm_scratch);
        free_opt(gpu, &mut self.mtp_last_hidden);
        free_opt(gpu, &mut self.q_lat);
        free_opt(gpu, &mut self.q_lat_rot);
        free_opt(gpu, &mut self.q);
        free_opt(gpu, &mut self.kv);
        free_opt(gpu, &mut self.pos_buf);
        free_opt(gpu, &mut self.comp_pos_buf);
        free_opt(gpu, &mut self.pos_array_device);
        free_opt(gpu, &mut self.token_id_buf);
        free_opt(gpu, &mut self.attn_state_buf);
        free_opt(gpu, &mut self.attn_out);
        free_opt(gpu, &mut self.ffn_out);
        free_opt(gpu, &mut self.ffn_x_rot);
        free_opt(gpu, &mut self.ffn_x_plain);
        free_opt(gpu, &mut self.ffn_gate);
        free_opt(gpu, &mut self.ffn_up);
        free_opt(gpu, &mut self.ffn_silu_rot);
        free_opt(gpu, &mut self.final_norm);
        free_opt(gpu, &mut self.logits);
        free_opt(gpu, &mut self.final_norm_rot);
        free_opt(gpu, &mut self.hc_x_in);
        free_opt(gpu, &mut self.hc_c);
        free_opt(gpu, &mut self.router_scores);
        free_opt(gpu, &mut self.topk_indices);
        free_opt(gpu, &mut self.routed_expert_out);
        free_opt(gpu, &mut self.moe_topk_indices);
        free_opt(gpu, &mut self.moe_topk_weights);
        free_opt(gpu, &mut self.moe_gate_batch);
        free_opt(gpu, &mut self.moe_up_batch);
        free_opt(gpu, &mut self.moe_rot_batch);
        free_opt(gpu, &mut self.moe_down_expert_outputs);
        free_opt(gpu, &mut self.q_head_ones);
        free_opt(gpu, &mut self.attn_out_raw);
        free_opt(gpu, &mut self.attn_out_raw_rot);
        free_opt(gpu, &mut self.wo_a_out);
        free_opt(gpu, &mut self.wo_a_out_rot);
        free_opt(gpu, &mut self.head_hc_pre);
        free_opt(gpu, &mut self.head_hc_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEEPSEEK4_CONFIG_JSON: &str = r#"{
        "vocab_size": 129280, "hidden_size": 4096, "num_hidden_layers": 43,
        "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
        "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6,
        "q_lora_rank": 1024, "o_lora_rank": 1024, "qk_rope_head_dim": 64,
        "o_groups": 8,
        "n_routed_experts": 256, "n_shared_experts": 1,
        "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
        "routed_scaling_factor": 1.5, "topk_method": "noaux_tc",
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": true,
        "swiglu_limit": 10.0,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 0],
        "compress_rope_theta": 160000,
        "rope_theta": 10000,
        "rope_scaling": {
            "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1, "type": "yarn"
        },
        "sliding_window": 128,
        "num_nextn_predict_layers": 1, "num_hash_layers": 3
    }"#;

    #[test]
    fn parses_deepseek4_config_shape() {
        let raw: RawDeepseekV4Config = serde_json::from_str(DEEPSEEK4_CONFIG_JSON).unwrap();
        assert_eq!(raw.num_hidden_layers, 43);
        assert_eq!(raw.head_dim, 512);
        assert_eq!(raw.qk_rope_head_dim, 64);
        assert_eq!(raw.q_lora_rank, 1024);
        assert_eq!(raw.o_lora_rank, 1024);
        assert_eq!(raw.n_routed_experts, 256);
        assert_eq!(raw.num_experts_per_tok, 6);
        assert_eq!(raw.hc_mult, 4);
        assert_eq!(raw.hc_sinkhorn_iters, 20);
        assert_eq!(raw.index_n_heads, 64);
        assert_eq!(raw.index_head_dim, 128);
        assert_eq!(raw.index_topk, 512);
        assert_eq!(raw.sliding_window, 128);
        assert_eq!(raw.compress_ratios.len(), 8);
    }

    /// Verify the parser handles the actual released DeepSeek V4 config.json
    /// (snapshot 6976c7ff). Catches schema drift if the upstream
    /// model card adds or renames fields.
    #[test]
    fn parses_real_deepseek4_config_json() {
        let real_config_path =
            "/home/nick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/\
             snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136/config.json";
        let raw_json = match std::fs::read_to_string(real_config_path) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("skipping real-config test — DeepSeek V4 not locally available");
                return;
            }
        };
        // Real config has extra fields beyond what RawDeepseekV4Config
        // reads (architectures, attention_bias, etc). serde silently
        // ignores them — verify we still parse the fields we care about.
        let raw: RawDeepseekV4Config = serde_json::from_str(&raw_json)
            .expect("real DeepSeek V4 config.json must parse — schema drift detected");

        // Cross-check against the documented DeepSeek V4 constants.
        assert_eq!(raw.num_hidden_layers, 43);
        assert_eq!(raw.head_dim, 512);
        assert_eq!(raw.qk_rope_head_dim, 64);
        assert_eq!(raw.q_lora_rank, 1024);
        assert_eq!(raw.o_lora_rank, 1024);
        assert_eq!(raw.n_routed_experts, 256);
        assert_eq!(raw.num_experts_per_tok, 6);
        assert_eq!(raw.hc_mult, 4);
        assert_eq!(raw.index_n_heads, 64);
        assert_eq!(raw.sliding_window, 128);

        // The released checkpoint's compress_ratios has length
        // num_hidden_layers + num_nextn_predict_layers = 44.
        assert_eq!(
            raw.compress_ratios.len(),
            raw.num_hidden_layers + raw.num_nextn_predict_layers
        );

        // Check the alternating pattern in the middle layers.
        // DeepSeek V4 shipped pattern: [0, 0, 4, 128, 4, 128, ..., 4, 0].
        for (i, &r) in raw.compress_ratios.iter().enumerate() {
            if i < 2 || i == raw.compress_ratios.len() - 1 {
                assert_eq!(r, 0, "layer {i}: expected ratio=0, got {r}");
            } else {
                let expected = if i % 2 == 0 { 4 } else { 128 };
                assert_eq!(r, expected, "layer {i}: expected ratio={expected}, got {r}");
            }
        }

        // Verify DeepseekV4State::new accepts the real config.
        // (Reconstruct the full Config from raw to drive State::new.)
        // Skip — would require a fake HfqFile. The shape test above
        // is enough for the schema-drift gate.
    }
}
