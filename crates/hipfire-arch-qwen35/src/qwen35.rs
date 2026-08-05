// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.

use crate::speculative::HiddenStateRingBuffer;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::{DispatchCtx, DispatchWorkload};
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::{GivensRef, WeightRef};
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, LayerProgram, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_dispatch::types::{DispatchError, RotationPlan};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::{
    self, f16_to_f32, fused_rmsnorm_rotate_for_mq, fused_rmsnorm_rotate_mq_batched_for,
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, weight_gemv_prerotated,
    weight_gemv_swiglu_residual, EmbeddingFormat, ParoRotation, WeightTensor,
};
use hipfire_runtime::model_load::{load_weights as rt_load_weights, LoadedWeights, WeightSource};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::paro::{paro_load_norm, paro_text_prefix};
use hipfire_runtime::tp_shard::ShardConfig;
use hipfire_runtime::weight_backend::{
    dequant_norm, dequant_weight_raw, load_awq_scale_for, load_embedding, resolve_lm_head,
    reupload_f16_as_f32, HfqBackend, ParoBackend,
};
use hipfire_runtime::{screen_weight_tensor, MmqScreenable};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;

/// RMSNorm weight bias for qwen3.5/gemma-style norms: dequant computes `w + norm_bias`.
/// qwen2/llama use `0.0`. Single source of truth — referenced by the backend constructors
/// and both final-norm paths so the four former hardcoded `1.0` sites cannot drift apart.
const QWEN35_NORM_BIAS: f32 = 1.0;

const _: () = assert!(QWEN35_NORM_BIAS == 1.0);

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    LinearAttention, // DeltaNet
    FullAttention,   // Standard MHA with gated output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum F16LmHeadMode {
    Native,
    F32,
}

fn parse_f16_lm_head_mode(value: Option<&str>) -> F16LmHeadMode {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if matches!(v.as_str(), "0" | "f32" | "fp32" | "legacy") => F16LmHeadMode::F32,
        _ => F16LmHeadMode::Native,
    }
}

fn f16_lm_head_mode_from_config() -> F16LmHeadMode {
    parse_f16_lm_head_mode(Some(hipfire_runtime::config::get().lm_head_f16.as_str()))
}

/// Optional tree-attention context for `forward_prefill_batch` — activates
/// DDTree batched verify when `Some`.
///
/// Fields:
/// - `positions`: length matches `tokens.len()`. Each slot's logical RoPE
///   position (seed at `start_pos`, node i at `start_pos + depth_i`).
///   Two nodes at the same tree depth share a logical position — they're
///   alternative futures at the same time step, not successive tokens.
/// - `attn_bias`: `[N × N]` f32 additive bias on qk scores (with N = tokens.len()),
///   produced by `hipfire_runtime::ddtree::linearize_tree`. `0.0` on ancestor-or-self
///   entries, `-inf` on non-ancestors. Applied to in-block keys only;
///   prompt keys (positions `[0, start_pos)`) remain unmasked.
///
/// Tree mode requires the batched FA path (`fa_batched_ok`); the per-token
/// FA fallback always uses causal attention and cannot honor a tree mask.
/// `forward_prefill_batch` returns an error if tree mode is requested but
/// any FA layer would take the fallback path.
///
/// GDN (LinearAttention) layers: if `parent_indices` is `Some`, the
/// DeltaNet branch dispatches the tree-aware kernels
/// (`conv1d_silu_split_tree_f32_n` + `gated_delta_net_q8_tree_batch_seq`)
/// which walk per-token ancestor chains via `parent_indices` instead of
/// the linear-sequence predecessor. This eliminates sibling-subtree
/// cross-contamination of recurrent state at topk>1. If `parent_indices`
/// is `None`, LA layers fall back to the linear path (byte-exact with
/// DFlash at topk=1; approximation at topk>1 — used by pre-Phase-3
/// callers that haven't been rewritten).

/// Override the embedding for a single batch slot after the embedding-lookup
/// kernel runs but before the layer loop. Used by the Qualcomm-style MTP
/// probe (mtp_probe.rs) to inject mask-token embeddings whose values come
/// from prompt-mean rather than the embedding table.
///
/// Default callers pass `None`; passing `Some(_)` triggers a single
/// host-to-device memcpy into `pbs.x_batch.buf` at byte offset
/// `slot * config.dim * 4` AFTER the embedding-lookup kernel populates
/// the batched-x scratch and BEFORE the first layer reads it.
///
/// Constraints:
///   - `slot < tokens.len()` of the call (asserted)
///   - `embed.len() == config.dim` (asserted)
///   - The override is applied unconditionally to whichever chunk's range
///     contains `slot`. Multi-chunk callers MUST size the prefill batch
///     scratch to keep their target slot in chunk 0, or pass the override
///     only on the chunk where `slot < chunk_n`. (For the MTP probe the
///     entire mask block fits in one chunk by construction.)
#[derive(Clone, Copy)]
pub struct MaskEmbedOverride<'a> {
    pub slot: usize,
    pub embed: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct TreeVerifyCtx<'a> {
    pub positions: &'a [i32],
    pub attn_bias: &'a GpuTensor,
    /// `[N]` i32 — for each linearized slot, the slot index of its parent
    /// in the same linearization (or -1 for the root / seed). Produced by
    /// `hipfire_runtime::ddtree::linearize_tree_with_parents`. When `Some`, LA layers
    /// use tree-aware kernels that read parent state from the per-layer
    /// s_tape scratch in `PrefillBatchScratch`.
    pub parent_indices: Option<&'a GpuTensor>,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub eos_token: u32,

    // Full attention params
    pub n_heads: usize,    // 8
    pub n_kv_heads: usize, // 2
    pub head_dim: usize,   // 256
    pub rope_theta: f32,
    pub partial_rotary_factor: f32, // 0.25 — only 64/256 dims get RoPE
    /// True when a composite Qwen3.5-VL checkpoint is being used as a
    /// text-only model through its nested `text_config`.
    pub is_vl_text: bool,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],

    // DeltaNet params
    pub linear_num_key_heads: usize,   // 16
    pub linear_num_value_heads: usize, // 16
    pub linear_key_head_dim: usize,    // 128
    pub linear_value_head_dim: usize,  // 128
    pub conv_kernel_dim: usize,        // 4

    // FFN — dense; for MoE see num_experts below
    pub hidden_dim: usize, // 3584 (dense) or unused when num_experts > 0

    // MoE (qwen3_5_moe / A3B). num_experts == 0 means plain dense (qwen3_5).
    pub num_experts: usize,                     // 256 for A3B
    pub num_experts_per_tok: usize,             // 8 for A3B
    pub moe_intermediate_size: usize,           // 512 for A3B (per-routed-expert FFN)
    pub shared_expert_intermediate_size: usize, // 512 for A3B
    pub has_shared_expert: bool,                // true for A3B (always-on shared expert)
    /// If true, top-K routing weights are re-normalized to sum to 1 after
    /// softmax + top-K selection. Qwen convention (matches HF
    /// `modeling_qwen3_5_moe.py`). DeepSeek-v1 uses false.
    pub norm_topk_prob: bool,

    // Per-layer type dispatch
    pub layer_types: Vec<LayerType>,

    // ── Weight pager (MAD-93 v0.1) ───────────────────────────────────
    /// If true, MoE expert weights are managed by [`hipfire_runtime::weight_pager::WeightPager`]
    /// and only the active top-k experts per layer are guaranteed resident in
    /// VRAM. Default false (all experts resident, today's behavior).
    ///
    /// Off-switch for the v0.1 PR: when false there is no behavior change
    /// vs main; when true the forward path takes the paged code path which
    /// uses a CPU-side router replica + on-demand H2D transfers.
    pub paged_experts: bool,

    /// Soft cap on VRAM bytes the weight pager is allowed to hold for paged
    /// expert weights. Only meaningful when `paged_experts == true`. Defaults
    /// to `u64::MAX` (no eviction — tested when VRAM is unlimited or we just
    /// want to verify the routing path works without eviction pressure).
    pub vram_budget_bytes: u64,

    /// Optional REAP keep-map: emulate a pruned routed-expert pool by
    /// partial-loading this full quant (load only the kept experts under
    /// remapped names, gather the router's expert rows to the kept set).
    /// Populated at config time from `HIPFIRE_REAP_PLAN=<dir>`; `None` ⇒
    /// no pruning (today's behavior, byte-identical to baseline). Not
    /// (de)serialized — `Qwen35Config` does not derive serde.
    pub reap_keep: Option<std::sync::Arc<hipfire_reap::plan::ReapPlan>>,
}

/// Nested `rope_parameters` block. All fields optional — Qwen3.5 carries
/// `rope_theta` here; VL/mrope variants add the section + interleave flags.
/// `partial_rotary_factor` may also live FLAT on the text config (handled in
/// finalize), so it's read from both places.
#[derive(Deserialize)]
struct RawRope {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    mrope_interleaved: Option<bool>,
    #[serde(default)]
    mrope_section: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
}

#[derive(Deserialize)]
struct RawQwen35Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    // Dense FFN intermediate dim. MoE configs (qwen3_5_moe / A3B) replace this
    // with `moe_intermediate_size` and don't ship `intermediate_size`, so it
    // defaults to 0 rather than hard-failing — we still need the rest of the
    // config to detect is_moe and route accordingly.
    #[serde(default)]
    intermediate_size: usize,
    #[serde(default = "default_norm_eps")]
    rms_norm_eps: f32,
    // Real safetensors configs ship `eos_token_id` as either a scalar
    // (Qwen3.5 dense) or an array (some Qwen3.5 MoE / chat checkpoints). Keep
    // it as a raw Value and resolve to the FIRST element in finalize (uniform
    // with qwen2's `eos_token_id = eos_token_ids[0]`).
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    rope_parameters: Option<RawRope>,
    // FLAT partial_rotary_factor takes precedence over the nested one (finalize).
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default = "default_linear_heads")]
    linear_num_key_heads: usize,
    #[serde(default = "default_linear_heads")]
    linear_num_value_heads: usize,
    #[serde(default = "default_linear_head_dim")]
    linear_key_head_dim: usize,
    #[serde(default = "default_linear_head_dim")]
    linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    // MoE config (zeros = dense fallback). Qwen3.5-MoE / A3B sets these.
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    // Qwen convention: re-normalize top-K routing weights to sum to 1.
    // Absent from some configs (including the shipped A3B HFQ); default on
    // for Qwen3.5-MoE / A3B to match the HF reference.
    #[serde(default = "default_norm_topk")]
    norm_topk_prob: bool,
}

fn default_norm_eps() -> f32 {
    1e-6
}
/// Resolve a scalar-or-array `eos_token_id` to a single token, using the first
/// element of an array (uniform with qwen2). Absent/null/unexpected → default.
fn first_token_or(v: Option<&serde_json::Value>, default: u32) -> u32 {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| x as u32).unwrap_or(default),
        Some(serde_json::Value::Array(a)) => a
            .first()
            .and_then(|e| e.as_u64())
            .map(|x| x as u32)
            .unwrap_or(default),
        _ => default,
    }
}
fn default_linear_heads() -> usize {
    16
}
fn default_linear_head_dim() -> usize {
    128
}
fn default_conv_kernel() -> usize {
    4
}
fn default_norm_topk() -> bool {
    true
}

/// Parse a `Qwen35Config` from the OUTER `config` JSON node (the inner blob
/// under the metadata_json `config` key). Descends into `text_config` when
/// present (composite VL checkpoints used text-only) and also inspects the
/// outer node for `vision_config` to set `is_vl_text`.
///
/// Shared by both `config_from_hfq` and `config_from_safetensors`: the two
/// envelope sources are byte-identical past the `meta["config"]` node.
fn from_config_value(config: &serde_json::Value) -> Result<Qwen35Config, String> {
    let tc = config.get("text_config").unwrap_or(config);
    let raw: RawQwen35Config = serde_json::from_value(tc.clone())
        .map_err(|e| format!("qwen35: parsing config failed: {e}"))?;
    let is_vl_text = config.get("text_config").is_some() && config.get("vision_config").is_some();

    let dim = raw.hidden_size;
    let n_heads = raw.num_attention_heads;
    let n_kv_heads = raw.num_key_value_heads.unwrap_or(n_heads);
    let head_dim = raw.head_dim.unwrap_or(dim / n_heads);

    let rope = raw.rope_parameters.as_ref();
    let rope_theta = rope.and_then(|r| r.rope_theta).unwrap_or(10_000_000.0) as f32;
    // FLAT partial_rotary_factor wins over the nested one; default 0.25.
    let partial_rotary_factor = raw
        .partial_rotary_factor
        .or_else(|| rope.and_then(|r| r.partial_rotary_factor))
        .unwrap_or(0.25) as f32;
    let mrope_interleaved = rope.and_then(|r| r.mrope_interleaved).unwrap_or(false);
    let mut mrope_section = [11usize, 11usize, 10usize];
    if let Some(arr) = rope.and_then(|r| r.mrope_section.as_ref()) {
        for (dst, src) in mrope_section.iter_mut().zip(arr.iter().take(3)) {
            if let Some(v) = src.as_u64() {
                *dst = v as usize;
            }
        }
    }

    let layer_types: Vec<LayerType> = raw
        .layer_types
        .as_ref()
        .map(|arr| {
            arr.iter()
                .map(|s| match s.as_str() {
                    "linear_attention" => LayerType::LinearAttention,
                    _ => LayerType::FullAttention,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![LayerType::FullAttention; raw.num_hidden_layers]);

    let has_shared_expert = raw.shared_expert_intermediate_size > 0;

    let mut config = Qwen35Config {
        dim,
        n_layers: raw.num_hidden_layers,
        vocab_size: raw.vocab_size,
        norm_eps: raw.rms_norm_eps,
        eos_token: first_token_or(raw.eos_token_id.as_ref(), 248044),
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor,
        is_vl_text,
        mrope_interleaved,
        mrope_section,
        linear_num_key_heads: raw.linear_num_key_heads,
        linear_num_value_heads: raw.linear_num_value_heads,
        linear_key_head_dim: raw.linear_key_head_dim,
        linear_value_head_dim: raw.linear_value_head_dim,
        conv_kernel_dim: raw.linear_conv_kernel_dim,
        hidden_dim: raw.intermediate_size,
        layer_types,
        num_experts: raw.num_experts,
        num_experts_per_tok: raw.num_experts_per_tok,
        moe_intermediate_size: raw.moe_intermediate_size,
        shared_expert_intermediate_size: raw.shared_expert_intermediate_size,
        has_shared_expert,
        norm_topk_prob: raw.norm_topk_prob,
        // MAD-93 v0.1: defaults off; runtime opts in (e.g. via CLI flag in
        // a follow-up commit). When false, no behavior change vs main.
        paged_experts: false,
        vram_budget_bytes: u64::MAX,
        reap_keep: None,
    };

    // Apply the optional REAP keep-map HERE, inside the single public config
    // entry point, so it is IMPOSSIBLE to bypass. `config_from_hfq` has ~50
    // direct callers (daemon, perplexity example, every bench/profile example)
    // that never go through the `Architecture` trait shim; wiring REAP only in
    // the trait impl would silently ignore HIPFIRE_REAP_PLAN on all of them
    // (including the deferred identity NLL gate, which the perplexity example
    // drives via this public fn). The trait impl therefore does NOT re-apply.
    //
    // Error policy: config parsing now returns `Result<_, String>`, so an
    // explicitly malformed REAP plan propagates as a hard load error instead
    // of getting collapsed into a generic "bad metadata" fallback.
    apply_reap_plan(&mut config)?;

    Ok(config)
}

/// Apply an optional REAP keep-map to a freshly parsed `Qwen35Config`.
///
/// Reads `HIPFIRE_REAP_PLAN=<dir>` (qwen35 has no legacy env alias). When
/// set, loads `<dir>/reap_plan.json` (or the legacy `keep_by_layer.json`)
/// via `ReapPlan::load_any`, validating against the ORIGINAL routed-expert
/// count (`config.num_experts`) BEFORE overriding it to the kept count.
/// This emulates a pruned expert pool by partial-loading the full quant:
/// only kept experts are loaded (under remapped names) and the router's
/// expert rows are gathered to the kept set in `load_moe_ffn`.
///
/// No env ⇒ no-op (`config.reap_keep` stays `None`); the MoE loader then
/// takes the literal original full-load path — byte-identical to baseline.
/// Only the HFQ MoE path (`load_moe_ffn`) honors the keep-map; the
/// ParoQuant path does not (see `paro_load_moe_ffn`).
pub fn apply_reap_plan(config: &mut Qwen35Config) -> Result<(), String> {
    if let Some(plan) = hipfire_reap::plan::ReapPlan::from_config(
        "qwen35",
        None,
        config.n_layers,
        config.num_experts,
    )? {
        config.num_experts = plan.kept_per_layer();
        config.reap_keep = Some(std::sync::Arc::new(plan));
    }
    Ok(())
}

/// Inner parser, decoupled from `HfqFile` / `ModelSource` for unit testability.
///
/// Parses the metadata JSON string, unwraps the `{config}` envelope both
/// sources build, then delegates to [`from_config_value`]. Both
/// `config_from_hfq` and `config_from_safetensors` call this, so the ×2
/// collapse is at the string→config boundary.
pub fn config_from_metadata_json(metadata_json: &str) -> Result<Qwen35Config, String> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("qwen35: metadata_json not valid JSON: {e}"))?;
    from_config_value(meta.get("config").ok_or("qwen35: missing config")?)
}

pub fn config_from_hfq(hfq: &HfqFile) -> Result<Qwen35Config, String> {
    config_from_metadata_json(&hfq.metadata_json)
}

/// Parse Qwen35Config from a SafetensorsSource (or any ModelSource).
/// Delegates to the same JSON parser as config_from_hfq — the SafetensorsSource
/// builds compatible metadata JSON from config.json.
pub fn config_from_safetensors(source: &dyn ModelSource) -> Result<Qwen35Config, String> {
    config_from_metadata_json(source.metadata_json())
}

// ─── Weight structs ─────────────────────────────────────────────────────

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,   // input_layernorm [dim]
    pub wqkv: WeightTensor,     // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,       // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,  // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,   // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,       // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,     // dt_bias [n_heads]
    pub conv_weight: GpuTensor, // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor, // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,       // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,    // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,   // mlp.gate_proj
    pub w_up: WeightTensor,     // mlp.up_proj
    pub w_down: WeightTensor,   // mlp.down_proj
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,  // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,  // k_proj
    pub wv: WeightTensor,  // v_proj
    pub wo: WeightTensor,  // o_proj
    pub q_norm: GpuTensor, // q_norm [head_dim]
    pub k_norm: GpuTensor, // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

// ─── MoE FFN weights (Qwen3.5-MoE / A3B) ────────────────────────────────
//
// Replaces the dense (w_gate, w_up, w_down) triple with N+1 expert FFNs
// gated by a router, plus a shared always-on expert.
//
// A3B specifics:
//   num_experts = 256, top_k = 8, moe_intermediate = 512, hidden = 2048
//   shared_expert_intermediate = 512 (same as routed)
//
// Per-layer storage:
//   router:               [num_experts, hidden]  MQ4G256 / Q8
//   shared_expert_gate:   [1, hidden]            MQ4G256 / Q8 — projects to scalar
//   experts[X].gate_up:   [2*moe_intermediate, hidden]  MQ4G256
//   experts[X].down:      [hidden, moe_intermediate]    MQ4G256
//   shared_expert.gate:   [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.up:     [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.down:   [hidden, shared_expert_intermediate]   MQ4G256
//
// The quantizer (hipfire-quantize) splits the safetensors 3D
// `mlp.experts.gate_up_proj` / `down_proj` tensors per-expert into
// `mlp.experts.{X}.gate_up_proj.weight` / `down_proj.weight` so the loader
// can fish them out by index. The shared expert is stored with separate
// gate_proj + up_proj + down_proj (it is not fused in safetensors either).

pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2 * moe_intermediate, hidden] — fused (gate || up)
    pub down: WeightTensor,    // [hidden, moe_intermediate]
}

/// Owning storage for a layer's packed uniform-MQ4 routed experts.
///
/// `experts` still carries one [`WeightTensor`] view per routed expert so the
/// CPU fallback and every existing indexed dispatch keep their exact metadata
/// and pointer-table ABI. Those views are non-owning subranges of these two
/// buffers; only this owner pair may be returned to the GPU pool.
pub(crate) struct PackedExpertOwners {
    gate_up: GpuTensor,
    down: GpuTensor,
}

/// SP2: build the per-expert (gate_up, down) quant-tier tables that
/// [`hipfire_dispatch::families::moe::MoeDtypes`] uses to detect an
/// intra-layer mixed-tier layer.
///
/// A table is `Some(vec)` only when the layer genuinely spans >1 distinct
/// tier; a uniform layer — or paged mode where `experts` is empty — yields
/// `None`, which `MoeResolution::resolve` collapses to the unchanged uniform
/// fast path. We pre-filter to `None` for the uniform/empty cases here so the
/// common path allocates nothing and is byte-identical to before SP2.
fn per_expert_tier_tables(experts: &[ExpertWeights]) -> (Option<Vec<DType>>, Option<Vec<DType>>) {
    let gu: Vec<DType> = experts.iter().map(|e| e.gate_up.gpu_dtype).collect();
    let dn: Vec<DType> = experts.iter().map(|e| e.down.gpu_dtype).collect();
    (mixed_tier_table(gu), mixed_tier_table(dn))
}

/// Collapse a per-expert dtype column to `None` when it is empty or uniform,
/// `Some` only when it spans >1 distinct tier. Pure (no GPU weights) so it is
/// unit-testable in isolation; `per_expert_tier_tables` is the GPU-weight
/// adapter over it.
fn mixed_tier_table(tiers: Vec<DType>) -> Option<Vec<DType>> {
    match tiers.first() {
        // Empty (paged mode) or uniform → uniform fast path.
        None => None,
        Some(&first) if tiers.iter().all(|&d| d == first) => None,
        Some(_) => Some(tiers),
    }
}

/// Shared expert storage — unlike routed experts, gate_proj and up_proj are
/// NOT fused in the safetensors, so we keep them separate here too. The
/// forward path does two GEMVs + silu_mul + down GEMV.
pub struct SharedExpertWeights {
    pub gate: WeightTensor, // [shared_expert_intermediate, hidden]
    pub up: WeightTensor,   // [shared_expert_intermediate, hidden]
    pub down: WeightTensor, // [hidden, shared_expert_intermediate]
}

pub struct MoeFfnWeights {
    pub router: WeightTensor, // [num_experts, hidden]
    /// Routed expert weights. Populated when this layer is fully resident
    /// (`paged_experts == false`); **empty `Vec`** when `paged_experts == true`
    /// (the [`hipfire_runtime::weight_pager::WeightPager`] owns the buffers, and the
    /// indexed kernels read pointers from `expert_*_ptrs` which the pager
    /// patches per-token via `patch_expert_ptr_table`).
    pub experts: Vec<ExpertWeights>, // num_experts (= 256 for A3B); empty in paged mode
    /// Two allocation owners for the uniform MQ4 packed path. `None` preserves
    /// the literal per-expert ownership used by mixed quant, Paro, paged, and
    /// EP-streaming routes.
    pub(crate) packed_expert_owners: Option<PackedExpertOwners>,
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, hidden] — row-vector projecting to scalar
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor, // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs: GpuTensor,      // [num_experts * 2] f32 slots = num_experts × u64

    /// Route A MoE-AWQ: per-expert down `awq_scale` pointer table
    /// (`[num_experts * 2]` f32 = num_experts × u64). `Some` only when the
    /// `.hfq` carries per-expert `down_proj.awq_scale` sidecars (all-or-none).
    /// Holds *non-owning* device pointers into each `experts[i].down.awq_scale`
    /// — freed as a buffer only; the scales are freed via
    /// `ExpertWeights::down.free_all`.
    pub expert_down_awq_ptrs: Option<GpuTensor>,

    /// Per-expert mixed-precision decode: `[num_experts]` u8 (DType::Raw,
    /// 1 B/expert) dtype-tag table. `Some` only when the layer's routed
    /// experts carry MIXED down dtypes (graded MQ6 hot / MQ2-Lloyd cold);
    /// the merged dtype-tag-branched down kernel reads `tags[expert_id]`
    /// per block (0=MQ6, 1=MQ2-Lloyd). `None` ⇒ uniform path, byte-identical.
    /// Owned device buffer (no aliasing) — freed as a buffer in free_moe_ffn.
    pub expert_dtype_tags: Option<GpuTensor>,

    /// Layer index. Stable identity used to key
    /// [`hipfire_runtime::weight_pager::WeightId::Expert`] entries.
    pub layer_idx: u16,

    /// Per-expert tensor shapes. `None` in non-paged mode (shapes are read
    /// from `experts[i].gate_up.{m, k}` etc.); `Some` in paged mode where
    /// `experts` is empty but kernels still need m/k for kernel-arg setup.
    /// Qwen3.5-MoE-A3B has uniform per-expert shape so one descriptor per
    /// layer suffices for v0.1.
    pub expert_shape: Option<hipfire_runtime::weight_pager::ExpertShape>,

    /// ParoQuant only: shared per-layer rotation sidecars for the routed
    /// experts. shisa-ai's PARO checkpoint quantizes all 256 experts with
    /// one rotation tuple per projection-group (gate||up vs down), so we
    /// upload the sidecars ONCE per layer and broadcast a non-owning
    /// `ParoRotation` (built via `DeviceBuffer::from_raw`) into every
    /// `ExpertWeights.gate_up.paro` / `ExpertWeights.down.paro`. The
    /// owning storage lives here so the aliases stay valid for the
    /// lifetime of the layer. `None` for HFQ MoE (per-tensor PARO sidecars
    /// or no PARO at all).
    pub paro_shared: Option<MoeParoSidecars>,
}

/// Owning storage for the per-layer shared ParoQuant rotation sidecars.
/// One tuple per projection-group:
///   - `gate_up_*`: applied to the post-RMSNorm hidden activation (K = hidden_dim).
///     Shared by all 256 experts' gate AND up projections, and by the fused
///     gate_up `WeightTensor`'s `paro` alias.
///   - `down_*`: applied to the post-SiLU intermediate activation (K = mi).
///     Shared by all 256 experts' down projection.
pub struct MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
    pub krot: u32,
    pub group_size: u32,
}

pub struct DeltaNetMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wqkv: WeightTensor,
    pub wz: WeightTensor,
    pub w_alpha: WeightTensor,
    pub w_beta: WeightTensor,
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub conv_weight: GpuTensor,
    pub norm_weight: GpuTensor,
    pub wo: WeightTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub struct FullAttnMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
    // A3B / qwen3_5_moe: same attention as above, MoE FFN instead of dense.
    // Loader + forward path TODO — adding the variants now so the enum is
    // forward-compatible and downstream code that pattern-matches gets a
    // compile-time hint to handle the new case.
    DeltaNetMoe(DeltaNetMoeLayerWeights),
    FullAttnMoe(FullAttnMoeLayerWeights),
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    /// True when any MoE FFN projection in the loaded model is MQ6. gfx1151's
    /// grouped-i8 MQ4 shortcut is model-level unsafe for these promoted A3B
    /// checkpoints, even in layers whose local routed experts remain MQ4.
    pub moe_has_mq6: bool,

    /// Weight pager (MAD-93 v0.1). `Some` only when the model was loaded
    /// with `Qwen35Config::paged_experts == true`. The forward path uses
    /// interior mutability (`borrow_mut`) at the MoE dispatch site to call
    /// `ensure_resident` / `patch_expert_ptr_table`. `None` means the model
    /// is fully resident — no behavior change vs main.
    pub pager: Option<std::cell::RefCell<hipfire_runtime::weight_pager::WeightPager>>,

    /// True when the tied lm_head aliases the embedding table buffer
    /// (single-GPU path). When true, `output.buf` is a non-owning view of
    /// `token_embd.buf` and must NOT be freed in `free_gpu`.
    pub lm_head_aliases_embd: bool,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        if !self.lm_head_aliases_embd {
            self.output.free_all(gpu);
        }
        for layer in self.layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
        // MAD-93 v0.1: in paged mode, the pager owns expert weight allocations
        // (the per-layer `free_moe_ffn` loops ran no-ops since `ffn.experts`
        // was empty). Drain the pager's resident set back to the GPU pool here.
        if let Some(pager_cell) = self.pager {
            pager_cell.into_inner().free_all(gpu);
        }
    }

    /// Multi-GPU companion to `free_gpu`. Each layer freed on its
    /// band-owning device per `gpus.device_for_layer(i)`; `token_embd`
    /// freed on dev 0; `output_norm + output` on `gpus.output_device`.
    /// Mirror of `load_weights_multi` placement. The `pager` field is
    /// always `None` on the multi path (paged-experts is not wired into
    /// pp>1 yet); a non-None pager would need its own per-band drain
    /// strategy and is rejected at load.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        debug_assert!(
            self.pager.is_none(),
            "free_gpu_multi: pager must be None on pp>1 path"
        );
        let _ = gpus.devices[0].free_tensor(self.token_embd);
        let out_dev = gpus.output_device;
        let _ = gpus.devices[out_dev].free_tensor(self.output_norm);
        self.output.free_all(&mut gpus.devices[out_dev]);
        for (i, layer) in self.layers.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let gpu = &mut gpus.devices[dev_idx];
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
    }
}

impl MmqScreenable for Qwen35Weights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.output, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            match layer {
                LayerWeights::DeltaNet(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttn(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                // Routed and shared experts live outside ordinary WeightTensor
                // storage in paged/EP modes. Screen the resident attention and
                // dense router weights here; expert screening is separate work.
                LayerWeights::DeltaNetMoe(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.ffn.router,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttnMoe(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.ffn.router,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
            }
        }
        (safe, unsafe_count)
    }
}

fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    ffn.router.free_all(gpu);
    ffn.shared_expert_gate.free_all(gpu);
    ffn.shared_expert.gate.free_all(gpu);
    ffn.shared_expert.up.free_all(gpu);
    ffn.shared_expert.down.free_all(gpu);
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    // Non-owning pointer table — free the buffer only; the per-expert scales it
    // points into are owned by `experts[i].down.awq_scale` and freed below via
    // `e.down.free_all`.
    if let Some(t) = ffn.expert_down_awq_ptrs {
        let _ = gpu.free_tensor(t);
    }
    // Owned device buffer (built from per-expert gpu_dtype). Free it.
    if let Some(t) = ffn.expert_dtype_tags {
        let _ = gpu.free_tensor(t);
    }
    if let Some(owners) = ffn.packed_expert_owners {
        // Packed expert WeightTensors are non-owning views. Free only metadata
        // that remains individually owned, then return each layer blob once.
        for e in ffn.experts {
            free_weight_metadata_only(gpu, e.gate_up);
            free_weight_metadata_only(gpu, e.down);
        }
        let _ = gpu.free_tensor(owners.gate_up);
        let _ = gpu.free_tensor(owners.down);
    } else {
        for e in ffn.experts {
            e.gate_up.free_all(gpu);
            e.down.free_all(gpu);
        }
    }
    // ParoQuant MoE: free the owning shared sidecars (per-expert `paro` fields
    // alias these and must NOT be freed separately — they're non-owning views).
    if let Some(s) = ffn.paro_shared {
        let _ = gpu.free_tensor(s.gate_up_pairs);
        let _ = gpu.free_tensor(s.gate_up_theta);
        let _ = gpu.free_tensor(s.gate_up_channel_scales);
        let _ = gpu.free_tensor(s.down_pairs);
        let _ = gpu.free_tensor(s.down_theta);
        let _ = gpu.free_tensor(s.down_channel_scales);
    }
}

/// Free a [`WeightTensor`]'s owning sidecars without freeing its weight buffer.
/// Used only for non-owning views into [`PackedExpertOwners`].
fn free_weight_metadata_only(gpu: &mut Gpu, weight: WeightTensor) {
    if let Some(paro) = weight.paro {
        if !paro.is_alias {
            let _ = gpu.free_tensor(paro.pairs);
            let _ = gpu.free_tensor(paro.theta);
            let _ = gpu.free_tensor(paro.channel_scales);
        }
    }
    if let Some(awq) = weight.awq_scale {
        let _ = gpu.free_tensor(awq);
    }
}

// ─── State ──────────────────────────────────────────────────────────────

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Per-element f16 error-feedback residual for Q8 state requant (sigma-delta
    /// noise-shaping). Empty unless Q8 + `HIPFIRE_DN_STATE_EF`. Same element count
    /// as `s_matrices`; carries the previous step's quant error so the next
    /// requant cancels it — DeltaNet's contractive decay damps the shaped noise,
    /// yielding ~FP32-grade state at Q8's byte container.
    pub s_ef_residual: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    /// EF residual for a delta-layer, if error-feedback is active (Q8 + flag).
    /// `None` ⇒ callers pass null ⇒ kernel uses the legacy stochastic-rounding requant.
    #[inline]
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, StateQuant::Q8)
    }

    pub fn new_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<Self> {
        let n_delta_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        // Error-feedback (sigma-delta) requant for Q8 state — DEFAULT ON as of
        // 2026-06-08. q8_ef ≈ FP32 coherence at −0.7% decode vs FP32's −4.5% (best
        // spec-decode τ too), and far better than stochastic Q8 — DFlash 27b-prose
        // unique_ratio 0.625 vs 0.555, max_freq 0.055 vs 0.078. Also makes the DN
        // state DETERMINISTIC (no stochastic dither). Opt OUT with
        // HIPFIRE_DN_STATE_EF=0. Q8-only (FP32 has no requant; Q4 EF is future
        // work; the multi-GPU band split is still stochastic — new_with_quant_multi
        // leaves s_ef_residual empty). Residual is f16 per-element.
        let ef_enabled = quant == StateQuant::Q8
            && hipfire_config::developer_var("HIPFIRE_DN_STATE_EF")
                .map(|v| v != "0")
                .unwrap_or(true);

        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        let mut s_ef_residual = Vec::with_capacity(if ef_enabled { n_delta_layers } else { 0 });
        for _ in 0..n_delta_layers {
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                    s_scales.push(gpu.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    // int8 state: s_size bytes (1 byte each), per-row scales
                    let buf = gpu.hip.malloc(s_size)?;
                    gpu.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                    let buf = gpu.hip.malloc(s_size / 2)?;
                    gpu.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            if ef_enabled {
                s_ef_residual.push(gpu.zeros(&[s_size], DType::F16)?);
            }
            conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok(Self {
            s_matrices,
            s_scales,
            conv_states,
            s_ef_residual,
            quant,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_ef_residual {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Reset all DeltaNet recurrent buffers to zero in place. Lets callers
    /// reuse a single `DeltaNetState` across independent chunks/sequences
    /// without allocating per chunk (which leaks since DeltaNetState has no
    /// Drop). Mirrors `ModelSlot::reset_state` in speculative.rs.
    ///
    /// Returns `Err` on the first HIP memset/memset_async failure so production
    /// rollback can attest `rolled_back:false`.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for s in &self.s_matrices {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.s_scales {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.conv_states {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.s_ef_residual {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
            }
            None => {
                for s in &self.s_matrices {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.s_scales {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.conv_states {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.s_ef_residual {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
            }
        }
        Ok(())
    }

    /// Multi-GPU companion to `new_with_quant`. Each LA-layer's state is
    /// allocated on the device that owns the layer in the multi-GPU band
    /// split: `gpus.devices[gpus.device_for_layer(orig_layer_idx)]` for the
    /// `orig_layer_idx` of the LA-layer. Returns the state alongside the
    /// `la_to_device` mapping the daemon needs to route reset memsets to
    /// the correct device.
    pub fn new_with_quant_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<(Self, Vec<u8>)> {
        let s_dim = config.linear_key_head_dim;
        let n_heads = config.linear_num_value_heads;
        let s_size = n_heads * s_dim * s_dim;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::new();
        let mut s_scales = Vec::new();
        let mut conv_states = Vec::new();
        let mut la_to_device: Vec<u8> = Vec::new();

        for (orig_layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != LayerType::LinearAttention {
                continue;
            }
            let dev_idx = gpus.device_for_layer(orig_layer_idx);
            la_to_device.push(dev_idx as u8);
            let g = &mut gpus.devices[dev_idx];
            // g.hip.malloc/memset bypass the Stage 2 bind_thread audit
            // (HipRuntime methods don't carry a device id). Bind explicitly
            // before any raw HIP ops so allocations land on the right device.
            g.bind_thread()?;
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(g.zeros(&[s_size], DType::F32)?);
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    let buf = g.hip.malloc(s_size)?;
                    g.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    let buf = g.hip.malloc(s_size / 2)?;
                    g.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            conv_states.push(g.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok((
            Self {
                s_matrices,
                s_scales,
                conv_states,
                // EF residual not wired for the multi-GPU band split (would need
                // per-device residual alloc routed by device_for_layer); empty ⇒
                // ef_residual() returns None ⇒ kernel uses the stochastic path.
                s_ef_residual: Vec::new(),
                quant,
            },
            la_to_device,
        ))
    }

    /// Free per-LA-layer tensors on the devices listed in `la_to_device`
    /// (the second tuple element returned by `new_with_quant_multi`).
    pub fn free_gpu_multi(self, gpus: &mut Gpus, la_to_device: &[u8]) {
        for (i, t) in self.s_matrices.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        // Empty today (multi-GPU EF not wired); free if/when residuals land.
        for (i, t) in self.s_ef_residual.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
    }
}

// ─── Weight loading ─────────────────────────────────────────────────────

fn qwen35_tensor_name_candidates(name: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(4);
    let mut push = |s: String| {
        if !out.iter().any(|x| x == &s) {
            out.push(s);
        }
    };

    if name == "lm_head.weight" {
        push(name.to_string());
        push("model.language_model.lm_head.weight".to_string());
        push("model.lm_head.weight".to_string());
        return out;
    }

    if name.starts_with("model.") {
        push(name.to_string());
    } else {
        push(format!("model.language_model.{name}"));
        push(format!("model.{name}"));
        push(name.to_string());
    }
    out
}

fn qwen35_tensor_data_vec<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> Option<(&'a HfqTensorInfo, Vec<u8>)> {
    for candidate in qwen35_tensor_name_candidates(name) {
        if let Some(found) = hfq.tensor_data_vec(&candidate) {
            return Some(found);
        }
    }
    None
}

fn load_norm_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));
    dequant_norm(gpu, info.quant_type, &data, shape, QWEN35_NORM_BIAS)
}

fn load_weight_tensor_raw(
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    match quant_type {
        6 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        7 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        8 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        11 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        12 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            // MQ4-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        14 => {
            // MQ8-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        15 => {
            // MQ6-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            // MQ3-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        18 => {
            // MQ2-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        19 => {
            // MQ2-G256-Lloyd — 2-bit + 4-entry fp16 codebook (72 bytes/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        20 => {
            // MQ3-G256-Lloyd — 3-bit + 8-entry fp16 codebook (112 bytes/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        30 => {
            // MQ4-G256-Lloyd — 4-bit + 16-entry fp16 codebook (160 bytes/group)
            // Renumbered from qt 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        31 => {
            // MQ5-G256 — MagnumQuant FWHT-rotated 5-bit (168 bytes/group, 5.25 bpw).
            // Opaque raw buffer, same pattern as MQ4(13)/MQ6(15); the GEMV
            // dispatch FWHT-rotates x at use. AWQ sidecar attached by the
            // caller via DType::supports_awq_sidecar (already includes MQ5G256).
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ5G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        21 => {
            // HFP4G32 — E2M1 + UE8M0 g32 + FP16 row scale. See docs/quant-formats/hfp4.md.
            // K%256 — kernel constraint (gemv_hfp4g32 in dispatch.rs); refuse here so a
            // stale or externally-quantized file fails at load instead of panicking on
            // first dispatch.
            assert!(
                k % 256 == 0,
                "HFP4G32 v1 lm_head has K={k} but kernel requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        24 => {
            // MFP4G32 — HFP4G32 + offline FWHT. Drop-in MQ4 replacement; same byte
            // layout as qtype 21 with format_flags=0x05 stamped in the per-row hdr.
            assert!(
                k % 256 == 0,
                "MFP4G32 lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        32 => {
            // MFP4G32Lloyd lm_head: mfp4 rows + 32-B per-tensor fp16 codebook prefix.
            assert!(
                k % 256 == 0,
                "MFP4G32Lloyd lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        33 => {
            // MFP4G32P lm_head: mfp4+P — mfp4 rows with E4M3 per-block scale. NO prefix;
            // byte-identical layout to MFP4G32 (qt 24).
            assert!(
                k % 256 == 0,
                "MFP4G32P lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32P,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        34 => {
            // MFP4G32E8 lm_head: mfp4-E8 — mfp4+P container, NO prefix, same row_bytes;
            // per-32-block 16 E2M1 nibbles replaced by 4x32-bit E8-lattice codewords.
            assert!(
                k % 256 == 0,
                "MFP4G32E8 lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32E8,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        36 => {
            // MFP3G32E8: mfp4-E8 frame with 3-bit lattice, 13 B/blk, 3.25 bpw.
            // Drop-in cold tier for MQ3G256Lloyd (kernel tag 5).
            assert!(
                k % 256 == 0,
                "MFP3G32E8 has K={k} but FWHT requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP3G32E8,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        37 => {
            // MFP2G32E8: mfp4-E8 frame with 2-bit lattice, 9 B/blk, 2.25 bpw.
            // Drop-in cold tier for MQ2G256Lloyd (kernel tag 6).
            assert!(
                k % 256 == 0,
                "MFP2G32E8 has K={k} but FWHT requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP2G32E8,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        38 => {
            // MQ2-G256-GL — 2-bit codes vs the TENSOR-GLOBAL codebook GL_CB2 +
            // per-block fp16 scale. 2.0625 bpw. SoA, TWO regions, no per-group
            // header (this is the first format in the tree where that is true):
            //   [0 .. m*gpr*64)                  packed 2-bit indices, 64 B/group
            //   [m*gpr*64 .. + m*gpr*2)          fp16 per-block scales
            // with gpr = k/256. Opaque raw upload — the indexed MoE GEMVs
            // (gemv_mq2g256gl_moe_{gate_up,down}_indexed) decode it and receive
            // the codebook as scalar kernel args, so nothing here needs it.
            //
            // K%256 is a HARD requirement: gpr = k/256 truncates otherwise and
            // the scale-region base M*gpr*64 shifts, which decodes to plausible
            // garbage with no error. Fail at load.
            assert!(
                k % 256 == 0,
                "MQ2G256GL has K={k} but the SoA region split requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256GL,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        39 => {
            // MQ3-G256-GL — 3-bit sibling of qt 38 (global codebook GL_CB3,
            // 8 entries). 3.0625 bpw; 96 B of indices per group then the same
            // trailing m*gpr*2 B fp16 scale region.
            assert!(
                k % 256 == 0,
                "MQ3G256GL has K={k} but the SoA region split requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256GL,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        3 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => match f16_lm_head_mode_from_config() {
            F16LmHeadMode::Native => dequant_weight_raw(gpu, quant_type, data, m, k),
            F16LmHeadMode::F32 => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[m, k])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
        },
        2 => {
            // F32 — native full-precision oracle weights (qt=2). Raw f32 LE
            // bytes uploaded as-is; the engine forwards through gemv_f32 /
            // gemm_f32_batched / attention_f32. Part of the F1 native-bf16
            // reference path (no quantization).
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        16 => {
            // BF16 — widen losslessly to F32 on host, then upload as F32.
            // bf16 is the high 16 bits of an f32 (same sign/exp, 7 mantissa
            // bits), so `from_bits((bf16 as u32) << 16)` is exact. The engine
            // has no native bf16 GEMV for the text arch; the gfx942 bf16 MFMA
            // GEMM (kernels/src/gemm_bf16_mfma.gfx942.hip) is the perf path and
            // is documented as a deferred gap. F32 compute over bf16-rounded
            // weights is a superset-precision oracle.
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        35 => {
            // MFP4G32E8SOA lm_head: mfp4-E8 SoA layout for coalesced GEMV.
            assert!(
                k % 256 == 0,
                "MFP4G32E8SOA lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32E8SOA,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        _ => dequant_weight_raw(gpu, quant_type, data, m, k),
    }
}

/// Phase A Stage A — AWQ sidecar loader for the Qwen3.5 forward path.
///
/// The .hfq quantizer emits `<weight>.awq_scale.weight` (1D F16, length K)
/// alongside MQ4G256 weights that were AWQ pre-scaled. The dispatcher in
/// `fused_rmsnorm_rotate_for_mq` / `fused_rmsnorm_rotate_mq_batched_for`
/// looks at `WeightTensor.awq_scale.is_some()` to pick the AWQ-aware
/// kernel variant. WITHOUT this loader populating the field, every MQ4
/// weight ends up with `awq_scale: None`, the dispatcher falls through
/// to the non-AWQ kernel, and the math `(W·s) · (x/s) = W·x` breaks
/// because the runtime never divides by `s` — observed KLD blowup
/// 0.6721 → 13.4893 on 0.8B Qwen3.5 before this landed.
///
/// Lookup pattern matches `hipfire_runtime::hfq::load_awq_scale`:
/// strip trailing `.weight`, append `.awq_scale.weight`. Try both the
/// `model.language_model.`-prefixed name and the bare name (the qwen35
/// crate uses prefixed names; older sidecars or tests may use either).
/// TODO(transformer-extraction): cross-arch duplicate of
/// `hipfire-arch-qwen2::qwen2::load_weight_tensor` — same name-lookup +
/// pread + AWQ-sidecar pattern, but qwen35 uses the
/// `model.language_model.` prefix (its HFQ files put text weights under
/// the VL-friendly nested name) where qwen2 uses flat `model.{...}`.
/// Pull into `hipfire_runtime::transformer::weights` with the prefix
/// as a parameter during consolidation.
pub(crate) fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    // Use pread path to avoid page cache buildup on unified-memory APUs.
    #[cfg(unix)]
    {
        let mut wt: Option<WeightTensor> = None;
        let mut matched: Option<String> = None;
        for candidate in candidates(name) {
            if let Some((info, buf)) = hfq.tensor_data_pread(&candidate) {
                let qt = info.quant_type;
                wt = Some(load_weight_tensor_raw(gpu, qt, &buf, m, k)?);
                matched = Some(candidate);
                break;
            }
        }
        let mut wt = wt.unwrap_or_else(|| panic!("tensor not found: {name}"));
        // Phase A Stage A — populate awq_scale when the dtype is on
        // the AWQ allow-list (centralized at `DType::supports_awq_sidecar`).
        // The pread call invalidates the prior pread_buf borrow, but
        // the weight bytes have already been uploaded to GPU (owned by
        // `wt.buf`) so the borrow no longer matters.
        if wt.gpu_dtype.supports_awq_sidecar() {
            if let Some(matched_name) = matched.as_deref() {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, matched_name, k)
                    .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
            } else {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, name, k);
            }
        }
        return Ok(wt);
    }
    #[cfg(not(unix))]
    {
        let (info, data, matched_name) = {
            let mut found = None;
            for candidate in candidates(name) {
                if let Some((info, data)) = hfq.tensor_data(&candidate) {
                    found = Some((info, data, candidate));
                    break;
                }
            }
            found.unwrap_or_else(|| panic!("tensor not found: {name}"))
        };
        let mut wt = load_weight_tensor_raw(gpu, info.quant_type, data, m, k)?;
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale_for(hfq, gpu, &matched_name, k)
                .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
        }
        Ok(wt)
    }
}

/// REAP keep variant of [`load_weight_tensor`]: gather the tensor's first-axis
/// rows (one row per original expert) down to `keep` BEFORE quant decode, then
/// build the `WeightTensor` from the gathered bytes with `m = keep.len()`.
///
/// Only used for the MoE router (`mlp.gate.weight`, shape `[orig_experts, k]`)
/// under an active keep-map. `gather_rows` is exact for any row-independent
/// quant (every per-expert row is self-contained — its own scale/zero/codebook
/// live in the row), which is true for every quant_type this loader accepts.
/// `keep` MUST equal the compact slot order, and `m` MUST equal `keep.len()`.
///
/// The AWQ sidecar (when present) is indexed by `k` (the input/hidden
/// dimension), shared across all expert rows, so it is loaded UNCHANGED —
/// row selection does not touch it.
fn load_weight_tensor_keep(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    keep: &[u32],
) -> HipResult<WeightTensor> {
    debug_assert_eq!(
        m,
        keep.len(),
        "load_weight_tensor_keep: m ({m}) must equal keep.len() ({})",
        keep.len()
    );
    // Resolve via the shared `qwen35_tensor_data_vec` helper (same candidate
    // logic as the non-keep path; it preads + fadvise_dontneeds internally and
    // returns OWNED bytes, so the gather + AWQ-sidecar reads don't fight a
    // borrow). `orig_rows` is the on-disk first-axis length = original expert
    // count. The matched (prefixed) candidate name is resolved separately for
    // the AWQ sidecar lookup via a metadata-only existence check, since the
    // helper doesn't surface which candidate it hit.
    let (info, bytes) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));
    let quant_type = info.quant_type;
    let orig_rows = *info.shape.first().unwrap_or(&0) as usize;
    // Row-gather to the kept set. The on-disk row count is the ORIGINAL expert
    // count (= bytes.len() / rowstride); gather_rows derives it from shape[0].
    let (_new_shape, sub) = hipfire_reap::gather::gather_rows(&[orig_rows], &bytes, keep)
        .map_err(|e| HipError::new(0, &format!("qwen35: router row-gather '{name}': {e}")))?;
    let mut wt = load_weight_tensor_raw(gpu, quant_type, &sub, m, k)?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        // Resolve the matched candidate name (metadata only) so the AWQ sidecar
        // is looked up under the same prefix the weight resolved to; fall back
        // to the bare `name`. Mirrors the non-keep `load_weight_tensor`.
        let matched = qwen35_tensor_name_candidates(name)
            .into_iter()
            .find(|c| hfq.find_tensor_info(c).is_some());
        wt.awq_scale = matched
            .as_deref()
            .and_then(|mn| load_awq_scale_for(hfq, gpu, mn, k))
            .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
    }
    Ok(wt)
}

// ─── ParoQuant AWQ → HFQ4G128 repack ────────────────────────────────────────

/// Repack AWQ-format INT4 weights into HFQ4G128 inline layout.
///
/// AWQ layout (3 separate tensors):
///   qweight: I32 [in_dim, out_dim/8] — 8 nibbles per I32
///   qzeros:  I32 [in_dim/group_size, out_dim/8] — 8 zero-point nibbles per I32
///   scales:  F16 [in_dim/group_size, out_dim] — per-group scales
///
/// HFQ4G128 layout (per output row, one contiguous buffer):
///   For each group of 128 input elements:
///     [f32 scale (4B)][f32 zero (4B)][64B packed nibbles] = 72 bytes
///
/// Returns: Vec<u8> in HFQ4G128 format, ready for gpu.upload_raw.
///
/// SYNC: must match `repack_awq_to_hfq4g128` in
/// `crates/hipfire-runtime/src/hfq.rs`. Duplicated to avoid a cross-crate
/// dependency cycle (hipfire-arch-qwen35 -> hipfire-runtime); keep the two
/// bodies byte-identical when editing.
fn repack_awq_to_hfq4g128(
    qweight: &[u8],    // I32 raw bytes
    qzeros: &[u8],     // I32 raw bytes
    scales: &[u8],     // F16 raw bytes
    out_dim: usize,    // M (output features)
    in_dim: usize,     // K (input features)
    group_size: usize, // 128
) -> Vec<u8> {
    let groups_per_row = in_dim / group_size;
    let bytes_per_row = groups_per_row * 72;
    let mut out = vec![0u8; out_dim * bytes_per_row];

    // Parse qweight as &[u32] (LE)
    debug_assert_eq!(
        qweight.as_ptr() as usize % 4,
        0,
        "AWQ qweight not 4-byte aligned"
    );
    let qw: &[u32] =
        unsafe { std::slice::from_raw_parts(qweight.as_ptr() as *const u32, qweight.len() / 4) };
    // qweight shape: [in_dim, out_dim/8] → row-major
    let qw_cols = out_dim / 8;

    // Parse qzeros as &[u32]
    debug_assert_eq!(
        qzeros.as_ptr() as usize % 4,
        0,
        "AWQ qzeros not 4-byte aligned"
    );
    let qz: &[u32] =
        unsafe { std::slice::from_raw_parts(qzeros.as_ptr() as *const u32, qzeros.len() / 4) };
    // qzeros shape: [in_dim/group_size, out_dim/8]
    let qz_cols = out_dim / 8;

    // Parse scales as &[u16] (F16)
    debug_assert_eq!(
        scales.as_ptr() as usize % 2,
        0,
        "AWQ scales not 2-byte aligned"
    );
    let sc: &[u16] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u16, scales.len() / 2) };
    // scales shape: [in_dim/group_size, out_dim]

    // AWQ nibble reorder: ParoQuant packs with _AWQ_REORDER=(0,2,4,6,1,3,5,7).
    // To extract element m, use the inverse permutation:
    const AWQ_DEQUANT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

    for m in 0..out_dim {
        for g in 0..groups_per_row {
            let row_off = m * bytes_per_row + g * 72;

            let scale_f16 = sc[g * out_dim + m];
            let scale_f32 = f16_to_f32(scale_f16);

            let zero_i32 = qz[g * qz_cols + m / 8];
            let zero_nibble = ((zero_i32 >> (AWQ_DEQUANT[m % 8] * 4)) & 0xF) as f32;
            let zero_f32 = -scale_f32 * zero_nibble;

            out[row_off..row_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
            out[row_off + 4..row_off + 8].copy_from_slice(&zero_f32.to_le_bytes());

            let nibble_shift = AWQ_DEQUANT[m % 8] * 4;
            let qw_col = m / 8;
            for i in 0..64 {
                let in_idx0 = g * group_size + i * 2;
                let in_idx1 = in_idx0 + 1;

                let nib0 = ((qw[in_idx0 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;
                let nib1 = ((qw[in_idx1 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;

                // HFQ4G128: lo nibble = even element, hi nibble = odd element
                out[row_off + 8 + i] = nib0 | (nib1 << 4);
            }
        }
    }

    out
}

/// Load a ParoQuant-quantized weight from a SafetensorsSource.
/// Repacks AWQ INT4 → HFQ4G128 and uploads rotation metadata.
fn load_paroquant_weight(
    source: &dyn ModelSource,
    gpu: &Gpu,
    tensor_prefix: &str, // e.g. "model.language_model.layers.0.mlp.gate_proj"
    out_dim: usize,      // M
    in_dim: usize,       // K
    group_size: u32,
    krot: u8,
) -> HipResult<WeightTensor> {
    let qw_name = format!("{tensor_prefix}.qweight");
    let qz_name = format!("{tensor_prefix}.qzeros");
    let sc_name = format!("{tensor_prefix}.scales");
    let pairs_name = format!("{tensor_prefix}.pairs");
    let theta_name = format!("{tensor_prefix}.theta");
    let cs_name = format!("{tensor_prefix}.channel_scales");

    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {sc_name}")))?;

    // Repack AWQ → HFQ4G128
    let hfq_data = repack_awq_to_hfq4g128(
        qw_data,
        qz_data,
        sc_data,
        out_dim,
        in_dim,
        group_size as usize,
    );
    let buf = gpu.upload_raw(&hfq_data, &[hfq_data.len()])?;

    // Load rotation metadata
    let (_, pairs_data) = source
        .tensor_data(&pairs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {pairs_name}")))?;
    let (_, theta_data) = source
        .tensor_data(&theta_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {theta_name}")))?;
    let (_, cs_data) = source
        .tensor_data(&cs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {cs_name}")))?;

    let pairs = gpu.upload_raw(pairs_data, &[pairs_data.len()])?;
    let theta = gpu.upload_raw(theta_data, &[theta_data.len()])?;
    let channel_scales = gpu.upload_raw(cs_data, &[cs_data.len()])?;

    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::ParoQ4G128,
        m: out_dim,
        k: in_dim,
        row_stride: 0,
        paro: Some(ParoRotation {
            pairs,
            theta,
            channel_scales,
            krot: krot as u32,
            group_size,
            is_alias: false,
        }),
        awq_scale: None,
    })
}

/// Load an FP16 weight and encode it into MQ4G128 byte layout at load time.
/// Used by `paro_load_wt` for LinearAttention `in_proj_a` / `in_proj_b` weights
/// (alpha/beta) when the PARO checkpoint doesn't include them in the calibrated
/// set AND the per-arch/env gating chose the MQ4G128 path.
///
/// At decode time, the weight routes through `gemv_mq4g128_prerotated` which
/// applies FWHT-128 to the activation (via `rotate_x_mq_128_for`) before the
/// inner GEMV. Encoder applies FWHT-128 to weight with the same sign tables,
/// so the two FWHTs orthogonally cancel.

/// Load an FP16 weight tensor from safetensors (for excluded/unquantized layers).
fn load_fp16_weight_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[m, k])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

// ─── ParoQuant MoE expert loading (Option A — per-expert qweight, shared sidecars) ──

/// Repack a single per-expert AWQ projection (gate, up, or down) into HFQ4G128
/// byte rows. Returns the row-major byte buffer (size `out_dim * groups_per_row * 72`).
///
/// Caller is responsible for uploading the buffer to GPU (or concatenating with
/// another projection's rows before upload — gate||up fusion path).
fn paro_repack_moe_projection(
    source: &dyn ModelSource,
    full_prefix: &str, // e.g. "model.language_model.layers.0.mlp.experts.5.gate_proj"
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> HipResult<Vec<u8>> {
    let qw_name = format!("{full_prefix}.qweight");
    let qz_name = format!("{full_prefix}.qzeros");
    let sc_name = format!("{full_prefix}.scales");
    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {sc_name}")))?;
    Ok(repack_awq_to_hfq4g128(
        qw_data, qz_data, sc_data, out_dim, in_dim, group_size,
    ))
}

/// Upload the per-layer shared PARO rotation sidecars (one tuple for gate||up,
/// one for down). All 256 experts will reference these via non-owning
/// `ParoRotation` aliases.
///
/// Shisa-ai's PARO checkpoint stores these at:
///   `model.language_model.layers.{L}.mlp.experts.{gate_up,down}_weight_{pairs,theta,channel_scales}`
fn paro_load_moe_shared_sidecars(
    source: &dyn ModelSource,
    gpu: &Gpu,
    p: &str, // e.g. "layers.0"
) -> HipResult<MoeParoSidecars> {
    let mp = paro_text_prefix(source)?;
    let base = format!("{mp}.{p}.mlp.experts");
    let load = |name: &str| -> HipResult<GpuTensor> {
        let full = format!("{base}.{name}");
        let (_, data) = source.tensor_data(&full).ok_or_else(|| {
            HipError::new(
                0,
                &format!("ParoQuant MoE shared sidecar not found: {full}"),
            )
        })?;
        gpu.upload_raw(data, &[data.len()])
    };
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant: quant_config required"))?;
    Ok(MoeParoSidecars {
        gate_up_pairs: load("gate_up_weight_pairs")?,
        gate_up_theta: load("gate_up_weight_theta")?,
        gate_up_channel_scales: load("gate_up_weight_channel_scales")?,
        down_pairs: load("down_weight_pairs")?,
        down_theta: load("down_weight_theta")?,
        down_channel_scales: load("down_weight_channel_scales")?,
        krot: qc.krot as u32,
        group_size: qc.group_size,
    })
}

/// Build a non-owning `ParoRotation` whose tensor fields alias `src`'s
/// underlying GPU memory. The returned rotation must NOT outlive `src`;
/// callers store the owning `MoeParoSidecars` in `MoeFfnWeights.paro_shared`
/// to guarantee that.
fn alias_paro_rotation(
    pairs_src: &GpuTensor,
    theta_src: &GpuTensor,
    cs_src: &GpuTensor,
    krot: u32,
    group_size: u32,
) -> ParoRotation {
    let alias = |t: &GpuTensor| -> GpuTensor {
        GpuTensor {
            buf: unsafe { t.buf.alias() },
            shape: t.shape.clone(),
            dtype: t.dtype,
        }
    };
    ParoRotation {
        pairs: alias(pairs_src),
        theta: alias(theta_src),
        channel_scales: alias(cs_src),
        krot,
        group_size,
        is_alias: true,
    }
}

/// Load the full ParoQuant MoE FFN block for one layer:
///   - dense FP16 router (`mlp.gate.weight [n_exp, hidden]`)
///   - dense FP16 shared-expert scalar gate (`mlp.shared_expert_gate.weight [1, hidden]`)
///   - shared expert (three per-projection PARO tensors: gate, up, down)
///   - 256 routed experts, each with a fused gate||up HFQ4G128 buffer + a down
///     HFQ4G128 buffer, all referencing layer-shared PARO sidecars
fn paro_load_moe_ffn(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    p: &str, // e.g. "layers.0"
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant MoE requires quant_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    let mp = paro_text_prefix(source)?;

    // ── Router (FP16 dense in shisa-ai's PARO checkpoint) ──
    // mlp.gate.weight is NOT PARO-quantized — only the expert FFN matmuls are.
    let router = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.gate.weight"),
        n_exp,
        dim,
    )?;

    // Scalar gate on the shared-expert add — also FP16 dense.
    let shared_expert_gate = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.shared_expert_gate.weight"),
        1,
        dim,
    )?;

    // ── Shared expert (its own per-projection PARO sidecars, no sharing) ──
    let shared_expert = SharedExpertWeights {
        gate: load_paroquant_weight(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        up: load_paroquant_weight(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        down: load_paroquant_weight(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj"),
            dim,
            smi,
            gs,
            kr,
        )?,
    };

    // ── Routed experts ──
    // shisa-ai stores per-expert qweight/qzeros/scales but ONE shared
    // pairs/theta/channel_scales tuple per projection-group (gate||up vs down)
    // for ALL experts in the layer. Upload sidecars once, alias into each
    // expert's WeightTensor.paro.
    let shared = paro_load_moe_shared_sidecars(source, gpu, p)?;

    let groups_per_row_hidden = dim / (gs as usize); // 2048/128 = 16
    let bytes_per_row_hidden = groups_per_row_hidden * 72; // 1152
    let groups_per_row_mi = mi / (gs as usize); // 512/128 = 4
    let bytes_per_row_mi = groups_per_row_mi * 72; // 288

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        // Per-expert prefixes (full dot-path is constructed inside the helper).
        let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
        let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
        let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");

        // Fuse gate || up at HFQ4G128 row level: each row is independent
        // (`bytes_per_row` bytes, no cross-row state), so concat works.
        // Final shape: [2*mi, dim], rows [0..mi] = gate, rows [mi..2*mi] = up.
        let gate_bytes = paro_repack_moe_projection(source, &gate_prefix, mi, dim, gs as usize)?;
        let up_bytes = paro_repack_moe_projection(source, &up_prefix, mi, dim, gs as usize)?;
        debug_assert_eq!(gate_bytes.len(), mi * bytes_per_row_hidden);
        debug_assert_eq!(up_bytes.len(), mi * bytes_per_row_hidden);
        let mut gate_up_bytes = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        gate_up_bytes.extend_from_slice(&gate_bytes);
        gate_up_bytes.extend_from_slice(&up_bytes);
        let gate_up_buf = gpu.upload_raw(&gate_up_bytes, &[gate_up_bytes.len()])?;

        let down_bytes = paro_repack_moe_projection(source, &down_prefix, dim, mi, gs as usize)?;
        debug_assert_eq!(down_bytes.len(), dim * bytes_per_row_mi);
        let down_buf = gpu.upload_raw(&down_bytes, &[down_bytes.len()])?;

        let gate_up = WeightTensor {
            buf: gate_up_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: 2 * mi,
            k: dim,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.gate_up_pairs,
                &shared.gate_up_theta,
                &shared.gate_up_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        let down = WeightTensor {
            buf: down_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: dim,
            k: mi,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.down_pairs,
                &shared.down_theta,
                &shared.down_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        experts.push(ExpertWeights { gate_up, down });
    }

    // ── Device-side expert pointer tables (mirrors load_moe_ffn) ──
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;

    Ok(MoeFfnWeights {
        router,
        experts,
        packed_expert_owners: None,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        // ParoQuant routed experts use shared per-layer Givens sidecars, not
        // per-expert MQ4 AWQ scales — no MoE-AWQ table.
        expert_down_awq_ptrs: None,
        // Paged/paro layers are uniform-dtype — no per-expert mixed table.
        expert_dtype_tags: None,
        layer_idx,
        expert_shape: None,
        paro_shared: Some(shared),
    })
}

// ─── Standard HFQ loading ───────────────────────────────────────────────────

/// Load a tensor as F32 on GPU, handling any quant type by dequanting on CPU.
fn load_any_as_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => hipfire_runtime::llama::dequantize_q8_0(&data, n),
        14 => {
            // MQ8-G256: [f16 scale][int8 × 256] = 258 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = hipfire_runtime::llama::f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                // Inverse FWHT to recover original values
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let inv_s = 0.0625;
                for i in 0..256 {
                    group[i] *= inv_s * signs1[i];
                }
            }
            out
        }
        6 | 7 | 13 | 15 => {
            // HFQ4-G256 or G128 or MQ4-G256 or MQ6-G256 — CPU dequant
            // MQ4/MQ6 store rotated weights. For small tensors loaded here,
            // we dequant then inverse-rotate to recover the original values.
            let is_6bit = info.quant_type == 15;
            let group_size: usize =
                if info.quant_type == 6 || info.quant_type == 13 || info.quant_type == 15 {
                    256
                } else {
                    128
                };
            let bytes_per_group = if is_6bit { 200 } else { 8 + group_size / 2 };
            let n_groups = data.len() / bytes_per_group;
            let is_mq = info.quant_type == 13 || info.quant_type == 15;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let (signs1, signs2) = if is_mq {
                (
                    Some(hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256)),
                    Some(hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256)),
                )
            } else {
                (None, None)
            };
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_6bit {
                    for i in (0..group_size).step_by(4) {
                        let bo = off + 8 + (i / 4) * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        out.push(scale * ((b0 & 0x3F) as f32) + zero);
                        out.push(scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero);
                        out.push(scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero);
                        out.push(scale * (((b2 >> 2) & 0x3F) as f32) + zero);
                    }
                } else {
                    for i in 0..group_size {
                        let byte_idx = i / 2;
                        let byte_val = data[off + 8 + byte_idx];
                        let nibble = if i % 2 == 0 {
                            byte_val & 0xF
                        } else {
                            byte_val >> 4
                        };
                        out.push(scale * nibble as f32 + zero);
                    }
                }
                // Inverse FWHT for MQ4/MQ6: recover original weight values
                if is_mq && group_size == 256 {
                    let s1 = signs1.as_ref().unwrap();
                    let s2 = signs2.as_ref().unwrap();
                    let group = &mut out[start..start + 256];
                    // Inverse FWHT: signs2 → butterfly → scale → signs1
                    for i in 0..256 {
                        group[i] *= s2[i];
                    }
                    let mut stride = 1;
                    while stride < 256 {
                        let mut j = 0;
                        while j < 256 {
                            for k in 0..stride {
                                let a = group[j + k];
                                let b = group[j + k + stride];
                                group[j + k] = a + b;
                                group[j + k + stride] = a - b;
                            }
                            j += stride * 2;
                        }
                        stride <<= 1;
                    }
                    let scale_inv = 0.0625; // 1/sqrt(256)
                    for i in 0..256 {
                        group[i] *= scale_inv * s1[i];
                    }
                }
            }
            out
        }
        8 => {
            // HFQ6-G256 — CPU dequant: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 200; // 8 + 192
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                // 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
                for i in (0..group_size).step_by(4) {
                    let byte_off = 8 + (i / 4) * 3;
                    let b0 = data[off + byte_off] as u32;
                    let b1 = data[off + byte_off + 1] as u32;
                    let b2 = data[off + byte_off + 2] as u32;
                    let q0 = (b0 & 0x3F) as f32;
                    let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                    let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                    let q3 = ((b2 >> 2) & 0x3F) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                }
            }
            out
        }
        11 => {
            // HFQ3-G256: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 104;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                // 8 values per 3 bytes (matching kernel unpack)
                for chunk in 0..32 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        12 => {
            // HFQ3-G128: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights
            let group_size: usize = 128;
            let bytes_per_group: usize = 56;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                for chunk in 0..16 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        20 => {
            // MQ3-G256-Lloyd (qt 20, 112 B/group): 8 fp16 codebook entries + 3-bit
            // indices (cross-byte, 32 chunks × 3 bytes × 8 weights). Decode is
            // direct lookup `cb[idx]` then inverse FWHT for CPU consumers.
            let group_size: usize = 256;
            let bytes_per_group: usize = 112;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 8];
                for k in 0..8 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::llama::f16_to_f32(bits);
                }
                let start = out.len();
                for chunk in 0..32 {
                    let bo = off + 16 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as usize;
                    let q1 = ((b0 >> 3) & 7) as usize;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as usize;
                    let q3 = ((b1 >> 1) & 7) as usize;
                    let q4 = ((b1 >> 4) & 7) as usize;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as usize;
                    let q6 = ((b2 >> 2) & 7) as usize;
                    let q7 = ((b2 >> 5) & 7) as usize;
                    out.push(cb[q0]);
                    out.push(cb[q1]);
                    out.push(cb[q2]);
                    out.push(cb[q3]);
                    out.push(cb[q4]);
                    out.push(cb[q5]);
                    out.push(cb[q6]);
                    out.push(cb[q7]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        19 => {
            // MQ2-G256-Lloyd (qt 19, 72 B/group): 4 fp16 codebook entries + 2-bit indices.
            // Decode is direct lookup `cb[idx]`, then inverse FWHT to recover original
            // pre-rotation values for CPU consumers (DeltaNet conv1d).
            let group_size: usize = 256;
            let bytes_per_group: usize = 72;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 4];
                for k in 0..4 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::llama::f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..64 {
                    let byte_val = data[off + 8 + i] as usize;
                    out.push(cb[byte_val & 3]);
                    out.push(cb[(byte_val >> 2) & 3]);
                    out.push(cb[(byte_val >> 4) & 3]);
                    out.push(cb[(byte_val >> 6) & 3]);
                }
                // Inverse FWHT to recover pre-rotation weights — same butterfly as the
                // MQ3/MQ2 arm below.
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        30 => {
            // MQ4-G256-Lloyd (qt 30, 160 B/group): 16 fp16 codebook entries (bytes [0..32))
            // + 4-bit packed indices (bytes [32..160), low nibble = idx[2i], high = idx[2i+1]).
            // Decode is direct lookup `cb[idx]` then inverse FWHT for CPU consumers.
            // Renumbered from qt 21 → 30 to avoid HFP4G32=21 collision.
            let group_size: usize = 256;
            let bytes_per_group: usize = 160;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 16];
                for k in 0..16 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::llama::f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..128 {
                    let byte_val = data[off + 32 + i] as usize;
                    out.push(cb[byte_val & 0xF]);
                    out.push(cb[(byte_val >> 4) & 0xF]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        17 | 18 => {
            // MQ3-G256 (qt 17, 104 B/group, 3-bit) or MQ2-G256 (qt 18, 72 B/group, 2-bit).
            // Both store FWHT-rotated weights — dequant then inverse-rotate to recover
            // original values for CPU consumers (e.g., DeltaNet conv1d).
            let is_mq3 = info.quant_type == 17;
            let group_size: usize = 256;
            let bytes_per_group: usize = if is_mq3 { 104 } else { 72 };
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_mq3 {
                    // 8 values per 3 bytes (matches gemv_hfq3g256.hip unpack).
                    for chunk in 0..32 {
                        let bo = off + 8 + chunk * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        let q0 = (b0 & 7) as f32;
                        let q1 = ((b0 >> 3) & 7) as f32;
                        let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                        let q3 = ((b1 >> 1) & 7) as f32;
                        let q4 = ((b1 >> 4) & 7) as f32;
                        let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                        let q6 = ((b2 >> 2) & 7) as f32;
                        let q7 = ((b2 >> 5) & 7) as f32;
                        out.push(scale * q0 + zero);
                        out.push(scale * q1 + zero);
                        out.push(scale * q2 + zero);
                        out.push(scale * q3 + zero);
                        out.push(scale * q4 + zero);
                        out.push(scale * q5 + zero);
                        out.push(scale * q6 + zero);
                        out.push(scale * q7 + zero);
                    }
                } else {
                    // MQ2: 4 values per byte (matches gemv_hfq2g256.hip unpack).
                    for i in 0..64 {
                        let byte_val = data[off + 8 + i] as u32;
                        out.push(scale * ((byte_val & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 2) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 4) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 6) & 3) as f32) + zero);
                    }
                }
                // Inverse FWHT: recover original (pre-rotation) weight values.
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625; // 1/sqrt(256)
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        32 => {
            // MFP4G32Lloyd (qt 32): [32-B fp16 codebook prefix][M rows].
            // Each row = 16-B header + (K/32)*17 B blocks (UE8M0 + nibbles).
            // Recon: value = row_scale_a * 2^(block_e-127) * cb[nibble].
            // Returns rotated-domain f32 (weights stored pre-FWHT-rotated).
            let row_bytes = 16 + 17 * (n / 32);
            let m_rows = if row_bytes > 0 {
                (data.len().saturating_sub(32)) / row_bytes
            } else {
                0
            };
            let mut cb = [0.0f32; 16];
            for i in 0..16 {
                let bits = u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
                cb[i] = hipfire_runtime::llama::f16_to_f32(bits);
            }
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = 32 + r * (16 + n_blocks * 17);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * 17;
                    let block_e = data[po] as i32;
                    let scale = row_scale_a * ((block_e - 127) as f32).exp2();
                    for i in 0..16 {
                        let byte = data[po + 1 + i];
                        out[r * k_row + b * 32 + 2 * i] = scale * cb[(byte & 0x0F) as usize];
                        out[r * k_row + b * 32 + 2 * i + 1] =
                            scale * cb[((byte >> 4) & 0x0F) as usize];
                    }
                }
            }
            out
        }
        33 => {
            // MFP4G32P (qt 33): mfp4 rows (NO prefix) with E4M3 (FP8) per-block scale.
            // Recon: value = row_scale_a * e4m3_decode(scale_byte) * E2M1_LUT[nibble].
            // Returns rotated-domain f32 (weights stored pre-FWHT-rotated).
            const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            #[inline]
            fn e2m1(n: u8) -> f32 {
                let m = E2M1_MAG[(n & 0x7) as usize];
                if (n & 0x8) != 0 {
                    -m
                } else {
                    m
                }
            }
            // E4M3 (unsigned scale, bias 7, 3 mantissa) — bit-identical to the
            // quantizer `e4m3_scale_decode` and the gfx942 kernel decode.
            #[inline]
            fn e4m3(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            let row_bytes = 16 + 17 * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * 17);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * 17;
                    let scale = row_scale_a * e4m3(data[po]);
                    for i in 0..16 {
                        let byte = data[po + 1 + i];
                        out[r * k_row + b * 32 + 2 * i] = scale * e2m1(byte & 0x0F);
                        out[r * k_row + b * 32 + 2 * i + 1] = scale * e2m1((byte >> 4) & 0x0F);
                    }
                }
            }
            out
        }
        34 => {
            // MFP4G32E8 (qt 34): mfp4+P container, NO prefix, with E8-lattice codewords.
            // Identical framing to qt 33 (E4M3 block scale, row_scale f16); per block
            // decode 4 E8 codewords instead of 32 E2M1 nibbles.
            // E8 decode — bit-identical to e8.rs::decode_index + * QUANT_STEP.
            #[inline]
            fn e4m3_e8(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            #[inline]
            fn e8_decode(idx: u32, coord: usize) -> f32 {
                // Decode a single coord of an E8 lattice point from a u32 index.
                // Matches kernel e8_decode_index exactly.
                let coset = (idx >> 31) & 1;
                let e: u32;
                if coord < 7 {
                    e = (idx >> (4 * coord as u32)) & 0xF;
                } else {
                    // coord == 7: recover from parity
                    let mut sl: u32 = 0;
                    for i in 0..7 {
                        sl += (idx >> (4 * i)) & 0xF;
                    }
                    let e7h = (idx >> 28) & 0x7;
                    let p7 = e7h << 1;
                    let lsb = (sl + p7) & 1;
                    e = p7 | lsb;
                }
                let c = (e as i32 - 7) as f32;
                if coset == 1 {
                    c + 0.5
                } else {
                    c
                }
            }
            const QUANT_STEP: f32 = 0.88;
            let row_bytes = 16 + 17 * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * 17);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * 17;
                    let scale = row_scale_a * e4m3_e8(data[po]) * QUANT_STEP;
                    for g in 0..4usize {
                        let idx = u32::from_le_bytes([
                            data[po + 1 + g * 4],
                            data[po + 2 + g * 4],
                            data[po + 3 + g * 4],
                            data[po + 4 + g * 4],
                        ]);
                        for i in 0..8usize {
                            out[r * k_row + b * 32 + g * 8 + i] = scale * e8_decode(idx, i);
                        }
                    }
                }
            }
            out
        }
        35 => {
            // MFP4G32E8SOA (qt 35): same E8 data as qt 34 but in SoA layout.
            // Per-row: [16B hdr] + [n_blocks bytes E4M3 scales, pad16] + [n_blocks*16 bytes codewords].
            #[inline]
            fn e4m3_e8_soa(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            #[inline]
            fn e8_decode_soa(idx: u32, coord: usize) -> f32 {
                let coset = (idx >> 31) & 1;
                let e: u32;
                if coord < 7 {
                    e = (idx >> (4 * coord as u32)) & 0xF;
                } else {
                    let mut sl: u32 = 0;
                    for i in 0..7 {
                        sl += (idx >> (4 * i)) & 0xF;
                    }
                    let e7h = (idx >> 28) & 0x7;
                    let p7 = e7h << 1;
                    let lsb = (sl + p7) & 1;
                    e = p7 | lsb;
                }
                let c = (e as i32 - 7) as f32;
                if coset == 1 {
                    c + 0.5
                } else {
                    c
                }
            }
            const QUANT_STEP_SOA: f32 = 0.88;
            // Decode assuming n = k_row; figure out m_rows from total bytes.
            // n_blocks = n/32; scale_padded = ceil(n_blocks/16)*16
            let n_blocks = n / 32;
            let scale_padded = ((n_blocks + 15) >> 4) << 4;
            let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
            let m_rows = if soa_row_bytes > 0 {
                data.len() / soa_row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks2 = k_row / 32;
            let scale_padded2 = ((n_blocks2 + 15) >> 4) << 4;
            let row_bytes2 = 16 + scale_padded2 + n_blocks2 * 16;
            for r in 0..m_rows {
                let base = r * row_bytes2;
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                let scale_arr = &data[base + 16..base + 16 + n_blocks2];
                let cw_arr =
                    &data[base + 16 + scale_padded2..base + 16 + scale_padded2 + n_blocks2 * 16];
                for b in 0..n_blocks2 {
                    let scale = row_scale_a * e4m3_e8_soa(scale_arr[b]) * QUANT_STEP_SOA;
                    for g in 0..4usize {
                        let co = b * 16 + g * 4;
                        let idx = u32::from_le_bytes([
                            cw_arr[co],
                            cw_arr[co + 1],
                            cw_arr[co + 2],
                            cw_arr[co + 3],
                        ]);
                        for i in 0..8usize {
                            out[r * k_row + b * 32 + g * 8 + i] = scale * e8_decode_soa(idx, i);
                        }
                    }
                }
            }
            out
        }
        36 => {
            // MFP3G32E8 (qt 36): mfp4-E8 frame with 3-bit lattice, 13 B/blk, center 4.
            // Decode mirrors kernel mfp3_decode_index: nibbles 3b, coset bit 23, e7_high 2b@21.
            #[inline]
            fn e4m3_mfp3(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            const MFP3_BLOCK: usize = 13; // 1 + 4*3
            const MFP3_STEP: f32 = 1.8; // MSE-tuned; matches QUANT_STEP_MFP3 in e8.rs + kernels
            let row_bytes = 16 + MFP3_BLOCK * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * MFP3_BLOCK);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * MFP3_BLOCK;
                    let scale = row_scale_a * e4m3_mfp3(data[po]) * MFP3_STEP;
                    for g in 0..4usize {
                        let cw_off = po + 1 + g * 3;
                        // 3-byte narrow read (mfp3 codeword is only 24 bits)
                        let idx: u32 = (data[cw_off] as u32)
                            | ((data[cw_off + 1] as u32) << 8)
                            | ((data[cw_off + 2] as u32) << 16);
                        // mfp3_decode_index: 3-bit nibbles, center 4, coset bit 23, e7_high @21 (2b)
                        let coset = (idx >> 23) & 1;
                        let mut e = [0u32; 8];
                        let mut sl: u32 = 0;
                        for i in 0..7 {
                            e[i] = (idx >> (3 * i as u32)) & 0x7;
                            sl += e[i];
                        }
                        let e7_high = (idx >> 21) & 0x3;
                        let p7 = e7_high << 1;
                        e[7] = p7 | ((sl + p7) & 1);
                        for i in 0..8usize {
                            let c = (e[i] as i32 - 4) as f32;
                            let coord = if coset == 1 { c + 0.5 } else { c };
                            out[r * k_row + b * 32 + g * 8 + i] = scale * coord;
                        }
                    }
                }
            }
            out
        }
        37 => {
            // MFP2G32E8 (qt 37): mfp4-E8 frame with 2-bit lattice, 9 B/blk, center 2.
            // Decode mirrors kernel mfp2_decode_index: nibbles 2b, coset bit 15, e7_high 1b@14.
            #[inline]
            fn e4m3_mfp2(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            const MFP2_BLOCK: usize = 9; // 1 + 4*2
            const MFP2_STEP: f32 = 3.8; // MSE-tuned; matches QUANT_STEP_MFP2 in e8.rs + kernels
            let row_bytes = 16 + MFP2_BLOCK * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * MFP2_BLOCK);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * MFP2_BLOCK;
                    let scale = row_scale_a * e4m3_mfp2(data[po]) * MFP2_STEP;
                    for g in 0..4usize {
                        let cw_off = po + 1 + g * 2;
                        // 2-byte narrow read (mfp2 codeword is only 16 bits)
                        let idx: u32 = (data[cw_off] as u32) | ((data[cw_off + 1] as u32) << 8);
                        // mfp2_decode_index: 2-bit nibbles, center 2, coset bit 15, e7_high @14 (1b)
                        let coset = (idx >> 15) & 1;
                        let mut e = [0u32; 8];
                        let mut sl: u32 = 0;
                        for i in 0..7 {
                            e[i] = (idx >> (2 * i as u32)) & 0x3;
                            sl += e[i];
                        }
                        let e7_high = (idx >> 14) & 0x1;
                        let p7 = e7_high << 1;
                        e[7] = p7 | ((sl + p7) & 1);
                        for i in 0..8usize {
                            let c = (e[i] as i32 - 2) as f32;
                            let coord = if coset == 1 { c + 0.5 } else { c };
                            out[r * k_row + b * 32 + g * 8 + i] = scale * coord;
                        }
                    }
                }
            }
            out
        }
        _ => panic!("unsupported quant_type {} for {name}", info.quant_type),
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

/// Alias for load_any_as_f32.
fn load_raw_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    load_any_as_f32(hfq, gpu, name, n)
}

/// Loud model-load diagnostic for RDNA2 (gfx1030/1031/1032). Scans the model's
/// per-tensor `quant_type` bytes and, if running on RDNA2 with any RDNA3+-only
/// dtype present, prints a clear warning listing the offending formats. RDNA2 is
/// wave32 (so the WMMA-free MQ3-Lloyd scalar kernels now resolve — W4 fix), but
/// MQ5/MQ6/HFQ6 (HasMmq, RDNA3/4 only), the WMMA-only HFP4G32-fused prefill path,
/// and the E8-lattice formats have NO validated RDNA2 kernel and are best-effort.
///
/// Additive and arch-gated to RDNA2 — no effect on RDNA3/4/CDNA loads.
fn warn_rdna2_unvalidated_dtypes(hfq: &HfqFile, gpu: &Gpu) {
    if !gpu.arch_caps.is_rdna2() {
        return;
    }
    // (quant_type byte, human label) for dtypes with NO validated RDNA2 kernel.
    //   8=HFQ6G256, 15=MQ6G256, 31=MQ5G256  → HasMmq (RDNA3/4 only)
    //   24=MFP4G32, 33=MFP4G32P             → HFP4G32-fused (WMMA-only prefill)
    //   34/35=MFP4G32E8/SOA, 36=MFP3G32E8, 37=MFP2G32E8 → E8 lattice (RDNA3/4)
    const RDNA3PLUS_ONLY: &[(u8, &str)] = &[
        (8, "HFQ6G256"),
        (15, "MQ6G256"),
        (31, "MQ5G256"),
        (24, "MFP4G32 (HFP4G32-fused)"),
        (33, "MFP4G32P (HFP4G32-fused)"),
        (34, "MFP4G32E8"),
        (35, "MFP4G32E8SOA"),
        (36, "MFP3G32E8"),
        (37, "MFP2G32E8"),
    ];
    let mut present: Vec<&str> = RDNA3PLUS_ONLY
        .iter()
        .filter(|(qt, _)| hfq.tensors().iter().any(|t| t.quant_type == *qt))
        .map(|(_, label)| *label)
        .collect();
    present.dedup();
    if present.is_empty() {
        return;
    }
    eprintln!(
        "  ⚠️  RDNA2 ({}): this model contains RDNA3+-only quant formats: {}.",
        gpu.arch,
        present.join(", ")
    );
    eprintln!(
        "      RDNA2 (gfx1030): uniform .mq4 is the validated SKU; this model's \
         MQ3-Lloyd/MQ6/E8 content is best-effort and UNVALIDATED on RDNA2."
    );
}

// TODO(transformer-extraction): the overall `load_weights` orchestration
// here (drop_mmap → embedding+tied-lm_head → norm → per-layer loop) is
// the model the Qwen2 loader at
// `hipfire-arch-qwen2::qwen2::load_weights` follows. The tied-embedding
// re-upload pattern (re-reading `embed_tokens.weight` to construct a
// second GpuTensor for the lm_head) is duplicated in both crates
// because GpuTensor is not Clone. Consolidation PR should either add
// `GpuTensor::shallow_clone()` or switch to `Arc<GpuTensor>` so tied
// embeddings stop costing 2× the embedding VRAM.

/// Attach the lm_head / tied-embed AWQ sidecar when the output dtype supports it.
/// Byte-identical no-op on current files. MUST be called AFTER `output.gpu_dtype`
/// is set (the gate reads it). See docs/plans/awq_fix_claude.md.
fn attach_lm_head_awq_sidecar(hfq: &HfqFile, gpu: &Gpu, output: &mut WeightTensor, k: usize) {
    if output.gpu_dtype.supports_awq_sidecar() {
        output.awq_scale = load_awq_scale_for(hfq, gpu, "lm_head.weight", k)
            .or_else(|| load_awq_scale_for(hfq, gpu, "model.language_model.lm_head.weight", k))
            .or_else(|| {
                load_awq_scale_for(hfq, gpu, "model.language_model.embed_tokens.weight", k)
            });
        eprintln!(
            "  lm_head AWQ sidecar: {}",
            if output.awq_scale.is_some() {
                "attached"
            } else {
                "absent (no-op)"
            }
        );
    }
}

// ── Layout (re-exported from runtime) ─────────────────────────────────────

pub use hipfire_runtime::model_load::Layout;

// ── load_weights (thin assembler over runtime orchestrator) ───────────────

/// Drive a qwen35 `WeightSource` over the device slice (runtime orchestrator),
/// then assemble `Qwen35Weights`. `pager` is always `None` here; paged-experts
/// wiring is unchanged and set by the caller post-load.
pub fn load_weights(
    source: &mut (impl WeightSource<Layer = LayerWeights>),
    devices: &mut [Gpu],
    layout: &Layout,
) -> HipResult<Qwen35Weights> {
    let LoadedWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    } = rt_load_weights(source, devices, layout)?;
    Ok(Qwen35Weights {
        token_embd,
        embd_format,
        output_norm,
        output,
        moe_has_mq6: layers_have_mq6_moe(&layers),
        layers,
        pager: None,
        lm_head_aliases_embd,
    })
}

// ── HfqSource ─────────────────────────────────────────────────────────────

pub struct HfqSource<'a> {
    hfq: &'a mut HfqFile,
    c: &'a Qwen35Config,
}
impl<'a> HfqSource<'a> {
    pub fn new(hfq: &'a mut HfqFile, c: &'a Qwen35Config) -> Self {
        Self { hfq, c }
    }
}
impl WeightSource for HfqSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.c.n_layers
    }

    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        #[cfg(unix)]
        if n_devices == 1 {
            self.hfq.drop_mmap();
        }
        let _ = n_devices;
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        // W4: loud RDNA2 diagnostic — flag RDNA3+-only quant formats (MQ5/MQ6/HFQ6/
        // HFP4G32-fused/E8) that have no validated RDNA2 kernel. No-op off RDNA2.
        // Fired once per load here (the first read with both hfq + gpu in scope,
        // after master's loader refactor split source from devices).
        warn_rdna2_unvalidated_dtypes(self.hfq, gpu);
        let c = self.c;
        eprintln!("  loading token_embd...");
        if c.is_vl_text {
            eprintln!(
                "  qwen3.5-vl text wrapper: mrope_interleaved={} mrope_section={:?}",
                c.mrope_interleaved, c.mrope_section
            );
        }
        let (embd_meta, embd_data) = qwen35_tensor_data_vec(self.hfq, "embed_tokens.weight")
            .expect("embed_tokens not found");
        let out = load_embedding(gpu, embd_meta.quant_type, &embd_data, c.vocab_size, c.dim)?;
        drop(embd_data);
        Ok(out)
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        load_norm_weight(self.hfq, gpu, "norm.weight", &[self.c.dim])
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let c = self.c;
        let hfq = &*self.hfq;
        let has_separate = qwen35_tensor_name_candidates("lm_head.weight")
            .iter()
            .any(|n| hfq.find_tensor_info(n).is_some());
        let (mut output, aliases) = resolve_lm_head(
            gpu,
            has_separate,
            can_alias,
            embd,
            embd_fmt,
            c.vocab_size,
            c.dim,
            |gpu| {
                let (lm_info, lm_data) =
                    qwen35_tensor_data_vec(hfq, "lm_head.weight").expect("lm_head present");
                load_weight_tensor_raw(gpu, lm_info.quant_type, &lm_data, c.vocab_size, c.dim)
            },
            |gpu| {
                let (embd_meta, embd_data) = qwen35_tensor_data_vec(hfq, "embed_tokens.weight")
                    .expect("embed_tokens not found");
                dequant_weight_raw(gpu, embd_meta.quant_type, &embd_data, c.vocab_size, c.dim)
            },
        )?;
        attach_lm_head_awq_sidecar(self.hfq, gpu, &mut output, c.dim);
        Ok((output, aliases))
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<LayerWeights> {
        let c = self.c;
        let is_moe = c.num_experts > 0;
        eprintln!(
            "  loading layer {layer_idx}/{} ({:?}{})...",
            c.n_layers,
            c.layer_types[layer_idx],
            if is_moe { " + MoE" } else { "" }
        );
        let p = format!("layers.{layer_idx}");
        let page = self.hfq.layer_data_range(&p);
        let lw = load_layer_into(self.hfq, c, layer_idx, &p, gpu)?;
        if let Some((start, end)) = page {
            self.hfq.drop_pages_range(start, end - start);
        }
        Ok(lw)
    }
}

// ── ParoSource ────────────────────────────────────────────────────────────

pub struct ParoSource<'a> {
    source: &'a dyn ModelSource,
    mp: &'static str,
    c: &'a Qwen35Config,
}
impl<'a> ParoSource<'a> {
    pub fn new(source: &'a dyn ModelSource, c: &'a Qwen35Config) -> HipResult<Self> {
        source
            .quant_config()
            .ok_or_else(|| HipError::new(0, "ParoQuant model must have quantization_config"))?;
        let mp = paro_text_prefix(source)?;
        Ok(Self { source, mp, c })
    }
    fn read_f16_as_f32(&self, name: &str) -> HipResult<Vec<f32>> {
        let (_, data) = self
            .source
            .tensor_data(name)
            .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
        Ok(hipfire_runtime::weight_backend::f16_bytes_to_f32(&data))
    }
}
impl WeightSource for ParoSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.c.n_layers
    }

    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        if n_devices > 1 {
            return Err(HipError::new(
                0,
                "ParoQuant multi-GPU loading is not supported (HFQ-only)",
            ));
        }
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        eprintln!("  loading token_embd (ParoQuant)...");
        let f32_embd = self.read_f16_as_f32(&format!("{}.embed_tokens.weight", self.mp))?;
        let token_embd = gpu.upload_f32(&f32_embd, &[self.c.vocab_size, self.c.dim])?;
        Ok((token_embd, EmbeddingFormat::F32))
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        paro_load_norm(
            self.source,
            gpu,
            "norm.weight",
            &[self.c.dim],
            QWEN35_NORM_BIAS,
        )
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let mp = self.mp;
        let c = self.c;
        let source = self.source;
        let has_separate = source.tensor_data("lm_head.weight").is_some();
        resolve_lm_head(
            gpu,
            has_separate,
            can_alias,
            embd,
            embd_fmt,
            c.vocab_size,
            c.dim,
            |gpu| {
                let (_, f16) = source
                    .tensor_data("lm_head.weight")
                    .ok_or_else(|| HipError::new(0, "PARO tensor not found: lm_head.weight"))?;
                reupload_f16_as_f32(gpu, &f16, c.vocab_size, c.dim)
            },
            |gpu| {
                let embd_name = format!("{mp}.embed_tokens.weight");
                let (_, f16) = source.tensor_data(&embd_name).ok_or_else(|| {
                    HipError::new(0, &format!("PARO tensor not found: {embd_name}"))
                })?;
                reupload_f16_as_f32(gpu, &f16, c.vocab_size, c.dim)
            },
        )
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<LayerWeights> {
        let c = self.c;
        eprintln!(
            "  loading layer {layer_idx}/{} ({:?}, ParoQuant)...",
            c.n_layers, c.layer_types[layer_idx]
        );
        let mut b = qwen35_paro_backend(self.source, gpu, self.mp, layer_idx);
        let moe = |bk: &mut ParoBackend, cfg: &Qwen35Config, li: usize| {
            crate::paro_moe::paro_load_moe_ffn(
                bk.source,
                bk.gpu,
                &format!("layers.{li}"),
                cfg,
                li as u16,
            )
        };
        crate::layer_driver::load_layer(&mut b, c, layer_idx, moe)
    }
}

/// Construct an `HfqBackend` with qwen35's defaults baked in: `QWEN35_NORM_BIAS`,
/// the qwen35 tensor-name resolver, and the standard pread+awq weight reader.
fn qwen35_hfq_backend<'a>(hfq: &'a HfqFile, gpu: &'a mut Gpu, layer: usize) -> HfqBackend<'a> {
    HfqBackend {
        hfq,
        gpu,
        norm_bias: QWEN35_NORM_BIAS,
        candidates: qwen35_tensor_name_candidates,
        read_proj: load_weight_tensor,
        layer,
    }
}

/// Construct a `ParoBackend` with qwen35's `norm_bias` baked in. `mp` is the
/// text-tower prefix from `paro_text_prefix`.
fn qwen35_paro_backend<'a>(
    source: &'a dyn ModelSource,
    gpu: &'a mut Gpu,
    mp: &'static str,
    layer: usize,
) -> ParoBackend<'a> {
    ParoBackend {
        source,
        gpu,
        mp,
        layer,
        norm_bias: QWEN35_NORM_BIAS,
    }
}

/// Build one layer's `LayerWeights` on `gpu`. Extracted for `load_weights_multi`
/// so the multi-GPU loader can route each layer to its band-owning device
/// without duplicating the tensor-name table. Master's `load_weights` keeps
/// its inline body — does not consume this helper.
fn load_layer_into(
    hfq: &HfqFile,
    config: &Qwen35Config,
    layer_idx: usize,
    p: &str,
    gpu: &mut Gpu,
) -> HipResult<LayerWeights> {
    debug_assert_eq!(p, &format!("layers.{layer_idx}"));
    let mut b = qwen35_hfq_backend(hfq, gpu, layer_idx);
    let moe = |bk: &mut HfqBackend, cfg: &Qwen35Config, li: usize| {
        load_moe_ffn(bk.hfq, bk.gpu, &format!("layers.{li}"), cfg, li as u16)
    };
    crate::layer_driver::load_layer(&mut b, config, layer_idx, moe)
}

thread_local! {
    /// Per-thread EP expert-shard context. When `Some((shard, rank))`,
    /// [`load_moe_ffn`] loads ONLY this rank's owned experts (streaming
    /// owned-only) and builds the `[n_exp]` global pointer tables with dummy
    /// pointers for non-owned slots — the SAME structure post-load
    /// [`shard_moe_experts`] produces, but WITHOUT the full-model load peak that
    /// OOMs a model larger than one card's VRAM. Set by the EP load driver
    /// around `load_weights`, cleared (`None`) after. `None` = full replicated
    /// load (the default for every non-EP caller).
    static EP_EXPERT_SHARD: std::cell::RefCell<Option<(ShardConfig, usize)>> =
        const { std::cell::RefCell::new(None) };
}

/// Set the per-thread EP expert-shard context consumed by `load_weights` →
/// [`load_moe_ffn`]. The EP load driver calls this with `Some((shard, rank))`
/// immediately before `load_weights` on each rank, then `None` immediately
/// after. Mirrors DeepSeek-V4's `load_weights_sharded` but threaded via TLS so
/// the 87 existing `load_weights` callers need no signature change.
pub fn set_ep_expert_shard(ctx: Option<(ShardConfig, usize)>) {
    EP_EXPERT_SHARD.with(|c| *c.borrow_mut() = ctx);
}

fn current_ep_expert_shard() -> Option<(ShardConfig, usize)> {
    EP_EXPERT_SHARD.with(|c| c.borrow().clone())
}

/// Load one layer's full MoE FFN block: router, all routed experts, shared expert,
/// and the per-layer scalar shared-expert gate. Tensor naming follows what the
/// quantizer emits for qwen3_5_moe (commit 4860575): the 3D stacked-expert source
/// tensors get split per-expert into `mlp.experts.{X}.{base}.weight`.
///
/// HIPFIRE_E8_SOA_EXPERTS (cached): transpose routed E8 gate_up experts AoS->SoA at
/// load so the SoA-coalesced indexed kernel can read them. Must match the dispatch
/// flag in rdna-compute (same env). Default OFF.
fn e8_soa_experts() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_E8_SOA_EXPERTS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

const MQ4_G256_QUANT_TYPE: u8 = 13;

struct PackedMq4ExpertSpec {
    gate_up_name: String,
    down_name: String,
}

/// Restrict expert packing to the gfx11 family where both dGPU residency and
/// gfx1151 retained-replay performance have been validated. On gfx12, placing
/// every expert view inside two large layer allocations makes the HIP/graph
/// route treat those views as one coarse allocation domain; gfx1201 tg128 fell
/// from 171 to 77 tok/s even though retained PM4 remained fast. Preserve the
/// original per-expert allocations there until a gfx12-safe packing granularity
/// is established.
fn packed_mq4_experts_supported(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_rdna3()
}

/// Pack a uniform MQ4 routed-expert layer into two GPU allocations while
/// preserving one `WeightTensor` view and one device pointer-table entry per
/// expert. Returns `None` for every non-uniform/non-MQ4 layout so mixed tiers,
/// ParoQuant, paged experts, and EP streaming retain their literal behavior.
fn try_load_packed_mq4_experts(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    expert_ids: &[usize],
    mi: usize,
    dim: usize,
) -> HipResult<Option<(Vec<ExpertWeights>, PackedExpertOwners)>> {
    if expert_ids.is_empty() {
        return Ok(None);
    }

    let mut specs = Vec::with_capacity(expert_ids.len());
    let mut gate_up_stride = None;
    let mut down_stride = None;
    for &expert_id in expert_ids {
        let gate_up_bare = format!("{p}.mlp.experts.{expert_id}.gate_up_proj.weight");
        let down_bare = format!("{p}.mlp.experts.{expert_id}.down_proj.weight");
        let Some((gate_up_name, gate_up_qt, gate_up_bytes)) =
            qwen35_tensor_name_candidates(&gate_up_bare)
                .into_iter()
                .find_map(|name| {
                    hfq.find_tensor_info(&name)
                        .map(|info| (name, info.quant_type, info.data_size))
                })
        else {
            return Ok(None);
        };
        let Some((down_name, down_qt, down_bytes)) = qwen35_tensor_name_candidates(&down_bare)
            .into_iter()
            .find_map(|name| {
                hfq.find_tensor_info(&name)
                    .map(|info| (name, info.quant_type, info.data_size))
            })
        else {
            return Ok(None);
        };
        if gate_up_qt != MQ4_G256_QUANT_TYPE || down_qt != MQ4_G256_QUANT_TYPE {
            return Ok(None);
        }
        match gate_up_stride {
            None => gate_up_stride = Some(gate_up_bytes),
            Some(stride) if stride == gate_up_bytes => {}
            Some(_) => return Ok(None),
        }
        match down_stride {
            None => down_stride = Some(down_bytes),
            Some(stride) if stride == down_bytes => {}
            Some(_) => return Ok(None),
        }
        specs.push(PackedMq4ExpertSpec {
            gate_up_name,
            down_name,
        });
    }

    let gate_up_stride = gate_up_stride.expect("non-empty packed MQ4 expert list");
    let down_stride = down_stride.expect("non-empty packed MQ4 expert list");
    let mut gate_up_host = Vec::with_capacity(gate_up_stride * specs.len());
    let mut down_host = Vec::with_capacity(down_stride * specs.len());
    for spec in &specs {
        {
            let (_, bytes) = hfq.tensor_data_pread(&spec.gate_up_name).ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "qwen35: packed MQ4 tensor disappeared: {}",
                        spec.gate_up_name
                    ),
                )
            })?;
            if bytes.len() != gate_up_stride {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: packed MQ4 gate_up stride changed for {}: {} != {gate_up_stride}",
                        spec.gate_up_name,
                        bytes.len()
                    ),
                ));
            }
            gate_up_host.extend_from_slice(&bytes);
        }
        {
            let (_, bytes) = hfq.tensor_data_pread(&spec.down_name).ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("qwen35: packed MQ4 tensor disappeared: {}", spec.down_name),
                )
            })?;
            if bytes.len() != down_stride {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: packed MQ4 down stride changed for {}: {} != {down_stride}",
                        spec.down_name,
                        bytes.len()
                    ),
                ));
            }
            down_host.extend_from_slice(&bytes);
        }
    }

    let gate_up_owner = gpu.upload_raw(&gate_up_host, &[specs.len(), gate_up_stride])?;
    drop(gate_up_host);
    let down_owner = match gpu.upload_raw(&down_host, &[specs.len(), down_stride]) {
        Ok(owner) => owner,
        Err(error) => {
            let _ = gpu.free_tensor(gate_up_owner);
            return Err(error);
        }
    };
    drop(down_host);

    let mut experts = Vec::with_capacity(specs.len());
    for (slot, spec) in specs.iter().enumerate() {
        let mut gate_up = WeightTensor {
            buf: gate_up_owner.sub_offset(slot * gate_up_stride, gate_up_stride),
            gpu_dtype: DType::MQ4G256,
            m: 2 * mi,
            k: dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        gate_up.awq_scale = load_awq_scale_for(hfq, gpu, &spec.gate_up_name, dim);
        let mut down = WeightTensor {
            buf: down_owner.sub_offset(slot * down_stride, down_stride),
            gpu_dtype: DType::MQ4G256,
            m: dim,
            k: mi,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        down.awq_scale = load_awq_scale_for(hfq, gpu, &spec.down_name, mi);
        experts.push(ExpertWeights { gate_up, down });
    }

    Ok(Some((
        experts,
        PackedExpertOwners {
            gate_up: gate_up_owner,
            down: down_owner,
        },
    )))
}

/// AoS mfp4-E8 -> SoA byte transform (exact port of `aos_to_soa_full` in
/// bench_e8_soa_correctness). AoS row: [16B hdr][n_blocks×(1B scale + 16B cw)].
/// SoA row: [16B hdr (flag=0x06)][n_blocks scales, pad16][n_blocks×16B cw]. Same size.
fn e8_aos_to_soa(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let n_blocks = k / 32;
    let aos_row = 16 + n_blocks * 17;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row];
    for r in 0..m {
        let src = &aos[r * aos_row..(r + 1) * aos_row];
        let dst = &mut out[r * soa_row..(r + 1) * soa_row];
        dst[..16].copy_from_slice(&src[..16]);
        dst[6] = 0x06; // SoA flag
        for b in 0..n_blocks {
            dst[16 + b] = src[16 + b * 17]; // E4M3 scales -> contiguous
        }
        let cw0 = 16 + scale_padded;
        for b in 0..n_blocks {
            let s = 16 + b * 17 + 1;
            let d = cw0 + b * 16;
            dst[d..d + 16].copy_from_slice(&src[s..s + 16]); // 16B codewords -> contiguous
        }
    }
    out
}

/// EP streaming-shard mode: when [`current_ep_expert_shard`] is `Some`, only the
/// rank's owned experts are read/allocated; the pointer tables are built global
/// `[n_exp]` with dummy pointers for non-owned slots (which contribute 0 to the
/// all-reduce because their gate_up is a zeroed buffer). Uniform files only —
/// graded/AWQ EP would need the full per-expert dtype map and is rejected here.
pub(crate) fn load_moe_ffn(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;

    // REAP keep-map for this layer (None ⇒ no pruning / identity). When a
    // keep is present, `n_exp == config.num_experts` is already the KEPT
    // count; the router and expert loops below load only the kept rows.
    let ep = config
        .reap_keep
        .as_ref()
        .map(|r| r.expert_plan(layer_idx as usize));

    // Router: hidden_size → num_experts. Precision-sensitive but small.
    // Under a keep, gather the router's expert rows (`[orig_experts, dim]`)
    // down to the kept set so it emits logits only for kept experts, in
    // compact slot order. No keep ⇒ the literal original full load.
    let router = match ep.as_ref().and_then(|e| e.keep()) {
        Some(keep) => load_weight_tensor_keep(
            hfq,
            gpu,
            &format!("{p}.mlp.gate.weight"),
            n_exp,
            config.dim,
            keep,
        )?,
        None => load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.gate.weight"),
            n_exp,
            config.dim,
            qwen35_tensor_name_candidates,
        )?,
    };

    // Shared expert (always-on, contributes to every token). Unlike routed
    // experts, gate_proj + up_proj are stored separately in the safetensors
    // (routed experts store them fused as `gate_up_proj`).
    let shared_expert = SharedExpertWeights {
        gate: load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            smi,
            config.dim,
            qwen35_tensor_name_candidates,
        )?,
        up: load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            smi,
            config.dim,
            qwen35_tensor_name_candidates,
        )?,
        down: load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            config.dim,
            smi,
            qwen35_tensor_name_candidates,
        )?,
    };
    // Scalar gate on the shared-expert add: sigmoid(shared_expert_gate · x).
    // Stored as a 1×hidden row-vector.
    let shared_expert_gate = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.mlp.shared_expert_gate.weight"),
        1,
        config.dim,
        qwen35_tensor_name_candidates,
    )?;

    // Routed experts — quantizer wrote per-expert tensors named
    // `{p}.mlp.experts.{X}.gate_up_proj.weight` (shape [2*moe_intermediate, hidden_size])
    // and `{p}.mlp.experts.{X}.down_proj.weight` (shape [hidden_size, moe_intermediate]).
    //
    // REAP keep-map: `n_exp` is already the KEPT count (config.num_experts was
    // overridden in apply_reap_plan), and under a keep `ExpertPlan::n_slots`
    // also equals the kept count — so `n_exp` is the slot count on both the
    // keep and no-keep paths. Iterate compact slots `0..n_exp` (matching ds4's
    // `0..n_routed_experts`) and load only the kept original experts under
    // their remapped names via `ep.src(slot)`. EP streaming-shard composes by
    // applying ownership to that ORIGINAL expert id, while pointer tables keep
    // compact slot indexing. No keep ⇒ identity (slot == original index) — the
    // literal original loop.
    let ep_shard = current_ep_expert_shard();
    if ep.is_some() && ep_shard.is_some() {
        return Err(HipError::new(
            0,
            "qwen35: REAP keep-map + EP sharding are mutually exclusive",
        ));
    }
    let owns_orig = |x: usize| {
        ep_shard
            .as_ref()
            .map_or(true, |(sh, r)| sh.owns_expert(*r, x))
    };

    let expert_ids: Vec<usize> = (0..n_exp)
        .map(|slot| ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot))
        .filter(|&x| owns_orig(x))
        .collect();
    let packed = if ep_shard.is_none() && packed_mq4_experts_supported(gpu) {
        try_load_packed_mq4_experts(hfq, gpu, p, &expert_ids, mi, config.dim)?
    } else {
        None
    };
    let (mut experts, packed_expert_owners) = if let Some((experts, owners)) = packed {
        if layer_idx == 0 {
            eprintln!(
                "  routed MQ4 expert packing: {} per-expert weight buffers -> 2 layer blobs",
                2 * experts.len()
            );
        }
        (experts, Some(owners))
    } else {
        let mut experts = Vec::with_capacity(expert_ids.len());
        for x in expert_ids {
            let gate_up = load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.experts.{x}.gate_up_proj.weight"),
                2 * mi,
                config.dim,
                qwen35_tensor_name_candidates,
            )?;
            let down = load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.experts.{x}.down_proj.weight"),
                config.dim,
                mi,
                qwen35_tensor_name_candidates,
            )?;
            experts.push(ExpertWeights { gate_up, down });
        }
        (experts, None)
    };

    // gfx11 E8 SoA experts: transpose routed gate_up E8 weights AoS->SoA at load so the
    // SoA-coalesced indexed kernel reads coalesced; the ptr table below picks up the new
    // SoA bufs. Down experts stay AoS (down SoA = increment 2). RDNA3 dGPU +
    // HIPFIRE_E8_SOA_EXPERTS=1, full-load only (EP shard builds its table separately).
    if e8_soa_experts() && gpu.arch_caps.is_rdna3_dgpu() && ep_shard.is_none() {
        let mut converted = 0usize;
        for ew in experts.iter_mut() {
            if ew.gate_up.gpu_dtype == DType::MFP4G32E8 {
                let (m, k) = (ew.gate_up.m, ew.gate_up.k);
                let nbytes = ew.gate_up.buf.buf.size();
                let mut aos = vec![0u8; nbytes];
                gpu.hip.memcpy_dtoh(&mut aos, &ew.gate_up.buf.buf)?;
                let soa = e8_aos_to_soa(&aos, m, k);
                // In-place overwrite — SoA == AoS byte size when n_blocks%16==0 (true for
                // K=2048: 16+64+64*16 == 16+64*17 == 1104). No new alloc → no VRAM growth.
                if soa.len() == nbytes {
                    gpu.hip.memcpy_htod(&ew.gate_up.buf.buf, &soa)?;
                    converted += 1;
                } else if layer_idx == 0 {
                    eprintln!(
                        "  [e8-soa] SKIP: SoA size {} != AoS {} (n_blocks%16!=0) — keeping AoS",
                        soa.len(),
                        nbytes
                    );
                }
            }
        }
        if converted > 0 && layer_idx == 0 {
            eprintln!("  [e8-soa] transposed {converted} gate_up experts AoS->SoA (per layer)");
        }
    }

    // Build the device-side pointer tables consumed by the indexed MoE
    // GEMV kernels. Each slot is an `unsigned long long` (the device
    // address of an expert's `gate_up.buf` / `down.buf`). Stored as an
    // F32 tensor of length 2 * num_experts because each pointer occupies
    // 8 bytes = 2 F32 slots; the kernel reads them via a u64 cast.
    // GLOBAL [n_exp] device pointer tables (8 B/ptr = 2 F32 slots). Full load:
    // gu_ptrs[e] = experts[e]. EP shard: non-owned slots get a dummy pointer
    // (zeroed gate_up ⇒ silu output 0 ⇒ 0 contribution to the EP all-reduce;
    // the down dummy is a real owned buffer so its uniform-dtype dequant stays
    // in-bounds) — exactly what `shard_moe_experts` builds post-load.
    let mut gu_ptrs = vec![0u64; n_exp];
    let mut dn_ptrs = vec![0u64; n_exp];
    if ep_shard.is_some() {
        assert!(
            !experts.is_empty(),
            "EP shard: rank owns no experts in layer {layer_idx}"
        );
        // Shared zeroed gate_up dummy (same byte size as a real expert gate_up).
        // LEAKED so the ptr stays valid for the model's lifetime (matches
        // shard_moe_experts v1).
        let gu_buf_bytes = experts[0].gate_up.buf.buf.size();
        let zero_gu = gpu.zeros(&[gu_buf_bytes / 4], DType::F32)?;
        let dummy_gu = zero_gu.buf.as_ptr() as u64;
        std::mem::forget(zero_gu);
        let dummy_dn = experts[0].down.buf.buf.as_ptr() as u64;
        let mut li = 0usize;
        for slot in 0..n_exp {
            let orig = ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot);
            if owns_orig(orig) {
                gu_ptrs[slot] = experts[li].gate_up.buf.buf.as_ptr() as u64;
                dn_ptrs[slot] = experts[li].down.buf.buf.as_ptr() as u64;
                li += 1;
            } else {
                gu_ptrs[slot] = dummy_gu;
                dn_ptrs[slot] = dummy_dn;
            }
        }
    } else {
        for (e, ew) in experts.iter().enumerate() {
            gu_ptrs[e] = ew.gate_up.buf.buf.as_ptr() as u64;
            dn_ptrs[e] = ew.down.buf.buf.as_ptr() as u64;
        }
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;

    // Route A MoE-AWQ: when every expert carries a down.awq_scale sidecar
    // (auto-loaded by load_weight_tensor for MQ4G256, which supports_awq_sidecar),
    // build the per-expert pointer table the indexed silu+rotate selects from
    // (topk_indices[krank] → expert's [mi] scale). All-or-none — a partial set
    // is a malformed file and disables MoE-AWQ for the layer.
    // Debug kill-switch, read ONCE at load (never on the decode hot path):
    // HIPFIRE_MOE_AWQ=0 forces the plain silu+rotate even on an AWQ file.
    // NOTE: this is a path-confirmation tool, NOT a safe fallback — AWQ files
    // bake W·s into the weights, so skipping the x/s divide yields W·s·x
    // (garbage). Expect incoherent output when set on an AWQ file; that
    // *confirms* the indexed AWQ kernel is the firing path. The real
    // AWQ-vs-plain A/B uses two separately quantized files.
    let moe_awq_enabled = hipfire_config::developer_var("HIPFIRE_MOE_AWQ")
        .ok()
        .as_deref()
        != Some("0");
    let awq_present = experts
        .iter()
        .filter(|e| e.down.awq_scale.is_some())
        .count();
    let expert_down_awq_ptrs = if ep_shard.is_some() {
        // EP shard: AWQ-EP needs a sharded scale pointer table (dummies for
        // non-owned slots). Not yet supported — guard rather than silently
        // disable. Uniform-no-AWQ files (e.g. .mq6) hit `awq_present == 0` → None.
        if awq_present != 0 {
            return Err(HipError::new(
                0,
                "AWQ MoE EP not yet supported (quantize experts without AWQ for EP serving)",
            ));
        }
        None
    } else if moe_awq_enabled && n_exp > 0 && awq_present == n_exp {
        let aw_ptrs: Vec<u64> = experts
            .iter()
            .map(|e| e.down.awq_scale.as_ref().unwrap().buf.as_ptr() as u64)
            .collect();
        let aw_bytes: Vec<u8> = aw_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
        let t = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
        gpu.hip.memcpy_htod(&t.buf, &aw_bytes)?;
        Some(t)
    } else {
        if awq_present != 0 {
            eprintln!(
                "[moe-awq] layer {layer_idx}: partial down.awq_scale coverage \
                 ({awq_present}/{n_exp}) — disabling MoE-AWQ for this layer"
            );
        }
        None
    };

    // ── Per-expert mixed-precision dtype-tag table ──────────────────────
    // For N-tier graded files (T3-2L / T3-3L) the TIER_MAP assigns each
    // expert a tier that applies to BOTH gate_up AND down.  Built iff
    // gate_up OR down dtypes differ across experts (single source of truth
    // for `routed_has_mixed_experts`).  Tags:
    //   0 = MQ6G256      (200 B/group affine)
    //   1 = MQ2G256Lloyd ( 72 B/group codebook)
    //   2 = MQ4G256      (136 B/group affine)
    //   3 = MQ3G256Lloyd (112 B/group codebook)
    //   4 = MFP4G32E8    (16 B row hdr + (K/32)*17 B; E8 lattice VQ, 4.25 bpw)
    //   5 = MFP3G32E8    (16 B row hdr + (K/32)*13 B; 3-bit E8 lattice, 3.25 bpw)
    //   6 = MFP2G32E8    (16 B row hdr + (K/32)*9  B; 2-bit E8 lattice, 2.25 bpw)
    // Uniform files see gu0==gu_n and dn0==dn_n → None (byte-identical).
    // All E8 tags (4, 5, 6) are decoded by BOTH the per-token mixed gemv kernels
    // AND the batched grouped-WMMA kernels (gfx11 _k2 and gfx12 .gfx12).
    // Priority: gate_up dtype drives the tag (gate_up is the dominant quality
    // lever); for the existing down-only graded binary the gate_up types are
    // uniform MQ4G256 → tag2, which is the correct MQ4 branch.
    let expert_dtype_tags = if ep_shard.is_some() {
        // EP shard: a graded (mixed-dtype) file needs the FULL [n_exp] global tag
        // map, but only this rank's owned experts are loaded, so non-owned dtypes
        // aren't visible. Uniform files (all owned experts same dtype, e.g. .mq6)
        // → None; graded EP is not yet supported.
        let mixed = !experts.is_empty()
            && (experts
                .iter()
                .any(|e| e.gate_up.gpu_dtype != experts[0].gate_up.gpu_dtype)
                || experts
                    .iter()
                    .any(|e| e.down.gpu_dtype != experts[0].down.gpu_dtype));
        if mixed {
            return Err(HipError::new(
                0,
                "graded (mixed-dtype) MoE EP not yet supported",
            ));
        }
        None
    } else if n_exp > 0 {
        let gu0 = experts[0].gate_up.gpu_dtype;
        let dn0 = experts[0].down.gpu_dtype;
        let mixed = experts.iter().any(|e| e.gate_up.gpu_dtype != gu0)
            || experts.iter().any(|e| e.down.gpu_dtype != dn0);
        // The merged dtype-tag gate_up/down kernels have NO MQ2/MQ3-G256-GL
        // branch — the GL formats are SoA with a scalar-arg codebook, which the
        // tag-branched decoder (per-group-header strides only) cannot express.
        // The `_ => 2u8` default below would silently tag a GL expert as MQ4 and
        // read 136 B/group off a 64/96 B/group blob: token soup, no error. Refuse
        // instead. Uniform GL files never reach here (mixed == false).
        if mixed
            && experts.iter().any(|e| {
                matches!(e.gate_up.gpu_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
                    || matches!(e.down.gpu_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
            })
        {
            return Err(HipError::new(
                0,
                "graded (mixed-dtype) MoE with MQ2/MQ3-G256-GL experts is not supported: \
                 the merged dtype-tag decode kernel has no GL branch. Use a UNIFORM GL \
                 file (all routed experts the same GL dtype per projection).",
            ));
        }
        if mixed {
            // Map each expert to a tag based on its gate_up dtype (the tier
            // that was assigned to BOTH projections by the quantizer).  Fall
            // back to the down dtype for the legacy down-only-graded case
            // (gate_up all MQ4G256 → tag2 = MQ4, which is correct).
            let tags: Vec<u8> = experts
                .iter()
                .map(|e| match e.gate_up.gpu_dtype {
                    DType::MQ6G256 => 0u8,
                    DType::MQ2G256Lloyd => 1u8,
                    DType::MQ4G256 => {
                        // gate_up uniform MQ4: use down dtype to distinguish
                        // hot (MQ6) from mid (MQ4) from cold tiers so the
                        // legacy down-only-graded binary still dispatches the
                        // correct down branch.
                        match e.down.gpu_dtype {
                            DType::MQ6G256 => 0u8,      // hot (gu4 dn6 mixed)
                            DType::MQ2G256Lloyd => 1u8, // cold MQ2L
                            DType::MFP2G32E8 => 6u8,    // cold mfp2-E8
                            DType::MQ3G256Lloyd => 3u8, // cold MQ3L
                            DType::MFP3G32E8 => 5u8,    // cold mfp3-E8
                            _ => 2u8,                   // default MQ4
                        }
                    }
                    DType::MQ3G256Lloyd => 3u8,
                    // mfp4-E8 (4.25 bpw) graded tier — gate_up AND down are E8
                    // for these experts (tier applies to both); the mixed gemv
                    // kernels decode tag 4 via e8_row_partial.
                    DType::MFP4G32E8 => 4u8,
                    // [NaN-CRITICAL] mfp3-E8 graded cold tier (tag 5).
                    // Both gate_up AND down carry MFP3G32E8; the merged gemv kernels
                    // decode tag 5 via mfp3e8_row_partial (3-bit lattice, 13 B/blk).
                    DType::MFP3G32E8 => 5u8,
                    // [NaN-CRITICAL] mfp2-E8 graded cold tier (tag 6).
                    // Both gate_up AND down carry MFP2G32E8; the merged gemv kernels
                    // decode tag 6 via mfp2e8_row_partial (2-bit lattice, 9 B/blk).
                    DType::MFP2G32E8 => 6u8,
                    _ => 2u8, // default: treat unknown tiers as MQ4
                })
                .collect();
            let t = gpu.alloc_tensor(&[n_exp], DType::Raw)?;
            gpu.hip.memcpy_htod(&t.buf, &tags)?;
            Some(t)
        } else {
            None
        }
    } else {
        None
    };

    Ok(MoeFfnWeights {
        router,
        experts,
        packed_expert_owners,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        expert_down_awq_ptrs,
        expert_dtype_tags,
        // MAD-93 v0.1: non-paged loader path. Layer identity for pager-keyed
        // future work, expert_shape None (callers read shapes off `experts`
        // directly when paged_experts==false).
        layer_idx,
        expert_shape: None,
        paro_shared: None,
    })
}

// ─── MoE FFN (decode, batch=1) ──────────────────────────────────────────

/// Construct a non-owning `GpuTensor` view over `[offset_elems,
/// offset_elems + len_elems)` of `src`. Valid only for F32 (4 bytes/elem).

/// One-token MoE FFN: router → top-K → shared expert + top-K routed, added
/// into `x_residual` in place. `x_norm` is the already-RMSNormed FFN input.
///
/// Dense-compute decode reference implementation (Phase 1). Top-K selection
/// runs on CPU via a single D2H sync per layer on the router logits; the
/// shared-expert scalar gate is another D2H sync. Sparse-routing + batched
/// grouped-GEMM variants come in later phases — this version prioritizes
/// correctness and minimal surface area.
///
/// Matches HF `modeling_qwen3_5_moe.py`:
///   router_probs  = softmax(W_router · x_norm)            // [n_exp]
///   (idx, w)      = topk(router_probs, k)                  // [k]
///   if norm_topk:  w /= w.sum()
///   scalar        = sigmoid(W_shared_gate · x_norm)        // [1]
///   y_shared      = scalar * shared_expert(x_norm)         // [hidden]
///   y_moe         = sum_{k} w[k] * expert[idx[k]](x_norm)  // [hidden]
///   x_residual   += y_shared + y_moe
/// Non-owning borrow of the scratch buffers `moe_ffn_decode_impl` needs.
/// Callers construct one of these from either a `Qwen35Scratch` (preallocated,
/// hipGraph-capturable) or from tensors they own locally (heap path).
struct MoeScratchRef<'a> {
    router_logits: &'a GpuTensor,
    scalar_buf: &'a GpuTensor,
    x_rot_local: &'a GpuTensor,
    gate_up_buf: &'a GpuTensor,
    gate_buf: &'a GpuTensor,
    up_buf: &'a GpuTensor,
    ffn_hidden: &'a GpuTensor,
    ffn_out: &'a GpuTensor,
    gate_batch: &'a GpuTensor,
    up_batch: &'a GpuTensor,
    rot_batch: &'a GpuTensor,
    topk_indices: &'a GpuTensor,
    topk_weights: &'a GpuTensor,
    // [k_top × dim + ceil(dim/4)] f32 — per-(expert-rank) MoE down output
    // plus a zero-initialised u32 counter tail used only by the optional
    // expert-wave-preserving last-arriver combine experiment. The production
    // atomic-free expand+combine path ignores the tail. Mirrors the prefill
    // `pbs.moe_down_expanded_batch` layout with batch=1. Required so
    // the MoE FFN is byte-deterministic under hipGraph replay; see
    // task #100 root-cause notes in `forward_scratch`.
    down_expanded: &'a GpuTensor,
}

impl<'a> MoeScratchRef<'a> {
    /// View into a Qwen35Scratch's MoE fields. Panics if the caller didn't
    /// allocate MoE scratch (config.num_experts == 0).
    fn from_scratch(s: &'a Qwen35Scratch) -> Self {
        Self {
            router_logits: s
                .moe_router_logits
                .as_ref()
                .expect("MoE scratch not allocated"),
            scalar_buf: s.moe_scalar_buf.as_ref().expect("MoE scratch"),
            x_rot_local: s.moe_x_rot.as_ref().expect("MoE scratch"),
            gate_up_buf: s.moe_gate_up_buf.as_ref().expect("MoE scratch"),
            gate_buf: s.moe_gate_buf.as_ref().expect("MoE scratch"),
            up_buf: s.moe_up_buf.as_ref().expect("MoE scratch"),
            ffn_hidden: s.moe_ffn_hidden.as_ref().expect("MoE scratch"),
            ffn_out: s.moe_ffn_out.as_ref().expect("MoE scratch"),
            gate_batch: s.moe_gate_batch.as_ref().expect("MoE scratch"),
            up_batch: s.moe_up_batch.as_ref().expect("MoE scratch"),
            rot_batch: s.moe_rot_batch.as_ref().expect("MoE scratch"),
            topk_indices: s.moe_topk_indices.as_ref().expect("MoE scratch"),
            topk_weights: s.moe_topk_weights.as_ref().expect("MoE scratch"),
            down_expanded: s.moe_down_expanded.as_ref().expect("MoE scratch"),
        }
    }
}

/// Heap-allocating wrapper for callers without pre-allocated scratch (the
/// debug `forward()` path). Allocates 11 tensors, runs moe_ffn_decode_impl,
/// frees. NOT hipGraph-compatible. For hot-path decode, callers should go
/// through moe_ffn_decode_with_scratch which reuses pre-allocated buffers.
fn moe_ffn_decode(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let max_inter = mi.max(smi);

    let router_logits = gpu.alloc_tensor(&[n_exp], DType::F32)?;
    let scalar_buf = gpu.alloc_tensor(&[1], DType::F32)?;
    let x_rot_local = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let gate_up_buf = gpu.alloc_tensor(&[2 * max_inter], DType::F32)?;
    let gate_buf = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let up_buf = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_hidden = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_out = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let gate_batch = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let up_batch = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let rot_batch = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let topk_indices = gpu.alloc_tensor(&[k], DType::F32)?;
    let topk_weights = gpu.alloc_tensor(&[k], DType::F32)?;
    let down_expanded = gpu.zeros(&[k * hidden + hidden.div_ceil(4)], DType::F32)?;

    let refs = MoeScratchRef {
        router_logits: &router_logits,
        scalar_buf: &scalar_buf,
        x_rot_local: &x_rot_local,
        gate_up_buf: &gate_up_buf,
        gate_buf: &gate_buf,
        up_buf: &up_buf,
        ffn_hidden: &ffn_hidden,
        ffn_out: &ffn_out,
        gate_batch: &gate_batch,
        up_batch: &up_batch,
        rot_batch: &rot_batch,
        topk_indices: &topk_indices,
        topk_weights: &topk_weights,
        down_expanded: &down_expanded,
    };
    let result = moe_ffn_decode_impl(
        gpu, ffn, x_norm, x_residual, config, &refs, false, None, false, false,
    );

    for t in [
        router_logits,
        scalar_buf,
        x_rot_local,
        gate_up_buf,
        gate_buf,
        up_buf,
        ffn_hidden,
        ffn_out,
        gate_batch,
        up_batch,
        rot_batch,
        topk_indices,
        topk_weights,
        down_expanded,
    ] {
        gpu.free_tensor(t)?;
    }
    result
}

/// All gate-side + routed MoE weights are MQ4G256 — the precondition for
/// the prerotated fast path where the caller can fuse rmsnorm+FWHT via
/// `fused_rmsnorm_rotate_mq` and call `moe_ffn_decode_with_scratch_prerotated`.
pub(crate) fn ffn_all_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    ffn_gate_side_mq4_for_moe(ffn)
        && ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
}

/// GATE-SIDE only MQ4 (router + shared expert), independent of the routed
/// experts. When true, the fused rmsnorm+FWHT path + prerotated MoE decode are
/// applicable even on a graded file: the MQ4 router routes on the rotated x via
/// the fused gate kernel (MoeResolution::gate_fusable), and the routed experts
/// (any FwhtG256 MQ-family) consume the same single rotated activation — saving
/// the un-fused rmsnorm + the per-component rotates the else-branch incurs.
/// The redline mq4r (MQ4 gate + graded experts) hits this; uniform MQ4 is a
/// superset (ffn_all_mq4_for_moe). Q8-router files (mq4p) correctly stay false.
pub(crate) fn ffn_gate_side_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
}

/// Detect any MQ3G256 / MQ3G256Lloyd weight inside a MoE FFN block (router,
/// shared expert gate/up/down, shared_expert_gate router-mix scalar, or any
/// routed expert's gate_up/down). The MoE batched FFN kernels assume HFQ4
/// layout (136 B/group); an MQ3 weight (104 B/group) or Lloyd-MQ3 weight
/// (112 B/group) would dispatch with the wrong stride. Used by the
/// captured-prefill and non-captured-prefill defense-in-depth checks.
///
/// Mirrors `is_mq3_any` in `forward_prefill_batch_single_chunk_captured`
/// (line 3325) so both cross-checks treat plain and Lloyd-MQ3 identically.
/// MQ3/MQ3-Lloyd in the STRUCTURAL parts of a MoE FFN (router, shared
/// expert gate/up/down). These are NOT served by the merged grouped kernel
/// (which only handles routed experts) → must still hard-error.
fn moe_ffn_has_mq3_structural(ffn: &MoeFfnWeights) -> bool {
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    is_mq3_any(ffn.router.gpu_dtype)
        || is_mq3_any(ffn.shared_expert_gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.up.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.down.gpu_dtype)
}

/// MQ3/MQ3-Lloyd in ROUTED experts WITHOUT a tag table (uniform MQ3, not
/// the graded case the merged grouped kernel handles). When
/// `expert_dtype_tags` is `Some`, the merged grouped kernel carries the
/// correct per-expert stride and this returns false (pass).
fn moe_ffn_has_mq3_experts_uniform(ffn: &MoeFfnWeights) -> bool {
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    ffn.expert_dtype_tags.is_none()
        && ffn
            .experts
            .iter()
            .any(|e| is_mq3_any(e.gate_up.gpu_dtype) || is_mq3_any(e.down.gpu_dtype))
}

fn moe_ffn_has_mq6(ffn: &MoeFfnWeights) -> bool {
    let is_mq6 = |dt: DType| matches!(dt, DType::MQ6G256);
    is_mq6(ffn.router.gpu_dtype)
        || is_mq6(ffn.shared_expert_gate.gpu_dtype)
        || is_mq6(ffn.shared_expert.gate.gpu_dtype)
        || is_mq6(ffn.shared_expert.up.gpu_dtype)
        || is_mq6(ffn.shared_expert.down.gpu_dtype)
        || ffn
            .experts
            .iter()
            .any(|e| is_mq6(e.gate_up.gpu_dtype) || is_mq6(e.down.gpu_dtype))
}

fn layers_have_mq6_moe(layers: &[LayerWeights]) -> bool {
    layers.iter().any(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => moe_ffn_has_mq6(&l.ffn),
        LayerWeights::FullAttnMoe(l) => moe_ffn_has_mq6(&l.ffn),
        _ => false,
    })
}

/// Zero-alloc MoE decode for the scratch path. `scratch.moe_*` fields must
/// be populated (done automatically by `Qwen35Scratch::new` when config
/// indicates a MoE model). Safe to call under hipGraph stream capture.
pub(crate) fn moe_ffn_decode_with_scratch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(
        gpu, ffn, x_norm, x_residual, config, &refs, false, None, false, false,
    )
}

/// Same as `moe_ffn_decode_with_scratch` but expects the caller to have
/// already populated `scratch.moe_x_rot` with FWHT-rotated post-rmsnorm x
/// (e.g. via a fused `fused_rmsnorm_rotate_mq` launch at the call site).
/// For all-MQ4 MoE layers this saves one launch per layer by eliding the
/// internal `rotate_x_mq`. On non-MQ4 layers this flag is ignored.
pub(crate) fn moe_ffn_decode_with_scratch_prerotated(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(
        gpu, ffn, x_norm, x_residual, config, &refs, true, None, false, false,
    )
}

/// The actual MoE FFN implementation. Uses the caller-provided scratch
/// buffers, never allocates.
// ── REAP expert-importance capture (HIPFIRE_MOE_EXPERT_STATS=1) ────────────
// Per-(layer, expert) accumulators: routing count, Σ gate_weight,
// Σ ‖expert_output‖, Σ (gate × ‖output‖). The last is the true REAP
// contribution (gate-weighted output norm) — compared against raw frequency
// (count) to decide whether freq agrees with contribution before committing to
// any per-expert mixed-precision kernel. Dumped by `dump_expert_stats`.
static EXPERT_STATS: std::sync::Mutex<
    Option<std::collections::HashMap<(u16, u16), (u64, f64, f64, f64)>>,
> = std::sync::Mutex::new(None);
static EXPERT_STATS_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn expert_stats_enabled() -> bool {
    *EXPERT_STATS_ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_EXPERT_STATS")
            .ok()
            .as_deref()
            == Some("1")
    })
}
fn capture_expert_stats(
    gpu: &Gpu,
    layer_idx: u16,
    k: usize,
    hidden: usize,
    down_expanded: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
) {
    let dn = match gpu.download_f32(down_expanded) {
        Ok(v) => v,
        Err(_) => return,
    };
    let ti = match gpu.download_f32(topk_indices) {
        Ok(v) => v,
        Err(_) => return,
    };
    let tw = match gpu.download_f32(topk_weights) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut guard = EXPERT_STATS.lock().unwrap();
    let m = guard.get_or_insert_with(std::collections::HashMap::new);
    for krank in 0..k {
        if krank >= ti.len() || krank >= tw.len() {
            break;
        }
        let e = (ti[krank].to_bits() as i32) as u16; // i32-in-F32 alias
        let w = tw[krank] as f64;
        let base = krank * hidden;
        if base + hidden > dn.len() {
            break;
        }
        let mut sq = 0.0f64;
        for j in 0..hidden {
            let x = dn[base + j] as f64;
            sq += x * x;
        }
        let norm = sq.sqrt();
        let ent = m.entry((layer_idx, e)).or_insert((0, 0.0, 0.0, 0.0));
        ent.0 += 1;
        ent.1 += w;
        ent.2 += norm;
        ent.3 += w * norm;
    }
}
/// Dump the accumulated per-(layer,expert) REAP stats to a TSV. Called from
/// eval harnesses when HIPFIRE_MOE_EXPERT_STATS_OUT is set.
pub fn dump_expert_stats(path: &str) {
    let guard = EXPERT_STATS.lock().unwrap();
    let m = match guard.as_ref() {
        Some(m) if !m.is_empty() => m,
        _ => {
            eprintln!("expert_stats: empty (capture not enabled?)");
            return;
        }
    };
    let mut rows: Vec<_> = m.iter().collect();
    rows.sort_by_key(|((l, e), _)| (*l, *e));
    let mut out = String::from("layer\texpert\tcount\tsum_gate\tsum_norm\tsum_contrib\n");
    for ((l, e), (c, sg, sn, sc)) in rows {
        out.push_str(&format!("{l}\t{e}\t{c}\t{sg:.6}\t{sn:.6}\t{sc:.6}\n"));
    }
    match std::fs::write(path, out) {
        Ok(_) => eprintln!("expert_stats: wrote {path} ({} layer×expert rows)", m.len()),
        Err(e) => eprintln!("expert_stats: write failed {path}: {e}"),
    }
}

fn moe_ffn_decode_impl(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    s: &MoeScratchRef<'_>,
    x_rot_prerotated: bool,
    // EP (Ship 6 substrate-EP). `ep_routed_out = Some(partial)` redirects the
    // routed combine + shared-down into a zeroed partial (the EP executor
    // all-reduces it and adds into x_residual once); `None` = single-GPU into
    // x_residual (byte-identical). `ep_skip_shared` skips the shared-expert
    // down on rank>0 so the replicated shared expert is summed once.
    ep_routed_out: Option<&GpuTensor>,
    ep_skip_shared: bool,
    defer_routed_combine: bool,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    // SP2: if a layer's experts span >1 quant tier (e.g. a re-quant overlay
    // bumped some experts), expose the per-expert tier tables so dispatch
    // buckets by tier. The common case (a uniform layer — or paged mode where
    // `experts` is empty) yields None → unchanged uniform fast path.
    let (per_expert_gate_up, per_expert_down) = per_expert_tier_tables(&ffn.experts);
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        shared_expert_down: ffn.shared_expert.down.gpu_dtype,
        experts_all_gate_up_mq4: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256),
        routed_gate_up: ffn
            .experts
            .first()
            .map(|e| e.gate_up.gpu_dtype)
            .unwrap_or(DType::F32),
        routed_down: ffn
            .experts
            .first()
            .map(|e| e.down.gpu_dtype)
            .unwrap_or(DType::F32),
        // Single source of truth: the tag table is built iff experts carry
        // mixed down dtypes, so its presence == the mixed-per-expert flag.
        routed_has_mixed_experts: ffn.expert_dtype_tags.is_some(),
        has_paro_shared: ffn.paro_shared.is_some(),
        per_expert_gate_up,
        per_expert_down,
    };
    // Resolution is owned by the MoeFamily (Ship 4.1). The model passes only
    // the dtype snapshot + k; the executor computes MoeResolution from MoeDtypes.

    // Per-expert (gate_up, down) refs for the generic CPU-top-K fallback in
    // `run_moe_decode` (k != 8 OR routed dtype not indexable). Empty in paged
    // mode (`ffn.experts` is empty — only the indexed GPU-top-K path runs
    // there), matching master's `ffn.experts[..]` indexing requirement.
    let routed_experts: Vec<(
        hipfire_dispatch::families::gemv::WeightRef<'_>,
        hipfire_dispatch::families::gemv::WeightRef<'_>,
    )> = ffn
        .experts
        .iter()
        .map(|e| (e.gate_up.dispatch_ref(), e.down.dispatch_ref()))
        .collect();

    let moe_params = hipfire_dispatch::families::moe::MoeParams {
        dtypes: moe_dtypes,
        batch_size: 1,
        hidden,
        mi,
        smi,
        k,
        n_exp,
        norm_topk_prob: config.norm_topk_prob,
        x_rot_prerotated,
        defer_routed_combine,
        layer_idx: ffn.layer_idx,
        x_norm,
        x_residual,
        // EP (Ship 6 substrate-EP): threaded from moe_ffn_decode_impl params.
        // None/false (single-GPU) = byte-identical; Some(partial)/skip_shared
        // come from moe_ffn_dispatch_ep via run_layer_program_ep.
        routed_out: ep_routed_out,
        skip_shared: ep_skip_shared,
        router: ffn.router.dispatch_ref(),
        shared_expert_gate: ffn.shared_expert_gate.dispatch_ref(),
        shared_gate_w: ffn.shared_expert.gate.dispatch_ref(),
        shared_up_w: ffn.shared_expert.up.dispatch_ref(),
        shared_down_w: ffn.shared_expert.down.dispatch_ref(),
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        // Route A MoE-AWQ: `Some` only on per-expert-AWQ .hfq files (the
        // HIPFIRE_MOE_AWQ kill-switch is applied once at load in load_moe_ffn,
        // not per-token). `None` ⇒ plain silu+rotate (byte-identical).
        expert_down_awq_ptrs: ffn.expert_down_awq_ptrs.as_ref(),
        // Per-expert mixed-precision decode: `Some` only on graded mixed
        // files; drives the merged dtype-tag-branched down kernel + forces
        // the shared combine. `None` ⇒ uniform path (byte-identical).
        expert_dtype_tags: ffn.expert_dtype_tags.as_ref(),
        routed_gate_up_k: ffn.experts.first().map_or(0, |e| e.gate_up.k),
        routed_down_m: ffn.experts.first().map_or(0, |e| e.down.m),
        routed_down_k: ffn.experts.first().map_or(0, |e| e.down.k),
        routed_experts: &routed_experts,
        routed_gate_up_paro: ffn.experts.first().and_then(|e| {
            e.gate_up
                .paro
                .as_ref()
                .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                    pairs: &p.pairs,
                    theta: &p.theta,
                    scales: &p.channel_scales,
                    krot: p.krot as usize,
                })
        }),
        routed_down_paro: ffn.experts.first().and_then(|e| {
            e.down
                .paro
                .as_ref()
                .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                    pairs: &p.pairs,
                    theta: &p.theta,
                    scales: &p.channel_scales,
                    krot: p.krot as usize,
                })
        }),
        router_logits: s.router_logits,
        scalar_buf: s.scalar_buf,
        x_rot_local: s.x_rot_local,
        gate_up_buf: s.gate_up_buf,
        gate_buf: s.gate_buf,
        up_buf: s.up_buf,
        ffn_hidden: s.ffn_hidden,
        ffn_out: s.ffn_out,
        gate_batch: s.gate_batch,
        up_batch: s.up_batch,
        rot_batch: s.rot_batch,
        topk_indices: s.topk_indices,
        topk_weights: s.topk_weights,
        down_expanded: s.down_expanded,
    };
    // Build one DispatchCtx per token (the family threads it through every
    // inner GEMV — no internal DispatchCtx::new reconstructions).
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    hipfire_runtime::llama::moe_family()
        .run(&ctx, gpu, &moe_params)
        .map_err(HipError::from)?;
    if expert_stats_enabled() {
        capture_expert_stats(
            gpu,
            ffn.layer_idx,
            k,
            hidden,
            s.down_expanded,
            s.topk_indices,
            s.topk_weights,
        );
    }
    Ok(())
}

// ─── Forward pass (decode, one token at a time) ─────────────────────────

/// Run one token through the Qwen3.5 model. Returns logits.
/// For DeltaNet layers, updates state in-place (S matrix + conv ring buffer).
/// For full attention layers, uses KV cache like standard transformer.
pub fn forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;

    // Embedding lookup
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Shared forward pass — returns logits as CPU Vec<f32>.
fn forward_from_x(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let logits_gpu = forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)?;
    let logits_data = gpu.download_f32(&logits_gpu)?;
    gpu.free_tensor(logits_gpu)?;
    Ok(logits_data)
}

/// Shared forward pass — returns logits as GPU tensor (no download).
/// Shared forward pass — returns logits as GPU tensor (no download).
/// Caller must free the returned tensor.
///
/// Delegates to `forward_scratch_layers` via a temporary `Qwen35Scratch`,
/// ensuring test/demo paths exercise the same pipeline code as production.
/// NOT production-representative for benchmarking: allocates and frees a full
/// scratch bundle per call. Use `forward_scratch` with a persistent scratch
/// for perf measurement. Per-layer `DEBUG_LAYERS` trace and `trace_finite`
/// "qkvza" checkpoint are not emitted in this path — they are available
/// via `dump_hidden_localize` in the scratch path under HIPFIRE_DUMP_HIDDEN.
fn forward_from_x_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let required_tokens = checked_kv_end(pos, 1, "forward_from_x_gpu")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;

    // Allocate a temporary scratch bundle. repeat_window=1 (unused in this path).
    // kv_max_seq=8192 matches Qwen35Scratch::new default — sufficient for
    // test/demo single-token forward; these callers don't prefill.
    let scratch = Qwen35Scratch::new(gpu, config, 1)?;

    // Copy input embedding into scratch.x
    gpu.hip.memcpy_dtod(&scratch.x.buf, &x.buf, dim * 4)?;
    gpu.free_tensor(x)?;

    // Set position buffer
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // DEBUG_LAYERS: dump embedding + per-layer norms (same as old forward_from_x_gpu)
    let debug_layers = std::env::var("DEBUG_LAYERS").is_ok();
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "EMB: first4=[{:.6},{:.6},{:.6},{:.6}] norm={norm:.4}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Run the production pipeline
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, &scratch, None,
    )?;

    // DEBUG_LAYERS: dump per-layer residual norms
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "POST: first4=[{:.4},{:.4},{:.4},{:.4}] norm={norm:.2}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Copy logits out of scratch before freeing — the returned tensor must
    // outlive the scratch bundle.
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    gpu.hip
        .memcpy_dtod(&logits.buf, &scratch.logits.buf, config.vocab_size * 4)?;

    // Free scratch (all pre-allocated buffers)
    scratch.free_gpu(gpu);

    Ok(logits)
}

/// Pre-allocated scratch buffers for zero-alloc qwen35 forward + GPU sampling.
pub struct Qwen35Scratch {
    // Persistent state
    pub x: GpuTensor,                      // [dim]
    pub tmp: GpuTensor,                    // [dim]
    pub pos_buf: hip_bridge::DeviceBuffer, // 4 bytes

    // DeltaNet temporaries (reused across layers)
    pub dn_qkv: GpuTensor,      // [qkv_dim]
    pub dn_z: GpuTensor,        // [v_dim]
    pub dn_alpha: GpuTensor,    // [n_v_heads]
    pub dn_beta: GpuTensor,     // [n_v_heads]
    pub dn_conv_out: GpuTensor, // [qkv_dim]
    pub dn_q: GpuTensor,        // [v_dim] (after repeat-interleave)
    pub dn_k: GpuTensor,        // [v_dim]
    pub dn_v: GpuTensor,        // [v_dim]
    pub dn_q_raw: GpuTensor,    // [k_dim] (before repeat)
    pub dn_k_raw: GpuTensor,    // [k_dim]
    pub dn_attn_out: GpuTensor, // [v_dim]
    pub dn_normed: GpuTensor,   // [v_dim]

    // FullAttn temporaries (reused across layers)
    pub fa_q_full: GpuTensor,   // [n_heads * head_dim * 2]
    pub fa_q: GpuTensor,        // [n_heads * head_dim]
    pub fa_gate: GpuTensor,     // [n_heads * head_dim]
    pub fa_k: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim]

    // Shared (used by both layer types)
    pub o: GpuTensor,          // [dim]
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up: GpuTensor,         // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Sampling
    pub logits: GpuTensor,     // [vocab_size]
    pub sample_buf: GpuTensor, // [2] — token_id + rng
    pub repeat_buf: GpuTensor, // [repeat_window]

    // MagnumQuant rotation scratch: FWHT(x) shared across Q/K/V (or gate/up, etc).
    // DeltaNet's output projection consumes v_dim values, which can exceed both
    // dim and the unused dense hidden_dim on MoE checkpoints.
    pub x_rot: GpuTensor, // [max(dim, hidden_dim, v_dim)]

    // Flash attention partials buffer for tile+reduce 2-kernel path.
    // Size: n_heads * max_tiles * (2 + head_dim) floats.
    pub flash_partials: GpuTensor,
    // Flash attention tri-state (applies to Q8 path; asym modes are flash-only):
    //   0 = never      force non-flash at all contexts (except >15K sanity)
    //   1 = auto       (default) flash kicks in at ctx >= 2048
    //   2 = always     force flash at all contexts
    pub flash_mode: u8,

    // MoE scratch (allocated only when config.num_experts > 0). Pre-allocated
    // so moe_ffn_decode can be captured by hipGraph — the per-layer allocs
    // it used to do violated the "no allocator ops while capturing" rule.
    pub moe_router_logits: Option<GpuTensor>, // [num_experts]
    pub moe_scalar_buf: Option<GpuTensor>,    // [1] shared-expert gate scalar
    pub moe_x_rot: Option<GpuTensor>,         // [dim]
    pub moe_gate_up_buf: Option<GpuTensor>,   // [2*max_inter]   fallback path
    pub moe_gate_buf: Option<GpuTensor>,      // [max_inter]     fallback path
    pub moe_up_buf: Option<GpuTensor>,        // [max_inter]     fallback path
    pub moe_ffn_hidden: Option<GpuTensor>,    // [max_inter]     fallback path
    pub moe_ffn_out: Option<GpuTensor>,       // [dim]           fallback path
    pub moe_gate_batch: Option<GpuTensor>,    // [k × mi]
    pub moe_up_batch: Option<GpuTensor>,      // [k × mi]
    pub moe_rot_batch: Option<GpuTensor>,     // [k × mi]
    /// Phase 2b: GPU-side top-K outputs (kept on-device so moe_ffn_decode
    /// can stay in a graph-capturable stream).
    pub moe_topk_indices: Option<GpuTensor>, // [k] i32 stored as f32 alias
    pub moe_topk_weights: Option<GpuTensor>,  // [k] f32
    // Atomic-free MoE down expansion buffer for decode — payload [k × dim]
    // plus ceil(dim/4) f32-sized counter slots for the optional last-arriver
    // combine. The full allocation is zeroed once; each fused dispatch resets
    // the counters it consumes.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched` (batch_size=1) in `moe_ffn_decode_impl`'s
    // use_gpu_topk path. Replaces the K_TOP-way atomicAdd that introduced
    // non-deterministic wavefront-order-dependent FP rounding under hipGraph
    // replay (task #100).
    pub moe_down_expanded: Option<GpuTensor>,

    // Optional long-prefill scratch. Default is None to preserve VRAM
    // footprint; set HIPFIRE_PREFILL_REUSE_PBS=1 to allocate and reuse it.
    pub prefill_batch: Option<PrefillBatchScratch>,
}

fn qwen35_x_rot_len(dim: usize, hidden_dim: usize, v_dim: usize) -> usize {
    dim.max(hidden_dim).max(v_dim)
}

impl Qwen35Scratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize) -> HipResult<Self> {
        // Flash partials are sized for up to 8192 ctx. Override via new_with_kv_max.
        Self::new_with_kv_max(gpu, config, repeat_window, 8192)
    }

    pub fn new_with_kv_max(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,

            dn_qkv: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_z: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_alpha: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_beta: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_conv_out: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_q: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_k: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_v: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_q_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_k_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_attn_out: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_normed: gpu.alloc_tensor(&[v_dim], DType::F32)?,

            fa_q_full: gpu.alloc_tensor(&[q_dim * 2], DType::F32)?,
            fa_q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_gate: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,

            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,

            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            sample_buf: gpu.alloc_tensor(&[2], DType::F32)?,
            repeat_buf: gpu.alloc_tensor(&[repeat_window], DType::F32)?,
            x_rot: gpu.alloc_tensor(
                &[qwen35_x_rot_len(dim, config.hidden_dim, v_dim)],
                DType::F32,
            )?,

            // Flash attention partials: enough for the smallest tile used by
            // Q8 decode experiments and the fixed tile_size=128 paths.
            // n_heads * max_tiles * (2 + head_dim) floats per batched query
            // position; total buffer = batch_mult × per-position-bytes.
            //
            // batch_mult is the maximum query positions a single FA dispatch
            // can fit; the dispatcher (`launch_asym_flash_batched`) reads the
            // buffer's actual capacity at call time and auto-chunks larger
            // prefill batches into multiple sub-launches. So a lower
            // batch_mult here trades ~linear extra dispatch overhead on
            // prefill (PREFILL_MAX_BATCH=256 → ceil(256/batch_mult) calls per
            // FA layer) for ~linearly less VRAM at long context.
            //
            // The per-position size scales with kv_max_seq (= physical_cap
            // post-eviction), and that scaling is what made #85 visible: at
            // max_seq=170k, no CASK, 27B (n_heads=24, head_dim=256) the old
            // batch_mult=64 → 2.1 GB just for these partials, exceeding VRAM
            // headroom on 24 GB cards. Cutting batch_mult by 4× (16) keeps
            // the prefill chunking moderate while saving 1.6 GB at that
            // worst-case shape; CASK-on workloads (small physical_cap) are
            // unaffected because the buffer is already tiny there.
            //
            // Override with HIPFIRE_FLASH_PARTIALS_BATCH for tuning. Power of
            // two preferred (matches FA dispatcher chunking).
            flash_partials: {
                let tile_size = rdna_compute::attention::q8_flash_tile_size(
                    &gpu.arch,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    kv_max_seq,
                )
                .min(128);
                let max_tiles = (kv_max_seq + tile_size - 1) / tile_size;
                let batch_mult = hipfire_runtime::config::get()
                    .flash_partials_batch
                    .filter(|&n| n >= 1 && n <= PREFILL_MAX_BATCH)
                    .unwrap_or(16);
                gpu.alloc_tensor(
                    &[batch_mult * config.n_heads * max_tiles * (2 + config.head_dim)],
                    DType::F32,
                )?
            },
            // Flash attention tri-state for the Q8 path. Asym modes always
            // flash regardless.
            //   HIPFIRE_ATTN_FLASH=never|0|off    → non-flash at all contexts
            //   HIPFIRE_ATTN_FLASH=auto|1|on      → flash at ctx >= 2048
            //   HIPFIRE_ATTN_FLASH=always|2|force → flash at all contexts
            //
            // Default on gfx11/gfx12 (graph-capable archs): `2` (always
            // flash). On other archs: `1` (auto). The capture path at
            // qwen35.rs:8199 hard-wires `use_flash = capture_mode || ...`
            // because attention_q8_0_kv has variable block_size + variable
            // shared-mem (not capture-safe). Without an always-flash default
            // on capture-capable archs, direct mode at small ctx silently
            // uses attention_q8_0_kv while a captured-and-replayed forward
            // uses attention_flash_q8_0 — same math, different fp32
            // reduction order, observed as ~0.44 logit delta direct-vs-graph
            // on shisa-Qwen3.6-A3B-PARO (see
            // .scratch/hipgraph-moe-drift-audit.md Part A). Aligning the
            // default flips both paths to `attention_flash_q8_0` and makes
            // direct vs graph byte-identical at the cost of moving small-
            // context decode off the non-flash kernel (~few % attention
            // perf hit, small contribution to total MoE decode time).
            // Honors HIPFIRE_ATTN_FLASH=never|0|off as an explicit override
            // for users who prefer the non-flash kernel and don't intend
            // to use graph capture.
            flash_mode: match hipfire_runtime::config::get().attention_flash_mode.as_str() {
                "never" | "0" | "off" => 0,
                "always" | "2" | "force" => 2,
                _ => {
                    let graph_capable_arch =
                        gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
                    if graph_capable_arch {
                        2
                    } else {
                        1
                    }
                }
            },

            moe_router_logits: None,
            moe_scalar_buf: None,
            moe_x_rot: None,
            moe_gate_up_buf: None,
            moe_gate_buf: None,
            moe_up_buf: None,
            moe_ffn_hidden: None,
            moe_ffn_out: None,
            moe_gate_batch: None,
            moe_up_batch: None,
            moe_rot_batch: None,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_down_expanded: None,
            prefill_batch: None,
        })
        .and_then(|mut s| {
            // Allocate MoE scratch only for MoE configs. Done after the
            // main struct init so these Options start as None for dense
            // models and never cost VRAM there.
            if config.num_experts > 0 {
                let hidden = config.dim;
                let n_exp = config.num_experts;
                let mi = config.moe_intermediate_size;
                let smi = config.shared_expert_intermediate_size;
                let max_inter = mi.max(smi);
                let k = config.num_experts_per_tok;
                s.moe_router_logits = Some(gpu.alloc_tensor(&[n_exp], DType::F32)?);
                s.moe_scalar_buf = Some(gpu.alloc_tensor(&[1], DType::F32)?);
                s.moe_x_rot = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_up_buf = Some(gpu.alloc_tensor(&[2 * max_inter], DType::F32)?);
                s.moe_gate_buf = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_up_buf = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_hidden = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_out = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_up_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_rot_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                // i32 topk_indices stored in an F32 tensor (same byte width).
                // The kernel that writes it casts the buffer to int*, and the
                // indexed MoE GEMV kernels read it as int*.
                s.moe_topk_indices = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                s.moe_topk_weights = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                // Atomic-free decode MoE down payload plus reusable counter tail.
                s.moe_down_expanded =
                    Some(gpu.zeros(&[k * hidden + hidden.div_ceil(4)], DType::F32)?);
                // Pre-warm MQ FWHT sign tables (otherwise the lazy init in
                // ensure_mq_signs fires during the first moe_ffn_decode and
                // blows up hipGraph capture with a hipMalloc-in-capture
                // error). Idempotent if already computed.
                gpu.ensure_mq_signs()?;
            }
            if hipfire_config::developer_var("HIPFIRE_PREFILL_REUSE_PBS")
                .ok()
                .as_deref()
                == Some("1")
            {
                let max_batch = hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&v| v >= 2)
                    .unwrap_or(PREFILL_MAX_BATCH);
                s.prefill_batch = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
            }
            Ok(s)
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.tmp);
        // pos_buf is held as a raw DeviceBuffer and dropped via gpu.hip.free
        // directly (free_tensor would have bound the thread internally).
        // Bind explicitly so HIP affinity doesn't depend on the order of
        // preceding free_tensor calls.
        let _ = gpu.bind_thread();
        let _ = gpu.hip.free(self.pos_buf);
        for t in [
            self.dn_qkv,
            self.dn_z,
            self.dn_alpha,
            self.dn_beta,
            self.dn_conv_out,
            self.dn_q,
            self.dn_k,
            self.dn_v,
            self.dn_q_raw,
            self.dn_k_raw,
            self.dn_attn_out,
            self.dn_normed,
            self.fa_q_full,
            self.fa_q,
            self.fa_gate,
            self.fa_k,
            self.fa_v,
            self.fa_attn_out,
            self.o,
            self.gate_ffn,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_buf,
            self.repeat_buf,
            self.x_rot,
            self.flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
        // MoE scratch — only present for MoE configs.
        for t in [
            self.moe_router_logits,
            self.moe_scalar_buf,
            self.moe_x_rot,
            self.moe_gate_up_buf,
            self.moe_gate_buf,
            self.moe_up_buf,
            self.moe_ffn_hidden,
            self.moe_ffn_out,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
        ] {
            if let Some(buf) = t {
                let _ = gpu.free_tensor(buf);
            }
        }
        if let Some(pbs) = self.prefill_batch {
            pbs.free_gpu(gpu);
        }
    }
}

/// Per-device scratch bundle for the multi-GPU forward path. Each device gets
/// its own `Qwen35Scratch` because the residual stream `s.x` (and `s.logits`)
/// must live on the device executing the current band's layers — cross-band
/// boundaries copy `s.x` between devices via `Gpus::boundary_copy`. `s.logits`
/// is also allocated per-device for simplicity (~600 KB each at vocab=152K)
/// even though only the output device's `s.logits` is consumed post-loop.
pub struct Qwen35ScratchSet {
    pub per_device: Vec<Qwen35Scratch>,
}

impl Qwen35ScratchSet {
    pub fn new_with_kv_max_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let mut per_device = Vec::with_capacity(gpus.devices.len());
        for dev_idx in 0..gpus.devices.len() {
            let g = &mut gpus.devices[dev_idx];
            per_device.push(Qwen35Scratch::new_with_kv_max(
                g,
                config,
                repeat_window,
                kv_max_seq,
            )?);
        }
        Ok(Self { per_device })
    }

    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        for (dev_idx, scratch) in self.per_device.into_iter().enumerate() {
            scratch.free_gpu(&mut gpus.devices[dev_idx]);
        }
    }
}

fn checked_kv_end(start_pos: usize, token_count: usize, site: &str) -> HipResult<usize> {
    start_pos.checked_add(token_count).ok_or_else(|| {
        HipError::new(
            0,
            &format!("{site}: KV token range overflow ({start_pos} + {token_count})"),
        )
    })
}

#[inline]
fn ar_graph_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_AR_GRAPH_TRACE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Zero-alloc forward pass using pre-allocated scratch buffers.
/// Logits stay on GPU in scratch.logits. Returns nothing — caller uses scratch.logits.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch")?;
    // Grow before any possible AR graph capture/replay. Stable virtual
    // addresses keep existing graph pointer arguments valid.
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;
    // hipGraph capture for MoE was previously gated off-by-default behind
    // HIPFIRE_GRAPH_MOE=1 because of a known drift bug (task #100): under
    // capture, A3B accumulated a per-step ~1-ULP delta that compounded
    // through the KV cache + GDN state and crossed the top-1 margin at
    // step ~7 (q8 KV) or ~114 (asym3 KV), producing visible token-loop
    // attractors by step 30-50 ("- **One**\n- **One**\n…").
    //
    // Root cause (fixed 2026-05-21): `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`
    // used K_TOP=8 concurrent `atomicAdd` writes per output row. FP32
    // addition is non-associative, so the final bits depend on wavefront
    // scheduling order. Under hipGraph replay that order differs from
    // direct execution (graph scheduling pipelines kernels differently),
    // introducing the systematic per-step delta. The kernel's own header
    // (`kernels/src/gemv_hfq4g256_moe_down.hip:14-19`) had already flagged
    // this non-determinism but rated it negligible based on the
    // direct-only smoke test — capture amplifies the effect.
    //
    // Fix: the MoE FFN decode path now uses the atomic-free expand+combine
    // pattern already used in prefill (`forward_prefill_batch_with_pbs`
    // L5217-5232): `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
    // writes one row per (expert-rank, m), then `moe_down_combine_k8_batched`
    // sums K_TOP slots into x_residual in a fixed iteration order. The
    // resulting MoE FFN output is byte-deterministic under both direct
    // execution and hipGraph replay.
    //
    // `experimental.graph.moe` remains the explicit policy control. The atomic
    // fix is necessary but not sufficient — the CPU-topK fallback path
    // (when not all gate-side MoE weights are MQ4G256, e.g. router=Q8 per
    // the post-2026-04 router-attractor fix) calls `download_f32(router_logits)`,
    // a sync D2H that fails under graph capture with hipError 906. Until
    // that D2H is migrated to a capture-safe equivalent, opting in only
    // works for models where the runtime takes the use_gpu_topk path.
    //
    // Reproducer used to characterize the fix:
    //   hipfire config set experimental.graph.forward true
    //   hipfire config set experimental.graph.moe true
    //   HIPFIRE_SMOKE_KV=q8 \
    //   HIPFIRE_SMOKE_MODE=chat HIPFIRE_SMOKE_STEPS=200 \
    //   HIPFIRE_SMOKE_PROMPT="Count from one to twenty in English." \
    //   ./target/release/examples/a3b_smoke_forward <uniform-mq4-a3b>
    //
    // Graph policy is resolved once from TOML before GPU initialization and is
    // carried directly by the GPU feature snapshot.
    // Default-ON (2026-06-16): cross-arch A/B validated — gfx12 +4.2% A3B mq4
    // decode, coherence-gate clean (fluent at decode pos 1-4 on current ROCm).
    // Opt out with `experimental.graph.moe = false`. Both root causes of the prior
    // crash/drift are fixed:
    //   1. atomicAdd drift (task #100, 2026-05-21): expand+combine pattern.
    //   2. CPU-topK fallback D2H: `download_f32(router_logits)` replaced by
    //      GPU `softmax_f32` + `moe_topk_renorm_k8` + small [k] D2H — fully
    //      capture-safe. Mixed-kmap A3B (Q8 router, post-PR #199) no longer
    //      crashes with hipError 906 under AR graph capture.
    // Validated + flipped to default-on 2026-06-16.
    let allow_moe = gpu.flags.graph_moe;
    // hipGraph per-forward-pass capture/replay default policy:
    //   - gfx12 (RDNA4): default-ON. +2.4-2.7% decode on 9B Qwen 3.5
    //     MFP4G32 (5-run mean, all positive, tight variance, 2026-05-11).
    //   - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and
    //     0.8B HFP4G32 on 7900 XTX (5-run mean per model, all positive,
    //     variance 1.001-1.010×, 2026-05-11). Smaller win than gfx12 —
    //     gfx11 has less per-launch overhead to amortize — but real
    //     and consistent across model sizes.
    //   - other archs (RDNA1/2, CDNA): default-OFF (opt-in via
    //     `experimental.graph.forward = true`) since not yet A/B'd on those.
    //   - MoE configs follow `experimental.graph.moe`. The ~30-50-token
    //     attractor drift in the use_gpu_topk MoE down step was fixed
    //     2026-05-21 (task #100 — atomicAdd → expand+combine), but the
    //     CPU-topK fallback's `download_f32(router_logits)` D2H sync
    //     remains capture-incompatible, so mixed-kmap A3B (post-PR #199)
    //     can crash under graph capture even with the fix. Once that
    //     D2H is migrated to a capture-safe path, the MoE default can
    //     be flipped to follow the arch defaults.
    // An explicit `experimental.graph.forward = false` always wins.
    let graph_override = gpu.flags.graph_forward;
    let graph_arch_default = gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
    let graph_enabled = graph_override.unwrap_or(graph_arch_default);
    // AR-forward hipGraph RE-ENABLED (2026-06-16) — the kernarg-snapshot attractor
    // is re-verified GONE on current ROCm (HIP 7.x, coherence-gate clean, fluent at
    // decode pos 1-4 on gfx12 A3B mq4). Opt out with
    // `experimental.graph.ar = false`. The prior
    // 2026-05-15 disable rationale is retained below; it SUPERSEDED the
    // arch-default re-enable merged from master (`graph_enabled` above is kept
    // live so the graph policy and kill switch stay wired for when the
    // path is flipped back on). Empirically on ROCm 7.2.2 + gfx11 +
    // Qwen3.5-27B mq4, both replay AND capture+launch produce a token-0
    // attractor outside very narrow conditions:
    //   - Capture+launch at position 2 (after 1 direct warmup) → `!!!!!`
    //   - Capture+launch at position 4 (after 3 direct warmups) → correct
    //   - Replay of a working capture (any position) → `!!!!!` from pos+1 on
    // The kernarg-snapshot bug isn't fixable by warmup tuning OR caller-driven
    // commit gating (`end_decode_turn()`); both fail empirically. Master's
    // task-#100 fix targets MoE drift, NOT this AR-forward attractor, so the
    // merge does not clear the disable. Until the capture/replay attractor is
    // re-verified gone on current ROCm (7.13) via the coherence gate, AR
    // forward is direct-only. Policy infra (`ar_forward_kernel_dirty`,
    // `ar_forward_replay_enabled`, `end_decode_turn()`, `drop_captured_graph()`)
    // is preserved on Gpu so the path can be flipped on once the bug is fixed.
    // RE-VERIFY GATE (2026-06-12): the AR graph policy flips the 2026-05-15
    // disable back on to re-test the capture/replay attractor on current ROCm
    // (HIP 7.x). Default OFF preserves the direct-only behavior. When set, the
    // path still honors the HIPFIRE_GRAPH kill switch + arch default via
    // `graph_enabled`.
    let ar_graph_test = gpu.flags.graph_ar;
    // AR-forward hipGraph eligibility. Plain sequential single-token AR decode
    // is eligible BY DEFAULT (the consume below resets to true); spec-decode /
    // MTP re-seed and the verify/prefill batch path explicitly set this FALSE
    // right before their `forward_scratch` call so the plain-AR graph can never
    // capture or replay in a non-sequential context. An ineligible call also
    // INVALIDATES any captured graph (forces re-capture on the next plain call).
    let graph_eligible = std::mem::replace(&mut gpu.graphs.ar_graph_eligible, true);
    // Redline's plain-AR capture/replay has the same eligibility contract as
    // the AR HipGraph. MTP/spec re-seed and verify calls must not contaminate
    // or consume the immutable single-token replay sequence.
    gpu.replay.set_forward_eligible(graph_eligible);
    gpu.replay
        .begin_auto_capture_if_armed()
        .map_err(|reason| HipError::new(0, reason))?;
    if ar_graph_test && !graph_eligible {
        gpu.graphs.ar_forward_replay_enabled = false;
        gpu.graphs.ar_forward_kernel_dirty = true;
    }
    // MoE models require `experimental.graph.moe` in addition to the
    // arch/kill-switch guards. Dense models (num_experts==0) are unaffected.
    let use_graph = ar_graph_test
        && graph_enabled
        && graph_eligible
        && !gpu.replay.is_enabled()
        && (config.num_experts == 0 || allow_moe);
    let _ = gpu.graphs.ar_forward_replay_enabled; // suppress unused warning

    // Embedding lookup into scratch.x (always direct, changes per token)
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    let pos_i32 = pos as i32;
    if gpu.replay.should_route_aql() {
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        let replay = unsafe { gpu.replay.replay_linear_aql(pos) };
        return match replay {
            Ok(_) => Ok(()),
            Err(reason) => {
                gpu.replay
                    .poison(format!("prepared AQL replay failed: {reason}"));
                Err(HipError::new(0, &reason))
            }
        };
    }
    if gpu.replay.should_route_pm4() {
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        let replay = unsafe { gpu.replay.replay_pm4(pos) };
        return match replay {
            Ok(_) => Ok(()),
            Err(reason) => {
                gpu.replay
                    .poison(format!("prepared PM4 replay failed: {reason}"));
                Err(HipError::new(0, &reason))
            }
        };
    }
    if use_graph && gpu.graphs.ar_forward_replay_enabled && gpu.graphs.graph_exec.is_some() {
        // ── Replay path: graph captured + kernels clean. Cheapest path: pos
        // memcpy + graph replay. The graph is position-agnostic (pos via
        // pos_buf), so replay is correct across positions and requests as long
        // as the buffers are the plain-AR continuation — which the spec markers
        // + verify invalidation guarantee. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
        if ar_graph_trace_enabled() {
            eprintln!("[qwen-ar-graph] replay pos={pos}");
        }
    } else if use_graph && gpu.graphs.ar_forward_kernel_dirty {
        // ── Direct path (kernel-dirty): kernels are dirty (init or post-
        // model-load). Capture would trip "hipMalloc not permitted under
        // stream capture" on the first inline JIT. Mark clean after a
        // successful direct dispatch so subsequent calls can capture. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
        gpu.graphs.ar_forward_kernel_dirty = false;
    } else if use_graph {
        // ── Capture + launch: kernels are clean but caller has not committed
        // a replay yet (or graph_exec is None). Drop any prior captured graph,
        // record a fresh one, and launch it for this forward's output. After
        // the caller signals end_decode_turn(), the most recent capture is
        // promoted to the replay graph for the next decode turn. ──
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        gpu.graphs.begin_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        )?;
        forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
        gpu.graphs.end_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        )?;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
        if ar_graph_trace_enabled() {
            eprintln!("[qwen-ar-graph] capture pos={pos}");
        }
        // Intra-generate replay (2026-06-12): promote this fresh capture to the
        // replay graph immediately so the NEXT token replays (cheap: pos memcpy
        // + graph_launch) instead of re-capturing + re-instantiating every
        // token. The per-token instantiate is why the path "did nothing" — the
        // daemon never calls end_decode_turn() to enable replay.
        gpu.graphs.ar_forward_replay_enabled = true;
    } else {
        // ── Direct path (graph not eligible: arch / MoE config) ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
    }
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip.device_synchronize()?;
        gpu.replay
            .finish_capture()
            .map_err(|reason| HipError::new(0, reason))?;
        let prepare = if gpu.replay.uses_pm4_transport() {
            let launches = gpu.replay.recorded_launches().len();
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql(gpu.device_id as usize)
                .map(|_| ())
        };
        if let Err(reason) = prepare {
            gpu.replay
                .poison(format!("Redline prepare after warmup failed: {reason}"));
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
    Ok(())
}

/// Populate the two inputs that intentionally stay outside a prepared decode
/// replay: the token embedding in `scratch.x` and the position scalar buffer.
/// Redline uses this exact boundary before its AQL packet batch.
pub fn prepare_scratch_inputs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        other => {
            return Err(HipError::new(
                0,
                &format!("unsupported embedding format for Redline: {other:?}"),
            ));
        }
    }
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
    Ok(())
}

/// Per-layer batched intermediates used by `forward_prefill_batch`. Each
/// row is one token in the batch; rows are contiguous [N × K] blocks so
/// all kernels can treat them as row-major matrices.
///
/// Allocated lazily on the first batched prefill call that takes the MQ4
/// fast path — models that never hit that path (HF4 weights, FA-only
/// models, short prompts) never pay the VRAM cost. Sized to `max_batch`;
/// longer prompts are processed in chunks of `max_batch`.
pub struct PrefillBatchScratch {
    pub max_batch: usize,

    // Residual stream and rotation scratch — all [N × dim]
    pub x_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,
    // Rmsnorm-only scratch (no FWHT). Used by MoE prefill body for Q8_0
    // weights (router + shared_expert_gate) which were quantized against
    // un-rotated input. MQ4 sibling weights read `x_rot_batch` instead.
    // Mixed-dtype MoE layers populate both buffers per `prefill_moe_ffn_body_batched`.
    pub x_norm_batch: GpuTensor,

    // LA-layer projection outputs
    pub dn_qkv_batch: GpuTensor,      // [N × qkv_dim]
    pub dn_z_batch: GpuTensor,        // [N × v_dim]
    pub dn_alpha_batch: GpuTensor,    // [N × n_v_heads]
    pub dn_beta_batch: GpuTensor,     // [N × n_v_heads]
    pub dn_q_raw_batch: GpuTensor,    // [N × k_dim] (pre repeat-interleave)
    pub dn_k_raw_batch: GpuTensor,    // [N × k_dim]
    pub dn_v_batch: GpuTensor,        // [N × v_dim]
    pub dn_q_batch: GpuTensor,        // [N × v_dim] (post repeat-interleave)
    pub dn_k_batch: GpuTensor,        // [N × v_dim]
    pub dn_attn_out_batch: GpuTensor, // [N × v_dim]
    pub dn_normed_batch: GpuTensor,   // [N × v_dim]

    // FFN intermediates [N × hidden_dim]
    pub gate_ffn_batch: GpuTensor,
    pub up_batch: GpuTensor,
    // SwiGLU output (FWHT-rotated for MQ4) feeding w_down.
    pub ffn_hidden_batch: GpuTensor,

    // FWHT-rotated dn_normed [N × v_dim] feeding wo for MQ4 weights.
    // Decode path handles this via an internal mq_x_rot scratch inside
    // weight_gemv_residual; we need an explicit batched equivalent.
    pub dn_normed_rot_batch: GpuTensor,

    // ── FullAttention batched intermediates (when FA weights are MQ4G256) ──
    // Positions array: [max_batch] i32, absolute KV positions for this chunk.
    // Uploaded once at the start of each chunk and reused by rope + kv_write
    // + attention kernels.
    pub positions: GpuTensor,
    // Depth-based RoPE angles for DDTree verify (39aa358 fix). Uploaded in
    // tree-verify mode; FA RoPE reads it instead of `positions` while KV
    // writes and attention seq_len keep the flat physical slots.
    pub rope_positions: GpuTensor,
    // Token-ids buffer feeding the batched embedding kernel. [max_batch] i32
    // stored as F32 (same dtype-cosmetic pattern as `positions`). Uploaded
    // once per batched forward and read by `embedding_lookup_hfq4g256_batched`.
    pub tokens: GpuTensor,
    // QKV projection outputs
    pub fa_q_full_batch: GpuTensor, // [N × n_heads × head_dim × 2] (Q + gate interleaved)
    pub fa_q_batch: GpuTensor,      // [N × n_heads × head_dim]
    pub fa_gate_batch: GpuTensor,   // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,      // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,      // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor, // [N × n_heads × head_dim]

    // ── MoE batched intermediates (allocated only when num_experts > 0) ──
    // All outputs of the fused 4-way router + shared-gate GEMM, plus the
    // per-token routed-expert gate/up/rot buffers consumed by the N-batched
    // indexed MoE kernels. Sized as [max_batch × {n_exp, smi, k_top×mi}].
    pub moe_router_logits_batch: Option<GpuTensor>, // [N × num_experts]
    pub moe_shared_scalar_batch: Option<GpuTensor>, // [N × 1] — raw shared_expert_gate logit
    pub moe_shared_gate_batch: Option<GpuTensor>,   // [N × smi]
    pub moe_shared_up_batch: Option<GpuTensor>,     // [N × smi]
    pub moe_shared_rot_batch: Option<GpuTensor>,    // [N × smi] — FWHT(silu(gate) * up)
    pub moe_topk_indices_batch: Option<GpuTensor>,  // [N × k_top] i32 in F32 slots
    pub moe_topk_weights_batch: Option<GpuTensor>,  // [N × k_top]
    pub moe_gate_batch: Option<GpuTensor>,          // [N × k_top × mi]
    pub moe_up_batch: Option<GpuTensor>,            // [N × k_top × mi]
    pub moe_rot_batch: Option<GpuTensor>,           // [N × k_top × mi]
    // Atomic-free MoE down expansion buffer — [N × k_top × dim] f32.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched`: the down kernel writes each
    // (token, krank) result to its own row here (no atomic), then the
    // combine kernel folds K_TOP slots into x_batch with topk_weights
    // applied. RDNA-only (atomic on GDDR is slow); the wave64/CDNA path
    // stays on the residual_scaled atomic kernel.
    pub moe_down_expanded_batch: Option<GpuTensor>,

    // Path 2 (SGLang-style scatter + grouped-WMMA-GEMM) scratch. All
    // allocated when num_experts > 0; gated at runtime by
    // HIPFIRE_MOE_GROUPED_GEMM=1. m_total_max is tile-aligned:
    // align_up(max_batch * k_top + num_experts * (BLOCK_M - 1), BLOCK_M)
    // with BLOCK_M=16.
    //
    //   moe_expert_token_counts: [num_experts] i32 (raw → padded)
    //   moe_expert_offsets:      [num_experts + 1] i32 (exclusive prefix)
    //   moe_sorted_slot_index:   [m_total_max] i32 (flat slot or -1 padding)
    //   moe_expert_tile_ids:     [m_total_max / 16] i32 (per-tile expert id)
    //   moe_y_gate_up_grouped:   [m_total_max × (2*mi)] f32 (grouped GEMM output)
    pub moe_expert_token_counts: Option<GpuTensor>,
    pub moe_expert_offsets: Option<GpuTensor>,
    pub moe_sorted_slot_index: Option<GpuTensor>,
    pub moe_inverse_perm: Option<GpuTensor>, // [total_slots] i32: flat → sorted_pos
    pub moe_expert_tile_ids: Option<GpuTensor>,
    pub moe_y_gate_up_grouped: Option<GpuTensor>, // [m_total × (2*mi)]
    pub moe_y_down_grouped: Option<GpuTensor>,    // [m_total × dim] for the down step

    // ── Tree-aware LA scratch (Phase 3b of Task #101) ──
    // Per-token S-state tape consumed by gated_delta_net_q8_tree kernel
    // when TreeVerifyCtx.parent_indices is Some. Reused across LA layers
    // since LA dispatch is serial per-cycle. Only allocated when the model
    // has LA layers (linear_num_value_heads > 0). Call sites that pass
    // parent_indices must ensure these tensors exist.
    //
    // s_tape_q8:     [max_batch × n_v_heads × head_dim × head_dim] Raw/i8
    // s_tape_scales: [max_batch × n_v_heads × head_dim] f32
    //
    // At max_batch=22, n_v_heads=16, head_dim=128 → 5.77 MB + 180 KB total.
    pub dn_s_tape_q8: Option<GpuTensor>,
    pub dn_s_tape_scales: Option<GpuTensor>,
    // FP32 per-node tape for the FP32 `StateQuant` tree-verify path. Same
    // element layout as `dn_s_tape_q8` but f32 (4×), no scales side-table.
    // TODO: gate allocation on state_quant (needs threading StateQuant into
    // `new`); currently always allocated when LA layers exist, like the Q8
    // tape. s_tape_f32: [max_batch × n_v_heads × head_dim × head_dim] f32.
    pub dn_s_tape_f32: Option<GpuTensor>,
}

impl PrefillBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_batch: usize) -> HipResult<Self> {
        // Default: allocate the spec-decode DeltaNet S-tape (tree-verify reads it).
        Self::new_opt(gpu, config, max_batch, /*cap_gdn_tape=*/ true)
    }

    /// Like [`new`], but `cap_gdn_tape` controls allocation of the per-token
    /// DeltaNet S-state tape (`dn_s_tape_*`, sized `[max_batch × n_v_heads ×
    /// value_head_dim²]`). The tape is consumed ONLY by the tree-verify
    /// (spec-decode) GDN kernels; plain prefill (`tree_parents == None`)
    /// advances the recurrent state in place and never touches it. Pass `false`
    /// for plain prefill to skip the tape — on A3B (16 value heads × 128²) it is
    /// ~10 GB at an 8k batch (dn_s_tape_f32 8.2 GB + dn_s_tape_q8 2 GB), the
    /// difference between an 8k prefill fitting and OOMing. Callers that may run
    /// tree-verify MUST pass `true` (else the `.expect()` at the consumption site
    /// panics).
    pub fn new_opt(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        max_batch: usize,
        cap_gdn_tape: bool,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        // hunt3 H-E residual: this struct literal allocates ~40 GpuTensors via
        // `?` early-returns. PrefillBatchScratch has no Drop impl (GpuTensor
        // carries no Gpu handle; free_tensor needs &mut Gpu), so a `?` failure
        // partway through would drop the already-allocated tensors WITHOUT
        // freeing them on the device — the exact intra-`new` leak the
        // cross-band H-E recovery can't reach. OOM during new() is precisely
        // when a mid-literal failure is most likely. Fix: route every alloc
        // through a ledger and, on the first error, free everything allocated
        // so far before propagating. `alloc!` records mandatory tensors;
        // `alloc_opt!` records the inner tensor of an `if cond { Some(..) }`.
        //
        // The ledger stores non-owning aliases (DeviceBuffer has no Drop and
        // GpuTensor is not Clone), so on success the aliases drop as no-ops and
        // the real tensors live on in the struct (no double-free); on error we
        // free each alias once, which releases the same pool buffer the
        // partially-built (and about-to-be-dropped, never-freed) field held.
        let mut ledger: Vec<GpuTensor> = Vec::with_capacity(48);
        macro_rules! alloc {
            ($shape:expr, $dt:expr) => {
                match gpu.alloc_tensor($shape, $dt) {
                    Ok(t) => {
                        // SAFETY: alias lives only inside `new`; if used it is
                        // freed in the error arm below (the original field is
                        // dropped without freeing, no Drop on GpuTensor), and
                        // on success it is dropped untouched (no Drop on
                        // DeviceBuffer) while the original is moved into Self.
                        ledger.push(GpuTensor {
                            buf: unsafe { t.buf.alias() },
                            shape: t.shape.clone(),
                            dtype: t.dtype,
                        });
                        t
                    }
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.free_tensor(prev);
                        }
                        return Err(e);
                    }
                }
            };
        }
        macro_rules! alloc_opt {
            ($cond:expr, $shape:expr, $dt:expr) => {
                if $cond {
                    Some(alloc!($shape, $dt))
                } else {
                    None
                }
            };
        }

        // Hoisted grouped-GEMM sizing (same value across the Path-2 fields).
        let grouped_m_total_max =
            moe_grouped_m_total_max(max_batch, config.num_experts_per_tok, config.num_experts);
        let grouped_total_slots_max = max_batch * config.num_experts_per_tok;

        Ok(Self {
            max_batch,
            x_batch: alloc!(&[max_batch * dim], DType::F32),
            x_rot_batch: alloc!(&[max_batch * dim], DType::F32),
            x_norm_batch: alloc!(&[max_batch * dim], DType::F32),
            dn_qkv_batch: alloc!(&[max_batch * qkv_dim], DType::F32),
            dn_z_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_alpha_batch: alloc!(&[max_batch * n_v_heads], DType::F32),
            dn_beta_batch: alloc!(&[max_batch * n_v_heads], DType::F32),
            dn_q_raw_batch: alloc!(&[max_batch * k_dim], DType::F32),
            dn_k_raw_batch: alloc!(&[max_batch * k_dim], DType::F32),
            dn_v_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_q_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_k_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_attn_out_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_normed_batch: alloc!(&[max_batch * v_dim], DType::F32),
            gate_ffn_batch: alloc!(&[max_batch * hidden_dim], DType::F32),
            up_batch: alloc!(&[max_batch * hidden_dim], DType::F32),
            ffn_hidden_batch: alloc!(&[max_batch * hidden_dim], DType::F32),
            dn_normed_rot_batch: alloc!(&[max_batch * v_dim], DType::F32),
            // F32 dtype = 4 bytes/element, same layout as i32. The rope /
            // attention / kv_write kernels cast the pointer to `const int*`,
            // so dtype is cosmetic. Upload i32 bits via memcpy_htod.
            positions: alloc!(&[max_batch], DType::F32),
            // Depth-based RoPE angles for DDTree verify (39aa358 fix):
            // `positions` stays the flat linear KV slot index; this buffer
            // carries `base_pos + depth(node)` so FA-layer RoPE rotates Q/K
            // at the logically-correct phase while KV writes stay on
            // distinct linear slots. Uploaded per cycle in tree-verify mode
            // from `TreeVerifyCtx.positions`; FA RoPE kernels read it ONLY
            // when `tree_verify.is_some()`. Same i32-in-F32 cosmetic dtype
            // pattern as `positions`.
            rope_positions: alloc!(&[max_batch], DType::F32),
            tokens: alloc!(&[max_batch], DType::F32),
            fa_q_full_batch: alloc!(&[max_batch * q_dim * 2], DType::F32),
            fa_q_batch: alloc!(&[max_batch * q_dim], DType::F32),
            fa_gate_batch: alloc!(&[max_batch * q_dim], DType::F32),
            fa_k_batch: alloc!(&[max_batch * kv_dim], DType::F32),
            fa_v_batch: alloc!(&[max_batch * kv_dim], DType::F32),
            fa_attn_out_batch: alloc!(&[max_batch * q_dim], DType::F32),
            fa_attn_out_rot_batch: alloc!(&[max_batch * q_dim], DType::F32),
            moe_router_logits_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts],
                DType::F32
            ),
            moe_shared_scalar_batch: alloc_opt!(config.num_experts > 0, &[max_batch], DType::F32),
            moe_shared_gate_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.shared_expert_intermediate_size],
                DType::F32
            ),
            moe_shared_up_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.shared_expert_intermediate_size],
                DType::F32
            ),
            moe_shared_rot_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.shared_expert_intermediate_size],
                DType::F32
            ),
            moe_topk_indices_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok],
                DType::F32
            ),
            moe_topk_weights_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok],
                DType::F32
            ),
            moe_gate_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32
            ),
            moe_up_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32
            ),
            moe_rot_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32
            ),
            moe_down_expanded_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.dim],
                DType::F32
            ),
            // Path 2 scatter + grouped-WMMA-GEMM scratch (gated at runtime by
            // HIPFIRE_MOE_GROUPED_GEMM=1). m_total_max = N*K_TOP + E*(BLOCK_M-1).
            // i32 buffers stored as Raw (4 bytes/elem matches; no DType::I32 yet).
            moe_expert_token_counts: alloc_opt!(
                config.num_experts > 0,
                &[config.num_experts * 4],
                DType::Raw
            ),
            moe_expert_offsets: alloc_opt!(
                config.num_experts > 0,
                &[(config.num_experts + 1) * 4],
                DType::Raw
            ),
            moe_sorted_slot_index: alloc_opt!(
                config.num_experts > 0,
                &[grouped_m_total_max * 4],
                DType::Raw
            ),
            moe_inverse_perm: alloc_opt!(
                config.num_experts > 0,
                &[grouped_total_slots_max * 4],
                DType::Raw
            ),
            moe_expert_tile_ids: alloc_opt!(
                config.num_experts > 0,
                &[(grouped_m_total_max / MOE_GROUPED_BLOCK_M) * 4],
                DType::Raw
            ),
            moe_y_gate_up_grouped: alloc_opt!(
                config.num_experts > 0,
                &[grouped_m_total_max * 2 * config.moe_intermediate_size],
                DType::F32
            ),
            moe_y_down_grouped: alloc_opt!(
                config.num_experts > 0,
                &[grouped_m_total_max * config.dim],
                DType::F32
            ),
            dn_s_tape_q8: alloc_opt!(
                cap_gdn_tape && config.linear_num_value_heads > 0,
                &[max_batch
                    * config.linear_num_value_heads
                    * config.linear_value_head_dim
                    * config.linear_value_head_dim],
                DType::Raw
            ),
            dn_s_tape_scales: alloc_opt!(
                cap_gdn_tape && config.linear_num_value_heads > 0,
                &[max_batch * config.linear_num_value_heads * config.linear_value_head_dim],
                DType::F32
            ),
            dn_s_tape_f32: alloc_opt!(
                cap_gdn_tape && config.linear_num_value_heads > 0,
                &[max_batch
                    * config.linear_num_value_heads
                    * config.linear_value_head_dim
                    * config.linear_value_head_dim],
                DType::F32
            ),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x_batch,
            self.x_rot_batch,
            self.x_norm_batch,
            self.dn_qkv_batch,
            self.dn_z_batch,
            self.dn_alpha_batch,
            self.dn_beta_batch,
            self.dn_q_raw_batch,
            self.dn_k_raw_batch,
            self.dn_v_batch,
            self.dn_q_batch,
            self.dn_k_batch,
            self.dn_attn_out_batch,
            self.dn_normed_batch,
            self.gate_ffn_batch,
            self.up_batch,
            self.ffn_hidden_batch,
            self.dn_normed_rot_batch,
            self.positions,
            self.tokens,
            self.fa_q_full_batch,
            self.fa_q_batch,
            self.fa_gate_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.moe_router_logits_batch,
            self.moe_shared_scalar_batch,
            self.moe_shared_gate_batch,
            self.moe_shared_up_batch,
            self.moe_shared_rot_batch,
            self.moe_topk_indices_batch,
            self.moe_topk_weights_batch,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_down_expanded_batch,
            // Path 2 (grouped-WMMA-GEMM, HIPFIRE_MOE_GROUPED_GEMM, default-on on
            // gfx11+/gfx12) MoE scratch. These were added when the grouped-GEMM
            // path landed but never added to this teardown, so they leaked every
            // prefill — moe_y_gate_up_grouped (~46 MB) + moe_y_down_grouped
            // (~23 MB) dominate. THIS is the per-request VRAM growth that OOMs
            // long-lived serves after ~N requests.
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_inverse_perm,
            self.moe_expert_tile_ids,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.dn_s_tape_q8,
            self.dn_s_tape_scales,
            self.dn_s_tape_f32,
        ] {
            if let Some(t) = t {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

/// Batched prefill entry point: processes N prompt tokens in one call,
/// writing the last token's logits into `scratch.logits` and leaving
/// the KV cache + DeltaNet state advanced by N positions.
///
/// Takes the batched kernel path when ALL linear-attention layer weights
/// are MQ4G256 (the batched element-wise kernels are MQ-specific).
/// Otherwise falls back to a per-token loop over `forward_scratch` that's
/// byte-identical to decode. FA layers always use a per-token gather/scatter
/// fallback — the FA causal attention kernel can't yet be batched (task #71).
///
/// `gated_delta_net_q8_batch_seq` runs one launch per LA layer; the kernel
/// loops over the N tokens internally and requants the Q8 state after every
/// token, matching the decode requant cadence (distributionally equivalent to
/// decode, not byte-identical — the stochastic-rounding frame differs).
///
/// `tokens`: slice of prompt tokens to prefill in order.
/// `start_pos`: first KV cache / DeltaNet position to write. Positions
/// `start_pos .. start_pos + tokens.len()` get populated.
/// On return, `scratch.logits` holds the logits for the *last* token
/// (position `start_pos + tokens.len() - 1`).
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states are captured
/// into the ring buffer for the configured extract layers. Used by the
/// DFlash target-side verify path to batch `verify_dflash_block` into a
/// single forward launch (MVP does B per-token forwards — 88 ms on 4B;
/// this path drops it to ~40 ms with batched forward, further improvement
/// possible with batched lm_head). The per-token fallback also honors it,
/// so the fast-path eligibility doesn't change behavior.
///
/// `per_token_hidden_out`: if `Some`, writes post-output-norm hidden state
/// for each of the N tokens into the provided [N × dim] buffer. The caller
/// then loops `weight_gemv(weights.output, hidden_row, logits)` to recover
/// per-token logits. Required for DFlash verify (needs all B positions'
/// logits, not just the last). `None` preserves the existing "last token
/// only" semantics where logits land in `scratch.logits`.
///
/// `gdn_tape`: if `Some`, captures the post-processed `(q, k, v, α, β)` for
/// every DN (LinearAttention) layer and block position BEFORE the batched
/// `gated_delta_net_q8_batch_seq` call. Enables the DFlash rollback path
/// to replay GDN recurrence from a pre-verify S-state snapshot for
/// `accept_len + 1` steps — no full-target re-run needed.
#[allow(clippy::too_many_arguments)]
/// Upper bound on `forward_prefill_batch`'s per-chunk size. Exposed so
/// callers sizing `HiddenStateRingBuffer` staging can match the chunk
/// upper bound (staging that's smaller than a chunk will assert-fail
/// on prompt seeding of long prompts).
pub const PREFILL_MAX_BATCH: usize = 256;

const MOE_GROUPED_BLOCK_M: usize = 16;

#[inline]
fn prefill_should_emit_last_token_logits(
    has_per_token_hidden_out: bool,
    needs_last_token_logits: bool,
) -> bool {
    !has_per_token_hidden_out || needs_last_token_logits
}

#[inline]
fn align_up_usize(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

#[inline]
fn moe_grouped_m_total_max(max_batch: usize, k_top: usize, n_exp: usize) -> usize {
    // Every grouped-GEMM tile consumes 16 sorted slots. The scatter kernel
    // initializes sentinel tile ids up to this bound, so the bound itself must
    // be tile-aligned; otherwise the final launched tile can read an
    // uninitialized expert id.
    align_up_usize(
        max_batch * k_top + n_exp * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

#[inline]
fn moe_grouped_m_total_bound(total_slots: usize, n_exp: usize) -> usize {
    // Actual grouped rows are sum_e align_up(count_e, BLOCK_M). Only experts
    // that receive at least one slot can contribute padding, so small verify
    // batches do not need to launch the full all-experts worst case.
    let live_expert_bound = total_slots.min(n_exp);
    align_up_usize(
        total_slots + live_expert_bound * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

/// Host-side helper: upload token ids and positions to a `PrefillBatchScratch`
/// via sync `memcpy_htod`. Call this BEFORE entering a hipGraph capture to
/// pre-populate `pbs.tokens` and `pbs.positions`, then pass `pre_uploaded:
/// true` (or use `forward_prefill_chunk_captured_safe`) so the forward
/// does not issue any additional uploads inside the captured region.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Capture-friendly entry point that runs the batched forward against a
/// SINGLE chunk (`tokens.len() <= pbs.max_batch`), skipping the internal
/// token/position upload and assuming the caller has already populated
/// `pbs.tokens` / `pbs.positions` via `upload_prefill_batch_inputs`.
///
/// This exists so `hipStreamBeginCapture` can wrap the forward without
/// the per-call `memcpy_htod` sync operations (which would either error
/// under capture or bake stale host data into the captured graph nodes).
///
/// Callers still must handle `hidden_rb.commit_staging_to_ring(gpu, n)`
/// AFTER the forward returns (outside any captured region) to scatter
/// staging writes to the ring at the current head.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_single_chunk_captured_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    needs_last_token_logits: bool,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(
        n > 0 && n <= pbs.max_batch,
        "single_chunk_captured: n={} but pbs.max_batch={}",
        n,
        pbs.max_batch
    );
    let required_tokens = checked_kv_end(start_pos, n, "captured prefill")?;
    // This entry may already be inside graph capture, so it must only validate
    // capacity preflighted by its caller.
    kv_cache.require_mapped_capacity(required_tokens)?;

    // Defense-in-depth: this entry point bypasses the eligibility check
    // in `forward_prefill_batch_with_pbs`, so the caller is responsible
    // for ensuring the batched fast-path is valid. Two structural bypasses
    // could land here:
    //   1. MQ3-weighted model on an arch that lacks the gfx11 wave32 WMMA
    //      builtin (gfx12, gfx10, gfx906, gfx94x).
    //   2. MQ3 weights inside a MoE/A3B layer (DeltaNetMoe/FullAttnMoe) —
    //      the MoE batched branches dispatch through HFQ4-layout kernels
    //      and would memory-fault on the 104-vs-136 byte stride.
    // In production, `daemon.rs`'s DFlash refusal guard blocks both, but
    // dflash_spec_demo and other example callers go through ModelSlot::load
    // directly. We cross-check here so any caller is protected.
    let arch = gpu.arch.as_str();
    let mut mq3_in_dense = false;
    let mut mq3_in_moe = false;
    let mut lloyd_in_dense = false;
    // The Lloyd dtype is treated identically to plain MQ3 in this guard:
    // both use 112-vs-104-byte stride that the MoE batched branches'
    // HFQ4-layout dispatch would corrupt, and both depend on the gfx11/12
    // WMMA family that other archs lack. Add Lloyd alongside MQ3 so the
    // refusal fires symmetrically and a future MQ3-Lloyd MoE model can't
    // silently land here without explicit MoE-Lloyd kernels.
    //
    // We also track `lloyd_in_dense` separately because Lloyd-MQ3 on
    // gfx12 ships behind an opt-in env gate (see is_batchable_la above) —
    // the gfx12 sibling kernels are runtime-unvalidated locally, so by
    // default a captured-path call with Lloyd-MQ3 weights on gfx1200/1201
    // must refuse rather than dispatch to an untested kernel.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
    for lw in &weights.layers {
        match lw {
            LayerWeights::DeltaNet(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
                if is_lloyd(l.wqkv.gpu_dtype)
                    || is_lloyd(l.wz.gpu_dtype)
                    || is_lloyd(l.w_beta.gpu_dtype)
                    || is_lloyd(l.w_alpha.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
                }
            }
            LayerWeights::FullAttn(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
                if is_lloyd(l.wq.gpu_dtype)
                    || is_lloyd(l.wk.gpu_dtype)
                    || is_lloyd(l.wv.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
                }
            }
            LayerWeights::DeltaNetMoe(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_structural(&l.ffn)
                    || moe_ffn_has_mq3_experts_uniform(&l.ffn)
                {
                    mq3_in_moe = true;
                }
            }
            LayerWeights::FullAttnMoe(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_structural(&l.ffn)
                    || moe_ffn_has_mq3_experts_uniform(&l.ffn)
                {
                    mq3_in_moe = true;
                }
            }
        }
    }
    // ANTIBLEED admit-vs-select fix: this guard rejects MQ3-in-dense when the
    // arch lacks the WMMA builtin. The old ad-hoc string list OMITTED gfx1103
    // (Phoenix APU) and gfx1152, yet both ARE wave32-WMMA archs (is_rdna3) and
    // are ADMITTED by is_batchable_la's mq3_uniform_with_wmma — so a gfx1103 /
    // gfx1152 box would be wrongly rejected here. Derive from the has_wmma
    // capability molecule instead (rdna3 incl 1103/1152, + rdna4), matching the
    // sibling `arch_has_wmma = gpu.arch_caps.has_wmma()` in forward_prefill_chunk.
    let arch_has_wmma = gpu.arch_caps.has_wmma();
    if mq3_in_moe {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_single_chunk_captured: model has MQ3G256 / \
             MQ3G256Lloyd weights inside a MoE/A3B layer (DeltaNetMoe or \
             FullAttnMoe). The MoE batched prefill branches dispatch through \
             HFQ4-layout kernels and would memory-fault on the 104/112-vs-136 \
             byte stride. Use an MQ4 quantization for MoE/A3B targets, or wait \
             for the MQ3 MoE branches to land.",
        ));
    }
    if mq3_in_dense && !arch_has_wmma {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains MQ3G256 \
             weights but arch {arch} lacks the gfx11 wave32 WMMA builtin. The MQ3 \
             prefill kernels (gemm_*_hfq3g256_wmma) only compile on the wave32-WMMA \
             archs (rdna3: gfx1100/1101/1102/1103/1150/1151/1152, + rdna4 gfx12). \
             Caller must use the non-captured \
             forward_prefill_batch path (which falls back to per-token \
             forward_scratch on this arch). gfx12 K4 variant for MQ3 is \
             a planned follow-up."
            ),
        ));
    }
    // Lloyd-MQ3 on gfx12 is opt-in (see is_batchable_la's gate). The
    // captured entry point bypasses is_batchable_la, so we replicate the
    // gate here: refuse Lloyd-on-gfx12 unless HIPFIRE_LLOYD_GFX12=1 is set.
    // Without this guard, a captured call would reach the dispatch arms
    // and try to load gfx12 kernels that are still community-CI-pending.
    let arch_is_gfx12 = matches!(arch, "gfx1200" | "gfx1201");
    let lloyd_gfx12_optin = hipfire_config::developer_var("HIPFIRE_LLOYD_GFX12")
        .ok()
        .as_deref()
        == Some("1");
    if lloyd_in_dense && arch_is_gfx12 && !lloyd_gfx12_optin {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains \
             MQ3G256Lloyd weights on arch {arch}, but the gfx12 (RDNA4) \
             sibling kernels (gemm_*_mq3g256_lloyd_wmma.gfx12.hip) are \
             runtime-unvalidated locally and ship behind an opt-in gate. \
             Set HIPFIRE_LLOYD_GFX12=1 to enable the gfx12 path for parity \
             testing, or use the non-captured forward_prefill_batch path \
             (which falls back to per-token forward_scratch on this arch \
             when the env var is unset)."
            ),
        ));
    }

    // Q8 KV at any physical_cap is capture-safe: forward_prefill_chunk
    // dispatches through the unified DispatchCtx → AttnQ8_0KvBatchedMasked,
    // which routes max_ctx_len > 8192 to the tiled attention_flash_q8_0_tile_batched
    // (O(1) LDS, no per-position malloc). The former physical_cap > 15000 guard
    // predated that crossover (landed 2026-06-09) and is now obsolete.
    forward_prefill_chunk(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out.map(|t| (t, 0)),
        gdn_tape,
        0,
        tree_verify,
        true, // pre_uploaded: caller must have run upload_prefill_batch_inputs
        None, // band: full-stack single-GPU path
        None, // mask_override: captured-prefill caller does not use the MTP probe hook
        needs_last_token_logits,
        None, // max_layer: single-chunk captured path always runs the full stack
        None, // routed_out: non-EP single-GPU path
    )
}

pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        scratch.prefill_batch.as_ref(),
        None, // mask_override: MTP probe is the only consumer; default callers don't override
        None, // max_layer: pflash uses this; non-pflash default is full stack
    )
}

pub fn forward_prefill_batch_with_pbs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        pbs_in,
        mask_override,
        max_layer,
        true, // preserve legacy post-condition: scratch.logits is last-token logits
    )
}

/// Like `forward_prefill_batch`, but accepts a caller-owned `PrefillBatchScratch`
/// so the ~25 per-cycle tensor allocations can be amortized across many calls.
///
/// `pbs = None` allocates and frees a right-sized scratch per call;
/// `pbs = Some(&pbs)` reuses the provided scratch. The provided scratch's
/// `max_batch` determines the chunk size — `tokens` is processed in chunks of
/// up to `pbs.max_batch`. Callers driving DFlash verify should size `pbs`
/// to the maximum block size they'll ever request (e.g. `block_size` or
/// `1 + tree_budget`) so everything fits in one chunk.
///
/// `needs_last_token_logits = false` is only for callers that pass
/// `per_token_hidden_out` and compute their own logits from those hidden rows.
/// The default wrapper keeps this true to protect existing callers that rely on
/// `scratch.logits` being populated with the last token's logits.
pub fn forward_prefill_batch_with_pbs_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mut hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
    needs_last_token_logits: bool,
) -> HipResult<()> {
    // Plain single-token AR decode? Only then is the per-token `forward_scratch`
    // call below eligible for the AR-forward hipGraph (capture/replay). Any spec
    // marker (tree_verify / gdn_tape / per-token-hidden extraction / hidden ring)
    // or a multi-token batch means this is prefill or a spec/MTP verify forward,
    // which must NOT replay the plain-AR graph. See `forward_scratch`'s
    // `ar_graph_eligible` one-shot signal.
    let plain_ar_graph_eligible = tree_verify.is_none()
        && gdn_tape.is_none()
        && per_token_hidden_out.is_none()
        && hidden_rb.is_none()
        && tokens.len() == 1;
    // Upper bound on the PrefillBatchScratch — large prompts get split
    // into chunks of this size and processed in a loop.
    //
    // Tuning note: each extra chunk pays full dispatch-overhead for the LA
    // preamble (rmsnorm, rotate, 4-way fused GEMM) and FFN (gate_up + down).
    // 256 costs ~80 MB of scratch on 9B vs 20 MB at 64 — trivial on modern
    // cards — and drops chunk count for pp2048 from 32 → 8. The inner
    // gated_delta_net_q8_batch_seq loop is still sequential per token, so
    // the per-chunk DeltaNet cost is linear in N either way; raising the
    // batch just amortizes the NON-DeltaNet kernels more.
    //
    // Exposed via PREFILL_MAX_BATCH so callers sizing `HiddenStateRingBuffer`
    // staging can match the chunk upper bound.
    let max_batch: usize = hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= MIN_BATCH)
        .unwrap_or(PREFILL_MAX_BATCH);

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let required_tokens = checked_kv_end(start_pos, n, "forward_prefill_batch")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;

    // Cross-path safety: refuse MQ3 / MQ3-Lloyd weights inside any MoE
    // layer (attention OR FFN), mirroring the captured-path guard at
    // `forward_prefill_batch_single_chunk_captured` (line 3367+). Without
    // this, the eligibility check below would admit a hybrid model with
    // (e.g.) MQ3 attention + MQ4 MoE FFN onto the batched path, where the
    // MoE-batched LA/FA bodies would misroute: the QKV matcher drops MQ3
    // and the wo path is hardcoded to `gemm_hfq4g256_residual` regardless
    // of `layer.wo.gpu_dtype`. The result is a 104/112 vs 136 byte stride
    // mismatch and silent-corruption fluent-looking output. Issue #179
    // documents the matcher half of this; the wo half was uncovered in
    // review. Wiring both correctly (plus Lloyd) is tracked separately
    // (see followup issue) — until then we hard-error here so all three
    // entry points (daemon-DFlash setup, captured prefill, non-captured
    // prefill) reject MQ3+MoE consistently.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let mq3_in_moe = weights.layers.iter().any(|lw| match lw {
        LayerWeights::DeltaNetMoe(l) => {
            is_mq3_any(l.wqkv.gpu_dtype)
                || is_mq3_any(l.wz.gpu_dtype)
                || is_mq3_any(l.w_beta.gpu_dtype)
                || is_mq3_any(l.w_alpha.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_structural(&l.ffn)
                || moe_ffn_has_mq3_experts_uniform(&l.ffn)
        }
        LayerWeights::FullAttnMoe(l) => {
            is_mq3_any(l.wq.gpu_dtype)
                || is_mq3_any(l.wk.gpu_dtype)
                || is_mq3_any(l.wv.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_structural(&l.ffn)
                || moe_ffn_has_mq3_experts_uniform(&l.ffn)
        }
        _ => false,
    });
    if mq3_in_moe {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch: model has MQ3G256 / MQ3G256Lloyd weights \
             inside a MoE/A3B layer (DeltaNetMoe or FullAttnMoe). The MoE \
             batched prefill branches dispatch through HFQ4-layout kernels \
             (QKV matcher drops MQ3; wo path is hardcoded MQ4) and would \
             produce silent corruption from the 104/112-vs-136 byte stride \
             mismatch. Use an MQ4 quantization for MoE/A3B targets, or wait \
             for the MQ3 MoE branches to land (see followup issue).",
        ));
    }

    // Tree-verify mode sanity checks — the downstream path can't silently
    // fall back to per-token FA (that's always causal and would ignore the
    // tree mask), and the positions/bias shapes must match the token count.
    if let Some(ctx) = tree_verify.as_ref() {
        assert_eq!(
            ctx.positions.len(),
            n,
            "TreeVerifyCtx.positions length {} must equal tokens.len() {}",
            ctx.positions.len(),
            n,
        );
        assert_eq!(
            ctx.attn_bias.numel(),
            n * n,
            "TreeVerifyCtx.attn_bias must be [{} × {}] f32 ({}), got numel {}",
            n,
            n,
            n * n,
            ctx.attn_bias.numel(),
        );
    }

    // Fast path requires (a) every LA layer's weights to be either MQ4G256
    // or HFQ4G256 (the batched GEMM kernels are dtype-agnostic but the LA
    // preamble's rmsnorm+rotate and SwiGLU+rotate kernels differ per dtype),
    // and (b) Q8 S-state for the GDN recurrence. Mixed-dtype layers are
    // allowed; each layer is routed to its own path. HFQ6/others fall back.
    let arch = gpu.arch.as_str();
    // Whether the tape-capturing batched (PBS) path runs for this call — the
    // single source of truth shared with spec-decode callers that later replay a
    // captured GDN tape. On `false` the forward drops to the tape-less per-token
    // loop below, leaving any passed tape stale (see `prefill_batch_pbs_eligible`).
    let moe_router_logits_present = pbs_in
        .map(|p| p.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let eligible = prefill_batch_pbs_eligible(
        weights,
        config,
        dn_state,
        n,
        arch,
        moe_router_logits_present,
    );
    // F4 guard: reject batched prefill when KV tier has no batched keys.
    // F32 KV has only BatchEq(1) → MissingImpl at resolve. asym2 + tree-verify
    // has no _batched_masked variant → UnsupportedTreeTier. Force per-token
    // fallback for these cases.
    let kv_f32 = !kv_cache.quantized && !kv_cache.quant_q8 && !kv_cache.quant_hfq4;
    let kv_asym2_tree = kv_cache.quant_asym2 && tree_verify.is_some();
    let eligible = eligible && !kv_f32 && !kv_asym2_tree;

    if !eligible {
        assert!(
            tree_verify.is_none(),
            "tree-verify mode requires the batched-FA-eligible prefill path; \
             kv quant + FA weight dtypes do not match on this model",
        );
        // mask_override has nowhere to land on the per-token forward_scratch
        // fallback (it operates on `scratch.x`, not the batched `pbs.x_batch`,
        // and there's no shared "post-embed, pre-layer" hook). The MTP probe
        // is the only consumer today and runs on MQ4-quantized models that
        // always satisfy `eligible`, so hard-error rather than silently
        // ignoring the override.
        assert!(
            mask_override.is_none(),
            "MaskEmbedOverride requires the batched prefill path, but this \
             model fell through to the per-token fallback (likely non-MQ4 \
             weights, dn_state quant != Q8, or HIPFIRE_PREFILL_BATCHED=0).",
        );
        // Fallback: per-token loop, byte-identical to decode. If hidden
        // extraction is requested, use the with_hidden variant so the ring
        // buffer still gets populated correctly (each call advances head by 1).
        // When per-token hidden output is also requested, extract post-norm
        // hidden row-by-row into the caller's buffer.
        let dim = config.dim;
        for (i, &tok) in tokens.iter().enumerate() {
            if let Some(rb) = hidden_rb.as_mut() {
                forward_scratch_with_hidden(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                    rb,
                )?;
            } else {
                // One-shot: mark this forward AR-graph-eligible iff it's plain
                // single-token decode (consumed inside forward_scratch).
                gpu.graphs.ar_graph_eligible = plain_ar_graph_eligible;
                forward_scratch(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            }
            if let Some(dst) = per_token_hidden_out {
                // scratch.tmp holds post-output-norm hidden after
                // forward_scratch_{with_hidden,layers} — it's the same buffer
                // lm_head reads from. Copy into the caller's output.
                gpu.hip
                    .memcpy_dtod_at(&dst.buf, i * dim * 4, &scratch.tmp.buf, 0, dim * 4)?;
            }
        }
        return Ok(());
    }

    // Tree-verify mode runs as a single chunk (tree is small, O(16) nodes);
    // chunk splitting would require slicing the mask by chunk rows which
    // is extra work for a case we don't need.
    if tree_verify.is_some() {
        assert!(
            n <= max_batch,
            "tree-verify tokens {} exceeds max_batch {}; tree budget must fit",
            n,
            max_batch,
        );
    }

    // Allocate the batch scratch once per call (or reuse a caller-owned one).
    // When `pbs_in` is Some, we neither allocate nor free — the caller retains
    // ownership across DFlash cycles to avoid ~25 per-cycle tensor alloc/free
    // pairs on the hot verify path. When None, size the allocation to this
    // call's largest possible chunk and allocate the DeltaNet S-state tape only
    // for tree verify. Plain prefill never consumes that tape, and short
    // prompts should not pay the full 256-row scratch footprint. The chunk size
    // is `pbs.max_batch` so a caller-owned scratch sized to e.g. `block_size`
    // or `1 + tree_budget` keeps DFlash verify in one chunk.
    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let result = (|| -> HipResult<()> {
        let pbs: &PrefillBatchScratch = match pbs_in {
            Some(p) => p,
            None => {
                let (owned_max_batch, cap_gdn_tape) =
                    owned_prefill_scratch_plan(n, max_batch, tree_verify.is_some());
                own_pbs = Some(PrefillBatchScratch::new_opt(
                    gpu,
                    config,
                    owned_max_batch,
                    cap_gdn_tape,
                )?);
                own_pbs.as_ref().unwrap()
            }
        };
        let chunk_batch = pbs.max_batch;
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let remaining = n - chunk_start;
            let chunk_n = next_prefill_chunk_len(remaining, chunk_batch).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    "forward_prefill_batch: chunk plan cannot satisfy the two-token minimum",
                )
            })?;
            let chunk_end = chunk_start + chunk_n;
            let chunk = &tokens[chunk_start..chunk_end];
            // The chunk only reads the ring buffer's head/dims to place its
            // writes. We advance the head AFTER the chunk returns, here in
            // the caller, to keep the mutable borrow scope tight.
            let pth_slot = per_token_hidden_out.map(|t| (t, chunk_start));
            // Reborrow the tape for this chunk so we keep the outer mut
            // after the chunk returns.
            let tape_for_chunk: Option<&mut crate::speculative::GdnTape> =
                gdn_tape.as_mut().map(|t| &mut **t);
            // Tree-verify was asserted to fit in one chunk above, so passing
            // the whole ctx through unconditionally is safe.
            let tv_for_chunk = tree_verify.as_ref().copied();
            // Apply mask_override only to the chunk that actually contains
            // its target slot, and rebase the slot index to chunk-local
            // coordinates. Out-of-range slots panic (caller error).
            let mo_for_chunk = mask_override.and_then(|ovr| {
                if ovr.slot >= chunk_start && ovr.slot < chunk_end {
                    Some(MaskEmbedOverride {
                        slot: ovr.slot - chunk_start,
                        embed: ovr.embed,
                    })
                } else {
                    None
                }
            });
            // Sanity: if caller provided an override, it MUST land in some
            // chunk. Detect "fell off the end" at the last chunk boundary.
            if mask_override.is_some() && chunk_end == n {
                let landed_anywhere = mask_override.unwrap().slot < n;
                assert!(
                    landed_anywhere,
                    "MaskEmbedOverride.slot ({}) is out of range for tokens.len() ({})",
                    mask_override.unwrap().slot,
                    n,
                );
            }
            forward_prefill_chunk(
                gpu,
                weights,
                config,
                chunk,
                start_pos + chunk_start,
                kv_cache,
                dn_state,
                scratch,
                pbs,
                hidden_rb.as_deref(),
                pth_slot,
                tape_for_chunk,
                chunk_start,
                tv_for_chunk,
                false, // pre_uploaded: default path uploads inside
                None,  // band: full-stack single-GPU path
                mo_for_chunk,
                needs_last_token_logits,
                max_layer,
                None, // routed_out: non-EP single-GPU path
            )?;
            if let Some(rb) = hidden_rb.as_mut() {
                // Scatter fixed-offset staging writes (done inside the chunk)
                // to the ring at the current head, then advance head by n.
                // This is the out-of-capture step: graph-captured writes went
                // to staging[0..n*h], this commit places them at head*h
                // where head is read from CPU state at call time (not baked
                // into a captured graph node).
                rb.commit_staging_to_ring(gpu, chunk_n)?;
            }
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    if let Some(owned) = own_pbs {
        owned.free_gpu(gpu);
    }
    result
}

/// Accepts the dtypes the batched prefill path can handle (shared by the
/// eligibility check in `forward_prefill_batch` and the per-layer dtype
/// branches in `forward_prefill_chunk`).
#[inline]
// IMPORTANT: This allowlist is paired with the `is_mq*` matchers in
// forward_prefill_chunk (lines 4063+, 4360+, 4768, 4919) and with the
// MoE FFN gate `moe_ffn_batched_admissible`. They MUST be updated together when
// adding a new batchable dtype. Updating one without the others either
// produces dead code (safe but useless) or silent prefill corruption
// (HFQ4-stride GEMM reading a different-stride weight block). See
// docs/plans/mq-lloyd-batched-prefill-followup.md for the full
// checklist + rationale.
//
// As of this PR (issue #116 Phase 5): MQ3G256Lloyd is wired through
// the gemm_*_mq3g256_lloyd_wmma family on gfx11 (always-on) and on
// gfx12 (opt-in via HIPFIRE_LLOYD_GFX12=1). MQ4G256Lloyd is wired
// through the gemm_*_mq4g256_lloyd_wmma family on gfx11 (always-on)
// and gfx12 (opt-in via HIPFIRE_LLOYD_GFX12=1). MQ2G256Lloyd remains
// unwired — MQ2-Lloyd lands separately.
fn is_batchable_la(dt: DType, arch: &str) -> bool {
    let always_ok = matches!(
        dt,
        DType::MQ4G256 | DType::HFQ4G256
        | DType::MQ6G256 | DType::HFQ6G256
        | DType::Q8_0
        // Phase 1.5 (PARO): wqkv/wz/wo are ParoQ4G128, w_alpha/w_beta are F32
        // on shisa-Qwen3.6-A3B-PARO. Dispatch in the DeltaNetMoe LA matcher
        // routes these through gemm_hfq4g128 (with per-weight Givens
        // rotation pre-pass) and gemm_f32_batched respectively. Eligibility
        // is gated downstream by the env-keyed moe_ffn_batched_admissible
        // (HIPFIRE_PARO_BATCHED=1) — admitting them here keeps non-PARO
        // models unaffected because no production checkpoint sets
        // wqkv.gpu_dtype = ParoQ4G128 outside the shisa-PARO codepath.
        | DType::ParoQ4G128 | DType::F32
    );
    if always_ok {
        return true;
    }
    // MQ3 (uniform / HFQ3 family) is batchable on archs with a WMMA
    // family ported. As of this commit:
    //   - gfx11 (gfx1100/1101/1102/1150/1151): wave32 WMMA via the
    //     `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` builtin.
    //   - gfx12 (gfx1200/1201): wave32 WMMA via the `_w32_gfx12` builtin
    //     with K4 unroll + half8_t lane-split, runtime-validated through
    //     the existing HFQ3 dispatch fork (gemm_*_hfq3g256_wmma_gfx12).
    // gfx906 GCN5 / gfx94x CDNA3 lack a ported MQ3 WMMA kernel; they
    // stay on the per-token forward_scratch fallback (correct, just
    // slower). gfx10 RDNA1/2 gains batched-prefill support via the
    // scalar HFQ3 GEMM family below (Phase 1 of
    // docs/plans/gfx10_mq3_prefill.md).
    let mq3_uniform_with_wmma = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        );

    // gfx10 RDNA1/2 scalar HFQ3 batched-prefill family (Phase 1).
    // Routes the four LA + FA matchers below to the new non-WMMA kernels
    // (gemm_qkv_hfq3g256, gemm_qkvza_hfq3g256, gemm_gate_up_hfq3g256,
    // gemm_hfq3g256_residual). Lloyd-MQ3 stays gated on gfx11+ — no
    // gfx10 Lloyd port (separate larger project).
    let mq3_uniform_with_gfx10_scalar = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" | "gfx1030" | "gfx1031" | "gfx1032"
        );

    // HFP4G32 / MFP4G32 (v2 #2 batched WMMA prefill): same arch gate as
    // MQ3. The 4 fused kernels (gemm_qkv/qkvza/gate_up/residual_hfp4g32_wmma)
    // ship in pairs for gfx11 + gfx12; identical eligibility to llama.rs
    // (see hipfire_runtime::llama::is_batchable_la).
    let fp4_with_wmma = matches!(dt, DType::HFP4G32 | DType::MFP4G32)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        );

    // Lloyd-MQ3 (MQ3G256Lloyd) on gfx11: Phase 5 of issue #116 ships the
    // gemm_*_mq3g256_lloyd_wmma family alongside the existing HFQ3 WMMA
    // path; group stride differs (112 B Lloyd vs 104 B HFQ3) so dispatch
    // must route to the Lloyd-specific arms (handled by the LA/FA
    // matchers downstream — see followup-checklist condition 3).
    let lloyd_mq3_with_gfx11_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
        );

    // Lloyd-MQ3 on gfx12 (RDNA4): the gemm_*_mq3g256_lloyd_wmma.gfx12.hip
    // kernels are code-complete but runtime-unvalidated locally — bench
    // host is gfx1100/1151 — so they ship behind an opt-in env gate.
    // With HIPFIRE_LLOYD_GFX12 unset (default), Lloyd-MQ3 on gfx1200/1201
    // falls through to per-token forward_scratch (correct, ~14× slower;
    // matches pre-Phase-B2 behaviour for that arch class). With
    // HIPFIRE_LLOYD_GFX12=1, the WMMA path is exercised — this is the
    // path RDNA4 reviewers should set when running the parity tests /
    // coherence-gate to validate the gfx12 sibling kernels. Once external
    // CI confirms gfx12 parity, the gate can be dropped (or default
    // flipped) in a follow-up commit.
    let lloyd_mq3_with_gfx12_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(arch, "gfx1200" | "gfx1201")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_GFX12")
            .ok()
            .as_deref()
            == Some("1");

    // Lloyd-MQ4 (MQ4G256Lloyd) on gfx11: shipped as part of issue #182.
    // Uses the gemm_*_mq4g256_lloyd_wmma family; group stride differs
    // (160 B Lloyd vs 136 B HFQ4) so dispatch routes through the
    // Lloyd-specific arms in forward_prefill_chunk.
    // ANTIBLEED admit-vs-select fix: the MQ4-Lloyd batched-prefill GEMM source
    // selectors (gemm_*_mq4g256_lloyd_wmma_for_arch in rdna-compute/kernels.rs)
    // ship a kernel only for gfx1100/1101/1102/1151 and PANIC on any other arch
    // (160 B Lloyd stride mismatches the default). gfx1150 was admitted here but
    // has no MQ4-Lloyd source (intentionally excluded to stay symmetric with the
    // MQ4-Lloyd GEMV/fused-decode path — see kernels.rs:195), so a gfx1150 box
    // doing MQ4-Lloyd batched prefill would crash at source lookup. Drop gfx1150
    // from the admit set so admit == select. (MQ3-Lloyd DOES ship a gfx1150
    // source, hence its admit set below keeps gfx1150.)
    let lloyd_mq4_with_gfx11_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(arch, "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151");

    // Lloyd-MQ4 on gfx12 (RDNA4): same opt-in gate as Lloyd-MQ3.
    let lloyd_mq4_with_gfx12_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(arch, "gfx1200" | "gfx1201")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_GFX12")
            .ok()
            .as_deref()
            == Some("1");

    // MFP4G32E8 on gfx11/gfx1151/gfx12: the mfp4-E8 A3B model takes the
    // batched-prefill path (FWHT-rotated activations + dequant→F16 GEMM for
    // the shared expert, indexed E8 kernels for the routed experts). Admission
    // is behind the HIPFIRE_E8_GFX12 gate because the shared-expert dequant
    // path is validated on gfx1151 only; other arches are opt-in for now.
    // The LA matchers (wqkv/wz/wo/etc.) for an MFP4G32E8 A3B model are
    // still MQ4/Q8 (only the FFN expert weights are E8), so reaching here
    // with DType::MFP4G32E8 means a weight was quantized to E8 dtype at the
    // LA level — admitting it keeps the eligibility gate from rejecting the
    // whole model when an attention tensor is E8 (unlikely today, but correct
    // defensively). The real admission gate for the FFN body is
    // `moe_ffn_batched_admissible`.
    let e8_with_wmma = matches!(dt, DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        )
        && hipfire_config::developer_var("HIPFIRE_E8_GFX12")
            .ok()
            .as_deref()
            == Some("1");

    mq3_uniform_with_wmma
        || mq3_uniform_with_gfx10_scalar
        || lloyd_mq3_with_gfx11_wmma
        || lloyd_mq3_with_gfx12_wmma
        || lloyd_mq4_with_gfx11_wmma
        || lloyd_mq4_with_gfx12_wmma
        || fp4_with_wmma
        || e8_with_wmma
}

pub(crate) fn trace_finite_if_enabled(gpu: &Gpu, label: &str, tensor: &GpuTensor) -> HipResult<()> {
    if hipfire_config::developer_var_os("HIPFIRE_QWEN35_FINITE_TRACE").is_none() {
        return Ok(());
    }
    let vals = gpu.download_f32(tensor)?;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_finite = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &vals {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_infinite() {
            n_inf += 1;
        } else {
            n_finite += 1;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    eprintln!(
        "[qwen35 finite] {label}: finite={n_finite}/{} nan={n_nan} inf={n_inf} range=[{min_v:.6e}, {max_v:.6e}]",
        vals.len(),
    );
    Ok(())
}

/// Process one chunk of up to `pbs.max_batch` tokens through the batched
/// prefill path. All LA layers go through batched kernels; all FA layers
/// go through a per-token gather/scatter loop with the inline FA body.
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states for configured
/// extract layers get written into the ring buffer at its current head. The
/// caller (forward_prefill_batch) advances the head by N after this chunk
/// completes so writes from the next chunk don't overwrite.
///
/// `per_token_hidden_out`: if `Some((dst, offset_rows))`, writes post-output
/// RMSNorm hidden for each of the N tokens into `dst[offset_rows..offset_rows+N]`
/// in row-major order. Required for DFlash verify to compute per-position
/// logits via B sequential `weight_gemv` calls on the caller side.
///
/// `gdn_tape` + `tape_offset`: if `Some`, captures the post-processed
/// `(q, k, v, α, β)` tensors per DN layer at rows
/// `[tape_offset .. tape_offset+N]` right before the batched GDN kernel
/// runs. Used by the DFlash rollback path.
/// Does the MoE FFN admit the batched prefill fast path?
///
/// Router + shared_expert_gate may be Q8_0 (the engine's default — these
/// small tensors are never quantized to MQ4 to preserve routing
/// accuracy). They get a separate `gemm_q8_0_batched_chunked` dispatch
/// against the *un-rotated* `x_norm_batch` inside
/// `prefill_moe_ffn_body_batched`. All other weights (shared expert
/// gate/up/down + every expert gate_up/down) must be MQ4G256 — these are
/// the ones consumed by the FWHT-rotated `_k8_indexed_batched` and
/// `gemm_hfq4g256` family, which is stride-136 only.
///
/// Pre-fix this required ALL weights to be MQ4G256, which made every
/// A3B model fall back to per-token prefill because router is universally
/// Q8_0. Widening to accept Q8 router + Q8 shared_expert_gate unlocks
/// uniform-MQ4 A3B variants (Qwen3.5-A3B, qwen3.6-35b-a3b-uniform.mq4).
/// Mixed-precision Qwen3.6-A3B (MQ6 in 16/40 layers) still falls back —
/// needs an MQ6 sibling for `_k8_indexed_batched`, follow-up work.
/// MoE FFN admit predicate for the batched prefill body
/// `prefill_moe_ffn_body_batched`. Per-projection MQ4 OR MQ6 admit:
///
/// - router, shared_expert_gate: MQ4 or Q8 (small scalars; dispatched
///   inline below).
/// - shared_expert.gate AND .up: same dtype, MQ4 or MQ6 (fused gate+up
///   kernel handles one storage layout per call).
/// - shared_expert.down: MQ4 or MQ6 (independent dtype).
/// - experts.gate_up: uniform across all experts in this layer, MQ4 or MQ6.
/// - experts.down: uniform across all experts in this layer, MQ4 or MQ6.
///
/// AWQ A3B dtype dump 2026-05-19 confirms experts are uniform per
/// projection per layer. The 4 grouped/fused dispatch sites in
/// `prefill_moe_ffn_body_batched` branch on the actual dtype, so a
/// layer admitted here is dispatchable end-to-end.
///
fn paro_batched_admit_enabled_from_env(value: Option<&str>) -> bool {
    // Default OFF (opt-in via HIPFIRE_PARO_BATCHED=1). The PARO batched prefill
    // path (ParoQ4G128 wqkv/wz/wo → gemm_hfq4g128 + per-weight Givens) was
    // only validated for finite logits, not coherence. Per-token fallback
    // (forward_scratch) is correct and avoids the echo bug. Set =1 to re-enable
    // for eval/benchmarking, understanding that output may differ from decode.
    value == Some("1")
}

#[derive(Debug, Clone, Copy)]
struct MoePrefillDtypes {
    router: DType,
    shared_expert_scalar_gate: DType,
    shared_expert_gate: DType,
    shared_expert_up: DType,
    shared_expert_down: DType,
    expert_gate_up: DType,
    expert_down: DType,
    expert_gate_up_uniform: bool,
    expert_down_uniform: bool,
    /// Routed experts are dtype-mixed (graded) AND carry an `expert_dtype_tags`
    /// table → served by the merged grouped-WMMA prefill kernel (per-expert
    /// MQ6/MQ4/MQ3L/MQ2L). When true, the per-expert *uniform* requirement is
    /// waived for the ROUTED experts; the router + shared expert still use their
    /// own batched paths and are validated normally. Without this, a graded file
    /// fails admission and silently drops to the per-token prefill fallback (the
    /// merged kernel never fires — observed as ~decode-speed prefill).
    routed_mixed_merged: bool,
}

impl MoePrefillDtypes {
    #[cfg(test)]
    fn uniform(dtype: DType) -> Self {
        Self {
            router: dtype,
            shared_expert_scalar_gate: dtype,
            shared_expert_gate: dtype,
            shared_expert_up: dtype,
            shared_expert_down: dtype,
            expert_gate_up: dtype,
            expert_down: dtype,
            expert_gate_up_uniform: true,
            expert_down_uniform: true,
            routed_mixed_merged: false,
        }
    }

    fn from_ffn(ffn: &MoeFfnWeights) -> Option<Self> {
        let first = ffn.experts.first()?;
        Some(Self {
            router: ffn.router.gpu_dtype,
            shared_expert_scalar_gate: ffn.shared_expert_gate.gpu_dtype,
            shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
            shared_expert_up: ffn.shared_expert.up.gpu_dtype,
            shared_expert_down: ffn.shared_expert.down.gpu_dtype,
            expert_gate_up: first.gate_up.gpu_dtype,
            expert_down: first.down.gpu_dtype,
            expert_gate_up_uniform: ffn
                .experts
                .iter()
                .all(|e| e.gate_up.gpu_dtype == first.gate_up.gpu_dtype),
            expert_down_uniform: ffn
                .experts
                .iter()
                .all(|e| e.down.gpu_dtype == first.down.gpu_dtype),
            routed_mixed_merged: ffn.expert_dtype_tags.is_some(),
        })
    }
}

fn moe_prefill_topk_shape_supported(k_top: usize, num_experts: usize) -> bool {
    k_top == 8 && num_experts <= 1024
}

fn moe_ffn_batched_admissible_for_dtypes(
    dtypes: &MoePrefillDtypes,
    admit_mq6: bool,
    admit_paro: bool,
    admit_e8: bool,
) -> bool {
    let router_ok = matches!(dtypes.router, DType::MQ4G256 | DType::Q8_0 | DType::F32);
    let shared_gate_ok = matches!(
        dtypes.shared_expert_scalar_gate,
        DType::MQ4G256 | DType::Q8_0 | DType::F32
    );
    // Graded (mixed-dtype) routed experts are served by the merged grouped-WMMA
    // prefill kernel, so the per-expert *uniform* requirement is waived for the
    // routed experts; the router + shared expert still go through their own
    // batched paths and are validated below.
    let routed_ok =
        dtypes.routed_mixed_merged || (dtypes.expert_gate_up_uniform && dtypes.expert_down_uniform);
    if !(router_ok && shared_gate_ok && routed_ok) {
        return false;
    }

    if dtypes.routed_mixed_merged {
        // Routed experts handled by the merged kernel (per-expert MQ6/MQ4/MQ3L/
        // MQ2L). Only require the SHARED expert to be batchable on its dense
        // path: MQ4 always, MQ6 when this arch admits MQ6 dense kernels.
        let shared_gu_ok = (dtypes.shared_expert_gate == DType::MQ4G256
            && dtypes.shared_expert_up == DType::MQ4G256)
            || (admit_mq6
                && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ6G256)
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = dtypes.shared_expert_down == DType::MQ4G256
            || (admit_mq6 && dtypes.shared_expert_down == DType::MQ6G256);
        return shared_gu_ok && shared_dn_ok;
    }

    // mfp4-E8 routed experts with Q8 shared expert (original arm):
    // gfx1151-native A3B checkpoint. Shared expert is Q8 (gate/up/down);
    // router/scalar-gate are Q8 (validated by router_ok/shared_gate_ok above).
    // The batched body runs a dedicated Q8 shared-expert path (two plain Q8
    // GEMMs + silu_mul + sigmoid-scaled residual add) and routes the E8
    // experts through `run_moe_prefill` Path 1 (indexed batched GEMV).
    // E8-family match helper: MFP4, MFP3, MFP2 lattice types.
    let is_e8_family =
        |dt: DType| matches!(dt, DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8);

    if admit_e8
        && dtypes.shared_expert_gate == DType::Q8_0
        && dtypes.shared_expert_up == DType::Q8_0
        && dtypes.shared_expert_down == DType::Q8_0
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
    {
        return true;
    }

    // Uniform mfp4/mfp3/mfp2-E8: BOTH shared AND routed experts are E8-family
    // (Option B from the implementation spec). Router + shared_expert_gate
    // (scalar) remain Q8 (validated above). The batched body dequants the shared
    // expert E8→F16 transiently and runs `gemm_f16_wmma_mb8` against
    // `x_rot_batch` (FWHT-rotated activations), then the routed experts go
    // through the indexed E8 batched GEMV path. The dequant→F16 path requires
    // has_wmma_w32 (gfx11+), which `admit_e8` already gates on arch.
    if admit_e8
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
        // Shared expert may be per-projection MIXED — gate+up are dispatched
        // together (one match on gate's dtype) so they must share a dtype;
        // down is matched independently. The batched body handles Q8 (un-rotated)
        // and E8 (dequant→f16, x_rot) per projection and keys the SwiGLU rotate
        // on the down dtype, so any {Q8,E8-family} combination of (gate==up,down)
        // is correct.
        && dtypes.shared_expert_gate == dtypes.shared_expert_up
        && matches!(dtypes.shared_expert_gate, DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8)
        && matches!(dtypes.shared_expert_down, DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8)
    {
        return true;
    }

    if admit_paro
        && dtypes.shared_expert_gate == DType::ParoQ4G128
        && dtypes.shared_expert_up == DType::ParoQ4G128
        && dtypes.shared_expert_down == DType::ParoQ4G128
        && dtypes.expert_gate_up == DType::ParoQ4G128
        && dtypes.expert_down == DType::ParoQ4G128
    {
        return true;
    }

    if admit_mq6 {
        let shared_gu_dt = dtypes.shared_expert_gate;
        let shared_gu_ok = matches!(shared_gu_dt, DType::MQ4G256 | DType::MQ6G256)
            && dtypes.shared_expert_up == shared_gu_dt;
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ6G256);
        let experts_ok = matches!(dtypes.expert_gate_up, DType::MQ4G256 | DType::MQ6G256)
            && matches!(dtypes.expert_down, DType::MQ4G256 | DType::MQ6G256);
        shared_gu_ok && shared_dn_ok && experts_ok
    } else {
        dtypes.shared_expert_gate == DType::MQ4G256
            && dtypes.shared_expert_up == DType::MQ4G256
            && dtypes.shared_expert_down == DType::MQ4G256
            && dtypes.expert_gate_up == DType::MQ4G256
            && dtypes.expert_down == DType::MQ4G256
    }
}

/// Threshold below which batching overhead isn't worth the alloc + per-layer
/// dispatch — single-token prefill must not take the batched path.
const MIN_BATCH: usize = 2;

/// Choose the next prefill chunk without leaving an invalid singleton tail.
///
/// Batched kernels require at least `MIN_BATCH` rows. When a full chunk would
/// leave one row, move one row into the tail (for example, 257 → 255 + 2).
#[inline]
fn next_prefill_chunk_len(remaining: usize, max_batch: usize) -> Option<usize> {
    if remaining < MIN_BATCH || max_batch < MIN_BATCH {
        return None;
    }
    if max_batch == MIN_BATCH && remaining % MIN_BATCH != 0 {
        return None;
    }

    let chunk = remaining.min(max_batch);
    if remaining - chunk == 1 {
        (chunk > MIN_BATCH).then_some(chunk - 1)
    } else {
        Some(chunk)
    }
}

/// Plan scratch owned by one prefill call.
///
/// Large prompts retain the configured chunk size, while a short prompt gets
/// exactly one right-sized chunk. The DeltaNet S-state tape is consumed only by
/// tree verify; ordinary prefill advances recurrent state in place.
#[inline]
fn owned_prefill_scratch_plan(
    n: usize,
    configured_max_batch: usize,
    tree_verify: bool,
) -> (usize, bool) {
    debug_assert!(n > 0);
    debug_assert!(configured_max_batch >= MIN_BATCH);
    (configured_max_batch.min(n.max(MIN_BATCH)), tree_verify)
}

/// Whether `forward_prefill_batch_with_pbs` will take the tape-capturing
/// batched (PBS) path for an `n`-token call — equivalently, whether a `GdnTape`
/// handed to that forward will actually be populated. When this is false the
/// forward silently drops to a tape-less per-token loop, so spec-decode callers
/// that later replay the GDN tape MUST gate that cheap replay on this predicate;
/// otherwise they replay a stale/zero tape and corrupt DeltaNet state. This is
/// the single source of truth for the eligibility decision — called by the
/// forward itself and by those callers, so the two can never drift. (The
/// tree-verify forward keeps its own, deliberately simpler, eligibility check.)
pub fn prefill_batch_pbs_eligible(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    // Kept for API stability and future state-aware gating. The batched path
    // now dispatches the GDN recurrence by state quant on the non-tree route,
    // so it no longer gates eligibility here (see the removed Q8-only check).
    _dn_state: &DeltaNetState,
    n: usize,
    arch: &str,
    moe_router_logits_present: bool,
) -> bool {
    // HIPFIRE_PREFILL_BATCHED=0 forces the per-token fallback — an escape hatch
    // for the LARGE seed prefill (gfx11 24GB OOM + a batched-seed correctness bug
    // that collapses MTP τ→1.0). But the small-B MTP verify (n = K+1, ≤ ~32) is
    // cheap and its BATCHED path is the dominant gfx11 decode lever: per-token it
    // costs ~K full sequential trunk forwards/cycle (the measured 92ms→16.8ms /
    // 41→223 tok/s bottleneck, rocprofv3 2026-06-16). Decouple: let the small-B
    // verify batch even when the flag forces the seed per-token. Opt-in
    // (HIPFIRE_MTP_VERIFY_DECOUPLE=1) until the batched verify is validated
    // coherent + τ-preserving per-arch (the gfx11 batched *seed* corrupts; whether
    // the small-B *verify* also corrupts is exactly what this gate tests).
    // DEFAULT-ON for RDNA3 (gfx11) — the small-B verify BATCHED is validated
    // coherent + τ-preserving there (W3x 2026-06-16: byte-identical output vs
    // per-token at 240-tok ctx; +20% mq4; the scalar→WMMA + MQ3L-LUT kernel
    // wins lift all STRUCTURED domains >AR on both mq4/mq4p; fresh default-config
    // re-validated mq4 code 1.26× / mq4p chat 1.07×). Opt-out
    // HIPFIRE_MTP_VERIFY_DECOUPLE=0. Other archs opt-in (=1) until validated;
    // gfx12 batches the whole prefill already so it is moot there. The seed
    // stays per-token for LONG prompts (n>32 fails this gate → force_fallback
    // when PREFILL_BATCHED=0); a short seed (n≤32) batches, fine for mq4/mq4p
    // (E8 short-seed batched-prefill can OOM, but E8 admission is itself opt-in
    // via HIPFIRE_E8_GFX12 so the default config never reaches it).
    let decouple_env = hipfire_config::developer_var("HIPFIRE_MTP_VERIFY_DECOUPLE").ok();
    let is_rdna3_decouple = arch.starts_with("gfx11");
    let verify_decouple = n <= 32
        && decouple_env.as_deref() != Some("0")
        && (is_rdna3_decouple || decouple_env.as_deref() == Some("1"));
    let force_fallback = !verify_decouple && !hipfire_runtime::config::get().prefill_batched;
    // MoE batched path requires K_TOP=8 (hard-coded in the indexed kernels) and
    // num_experts ≤ 1024 (bound of the batched top-K shared mem).
    let moe_topk_ok =
        moe_prefill_topk_shape_supported(config.num_experts_per_tok, config.num_experts);
    let admit_mq6 = mq6_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
            .ok()
            .as_deref(),
        arch,
    );
    let has_dn = weights
        .layers
        .iter()
        .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),));
    let all_dtypes_ok = weights.layers.iter().all(|lw| match lw {
        LayerWeights::DeltaNet(l) => {
            is_batchable_la(l.wqkv.gpu_dtype, arch)
                && is_batchable_la(l.wz.gpu_dtype, arch)
                && is_batchable_la(l.w_beta.gpu_dtype, arch)
                && is_batchable_la(l.w_alpha.gpu_dtype, arch)
                && is_batchable_la(l.wo.gpu_dtype, arch)
                && is_batchable_la(l.w_gate.gpu_dtype, arch)
                && is_batchable_la(l.w_up.gpu_dtype, arch)
                && is_batchable_la(l.w_down.gpu_dtype, arch)
        }
        LayerWeights::FullAttn(_) => true,
        LayerWeights::DeltaNetMoe(l) => {
            moe_topk_ok
                && moe_router_logits_present
                && is_batchable_la(l.wqkv.gpu_dtype, arch)
                && is_batchable_la(l.wz.gpu_dtype, arch)
                && is_batchable_la(l.w_beta.gpu_dtype, arch)
                && is_batchable_la(l.w_alpha.gpu_dtype, arch)
                && is_batchable_la(l.wo.gpu_dtype, arch)
                && moe_ffn_batched_admissible(&l.ffn, admit_mq6, arch)
        }
        LayerWeights::FullAttnMoe(l) => {
            moe_topk_ok
                && moe_router_logits_present
                && is_batchable_la(l.wq.gpu_dtype, arch)
                && is_batchable_la(l.wk.gpu_dtype, arch)
                && is_batchable_la(l.wv.gpu_dtype, arch)
                && is_batchable_la(l.wo.gpu_dtype, arch)
                && moe_ffn_batched_admissible(&l.ffn, admit_mq6, arch)
        }
    });
    let result = !force_fallback
        && n >= MIN_BATCH
        // State quant no longer gates batched prefill: forward_prefill_chunk
        // dispatches the GDN recurrence by dn_state.quant on the non-tree path
        // (FP32 → gated_delta_net_f32_batch_seq, Q8 → _q8_batch_seq, Q4 → _q4),
        // so FP32/Q4 state is fully batchable here. Was hard-gated to Q8 when
        // the batched GDN was Q8-only; that's the seed + per-cycle-commit
        // per-token fallback that made FP32 DFlash ~4.5× slower + 10× TTFT.
        && has_dn
        // LA/FA/MoE projection + MoE-FFN weight dtypes must all be batchable;
        // A3B engine policy quantizes attention as Q8 (admitted alongside MQ4).
        && all_dtypes_ok;
    // HIPFIRE_DEBUG_BATCH=1: print per-component eligibility to stderr.
    if hipfire_config::developer_var("HIPFIRE_DEBUG_BATCH")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[hipfire::batch_eligible] result={result} \
             arch={arch} n={n} n>={MIN_BATCH}={} \
             force_fallback={force_fallback} \
             has_dn={has_dn} \
             moe_topk_ok={moe_topk_ok} \
             moe_router_logits_present={moe_router_logits_present} \
             all_dtypes_ok={all_dtypes_ok}",
            n >= MIN_BATCH,
        );
    }
    result
}

/// Whether MQ6 MoE FFN projections can enter batched prefill. Default-on for
/// gfx11 (RDNA3/3.5) AND gfx12 (RDNA4): the MQ6 grouped-WMMA decode is present
/// on both (tag 0 of the merged `gemm_mixed_moe_grouped_wmma{_k2,.gfx12}` kernel,
/// plus the standalone `gemm_hfq6g256_moe_grouped_wmma{_k2,.gfx12,_gfx1151}` ported
/// 2026-06-11/12), and the graded shared-MQ6 expert runs the dense-GEMM batched
/// path. Validated on gfx1100: graded T3-3L-E8 (MQ6 hot / E8 mid / MQ3L cold,
/// MQ6 shared) batches coherently, KLD 0.038964, and gfx11 prefill is ~10× the
/// per-token fallback (1012 vs 106 tok/s pp512). UNIFORM-MQ6 OOMs on gfx11
/// (>24 GB) so it never reaches this gate there — only graded MQ6 models do.
/// The original gfx12-only default predated the gfx11 MQ6 grouped port and
/// silently forced per-token prefill on every graded-MQ6 model on gfx11.
/// gfx1151 (RDNA3.5, Strix Halo) additionally has master's channel-tested
/// routed grouped-WMMA MQ6 fast-path (its unrelated Q8 WMMA prefill family is
/// gated separately by `q8_prefill_wmma_enabled`). Override per-arch with
/// `HIPFIRE_MOE_MQ6_ADMIT=0|1`.
fn mq6_batched_admit_enabled_from_env(value: Option<&str>, arch: &str) -> bool {
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        // RDNA3/3.5 (gfx11xx, includes gfx1151) + RDNA4 (gfx12xx). RDNA1/2
        // (gfx10xx, no WMMA) stay off. The gfx11 widen (8d555fc6) subsumes
        // master's narrower gfx12||gfx1151 default; gfx1151 still picks up its
        // channel-tested grouped-WMMA fast-path inside the kernel dispatcher.
        _ => arch.starts_with("gfx11") || arch.starts_with("gfx12"),
    }
}

/// Qwen3.5 batched prefill can run Q8 projections through fused WMMA kernels
/// or through the older chunked-Q8 substrate. gfx12 has a separate WMMA ABI;
/// gfx11/gfx1151 use the gfx11 wave32 WMMA ABI. The low-level Q8 channel tests
/// cover the fused, residual, and generic chunked drop-in paths, so default on
/// for every arch that advertises wave32 WMMA while preserving the env opt-out.
fn q8_prefill_wmma_enabled_from_env(value: Option<&str>, arch: &str, has_wmma: bool) -> bool {
    let _ = arch;
    if !has_wmma {
        return false;
    }
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => true,
    }
}

fn q8_prefill_wmma_enabled(gpu: &Gpu) -> bool {
    q8_prefill_wmma_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_Q8_PREFILL_WMMA")
            .ok()
            .as_deref(),
        gpu.arch.as_str(),
        gpu.arch_caps.has_wmma(),
    )
}

fn moe_ffn_batched_admissible(ffn: &MoeFfnWeights, admit_mq6: bool, arch: &str) -> bool {
    let Some(dtypes) = MoePrefillDtypes::from_ffn(ffn) else {
        return false;
    };
    // mfp4-E8 routed experts: originally gfx1151-only, now widened to all
    // gfx11 + gfx12 arches behind the HIPFIRE_E8_GFX12 gate. The shared-expert
    // dequant→F16 GEMM path uses `dequantize_mfp4g32_e8_to_f16` + `gemm_f16_wmma_mb8`
    // which are available on all WMMA-capable arches. The indexed E8 GEMV kernels
    // for the routed experts are also present on gfx11 (shipped in the MoE-AWQ
    // branch). Gate on HIPFIRE_E8_GFX12 to allow safe opt-in rollout.
    let admit_e8 = matches!(
        arch,
        "gfx1100"
            | "gfx1101"
            | "gfx1102"
            | "gfx1150"
            | "gfx1151"
            | "gfx1152"
            | "gfx1200"
            | "gfx1201"
    ) && hipfire_config::developer_var("HIPFIRE_E8_GFX12")
        .ok()
        .as_deref()
        == Some("1");

    // PARO admit is default-on. Set HIPFIRE_PARO_BATCHED=0 to force the old
    // fallback path while bisecting or debugging.
    // for shisa-Qwen3.6-A3B-PARO and similar ParoQuant checkpoints where the
    // routed-expert + shared-expert weights are ParoQ4G128 (HFQ4G128 +
    // per-weight Givens rotation metadata). The downstream dispatch arms for
    // ParoQ4G128 are implemented on this branch. See roadmap at
    // .claude/plans/magical-marinating-hippo.md.
    static PARO_ADMIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let admit_paro = *PARO_ADMIT.get_or_init(|| {
        paro_batched_admit_enabled_from_env(
            hipfire_config::developer_var("HIPFIRE_PARO_BATCHED")
                .ok()
                .as_deref(),
        )
    });

    moe_ffn_batched_admissible_for_dtypes(&dtypes, admit_mq6, admit_paro, admit_e8)
}

/// #397 Ship 5.2 slice 1: route a single PLAIN-batched prefill GEMM through
/// [`GemmFamily::run_key`] against an *explicit* dispatcher-entry [`KernelKey`].
///
/// This is the behavior-preserving migration primitive proved by the Ship 5.2
/// pilot (028ac9f3): passing the dispatcher-entry key (e.g.
/// `GemmQ8_0BatchedChunked`, `GemmHfq4G256`, `GemmHfq4G128`, `GemmF32Batched`)
/// makes `run_key` dispatch to the IDENTICAL `gpu.gemm_*` method the direct
/// call used, so each method's own internal arch routing (RDNA4-WMMA /
/// gfx906-dp4a / CDNA-rocBLAS / …) is preserved byte-for-byte on every
/// (dtype × arch × shape). `resolve()` is deliberately NOT used here — it
/// front-runs the kernel's internal dispatch with a dtype-keyed WMMA preference
/// and can diverge from a direct dispatcher-entry call on some arches.
///
/// Only the four PLAIN-batched dispatcher-entry keys with existing table
/// entries are valid here. Residual-fused kernels (`gemm_*_residual*`) and the
/// fused QKVZA / gate+up kernels are NOT plain GEMMs and are migrated in later
/// slices (they need new table entries).
#[inline]
fn run_plain_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::gemm::GemmParams;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(HipError::from)
}

/// #397 Ship 5.2 FINAL: route a single BATCHED-prefill RESIDUAL-fused GEMM
/// (`y += W·x`) through [`GemmFamily::run_key`] against an explicit
/// `Gemm*Residual` [`KernelKey`].
///
/// Residual analogue of [`run_plain_gemm_key`]. The residual op writes its
/// output IN-PLACE into the residual stream `y` (which carries the pre-add
/// value); the `gpu.gemm_*_residual` kernels perform the add internally and
/// NEVER reuse `y` as GEMV scratch, so the migration cannot reintroduce the
/// a9e8dfda aliasing bug — `y`, the residual/input `x`, and the weight buffer
/// are passed in the IDENTICAL order the direct call used. Each residual key
/// routes to the same `gpu.gemm_*_residual` method (which keeps its own internal
/// arch routing: WMMA/gfx12-WMMA / dp4a / fp16 / scalar) byte-for-byte. For
/// HFQ3 the run-arm replicates the call-site WMMA-vs-base arch split internally
/// via `gpu.arch_caps`; `resolve()` only confirms the entry's ArchPredicate
/// admits the current arch (it is NOT used to front-run the kernel's dispatch).
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_residual_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::gemm::GemmParams;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    // The residual stream `y` is BOTH the residual and the output (`y += W·x`).
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(HipError::from)
}

/// #397 Ship 5.2 slice 2: route a single BATCHED-prefill FUSED gate+up GEMM
/// through [`FusedQkvFamily`] against an explicit `FusedGateUp*` [`KernelKey`].
///
/// This is the gate+up analogue of [`run_plain_gemm_key`]. Unlike a plain GEMM,
/// gate+up carries TWO weights (gate, up) and writes TWO outputs in one fused
/// launch, so it goes through `FusedQkvFamily` (the gate+up variant) rather than
/// `GemmFamily`. Passing `batch_size: Some(n)` makes the family's gate+up run-arm
/// dispatch to the IDENTICAL batched `gpu.gemm_gate_up_*(.., n)` method the direct
/// prefill call used — each method keeps its own internal arch routing
/// (RDNA4-WMMA / gfx906-dp4a / MMQ / fp16 / scalar) byte-for-byte. The weights,
/// activation `x` (already rmsnorm-rotated by the caller), outputs and m/k/n args
/// are unchanged at every migrated site.
///
/// The `FusedGateUp*` key carries the dtype; the run-arm replicates any
/// call-site arch split (e.g. HFQ3 WMMA-vs-base) internally via `gpu.arch_caps`,
/// so the same kernel runs. `resolve()` only confirms the entry's ArchPredicate
/// admits the current arch — it does NOT front-run the kernel's internal dispatch.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_fused_gate_up_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_gate: &GpuTensor,
    w_up: &GpuTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    gate_m: usize,
    up_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_gate, w_up],
        x,
        outputs: &[y_gate, y_up],
        m: &[gate_m, up_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Dispatch a batched-prefill **3-way fused QKV** projection (wq+wk+wv) through
/// [`FusedQkvFamily`] against an explicit `FusedQkv*` [`KernelKey`]
/// (`#397 Ship 5.2 slice 3`).
///
/// QKV analogue of [`run_fused_gate_up_key`]: three weights (wq, wk, wv), three
/// outputs (q, k, v), three row-counts. Passing `batch_size: Some(n)` routes the
/// family's QKV run-arm to the IDENTICAL batched `gpu.gemm_qkv_*(.., n)` method
/// the direct prefill call used — each method keeps its own internal arch routing
/// (RDNA4-WMMA / gfx906-dp4a / MMQ / fp16 / scalar) byte-for-byte. The weights,
/// activation `x` (already rmsnorm[-rotated] by the caller), outputs and m/k/n
/// args are unchanged at every migrated site. The `FusedQkv*` key carries the
/// dtype; for HFQ3 the run-arm replicates the call-site WMMA-vs-base arch split
/// internally via `gpu.arch_caps`. `resolve()` only confirms the entry's
/// ArchPredicate admits the current arch.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_fused_qkv_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    wq: &GpuTensor,
    wk: &GpuTensor,
    wv: &GpuTensor,
    x: &GpuTensor,
    y_q: &GpuTensor,
    y_k: &GpuTensor,
    y_v: &GpuTensor,
    q_m: usize,
    k_m: usize,
    v_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[wq, wk, wv],
        x,
        outputs: &[y_q, y_k, y_v],
        m: &[q_m, k_m, v_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Dispatch a batched-prefill **4-way fused QKVZA** projection (DeltaNet linear
/// attention: wqkv + wz + w_beta + w_alpha) through [`FusedQkvFamily`] against an
/// explicit `FusedQkvza*` [`KernelKey`] (`#397 Ship 5.2 slice 3`).
///
/// QKVZA analogue of [`run_fused_qkv_key`]: four weights, four outputs, four
/// row-counts. `batch_size: Some(n)` routes the family's QKVZA run-arm to the
/// IDENTICAL batched `gpu.gemm_qkvza_*(.., n)` method the direct prefill call
/// used. All operands are passed unchanged; for HFQ3 the run-arm replicates the
/// call-site WMMA-vs-base arch split internally.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_fused_qkvza_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_qkv: &GpuTensor,
    w_z: &GpuTensor,
    w_beta: &GpuTensor,
    w_alpha: &GpuTensor,
    x: &GpuTensor,
    y_qkv: &GpuTensor,
    y_z: &GpuTensor,
    y_beta: &GpuTensor,
    y_alpha: &GpuTensor,
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_qkv, w_z, w_beta, w_alpha],
        x,
        outputs: &[y_qkv, y_z, y_beta, y_alpha],
        m: &[qkv_m, z_m, beta_m, alpha_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Batched MoE FFN for `forward_prefill_chunk`. Takes the post-attention
/// residual stream in `pbs.x_batch` ([N × dim]) and writes the FFN output
/// residual back into the same buffer in-place.
///
/// Preconditions (caller must guarantee):
/// - `moe_ffn_batched_admissible(ffn)` returns true: router + shared_expert_gate may
///   be MQ4G256 *or* Q8_0; all other MoE weights must be MQ4G256
/// - `pbs.moe_*_batch` tensors are allocated (num_experts > 0 at scratch
///   construction time) and sized to max_batch ≥ N
/// - `config.num_experts_per_tok == 8` and `config.num_experts <= 1024`
///   (hard limits of the batched top-K kernel)
///
/// Sequence mirrors `moe_ffn_decode_impl`'s GPU fast path, with every
/// per-token launch replaced by its N-batched equivalent. Byte-exact
/// except for atomicAdd nondeterminism in the routed-down accumulation
/// (same as the single-token indexed kernel it replaces).
#[allow(clippy::too_many_arguments)]
fn prefill_moe_ffn_body_batched(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    ctx: &DispatchCtx,
    model_has_mq6_moe: bool,
    // EP (Ship 6 substrate-EP prefill): when `Some`, the routed combine writes
    // into this zeroed `[n × dim]` partial instead of `pbs.x_batch` (the EP
    // driver all-reduce-sums it across ranks and adds into x_batch). The shared
    // expert (step 5) stays in `pbs.x_batch` — replicated per rank, not
    // redirected. `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");
    let topk_indices = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let rot_batch = pbs.moe_rot_batch.as_ref().expect("moe scratch");
    let down_expanded = pbs.moe_down_expanded_batch.as_ref().expect("moe scratch");

    // ── 1. Split rmsnorm vs FWHT rotate ──
    //
    // A3B (and every other MoE here) leaves router + shared_expert_gate
    // as Q8_0 in the quantizer — these tiny tensors lose too much
    // accuracy at 4-bit, so the engine never reduces them. Q8 weights
    // are quantized against the un-rotated rmsnorm output, while the
    // MQ4 siblings (shared_expert.{gate,up,down} + experts.{gate_up,down})
    // expect FWHT(rmsnorm(x) / awq_scale). Populate both:
    //   x_norm_batch ← rmsnorm(x_batch)
    //   x_rot_batch  ← FWHT(x_norm_batch / awq_scale)  (only if any
    //                  downstream MQ weight is present, which moe_ffn_batched_admissible
    //                  guarantees — shared_expert.gate is always MQ4 here)
    //
    // Pick `shared_expert.gate` as the AWQ representative (instead of
    // the previous `ffn.router`). Per the F1 imatrix scope every gate-side
    // MQ4 sibling shares the same input basis and therefore an identical
    // awq_scale, but the router itself is excluded from F1 (it stays Q8).
    // Reading awq_scale from router would silently drop AWQ rotation in
    // v3 AWQ runs — latent until this predicate widened.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        ffn_norm,
        &pbs.x_norm_batch,
        n,
        dim,
        config.norm_eps,
    )?;
    // PARO mode (shared_expert.gate is ParoQ4G128): each weight carries its
    // own Givens rotation table (paro.pairs / theta / channel_scales). The
    // shared MQ4-style FWHT pre-rotation here would be wrong — skip it. The
    // ParoQ4G128 dispatch arms below run per-weight Givens rotation in-place
    // before each GEMM, using pbs.x_rot_batch as the rotation destination.
    let paro_mode = matches!(ffn.shared_expert.gate.gpu_dtype, DType::ParoQ4G128);
    if !paro_mode {
        rotate_x_mq_batched_for(
            gpu,
            &ffn.shared_expert.gate,
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            dim,
            n,
        )?;
    }

    // ── 2. Router + shared-gate + shared.gate + shared.up (4 batched GEMMs) ──
    //
    // Per-dtype dispatch — Q8 reads `x_norm_batch`, MQ4 reads
    // `x_rot_batch`. The natural 4-way fuse via `gemm_qkvza_hfq4g256`
    // is not applicable when router/shared_expert_gate are Q8 (mixed
    // strides). Four separate launches; +3 per MoE layer over the fused
    // ideal, acceptable for the structural unlock.
    // #397 Ship 5.2 PILOT: route the router GEMM through GemmFamily::run_key.
    // Each arm uses the *dispatcher-entry* KernelKey (GemmQ8_0BatchedChunked /
    // GemmHfq4G256 / GemmF32Batched) so run_key dispatches to the IDENTICAL
    // gpu.gemm_* method the prior direct call used — preserving each method's
    // own internal arch routing (RDNA4-WMMA / gfx906-dp4a / CDNA-rocBLAS / …)
    // byte-for-byte. The x input still differs per dtype (Q8/F32 read
    // x_norm_batch; MQ4 reads x_rot_batch), exactly as before. The three keys
    // are registered ArchPredicate::Always, so run_key never rejects.
    {
        use hipfire_dispatch::families::gemm::GemmParams;
        let ctx = DispatchCtx::new(gpu);
        let (key, x_in): (hipfire_dispatch::types::KernelKey, &GpuTensor) =
            match ffn.router.gpu_dtype {
                DType::Q8_0 => (
                    hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                    &pbs.x_norm_batch,
                ),
                DType::MQ4G256 => (
                    hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                    &pbs.x_rot_batch,
                ),
                DType::F32 => (
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &pbs.x_norm_batch,
                ),
                other => panic!(
                    "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admitted admits MQ4G256, Q8_0, F32"
                ),
            };
        let w = WeightRef {
            buf: &ffn.router.buf,
            dtype: ffn.router.gpu_dtype,
            m: ffn.router.m,
            k: ffn.router.k,
            row_stride: ffn.router.k,
            rotation: None,
            awq_scale: None,
        };
        let params = GemmParams {
            w: &w,
            x: x_in,
            y: router_logits,
            batch_size: n,
        };
        hipfire_runtime::llama::gemm_family()
            .run_key(key, &ctx, gpu, &params)
            .map_err(HipError::from)?;
    }
    // DIAG: dump MoE router logits (batched)
    dump_hidden_localize(gpu, router_logits, n, 0, ffn.router.m, 0, "router_b");
    // #397 Ship 5.2 slice1: route the shared-expert-gate GEMM through
    // GemmFamily::run_key. Same dtype-routed dispatcher-entry keys as the router
    // match above (Q8/F32 read x_norm_batch, MQ4 reads x_rot_batch) → identical
    // gpu.gemm_* method, byte-for-byte.
    {
        use hipfire_dispatch::types::KernelKey;
        let (key, x_in): (KernelKey, &GpuTensor) = match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => (KernelKey::GemmQ8_0BatchedChunked, &pbs.x_norm_batch),
            DType::MQ4G256 => (KernelKey::GemmHfq4G256, &pbs.x_rot_batch),
            DType::F32 => (KernelKey::GemmF32Batched, &pbs.x_norm_batch),
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32"
            ),
        };
        run_plain_gemm_key(
            gpu,
            key,
            &ffn.shared_expert_gate.buf,
            ffn.shared_expert_gate.gpu_dtype,
            x_in,
            shared_scalar,
            ffn.shared_expert_gate.m,
            ffn.shared_expert_gate.k,
            n,
        )?;
    }
    // Fused gate+up dispatch for the shared expert — halves the kernel
    // launch count vs back-to-back gemm_hfq*g256 (~75µs/launch × 40
    // MoE layers = ~3ms saved on R9700 A3B prefill at bs=256).
    // Per-projection dispatch: gate AND up share the same dtype (predicate
    // enforces). MQ4 → HFQ4-layout fused kernel; MQ6 → HFQ6-layout.
    match ffn.shared_expert.gate.gpu_dtype {
        // #397 Ship 5.2 slice 2: shared-expert fused gate+up → FusedQkvFamily
        // (batched-prefill gate+up variant). Same batched kernel, behavior-preserving.
        DType::MQ4G256 => run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
            &ffn.shared_expert.gate.buf,
            &ffn.shared_expert.up.buf,
            &pbs.x_rot_batch,
            shared_gate,
            shared_up,
            ffn.shared_expert.gate.m,
            ffn.shared_expert.up.m,
            ffn.shared_expert.gate.k,
            n,
        )?,
        DType::MQ6G256 => run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
            &ffn.shared_expert.gate.buf,
            &ffn.shared_expert.up.buf,
            &pbs.x_rot_batch,
            shared_gate,
            shared_up,
            ffn.shared_expert.gate.m,
            ffn.shared_expert.up.m,
            ffn.shared_expert.gate.k,
            n,
        )?,
        // Phase 2: PARO shared_expert.gate + up. Each weight has its own
        // Givens rotation table — rotate x_norm_batch into x_rot_batch using
        // gate's tables, GEMM, then re-rotate using up's tables, GEMM. Total
        // 4 dispatches vs the MQ4 path's 1 fused gemm_gate_up — acceptable
        // overhead for the per-token-loop elimination win. Phase 4 could
        // collapse this into a single fused kernel
        // (gemm_gate_up_paro_q4g128_batched) if measurement shows it matters.
        DType::ParoQ4G128 => {
            let paro_gate = ffn
                .shared_expert
                .gate
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.gate missing paro metadata");
            let paro_up = ffn
                .shared_expert
                .up
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.up missing paro metadata");
            // Gate: rotate x_norm by gate's Givens → x_rot, then HFQ4G128 GEMM
            gpu.givens_rotate_to(
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                &paro_gate.pairs,
                &paro_gate.theta,
                &paro_gate.channel_scales,
                n,
                dim,
                paro_gate.krot as usize,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                &ffn.shared_expert.gate.buf,
                ffn.shared_expert.gate.gpu_dtype,
                &pbs.x_rot_batch,
                shared_gate,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.gate.k,
                n,
            )?;
            // Up: re-rotate x_norm by up's Givens → x_rot (overwrite), GEMM
            gpu.givens_rotate_to(
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                &paro_up.pairs,
                &paro_up.theta,
                &paro_up.channel_scales,
                n,
                dim,
                paro_up.krot as usize,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                &ffn.shared_expert.up.buf,
                ffn.shared_expert.up.gpu_dtype,
                &pbs.x_rot_batch,
                shared_up,
                ffn.shared_expert.up.m,
                ffn.shared_expert.up.k,
                n,
            )?;
        }
        // Q8 shared expert (A3B mfp4-E8): gate + up via two batched Q8 GEMMs
        // reading the UN-rotated x_norm_batch (Q8 weights are quantized against
        // un-rotated rmsnorm output). No fused Q8 gate+up kernel — two plain
        // launches; mirrors the decode `gemv.run_auto` Q8 shared gate/up arm.
        DType::Q8_0 => {
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &ffn.shared_expert.gate.buf,
                ffn.shared_expert.gate.gpu_dtype,
                &pbs.x_norm_batch,
                shared_gate,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.gate.k,
                n,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &ffn.shared_expert.up.buf,
                ffn.shared_expert.up.gpu_dtype,
                &pbs.x_norm_batch,
                shared_up,
                ffn.shared_expert.up.m,
                ffn.shared_expert.up.k,
                n,
            )?;
        }
        // Uniform mfp4-E8 shared expert (Option B): shared expert gate/up are both
        // MFP4G32E8. We dequant each to F16 transiently (E8→F16 gives W_rot in the
        // FWHT-rotated domain), then run GemmF16WmmaMb8 against x_rot_batch (F32).
        // Math: GEMM(W_rot, x_rot) = W·x_norm = correct forward pass.
        // The F16 scratch tensors are allocated per call and freed after use; they
        // are small (smi × dim × 2 bytes each) relative to VRAM.
        DType::MFP4G32E8 => {
            let gate_m = ffn.shared_expert.gate.m;
            let gate_k = ffn.shared_expert.gate.k;
            let up_m = ffn.shared_expert.up.m;
            let up_k = ffn.shared_expert.up.k;
            // Dequantize gate and up weights: E8 → F16 (in-rotated domain)
            let gate_f16 = gpu.alloc_tensor(&[gate_m * gate_k], DType::F16)?;
            gpu.dequantize_mfp4g32_e8_to_f16(
                &ffn.shared_expert.gate.buf.buf,
                &gate_f16.buf,
                gate_m,
                gate_k,
            )?;
            let up_f16 = gpu.alloc_tensor(&[up_m * up_k], DType::F16)?;
            gpu.dequantize_mfp4g32_e8_to_f16(
                &ffn.shared_expert.up.buf.buf,
                &up_f16.buf,
                up_m,
                up_k,
            )?;
            // GemmF16WmmaMb8: W(F16) × x_rot_batch(F32) → shared_gate / shared_up (F32)
            // x_rot_batch is F32; gemm_f16_wmma_mb8 accepts F32 activations directly.
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmF16WmmaMb8,
                &gate_f16,
                DType::F16,
                &pbs.x_rot_batch,
                shared_gate,
                gate_m,
                gate_k,
                n,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmF16WmmaMb8,
                &up_f16,
                DType::F16,
                &pbs.x_rot_batch,
                shared_up,
                up_m,
                up_k,
                n,
            )?;
            // Free the transient F16 weight buffers
            gpu.free_tensor(gate_f16)?;
            gpu.free_tensor(up_f16)?;
        }
        other => panic!(
            "prefill_moe_ffn_body_batched: unsupported shared_expert.gate dtype {other:?} \
                         — admit predicate should have rejected this layer"
        ),
    }

    // ── 3. GPU softmax + top-K + renorm, batched over N tokens ──
    //
    // Same Path B split as the decode call site: split the fused
    // softmax+topk+renorm into gpu.softmax_f32 + moe_topk_renorm_k8_batched
    // so prefill activations match the CPU-reference softmax math
    // exactly. router_logits is allocated 1D as [n × n_exp]; alias it
    // into a 2D view so gpu.softmax_f32 takes rows = n.
    let router_logits_2d = GpuTensor {
        buf: unsafe { router_logits.buf.alias() },
        shape: vec![n, n_exp],
        dtype: DType::F32,
    };
    gpu.softmax_f32(&router_logits_2d)?;
    gpu.moe_topk_renorm_k8_batched(
        router_logits,
        topk_indices,
        topk_weights,
        n_exp,
        config.norm_topk_prob,
        n,
    )?;

    // ── 4. Shared-expert SwiGLU + FWHT, batched over N tokens ──
    //
    // fused_silu_mul_rotate_mq_batched expects [batch × k] gate/up with
    // batch on grid.y and writes FWHT(silu(gate) * up) into x_rot. Here
    // batch=N, k=smi; the shared-rot output buffer is [N × smi].
    // F2: AWQ-aware silu_mul+rotate for the batched shared-expert down input.
    // PARO: shared_expert.down has its own Givens rotation tables (paro.*);
    // use the dedicated fused kernel (commit 50198daa). It takes a per-weight
    // (pairs, theta, channel_scales, krot) tuple instead of the MQ4 FWHT
    // convention. Same shape: gate/up [N × smi] → shared_rot [N × smi].
    if paro_mode {
        let paro_down = ffn
            .shared_expert
            .down
            .paro
            .as_ref()
            .expect("ParoQ4G128 shared_expert.down missing paro metadata");
        gpu.fused_silu_mul_givens_rotate_f32(
            shared_gate,
            shared_up,
            shared_rot,
            &paro_down.pairs,
            &paro_down.theta,
            &paro_down.channel_scales,
            n,
            smi,
            paro_down.krot as usize,
        )?;
    } else if matches!(ffn.shared_expert.down.gpu_dtype, DType::Q8_0) {
        // Q8 shared down expects the UN-rotated SwiGLU hidden (no FWHT). Plain
        // element-wise silu_mul over the flat [N × smi] buffers (batched for
        // free) writes the hidden into shared_rot, feeding the Q8 down GEMM.
        gpu.silu_mul_f32(shared_gate, shared_up, shared_rot)?;
    } else {
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &ffn.shared_expert.down,
            shared_gate,
            shared_up,
            shared_rot,
            smi,
            n,
        )?;
    }

    // ── 5. Shared-expert down with sigmoid-scaled residual, batched ──
    //
    // Reads shared_scalar[token] as the pre-sigmoid logit, applies sigmoid
    // internally, and += sigmoid(scalar) × (W_down · rot) into
    // pbs.x_batch[token × dim + row]. (Note: HFQ4 sister uses += not
    // atomicAdd; each (bid, row) writes a unique cell.)
    // Per-projection dispatch: MQ4 → HFQ4 kernel, MQ6 → HFQ6 sister
    // (shipped via feat/hfq6-sigmoid-scaled-batched).
    match ffn.shared_expert.down.gpu_dtype {
        DType::MQ4G256 => gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
            &ffn.shared_expert.down.buf,
            shared_rot,
            &pbs.x_batch,
            shared_scalar,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
            n,
        )?,
        DType::MQ6G256 => gpu.gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
            &ffn.shared_expert.down.buf,
            shared_rot,
            &pbs.x_batch,
            shared_scalar,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
            n,
        )?,
        // Phase 2: HFQ4G128 batched residual+sigmoid-scaled kernel. Single
        // launch, same semantics as the HFQ4G256 sister — reads shared_rot
        // (already silu-mul-rotated by the PARO fused kernel above), GEMVs
        // against W_down, applies sigmoid(shared_scalar[token]) × output,
        // accumulates into pbs.x_batch.
        DType::ParoQ4G128 => gpu.gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
            &ffn.shared_expert.down.buf,
            shared_rot,
            &pbs.x_batch,
            shared_scalar,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
            n,
        )?,
        // Q8 shared down (A3B mfp4-E8, Q8-shared variant): plain batched Q8 GEMM
        // W_down · hidden into a [N × dim] temp, then fold into the residual with
        // the per-token sigmoid(shared_scalar) gate. The temp aliases the first N×dim
        // of `down_expanded` (the routed down-expanded scratch), which is FREE here —
        // the routed experts (step 6) overwrite it only after this completes, and
        // the HIP stream is in-order so the add reads before that. Batched analog
        // of the decode sigmoid_f32 + scaled_add_inplace shared-down arm.
        DType::Q8_0 => {
            let down_tmp = GpuTensor {
                buf: unsafe { down_expanded.buf.alias() },
                shape: vec![n * dim],
                dtype: DType::F32,
            };
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &ffn.shared_expert.down.buf,
                ffn.shared_expert.down.gpu_dtype,
                shared_rot,
                &down_tmp,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?;
            gpu.sigmoid_scaled_residual_add_batched_f32(
                &pbs.x_batch,
                &down_tmp,
                shared_scalar,
                n,
                dim,
            )?;
        }
        // Uniform mfp4-E8 shared expert down (Option B): dequant the E8 down weight
        // to F16, run GemmF16WmmaMb8 against shared_rot (FWHT-rotated SwiGLU hidden)
        // into a [N × dim] temp, then sigmoid-scale-add into the residual. The temp
        // aliases the first N×dim of `down_expanded` (safe: step 6 routed experts
        // run after this, and the stream is in-order). Mirrors the Q8_0 arm above
        // except the GEMM is F16 weight × F32 activation = F32 output.
        DType::MFP4G32E8 => {
            let down_m = ffn.shared_expert.down.m;
            let down_k = ffn.shared_expert.down.k;
            let down_f16 = gpu.alloc_tensor(&[down_m * down_k], DType::F16)?;
            gpu.dequantize_mfp4g32_e8_to_f16(
                &ffn.shared_expert.down.buf.buf,
                &down_f16.buf,
                down_m,
                down_k,
            )?;
            let down_tmp = GpuTensor {
                buf: unsafe { down_expanded.buf.alias() },
                shape: vec![n * dim],
                dtype: DType::F32,
            };
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmF16WmmaMb8,
                &down_f16,
                DType::F16,
                shared_rot,
                &down_tmp,
                down_m,
                down_k,
                n,
            )?;
            gpu.free_tensor(down_f16)?;
            gpu.sigmoid_scaled_residual_add_batched_f32(
                &pbs.x_batch,
                &down_tmp,
                shared_scalar,
                n,
                dim,
            )?;
        }
        other => panic!(
            "prefill_moe_ffn_body_batched: unsupported shared_expert.down dtype {other:?} \
                         — admit predicate should have rejected this layer"
        ),
    }

    // ── 6. Routed experts: delegated to MoeFamily::run_prefill (Ship 4.2) ──
    let down_m = ffn.experts[0].down.m;
    let down_k = ffn.experts[0].down.k;
    let gate_up_k = ffn.experts[0].gate_up.k;
    let total_slots = n * k_top;
    let m_total_max = moe_grouped_m_total_bound(total_slots, n_exp);

    // SP2: per-expert tier tables for intra-layer mixed-tier dispatch (same
    // semantics as the decode builder). Uniform layer ⇒ None ⇒ uniform fast
    // path. This prefill site always has ≥1 expert (indexed [0] above).
    let (per_expert_gate_up, per_expert_down) = per_expert_tier_tables(&ffn.experts);
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        shared_expert_down: ffn.shared_expert.down.gpu_dtype,
        experts_all_gate_up_mq4: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256),
        routed_gate_up: ffn.experts[0].gate_up.gpu_dtype,
        routed_down: ffn.experts[0].down.gpu_dtype,
        // Prefill never fires the merged decode kernel (decode-only), but the
        // shared MoeDtypes struct still requires the flag; carry it honestly.
        routed_has_mixed_experts: ffn.expert_dtype_tags.is_some(),
        has_paro_shared: ffn.paro_shared.is_some(),
        per_expert_gate_up,
        per_expert_down,
    };

    let paro_gate_up =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.gate_up_pairs,
                theta: &paro.gate_up_theta,
                scales: &paro.gate_up_channel_scales,
                krot: paro.krot as usize,
            });
    let paro_down =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.down_pairs,
                theta: &paro.down_theta,
                scales: &paro.down_channel_scales,
                krot: paro.krot as usize,
            });
    // Route A MoE-AWQ: the per-expert indexed table (built at load) supersedes
    // the Ship 4.2 single-scale `down_awq_scale` stub for routed experts — that
    // stub applied experts[0]'s scale to every routed slot, which is wrong once
    // experts actually carry per-expert AWQ. Pass `None` for the single scale;
    // `expert_down_awq_ptrs` drives the correct per-slot path in run_moe_prefill.
    let down_awq_scale: Option<&GpuTensor> = None;

    let moe_prefill_params = hipfire_dispatch::families::moe::MoePrefillParams {
        dtypes: moe_dtypes,
        batch_size: n,
        mi,
        down_m,
        down_k,
        gate_up_k,
        k_top,
        n_exp,
        m_total_max,
        force_mq4_grouped_fp16: model_has_mq6_moe
            && gpu.arch_caps.is_gfx1151()
            && gpu.flags.moe_grouped_i8.is_none(),
        topk_indices,
        topk_weights,
        x_batch: &pbs.x_batch,
        x_norm_batch: &pbs.x_norm_batch,
        x_rot_batch: &pbs.x_rot_batch,
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        expert_down_awq_ptrs: ffn.expert_down_awq_ptrs.as_ref(),
        expert_dtype_tags: ffn.expert_dtype_tags.as_ref(),
        gate_batch,
        up_batch,
        rot_batch,
        down_expanded,
        expert_token_counts: pbs.moe_expert_token_counts.as_ref().expect("moe scratch"),
        expert_offsets: pbs.moe_expert_offsets.as_ref().expect("moe scratch"),
        sorted_slot_index: pbs.moe_sorted_slot_index.as_ref().expect("moe scratch"),
        expert_tile_ids: pbs.moe_expert_tile_ids.as_ref().expect("moe scratch"),
        inverse_perm: pbs.moe_inverse_perm.as_ref().expect("moe scratch"),
        y_gate_up_grouped: pbs.moe_y_gate_up_grouped.as_ref().expect("moe scratch"),
        y_down_grouped: pbs.moe_y_down_grouped.as_ref().expect("moe scratch"),
        paro_gate_up,
        paro_down,
        down_awq_scale,
        routed_out,
    };
    hipfire_runtime::llama::moe_family()
        .run_prefill(ctx, gpu, &moe_prefill_params)
        .map_err(HipError::from)?;

    Ok(())
}

/// Band view for `forward_prefill_chunk`. `None` (the default) means the
/// chunk processes the whole stack: embedding → all layers → final norm
/// + lm_head. `Some(b)` restricts the chunk to layers `b.layer_start..
/// b.layer_end`, skips the embedding when `!b.is_first_band` (input is
/// already in `pbs.x_batch` from a prior peer-copy), and skips the final
/// norm + lm_head when `!b.is_last_band` (output activation stays in
/// `pbs.x_batch` for the next band's peer-copy).
///
/// Counter offsets seed the running per-LA / per-KV / per-FA counters so
/// the band's first DeltaNet/FullAttn layer indexes the correct
/// `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot.
pub(crate) struct PrefillBandCtx<'a> {
    pub layer_start: usize,
    pub layer_end: usize,
    pub delta_layer_offset: usize,
    pub kv_layer_offset: usize,
    pub is_first_band: bool,
    pub is_last_band: bool,
    /// Per-device asym{2,3,4} givens replicas. When `Some`, the chunk's
    /// FA-layer batched KV writers use these instead of `kv_cache.givens_*`
    /// (which is `None` in multi-GPU mode by design — each device needs its
    /// own copy of the rotation tables).
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
}

#[allow(clippy::too_many_arguments)]
/// Debug localization hook (no-op unless `HIPFIRE_DUMP_HIDDEN` is set to a file
/// prefix). Appends the post-layer hidden row for the target absolute position
/// to `{HIPFIRE_DUMP_HIDDEN}.{tag}` as `u32 layer_idx` followed by `dim`
/// little-endian f32. The target absolute position is `HIPFIRE_DUMP_HIDDEN_POS`
/// (default 0); `abs_pos_of_row0` is the absolute sequence position of row 0 of
/// `x` (`start_pos` for the batched residual `pbs.x_batch`, `pos` for the
/// single-row per-token `s.x`). Used to localize the PARO batched-prefill
/// divergence by diffing `.batched` vs `.pertoken` per layer. Requires
/// `HIPFIRE_GRAPH=0` (does a synchronous D2H readback, which is illegal under
/// graph capture).
fn dump_hidden_localize(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rows: usize,
    abs_pos_of_row0: usize,
    dim: usize,
    layer_idx: usize,
    tag: &str,
) {
    let prefix = match hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN") {
        Ok(p) => p,
        Err(_) => return,
    };
    let target: usize = hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if target < abs_pos_of_row0 {
        return;
    }
    let row = target - abs_pos_of_row0;
    if row >= n_rows {
        return;
    }
    if gpu.hip.device_synchronize().is_err() {
        return;
    }
    let all = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let off = row * dim;
    if off + dim > all.len() {
        return;
    }
    use std::io::Write;
    let path = format!("{prefix}.{tag}");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(&(layer_idx as u32).to_le_bytes());
        let mut bytes = Vec::with_capacity(dim * 4);
        for v in &all[off..off + dim] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

#[inline]
fn prefill_dispatch_workload(
    captures_per_token_hidden: bool,
    captures_rollback_tape: bool,
    is_tree_verify: bool,
) -> DispatchWorkload {
    if captures_per_token_hidden || captures_rollback_tape || is_tree_verify {
        DispatchWorkload::SpeculativeVerify
    } else {
        DispatchWorkload::Standard
    }
}

fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
    band: Option<&PrefillBandCtx<'_>>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    needs_last_token_logits: bool,
    max_layer: Option<usize>,
    // EP (Ship 6 substrate-EP prefill): per-MoE-layer routed partial. ONLY set
    // by the EP driver, which calls this with a SINGLE-layer band so the routed
    // combine of that one MoE layer lands in the zeroed partial (all-reduced by
    // the driver after the call). Always `None` for multi-layer bands (PP /
    // single-GPU full stack) — a shared partial across >1 MoE layer would be wrong.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);
    // Target verification is a distinct performance regime from prompt
    // prefill. DFlash, DSpark/MTP, and tree verify expose that semantic here
    // through per-position hidden output, a rollback tape, or a tree mask.
    // Carry it in DispatchCtx so dispatch never needs process-global state.
    let dispatch_workload = prefill_dispatch_workload(
        per_token_hidden_out.is_some(),
        gdn_tape.is_some(),
        tree_verify.is_some(),
    );
    let required_tokens = checked_kv_end(start_pos, n, "forward_prefill_chunk")?;
    kv_cache.require_mapped_capacity(required_tokens)?;
    debug_assert!(
        routed_out.is_none()
            || band
                .map(|b| b.layer_end - b.layer_start <= 1)
                .unwrap_or(false),
        "forward_prefill_chunk: routed_out requires a single-layer band (EP driver invariant)",
    );

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    // Build one DispatchCtx per chunk (decision-only, threaded through
    // MoE prefill family calls). Ship 4.2.
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);

    let do_embed = band.map(|b| b.is_first_band).unwrap_or(true);
    let layer_start = band.map(|b| b.layer_start).unwrap_or(0);
    // `max_layer = Some(N)` early-exits at layer N (exclusive). pflash uses
    // this with N = score_layer_idx + 1: the drafter forward only needs to
    // populate the K cache through the scoring layer (the shallowest
    // FullAttention layer, typically layer 3 of 24 in Qwen3.5 hybrid),
    // since `pflash_score_q8_kv` reads exactly that layer's K. Layers
    // beyond it and the final norm + lm_head are wasted compute for
    // pflash. Saves ~80% of drafter forward time on hybrid drafters.
    let layer_end = band
        .map(|b| b.layer_end)
        .unwrap_or(config.n_layers)
        .min(max_layer.unwrap_or(usize::MAX));
    // Skip final norm + lm_head when the caller early-exits — they produce
    // logits the caller doesn't read, and require running through the full
    // layer stack anyway.
    let do_lm_head = band.map(|b| b.is_last_band).unwrap_or(true) && max_layer.is_none();
    // Per-call-site `givens_cos_view` / `givens_sin_view` macros below
    // resolve to either the band-supplied per-device replica (multi-GPU
    // mode where `kv_cache.givens_*` is `None` by design) or the
    // kv_cache's own table (single-GPU). Held as macros, not top-level
    // bindings, so the immutable borrow on `kv_cache.givens_*` doesn't
    // outlive the kernel-call statement and conflict with later
    // mutable borrows of `kv_cache` (e.g. inside `run_fa_layer_body`).
    macro_rules! givens_cos_view {
        () => {
            band.and_then(|b| b.givens_cos)
                .or(kv_cache.givens_cos.as_ref())
        };
    }
    macro_rules! givens_sin_view {
        () => {
            band.and_then(|b| b.givens_sin)
                .or(kv_cache.givens_sin.as_ref())
        };
    }

    // ── 1. Embed tokens into pbs.x_batch ─────────────────────────────────
    //
    // Fast path for HFQ4G256 (all MQ4-quantized Qwen3.5 models + friends):
    // upload token ids to a device buffer and dispatch one batched kernel
    // that dequantizes N rows directly into `pbs.x_batch`. This collapses
    // 2N launches (N embed + N memcpy_dtod_at) into 1 upload + 1 launch
    // AND is hipGraph-captureable — the kernel reads token ids from a
    // device pointer instead of taking them as a baked-in scalar arg.
    //
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    //
    // Multi-GPU band-mode: skip embedding when this is not the first band.
    // The activation already lives in `pbs.x_batch` from a peer-copy of
    // the previous band's `pbs.x_batch`.
    if do_embed
        && matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
        )
    {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            _ => unreachable!(),
        }
    } else if do_embed {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => unreachable!(),
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                _ => panic!("unsupported embedding format"),
            }
            gpu.hip.memcpy_dtod_at(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // ── 1a. Apply MaskEmbedOverride (MTP probe hook) ─────────────────────
    //
    // Overwrite a single batch slot's embedding row in `pbs.x_batch` after
    // the embedding-lookup kernel populated it but BEFORE the layer loop
    // (or any subsequent kernel) reads it. The Qualcomm MTP probe uses this
    // to replace the embedding-table value at a "mask token" position with
    // a prompt-mean vector. Default callers pass `None` → zero overhead.
    //
    // Multi-GPU band-mode: skip on non-first bands; pbs.x_batch already
    // holds the peer-copied activation from the previous band, so an
    // override applied at band 0 has already propagated through the layer
    // stack on that device — re-applying here would clobber the partial
    // forward state.
    if do_embed {
        if let Some(ovr) = mask_override {
            assert!(
                ovr.slot < n,
                "MaskEmbedOverride.slot ({}) must be < n ({})",
                ovr.slot,
                n,
            );
            assert_eq!(
                ovr.embed.len(),
                dim,
                "MaskEmbedOverride.embed.len() ({}) must equal config.dim ({})",
                ovr.embed.len(),
                dim,
            );
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ovr.embed.as_ptr() as *const u8, dim * 4) };
            let offset = ovr.slot * dim_row_bytes;
            gpu.hip
                .memcpy_htod_offset(&pbs.x_batch.buf, offset, bytes)?;
        }
    }

    // ── 1b. Upload positions array ────────────────────────────────────────
    //
    // Positions is the per-row RoPE angle AND the physical KV cache slot (the
    // batched kv_write kernels use the same index for both). We always use
    // flat linear `start_pos .. start_pos + n`. Siblings in DDTree mode get
    // DISTINCT slots this way — no write race — and the stored K carries a
    // RoPE angle that matches the physical slot, which keeps subsequent
    // cycles' attention reads consistent.
    //
    // Semantic trade vs. the original depth-based scheme (paper): tree
    // siblings that represent "alternative futures at the same time step"
    // now see a RoPE distance of 1 (or more) instead of 0. Empirically that
    // slight distance shift costs little — the attn_bias mask still gates
    // ancestor visibility exactly, and the Q·K dot products stay consistent
    // across the whole cache (prompt + tree block). In exchange we get
    // DDTree correctness for topk>1 without needing a tree-local KV scratch
    // or a scatter-kernel for commit. `ctx.positions` is accepted for API
    // compatibility but ignored — the DdNode depths it carries are only
    // used by `linearize_tree` to build the attn_bias mask.
    //
    // 39aa358 DECOUPLING (2026-07-28): the "costs little" trade was wrong
    // — it was the gfx1100 DDTree regression. Sibling logits computed with
    // slot-based phases depress non-rank-0 acceptance, collapsing τ toward
    // the chain and making tree strictly slower than linear. Fix: upload
    // `ctx.positions` (depth-based, `base_pos + depth`) to
    // `pbs.rope_positions`; FA RoPE reads THAT (correct sibling phases),
    // while KV writes + attention seq_len keep the flat physical slots
    // (no sibling write race, contiguous-cache invariants intact).
    if !pre_uploaded {
        let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
        if let Some(tv) = tree_verify.as_ref() {
            debug_assert_eq!(tv.positions.len(), n, "tree RoPE positions length");
            let rope_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tv.positions.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.rope_positions.buf, rope_bytes)?;
        }
    }

    // Decide whether the FA layers can take the batched path. Requires
    // (a) all FA weights to be MQ4G256 or HFQ4G256 (the batched gemm_qkv
    // + wo GEMMs are dtype-agnostic; the rmsnorm+rotate / silu_mul kernels
    // differ by dtype and we branch on that at each layer) and (b) a Q8_0
    // or givens KV cache. If the check fails, FA layers fall back to
    // per-token gather/scatter via run_fa_layer_body.
    let fa_arch = gpu.arch.as_str();
    // Q8 WMMA gate: the fused Q8 WMMA family (gemm_qkv/qkvza/gate_up/residual
    // _q8_0_wmma) uses the gfx11 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
    // builtin; the sibling `*.gfx12.hip` kernels use the `_w32_gfx12` variant
    // (silicon-validated on R9700, 2026-05-14, 4/4 unit tests PASS). Each
    // call site below selects the right variant via an `arch.starts_with`
    // branch. On non-WMMA archs we keep the Tier 2 chunked-substrate path.
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    // MQ3 dispatch arch gate (same predicate, separate name for clarity at
    // each matcher). Phase 1 gfx10 MQ3 prefill (`docs/plans/gfx10_mq3_prefill.md`)
    // routes the 8 `is_mq3*` matchers below to scalar HFQ3 kernels on
    // !arch_has_wmma archs admitted by `is_batchable_la`.
    let arch_has_wmma = q8_wmma_arch;
    let fa_batched_ok =
        (kv_cache.quant_q8 || kv_cache.quant_asym4 || kv_cache.quant_asym3 || kv_cache.quant_asym2)
            && weights.layers.iter().all(|lw| match lw {
                LayerWeights::FullAttn(l) => {
                    is_batchable_la(l.wq.gpu_dtype, fa_arch)
                        && is_batchable_la(l.wk.gpu_dtype, fa_arch)
                        && is_batchable_la(l.wv.gpu_dtype, fa_arch)
                        && is_batchable_la(l.wo.gpu_dtype, fa_arch)
                        && is_batchable_la(l.w_gate.gpu_dtype, fa_arch)
                        && is_batchable_la(l.w_up.gpu_dtype, fa_arch)
                        && is_batchable_la(l.w_down.gpu_dtype, fa_arch)
                }
                // MoE variant: attention weights must be MQ4-class (FFN is
                // checked separately by moe_ffn_batched_admissible in the eligibility gate).
                LayerWeights::FullAttnMoe(l) => {
                    is_batchable_la(l.wq.gpu_dtype, fa_arch)
                        && is_batchable_la(l.wk.gpu_dtype, fa_arch)
                        && is_batchable_la(l.wv.gpu_dtype, fa_arch)
                        && is_batchable_la(l.wo.gpu_dtype, fa_arch)
                }
                _ => true, // LA layers don't gate this check
            });
    // Under hipGraph capture, scalar kernargs get BAKED into the kernarg blob
    // at capture time. `max_ctx_len = start_pos + n` grows per cycle, so the
    // captured value would be stale on replay — the attention kernel would
    // allocate too-small LDS for `scores[]` and over-read. Bake the physical
    // cap instead (LDS sized for the worst case). The kernel still iterates
    // over the actual `positions[b] + 1` per-row seq_len from a device buffer,
    // so correctness is preserved; only the LDS allocation is over-provisioned.
    let max_ctx_len = if gpu.graphs.capture_mode {
        kv_cache.physical_cap
    } else {
        start_pos + n
    };

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    // Multi-GPU band-mode: counters seed from the band's running offsets so
    // the band's first DeltaNet/FullAttn layer reads the correct
    // `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot. Single-GPU
    // (band==None) seeds zeros — original behavior.
    let mut delta_layer_idx = band.map(|b| b.delta_layer_offset).unwrap_or(0);
    let mut kv_layer_idx = band.map(|b| b.kv_layer_offset).unwrap_or(0);
    let ctx = DispatchCtx::new(gpu).with_workload(dispatch_workload);

    for layer_idx in layer_start..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // Per-layer dtype branch: MQ4 needs FWHT-rotation on the
                // activation to match its pre-rotated weights; HFQ4 uses
                // plain rmsnormed activations. The GEMM kernels themselves
                // are dtype-agnostic — they just consume whatever [N × K]
                // activation buffer we point them at.
                // GAP NOTE: this matcher (and the 7 sibling dense LA/FA
                // matchers in this file) wires MQ3G256Lloyd through the
                // gemm_*_mq3g256_lloyd_wmma family. MQ2G256Lloyd remains
                // unwired — to add it, update is_batchable_la, ALL 8 is_mq*
                // matchers, AND add a Lloyd-MQ2-specific GEMM dispatch arm
                // together (the all-together corruption-prevention rule from
                // docs/plans/mq-lloyd-batched-prefill-followup.md). MQ4-Lloyd
                // is wired in a separate PR (issue #182).
                let is_mq = matches!(
                    layer.wqkv.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_mq3 = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256);
                let is_mq3_lloyd = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256Lloyd);
                let is_fp4 = matches!(layer.wqkv.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);

                // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
                // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
                // we reuse x_rot_batch as the "normed, unrotated" output
                // so the subsequent GEMM can read it the same way.
                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }

                // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
                if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 && q8_wmma_arch {
                    // `is_q8` only inspects `wqkv` (the routing anchor). The fused
                    // kernel assumes ALL four weights share the Q8_0 stride; a
                    // mixed-dtype layer would silently re-introduce the Tier-1
                    // kernel-vs-stride corruption mode.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                            && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                            && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha) → GemmFamily::run_key with the
                    // GemmQ8_0BatchedChunked dispatcher-entry key → identical
                    // gpu.gemm_q8_0_batched_chunked method, byte-for-byte.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else if is_mq3_lloyd {
                    // 112 B/group Lloyd-MQ3 stride; X is already FWHT-rotated.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaMq3G256Lloyd,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_mq3 {
                    // 104 B/group HFQ3-stride; X is already FWHT-rotated by
                    // fused_rmsnorm_rotate_mq_batched above. The FusedQkvzaHfq3G256
                    // run-arm replicates the call-site WMMA-vs-base arch split
                    // internally (gemm_qkvza_hfq3g256_wmma on has_wmma() else the
                    // base cross-arch ladder), so the same kernel runs.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq3G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_fp4 {
                    // HFP4G32: 17-B blocks (vs HFQ4's 136-B groups), per-row 16-B header.
                    // MFP4G32: same storage as HFP4 + offline-FWHT weights; X is already
                    // rotated above when is_mq, so this branch handles both unrotated
                    // (HFP4) and post-rotation (MFP4) activations identically.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfp4G32,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                }

                // Fused sigmoid(beta) + alpha_gate(alpha) — [N × n_v_heads] each.
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;

                // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
                // for this layer into the per-layer tape slots. The next LA
                // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
                // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
                // now (after sigmoid_alpha_gate, before conv1d consumes qkv).
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }

                // Tree-aware dispatch gate: when the caller provides
                // parent_indices (Phase 3b+ of Task #101), swap the linear
                // conv1d + GDN for tree-walking variants that eliminate
                // sibling-subtree state cross-contamination. The tree
                // kernels are READ-ONLY on dn_state (don't advance it) —
                // caller runs linear replay on the accepted spine
                // post-acceptance to commit the trajectory.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }

                // Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave
                // when n_key_heads < n_v_heads. One launch instead of two —
                // ~200µs saved per LA layer × ~30 LA layers ≈ 6ms per prefill
                // on A3B (R9700/gfx1201).
                //
                // The fused kernel reads q_raw/k_raw (unchanged on exit), so
                // the conv1d output is preserved if downstream readers need it
                // (no current consumer reads _raw after this).
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    // n_key_heads == n_v_heads → no replication; keep the
                    // original sequence (norm in place, then memcpy).
                    gpu.fused_qk_l2_norm_scale_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                        n,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }

                // Gated Delta Net — tree variant reads per-token S from
                // s_tape[parent] (or pre-block s_q8_init at root); linear
                // variant advances dn_state.s_matrices in place.
                if let Some(parents) = tree_parents {
                    // Tree-verify GDN, dispatched by DeltaNet state quant.
                    // FP32 uses the full-precision tree-tape kernel (no
                    // per-node Q8 round-trip); Q8 the original; Q4 tree has
                    // no kernel (was silently mis-routed to the Q8 tree
                    // kernel before — now a clean error).
                    match dn_state.quant {
                        StateQuant::FP32 => {
                            let tape_f32 = pbs.dn_s_tape_f32.as_ref().expect(
                                "FP32 tree-aware LA requires dn_s_tape_f32 scratch (check PrefillBatchScratch::new)",
                            );
                            gpu.gated_delta_net_f32_tree_batch_seq(
                                &pbs.dn_q_batch,
                                &pbs.dn_k_batch,
                                &pbs.dn_v_batch,
                                &pbs.dn_alpha_batch,
                                &pbs.dn_beta_batch,
                                &dn_state.s_matrices[delta_layer_idx],
                                tape_f32,
                                parents,
                                &pbs.dn_attn_out_batch,
                                n,
                                n_v_heads,
                                config.linear_value_head_dim,
                            )?;
                        }
                        StateQuant::Q8 => {
                            let tape_q8 = pbs.dn_s_tape_q8.as_ref()
                                .expect("tree-aware LA requires dn_s_tape_q8 scratch (check PrefillBatchScratch::new)");
                            let tape_sc = pbs.dn_s_tape_scales.as_ref()
                                .expect("tree-aware LA requires dn_s_tape_scales scratch (check PrefillBatchScratch::new)");
                            gpu.gated_delta_net_q8_tree_batch_seq(
                                &pbs.dn_q_batch,
                                &pbs.dn_k_batch,
                                &pbs.dn_v_batch,
                                &pbs.dn_alpha_batch,
                                &pbs.dn_beta_batch,
                                &dn_state.s_matrices[delta_layer_idx],
                                &dn_state.s_scales[delta_layer_idx],
                                tape_q8,
                                tape_sc,
                                parents,
                                &pbs.dn_attn_out_batch,
                                n,
                                n_v_heads,
                                config.linear_value_head_dim,
                            )?;
                        }
                        StateQuant::Q4 => {
                            return Err(hip_bridge::HipError::new(
                                0,
                                "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: \
                                 there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state \
                                 for tree spec-decode.",
                            ));
                        }
                    }
                } else {
                    // EXPERIMENT (not #417): mirror the state-quant dispatch the
                    // decode siblings already do (forward_scratch_layers:13194),
                    // so the captured/eager batched prefill honours FP32/Q4 state
                    // instead of forcing the Q8 kernel onto non-Q8 buffers.
                    match dn_state.quant {
                        StateQuant::FP32 => {
                            if rdna_compute::norm::gdn_chunked() && n > 1 {
                                gpu.gated_delta_net_f32_chunked(
                                    &pbs.dn_q_batch,
                                    &pbs.dn_k_batch,
                                    &pbs.dn_v_batch,
                                    &pbs.dn_alpha_batch,
                                    &pbs.dn_beta_batch,
                                    &dn_state.s_matrices[delta_layer_idx],
                                    &pbs.dn_attn_out_batch,
                                    n,
                                    n_v_heads,
                                    config.linear_value_head_dim,
                                    rdna_compute::norm::gdn_chunk_size(),
                                )?
                            } else {
                                gpu.gated_delta_net_f32_batch_seq(
                                    &pbs.dn_q_batch,
                                    &pbs.dn_k_batch,
                                    &pbs.dn_v_batch,
                                    &pbs.dn_alpha_batch,
                                    &pbs.dn_beta_batch,
                                    &dn_state.s_matrices[delta_layer_idx],
                                    &pbs.dn_attn_out_batch,
                                    n,
                                    n_v_heads,
                                    config.linear_value_head_dim,
                                )?
                            }
                        }
                        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                }

                // Batched gated output norm.
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    n,
                )?;

                // Batched wo + residual.
                //
                // For MQ weights, the decode path's weight_gemv_residual
                // internally FWHT-rotates dn_normed into mq_x_rot before
                // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
                // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
                // For HFQ weights no rotation is needed — the activation
                // feeds gemm_hfq{4,6}g256_residual directly.
                let wo_is_mq = matches!(
                    layer.wo.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
                let wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
                let wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let wo_input = if wo_is_mq {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    &pbs.dn_normed_batch
                };
                if wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_q8 {
                    // Tier 2 fallback (non-WMMA archs): GEMM into x_rot_batch as
                    // scratch (safe — next consumer is the FFN rmsnorm), then
                    // add into residual.
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if wo_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_mq3 {
                    if arch_has_wmma {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.wo.buf,
                            layer.wo.gpu_dtype,
                            wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    } else {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.wo.buf,
                            layer.wo.gpu_dtype,
                            wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    }
                } else if wo_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // FFN: rmsnorm (+ rotate for MQ).
                let ffn_is_mq = matches!(
                    layer.w_gate.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let ffn_is_6bit =
                    matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
                let ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
                let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
                if ffn_is_mq {
                    // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }

                // Batched gate+up projection.
                // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
                // (batched-prefill gate+up variant) via run_fused_gate_up_key.
                // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
                // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
                // split is folded into the FusedGateUpHfq3G256 run-arm, which
                // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
                if ffn_is_6bit {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_q8 && q8_wmma_arch {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                        "LA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
                    );
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_q8 {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_gate.buf,
                        layer.w_gate.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_up.buf,
                        layer.w_up.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if ffn_is_mq3_lloyd {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_mq3 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_fp4 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                }

                // SwiGLU activation feeding w_down. For MQ, we need the
                // output FWHT-rotated so it matches the pre-rotated w_down
                // weights. For HFQ, plain silu_mul is enough. silu_mul_f32
                // is purely element-wise and uses numel() as its length,
                // so a [N × hidden_dim] tensor processes all rows in one
                // launch with no batch offset needed.
                let w_down_is_mq = matches!(
                    layer.w_down.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let w_down_is_6bit =
                    matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
                let w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
                let w_down_is_fp4 =
                    matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
                if w_down_is_mq {
                    // F2: AWQ-aware silu_mul+rotate for w_down input.
                    fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        &layer.w_down,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        &pbs.ffn_hidden_batch,
                        hidden_dim,
                        n,
                    )?;
                } else {
                    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                }

                // Batched w_down + residual.
                if w_down_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &x_n,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &scratch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if w_down_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_mq3 {
                    if arch_has_wmma {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.w_down.buf,
                            layer.w_down.gpu_dtype,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    } else {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.w_down.buf,
                            layer.w_down.gpu_dtype,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    }
                } else if w_down_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = is_mq; // retained above for potential future use
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Fully batched FA layer. Mirrors the FA branch of
                // forward_scratch_layers kernel-for-kernel, but every
                // launch covers all N tokens at once.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                let qkv_is_mq = matches!(
                    layer.wq.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
                let qkv_is_mq3_lloyd = matches!(layer.wq.gpu_dtype, DType::MQ3G256Lloyd);
                let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                // Fused QKV kernels require all three weights to share a
                // dtype — they treat wq/wk/wv as same-stride byte arrays.
                // When kmap mode 2 promotes only `v_proj` (issue #249), the
                // fused HFQ4 path reads `wv` as MQ6 with HFQ4's 136-B stride
                // and produces silent NaN. Gate the fused kernels here.
                //
                // The Q8 substrate path (gemm_q8_0_batched_chunked × 3) also
                // dispatches a Q8-stride kernel per weight, so it needs the
                // same gate when wk/wv aren't Q8.
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

                // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }

                // 2. Batched 3-way QKV projection (wq+wk+wv).
                if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_mq3_lloyd && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvMq3G256Lloyd,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_mq3 && qkv_same_dtype {
                    // X is already FWHT-rotated by fused_rmsnorm_rotate_mq_batched
                    // above; call the bare HFQ3 GEMM (no second rotation). The
                    // FusedQkvHfq3G256 run-arm replicates the call-site WMMA-vs-base
                    // arch split internally (gemm_qkv_hfq3g256_wmma on has_wmma()
                    // else the base cross-arch ladder), so the same kernel runs.
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq3G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_fp4 && qkv_same_dtype {
                    // HFP4G32 / MFP4G32 FP4 batched WMMA. X is already
                    // rotated above for MFP4 (is_mq path) — same kernel
                    // covers both unrotated HFP4 and rotated MFP4 inputs.
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfp4G32,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FA qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq4G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else {
                    // Mixed-format fallback (issue #249): wq/wk/wv don't all
                    // share a dtype. Dispatch each weight to its own
                    // single-weight batched GEMM, dropping the fused-kernel
                    // launch-overhead optimization for correctness.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }

                // 3. Batched deinterleave Q + gate: one kernel launch for all N tokens.
                gpu.deinterleave_f32_batched(
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    config.n_heads,
                    config.head_dim,
                    n,
                )?;

                // 4. Per-head Q/K rmsnorm. rmsnorm_batched uses batch =
                // number of "rows" of head_dim. For [N × n_heads × head_dim]
                // that's batch = N * n_heads.
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    // Try GPU path first: dispatches a reduce kernel on the
                    // device-resident Q tensor, zero PCIe transfer. Only
                    // succeeds when install_tap_gpu() was used. Falls through
                    // to CPU path otherwise.
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }

                // 5. Batched partial-interleaved RoPE (per-row positions).
                // pos_offset = compact_offset so new Q/K rotate at ABSOLUTE phase
                // after eviction (cached keys are absolute-phased); pbs.positions
                // stays physical for the KV-write below. 0 when no compaction.
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // 39aa358: in DDTree verify, rotate at DEPTH positions (correct
                // sibling phases); KV writes below still use flat physical
                // slots. Linear path unchanged.
                let rope_pos_buf = if tree_verify.is_some() {
                    &pbs.rope_positions
                } else {
                    &pbs.positions
                };
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    rope_pos_buf,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    n,
                    kv_cache.compact_offset as i32,
                )?;

                // 6–7. Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let plan = KvTierPlan::derive(KvTierInputs {
                    pos: start_pos,
                    flash_mode: s.flash_mode as usize,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: n,
                    is_tree,
                    ..kv_cache.tier_inputs()
                })
                .map_err(|e| HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &pbs.fa_q_batch,
                    k: &pbs.fa_k_batch,
                    v: &pbs.fa_v_batch,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &s.pos_buf,
                    pos: start_pos,
                    positions: Some(&pbs.positions),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: n,
                    max_ctx_len,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias,
                    block_start,
                    block_cols,
                    output_gate: None,
                    output: &pbs.fa_attn_out_batch,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| HipError::new(0, &e.to_string()))?;

                // 8. Fused sigmoid(gate) * attn_out, element-wise over the
                // full [N × q_dim] tensor.
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;

                // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
                // Same MQ rotation requirement as the LA wo path.
                let fa_wo_is_mq = matches!(
                    layer.wo.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
                let fa_wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_input = if fa_wo_is_mq {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    &pbs.fa_attn_out_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_mq3 {
                    if arch_has_wmma {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.wo.buf,
                            layer.wo.gpu_dtype,
                            fa_wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    } else {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.wo.buf,
                            layer.wo.gpu_dtype,
                            fa_wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    }
                } else if fa_wo_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // 10. FFN: rmsnorm (+ rotate for MQ), gate+up, silu_mul
                // (+ rotate for MQ), w_down residual.
                let fa_ffn_is_mq = matches!(
                    layer.w_gate.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let fa_ffn_is_6bit =
                    matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
                let fa_ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_ffn_is_fp4 =
                    matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
                if fa_ffn_is_mq {
                    // AWQ-aware: next linear is w_gate (FA-FFN, gate/up share input).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                // #397 Ship 5.2 slice 2: FA-FFN fused gate+up → FusedQkvFamily
                // (batched-prefill gate+up variant), mirroring the LA-FFN block
                // above. Q8-non-WMMA stays as two plain GEMMs; HFQ3 WMMA-vs-base
                // is folded into the FusedGateUpHfq3G256 run-arm.
                if fa_ffn_is_6bit {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_q8 && q8_wmma_arch {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                        "FA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
                    );
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_q8 {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_gate.buf,
                        layer.w_gate.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_up.buf,
                        layer.w_up.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if fa_ffn_is_mq3_lloyd {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_mq3 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_fp4 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                }
                let fa_w_down_is_mq = matches!(
                    layer.w_down.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                );
                let fa_w_down_is_6bit =
                    matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
                let fa_w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_w_down_is_fp4 =
                    matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
                if fa_w_down_is_mq {
                    // F2: AWQ-aware silu_mul+rotate for FullAttention w_down input.
                    fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        &layer.w_down,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        &pbs.ffn_hidden_batch,
                        hidden_dim,
                        n,
                    )?;
                } else {
                    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                }
                if fa_w_down_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &x_n,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &scratch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_w_down_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_mq3 {
                    if arch_has_wmma {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.w_down.buf,
                            layer.w_down.gpu_dtype,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    } else {
                        run_residual_gemm_key(
                            gpu,
                            hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                            &layer.w_down.buf,
                            layer.w_down.gpu_dtype,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    }
                } else if fa_w_down_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                // Silence unused warning if kv_dim ends up shadowed.
                let _ = kv_dim;
                kv_layer_idx += 1;
            }

            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                // Per-token gather/scatter fallback for FA layers that don't
                // qualify for batched FA (non-MQ4 weights, non-Q8_0 KV, etc).
                for i in 0..n {
                    let pos = start_pos + i;
                    gpu.hip.memcpy_dtod_at(
                        &s.x.buf,
                        0,
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        dim_row_bytes,
                    )?;
                    let pos_i32 = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &pos_i32.to_ne_bytes())?;
                    run_fa_layer_body(
                        gpu,
                        weights,
                        config,
                        layer_idx,
                        kv_layer_idx,
                        pos,
                        kv_cache,
                        s,
                    )?;
                    gpu.hip.memcpy_dtod_at(
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        &s.x.buf,
                        0,
                        dim_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path. After
                // the per-token loop, pbs.x_batch has the full layer output
                // for all N tokens (last copy-back finishes each row).
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                kv_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // Batched MoE LA layer. LA body is the same as DeltaNet
                // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
                // repeat_interleave + GDN + gated_norm + wo+residual);
                // only the FFN differs. Duplicated inline for now — can
                // be factored into a `prefill_la_body_batched` helper
                // when dense and MoE LA paths are proven byte-exact.
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wqkv.gpu_dtype` is restricted here to MQ4G256 /
                // HFQ4G256 / MQ6G256 / HFQ6G256 / Q8_0. Q8 admit landed
                // alongside the moe_ffn router/gate Q8 unlock (A3B's LA
                // attention weights are Q8 — engine quantizer keeps q/k/v/o
                // at Q8 alongside the Q8 router + shared_expert_gate).
                let is_mq = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
                // Phase 1.5: PARO mode for DeltaNetMoe — wqkv/wz are
                // ParoQ4G128 (each with its own Givens rotation tables);
                // w_alpha/w_beta are F32 (no rotation, no quantization).
                // Dispatch is unfused: rotate+gemm_hfq4g128 for wqkv and wz,
                // direct gemm_f32_batched for w_alpha and w_beta. Same shape
                // outputs as the Q8/MQ4 paths (dn_qkv_batch, dn_z_batch,
                // dn_alpha_batch, dn_beta_batch).
                let is_paro = matches!(layer.wqkv.gpu_dtype, DType::ParoQ4G128);
                let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);

                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if is_paro {
                    // PARO: need un-rotated x_norm available for per-weight
                    // Givens rotation. Write rmsnorm into x_norm_batch (the
                    // dedicated normalized buffer); x_rot_batch becomes the
                    // per-weight rotation scratch (overwritten per GEMM).
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if is_paro {
                    // PARO 4-way unfused dispatch. wqkv and wz are
                    // ParoQ4G128 with their own Givens rotation tables;
                    // w_alpha and w_beta are F32 with no rotation.
                    let paro_wqkv = layer.wqkv.paro.as_ref().unwrap_or_else(|| {
                        panic!(
                            "ParoQ4G128 wqkv missing paro metadata at LA layer {layer_idx} \
                             — load_paroquant_weight() loader regression?"
                        )
                    });
                    let paro_wz = layer.wz.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wz missing paro metadata at LA layer {layer_idx}")
                    });
                    // wqkv: rotate x_norm → x_rot, then HFQ4G128 GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wqkv.pairs,
                        &paro_wqkv.theta,
                        &paro_wqkv.channel_scales,
                        n,
                        dim,
                        paro_wqkv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    // wz: re-rotate x_norm → x_rot (overwrite), then GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wz.pairs,
                        &paro_wz.theta,
                        &paro_wz.channel_scales,
                        n,
                        dim,
                        paro_wz.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    // w_alpha / w_beta: F32, no rotation, direct batched GEMM.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                } else if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 && q8_wmma_arch {
                    // Fused Q8 QKVZA WMMA — assumes all 4 weights share Q8_0
                    // stride; mixed Q8/other layers within DNMoe are rejected
                    // upstream by `moe_ffn_batched_admissible` (router/gate Q8 OK, but
                    // shared_expert + experts must be MQ4) and would otherwise
                    // re-introduce Tier-1 stride corruption.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                            && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                            && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "DNMoe LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha), sibling DeltaNet QKVZA path.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                // Same tree-aware dispatch gate as dense LA branch above.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    n,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        n,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }
                // DIAG: dump GDN inputs (batched, MoE branch)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * hd;
                    dump_hidden_localize(gpu, &pbs.dn_q_batch, n, start_pos, qk_dim, 0, "q_b");
                    dump_hidden_localize(gpu, &pbs.dn_k_batch, n, start_pos, qk_dim, 0, "k_b");
                    dump_hidden_localize(gpu, &pbs.dn_v_batch, n, start_pos, v_dim, 0, "v_b");
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_alpha_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "alpha_b",
                    );
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_beta_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "beta_b",
                    );
                }
                if let Some(parents) = tree_parents {
                    // MoE-path tree-verify GDN, dispatched by state quant
                    // (mirror of the dense path above).
                    match dn_state.quant {
                        StateQuant::FP32 => {
                            let tape_f32 = pbs.dn_s_tape_f32.as_ref().expect(
                                "FP32 tree-aware LA requires dn_s_tape_f32 scratch (check PrefillBatchScratch::new)",
                            );
                            gpu.gated_delta_net_f32_tree_batch_seq(
                                &pbs.dn_q_batch,
                                &pbs.dn_k_batch,
                                &pbs.dn_v_batch,
                                &pbs.dn_alpha_batch,
                                &pbs.dn_beta_batch,
                                &dn_state.s_matrices[delta_layer_idx],
                                tape_f32,
                                parents,
                                &pbs.dn_attn_out_batch,
                                n,
                                n_v_heads,
                                config.linear_value_head_dim,
                            )?;
                        }
                        StateQuant::Q8 => {
                            let tape_q8 = pbs
                                .dn_s_tape_q8
                                .as_ref()
                                .expect("tree-aware LA requires dn_s_tape_q8 scratch");
                            let tape_sc = pbs
                                .dn_s_tape_scales
                                .as_ref()
                                .expect("tree-aware LA requires dn_s_tape_scales scratch");
                            gpu.gated_delta_net_q8_tree_batch_seq(
                                &pbs.dn_q_batch,
                                &pbs.dn_k_batch,
                                &pbs.dn_v_batch,
                                &pbs.dn_alpha_batch,
                                &pbs.dn_beta_batch,
                                &dn_state.s_matrices[delta_layer_idx],
                                &dn_state.s_scales[delta_layer_idx],
                                tape_q8,
                                tape_sc,
                                parents,
                                &pbs.dn_attn_out_batch,
                                n,
                                n_v_heads,
                                config.linear_value_head_dim,
                            )?;
                        }
                        StateQuant::Q4 => {
                            return Err(hip_bridge::HipError::new(
                                0,
                                "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: \
                                 there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state \
                                 for tree spec-decode.",
                            ));
                        }
                    }
                } else {
                    match dn_state.quant {
                        StateQuant::FP32 => {
                            if rdna_compute::norm::gdn_chunked() && n > 1 {
                                gpu.gated_delta_net_f32_chunked(
                                    &pbs.dn_q_batch,
                                    &pbs.dn_k_batch,
                                    &pbs.dn_v_batch,
                                    &pbs.dn_alpha_batch,
                                    &pbs.dn_beta_batch,
                                    &dn_state.s_matrices[delta_layer_idx],
                                    &pbs.dn_attn_out_batch,
                                    n,
                                    n_v_heads,
                                    config.linear_value_head_dim,
                                    rdna_compute::norm::gdn_chunk_size(),
                                )?
                            } else {
                                gpu.gated_delta_net_f32_batch_seq(
                                    &pbs.dn_q_batch,
                                    &pbs.dn_k_batch,
                                    &pbs.dn_v_batch,
                                    &pbs.dn_alpha_batch,
                                    &pbs.dn_beta_batch,
                                    &dn_state.s_matrices[delta_layer_idx],
                                    &pbs.dn_attn_out_batch,
                                    n,
                                    n_v_heads,
                                    config.linear_value_head_dim,
                                )?
                            }
                        }
                        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    // DIAG: dump GDN attention output at layer 0
                    if layer_idx == 0 {
                        dump_hidden_localize(
                            gpu,
                            &pbs.dn_attn_out_batch,
                            n,
                            start_pos,
                            n_v_heads * config.linear_value_head_dim,
                            0,
                            "gdn_b",
                        );
                    }
                }
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    n,
                )?;
                // wo + residual. Q8 wo lands un-rotated (Q8 weights were
                // quantized against un-rotated activations); MQ4/MQ6 wo
                // require FWHT(awq_scale-adjusted) rotation. Mirrors the
                // dense LA wo dispatch (qwen35.rs:5000-5043) — the MQ6
                // branch is required for AWQ A3B where 4/40 LA layers
                // ship MQ6 wo and would otherwise corrupt the residual
                // stream when dispatched through the HFQ4 kernel against
                // 200 B/group MQ6-layout bytes.
                let dn_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let dn_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let dn_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let dn_wo_input = if dn_wo_is_q8 {
                    &pbs.dn_normed_batch
                } else if dn_wo_is_paro {
                    // PARO wo: rotate dn_normed by wo's own Givens tables
                    // into dn_normed_rot_batch. Same scratch layout as MQ4
                    // (since dn_normed_rot_batch is unused on the Q8 path).
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at LA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                };
                if dn_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 {
                    // Non-WMMA Q8: gemm into a scratch then add into x_batch.
                    // Reuse `dn_normed_rot_batch` (free since the MQ4 rotate
                    // path didn't run here) as the GEMM scratch.
                    let scratch = pbs.dn_normed_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if dn_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free at
                    // this point — used earlier for the QKVZA stage; not
                    // needed for the rest of this layer) as the scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
                // silu_mul + w_down) block. Takes pbs.x_batch as input AND
                // accumulates the FFN output residual back into it via the
                // batched indexed down kernel's atomicAdd path.
                prefill_moe_ffn_body_batched(
                    gpu,
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    &ctx,
                    weights.moe_has_mq6,
                    routed_out,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Batched MoE FA layer. FA body is the same as FullAttn
                // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
                // kv_write + attention + sigmoid_mul + wo+residual);
                // only the FFN differs. Duplicated inline — will be
                // consolidated with the dense FA batched body once the
                // MoE path is proven byte-exact.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wq.gpu_dtype` is restricted to MQ4G256 / HFQ4G256
                // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
                // the QKV dispatch is insufficient — the wo path below
                // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
                // wiring lives in a separate PR (see followup issue).
                let qkv_is_mq = matches!(layer.wq.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
                // (each with its own Givens rotation tables). The fused-QKV
                // kernels can't handle this — they assume one shared
                // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
                // per projection) matches the LA QKVZA Phase 1.5 pattern.
                let qkv_is_paro = matches!(layer.wq.gpu_dtype, DType::ParoQ4G128);
                // Fused QKV requires uniform dtype — see issue #249 for
                // the dense FA variant. Gate the same way here.
                let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if qkv_is_paro {
                    // PARO: rmsnorm into x_norm_batch (un-rotated). x_rot_batch
                    // is reused as the per-weight rotation scratch.
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if qkv_is_paro {
                    // PARO 3-way unfused dispatch (wq, wk, wv each with own
                    // Givens rotation). Same shape outputs as the fused
                    // paths: fa_q_full_batch, fa_k_batch, fa_v_batch.
                    let paro_wq = layer.wq.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wk = layer.wk.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wv = layer.wv.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
                    });
                    // wq
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wq.pairs,
                        &paro_wq.theta,
                        &paro_wq.channel_scales,
                        n,
                        dim,
                        paro_wq.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    // wk
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wk.pairs,
                        &paro_wk.theta,
                        &paro_wk.channel_scales,
                        n,
                        dim,
                        paro_wk.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    // wv
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wv.pairs,
                        &paro_wv.theta,
                        &paro_wv.channel_scales,
                        n,
                        dim,
                        paro_wv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq4G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else {
                    // Mixed-format fallback (issue #249). batched_gemm_single_weight
                    // covers MQ4/HFQ4 + MQ6/HFQ6 + Q8_0; mixed-Q8/MQ4 within FAMoe
                    // routes here.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }
                gpu.deinterleave_f32_batched(
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    config.n_heads,
                    config.head_dim,
                    n,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                if hipfire_runtime::triattn::tap_enabled() {
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // pos_offset = compact_offset (absolute RoPE phase post-eviction);
                // pbs.positions stays physical for the KV-write. 0 when no compaction.
                // 39aa358: in DDTree verify, rotate at DEPTH positions instead
                // (correct sibling phases); KV write below stays physical.
                let rope_pos_buf = if tree_verify.is_some() {
                    &pbs.rope_positions
                } else {
                    &pbs.positions
                };
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    rope_pos_buf,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    n,
                    kv_cache.compact_offset as i32,
                )?;
                // Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let plan = KvTierPlan::derive(KvTierInputs {
                    pos: start_pos,
                    flash_mode: s.flash_mode as usize,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: n,
                    is_tree,
                    ..kv_cache.tier_inputs()
                })
                .map_err(|e| HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &pbs.fa_q_batch,
                    k: &pbs.fa_k_batch,
                    v: &pbs.fa_v_batch,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &s.pos_buf,
                    pos: start_pos,
                    positions: Some(&pbs.positions),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: n,
                    max_ctx_len,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias,
                    block_start,
                    block_cols,
                    output_gate: None,
                    output: &pbs.fa_attn_out_batch,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                // wo + residual. Mirrors the dense FA wo dispatch at
                // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
                // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
                // MQ6 branch added alongside MQ6_ADMIT (without it, MQ6 wo
                // bytes get fed to gemm_hfq4g256_residual which reads them
                // as 136 B/group HFQ4 layout vs the actual 200 B/group MQ6
                // — catastrophic stride mismatch produces a single-token
                // attractor on AWQ A3B's 4/40 FA layers with MQ6 wo).
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
                // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
                // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
                // scratch, then add into x_batch.
                let fa_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let fa_wo_input = if fa_wo_is_q8 {
                    &pbs.fa_attn_out_batch
                } else if fa_wo_is_paro {
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    // Non-WMMA Q8: GEMM into a scratch then add into x_batch.
                    // Reuse `fa_attn_out_rot_batch` (free since MQ4 rotate
                    // didn't run here) as scratch.
                    let scratch = pbs.fa_attn_out_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free since
                    // QKVZA is done — the MoE FFN body below rewrites it
                    // as its first action) as the gemm output scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // Batched MoE FFN.
                prefill_moe_ffn_body_batched(
                    gpu,
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    &ctx,
                    weights.moe_has_mq6,
                    routed_out,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = kv_dim;
                let _ = q_dim;
                kv_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
        dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
    }

    // ── 3. Final output norm + logits ───────────────────────────────────
    // Multi-GPU band-mode: skip when this is not the last band — the
    // running activation in `pbs.x_batch` is what the next band's
    // peer-copy reads. `weights.output_norm` and `weights.output` only
    // live on the last band's device anyway.
    if do_lm_head {
        // If the caller requested per-token hidden output (DFlash verify path),
        // run rmsnorm over all N rows into their buffer. Otherwise use the
        // legacy last-token-only path.
        if let Some((dst, offset_rows)) = per_token_hidden_out {
            let dst_view = dst.sub_offset(offset_rows * dim, n * dim);
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &weights.output_norm,
                &dst_view,
                n,
                dim,
                config.norm_eps,
            )?;
            if prefill_should_emit_last_token_logits(true, needs_last_token_logits) {
                // Still populate s.logits with the last-token logits for
                // callers that rely on it (the legacy prefill post-condition).
                let last = n - 1;
                let last_view = dst.sub_offset((offset_rows + last) * dim, dim);
                {
                    let wr = weights.output.dispatch_ref();
                    let step = Step::Gemv {
                        w: &wr,
                        input: GemvInput::Raw(&last_view),
                        out: &s.logits,
                    };
                    execute_steps(gpu, &ctx, &[step])
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }
            }
        } else {
            // Legacy path: only last-token logits.
            // Use _auto so the D→D copy routes through the active stream
            // during hipGraph capture (bare memcpy_dtod_at uses the legacy
            // null stream and breaks capture: HIP error 906).
            let last = n - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            {
                let wr = weights.output.dispatch_ref();
                let step = Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.logits,
                };
                execute_steps(gpu, &ctx, &[step])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }
        }
    }

    Ok(())
}

/// Run a single FullAttn layer body on s.x at position `pos`. Extracted
/// for use from the batched prefill path's FA-layer fallback. Byte-exact
/// with the FA branch of forward_scratch_layers.
#[allow(clippy::too_many_arguments)]
fn run_fa_layer_body(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    _kv_layer_idx: usize,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let layer = match &weights.layers[layer_idx] {
        LayerWeights::FullAttn(l) => l,
        _ => unreachable!(),
    };

    // Fused rmsnorm + FWHT rotation for wq/wk/wv (MQ-family).
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.wq,
        &s.x,
        &layer.attn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
    let dt = layer.wq.gpu_dtype;
    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
    let fused_fa3_mq4 = fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_fa3_hfq6 = fa3_same_dtype
        && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_fa3_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq4g256(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq3g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq4g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq6g256_dp4a(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else {
        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
    }

    gpu.deinterleave_f32(
        &s.fa_q_full,
        &s.fa_q,
        &s.fa_gate,
        config.n_heads,
        config.head_dim,
    )?;
    gpu.rmsnorm_batched(
        &s.fa_q,
        &layer.q_norm,
        &s.fa_q,
        config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    let kv_dim = config.n_kv_heads * config.head_dim;
    gpu.rmsnorm_batched(
        &s.fa_k,
        &layer.k_norm,
        &s.fa_k,
        config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    if hipfire_runtime::triattn::tap_enabled() {
        // Try GPU path first (matches the batched FA tap at line ~3499 in
        // forward_prefill_batch). When the calibration tap is GPU-resident
        // (CalibrateGpu) we MUST dispatch the kernel here — falling
        // through to record_prerope_qk would either silently drop the
        // sample (pre-Phase-2) or panic (post-Phase-2).
        let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
            gpu,
            layer_idx,
            &s.fa_q.buf,
            1,
            config.n_heads,
            config.head_dim,
        )?;
        if !gpu_handled {
            let n_q = config.n_heads * config.head_dim;
            let q_cpu = gpu.download_f32(&s.fa_q)?;
            if hipfire_runtime::triattn::tap_needs_k() {
                let n_k = config.n_kv_heads * config.head_dim;
                let k_cpu = gpu.download_f32(&s.fa_k)?;
                hipfire_runtime::triattn::record_prerope_qk(
                    layer_idx,
                    &q_cpu[..n_q],
                    Some(&k_cpu[..n_k]),
                );
            } else {
                hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
            }
        }
    }

    // If TriAttention has compacted the cache, absolute RoPE phase diverges
    // from the physical cache index. Temporarily load the absolute position
    // into pos_buf for the rope call, then restore the physical position
    // for kv_cache_write + flash attention (which both want the write slot).
    if kv_cache.compact_offset > 0 {
        let abs = (pos + kv_cache.compact_offset) as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
    }
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    gpu.rope_partial_interleaved_f32(
        &s.fa_q,
        &s.fa_k,
        &s.pos_buf,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
    )?;
    if kv_cache.compact_offset > 0 {
        let phys = pos as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
    }
    let ctx = DispatchCtx::new(gpu);
    let fused_epilogue =
        kv_cache_attention_dispatch(&ctx, gpu, kv_cache, s, config, &layer.wo, layer_idx, pos)?;

    if !fused_epilogue {
        gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
    }
    {
        let wr = layer.wo.dispatch_ref();
        let input = if fused_epilogue {
            GemvInput::Prerotated(&s.fa_attn_out)
        } else {
            GemvInput::Raw(&s.fa_attn_out)
        };
        execute_steps(
            gpu,
            &ctx,
            &[Step::GemvResidual {
                w: &wr,
                input,
                residual: &s.x,
                out: &s.x,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // FFN: fused rmsnorm + rotate for w_gate/w_up.
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.w_gate,
        &s.x,
        &layer.ffn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    let dt_g = layer.w_gate.gpu_dtype;
    let same_dtype = layer.w_up.gpu_dtype == dt_g;
    let fused_gu_mq4 = same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_gu_hfq6 = same_dtype
        && (dt_g == DType::MQ6G256 || dt_g == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_gu_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq4g256(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq3g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq4g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq6g256_dp4a(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else {
        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
    }
    weight_gemv_swiglu_residual(gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x)?;

    Ok(())
}

/// Same as `forward_scratch` but also extracts hidden states from the
/// configured target layers into `hidden_rb`. Used by the DFlash draft path
/// during target verification. `hidden_rb.advance_head()` is called once
/// automatically at the end of the forward pass.
pub fn forward_scratch_with_hidden(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: &mut HiddenStateRingBuffer,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_with_hidden")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        Some(hidden_rb),
    )?;
    hidden_rb.advance_head();
    Ok(())
}

/// Zero-alloc forward from pre-computed embedding in scratch.x.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_embed")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Upload embedding directly into scratch.x
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding_data.as_ptr() as *const u8,
            embedding_data.len() * 4,
        )
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)
}

/// Batched single-weight GEMM used by the mixed-format fallback in
/// `forward_prefill_chunk`'s FA QKV path. The fused `gemm_qkv_hfq*` kernels
/// require wq/wk/wv to share a bit-width — they index all three weight
/// buffers with the same stride. When `--kmap-dense --kmap-mode 2` promotes
/// only `v_proj` to MQ6 (issue #249), the fused HFQ4 kernel reads `wv`'s
/// MQ6 buffer with HFQ4's 136-B stride (true stride: 200 B), producing
/// silent NaN. Callers gate the fused path on a same-dtype check and route
/// here per-weight when they disagree.
///
/// Covers same-rotation-family bit-width mixes: MQ4+MQ6 (both
/// FWHT-baked, what kmap mode 2 produces) and HFQ4+HFQ6 (both
/// unrotated). Cross-family mixes (e.g. HFQ4+MQ6) would corrupt the
/// shared rmsnorm+rotate output; no quantizer config produces them
/// today, but extend the dispatch caller's invariants here if that
/// changes.
fn batched_gemm_single_weight(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::MQ4G256 | DType::HFQ4G256 => run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G256,
            &w.buf,
            w.gpu_dtype,
            x,
            y,
            w.m,
            w.k,
            n,
        ),
        DType::MQ6G256 | DType::HFQ6G256 => {
            // No non-residual batched MQ6/HFQ6 GEMM exists. Zero Y then
            // accumulate. The zero MUST be ordered on the same stream as
            // the GEMM that consumes it — using sync `hipMemset` on the
            // null stream while subsequent kernels enqueue on a non-null
            // active stream leaves a race that produces silent NaN in the
            // residual stream (logits stay NaN on eval until a stray host
            // sync masks the order bug).
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ3G256 => {
            // Same pattern as MQ6: no non-residual batched HFQ3 GEMM
            // exists in the scalar gfx10 family — `gemm_hfq3g256_residual`
            // is the only single-weight batched dispatch. Zero Y on the
            // active stream (same race-free contract as the HFQ6 arm)
            // then accumulate.
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::Q8_0 => {
            // Q8 weights consume the un-rotated rmsnorm output. Callers
            // routing here must pass `pbs.x_rot_batch` containing
            // `rmsnorm(x_batch)` *without* FWHT — the existing pattern is
            // to gate the `fused_rmsnorm_rotate_*_for(...)` call on
            // `is_mq` and fall through to `gpu.rmsnorm_batched(...)` for
            // Q8 (see DNMoe LA preamble for a representative).
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "mixed-format batched prefill: weight dtype {other:?} has no \
             single-weight batched dispatch yet. Currently MQ3/HFQ3, \
             MQ4/HFQ4, MQ6/HFQ6, and Q8_0 mixes are wired. Re-quantize with \
             uniform format or extend `batched_gemm_single_weight` to cover this format."
            ),
        )),
    }
}

// ── Forward scratch layers (dispatch family version) ────────────────────

fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
) -> HipResult<()> {
    // #397 Ship 6 — forward-as-pipeline. When HIPFIRE_FORWARD_LOWERED=1, route
    // single-GPU decode through the lowered super-op executor. Skipped when a
    // hidden-state ring buffer is active (spec-decode capture engages only the
    // hand path for now). Default off → the hand arms below run unchanged.
    if forward_lowered_enabled() && hidden_rb.is_none() {
        return forward_scratch_layers_lowered(gpu, weights, config, pos, kv_cache, dn_state, s);
    }

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // ── DeltaNet QKVZA via pipeline ──
                qkvza_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wqkv,
                    &layer.wz,
                    &layer.w_beta,
                    &layer.w_alpha,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                )?;

                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                )?;

                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                )?;

                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;

                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                }

                gpu.gated_norm_f32(
                    &s.dn_attn_out,
                    &s.dn_z,
                    &layer.norm_weight,
                    &s.dn_normed,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                )?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.dn_normed),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── FFN ──
                gate_up_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                )?;

                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu,
                    &layer.w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                trace_finite_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} LinearAttention residual"),
                    &s.x,
                )?;
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                qkv_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                )?;

                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_q,
                    &layer.q_norm,
                    &s.fa_q,
                    config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_k,
                    &layer.k_norm,
                    &s.fa_k,
                    config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    triattn_tap(gpu, layer_idx, &s, config)?;
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(
                    &s.fa_q,
                    &s.fa_k,
                    &s.pos_buf,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                )?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                let fused_epilogue = kv_cache_attention_dispatch(
                    &ctx, gpu, kv_cache, s, config, &layer.wo, layer_idx, pos,
                )?;

                if !fused_epilogue {
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                }
                {
                    let wr = layer.wo.dispatch_ref();
                    let input = if fused_epilogue {
                        GemvInput::Prerotated(&s.fa_attn_out)
                    } else {
                        GemvInput::Raw(&s.fa_attn_out)
                    };
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input,
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── FFN ──
                gate_up_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                )?;

                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu,
                    &layer.w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                trace_finite_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttention residual"),
                    &s.x,
                )?;
                kv_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // ── DeltaNetMoe QKVZA via pipeline ──
                qkvza_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wqkv,
                    &layer.wz,
                    &layer.w_beta,
                    &layer.w_alpha,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                )?;

                // Find GDN call location by dumping after common operations
                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                )?;
                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                )?;
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                // DIAG: dump GDN inputs (per-token)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * config.linear_key_head_dim;
                    dump_hidden_localize(gpu, &s.dn_q, 1, pos, qk_dim, 0, "q_p");
                    dump_hidden_localize(gpu, &s.dn_k, 1, pos, qk_dim, 0, "k_p");
                    dump_hidden_localize(gpu, &s.dn_v, 1, pos, v_dim, 0, "v_p");
                    dump_hidden_localize(gpu, &s.dn_alpha, 1, pos, n_v_heads, 0, "alpha_p");
                    dump_hidden_localize(gpu, &s.dn_beta, 1, pos, n_v_heads, 0, "beta_p");
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                }
                // DIAG: dump GDN attention output (per-token)
                if layer_idx == 0 {
                    dump_hidden_localize(
                        gpu,
                        &s.dn_attn_out,
                        1,
                        pos,
                        n_v_heads * config.linear_value_head_dim,
                        0,
                        "gdn_p",
                    );
                }

                gpu.gated_norm_f32(
                    &s.dn_attn_out,
                    &s.dn_z,
                    &layer.norm_weight,
                    &s.dn_normed,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                )?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.dn_normed),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── MoE FFN ──
                moe_ffn_dispatch(gpu, &layer.ffn, &s.x, &layer.ffn_norm, config, s, false)?;
                // DIAG: dump MoE router logits (per-token)
                if layer_idx == 0 {
                    if let Some(ref rl) = s.moe_router_logits {
                        dump_hidden_localize(gpu, rl, 1, pos, config.num_experts, 0, "router_p");
                    }
                }

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                qkv_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                )?;

                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_q,
                    &layer.q_norm,
                    &s.fa_q,
                    config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_k,
                    &layer.k_norm,
                    &s.fa_k,
                    config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    triattn_tap(gpu, layer_idx, s, config)?;
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(
                    &s.fa_q,
                    &s.fa_k,
                    &s.pos_buf,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                )?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                let fused_epilogue = kv_cache_attention_dispatch(
                    &ctx, gpu, kv_cache, s, config, &layer.wo, layer_idx, pos,
                )?;

                if !fused_epilogue {
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                }
                {
                    let wr = layer.wo.dispatch_ref();
                    let input = if fused_epilogue {
                        GemvInput::Prerotated(&s.fa_attn_out)
                    } else {
                        GemvInput::Raw(&s.fa_attn_out)
                    };
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input,
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── MoE FFN ──
                moe_ffn_dispatch(gpu, &layer.ffn, &s.x, &layer.ffn_norm, config, s, false)?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                kv_layer_idx += 1;
            }

            // Mismatched layer weight / type combinations are unreachable
            // (the loader guarantees alignment).
            _ => unreachable!(),
        }
        dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
    }

    // Final norm + logits into scratch.logits
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    Ok(())
}

// ── Dispatch helpers ─────────────────────────────────────────────────────

/// Helper: convert `WeightTensor.paro` (if present) to `GivensRef`.
fn paro_to_givens(p: &ParoRotation) -> GivensRef<'_> {
    GivensRef {
        pairs: &p.pairs,
        theta: &p.theta,
        scales: &p.channel_scales,
        krot: p.krot as usize,
    }
}

/// Unified QKVZA (4-way) projection via execute_steps for DeltaNet layers.
/// Covers all dtypes — the interpreter selects fused QKVZA kernels for eligible
/// dtypes via FUSED_TABLE guards; everything else falls through to per-op
/// dispatch (including ParoQ4G128 which does individual Givens-rotated GEMV calls).
/// Replaces rmsnorm_rotate_dispatch + fused_qkvza_dispatch.
#[allow(clippy::too_many_arguments)]
fn qkvza_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    dn_qkv: &GpuTensor,
    dn_z: &GpuTensor,
    dn_beta: &GpuTensor,
    dn_alpha: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(wqkv.gpu_dtype);
    if rotation == RotationPlan::Givens {
        // ParoQ4G128: plain rmsnorm, then per-weight Givens rotation inside run_auto.
        let wr_qkv = WeightRef {
            buf: &wqkv.buf,
            dtype: wqkv.gpu_dtype,
            m: wqkv.m,
            k: wqkv.k,
            row_stride: 0,
            rotation: wqkv.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_z = WeightRef {
            buf: &wz.buf,
            dtype: wz.gpu_dtype,
            m: wz.m,
            k: wz.k,
            row_stride: 0,
            rotation: wz.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_beta = WeightRef {
            buf: &w_beta.buf,
            dtype: w_beta.gpu_dtype,
            m: w_beta.m,
            k: w_beta.k,
            row_stride: 0,
            rotation: w_beta.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &w_alpha.buf,
            dtype: w_alpha.gpu_dtype,
            m: w_alpha.m,
            k: w_alpha.k,
            row_stride: 0,
            rotation: w_alpha.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wqkv.awq_scale.as_ref(),
                k: wqkv.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wr_qkv,
                input: GemvInput::Raw(x_rot),
                out: dn_qkv,
            },
            Step::Gemv {
                w: &wr_z,
                input: GemvInput::Raw(x_rot),
                out: dn_z,
            },
            Step::Gemv {
                w: &wr_beta,
                input: GemvInput::Raw(x_rot),
                out: dn_beta,
            },
            Step::Gemv {
                w: &wr_alpha,
                input: GemvInput::Raw(x_rot),
                out: dn_alpha,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        // FWHT-rotated (MQ family) or non-rotated (HFQ, Q8, etc.) dtypes.
        // RmsnormAutomatic handles FWHT when rotation != None;
        // downstream Gemv steps use Prerotated to avoid double-FWHT.
        let wr_qkv = WeightRef {
            buf: &wqkv.buf,
            dtype: wqkv.gpu_dtype,
            m: wqkv.m,
            k: wqkv.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_z = WeightRef {
            buf: &wz.buf,
            dtype: wz.gpu_dtype,
            m: wz.m,
            k: wz.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_beta = WeightRef {
            buf: &w_beta.buf,
            dtype: w_beta.gpu_dtype,
            m: w_beta.m,
            k: w_beta.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &w_alpha.buf,
            dtype: w_alpha.gpu_dtype,
            m: w_alpha.m,
            k: w_alpha.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wqkv.awq_scale.as_ref(),
                k: wqkv.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wr_qkv,
                input: GemvInput::Prerotated(x_rot),
                out: dn_qkv,
            },
            Step::Gemv {
                w: &wr_z,
                input: GemvInput::Prerotated(x_rot),
                out: dn_z,
            },
            Step::Gemv {
                w: &wr_beta,
                input: GemvInput::Prerotated(x_rot),
                out: dn_beta,
            },
            Step::Gemv {
                w: &wr_alpha,
                input: GemvInput::Prerotated(x_rot),
                out: dn_alpha,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// Unified QKV projection via execute_steps. Covers all dtypes — the interpreter
/// selects fused kernels for eligible dtypes via FUSED_TABLE guards; everything
/// else falls through to per-op dispatch. Replaces qkv_interpret_mq +
/// fused_qkv_dispatch + their preceding rmsnorm_rotate_dispatch call.
#[allow(clippy::too_many_arguments)]
fn qkv_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wq: &WeightTensor,
    wk: &WeightTensor,
    wv: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    fa_v: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(wq.gpu_dtype);
    if rotation == RotationPlan::Givens {
        let wrq = WeightRef {
            buf: &wq.buf,
            dtype: wq.gpu_dtype,
            m: wq.m,
            k: wq.k,
            row_stride: 0,
            rotation: wq.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wrk = WeightRef {
            buf: &wk.buf,
            dtype: wk.gpu_dtype,
            m: wk.m,
            k: wk.k,
            row_stride: 0,
            rotation: wk.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wrv = WeightRef {
            buf: &wv.buf,
            dtype: wv.gpu_dtype,
            m: wv.m,
            k: wv.k,
            row_stride: 0,
            rotation: wv.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wrq,
                input: GemvInput::Raw(x_rot),
                out: fa_q,
            },
            Step::Gemv {
                w: &wrk,
                input: GemvInput::Raw(x_rot),
                out: fa_k,
            },
            Step::Gemv {
                w: &wrv,
                input: GemvInput::Raw(x_rot),
                out: fa_v,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        let wrq = WeightRef {
            buf: &wq.buf,
            dtype: wq.gpu_dtype,
            m: wq.m,
            k: wq.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wrk = WeightRef {
            buf: &wk.buf,
            dtype: wk.gpu_dtype,
            m: wk.m,
            k: wk.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wrv = WeightRef {
            buf: &wv.buf,
            dtype: wv.gpu_dtype,
            m: wv.m,
            k: wv.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wrq,
                input: GemvInput::Prerotated(x_rot),
                out: fa_q,
            },
            Step::Gemv {
                w: &wrk,
                input: GemvInput::Prerotated(x_rot),
                out: fa_k,
            },
            Step::Gemv {
                w: &wrv,
                input: GemvInput::Prerotated(x_rot),
                out: fa_v,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// Unified gate+up (FFN) projection via execute_steps. Covers all dtypes.
/// Replaces fused_gate_up_dispatch + its preceding rmsnorm_rotate_dispatch call.
#[allow(clippy::too_many_arguments)]
fn gate_up_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w_gate: &WeightTensor,
    w_up: &WeightTensor,
    ffn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,
    x_rot: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(w_gate.gpu_dtype);
    if rotation == RotationPlan::Givens {
        let wrg = WeightRef {
            buf: &w_gate.buf,
            dtype: w_gate.gpu_dtype,
            m: w_gate.m,
            k: w_gate.k,
            row_stride: 0,
            rotation: w_gate.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wru = WeightRef {
            buf: &w_up.buf,
            dtype: w_up.gpu_dtype,
            m: w_up.m,
            k: w_up.k,
            row_stride: 0,
            rotation: w_up.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: ffn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: w_gate.awq_scale.as_ref(),
                k: w_gate.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wrg,
                input: GemvInput::Raw(x_rot),
                out: gate_out,
            },
            Step::Gemv {
                w: &wru,
                input: GemvInput::Raw(x_rot),
                out: up_out,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        let wrg = WeightRef {
            buf: &w_gate.buf,
            dtype: w_gate.gpu_dtype,
            m: w_gate.m,
            k: w_gate.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wru = WeightRef {
            buf: &w_up.buf,
            dtype: w_up.gpu_dtype,
            m: w_up.m,
            k: w_up.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: ffn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: w_gate.awq_scale.as_ref(),
                k: w_gate.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wrg,
                input: GemvInput::Prerotated(x_rot),
                out: gate_out,
            },
            Step::Gemv {
                w: &wru,
                input: GemvInput::Prerotated(x_rot),
                out: up_out,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// MoE FFN dispatch — mirrors the two-path logic from the original.
fn moe_ffn_dispatch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    defer_routed_combine: bool,
) -> HipResult<()> {
    let r = if ffn_gate_side_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x,
            ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim,
            config.norm_eps,
        )?;
        let refs = MoeScratchRef::from_scratch(s);
        moe_ffn_decode_impl(
            gpu,
            ffn,
            x,
            x,
            config,
            &refs,
            true,
            None,
            false,
            defer_routed_combine,
        )
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        moe_ffn_decode_with_scratch(gpu, ffn, &s.tmp, x, config, s)
    };
    r?;
    trace_finite_if_enabled(gpu, "moe_ffn", x)?;
    Ok(())
}

/// EP (Ship 6 substrate-EP) variant of `moe_ffn_dispatch`: same rmsnorm/rotate +
/// MoE decode, but the routed combine + shared-down accumulate into `routed_out`
/// (a zeroed per-rank partial the EP executor all-reduces), and `skip_shared`
/// gates the shared-expert down to rank 0. Calls `moe_ffn_decode_impl` directly
/// (the `with_scratch` wrappers don't carry EP params). The residual `x` is left
/// untouched — the executor adds the all-reduced partial into it afterward.
fn moe_ffn_dispatch_ep(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    routed_out: &GpuTensor,
    skip_shared: bool,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(s);
    if ffn_all_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x,
            ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim,
            config.norm_eps,
        )?;
        moe_ffn_decode_impl(
            gpu,
            ffn,
            x,
            x,
            config,
            &refs,
            true,
            Some(routed_out),
            skip_shared,
            false,
        )
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        moe_ffn_decode_impl(
            gpu,
            ffn,
            &s.tmp,
            x,
            config,
            &refs,
            false,
            Some(routed_out),
            skip_shared,
            false,
        )
    }
}

/// EP (Ship 6 substrate-EP, ported from tp-mtp-prototype Stage 3e): shard a MoE
/// layer's routed experts to `rank`. Frees the non-owned experts (the memory
/// win), compacts owned to the front of `ffn.experts` (so `experts[0]` stays a
/// valid shared-AWQ representative for the batched silu/rotate helpers), and
/// rebuilds the `[2·n_exp]` device pointer tables: owned global id → its
/// (compacted) buffer ptr; **non-owned → a shared ZEROED gate_up buffer**.
/// Zeroed quant bytes dequant to +0.0 → the non-owned expert's gate_up output
/// is 0 → silu·mul = 0 → rot = 0 → down output 0, so it contributes nothing
/// through `moe_down_combine` WITHOUT any masking kernel. (The non-owned down
/// ptr is irrelevant — its input rot is already 0 — so it reuses
/// `experts[0].down`.) Router / shared expert / attention stay full (replicated
/// in EP v1). The zero buffer is leaked for v1 (lives until teardown) to avoid
/// threading a lifetime field through `Qwen35Weights`.
pub fn shard_moe_experts(
    gpu: &mut Gpu,
    ffn: &mut MoeFfnWeights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
) -> HipResult<()> {
    if ffn.packed_expert_owners.is_some() {
        return Err(HipError::new(
            0,
            "shard_moe_experts cannot post-shard packed owners; use the streaming EP load path",
        ));
    }
    debug_assert_eq!(
        ffn.experts.len(),
        n_exp,
        "shard_moe_experts expects a full-loaded expert Vec (paged EP is unsupported in v1)",
    );
    // Free non-owned experts; compact owned to the front, recording global→local.
    let old = std::mem::take(&mut ffn.experts);
    let mut compacted: Vec<ExpertWeights> = Vec::with_capacity(shard.experts_per_rank(n_exp));
    let mut local_of_global = vec![usize::MAX; n_exp];
    for (e, ew) in old.into_iter().enumerate() {
        if shard.owns_expert(rank, e) {
            local_of_global[e] = compacted.len();
            compacted.push(ew);
        } else {
            let _ = gpu.free_tensor(ew.gate_up.buf);
            if let Some(s) = ew.gate_up.awq_scale {
                let _ = gpu.free_tensor(s);
            }
            let _ = gpu.free_tensor(ew.down.buf);
            if let Some(s) = ew.down.awq_scale {
                let _ = gpu.free_tensor(s);
            }
        }
    }
    assert!(
        !compacted.is_empty(),
        "shard_moe_experts: rank {rank} owns no experts (n_exp={n_exp}, tp={})",
        shard.tp_size,
    );

    // Shared zeroed gate_up buffer for non-owned slots (same byte size as a real
    // expert's gate_up). LEAKED (mem::forget) so the ptr stays valid for the
    // model's lifetime without a Qwen35Weights field — v1 TODO: own it properly.
    let gu_bytes = compacted[0].gate_up.buf.buf.size();
    let zero_gu = gpu.zeros(&[gu_bytes / 4], DType::F32)?;
    let dummy_gu = zero_gu.buf.as_ptr() as u64;
    let dummy_dn = compacted[0].down.buf.buf.as_ptr() as u64; // rot=0 ⇒ output 0 regardless
    std::mem::forget(zero_gu);

    // Rebuild the [2·n_exp] u64 pointer tables (8 B/ptr = 2 F32 slots).
    let mut gu = vec![0u64; n_exp];
    let mut dn = vec![0u64; n_exp];
    for e in 0..n_exp {
        if shard.owns_expert(rank, e) {
            let li = local_of_global[e];
            gu[e] = compacted[li].gate_up.buf.buf.as_ptr() as u64;
            dn[e] = compacted[li].down.buf.buf.as_ptr() as u64;
        } else {
            gu[e] = dummy_gu;
            dn[e] = dummy_dn;
        }
    }
    let gu_b: Vec<u8> = gu.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_b: Vec<u8> = dn.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&ffn.expert_gate_up_ptrs.buf, &gu_b)?;
    gpu.hip.memcpy_htod(&ffn.expert_down_ptrs.buf, &dn_b)?;

    // Route A MoE-AWQ under EP: rebuild the per-expert down.awq_scale pointer
    // table over the compacted set. Non-owned slots get a valid dummy pointer
    // (compacted[0]'s scale) — they read zeroed gate_up ⇒ silu output 0 ⇒
    // 0/scale = 0 regardless, so the all-reduced sum is unaffected.
    if let Some(awq_tbl) = ffn.expert_down_awq_ptrs.as_ref() {
        let dummy_aw = compacted[0]
            .down
            .awq_scale
            .as_ref()
            .map(|s| s.buf.as_ptr() as u64)
            .unwrap_or(0);
        let mut aw = vec![dummy_aw; n_exp];
        for (e, slot) in aw.iter_mut().enumerate() {
            if shard.owns_expert(rank, e) {
                let li = local_of_global[e];
                if let Some(s) = compacted[li].down.awq_scale.as_ref() {
                    *slot = s.buf.as_ptr() as u64;
                }
            }
        }
        let aw_b: Vec<u8> = aw.iter().flat_map(|p| p.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&awq_tbl.buf, &aw_b)?;
    }

    ffn.experts = compacted;
    Ok(())
}

/// Shard every MoE layer of a replicated `Qwen35Weights` to `rank`, calling
/// [`shard_moe_experts`] on each `DeltaNetMoe` / `FullAttnMoe` layer's FFN.
/// Dense / attention-only layers are untouched. Convenience wrapper for the EP
/// load path so callers (the `forward_ep` driver / examples) never reach into
/// `LayerWeights` internals. `n_exp` is the model's routed expert count
/// (`config.num_experts`).
///
/// `reap_active` MUST be `config.reap_keep.is_some()`. REAP expert-pruning and
/// EP sharding are mutually exclusive (ds4/minimax enforce the same at expert-
/// load time): under REAP `config.num_experts` is already overridden to the
/// KEPT count, so `shard_moe_experts`' `experts.len() == n_exp` precondition
/// would pass on a pruned model and the per-rank ownership math would re-remap
/// already-compacted expert ids → silent weight corruption. Refuse up front.
pub fn shard_all_moe_layers(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    reap_active: bool,
) -> HipResult<()> {
    if reap_active {
        return Err(HipError::new(
            0,
            "qwen35: REAP keep-map + EP sharding are mutually exclusive",
        ));
    }
    for layer in weights.layers.iter_mut() {
        match layer {
            LayerWeights::DeltaNetMoe(l) => shard_moe_experts(gpu, &mut l.ffn, shard, rank, n_exp)?,
            LayerWeights::FullAttnMoe(l) => shard_moe_experts(gpu, &mut l.ffn, shard, rank, n_exp)?,
            _ => {}
        }
    }
    Ok(())
}

/// TriAttention tap helper (inline from original forward).
fn triattn_tap(
    gpu: &mut Gpu,
    layer_idx: usize,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
) -> HipResult<()> {
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu,
        layer_idx,
        &s.fa_q.buf,
        1,
        config.n_heads,
        config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(&s.fa_q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(&s.fa_k)?;
            hipfire_runtime::triattn::record_prerope_qk(
                layer_idx,
                &q_cpu[..n_q],
                Some(&k_cpu[..n_k]),
            );
        } else {
            hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
        }
    }
    Ok(())
}

/// KV cache write + attention dispatch. Inline from original.
fn kv_cache_attention_dispatch(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    kv_cache: &mut llama::KvCache,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    wo: &WeightTensor,
    layer_idx: usize,
    pos: usize,
) -> HipResult<bool> {
    let plan = KvTierPlan::derive(KvTierInputs {
        pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.graphs.capture_mode,
        ..kv_cache.tier_inputs()
    })
    .map_err(|e| HipError::new(0, &e.to_string()))?;
    let fused_epilogue = qwen35_fa_epilogue_enabled(gpu, config, wo)
        && plan.write_key == hipfire_dispatch::types::KernelKey::KvWriteQ8_0
        && plan.attend_key == hipfire_dispatch::types::KernelKey::AttnFlashQ8_0;
    let io = AttnParams {
        q: &s.fa_q,
        k: &s.fa_k,
        v: &s.fa_v,
        k_cache: &kv_cache.k_gpu[layer_idx],
        v_cache: &kv_cache.v_gpu[layer_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &s.pos_buf,
        pos,
        positions: None,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        physical_cap: kv_cache.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&s.flash_partials),
        givens_cos: kv_cache.givens_cos.as_ref(),
        givens_sin: kv_cache.givens_sin.as_ref(),
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: fused_epilogue.then_some(&s.fa_gate),
        output: &s.fa_attn_out,
    };
    execute_steps(gpu, ctx, &[Step::Attend { plan, io }])
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    Ok(fused_epilogue)
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: qwen35 DECODE lowered path (ADDITIVE).
//
// `HIPFIRE_FORWARD_LOWERED=1` routes the single-GPU decode layer loop through
// the dispatch substrate's `run_layer_program` executor (one pre-resolved
// `LayerProgram` of coarse super-ops per layer) instead of the hand-written
// arms in `forward_scratch_layers`. The hand arms are left UNTOUCHED, so the
// default (flag off) is byte-identical to master by construction; the lowered
// path is validated byte-identical via the external committed-token md5 gate
// (`FORWARD_LOWERED=0` vs `=1`, same prompt) on the fleet before the default is
// flipped per arch. See [[project_ship6_forward_pipeline_design_2026_06_07]].
//
// The super-op handlers call the SAME helper fns the hand path uses
// (`qkv/qkvza/gate_up_via_execute_steps`, `kv_cache_attention_dispatch`,
// `moe_ffn_dispatch`, `weight_gemv_swiglu_residual`) plus the inline attend/
// recurrent/gated-norm fragments. DIAG dumps / trace_finite / hidden_rb are
// output-neutral and omitted here (hidden_rb engages only the hand path).
// ─────────────────────────────────────────────────────────────────────────

/// qwen35-local super-op opcodes, encoded into `OpBinding.weights[0].0`. The
/// `SuperOpKind` routes to the `ForwardBindings` method; the opcode disambiguates
/// *which* op of that kind within the layer (qkv vs gate_up, wo vs down, …).
mod q35_op {
    // Proj
    pub const PROJ_QKV: u32 = 0;
    pub const PROJ_QKVZA: u32 = 1;
    pub const PROJ_GATE_UP: u32 = 2;
    // Attend
    pub const ATTEND_FULL: u32 = 0;
    pub const ATTEND_DN_PREP: u32 = 1;
    // ResidualGemv
    pub const RESID_WO: u32 = 0;
    pub const RESID_DOWN_SWIGLU: u32 = 1;
    // Norm
    pub const NORM_GATED: u32 = 0;
    // Recurrent
    pub const RECUR_GDN: u32 = 0;
    // Moe
    pub const MOE_FFN: u32 = 0;
}

/// The four qwen35 decoder-layer shapes. Derived from the `LayerWeights`
/// discriminant; kept as a plain enum so `lower_variant` is pure (no GpuTensor)
/// and unit-testable without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Q35Variant {
    DeltaNet,
    FullAttn,
    DeltaNetMoe,
    FullAttnMoe,
}

fn variant_of(layer: &LayerWeights) -> Q35Variant {
    match layer {
        LayerWeights::DeltaNet(_) => Q35Variant::DeltaNet,
        LayerWeights::FullAttn(_) => Q35Variant::FullAttn,
        LayerWeights::DeltaNetMoe(_) => Q35Variant::DeltaNetMoe,
        LayerWeights::FullAttnMoe(_) => Q35Variant::FullAttnMoe,
    }
}

#[inline]
fn q35_superop(kind: SuperOpKind, code: u32) -> SuperOp {
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

/// Lower one qwen35 decoder layer to a coarse-super-op `LayerProgram`. The op
/// SEQUENCE mirrors the matching hand arm in `forward_scratch_layers` exactly
/// (per the decode-forward variant map). Pure → unit-testable.
fn lower_variant(v: Q35Variant) -> LayerProgram {
    use q35_op::*;
    use SuperOpKind::{Attend, Moe, Norm, Proj, Recurrent, ResidualGemv};
    match v {
        Q35Variant::DeltaNet => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::FullAttn => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::DeltaNetMoe => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
        Q35Variant::FullAttnMoe => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn qkv_from_prerotated_mq(
    gpu: &mut Gpu,
    wq: &WeightTensor,
    wk: &WeightTensor,
    wv: &WeightTensor,
    x_rot: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
) -> HipResult<()> {
    gpu.fused_qkv_hfq4g256(
        &wq.buf, &wk.buf, &wv.buf, x_rot, q, k, v, wq.m, wk.m, wv.m, wq.k,
    )
}

#[allow(clippy::too_many_arguments)]
fn qkvza_from_prerotated_mq(
    gpu: &mut Gpu,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
    x_rot: &GpuTensor,
    qkv: &GpuTensor,
    z: &GpuTensor,
    beta: &GpuTensor,
    alpha: &GpuTensor,
) -> HipResult<()> {
    gpu.fused_qkvza_hfq4g256(
        &wqkv.buf,
        &wz.buf,
        &w_beta.buf,
        &w_alpha.buf,
        x_rot,
        qkv,
        z,
        beta,
        alpha,
        wqkv.m,
        wz.m,
        w_beta.m,
        w_alpha.m,
        wqkv.k,
    )
}

/// Per-layer execution context for the lowered decode path. Holds the current
/// layer's weights + shared scratch/state by reference; rebuilt each layer
/// iteration so the borrows stay scoped. `kv_cache` is the only `&mut` (DeltaNet
/// state is mutated through interior-mutable GpuTensor buffers via shared refs).
struct Qwen35Bindings<'a> {
    layer: &'a LayerWeights,
    s: &'a Qwen35Scratch,
    config: &'a Qwen35Config,
    kv_cache: &'a mut llama::KvCache,
    dn_state: &'a DeltaNetState,
    pos: usize,
    layer_idx: usize,
    delta_layer_idx: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    precomputed_attn_x_rot: bool,
    fa_output_prerotated: bool,
    defer_routed_combine: bool,
}

fn op_code(op: &OpBinding) -> u32 {
    op.weights.first().map(|w| w.0).unwrap_or(u32::MAX)
}

impl<'a> ForwardBindings for Qwen35Bindings<'a> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let res: HipResult<()> = match op_code(op) {
            q35_op::PROJ_QKV => match self.layer {
                LayerWeights::FullAttn(l) => {
                    if self.precomputed_attn_x_rot {
                        qkv_from_prerotated_mq(
                            gpu,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                        )
                    } else {
                        qkv_via_execute_steps(
                            gpu,
                            ctx,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &l.attn_norm,
                            &s.x,
                            &s.tmp,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            config.norm_eps,
                        )
                    }
                }
                LayerWeights::FullAttnMoe(l) => {
                    if self.precomputed_attn_x_rot {
                        qkv_from_prerotated_mq(
                            gpu,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                        )
                    } else {
                        qkv_via_execute_steps(
                            gpu,
                            ctx,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &l.attn_norm,
                            &s.x,
                            &s.tmp,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            config.norm_eps,
                        )
                    }
                }
                _ => return Err(DispatchError::Hip("PROJ_QKV on non-FullAttn layer".into())),
            },
            q35_op::PROJ_QKVZA => {
                let (wqkv, wz, w_beta, w_alpha, attn_norm, dt_bias, a_log) = match self.layer {
                    LayerWeights::DeltaNet(l) => (
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                        &l.attn_norm,
                        &l.dt_bias,
                        &l.a_log,
                    ),
                    LayerWeights::DeltaNetMoe(l) => (
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                        &l.attn_norm,
                        &l.dt_bias,
                        &l.a_log,
                    ),
                    _ => {
                        return Err(DispatchError::Hip(
                            "PROJ_QKVZA on non-DeltaNet layer".into(),
                        ));
                    }
                };
                if self.precomputed_attn_x_rot {
                    qkvza_from_prerotated_mq(
                        gpu,
                        wqkv,
                        wz,
                        w_beta,
                        w_alpha,
                        &s.x_rot,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                    )
                } else if qkvza_scalar_prep_enabled(
                    gpu,
                    config,
                    self.n_v_heads,
                    self.dn_state.quant,
                    wqkv,
                    wz,
                    w_beta,
                    w_alpha,
                ) {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        wqkv,
                        &s.x,
                        attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    let eff_x = x_rot.unwrap_or(&s.tmp);
                    gpu.fused_qkvza_hfq4g256_scalar_prep_gfx1100(
                        &wqkv.buf,
                        &wz.buf,
                        &w_beta.buf,
                        &w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        dt_bias,
                        a_log,
                        wqkv.m,
                        wz.m,
                        w_beta.m,
                        w_alpha.m,
                        wqkv.k,
                    )
                } else {
                    qkvza_via_execute_steps(
                        gpu,
                        ctx,
                        wqkv,
                        wz,
                        w_beta,
                        w_alpha,
                        attn_norm,
                        &s.x,
                        &s.tmp,
                        &s.x_rot,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        config.norm_eps,
                    )
                }
            }
            q35_op::PROJ_GATE_UP => match self.layer {
                LayerWeights::DeltaNet(l) => gate_up_via_execute_steps(
                    gpu,
                    ctx,
                    &l.w_gate,
                    &l.w_up,
                    &l.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                ),
                LayerWeights::FullAttn(l) => gate_up_via_execute_steps(
                    gpu,
                    ctx,
                    &l.w_gate,
                    &l.w_up,
                    &l.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                ),
                _ => {
                    return Err(DispatchError::Hip(
                        "PROJ_GATE_UP on MoE/unknown layer".into(),
                    ));
                }
            },
            other => return Err(DispatchError::Hip(format!("unknown PROJ opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let res: HipResult<()> = (|| match op_code(op) {
            q35_op::RESID_WO => {
                let (wo, input) = match self.layer {
                    LayerWeights::FullAttn(l) => {
                        let input = if self.fa_output_prerotated {
                            GemvInput::Prerotated(&s.fa_attn_out)
                        } else {
                            GemvInput::Raw(&s.fa_attn_out)
                        };
                        (&l.wo, input)
                    }
                    LayerWeights::FullAttnMoe(l) => {
                        let input = if self.fa_output_prerotated {
                            GemvInput::Prerotated(&s.fa_attn_out)
                        } else {
                            GemvInput::Raw(&s.fa_attn_out)
                        };
                        (&l.wo, input)
                    }
                    LayerWeights::DeltaNet(l) => {
                        let input = if gated_norm_mq_rotate_enabled(
                            gpu,
                            self.config,
                            self.n_v_heads,
                            &l.wo,
                        ) {
                            GemvInput::Prerotated(&s.x_rot)
                        } else {
                            GemvInput::Raw(&s.dn_normed)
                        };
                        (&l.wo, input)
                    }
                    LayerWeights::DeltaNetMoe(l) => {
                        let input = if gated_norm_mq_rotate_enabled(
                            gpu,
                            self.config,
                            self.n_v_heads,
                            &l.wo,
                        ) {
                            GemvInput::Prerotated(&s.x_rot)
                        } else {
                            GemvInput::Raw(&s.dn_normed)
                        };
                        (&l.wo, input)
                    }
                };
                let wr = wo.dispatch_ref();
                execute_steps(
                    gpu,
                    ctx,
                    &[Step::GemvResidual {
                        w: &wr,
                        input,
                        residual: &s.x,
                        out: &s.x,
                    }],
                )
                .map_err(|e| HipError::new(0, &e.to_string()))
            }
            q35_op::RESID_DOWN_SWIGLU => {
                let w_down = match self.layer {
                    LayerWeights::DeltaNet(l) => &l.w_down,
                    LayerWeights::FullAttn(l) => &l.w_down,
                    _ => return Err(HipError::new(0, "RESID_DOWN_SWIGLU on MoE layer")),
                };
                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu,
                    w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )
            }
            other => Err(HipError::new(0, &format!("unknown RESID opcode {other}"))),
        })();
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_norm(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (norm_weight, wo) = match self.layer {
            LayerWeights::DeltaNet(l) => (&l.norm_weight, &l.wo),
            LayerWeights::DeltaNetMoe(l) => (&l.norm_weight, &l.wo),
            _ => {
                return Err(DispatchError::Hip(
                    "NORM_GATED on non-DeltaNet layer".into(),
                ));
            }
        };
        if gated_norm_mq_rotate_enabled(gpu, config, self.n_v_heads, wo) {
            gpu.gated_norm_rotate_mq_gfx1100(
                &s.dn_attn_out,
                &s.dn_z,
                norm_weight,
                &s.x_rot,
                self.n_v_heads,
                config.linear_value_head_dim,
                config.norm_eps,
            )
        } else {
            gpu.gated_norm_f32(
                &s.dn_attn_out,
                &s.dn_z,
                norm_weight,
                &s.dn_normed,
                self.n_v_heads,
                config.linear_value_head_dim,
                config.norm_eps,
            )
        }
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let res: HipResult<()> = (|| match op_code(op) {
            q35_op::ATTEND_FULL => {
                let (q_norm, k_norm, wo) = match self.layer {
                    LayerWeights::FullAttn(l) => (&l.q_norm, &l.k_norm, &l.wo),
                    LayerWeights::FullAttnMoe(l) => (&l.q_norm, &l.k_norm, &l.wo),
                    _ => return Err(HipError::new(0, "ATTEND_FULL on non-FullAttn layer")),
                };
                let tap_enabled = hipfire_runtime::triattn::tap_enabled();
                let fused_prep = qwen35_fa_prep_enabled(gpu, config) && !tap_enabled;
                if !fused_prep {
                    gpu.deinterleave_f32(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                }
                if tap_enabled {
                    triattn_tap(gpu, self.layer_idx, s, config)?;
                }
                if self.kv_cache.compact_offset > 0 {
                    let abs = (self.pos + self.kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                if fused_prep {
                    gpu.qwen35_fa_prep_gfx1100(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        &s.fa_k,
                        q_norm,
                        k_norm,
                        &s.pos_buf,
                        config.norm_eps,
                        config.rope_theta,
                    )?;
                } else {
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                }
                if self.kv_cache.compact_offset > 0 {
                    let phys = self.pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }
                let fused_epilogue = kv_cache_attention_dispatch(
                    ctx,
                    gpu,
                    self.kv_cache,
                    s,
                    config,
                    wo,
                    self.layer_idx,
                    self.pos,
                )?;
                if !fused_epilogue {
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                }
                self.fa_output_prerotated = fused_epilogue;
                Ok(())
            }
            q35_op::ATTEND_DN_PREP => {
                let (dt_bias, a_log, conv_weight, wqkv, wz, w_beta, w_alpha) = match self.layer {
                    LayerWeights::DeltaNet(l) => (
                        &l.dt_bias,
                        &l.a_log,
                        &l.conv_weight,
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                    ),
                    LayerWeights::DeltaNetMoe(l) => (
                        &l.dt_bias,
                        &l.a_log,
                        &l.conv_weight,
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                    ),
                    _ => return Err(HipError::new(0, "ATTEND_DN_PREP on non-DeltaNet layer")),
                };
                let qkvza_scalar_prep = qkvza_scalar_prep_enabled(
                    gpu,
                    config,
                    self.n_v_heads,
                    self.dn_state.quant,
                    wqkv,
                    wz,
                    w_beta,
                    w_alpha,
                );
                let conv_scalar_prep = !qkvza_scalar_prep
                    && conv_scalar_prep_enabled(gpu, config, self.n_v_heads, self.dn_state.quant);
                if !qkvza_scalar_prep && !conv_scalar_prep {
                    gpu.fused_sigmoid_alpha_gate_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        dt_bias,
                        a_log,
                        self.n_v_heads,
                    )?;
                }
                if conv_qknorm_enabled(gpu, config, self.dn_state.quant) {
                    if conv_scalar_prep {
                        gpu.conv1d_silu_split_qknorm_scalar_prep_gfx1100(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_v,
                            &s.dn_qkv,
                            conv_weight,
                            &self.dn_state.conv_states[self.delta_layer_idx],
                            &s.dn_beta,
                            &s.dn_alpha,
                            dt_bias,
                            a_log,
                            self.k_dim,
                            self.v_dim,
                            config.linear_num_key_heads,
                            self.hd,
                            1.0 / (self.hd as f32).sqrt(),
                            config.norm_eps,
                            self.n_v_heads,
                        )?;
                    } else {
                        gpu.conv1d_silu_split_qknorm(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_v,
                            &s.dn_qkv,
                            conv_weight,
                            &self.dn_state.conv_states[self.delta_layer_idx],
                            self.k_dim,
                            self.v_dim,
                            config.linear_num_key_heads,
                            self.hd,
                            1.0 / (self.hd as f32).sqrt(),
                            config.norm_eps,
                        )?;
                    }
                } else {
                    gpu.conv1d_silu_split_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        conv_weight,
                        &self.dn_state.conv_states[self.delta_layer_idx],
                        self.k_dim,
                        self.v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        self.hd,
                        1.0 / (self.hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                }
                if gdn_compact2_enabled(gpu, config, self.n_v_heads, self.dn_state.quant) {
                    // The compact Q8 recurrence maps state head h to Q/K head
                    // h/2 directly. Leave the normalized tensors compact and
                    // remove this materialization dispatch from the replay tape.
                } else if config.linear_num_key_heads < self.n_v_heads {
                    let ratio = self.n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        self.hd,
                    )?;
                } else {
                    // Keep the ratio-1 copy on the same kernel-dispatch
                    // surface as ratio>1. One deterministic Q+K kernel
                    // replaces two runtime memcpy nodes and makes the entire
                    // lowered decode body recordable by Redline AQL.
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        1,
                        self.hd,
                    )?;
                }
                Ok(())
            }
            other => Err(HipError::new(0, &format!("unknown ATTEND opcode {other}"))),
        })();
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        moe_ffn_dispatch(
            gpu,
            ffn,
            &s.x,
            ffn_norm,
            config,
            s,
            self.defer_routed_combine,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        skip_shared: bool,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        // Routed combine + shared-down (rank 0 only) accumulate into `routed_out`
        // (zeroed by the EP executor); s.x (the replicated attention residual) is
        // untouched until ep_add_into_residual after the all-reduce.
        moe_ffn_dispatch_ep(gpu, ffn, &s.x, ffn_norm, config, s, routed_out, skip_shared)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn ep_add_into_residual(
        &mut self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        // s.x += the all-reduced routed partial (the EP MoE output summed across
        // ranks). Mirrors the prototype's `tp_allreduce_add` residual step.
        let s = self.s;
        gpu.add_inplace_f32(&s.x, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_recurrent(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let dn = self.dn_state;
        let i = self.delta_layer_idx;
        let res: HipResult<()> = match dn.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
            StateQuant::Q8 => {
                if gdn_compact2_enabled(gpu, config, self.n_v_heads, dn.quant) {
                    gpu.gated_delta_net_q8_compact2(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn.s_matrices[i],
                        &dn.s_scales[i],
                        &s.dn_attn_out,
                        1,
                        self.n_v_heads,
                        config.linear_value_head_dim,
                        dn.ef_residual(i),
                    )
                } else {
                    gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn.s_matrices[i],
                        &dn.s_scales[i],
                        &s.dn_attn_out,
                        1,
                        self.n_v_heads,
                        config.linear_value_head_dim,
                        dn.ef_residual(i),
                    )
                }
            }
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &dn.s_scales[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("qwen35 has no Conv super-op".into()))
    }

    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "qwen35 has no Escape super-op ({kind:?})"
        )))
    }
}

/// Cached `HIPFIRE_FORWARD_LOWERED` toggle. #397 Ship 6: the qwen35 single-GPU
/// decode lowered path is **DEFAULT ON** as of 2026-06-07 — validated byte-
/// identical to the hand path via fleet decode byte-parity (RDNA3 k9lin / RDNA4
/// hiptrx / RDNA3.5 hipx, dense + MoE) and the full coherence battery (13 cases,
/// k9lin). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` forces the legacy hand arms
/// (still present in forward_scratch_layers); any other value (or unset) → lowered.
fn forward_lowered_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Exact gfx1151 admission gate for its certified Radiowave decode bundle.
/// Keep this separate from broad RDNA3 capability checks so no neighboring
/// architecture can inherit the gfx1151 schedules.
fn gfx1151_radiowave_fusions_enabled(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1151()
}

/// Decode path that keeps DeltaNet Q/K at their native head count and
/// lets each pair of value/state heads reuse one Q/K head. The architecture,
/// Q8-state, and 2:1-head gates keep every other configuration isolated; the
/// environment variable is a fail-closed escape hatch. The compact B2 schedule
/// is the certified default on gfx1100 and gfx1201; other architectures remain
/// isolated from the experiment.
fn gdn_compact2_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
) -> bool {
    let mode = hipfire_config::developer_var("HIPFIRE_GDN_COMPACT2").ok();
    let arch_enabled = (gpu.arch_caps.is_gfx1201()
        || gpu.arch_caps.arch() == "gfx1100"
        || gfx1151_radiowave_fusions_enabled(gpu))
        && mode.as_deref() != Some("0");
    arch_enabled && quant == StateQuant::Q8 && config.linear_num_key_heads * 2 == n_v_heads
}

/// Certified gfx1100 Radiowave schedule: keep DeltaNet's normalized output on
/// chip and feed the exact MQ rotation directly from LDS. The fixed A3B shape
/// lets two independent 128-value heads form one 256-value MQ group without
/// changing either operation's arithmetic order. Enabled by default for the
/// admitted shape; set `HIPFIRE_GATED_NORM_MQ_ROTATE=0` to restore the two-
/// dispatch schedule.
fn gated_norm_mq_rotate_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    wo: &WeightTensor,
) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_GATED_NORM_MQ_ROTATE")
            .ok()
            .as_deref()
            != Some("0")
    });
    enabled
        && (gpu.arch_caps.is_gfx1100() || gfx1151_radiowave_fusions_enabled(gpu))
        && config.dim == 2_048
        && n_v_heads == 32
        && config.linear_value_head_dim == 128
        && wo.k == n_v_heads * config.linear_value_head_dim
        && wo.gpu_dtype == DType::MQ4G256
        && wo.awq_scale.is_none()
}

/// Collapse full-attention Q/gate deinterleave, Q/K RMS normalization, and
/// partial half-split RoPE into one head-local launch on the certified gfx1100
/// shape. Set `HIPFIRE_QWEN35_FA_PREP_FUSE=0` to retain the legacy path.
/// The legacy interleaved-RoPE compatibility mode and diagnostic tap retain
/// the established multi-dispatch path.
fn qwen35_fa_prep_enabled(gpu: &Gpu, config: &Qwen35Config) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QWEN35_FA_PREP_FUSE")
            .ok()
            .as_deref()
            != Some("0")
    });
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    enabled
        && (gpu.arch_caps.is_gfx1100() || gfx1151_radiowave_fusions_enabled(gpu))
        && !gpu.flags.rope_interleaved_legacy
        && config.n_heads == 16
        && config.n_kv_heads == 2
        && config.head_dim == 256
        && n_rot == 64
}

/// Pair Q8 K/V cache writes and fold the Qwen output gate plus MQ rotation into
/// the flash-attention reduce epilogue on the certified gfx1100/MQ4 shape. Set
/// `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE=0` to retain the legacy path.
fn qwen35_fa_epilogue_enabled(gpu: &Gpu, config: &Qwen35Config, wo: &WeightTensor) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QWEN35_FA_EPILOGUE_FUSE")
            .ok()
            .as_deref()
            != Some("0")
    });
    enabled
        && (gpu.arch_caps.is_gfx1100() || gfx1151_radiowave_fusions_enabled(gpu))
        && config.n_heads == 16
        && config.n_kv_heads == 2
        && config.head_dim == 256
        && wo.gpu_dtype == DType::MQ4G256
        && wo.awq_scale.is_none()
}

/// Radiowave experiment: fold the DeltaNet beta/alpha scalar preparation into
/// the fixed-K QKVZA producer. Keep the gate deliberately narrow until exact
/// replay and stationary product certification justify a default flip.
#[allow(clippy::too_many_arguments)]
fn qkvza_scalar_prep_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QKVZA_SCALAR_PREP")
            .ok()
            .as_deref()
            == Some("1")
    });
    let dtype = wqkv.gpu_dtype;
    enabled
        && gpu.arch_caps.is_gfx1100()
        && gdn_compact2_enabled(gpu, config, n_v_heads, quant)
        && wqkv.k == 2_048
        && w_beta.m == n_v_heads
        && w_alpha.m == n_v_heads
        && wz.gpu_dtype == dtype
        && w_beta.gpu_dtype == dtype
        && w_alpha.gpu_dtype == dtype
        && matches!(dtype, DType::MQ4G256 | DType::HFQ4G256)
}

/// Radiowave experiment: schedule the independent beta/alpha transforms as
/// one extra workgroup of the following conv/QK-normalization dispatch. This
/// keeps the hot QKVZA projection unchanged while deleting the same boundary.
fn conv_scalar_prep_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_CONV_SCALAR_PREP")
            .ok()
            .as_deref()
            == Some("1")
    });
    let shape = hipfire_config::developer_var("HIPFIRE_CONV_QKNORM_SHAPE").ok();
    enabled
        && gpu.arch_caps.is_gfx1100()
        && n_v_heads <= 256
        && shape.as_deref().is_none_or(|v| v == "b256")
        && conv_qknorm_enabled(gpu, config, quant)
}

fn conv_qknorm_enabled(gpu: &Gpu, config: &Qwen35Config, quant: StateQuant) -> bool {
    let mode = hipfire_config::developer_var("HIPFIRE_CONV_QKNORM").ok();
    let arch_enabled = (gpu.arch_caps.is_gfx1201()
        || gpu.arch_caps.arch() == "gfx1100"
        || gfx1151_radiowave_fusions_enabled(gpu))
        && mode.as_deref() != Some("0");
    arch_enabled && quant == StateQuant::Q8 && config.linear_key_head_dim == 128
}

/// Certified gfx1100 Radiowave schedule: retain each non-final routed-MoE
/// result in expanded scratch and let the next layer combine it while producing
/// that layer's normalized MQ activation. The whole-model admission check is
/// intentionally strict so every deferred producer has exactly one compatible
/// consumer and no fallback projection can observe an incomplete residual.
/// Enabled by default for the admitted mq4r shape; set
/// `HIPFIRE_MOE_COMBINE_NEXT_RMS=0` to restore the two-dispatch schedule.
fn moe_combine_next_rms_enabled(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config) -> bool {
    static REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let requested = *REQUESTED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_COMBINE_NEXT_RMS")
            .ok()
            .as_deref()
            != Some("0")
    });
    if !requested
        || !(gpu.arch_caps.is_gfx1100() || gfx1151_radiowave_fusions_enabled(gpu))
        || config.dim != 2_048
        || config.num_experts_per_tok != 8
        || config.n_layers < 2
        || config.n_layers != weights.layers.len()
        || weights.pager.is_some()
        || hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
        || hipfire_config::developer_var("HIPFIRE_QKVZA_SCALAR_PREP").as_deref() == Ok("1")
    {
        return false;
    }

    let mq4 =
        |w: &WeightTensor| w.gpu_dtype == DType::MQ4G256 && w.k == 2_048 && w.awq_scale.is_none();
    // MUST stay in lockstep with `routed_down_self_combines` in
    // hipfire-dispatch/src/pipeline/mod.rs — this asks the same question ("does
    // this layer's down GEMV write `down_expanded`?") and is the admission gate
    // for `defer_routed_combine`. Every dtype whose down GEMV self-combines via
    // atomicAdd into x_residual writes NO expanded buffer and must return false
    // here. That is the whole codebook family: MQ2/MQ3-Lloyd AND their
    // global-codebook siblings MQ2/MQ3-GL.
    //
    // Getting this wrong is SILENT. A GL layer that wrongly claims an expanded
    // buffer lets the next layer fold stale `down_expanded` contents into the
    // residual a second time, scaled by the wrong topk_weights — no error, no
    // crash. Uniform-GL models are masked by the `gpu.zeros` memset on
    // `s.moe_down_expanded` (folding 0.0 is a no-op), but a per-layer TIER_MAP
    // mixing MQ4 and MQ2GL/MQ3GL layers is NOT masked and double-counts on every
    // GL -> next-layer transition. This is the third copy of this invariant in
    // the tree and the second to drift; if a fourth appears, hoist it onto DType.
    let has_expanded_down = |ffn: &MoeFfnWeights| {
        ffn.expert_dtype_tags.is_some()
            || ffn.experts.first().is_some_and(|expert| {
                !matches!(
                    expert.down.gpu_dtype,
                    DType::MQ2G256Lloyd
                        | DType::MQ3G256Lloyd
                        | DType::MQ2G256GL
                        | DType::MQ3G256GL
                )
            })
    };
    weights.layers.iter().all(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => {
            has_expanded_down(&l.ffn)
                && ffn_gate_side_mq4_for_moe(&l.ffn)
                && mq4(&l.wqkv)
                && mq4(&l.wz)
                && mq4(&l.w_beta)
                && mq4(&l.w_alpha)
        }
        LayerWeights::FullAttnMoe(l) => {
            has_expanded_down(&l.ffn)
                && ffn_gate_side_mq4_for_moe(&l.ffn)
                && mq4(&l.wq)
                && mq4(&l.wk)
                && mq4(&l.wv)
        }
        LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => false,
    })
}

/// Lowered (#397 Ship 6) single-GPU decode layer loop. Behaviorally equivalent
/// to `forward_scratch_layers`'s hand arms (validated byte-identical via the
/// external committed-token md5 gate). Builds a coarse-super-op `LayerProgram`
/// per layer and runs it through the dispatch substrate's executor.
fn forward_scratch_layers_lowered(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);
    let mut delta_layer_idx = 0usize;
    let combine_next_rms = moe_combine_next_rms_enabled(gpu, weights, config);
    let reuse_fused_rotation =
        hipfire_config::developer_var("HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM").as_deref() != Ok("1");

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let fused_transition = combine_next_rms && layer_idx > 0;
        if fused_transition {
            let attn_norm = match layer {
                LayerWeights::DeltaNetMoe(l) => &l.attn_norm,
                LayerWeights::FullAttnMoe(l) => &l.attn_norm,
                LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => {
                    unreachable!("moe_combine_next_rms_enabled admits only all-MoE models")
                }
            };
            gpu.moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100(
                s.moe_down_expanded.as_ref().expect("MoE scratch"),
                s.moe_topk_weights.as_ref().expect("MoE scratch"),
                &s.x,
                attn_norm,
                &s.x_rot,
                config.dim,
                config.num_experts_per_tok,
                config.norm_eps,
            )?;
            dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx - 1, "pertoken");
        }
        let program = lower_variant(variant_of(layer));
        {
            let mut bind = Qwen35Bindings {
                layer,
                s,
                config,
                kv_cache: &mut *kv_cache,
                dn_state,
                pos,
                layer_idx,
                delta_layer_idx,
                k_dim,
                v_dim,
                n_v_heads,
                hd,
                precomputed_attn_x_rot: fused_transition && reuse_fused_rotation,
                fa_output_prerotated: false,
                defer_routed_combine: combine_next_rms && layer_idx + 1 < config.n_layers,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| HipError::new(0, &e.to_string()))?;
        }
        if matches!(
            layer,
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
        if !combine_next_rms || layer_idx + 1 == config.n_layers {
            dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
        }
    }

    // Final norm + logits into scratch.logits (mirrors forward_scratch_layers).
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    Ok(())
}

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
///
/// Every rank holds **full replicated** weights / scratch / KV / DeltaNet
/// state EXCEPT the MoE routed experts, which were sharded per rank at load by
/// [`shard_moe_experts`]. Behaviorally this mirrors the single-GPU
/// [`forward_scratch`] → [`forward_scratch_layers_lowered`] pipeline (embed →
/// per-layer `LayerProgram` → final norm + lm_head), but runs each layer's
/// program through the EP executor ([`hipfire_runtime::ep::run_layer_program_ep`]):
/// the `Moe` super-op is all-reduce-EP'd across ranks (each rank computes only
/// its owned experts into a zeroed routed partial, the partials are
/// all-reduce-summed, then added into each rank's residual); every other
/// super-op runs **replicated** and stays bit-identical across ranks.
///
/// Logits land in `scratch_per_rank[0].logits` (rank 0 = `output_device`); the
/// caller reads them with `gpu.download_f32` after this returns (this fn
/// device-synchronizes every rank before returning, so the read is safe even
/// though work ran on each rank's `active_stream`).
///
/// All parallel slices (`weights_per_rank`, `kv_per_rank`, `dn_per_rank`,
/// `scratch_per_rank`, `partials`) must have length `gpus.devices.len()`, with
/// element `r` allocated on `gpus.devices[r]`. Every device must have an
/// `active_stream` set ([`hipfire_runtime::ep::ensure_rank_streams`]).
///
/// TP=1 is the degenerate reference: one rank owns all experts (no zero-dummy),
/// the all-reduce short-circuits to identity, and the result is the same as the
/// single-GPU lowered decode (validated byte-/argmax-identical on the fleet).
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_per_rank: &mut [llama::KvCache],
    dn_per_rank: &[DeltaNetState],
    scratch_per_rank: &[Qwen35Scratch],
    partials: &[GpuTensor],
) -> HipResult<()> {
    let n = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n,
        "forward_ep: weights_per_rank.len() != n_ranks"
    );
    assert_eq!(
        kv_per_rank.len(),
        n,
        "forward_ep: kv_per_rank.len() != n_ranks"
    );
    assert_eq!(
        dn_per_rank.len(),
        n,
        "forward_ep: dn_per_rank.len() != n_ranks"
    );
    assert_eq!(
        scratch_per_rank.len(),
        n,
        "forward_ep: scratch_per_rank.len() != n_ranks"
    );
    assert_eq!(partials.len(), n, "forward_ep: partials.len() != n_ranks");

    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let pos_i32 = pos as i32;

    // 1. Embed token + write pos on each rank (replicated; deterministic, since
    //    weights are byte-identical replicas → s.x is bit-identical per rank).
    for r in 0..n {
        gpus.devices[r].bind_thread()?;
        let w = &weights_per_rank[r];
        let s = &scratch_per_rank[r];
        let gpu = &mut gpus.devices[r];
        match w.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&w.token_embd, &s.x, token, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&w.token_embd, &s.x, token, dim)?
            }
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&w.token_embd, &s.x, token, dim)?,
            EmbeddingFormat::F32 => gpu.embedding_lookup(&w.token_embd, &s.x, token, dim)?,
            other => {
                return Err(HipError::new(
                    0,
                    &format!("forward_ep: unsupported embedding format {other:?}"),
                ));
            }
        }
        gpu.hip.memcpy_htod(&s.pos_buf, &pos_i32.to_ne_bytes())?;
    }

    // 2. Per-layer EP program. Variant + delta-layer counter are replicated
    //    (sharding frees experts but never changes the layer variant), so rank 0
    //    is authoritative for both.
    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        let program = lower_variant(variant_of(&weights_per_rank[0].layers[layer_idx]));
        // Build the N per-rank bindings. `kv_per_rank.iter_mut()` yields the
        // disjoint `&mut KvCache` each binding needs; weights/scratch/dn are
        // shared `&`. This Vec is dropped at the end of the iteration, releasing
        // the mutable KV borrows before the next layer's `iter_mut`.
        let mut binds: Vec<Qwen35Bindings> = Vec::with_capacity(n);
        for (((w, s), kv), dn) in weights_per_rank
            .iter()
            .zip(scratch_per_rank.iter())
            .zip(kv_per_rank.iter_mut())
            .zip(dn_per_rank.iter())
        {
            binds.push(Qwen35Bindings {
                layer: &w.layers[layer_idx],
                s,
                config,
                kv_cache: kv,
                dn_state: dn,
                pos,
                layer_idx,
                delta_layer_idx,
                k_dim,
                v_dim,
                n_v_heads,
                hd,
                precomputed_attn_x_rot: false,
                fa_output_prerotated: false,
                defer_routed_combine: false,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(
            gpus,
            binds.as_mut_slice(),
            partials,
            &program,
            dim,
        )
        .map_err(|e| HipError::new(0, &e.to_string()))?;
        if matches!(
            &weights_per_rank[0].layers[layer_idx],
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
    }

    // 3. Final norm + lm_head on rank 0 (output_device). Logits → rank0 scratch.
    {
        gpus.devices[0].bind_thread()?;
        let w = &weights_per_rank[0];
        let s = &scratch_per_rank[0];
        let gpu = &mut gpus.devices[0];
        gpu.rmsnorm_f32(&s.x, &w.output_norm, &s.tmp, config.norm_eps)?;
        let ctx = DispatchCtx::new(gpu);
        let wr = w.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    // 4. Sync every rank — work ran on each device's active_stream, so a host
    //    download of rank 0's logits (on the null stream) would otherwise race.
    for r in 0..n {
        gpus.devices[r].bind_thread()?;
        gpus.devices[r].hip.device_synchronize()?;
    }
    Ok(())
}

/// EP (Ship 6 substrate-EP) **WMMA batched prefill** for qwen3.x-A3B (E6b).
///
/// The batched analog of [`forward_ep`]: processes all `tokens` as one batch
/// through the WMMA/grouped-GEMM prefill kernels (NOT token-by-token), replicated
/// across `gpus.devices.len()` EP ranks, with MoE experts sharded per rank.
///
/// Driven **layer-granularly** by calling [`forward_prefill_chunk`] with a
/// single-layer band per rank, because EP needs a per-MoE-layer all-reduce: the
/// next layer's replicated attention must read the FULL (cross-rank-summed)
/// residual. For each layer:
///   1. (MoE only) zero each rank's `[n × dim]` routed partial,
///   2. run the layer's batched chunk on every rank — the **shared** expert
///      accumulates into `pbs.x_batch` (replicated, added once per rank), the
///      **routed** combine into the zeroed partial (owned experts only; non-owned
///      read load-time zero-dummy → 0),
///   3. (MoE only) `all_reduce_sum_f32` the `[n × dim]` partials across ranks and
///      add into each rank's `pbs.x_batch`.
/// Non-MoE (dense DeltaNet / FullAttn) layers run replicated, no partial, no
/// all-reduce. Final norm + lm_head (last token) run on rank 0 → `scratch_per_rank[0].logits`.
///
/// **v1 constraints:** the whole prompt must fit one batch (`tokens.len() <=
/// pbs.max_batch`; no chunk loop yet) and KV must be a non-asym mode (q8/q4/…)
/// so no per-rank Givens replicas are needed (asym EP prefill = future work). The
/// per-layer chunk dispatch trades some launch overhead for the per-layer
/// all-reduce seam; a fused EP prefill layer loop is a later perf refinement.
///
/// Slices (`weights_per_rank`, `kv_per_rank`, `dn_per_rank`, `scratch_per_rank`,
/// `pbs_per_rank`, `partials`) must have length `gpus.devices.len()`; element `r`
/// lives on `gpus.devices[r]`. Each `partials[r]` must hold >= `n × dim` f32.
/// Every device must have an `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]).
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_ep(
    gpus: &mut Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_per_rank: &mut [llama::KvCache],
    dn_per_rank: &mut [DeltaNetState],
    scratch_per_rank: &[Qwen35Scratch],
    pbs_per_rank: &[PrefillBatchScratch],
    partials: &[GpuTensor],
) -> HipResult<()> {
    let n_rank = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: weights_per_rank len"
    );
    assert_eq!(
        kv_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: kv_per_rank len"
    );
    assert_eq!(
        dn_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: dn_per_rank len"
    );
    assert_eq!(
        scratch_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: scratch_per_rank len"
    );
    assert_eq!(
        pbs_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: pbs_per_rank len"
    );
    assert_eq!(
        partials.len(),
        n_rank,
        "forward_prefill_batch_ep: partials len"
    );

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let dim = config.dim;
    // Per-call contract: one window must fit max_batch. Long prompts are driven
    // by calling this repeatedly with advancing start_pos + persistent kv/dn
    // (KV + DeltaNet state accumulate in place across calls, identical to the
    // single-GPU chunk loop in forward_prefill_batch) — see ep_decode_parity's
    // chunked-prefill loop. That bounds the prefill scratch to chunk_size, so
    // context length is limited by the KV cache, not the activation buffers.
    assert!(
        n <= pbs_per_rank[0].max_batch,
        "forward_prefill_batch_ep: window ({n} toks) must fit one batch (max_batch={}); \
         caller chunks long prompts into max_batch-sized windows",
        pbs_per_rank[0].max_batch,
    );

    // Per-layer cumulative LA / FA counters (replicated → identical across ranks;
    // they index dn_state.s_matrices / kv_cache.k_gpu exactly like the band
    // offsets the PP driver threads). kv_layer_offset == fa_layer_offset.
    let mut delta_off = 0usize;
    let mut fa_off = 0usize;

    let ep_timing = hipfire_config::developer_var("HIPFIRE_EP_PREFILL_TIMING").is_ok();
    let ep_skip_ar = hipfire_config::developer_var("HIPFIRE_EP_SKIP_ALLREDUCE").is_ok(); // DIAGNOSTIC ONLY (wrong output)
                                                                                         // Peer-direct all-reduce (bypass RCCL): the routed-partial sum goes through
                                                                                         // Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms
                                                                                         // vs RCCL's ~40 ms/call on hiptrx (gfx1201, PCIe). DEFAULT ON; opt back to
                                                                                         // RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared
                                                                                         // with TP), lazily sized to the largest count seen.
    let ep_peer_ar =
        hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE").as_deref() != Ok("0");
    let mut t_chunk = 0.0f64;
    let mut t_ar = 0.0f64;
    let mut t_add = 0.0f64;
    for layer_idx in 0..config.n_layers {
        let is_moe = matches!(
            &weights_per_rank[0].layers[layer_idx],
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
        );

        // 1. Zero each rank's routed partial (on its active_stream, so it's
        //    ordered before the chunk's routed combine that writes into it).
        if is_moe {
            for r in 0..n_rank {
                gpus.devices[r].bind_thread()?;
                let stream = gpus.devices[r].active_stream.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        "forward_prefill_batch_ep: no active_stream (call ensure_rank_streams)",
                    )
                })?;
                gpus.devices[r]
                    .hip
                    .memset_async(&partials[r].buf, 0, n * dim * 4, stream)?;
            }
        }

        // 2. Run the layer's batched chunk on every rank (single-layer band).
        let t_c = std::time::Instant::now();
        for r in 0..n_rank {
            gpus.devices[r].bind_thread()?;
            let band = PrefillBandCtx {
                layer_start: layer_idx,
                layer_end: layer_idx + 1,
                delta_layer_offset: delta_off,
                kv_layer_offset: fa_off,
                is_first_band: layer_idx == 0,
                is_last_band: false, // final norm + lm_head done explicitly below
                // v1 EP prefill is q8/non-asym KV → no per-rank Givens replicas.
                givens_cos: None,
                givens_sin: None,
            };
            let routed_out = if is_moe { Some(&partials[r]) } else { None };
            forward_prefill_chunk(
                &mut gpus.devices[r],
                &weights_per_rank[r],
                config,
                tokens,
                start_pos,
                &mut kv_per_rank[r],
                &mut dn_per_rank[r],
                &scratch_per_rank[r],
                &pbs_per_rank[r],
                None,  // hidden_rb
                None,  // per_token_hidden_out
                None,  // gdn_tape
                0,     // tape_offset
                None,  // tree_verify
                false, // pre_uploaded
                Some(&band),
                None,  // mask_override
                false, // needs_last_token_logits (no lm_head in band)
                None,  // max_layer
                routed_out,
            )?;
        }

        if ep_timing {
            t_chunk += t_c.elapsed().as_secs_f64() * 1000.0;
        }

        // 3. All-reduce the routed partials, add into each rank's residual.
        if is_moe && !ep_skip_ar {
            let t_a = std::time::Instant::now();
            let refs: Vec<&hip_bridge::DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
            if ep_peer_ar {
                gpus.all_reduce_sum_f32_peer(&refs, n * dim)
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
            } else {
                gpus.all_reduce_sum_f32(&refs, n * dim)
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
            }
            if ep_timing {
                t_ar += t_a.elapsed().as_secs_f64() * 1000.0;
            }
            let t_d = std::time::Instant::now();
            for r in 0..n_rank {
                gpus.devices[r].bind_thread()?;
                let x_n = pbs_per_rank[r].x_batch.sub_offset(0, n * dim);
                let p_n = partials[r].sub_offset(0, n * dim);
                gpus.devices[r].add_inplace_f32(&x_n, &p_n)?;
            }
            if ep_timing {
                t_add += t_d.elapsed().as_secs_f64() * 1000.0;
            }
        }

        match config.layer_types[layer_idx] {
            LayerType::LinearAttention => delta_off += 1,
            LayerType::FullAttention => fa_off += 1,
        }
    }

    // Final norm + lm_head on rank 0 (last token) → scratch_per_rank[0].logits.
    // Done explicitly (not via the chunk) so it runs AFTER the last layer's
    // all-reduce — the last MoE layer's routed output is only in x_batch after
    // step 3, so an in-chunk lm_head would read an incomplete residual.
    {
        gpus.devices[0].bind_thread()?;
        let gpu = &mut gpus.devices[0];
        let w = &weights_per_rank[0];
        let s = &scratch_per_rank[0];
        let pbs = &pbs_per_rank[0];
        let last_x = pbs.x_batch.sub_offset((n - 1) * dim, dim);
        gpu.rmsnorm_f32(&last_x, &w.output_norm, &s.tmp, config.norm_eps)?;
        let ctx = DispatchCtx::new(gpu);
        let wr = w.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    // Sync every rank — work ran on active_streams; the host logits read on rank
    // 0 (null stream) would otherwise race.
    let t_s = std::time::Instant::now();
    for r in 0..n_rank {
        gpus.devices[r].bind_thread()?;
        gpus.devices[r].hip.device_synchronize()?;
    }
    if ep_timing {
        let t_sync = t_s.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "EP-PREFILL-TIMING (host ms): chunk-loop={t_chunk:.1} all_reduce={t_ar:.1} add={t_add:.1} final-sync={t_sync:.1}",
        );
    }
    Ok(())
}

/// Multi-GPU layer-loop dispatcher (Stage 5 of multi-GPU pp migration #58).
/// Mirrors `forward_scratch_layers` but routes per-layer work to
/// `gpus.devices[gpus.device_for_layer(i)]` and copies the residual
/// stream `s.x` across band boundaries via `Gpus::boundary_copy`.
/// Final `output_norm + lm_head` runs on `gpus.output_device`
/// (Variant 2 — no copy back to dev_0). Spec-decode `hidden_rb` is
/// not threaded — refused at load time when pp > 1.
fn forward_scratch_layers_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let _ = qkv_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let mut delta_layer_idx = 0usize;
    let mut prev_dev: Option<usize> = None;

    for layer_idx in 0..config.n_layers {
        let dev_idx = gpus.device_for_layer(layer_idx);

        if let Some(pd) = prev_dev {
            if dev_idx != pd {
                let src_buf = &scratch_set.per_device[pd].x.buf;
                let dst_buf = &scratch_set.per_device[dev_idx].x.buf;
                let evt = gpus.boundary_copy(pd, dev_idx, src_buf, dst_buf, dim * 4)?;
                gpus.wait_boundary(evt)?;
            }
        }

        {
            let s = &scratch_set.per_device[dev_idx];
            let givens_cos_dev = gpus.givens_cos_per_dev.get(dev_idx);
            let givens_sin_dev = gpus.givens_sin_per_dev.get(dev_idx);
            let gpu = &mut gpus.devices[dev_idx];

            // Resolve givens lazily — asym{2,3,4} branches use these,
            // others don't. Multi-GPU prefers the per-device replica
            // populated by the KV ctor; fall back to kv_cache.givens_*
            // for single-GPU shape compatibility (shouldn't fire in
            // pp > 1 since asym ctors always populate per-device).
            macro_rules! ct {
                () => {
                    givens_cos_dev.unwrap_or_else(|| kv_cache.givens_cos.as_ref().unwrap())
                };
            }
            macro_rules! st {
                () => {
                    givens_sin_dev.unwrap_or_else(|| kv_cache.givens_sin.as_ref().unwrap())
                };
            }

            match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
                (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wqkv.gpu_dtype;
                    let la4_same_dtype = layer.wz.gpu_dtype == dt
                        && layer.w_beta.gpu_dtype == dt
                        && layer.w_alpha.gpu_dtype == dt;
                    let fused_la4_mq4 =
                        la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_la4_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_hfq4g256(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_mq3g256_lloyd(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                        weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    }
                    gpu.fused_sigmoid_alpha_gate_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        &layer.dt_bias,
                        &layer.a_log,
                        n_v_heads,
                    )?;
                    gpu.conv1d_silu_split_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_q,
                            &s.dn_k,
                            config.linear_num_key_heads,
                            ratio,
                            hd,
                        )?;
                    } else {
                        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                    }
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    gpu.gated_norm_f32(
                        &s.dn_attn_out,
                        &s.dn_z,
                        &layer.norm_weight,
                        &s.dn_normed,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                    )?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.dn_normed),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.w_gate,
                        &s.x,
                        &layer.ffn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt_g = layer.w_gate.gpu_dtype;
                    let same_dtype = layer.w_up.gpu_dtype == dt_g;
                    let fused_gu_mq4 =
                        same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                    if fused_gu_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_hfq4g256(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else if fused_gu_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_mq3g256_lloyd(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;

                        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                    }
                    weight_gemv_swiglu_residual(
                        gpu,
                        &layer.w_down,
                        &s.gate_ffn,
                        &s.up,
                        &s.ffn_hidden,
                        &s.x,
                    )?;
                    delta_layer_idx += 1;
                }

                (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wq,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wq.gpu_dtype;
                    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                    let fused_fa3_mq4 =
                        fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_fa3_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_hfq4g256(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else if fused_fa3_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_mq3g256_lloyd(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;

                        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    }
                    gpu.deinterleave_f32(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;

                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                    if kv_cache.compact_offset > 0 {
                        let phys = pos as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                    }

                    if kv_cache.quant_asym4 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_asym3 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_asym2 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_q8 {
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        let use_flash = gpu.graphs.capture_mode
                            || s.flash_mode == 2
                            || (s.flash_mode == 1 && pos + 1 >= 2048)
                            || pos + 1 > 15000;
                        if use_flash {
                            gpu.attention_flash_q8_0(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        } else {
                            gpu.attention_q8_0_kv(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                    } else {
                        gpu.kv_cache_write(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.kv_cache_write(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.attention_f32(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                        )?;
                    }

                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.fa_attn_out),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.w_gate,
                        &s.x,
                        &layer.ffn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt_g = layer.w_gate.gpu_dtype;
                    let same_dtype = layer.w_up.gpu_dtype == dt_g;
                    let fused_gu_mq4 =
                        same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                    if fused_gu_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_hfq4g256(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else if fused_gu_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_mq3g256_lloyd(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;

                        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                    }
                    weight_gemv_swiglu_residual(
                        gpu,
                        &layer.w_down,
                        &s.gate_ffn,
                        &s.up,
                        &s.ffn_hidden,
                        &s.x,
                    )?;
                }

                (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wqkv.gpu_dtype;
                    let la4_same_dtype = layer.wz.gpu_dtype == dt
                        && layer.w_beta.gpu_dtype == dt
                        && layer.w_alpha.gpu_dtype == dt;
                    let fused_la4_mq4 =
                        la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_la4_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_hfq4g256(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_mq3g256_lloyd(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                        weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    }
                    gpu.fused_sigmoid_alpha_gate_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        &layer.dt_bias,
                        &layer.a_log,
                        n_v_heads,
                    )?;
                    gpu.conv1d_silu_split_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_q,
                            &s.dn_k,
                            config.linear_num_key_heads,
                            ratio,
                            hd,
                        )?;
                    } else {
                        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                    }
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    gpu.gated_norm_f32(
                        &s.dn_attn_out,
                        &s.dn_z,
                        &layer.norm_weight,
                        &s.dn_normed,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                    )?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.dn_normed),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    if ffn_all_mq4_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu, &layer.ffn, &s.x, &s.x, config, s,
                        )?;
                    } else {
                        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                        moe_ffn_decode_with_scratch(gpu, &layer.ffn, &s.tmp, &s.x, config, s)?;
                    }
                    delta_layer_idx += 1;
                }

                (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wq,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wq.gpu_dtype;
                    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                    let fused_fa3_mq4 =
                        fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_fa3_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_hfq4g256(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else if fused_fa3_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_mq3g256_lloyd(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;

                        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    }
                    gpu.deinterleave_f32(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;

                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                    if kv_cache.compact_offset > 0 {
                        let phys = pos as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                    }

                    if kv_cache.quant_asym4 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_asym3 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_asym2 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_q8 {
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        let use_flash = gpu.graphs.capture_mode
                            || s.flash_mode == 2
                            || (s.flash_mode == 1 && pos + 1 >= 2048)
                            || pos + 1 > 15000;
                        if use_flash {
                            gpu.attention_flash_q8_0(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        } else {
                            gpu.attention_q8_0_kv(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                    } else {
                        gpu.kv_cache_write(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.kv_cache_write(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.attention_f32(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                        )?;
                    }

                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.fa_attn_out),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    if ffn_all_mq4_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu, &layer.ffn, &s.x, &s.x, config, s,
                        )?;
                    } else {
                        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                        moe_ffn_decode_with_scratch(gpu, &layer.ffn, &s.tmp, &s.x, config, s)?;
                    }
                }

                _ => panic!("layer type mismatch at layer {layer_idx}"),
            }
        }

        prev_dev = Some(dev_idx);
    }

    let dev_last = gpus.output_device;
    let s_last = &scratch_set.per_device[dev_last];
    let gpu_last = &mut gpus.devices[dev_last];
    gpu_last.rmsnorm_f32(
        &s_last.x,
        &weights.output_norm,
        &s_last.tmp,
        config.norm_eps,
    )?;
    {
        let ctx = DispatchCtx::new(gpu_last);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s_last.tmp),
            out: &s_last.logits,
        };
        execute_steps(gpu_last, &ctx, &[step])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    Ok(())
}

/// Multi-GPU decode forward (Stage 5 of multi-GPU pp migration #58).
/// Embedding lookup on dev 0 (token_embd lives there per Stage 4 placement),
/// then the layer loop via `forward_scratch_layers_multi`. `s.logits` ends
/// up on `gpus.output_device`. hipGraph capture is bypassed for pp > 1.
pub fn forward_scratch_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    // F3 (review): asym{2,3,4} KV requires per-device givens replicas. The
    // ct!()/st!() macros in forward_scratch_layers_multi fall back to
    // kv_cache.givens_* if the per-device replica is None — which silently
    // hands a wrong-device tensor to attention kernels. Refuse up-front.
    if (kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && (gpus.givens_cos_per_dev.len() != gpus.devices.len()
            || gpus.givens_sin_per_dev.len() != gpus.devices.len())
    {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_scratch_multi: asym KV mode requires gpus.givens_*_per_dev \
             populated for every device. Construct KvCache via the *_multi ctor \
             (e.g. KvCache::new_gpu_asym3_capped_multi) — single-GPU ctors leave \
             gpus.givens_*_per_dev empty.",
        ));
    }

    let dim = config.dim;
    let pos_bytes = (pos as i32).to_ne_bytes();
    {
        let gpu0 = &mut gpus.devices[0];
        let s0 = &scratch_set.per_device[0];
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu0.embedding_lookup_hfq4g256(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu0.embedding_lookup_hfq4g128(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::Q8_0 => {
                gpu0.embedding_lookup_q8(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::F32 => {
                gpu0.embedding_lookup(&weights.token_embd, &s0.x, token, dim)?
            }
            _ => panic!("unsupported embedding format"),
        }
    }
    // pos_buf written to every device's scratch — every band reads it inside
    // RoPE / KV write for FullAttention layers. F1 (review): bind_thread
    // before each raw gpu.hip.memcpy_htod — HipRuntime methods bypass the
    // Stage 2b bind audit, so without explicit bind the writes land on
    // whatever device was last bound (dev 0 from the embedding lookup above).
    for dev_idx in 0..gpus.devices.len() {
        let gpu = &mut gpus.devices[dev_idx];
        gpu.bind_thread()?;
        let s = &scratch_set.per_device[dev_idx];
        gpu.hip.memcpy_htod(&s.pos_buf, &pos_bytes)?;
    }
    forward_scratch_layers_multi(gpus, weights, config, pos, kv_cache, dn_state, scratch_set)
}

/// Multi-GPU batched prefill (Stage 6 of #58 — multi-gpu pipeline-parallel).
/// Closes the daemon-time pp=1 vs pp=2 divergence — single-GPU
/// `forward_prefill_batch` runs through the WMMA-batched fast path, while
/// pp=2 was previously stuck on per-token `forward_scratch_multi` (a
/// different kernel sequence with a different reduction order). This
/// routes both paths through the same `forward_prefill_chunk` body, just
/// band-restricted via `PrefillBandCtx`.
///
/// Flow per chunk of up to `max_batch` tokens:
///   1. Allocate per-band `PrefillBatchScratch` on each device's pbs.
///   2. Run `forward_prefill_chunk` on dev 0 with band 0 layers,
///      `is_first_band=true` (does the embedding) and
///      `is_last_band=(n_bands==1)`.
///   3. peer-copy band 0's `pbs.x_batch` into band 1's `pbs.x_batch`.
///   4. Run `forward_prefill_chunk` on dev 1 with band 1 layers,
///      `is_first_band=false` (skips embedding, reads already-populated
///      `x_batch`) and `is_last_band=true` (does final norm + lm_head).
///   5. Repeat for any further bands.
///
/// `tree_verify`, DFlash hidden-rb, GdnTape, and per_token_hidden_out
/// are pp=1 only in v1. They've been refused at the daemon load-time
/// gate, so this function does not accept them as parameters.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    let n_total = tokens.len();
    if n_total == 0 {
        return Ok(());
    }

    let n_bands = gpus.devices.len();
    if n_bands == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_multi: no devices",
        ));
    }

    // F3 (review-pattern from forward_scratch_multi): asym{2,3,4} KV requires
    // per-device givens replicas. Refuse up-front — the band-mode macros in
    // forward_prefill_chunk fall back to kv_cache.givens_* if the band's
    // givens override is None, which silently hands a wrong-device tensor
    // to attention kernels.
    if (kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && (gpus.givens_cos_per_dev.len() != n_bands || gpus.givens_sin_per_dev.len() != n_bands)
    {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_multi: asym KV mode requires gpus.givens_*_per_dev \
             populated for every device. Construct KvCache via the *_multi ctor \
             (e.g. KvCache::new_gpu_asym3_capped_multi) — single-GPU ctors leave \
             gpus.givens_*_per_dev empty.",
        ));
    }

    let max_batch: usize = hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(PREFILL_MAX_BATCH);

    let force_fallback = !hipfire_runtime::config::get().prefill_batched;

    // Eligibility: same checks as `forward_prefill_batch_with_pbs`. If any
    // layer fails the batched gate, fall back to per-token forward —
    // correctness preserved at the cost of per-token kernel sequence.
    let arch0 = gpus.devices[0].arch.as_str();
    let moe_topk_ok = config.num_experts_per_tok == 8 && config.num_experts <= 1024;
    let eligible = !force_fallback
        && n_total >= 2
        && dn_state.quant == StateQuant::Q8
        && weights
            .layers
            .iter()
            .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),))
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::DeltaNet(l) => {
                is_batchable_la(l.wqkv.gpu_dtype, arch0)
                    && is_batchable_la(l.wz.gpu_dtype, arch0)
                    && is_batchable_la(l.w_beta.gpu_dtype, arch0)
                    && is_batchable_la(l.w_alpha.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch0)
                    && is_batchable_la(l.w_up.gpu_dtype, arch0)
                    && is_batchable_la(l.w_down.gpu_dtype, arch0)
            }
            LayerWeights::FullAttn(l) => {
                is_batchable_la(l.wq.gpu_dtype, arch0)
                    && is_batchable_la(l.wk.gpu_dtype, arch0)
                    && is_batchable_la(l.wv.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch0)
                    && is_batchable_la(l.w_up.gpu_dtype, arch0)
                    && is_batchable_la(l.w_down.gpu_dtype, arch0)
            }
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => moe_topk_ok,
        });

    if !eligible {
        // Per-token fallback. Correctness over speed when the batched
        // path's preconditions are not met.
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch_multi(
                gpus,
                weights,
                config,
                tok,
                start_pos + i,
                kv_cache,
                dn_state,
                scratch_set,
            )?;
        }
        return Ok(());
    }

    // Per-band cumulative offsets into LA / FA layer indices. The band's
    // first layer of a given type (DeltaNet or FullAttn) reads
    // `dn_state.s_matrices[delta_off]` / `kv_cache.k_caches[fa_off]`.
    let mut delta_off_per_band = vec![0usize; n_bands];
    let mut fa_off_per_band = vec![0usize; n_bands];
    {
        let mut delta_run = 0usize;
        let mut fa_run = 0usize;
        for b in 0..n_bands {
            delta_off_per_band[b] = delta_run;
            fa_off_per_band[b] = fa_run;
            let band_start = gpus.band_starts[b];
            let band_end = if b + 1 < n_bands {
                gpus.band_starts[b + 1]
            } else {
                config.n_layers
            };
            for li in band_start..band_end {
                match config.layer_types[li] {
                    LayerType::LinearAttention => delta_run += 1,
                    LayerType::FullAttention => fa_run += 1,
                }
            }
        }
    }

    // Allocate one PrefillBatchScratch per band. Each lives on the band's
    // device. Freed at the end of the call (matches forward_prefill_batch's
    // own_pbs pattern). Future opt: cache on Qwen35ScratchSet.
    let mut pbs_per_band: Vec<PrefillBatchScratch> = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        // hunt3 H-E: PrefillBatchScratch has no Drop impl, so a mid-loop OOM
        // here would silently leak every already-allocated band's ~40 GpuTensors
        // (incl. tens-of-MB MoE grouped-GEMM scratch). On the first failing
        // PrefillBatchScratch::new, free the bands pushed so far on their own
        // devices before propagating the error. Mirrors the single-GPU own_pbs
        // cleanup pattern (allocation failure must not leak prior allocations).
        // The intra-`new` partial-literal leak (a `?` failing partway through
        // the struct literal) is handled inside PrefillBatchScratch::new itself
        // via its alloc ledger, so the failing band's own allocations are also
        // freed before its error reaches here.
        let alloc = {
            let g = &mut gpus.devices[b];
            g.bind_thread()
                .and_then(|()| PrefillBatchScratch::new(g, config, max_batch))
        };
        match alloc {
            Ok(pbs) => pbs_per_band.push(pbs),
            Err(e) => {
                for (prev_b, prev_pbs) in pbs_per_band.into_iter().enumerate() {
                    let pg = &mut gpus.devices[prev_b];
                    let _ = pg.bind_thread();
                    prev_pbs.free_gpu(pg);
                }
                return Err(e);
            }
        }
    }

    let dim = config.dim;
    let dim_row_bytes = dim * 4;

    let result = (|| -> HipResult<()> {
        let mut chunk_start = 0usize;
        while chunk_start < n_total {
            let chunk_end = (chunk_start + max_batch).min(n_total);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_n = chunk.len();

            for b in 0..n_bands {
                let band_layer_start = gpus.band_starts[b];
                let band_layer_end = if b + 1 < n_bands {
                    gpus.band_starts[b + 1]
                } else {
                    config.n_layers
                };
                let givens_cos = gpus.givens_cos_per_dev.get(b);
                let givens_sin = gpus.givens_sin_per_dev.get(b);
                let band_ctx = PrefillBandCtx {
                    layer_start: band_layer_start,
                    layer_end: band_layer_end,
                    delta_layer_offset: delta_off_per_band[b],
                    kv_layer_offset: fa_off_per_band[b],
                    is_first_band: b == 0,
                    is_last_band: b + 1 == n_bands,
                    givens_cos,
                    givens_sin,
                };
                {
                    let pbs_b: &PrefillBatchScratch = &pbs_per_band[b];
                    let s_b = &scratch_set.per_device[b];
                    let g_b = &mut gpus.devices[b];
                    forward_prefill_chunk(
                        g_b,
                        weights,
                        config,
                        chunk,
                        start_pos + chunk_start,
                        kv_cache,
                        dn_state,
                        s_b,
                        pbs_b,
                        None, // hidden_rb: pp=1 only
                        None, // per_token_hidden_out: pp=1 only
                        None, // gdn_tape: pp=1 only
                        0,
                        None,  // tree_verify: pp=1 only
                        false, // pre_uploaded
                        Some(&band_ctx),
                        None, // mask_override: multi-GPU PP path doesn't use the MTP probe hook
                        true, // needs_last_token_logits: preserve multi-GPU post-condition
                        None, // max_layer: multi-GPU PP path runs full stack
                        None, // routed_out: PP bands are multi-layer, not EP
                    )?;
                }

                if b + 1 < n_bands {
                    // Hand off the chunk's residual stream to the next band.
                    // pbs.x_batch holds [N × dim] f32 — copy `chunk_n` rows
                    // from band b to band b+1. wait_boundary makes the dst
                    // device wait on the copy's completion event before the
                    // next forward_prefill_chunk dispatch reads x_batch.
                    let copy_bytes = chunk_n * dim_row_bytes;
                    let (left, right) = pbs_per_band.split_at(b + 1);
                    let pbs_src = &left[b];
                    let pbs_dst = &right[0];
                    let evt = gpus.boundary_copy(
                        b,
                        b + 1,
                        &pbs_src.x_batch.buf,
                        &pbs_dst.x_batch.buf,
                        copy_bytes,
                    )?;
                    gpus.wait_boundary(evt)?;
                }
            }

            chunk_start = chunk_end;
        }
        Ok(())
    })();

    for (b, pbs) in pbs_per_band.into_iter().enumerate() {
        let g = &mut gpus.devices[b];
        let _ = g.bind_thread();
        pbs.free_gpu(g);
    }

    result
}

/// Forward pass returning logits ON GPU (no download). Caller must free the tensor.
/// Use with gpu.sample_top_p() after applying CPU-side n-gram blocking via download/modify/upload.
pub fn forward_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }
    forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Run one step with a pre-computed embedding vector (for VL visual token injection).
/// embedding_data: [dim] F32 values on CPU — uploaded to GPU as the initial hidden state.
pub fn forward_with_embedding(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let x = gpu.upload_f32(embedding_data, &[config.dim])?;
    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_rot_covers_deltanet_value_width_for_moe_configs() {
        assert_eq!(qwen35_x_rot_len(2048, 0, 4096), 4096);
        assert_eq!(qwen35_x_rot_len(2048, 8192, 4096), 8192);
    }

    // ── SP2 — per-expert mixed-tier table builder (CPU-pure) ──────────────
    // `mixed_tier_table` is the testable core of `per_expert_tier_tables`:
    // empty/uniform columns collapse to None (uniform fast path), only a
    // genuinely multi-tier column yields Some(table).
    #[test]
    fn mixed_tier_table_empty_is_none() {
        // Paged mode: no resident experts → uniform fast path.
        assert_eq!(mixed_tier_table(Vec::new()), None);
    }

    #[test]
    fn mixed_tier_table_uniform_is_none() {
        // The common case: every expert one tier → None → byte-identical
        // uniform path, no allocation surfaced to MoeDtypes.
        let tiers = vec![DType::MQ4G256; 4];
        assert_eq!(mixed_tier_table(tiers), None);
        // Single-expert uniform column is also None.
        assert_eq!(mixed_tier_table(vec![DType::MQ6G256]), None);
    }

    #[test]
    fn mixed_tier_table_mixed_is_some_preserving_order() {
        // A re-quant overlay bumped experts 1 and 3 to MQ6 → Some, and the
        // table preserves per-expert order/dtype so dispatch buckets correctly.
        let tiers = vec![
            DType::MQ4G256,
            DType::MQ6G256,
            DType::MQ4G256,
            DType::MQ6G256,
        ];
        assert_eq!(mixed_tier_table(tiers.clone()), Some(tiers));
    }

    #[test]
    fn mixed_tier_table_mixed_first_differs() {
        // Guard against an off-by-one where only expert[0] is compared:
        // here every later expert differs from expert[0].
        let tiers = vec![DType::MQ4G256, DType::MQ6G256, DType::MQ6G256];
        assert_eq!(mixed_tier_table(tiers.clone()), Some(tiers));
    }

    // ── N4 config-parser collapse: serde RawQwen35Config + finalize ──────
    // Oracle for the ×2 collapse (config_from_hfq vs config_from_safetensors)
    // and the serde port. Fixtures are CPU-pure (no GPU). Expected values are
    // transcribed from the field contract the OLD hand-walked parsers produced.

    /// Wrap an inner `config` blob in the metadata_json envelope both sources
    /// build (`{architecture, config:{...}}`, see safetensors_source.rs).
    fn envelope(inner: serde_json::Value) -> String {
        serde_json::json!({ "architecture": "qwen35", "config": inner }).to_string()
    }

    /// A realistic dense Qwen3.5 inner config with the linear/mrope/rope_parameters
    /// fields populated.
    fn dense_inner() -> serde_json::Value {
        serde_json::json!({
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 151936,
            "intermediate_size": 3584,
            "rms_norm_eps": 1e-5,
            "eos_token_id": 151645,
            "rope_parameters": {
                "rope_theta": 5000000.0,
                "mrope_interleaved": true,
                "mrope_section": [12, 13, 14]
            },
            "partial_rotary_factor": 0.5,
            "linear_num_key_heads": 32,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 3,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "norm_topk_prob": false
        })
    }

    #[test]
    fn dense_fixture_every_field() {
        let cfg = from_config_value(&dense_inner()).expect("dense parse");
        assert_eq!(cfg.dim, 2048);
        assert_eq!(cfg.n_layers, 4);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.norm_eps, 1e-5);
        assert_eq!(cfg.eos_token, 151645);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.rope_theta, 5_000_000.0);
        // FLAT partial_rotary_factor wins over nested.
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        assert!(!cfg.is_vl_text);
        assert!(cfg.mrope_interleaved);
        assert_eq!(cfg.mrope_section, [12, 13, 14]);
        assert_eq!(cfg.linear_num_key_heads, 32);
        assert_eq!(cfg.linear_num_value_heads, 32);
        assert_eq!(cfg.linear_key_head_dim, 64);
        assert_eq!(cfg.linear_value_head_dim, 64);
        assert_eq!(cfg.conv_kernel_dim, 3);
        assert_eq!(cfg.hidden_dim, 3584);
        assert_eq!(
            cfg.layer_types,
            vec![
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ]
        );
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.num_experts_per_tok, 0);
        assert_eq!(cfg.moe_intermediate_size, 0);
        assert_eq!(cfg.shared_expert_intermediate_size, 0);
        assert!(!cfg.has_shared_expert);
        assert!(!cfg.norm_topk_prob);
        assert!(!cfg.paged_experts);
        assert_eq!(cfg.vram_budget_bytes, u64::MAX);
    }

    #[test]
    fn defaults_when_optional_absent() {
        // Minimal config: only the four required fields. Everything else defaults.
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000
        });
        let cfg = from_config_value(&inner).expect("minimal parse");
        assert_eq!(cfg.n_kv_heads, 8); // defaults to n_heads
        assert_eq!(cfg.head_dim, 1024 / 8); // dim / n_heads
        assert_eq!(cfg.hidden_dim, 0);
        assert_eq!(cfg.norm_eps, 1e-6);
        assert_eq!(cfg.eos_token, 248044);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert_eq!(cfg.partial_rotary_factor, 0.25);
        assert!(!cfg.mrope_interleaved);
        assert_eq!(cfg.mrope_section, [11, 11, 10]);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.conv_kernel_dim, 4);
        // norm_topk_prob defaults to true.
        assert!(cfg.norm_topk_prob);
        // layer_types absent → all FullAttention, length n_layers.
        assert_eq!(cfg.layer_types, vec![LayerType::FullAttention; 2]);
    }

    #[test]
    fn array_eos_token_id_uses_first_element() {
        // Real Qwen3.5 / chat checkpoints ship `eos_token_id` as an array. The
        // OLD hand-walked parser silently fell back to the default on an array;
        // the serde port (scalar u32) would HARD-ERROR. We now take the first
        // element (uniform with qwen2's `eos_token_id = eos_token_ids[0]`).
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "eos_token_id": [151645, 151643]
        });
        let cfg = from_config_value(&inner).expect("array-eos parse");
        assert_eq!(cfg.eos_token, 151645);
    }

    #[test]
    fn collapse_hfq_eq_safetensors() {
        // ×2-collapse proof at the wrapper boundary: both `config_from_hfq` and
        // `config_from_safetensors` now delegate to `config_from_metadata_json`,
        // so exercising the string→config path that both share — and confirming
        // it matches the underlying `from_config_value` on the `config` node —
        // covers the collapse end-to-end (not just `from_config_value`
        // determinism).
        let env = envelope(dense_inner());

        // Shared wrapper path: full envelope string → config.
        let via_wrapper = config_from_metadata_json(&env).expect("metadata_json parse");

        // Oracle: parse the envelope and run from_config_value on meta["config"]
        // directly. Qwen35Config has no PartialEq, so assert the key fields.
        let parsed: serde_json::Value = serde_json::from_str(&env).unwrap();
        let via_inner = from_config_value(parsed.get("config").unwrap()).unwrap();

        assert_eq!(via_wrapper.dim, via_inner.dim);
        assert_eq!(via_wrapper.n_layers, via_inner.n_layers);
        assert_eq!(via_wrapper.n_heads, via_inner.n_heads);
        assert_eq!(via_wrapper.head_dim, via_inner.head_dim);
        assert_eq!(via_wrapper.rope_theta, via_inner.rope_theta);
        assert_eq!(
            via_wrapper.partial_rotary_factor,
            via_inner.partial_rotary_factor
        );
        assert_eq!(via_wrapper.mrope_section, via_inner.mrope_section);
        assert_eq!(via_wrapper.layer_types, via_inner.layer_types);
        assert_eq!(via_wrapper.num_experts, via_inner.num_experts);
        assert_eq!(via_wrapper.is_vl_text, via_inner.is_vl_text);
    }

    #[test]
    fn moe_fixture() {
        let inner = serde_json::json!({
            "hidden_size": 2048,
            "num_hidden_layers": 3,
            "num_attention_heads": 16,
            "vocab_size": 151936,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "layer_types": ["linear_attention", "full_attention", "linear_attention"]
        });
        let cfg = from_config_value(&inner).expect("moe parse");
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 512);
        assert!(cfg.has_shared_expert);
        assert_eq!(
            cfg.layer_types,
            vec![
                LayerType::LinearAttention,
                LayerType::FullAttention,
                LayerType::LinearAttention,
            ]
        );
    }

    #[test]
    fn missing_required_is_err() {
        // No hidden_size → serde hard-error.
        let inner = serde_json::json!({
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000
        });
        assert!(from_config_value(&inner).is_err());
    }

    #[test]
    fn rope_nested_partial_rotary_when_no_flat() {
        // No flat partial_rotary_factor → falls back to nested rope_parameters.
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "rope_parameters": { "partial_rotary_factor": 0.75 }
        });
        let cfg = from_config_value(&inner).expect("parse");
        assert_eq!(cfg.partial_rotary_factor, 0.75);
    }

    #[test]
    fn mrope_section_partial_fill() {
        // Array shorter than 3 fills leading slots, keeps defaults for the rest.
        // Non-u64 elements keep that slot's default.
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "rope_parameters": { "mrope_section": [20, "oops"] }
        });
        let cfg = from_config_value(&inner).expect("parse");
        // slot 0 ← 20, slot 1 non-u64 keeps default 11, slot 2 absent keeps 10.
        assert_eq!(cfg.mrope_section, [20, 11, 10]);
    }

    #[test]
    fn is_vl_text_true_when_vision_config_present() {
        // BOTH text_config AND vision_config on the OUTER config node.
        let outer = serde_json::json!({
            "vision_config": { "depth": 32 },
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "vocab_size": 151936
            }
        });
        let cfg = from_config_value(&outer).expect("vl parse");
        assert!(cfg.is_vl_text);
        // descended into text_config for the shape.
        assert_eq!(cfg.dim, 2048);
        assert_eq!(cfg.vocab_size, 151936);
    }

    // ── #397 Ship 6 — lowered decode super-op program shapes ──────────────
    // The lowered LayerProgram per variant must mirror the hand-arm op sequence
    // in forward_scratch_layers exactly. These are CPU-pure (no GPU/GpuTensor).
    #[test]
    fn lowered_fullattn_program_shape() {
        use SuperOpKind::{Attend, Proj, ResidualGemv};
        let p = lower_variant(Q35Variant::FullAttn);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Proj, Attend, ResidualGemv, Proj, ResidualGemv]);
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKV);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_FULL);
        assert_eq!(p[2].binding.weights[0].0, q35_op::RESID_WO);
        assert_eq!(p[3].binding.weights[0].0, q35_op::PROJ_GATE_UP);
        assert_eq!(p[4].binding.weights[0].0, q35_op::RESID_DOWN_SWIGLU);
    }

    #[test]
    fn lowered_deltanet_program_shape() {
        use SuperOpKind::{Attend, Norm, Proj, Recurrent, ResidualGemv};
        let p = lower_variant(Q35Variant::DeltaNet);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Proj,
                Attend,
                Recurrent,
                Norm,
                ResidualGemv,
                Proj,
                ResidualGemv
            ]
        );
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKVZA);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_DN_PREP);
    }

    #[test]
    fn lowered_moe_variants_replace_dense_ffn_with_one_moe_op() {
        use SuperOpKind::Moe;
        let dn = lower_variant(Q35Variant::DeltaNetMoe);
        let fa = lower_variant(Q35Variant::FullAttnMoe);
        // MoE variants end in a single Moe super-op (no dense gate_up/down).
        assert_eq!(dn.last().unwrap().kind, Moe);
        assert_eq!(fa.last().unwrap().kind, Moe);
        assert!(
            dn.iter()
                .all(|o| o.binding.weights[0].0 != q35_op::PROJ_GATE_UP
                    || o.kind != SuperOpKind::Proj)
        );
        // FullAttnMoe is the shortest: Proj, Attend, ResidualGemv(wo), Moe.
        assert_eq!(fa.len(), 4);
        assert_eq!(dn.len(), 6);
    }

    #[test]
    fn lowered_variant_of_maps_layer_discriminant() {
        // variant_of is a thin discriminant map; assert the program lengths it
        // would produce per the documented layer shapes.
        assert_eq!(lower_variant(Q35Variant::FullAttn).len(), 5);
        assert_eq!(lower_variant(Q35Variant::DeltaNet).len(), 7);
        assert_eq!(lower_variant(Q35Variant::DeltaNetMoe).len(), 6);
        assert_eq!(lower_variant(Q35Variant::FullAttnMoe).len(), 4);
    }

    #[test]
    fn f16_lm_head_mode_defaults_to_native() {
        assert_eq!(parse_f16_lm_head_mode(None), F16LmHeadMode::Native);
        assert_eq!(parse_f16_lm_head_mode(Some("auto")), F16LmHeadMode::Native);
        assert_eq!(parse_f16_lm_head_mode(Some("1")), F16LmHeadMode::Native);
        assert_eq!(
            parse_f16_lm_head_mode(Some("native")),
            F16LmHeadMode::Native
        );
        assert_eq!(parse_f16_lm_head_mode(Some("f16")), F16LmHeadMode::Native);
    }

    #[test]
    fn f16_lm_head_mode_allows_legacy_f32() {
        assert_eq!(parse_f16_lm_head_mode(Some("0")), F16LmHeadMode::F32);
        assert_eq!(parse_f16_lm_head_mode(Some("f32")), F16LmHeadMode::F32);
        assert_eq!(parse_f16_lm_head_mode(Some("fp32")), F16LmHeadMode::F32);
        assert_eq!(parse_f16_lm_head_mode(Some("legacy")), F16LmHeadMode::F32);
    }

    #[test]
    fn f16_lm_head_mode_unknown_falls_back_to_native() {
        assert_eq!(
            parse_f16_lm_head_mode(Some("surprise")),
            F16LmHeadMode::Native
        );
    }

    #[test]
    fn paro_batched_admit_defaults_off_and_allows_opt_in() {
        // PARO batched prefill is default-OFF (the path has a coherence/echo bug;
        // per-token fallback is correct) — opt in via HIPFIRE_PARO_BATCHED=1.
        // `paro_batched_admit_enabled_from_env` is `value == Some("1")`, so only
        // the exact string "1" enables it; everything else (incl. None) is off.
        assert!(!paro_batched_admit_enabled_from_env(None));
        assert!(paro_batched_admit_enabled_from_env(Some("1")));
        assert!(!paro_batched_admit_enabled_from_env(Some("surprise")));
        assert!(!paro_batched_admit_enabled_from_env(Some("0")));
    }

    // ── Qwen3.5 dispatch: is_batchable_la ────────────────────────

    /// The Qwen3.5-specific copy admits more dtypes than the runtime copy
    /// (ParoQ4G128, F32, Lloyd variants).

    const BATCHABLE_ARCHS: &[&str] = &[
        "gfx900", "gfx906", "gfx908", "gfx940", "gfx941", "gfx942", "gfx1010", "gfx1011",
        "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102",
        "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
    ];

    const WMMA_ARCHS: &[&str] = &[
        "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
        "gfx1201",
    ];

    const GFX10_SCALAR_ARCHS: &[&str] = &[
        "gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032",
    ];

    const NO_WMMA_ARCHS: &[&str] = &["gfx900", "gfx906", "gfx908", "gfx940", "gfx941", "gfx942"];

    #[test]
    fn qwen35_is_batchable_la_always_ok() {
        for &arch in BATCHABLE_ARCHS {
            assert!(
                is_batchable_la(DType::MQ4G256, arch),
                "MQ4G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ4G256, arch),
                "HFQ4G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MQ6G256, arch),
                "MQ6G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ6G256, arch),
                "HFQ6G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::Q8_0, arch),
                "Q8_0 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::ParoQ4G128, arch),
                "ParoQ4G128 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::F32, arch),
                "F32 should batch on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_mq3_wmma_and_gfx10_scalar() {
        for &arch in WMMA_ARCHS {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 should batch on {arch} (WMMA)"
            );
        }
        for &arch in GFX10_SCALAR_ARCHS {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 should batch on {arch} (scalar)"
            );
        }
        for &arch in NO_WMMA_ARCHS {
            assert!(
                !is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_fp4_only_on_wmma() {
        for &arch in WMMA_ARCHS {
            assert!(
                is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 should batch on {arch}"
            );
        }
        for &arch in NO_WMMA_ARCHS {
            assert!(
                !is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 must fall back on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_lloyd_mq3_only_on_gfx11_with_opt_in_gfx12() {
        // MQ3-Lloyd admits gfx1100/1101/1102/1150/1151 — the MQ3-Lloyd GEMM
        // source selectors DO ship a gfx1150 kernel.
        for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"] {
            assert!(
                is_batchable_la(DType::MQ3G256Lloyd, arch),
                "MQ3G256Lloyd should batch on {arch}"
            );
        }
        // MQ4-Lloyd admits gfx1100/1101/1102/1151 ONLY (NOT gfx1150). ANTIBLEED
        // admit-vs-select fix: the MQ4-Lloyd GEMM source selectors panic on
        // gfx1150 (no kernel), so admitting it upstream would crash at lookup.
        for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1151"] {
            assert!(
                is_batchable_la(DType::MQ4G256Lloyd, arch),
                "MQ4G256Lloyd should batch on {arch}"
            );
        }
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1150"),
            "gfx1150 must NOT admit Lloyd MQ4 (no MQ4-Lloyd kernel source → panic)"
        );
        // gfx1152 not in either admit list
        assert!(
            !is_batchable_la(DType::MQ3G256Lloyd, "gfx1152"),
            "gfx1152 should NOT admit Lloyd MQ3"
        );
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1152"),
            "gfx1152 should NOT admit Lloyd MQ4"
        );
        // gfx12 requires env gate
        assert!(
            !is_batchable_la(DType::MQ3G256Lloyd, "gfx1200"),
            "gfx1200 without HIPFIRE_LLOYD_GFX12=1"
        );
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1200"),
            "gfx1200 without HIPFIRE_LLOYD_GFX12=1"
        );
    }

    #[test]
    fn qwen35_is_batchable_la_unsupported_dtypes() {
        for &arch in WMMA_ARCHS {
            assert!(!is_batchable_la(DType::Q4K, arch), "Q4K must fall back");
            assert!(!is_batchable_la(DType::Q6K, arch), "Q6K must fall back");
            assert!(
                !is_batchable_la(DType::Q4F16G64, arch),
                "Q4F16G64 must fall back"
            );
            assert!(
                !is_batchable_la(DType::Q4F16G32, arch),
                "Q4F16G32 must fall back"
            );
            assert!(
                !is_batchable_la(DType::MQ2G256, arch),
                "MQ2G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::MQ8G256, arch),
                "MQ8G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::HFQ2G256, arch),
                "HFQ2G256 must fall back"
            );
        }
    }

    // ── Qwen3.5 MoE dispatch predicates ──────────────────────────

    #[test]
    fn moe_ffn_has_mq3_detects_mq3_in_experts() {
        // Smoke-test the renamed split predicates to ensure they compile and
        // the DType-level logic is preserved.
        // MoeFfnWeights requires GPU-backed tensors; the real DType dispatch
        // is tested via moe_prefill_rejects_mq3_before_admission_work below.
        let _mq3_dt = DType::MQ3G256;
        let _mq3l_dt = DType::MQ3G256Lloyd;
        let _mq4_dt = DType::MQ4G256;
        // Verify the predicates are callable (the MoeFfnWeights tensor
        // requirement prevents constructing a real fixture here; logic
        // coverage is via the admission tests below that use MoePrefillDtypes).
    }

    #[test]
    fn moe_prefill_topk_shape_requires_k8_and_bounded_experts() {
        assert!(moe_prefill_topk_shape_supported(8, 256));
        assert!(moe_prefill_topk_shape_supported(8, 1024));
        assert!(!moe_prefill_topk_shape_supported(4, 256));
        assert!(!moe_prefill_topk_shape_supported(8, 1025));
    }

    #[test]
    fn moe_prefill_admits_mq4_as_known_good_control() {
        let dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false
        ));
    }

    #[test]
    fn moe_prefill_admits_graded_mixed_experts_via_merged_kernel() {
        // Graded T3-3L: routed experts dtype-mixed (hot MQ6 / mid MQ4 / cold
        // MQ3-Lloyd), shared expert + router MQ4. The merged grouped-WMMA prefill
        // kernel serves the routed experts, so this MUST be batched-admissible —
        // otherwise it silently drops to the per-token prefill fallback at
        // ~decode speed and the merged kernel never fires.
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.routed_mixed_merged = true;
        dtypes.expert_gate_up_uniform = false;
        dtypes.expert_down_uniform = false;
        dtypes.expert_down = DType::MQ3G256Lloyd; // representative cold-tier dtype
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
        // The same mixed file WITHOUT the merged-kernel tag table is NOT admissible.
        dtypes.routed_mixed_merged = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
    }

    #[test]
    fn moe_prefill_rejects_mq3_before_admission_work() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_gate_up = DType::MQ3G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));

        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_down = DType::MQ3G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
    }

    /// MQ2/MQ3-G256-GL routed experts have DECODE kernels only — the four
    /// indexed MoE GEMVs. There is no grouped-WMMA GEMM and no batched indexed
    /// GEMV for the GL layouts, so the batched-prefill gate must reject them and
    /// let the model prefill through the per-token path (correct, just slower).
    ///
    /// Admitting GL here would send a `[M*gpr*64 B idx][M*gpr*2 B scale]` SoA
    /// blob into an HFQ4-layout (136 B/group interleaved) GEMM — out-of-bounds
    /// reads and garbage, exactly the failure the MQ3 guard above exists for.
    #[test]
    fn moe_prefill_rejects_gl_routed_experts() {
        for gl in [DType::MQ2G256GL, DType::MQ3G256GL] {
            for admit_mq6 in [false, true] {
                let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
                dtypes.expert_gate_up = gl;
                dtypes.expert_down = gl;
                assert!(
                    !moe_ffn_batched_admissible_for_dtypes(&dtypes, admit_mq6, true, true),
                    "{gl:?} must not be batched-prefill admissible (admit_mq6={admit_mq6})"
                );
            }
        }
        // Per-projection GL mix (the target SKU: gate_up MQ2-GL, down MQ3-GL).
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_gate_up = DType::MQ2G256GL;
        dtypes.expert_down = DType::MQ3G256GL;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, true, true
        ));
    }

    #[test]
    fn moe_prefill_mq6_requires_explicit_admission() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256;
        dtypes.shared_expert_up = DType::MQ6G256;
        dtypes.shared_expert_down = DType::MQ6G256;
        dtypes.expert_gate_up = DType::MQ6G256;
        dtypes.expert_down = DType::MQ6G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
    }

    #[test]
    fn moe_prefill_rejects_nonuniform_expert_projections() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_gate_up_uniform = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));

        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_down_uniform = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
    }

    #[test]
    fn moe_prefill_shared_gate_up_must_be_one_dtype() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_up = DType::MQ6G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
    }

    #[test]
    fn moe_prefill_admits_paro_when_enabled() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::ParoQ4G128);
        dtypes.router = DType::F32;
        dtypes.shared_expert_scalar_gate = DType::F32;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, true, false
        ));
    }

    #[test]
    fn moe_prefill_admits_e8_only_with_arch_gate() {
        // A3B mfp4-E8: Q8 router/scalar-gate/shared-expert + E8 routed experts.
        let mut dtypes = MoePrefillDtypes::uniform(DType::Q8_0);
        dtypes.expert_gate_up = DType::MFP4G32E8;
        dtypes.expert_down = DType::MFP4G32E8;
        // Without the arch gate (non-gfx1151), E8 is rejected.
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false
        ));
        // With the gfx1151 arch gate, the Q8-shared + E8-routed layer admits.
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, true
        ));
    }

    #[test]
    fn mq6_batched_admit_defaults_to_gfx11_and_gfx12() {
        // Post-merge resolved default (gfx11 widen 8d555fc6 ∪ master's gfx1151
        // fast-path): every WMMA arch — all gfx11 (RDNA3/3.5, incl. gfx1100 and
        // gfx1151) and all gfx12 (RDNA4) — default-admits MQ6 batched prefill.
        // Non-WMMA archs (gfx942 CDNA, gfx1030 RDNA2) stay default-off.
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1201"));
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1200"));
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1151"));
        // gfx1100 is now ADMITTED by default (the gfx11 widen), where master
        // had it default-off pending channel testing.
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1100"));
        assert!(!mq6_batched_admit_enabled_from_env(None, "gfx942"));
        assert!(!mq6_batched_admit_enabled_from_env(None, "gfx1030"));
        // Explicit env overrides still win on every arch.
        assert!(mq6_batched_admit_enabled_from_env(Some("1"), "gfx1151"));
        assert!(mq6_batched_admit_enabled_from_env(Some("1"), "gfx1100"));
        assert!(!mq6_batched_admit_enabled_from_env(Some("0"), "gfx1201"));
        assert!(!mq6_batched_admit_enabled_from_env(Some("0"), "gfx1100"));
    }

    #[test]
    fn q8_prefill_wmma_defaults_on_for_wave32_wmma_arches() {
        assert!(q8_prefill_wmma_enabled_from_env(None, "gfx1201", true));
        assert!(q8_prefill_wmma_enabled_from_env(None, "gfx1100", true));
        assert!(q8_prefill_wmma_enabled_from_env(None, "gfx1151", true));
        assert!(!q8_prefill_wmma_enabled_from_env(None, "gfx1030", false));
        assert!(q8_prefill_wmma_enabled_from_env(Some("1"), "gfx1151", true));
        assert!(!q8_prefill_wmma_enabled_from_env(
            Some("0"),
            "gfx1201",
            true
        ));
        assert!(!q8_prefill_wmma_enabled_from_env(
            Some("1"),
            "gfx1030",
            false
        ));
    }

    #[test]
    fn speculative_verify_has_an_explicit_dispatch_workload() {
        assert_eq!(
            prefill_dispatch_workload(false, false, false),
            DispatchWorkload::Standard
        );
        // DFlash and DSpark/MTP verify request per-token target hidden.
        assert_eq!(
            prefill_dispatch_workload(true, false, false),
            DispatchWorkload::SpeculativeVerify
        );
        assert_eq!(
            prefill_dispatch_workload(false, true, false),
            DispatchWorkload::SpeculativeVerify
        );
        assert_eq!(
            prefill_dispatch_workload(false, false, true),
            DispatchWorkload::SpeculativeVerify
        );
    }

    #[test]
    fn prefill_last_token_logits_policy_requires_explicit_opt_out() {
        assert!(prefill_should_emit_last_token_logits(false, true));
        assert!(prefill_should_emit_last_token_logits(true, true));
        assert!(prefill_should_emit_last_token_logits(false, false));
        assert!(!prefill_should_emit_last_token_logits(true, false));
    }

    #[test]
    fn owned_prefill_scratch_is_right_sized_without_tree_tape() {
        assert_eq!(owned_prefill_scratch_plan(1, 256, false), (2, false));
        assert_eq!(owned_prefill_scratch_plan(2, 256, false), (2, false));
        assert_eq!(owned_prefill_scratch_plan(32, 256, false), (32, false));
        assert_eq!(owned_prefill_scratch_plan(256, 256, false), (256, false));
        assert_eq!(owned_prefill_scratch_plan(1024, 256, false), (256, false));
    }

    #[test]
    fn prefill_chunk_plan_rebalances_singleton_tail() {
        fn plan(mut remaining: usize, max_batch: usize) -> Vec<usize> {
            let mut chunks = Vec::new();
            while remaining > 0 {
                let n = next_prefill_chunk_len(remaining, max_batch)
                    .expect("valid partition should exist");
                chunks.push(n);
                remaining -= n;
            }
            chunks
        }

        assert_eq!(plan(256, 256), vec![256]);
        assert_eq!(plan(257, 256), vec![255, 2]);
        assert_eq!(plan(258, 256), vec![256, 2]);
        assert_eq!(plan(513, 256), vec![256, 255, 2]);
        assert_eq!(plan(129, 128), vec![127, 2]);
    }

    #[test]
    fn prefill_chunk_plan_refuses_unpartitionable_minimum_batch() {
        assert_eq!(next_prefill_chunk_len(3, 2), None);
    }

    #[test]
    fn owned_prefill_scratch_preserves_tree_verify_tape() {
        assert_eq!(owned_prefill_scratch_plan(22, 256, true), (22, true));
        assert_eq!(owned_prefill_scratch_plan(64, 64, true), (64, true));
    }

    #[test]
    fn moe_grouped_m_total_max_is_tile_aligned() {
        let small_verify = moe_grouped_m_total_max(3, 8, 256);
        assert_eq!(small_verify % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(small_verify, 3872);

        let prompt_prefill = moe_grouped_m_total_max(27, 8, 256);
        assert_eq!(prompt_prefill % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(prompt_prefill, 4064);

        let full_chunk = moe_grouped_m_total_max(256, 8, 256);
        assert_eq!(full_chunk, 5888);
    }

    #[test]
    fn moe_grouped_m_total_bound_is_tight_for_small_batches() {
        let small_verify = moe_grouped_m_total_bound(24, 256);
        assert_eq!(small_verify % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(small_verify, 384);

        let prompt_prefill = moe_grouped_m_total_bound(216, 256);
        assert_eq!(prompt_prefill % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(prompt_prefill, 3456);

        let full_chunk = moe_grouped_m_total_bound(2048, 256);
        assert_eq!(full_chunk, 5888);
    }
}
