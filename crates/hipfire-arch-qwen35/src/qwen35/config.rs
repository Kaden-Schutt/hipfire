// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 config parsing (`Qwen35Config`), layer typing, EP batch attestation
//! types, and the tree-verify / mrope context structs.

use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_source::ModelSource;
use rdna_compute::GpuTensor;
use serde::Deserialize;

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    LinearAttention, // DeltaNet
    FullAttention,   // Standard MHA with gated output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum F16LmHeadMode {
    Native,
    F32,
}

fn parse_f16_lm_head_mode(value: Option<&str>) -> F16LmHeadMode {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if matches!(v.as_str(), "0" | "f32" | "fp32" | "legacy") => F16LmHeadMode::F32,
        _ => F16LmHeadMode::Native,
    }
}

pub(crate) fn f16_lm_head_mode_from_config() -> F16LmHeadMode {
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
/// Expert-parallel reduction mode. Only deterministic left-associated
/// peer-rooted sum is admitted for the batched EP route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35EpReduce {
    PeerRootedF32,
}

/// Expert-parallel classification for the attested EP route.
/// Only `ExpertParallel` (replicated attention, exactly-one-owner routed experts)
/// is admitted. Frozen interface for `Qwen35EpBatchReceipt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35BatchParallelism {
    ExpertParallel,
}

/// Attested receipt for one EP batch tick or seeded lane.
/// Created only after every rank synchronizes successfully.
/// Fields are private to prevent external fabrication; read-only getters
/// expose the attested values. Construction is module-private and
/// enforces `rank_count==4`, `rank_mask==0x0f`, `reduce==PeerRootedF32`,
/// `parallelism==ExpertParallel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35EpBatchReceipt {
    epoch: u64,
    rank_count: u8,
    rank_mask: u64,
    rows: u32,
    moe_collectives: u32,
    reduce: Qwen35EpReduce,
    parallelism: Qwen35BatchParallelism,
    _private: (),
}

impl Qwen35EpBatchReceipt {
    /// Attested epoch (checked, increments only on success).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn rank_count(&self) -> u8 {
        self.rank_count
    }
    pub fn rank_mask(&self) -> u64 {
        self.rank_mask
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn moe_collectives(&self) -> u32 {
        self.moe_collectives
    }
    pub fn reduce(&self) -> Qwen35EpReduce {
        self.reduce
    }
    pub fn parallelism(&self) -> Qwen35BatchParallelism {
        self.parallelism
    }
    /// Module-private attested constructor. Enforces frozen invariants.
    pub(crate) fn new_attested(epoch: u64, rows: u32, moe_collectives: u32) -> Self {
        Self {
            epoch,
            rank_count: 4,
            rank_mask: 0x0f,
            rows,
            moe_collectives,
            reduce: Qwen35EpReduce::PeerRootedF32,
            parallelism: Qwen35BatchParallelism::ExpertParallel,
            _private: (),
        }
    }
    /// Checked conversion for `rows` with error on overflow.
    pub(crate) fn rows_from_usize(rows: usize) -> HipResult<u32> {
        u32::try_from(rows).map_err(|_| HipError::new(0, "receipt rows overflow u32"))
    }
}

/// Loader-time batch shape knob for `Qwen35DecodeBatchEpState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35BatchLoadConfig {
    pub max_batch: usize,
    pub lane_capacity: usize,
    pub repeat_capacity: usize,
    pub prefill_chunk: usize,
}

impl Qwen35BatchLoadConfig {
    pub fn new(
        max_batch: usize,
        lane_capacity: usize,
        repeat_capacity: usize,
        prefill_chunk: usize,
    ) -> Self {
        Self {
            max_batch,
            lane_capacity,
            repeat_capacity,
            prefill_chunk,
        }
    }
}

/// Expert-parallel topology classification for the MQ4R route.
/// Only `ExpertParallel` (replicated attention, exactly-one-owner routed experts)
/// is admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35EpTopology {
    ExpertParallel,
}

/// Attested result of `validate_ep_batch_compatibility`. Every rank must pass;
/// refusal is `Err`, never `Ok` with `supported:false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35BatchCompatibility {
    pub(crate) rank_count: u8,
    pub(crate) rank_mask: u64,
    pub(crate) moe_layer_count: usize,
    pub(crate) topology: Qwen35EpTopology,
    pub(crate) reduce: Qwen35EpReduce,
    pub(crate) max_batch: usize,
    pub(crate) lane_capacity: usize,
    pub(crate) repeat_capacity: usize,
    pub(crate) prefill_chunk: usize,
    pub(crate) per_rank_decode_bytes: u64,
    pub(crate) per_rank_seed_pbs_bytes: u64,
    pub(crate) decode_partial_bytes: u64,
    pub(crate) seed_partial_bytes: u64,
    pub(crate) peer_bytes_per_rank: usize,
    pub(crate) per_rank_total_bytes: u64,
}

impl Qwen35BatchCompatibility {
    pub fn rank_count(&self) -> u8 {
        self.rank_count
    }
    pub fn rank_mask(&self) -> u64 {
        self.rank_mask
    }
    pub fn moe_layer_count(&self) -> usize {
        self.moe_layer_count
    }
    pub fn topology(&self) -> Qwen35EpTopology {
        self.topology
    }
    pub fn reduce(&self) -> Qwen35EpReduce {
        self.reduce
    }
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
    pub fn lane_capacity(&self) -> usize {
        self.lane_capacity
    }
    pub fn repeat_capacity(&self) -> usize {
        self.repeat_capacity
    }
    pub fn prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }
    pub fn per_rank_decode_bytes(&self) -> u64 {
        self.per_rank_decode_bytes
    }
    pub fn per_rank_seed_pbs_bytes(&self) -> u64 {
        self.per_rank_seed_pbs_bytes
    }
    pub fn decode_partial_bytes(&self) -> u64 {
        self.decode_partial_bytes
    }
    pub fn seed_partial_bytes(&self) -> u64 {
        self.seed_partial_bytes
    }
    pub fn peer_bytes_per_rank(&self) -> usize {
        self.peer_bytes_per_rank
    }
    pub fn per_rank_total_bytes(&self) -> u64 {
        self.per_rank_total_bytes
    }
}

/// Per-request 3D rope state. `None` for text-only requests, which keeps the
/// original 1D kernels and their dispatch identity (and hence the certified
/// retained-PM4 tape). Because a text token takes the same value on all three
/// axes, 3D mrope with `t == h == w` is bit-identical to 1D RoPE
/// (`crates/rdna-compute/examples/test_mrope_rope_parity.rs` asserts this),
/// so gating on "the request actually contains image tokens" is safe: a
/// text-only sequence would compute identical numbers either way.
///
/// Built by the daemon from the image span it already computes when splicing
/// visual tokens at `<|image_pad|>`; consumed by [`forward_scratch_mrope`] /
/// [`forward_scratch_embed_mrope`].
#[derive(Debug, Clone)]
pub struct MropeCtx {
    /// Sequence position that `positions[0]` describes — the value of the
    /// conversation cursor when this request's prefill started. Positions
    /// BELOW `base` belong to earlier turns and are not modelled here.
    pub base: usize,
    /// Per-token (t, h, w) for this request's prompt, already offset by
    /// `base` so the values are absolute rope phases.
    pub positions: Vec<[i32; 3]>,
    /// Added to the running sequence length for decode-step positions.
    /// `max(positions) + 1 - (base + positions.len())`.
    pub rope_delta: i32,
    /// `Qwen35Config::mrope_section` — [T, H, W] frequency counts.
    pub section: [usize; 3],
}

impl MropeCtx {
    /// Build from the loaded model config. `section` is read from
    /// [`Qwen35Config::mrope_section`] here rather than taken from the caller,
    /// so a request can never rotate with a section that disagrees with the
    /// checkpoint that produced the weights.
    pub fn new(
        config: &Qwen35Config,
        base: usize,
        positions: Vec<[i32; 3]>,
        rope_delta: i32,
    ) -> Self {
        Self {
            base,
            positions,
            rope_delta,
            section: config.mrope_section,
        }
    }

    /// (t, h, w) for sequence position `pos`.
    ///
    /// Inside the prompt this is the precomputed grid coordinate. Past the
    /// prompt (a generated token) all three axes collapse to
    /// `pos + rope_delta`, which is HF's decode formula: the cursor resumes
    /// at `max(image positions) + 1` rather than at the token index.
    pub fn pos3(&self, pos: usize) -> [i32; 3] {
        debug_assert!(
            pos >= self.base,
            "MropeCtx::pos3 called below base ({pos} < {})",
            self.base
        );
        match pos.checked_sub(self.base).and_then(|i| self.positions.get(i)) {
            Some(p) => *p,
            None => [pos as i32 + self.rope_delta; 3],
        }
    }
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

/// Exact dense Qwen3.6-27B shape shared by its gfx1100 decode selectors.
pub(crate) fn qwen36_27b_dense_shape(config: &Qwen35Config, n_v_heads: usize) -> bool {
    config.dim == 5_120
        && config.n_layers == 64
        && config.n_heads == 24
        && config.n_kv_heads == 4
        && config.head_dim == 256
        && config.linear_num_key_heads == 16
        && config.linear_key_head_dim == 128
        && n_v_heads == 48
        && config.linear_value_head_dim == 128
        && config.num_experts == 0
}
#[cfg(test)]
mod tests {
    use super::*;

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
    fn qwen36_27b_dense_shape_is_exact() {
        let inner = serde_json::json!({
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        });
        let mut cfg = from_config_value(&inner).expect("Qwen3.6-27B shape parse");
        assert!(qwen36_27b_dense_shape(&cfg, 48));
        assert!(!qwen36_27b_dense_shape(&cfg, 47));
        cfg.num_experts = 8;
        assert!(!qwen36_27b_dense_shape(&cfg, 48));
        cfg.num_experts = 0;
        cfg.dim = 2_048;
        assert!(!qwen36_27b_dense_shape(&cfg, 48));
    }
}
