// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 config / weights / state.
//!
//! Config parses from the HFQ `metadata_json` envelope. Weights/State mirror
//! the qwen35 GQA+MoE infrastructure (shared `WeightTensor`, `KvCache`, and the
//! `gemv_hfq4g256_moe_*` indexed-expert kernels) rather than deepseek4's MLA.
//! Expert weights ship pre-split (w1/w2/w3) in the HFQ; the loader byte-fuses
//! w1‖w3 into the per-expert `gate_up` blob the indexed GEMV kernels expect.

use crate::arch::MiniMaxM2;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::moe_plan::{select_moe_executor, MoEExecutionPolicy, MoeExecutorKind};
use hipfire_runtime::weight_manifest::{
    resolve_expert_manifest_for_policy, ExpertExecutionIdentity, ExpertGroupPlan,
    ExpertManifestResolution,
};
use hipfire_runtime::{screen_weight_tensor, MmqScreenable};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;

// ───────────────────────────── Config ─────────────────────────────

/// Typed MiniMax-M2 shape constants.
#[derive(Clone, Debug)]
pub struct MiniMaxConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Expert (MoE) FFN intermediate size (HF `intermediate_size`).
    pub intermediate_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    /// Rotated-dim count for partial RoPE (`rotary_dim`, < head_dim).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// Per-layer QK-norm on the flat q/k projection (RMSNorm pre-reshape).
    pub use_qk_norm: bool,
    /// Router uses `e_score_correction_bias` for top-k selection.
    pub use_routing_bias: bool,
    /// Router score activation; MiniMax-M2 = "sigmoid".
    pub scoring_func: String,
    /// MTP draft modules (spec-decode; 0 for the base forward / this ckpt).
    pub num_mtp_modules: usize,
    /// Optional REAP keep-map: emulate a pruned expert pool by partial-loading
    /// this full quant. Populated at config time from `HIPFIRE_REAP_PLAN=<dir>`;
    /// `None` ⇒ no pruning (literal original full load, byte-identical to
    /// baseline). Not (de)serialized — set in `apply_reap_plan`.
    pub reap_keep: Option<std::sync::Arc<hipfire_reap::plan::ReapPlan>>,
}

#[derive(Deserialize)]
struct RawMiniMaxConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    intermediate_size: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default = "default_rotary_dim")]
    rotary_dim: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default)]
    use_qk_norm: bool,
    #[serde(default)]
    use_routing_bias: bool,
    #[serde(default = "default_scoring")]
    scoring_func: String,
    #[serde(default)]
    num_mtp_modules: usize,
}

fn default_rotary_dim() -> usize {
    64
}
fn default_rope_theta() -> f32 {
    5_000_000.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    196_608
}
fn default_scoring() -> String {
    "sigmoid".to_string()
}

impl MiniMaxConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("minimax: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "minimax: metadata_json missing `config` wrapper".to_string())?;
        let raw: RawMiniMaxConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("minimax: parsing inner config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        let mut config = MiniMaxConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            intermediate_size: raw.intermediate_size,
            num_local_experts: raw.num_local_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            rotary_dim: raw.rotary_dim,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            use_qk_norm: raw.use_qk_norm,
            use_routing_bias: raw.use_routing_bias,
            scoring_func: raw.scoring_func,
            num_mtp_modules: raw.num_mtp_modules,
            reap_keep: None,
        };
        // Apply the optional REAP keep-map HERE, inside the single public config
        // entry point, so it is IMPOSSIBLE to bypass. Every caller funnels
        // through `MiniMaxConfig::from_hfq` — the daemon (via the `Architecture`
        // trait's `config_from_hfq`, which just delegates here) AND every
        // example (`infer_minimax`, `ep_minimax`, `dump_minimax_hidden_states`,
        // `debug_minimax_batch`) which call `MiniMaxConfig::from_hfq` directly,
        // never through the trait. Wiring REAP only in the trait shim would
        // silently ignore HIPFIRE_REAP_PLAN on all the direct callers. The trait
        // impl therefore does NOT re-apply (double-apply ⇒ kept-of-kept ⇒
        // load_any validation error). `from_hfq` returns `Result`, so a
        // malformed plan propagates cleanly via `?` (REAP is opt-in ⇒ a user who
        // explicitly set HIPFIRE_REAP_PLAN MUST hard-fail). With no env var,
        // `apply_reap_plan` is a no-op (Ok), so baseline behavior is unchanged.
        apply_reap_plan(&mut config)?;
        Ok(config)
    }

    /// q projection output width (n_heads * head_dim).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    /// k/v projection output width (n_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

/// Apply an optional REAP keep-map to a freshly parsed `MiniMaxConfig`.
///
/// Reads `HIPFIRE_REAP_PLAN=<dir>` (minimax has no legacy env alias). When set,
/// loads `<dir>/reap_plan.json` (or the legacy `keep_by_layer.json`) via
/// `ReapPlan::load_any`, validating against the ORIGINAL local-expert count
/// (`config.num_local_experts`) BEFORE overriding it to the kept count. This
/// emulates a pruned expert pool by partial-loading the full quant: only kept
/// experts are packed into the per-layer blob (under remapped names) and the
/// router's expert rows + routing bias are gathered to the kept set in
/// `MiniMaxWeights::load`.
///
/// No env ⇒ no-op (`config.reap_keep` stays `None`); the loader then takes the
/// literal original full-load path — byte-identical to baseline. REAP and EP
/// sharding are MUTUALLY EXCLUSIVE; that guard lives in `MiniMaxWeights::load`.
pub fn apply_reap_plan(config: &mut MiniMaxConfig) -> Result<(), String> {
    if let Some(plan) = hipfire_reap::plan::ReapPlan::from_config(
        "minimax",
        None,
        config.num_hidden_layers,
        config.num_local_experts,
    )? {
        config.num_local_experts = plan.kept_per_layer();
        config.reap_keep = Some(std::sync::Arc::new(plan));
    }
    Ok(())
}

/// Parse MiniMaxConfig from a ModelSource (safetensors or HFQ wrapper).
/// The metadata JSON should contain the same `{"architecture":..., "config":{...}}`
/// envelope as the HFQ format, as produced by SafetensorsSource::build_metadata_json.
pub fn config_from_safetensors(source: &dyn ModelSource) -> Result<MiniMaxConfig, String> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json())
        .map_err(|e| format!("minimax: metadata_json not valid JSON: {e}"))?;
    let inner = meta
        .get("config")
        .ok_or_else(|| "minimax: metadata_json missing 'config' key".to_string())?;
    let raw: RawMiniMaxConfig = serde_json::from_value(inner.clone())
        .map_err(|e| format!("minimax: failed to parse MiniMaxConfig from metadata: {e}"))?;
    let head_dim = raw
        .head_dim
        .unwrap_or(raw.hidden_size / raw.num_attention_heads);
    Ok(MiniMaxConfig {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim,
        intermediate_size: raw.intermediate_size,
        num_local_experts: raw.num_local_experts,
        num_experts_per_tok: raw.num_experts_per_tok,
        rotary_dim: raw.rotary_dim,
        rope_theta: raw.rope_theta,
        rms_norm_eps: raw.rms_norm_eps,
        max_position_embeddings: raw.max_position_embeddings,
        use_qk_norm: raw.use_qk_norm,
        use_routing_bias: raw.use_routing_bias,
        scoring_func: raw.scoring_func,
        num_mtp_modules: raw.num_mtp_modules,
        reap_keep: None,
    })
}

// ───────────────────────── HFQ load helpers ─────────────────────────
// Replicated from the qwen35 loader (those are crate-private). MiniMax HFQ
// files carry RAW HF tensor names, so we look them up by exact name.

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("minimax: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

/// Load a 1D norm vector (F16/F32) → F32 GpuTensor. MiniMax-M2 uses STANDARD
/// RMSNorm (`weight * x_normed`, no +1.0 offset — verified against
/// MiniMaxM2RMSNorm), so no offset is baked in.
fn load_norm(
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
        _ => {
            return Err(format!(
                "minimax: expected F16/F32 norm for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("minimax: upload norm {name}: {e:?}"))
}

/// Load a MiniMax AWQ shared-scale sidecar (1D F16, length k) → F32 GpuTensor.
fn load_mm_awq_scale(hfq: &HfqFile, gpu: &mut Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let (qt, data) = read_tensor(hfq, name).ok()?;
    if qt != 1 {
        return None;
    } // 1 = F16
    if data.len() != k * 2 {
        eprintln!(
            "minimax AWQ sidecar {name}: {} bytes != {} (k*2); skipping",
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

/// Load a quantized 2D weight → WeightTensor, tagging gpu_dtype from quant_type.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("minimax: load_wt {name}: {e}"))
}

/// REAP keep variant of [`load_wt`]: gather the tensor's first-axis rows (one
/// row per ORIGINAL expert) down to `keep` BEFORE quant decode, then build the
/// `WeightTensor` with `m = keep.len()`. Only used for the MoE router
/// (`block_sparse_moe.gate.weight`, shape `[orig_experts, hidden]`) under an
/// active keep-map: it then emits logits only for kept experts, in compact slot
/// order. `gather_rows` is exact for any row-independent quant (each per-expert
/// row carries its own scale/zero/codebook), which holds for every quant_type
/// `wt_from_raw` accepts. Reads owned bytes via `tensor_data_vec` to retain
/// `info.shape` for the original row count. `keep` MUST be compact-slot-ordered
/// and `m` MUST equal `keep.len()`.
fn load_wt_keep(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
    keep: &[u32],
) -> Result<WeightTensor, String> {
    debug_assert_eq!(
        m,
        keep.len(),
        "minimax load_wt_keep: m must equal keep.len()"
    );
    let (qt, sub) = hipfire_reap::load::gather_weight_rows("minimax", hfq, name, keep)?;
    wt_from_raw(gpu, qt, &sub, m, k).map_err(|e| format!("minimax: load_wt_keep {name}: {e}"))
}

/// REAP keep variant of [`load_norm`] for a 1-D per-expert F16/F32 vector (the
/// router's `e_score_correction_bias` `[orig_experts]`). Gathers the kept rows
/// (parallel to the router weight rows) so the per-expert routing bias aligns
/// with the gathered router logits. `gather_rows` over a 1-D shape is an exact
/// element select. Refuses block-packed quant (a single bias element is not a
/// whole quant block) — only F16/F32 1-D vectors can be element-gathered.
fn load_norm_keep(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    keep: &[u32],
) -> Result<GpuTensor, String> {
    debug_assert_eq!(
        m,
        keep.len(),
        "minimax load_norm_keep: m must equal keep.len()"
    );
    let f32_data = hipfire_reap::load::gather_f32_vec("minimax", hfq, name, keep)?;
    gpu.upload_f32(&f32_data, &[m])
        .map_err(|e| format!("minimax: upload routing_bias {name}: {e:?}"))
}

/// quant_type → DType mapping (subset used by MiniMax HFQ files; mirrors
/// qwen35::load_weight_tensor_raw). Uploads raw bytes and tags the dtype.
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

/// Bytes per packed-quant block (256 elements) for a given quant_type byte.
/// Used for TP-of-experts slicing (`expert_tp_column_pair`/`expert_tp_row_gather`).
/// Only the Lloyd variants used in M2.7.mq2 are currently supported; others
/// return an error so TP slicing fails fast rather than corrupting data.
fn block_bytes_for_qt(qt: u8) -> Result<usize, String> {
    match qt {
        19 => Ok(72),  // MQ2G256Lloyd (confirmed: weight_store.rs)
        20 => Ok(112), // MQ3G256Lloyd (confirmed: weight_store.rs)
        other => Err(format!(
            "minimax TP-expert-slice: quant_type {other} not supported for column/row slicing \
             (only MQ2G256Lloyd/MQ3G256Lloyd implemented)"
        )),
    }
}

// ──────────────────────────── Weights ────────────────────────────

/// Per-layer GPU-resident weights.
pub struct MiniMaxLayerWeights {
    pub attn_norm: GpuTensor, // input_layernorm
    pub ffn_norm: GpuTensor,  // post_attention_layernorm
    pub q_norm: GpuTensor,    // [n_heads*head_dim]
    pub k_norm: GpuTensor,    // [n_kv*head_dim]
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub router: WeightTensor, // block_sparse_moe.gate.weight [n_exp, hidden]
    pub routing_bias: GpuTensor, // e_score_correction_bias [n_exp] F32
    pub experts: Vec<MiniMaxExpertWeights>,
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
    /// EP-shard only: the shared zeroed gate_up buffer that non-owned experts'
    /// pointers index into (→ 0 silu output ⇒ 0 contribution). Owned here so it
    /// is reclaimed by `free_gpu` on unload and by the staging guard on a
    /// mid-load failure; `None` for single-GPU / fully-owned shards. Must
    /// outlive the device pointer table that bakes its address.
    pub dummy_gate_up: Option<GpuTensor>,
}

pub struct MiniMaxExpertWeights {
    /// Fused gate(w1)‖up(w3): [2*intermediate, hidden] MQ4G256.
    pub gate_up: WeightTensor,
    /// Down (w2): [hidden, intermediate] MQ4G256.
    pub down: WeightTensor,
}

pub struct MiniMaxWeights {
    pub embed: GpuTensor, // model.embed_tokens.weight (Q8 raw, for embedding_lookup_q8)
    pub final_norm: GpuTensor, // model.norm.weight
    pub lm_head: WeightTensor, // lm_head.weight
    pub layers: Vec<MiniMaxLayerWeights>,
    /// Model-owned immutable CPU state: the authoritative policy-aware expert
    /// manifest resolution for the exact key (load-bound manifest config
    /// identity + exact policy) of the model's forward path, cached here.
    /// SEALED: the cache type is private to this module; the public surface is
    /// [`expert_manifest_for_policy`](MiniMaxWeights::expert_manifest_for_policy)
    /// and the crate-visible
    /// [`single_policy`](MiniMaxWeights::single_policy).
    expert_manifest_cache: MiniMaxExpertManifestCache,
}

/// Model-owned immutable CPU cache of the authoritative policy-aware expert
/// manifest resolution for ONE exact key: the exact [`MoEExecutionPolicy`]
/// (kind + epoch-identity-sensitive mesh) AND the LOAD-BOUND manifest config
/// identity (every [`MiniMaxConfig`] field consumed by
/// [`MiniMaxM2::weight_manifest`] and [`MiniMaxM2::expert_group_manifest`]).
///
/// The cache is SEALED: type, field, constructors, resolver, and entry
/// internals are private to this module. The only authority doors are
/// [`MiniMaxWeights::expert_manifest_for_policy`] (public) and
/// [`MiniMaxWeights::single_policy`] (crate-visible for the forward path).
///
/// `MiniMaxWeights::load` cannot know the execution policy — the loader API
/// takes no policy, and threading one through would be a loader algorithm
/// change — so resolution is LAZY on the first forward, then cached. This is
/// the "lazy first-forward" case of the manifest-authority contract:
///
/// - **The config identity is bound at load**: both weight load paths
///   construct the cache with the identity of the config they loaded. Every
///   request compares its config identity against this load-time identity
///   BEFORE `get_or_init`; a wrong FIRST request is explicitly refused and
///   leaves the cache unseeded (no first-call-wins config binding).
/// - **Resolution** runs exactly once per exact key through
///   [`resolve_expert_manifest_for_policy`] over the authoritative
///   [`MiniMaxM2::weight_manifest`] + policy-aware
///   [`MiniMaxM2::expert_group_manifest`] specs, via `OnceLock::get_or_init`
///   — concurrent same-key callers never double-resolve. Success AND failure
///   are cached. Plans contain no GPU pointers.
/// - **Cached lookups are allocation-free**: the fast path compares the
///   BORROWED request (policy ref + load-bound config identity) against the
///   stored key; the owned epoch-sensitive policy/mesh is constructed only by
///   the winning `get_or_init` initializer (losing cold contenders clone
///   nothing). A different policy returns an explicit mismatch — never the
///   winner's resolution (no wrong-policy reuse), never a recompute, never a
///   replacement.
/// - **No fallback**: a failed resolution stays failed; validation is never
///   weakened.
struct MiniMaxExpertManifestCache {
    /// The manifest config identity the model was loaded with.
    load_identity: MiniMaxManifestConfigIdentity,
    entry: std::sync::OnceLock<CachedExpertManifestResolution>,
    /// The canonical single-rank execution policy of the model — ONE stable
    /// object (see [`Self::single_policy`]).
    single_policy: std::sync::OnceLock<MoEExecutionPolicy>,
    #[cfg(test)]
    resolver: MiniMaxManifestResolver,
    /// cfg(test)-only race seam: invoked after a caller observes the empty
    /// cell and immediately before `get_or_init`, so every contender reaches
    /// the once-boundary without deadlocking inside the initializer.
    #[cfg(test)]
    race_before_init: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
    /// cfg(test)-only construction counter: counts owned-key constructions by
    /// the WINNING `get_or_init` initializer (proves losing cold contenders
    /// clone nothing and repeated initialized lookups construct nothing).
    #[cfg(test)]
    cold_constructs: std::sync::atomic::AtomicUsize,
}

/// The complete cache key: the exact execution policy plus the load-bound
/// manifest/config identity. Do not under-key: every identity field is
/// consumed by `MiniMaxM2::weight_manifest` / `expert_group_manifest`
/// (arch.rs) and therefore changes the resolution.
#[derive(Clone, PartialEq, Eq, Debug)]
struct MiniMaxManifestKey {
    policy: MoEExecutionPolicy,
    config: MiniMaxManifestConfigIdentity,
}

/// Every `MiniMaxConfig` field consumed by the MiniMax manifest declarations.
/// `reap_keep` is deliberately absent: it is already folded into
/// `num_local_experts` at config-parse time (the only manifest-visible effect).
#[derive(Clone, PartialEq, Eq, Debug)]
struct MiniMaxManifestConfigIdentity {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    num_local_experts: usize,
}

impl MiniMaxManifestConfigIdentity {
    fn from_cfg(cfg: &MiniMaxConfig) -> Self {
        MiniMaxManifestConfigIdentity {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            intermediate_size: cfg.intermediate_size,
            num_local_experts: cfg.num_local_experts,
        }
    }
}

struct CachedExpertManifestResolution {
    key: MiniMaxManifestKey,
    resolution: Result<ExpertManifestResolution, String>,
}

/// Deterministic cache-level resolver seam: production resolves through the
/// authoritative arch manifests; tests inject a resolver to prove once-only
/// resolution and exact-key semantics without GPU objects.
#[cfg(test)]
enum MiniMaxManifestResolver {
    Arch,
    Test(
        Box<
            dyn Fn(&MiniMaxConfig, &MoEExecutionPolicy) -> Result<ExpertManifestResolution, String>
                + Send
                + Sync,
        >,
    ),
}

/// Resolve the authoritative MiniMax expert manifest for one exact policy.
fn resolve_arch_manifest(
    cfg: &MiniMaxConfig,
    policy: &MoEExecutionPolicy,
) -> Result<ExpertManifestResolution, String> {
    let specs = MiniMaxM2::expert_group_manifest(cfg, policy);
    let manifest = MiniMaxM2::weight_manifest(cfg);
    resolve_expert_manifest_for_policy(&specs, &manifest, policy)
}

fn identity_fmt(identity: &MiniMaxManifestConfigIdentity) -> String {
    format!(
        "(vocab {}, hidden {}, layers {}, heads {}, kv_heads {}, head_dim {}, inter {}, experts {})",
        identity.vocab_size,
        identity.hidden_size,
        identity.num_hidden_layers,
        identity.num_attention_heads,
        identity.num_key_value_heads,
        identity.head_dim,
        identity.intermediate_size,
        identity.num_local_experts,
    )
}

fn config_mismatch_error(
    load: &MiniMaxManifestConfigIdentity,
    request: &MiniMaxManifestConfigIdentity,
) -> String {
    format!(
        "minimax expert manifest cache: model was loaded with config identity {}; requested          config identity {} differs — refusing reuse (cache is per loaded model / load-bound          manifest config)",
        identity_fmt(load),
        identity_fmt(request),
    )
}

fn policy_mismatch_error(stored: &MoEExecutionPolicy, requested: &MoEExecutionPolicy) -> String {
    format!(
        "minimax expert manifest cache: bound to exact policy {:?} (mesh {:?}); requested policy          {:?} (mesh {:?}) differs — refusing reuse (cache is per loaded model / exact policy)",
        stored.kind(),
        stored.mesh(),
        requested.kind(),
        requested.mesh(),
    )
}

impl MiniMaxExpertManifestCache {
    /// Construct the cache bound to the manifest config identity of the
    /// config the model was loaded with. Both weight load paths
    /// (`MiniMaxWeights::load` and `load_weights_from_safetensors`) derive
    /// this identity from the SAME `cfg` they load weights with.
    fn new(load_identity: MiniMaxManifestConfigIdentity) -> Self {
        Self {
            load_identity,
            entry: std::sync::OnceLock::new(),
            single_policy: std::sync::OnceLock::new(),
            #[cfg(test)]
            resolver: MiniMaxManifestResolver::Arch,
            #[cfg(test)]
            race_before_init: None,
            #[cfg(test)]
            cold_constructs: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// cfg(test) seam: deterministic resolver + optional race hook (see
    /// `race_before_init`), so concurrency / exact-key / load-bound semantics
    /// are proven without GPU objects.
    #[cfg(test)]
    fn with_seams(
        load_identity: MiniMaxManifestConfigIdentity,
        resolver: Box<
            dyn Fn(&MiniMaxConfig, &MoEExecutionPolicy) -> Result<ExpertManifestResolution, String>
                + Send
                + Sync,
        >,
        race_before_init: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self {
            load_identity,
            entry: std::sync::OnceLock::new(),
            single_policy: std::sync::OnceLock::new(),
            resolver: MiniMaxManifestResolver::Test(resolver),
            race_before_init,
            cold_constructs: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The canonical single-rank execution policy of this model: ONE stable
    /// object for the model's lifetime. `DeviceMesh::single()` issues a fresh
    /// mesh epoch per call, so a per-token `MoEExecutionPolicy::single()`
    /// would be a different exact policy every token and the exact-key cache
    /// would reject every token after the first. The single forward path
    /// passes this object on every call — never a per-token reconstruction.
    fn single_policy(&self) -> &MoEExecutionPolicy {
        self.single_policy.get_or_init(MoEExecutionPolicy::single)
    }

    /// Return the cached resolution for the exact key, resolving and binding
    /// the cache on first use via `OnceLock::get_or_init` (same-key
    /// resolution runs exactly once, even under concurrency).
    ///
    /// Order of admission:
    /// 1. LOAD-BOUND config identity check (BEFORE `get_or_init`): a request
    ///    whose config identity differs from the load-time identity is
    ///    refused explicitly and leaves the cache unseeded.
    /// 2. Cached fast path: the BORROWED request policy is compared against
    ///    the stored key — no owned policy/mesh clone on the cached path.
    /// 3. Cold path: the owned policy/key is constructed only here, inside
    ///    the winning initializer; the post-init borrowed compare rejects a
    ///    concurrent winner with a different exact policy (never winner
    ///    reuse).
    fn get_or_resolve<'a>(
        &'a self,
        cfg: &MiniMaxConfig,
        policy: &MoEExecutionPolicy,
    ) -> Result<&'a ExpertManifestResolution, String> {
        // 1. Load-bound config identity — before get_or_init, so a wrong
        //    FIRST request refuses and leaves the cache unseeded.
        let request_identity = MiniMaxManifestConfigIdentity::from_cfg(cfg);
        if request_identity != self.load_identity {
            return Err(config_mismatch_error(
                &self.load_identity,
                &request_identity,
            ));
        }
        // 2. Cached fast path: borrowed comparison, zero owned-key clones.
        if let Some(entry) = self.entry.get() {
            if entry.key.policy != *policy {
                return Err(policy_mismatch_error(&entry.key.policy, policy));
            }
            return entry.resolution.as_ref().map_err(|err| err.clone());
        }
        // cfg(test) race seam: every contender that observed the empty cell
        // rendezvouses here BEFORE get_or_init (never inside the initializer,
        // so the once-boundary is reached without deadlock).
        #[cfg(test)]
        if let Some(hook) = &self.race_before_init {
            hook();
        }
        // 3. Cold path: the WINNING `get_or_init` initializer constructs the
        //    owned policy/key exactly once. Losing cold contenders never
        //    clone the epoch-sensitive policy/mesh axis and never build an
        //    owned key — they only borrow-compare against the winner's
        //    stored key below.
        let entry = self.entry.get_or_init(|| {
            #[cfg(test)]
            self.cold_constructs
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let key = MiniMaxManifestKey {
                policy: policy.clone(),
                config: request_identity,
            };
            #[cfg(test)]
            let resolution = match &self.resolver {
                MiniMaxManifestResolver::Arch => resolve_arch_manifest(cfg, &key.policy),
                MiniMaxManifestResolver::Test(resolve) => resolve(cfg, &key.policy),
            };
            #[cfg(not(test))]
            let resolution = resolve_arch_manifest(cfg, &key.policy);
            CachedExpertManifestResolution { key, resolution }
        });
        // A concurrent first caller may have won with a different exact
        // policy: NEVER winner reuse — borrowed compare, explicit mismatch.
        if entry.key.policy != *policy {
            return Err(policy_mismatch_error(&entry.key.policy, policy));
        }
        entry.resolution.as_ref().map_err(|err| err.clone())
    }
}

/// The common Single-rank MoE authority of one model: the stable canonical
/// single policy PLUS the model-owned cached expert-manifest resolution bound
/// to it (load-bound config identity + exact policy). BOTH forward entries —
/// sequential `decode_step_body` and production batched `forward_batch` —
/// obtain their Single authority through [`minimax_single_moe_authority`],
/// never a per-token policy construction, never a local plan fabrication.
#[derive(Debug)]
pub(crate) struct SingleMoeAuthority<'a> {
    policy: &'a MoEExecutionPolicy,
    resolution: &'a ExpertManifestResolution,
}

impl<'a> SingleMoeAuthority<'a> {
    /// CPU-testable construction from a policy + resolution pair (tests and
    /// malformed-seam cases). Production acquisition always goes through
    /// [`Self::from_cache`].
    #[cfg(test)]
    pub(crate) fn new(
        policy: &'a MoEExecutionPolicy,
        resolution: &'a ExpertManifestResolution,
    ) -> Self {
        SingleMoeAuthority { policy, resolution }
    }

    /// CPU-testable acquisition core from the model-owned cache: the stable
    /// canonical single policy and the cached resolution for it. A
    /// stale-config / different-policy / failed resolution is refused HERE —
    /// in `forward_batch` this runs BEFORE any GPU allocation or launch, in
    /// `decode_step_body` before the layer loop. Production goes through
    /// [`minimax_single_moe_authority`] (the same path via the weights
    /// accessors); the CPU tests use this core directly.
    #[cfg(test)]
    pub(crate) fn from_cache(
        cache: &'a MiniMaxExpertManifestCache,
        cfg: &MiniMaxConfig,
    ) -> Result<Self, String> {
        let policy = cache.single_policy();
        let resolution = cache
            .get_or_resolve(cfg, policy)
            .map_err(|e| format!("minimax: expert manifest (single): {e}"))?;
        Ok(SingleMoeAuthority { policy, resolution })
    }

    /// The stable canonical single policy of this authority.
    pub(crate) fn policy(&self) -> &'a MoEExecutionPolicy {
        self.policy
    }

    /// The model-owned cached resolution of this authority.
    #[cfg(test)]
    pub(crate) fn resolution(&self) -> &'a ExpertManifestResolution {
        self.resolution
    }

    /// Borrow and validate the layer-`l` plan BEFORE any kernel uses that
    /// layer. Admission (all allocation-free):
    /// 1. the resolution must cover the layer (out-of-range fails closed);
    /// 2. the plan must be layer-scoped (`plans[l].layer == Some(l)`);
    /// 3. the plan must declare the exact `sigmoid_topk` router identity —
    ///    the direct batched indexed sigmoid kernels require it;
    /// 4. the plan must admit [`ExpertExecutionIdentity::IndexedQuantized`] —
    ///    the direct batched kernels are the indexed quantized family;
    /// 5. Single executor admission ([`select_moe_executor`] — group size vs
    ///    rank count, parallelism, and the post-combine collective all agree
    ///    with the canonical single policy).
    /// The batched path runs this as an admission pin before its direct
    /// established batched MoE dispatch; the sequential path lowers with the
    /// returned plan.
    pub(crate) fn plan_for_layer(&self, l: usize) -> Result<&'a ExpertGroupPlan, String> {
        let plan = self.resolution.plans.get(l).ok_or_else(|| {
            format!(
                "minimax: expert manifest resolution has {} plan(s); layer {l} is out of range",
                self.resolution.plans.len()
            )
        })?;
        if plan.layer != Some(l) {
            return Err(format!(
                "minimax: expert manifest resolution plan[{l}] is scoped to layer {:?}, not Some({l})",
                plan.layer
            ));
        }
        if plan.router_identity != "sigmoid_topk" {
            return Err(format!(
                "minimax: expert manifest resolution plan[{l}] router identity '{}' is not                  'sigmoid_topk' (the direct batched indexed sigmoid kernels require it)",
                plan.router_identity
            ));
        }
        if !plan
            .allowed_executions
            .contains(&ExpertExecutionIdentity::IndexedQuantized)
        {
            return Err(format!(
                "minimax: expert manifest resolution plan[{l}] allowed executions {:?} do not                  admit IndexedQuantized (the direct batched indexed kernels require it)",
                plan.allowed_executions
            ));
        }
        match select_moe_executor(plan, self.policy) {
            Ok(MoeExecutorKind::SingleMesh) => Ok(plan),
            Ok(MoeExecutorKind::Parallel) => Err(format!(
                "minimax: expert manifest resolution plan[{l}] (group '{}', parallelism {:?},                  collective {:?}) is not Single-admitted",
                plan.group, plan.parallelism, plan.collective
            )),
            Err(err) => Err(format!(
                "minimax: expert manifest resolution plan[{l}] is not Single-admitted: {err:?}"
            )),
        }
    }
}

/// The common Single-authority accessor used by BOTH `decode_step_body` and
/// `forward_batch`: obtains the cache-owned stable `weights.single_policy()`
/// and `weights.expert_manifest_for_policy(cfg, policy)` (load-bound config /
/// policy / failure refusal propagated), exposed as the per-layer-validating
/// [`SingleMoeAuthority`].
pub(crate) fn minimax_single_moe_authority<'a>(
    weights: &'a MiniMaxWeights,
    cfg: &MiniMaxConfig,
) -> Result<SingleMoeAuthority<'a>, String> {
    let policy = weights.single_policy();
    let resolution = weights
        .expert_manifest_for_policy(cfg, policy)
        .map_err(|e| format!("minimax: expert manifest (single): {e}"))?;
    Ok(SingleMoeAuthority { policy, resolution })
}

impl MiniMaxWeights {
    /// The cached authoritative policy-aware expert-manifest resolution for
    /// `policy` — the single production plan source (see
    /// [`MiniMaxExpertManifestCache::get_or_resolve`]). The forward paths
    /// borrow plans by layer from the returned resolution; the object is
    /// identical (same pointer) on every repeat call for the same policy.
    pub fn expert_manifest_for_policy<'a>(
        &'a self,
        cfg: &MiniMaxConfig,
        policy: &MoEExecutionPolicy,
    ) -> Result<&'a ExpertManifestResolution, String> {
        self.expert_manifest_cache.get_or_resolve(cfg, policy)
    }

    /// The canonical single-rank execution policy of this model — ONE stable
    /// object for the model's lifetime (see
    /// [`MiniMaxExpertManifestCache::single_policy`]); the single forward
    /// path must never reconstruct the policy per token.
    pub(crate) fn single_policy(&self) -> &MoEExecutionPolicy {
        self.expert_manifest_cache.single_policy()
    }
}

impl MmqScreenable for MiniMaxWeights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.lm_head, gpu, &mut safe, &mut unsafe_count);
        // Routed experts use packed/indirect storage. Screen the resident
        // attention projections and dense router tensors here.
        for layer in &self.layers {
            for weight in [&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.router] {
                screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
            }
        }
        (safe, unsafe_count)
    }
}

impl MiniMaxWeights {
    /// Load MiniMax weights. `shard = Some((cfg, rank))` enables **EP shard-aware
    /// loading**: each layer's experts are read from the file but ONLY the
    /// rank-owned experts are uploaded into a compact packed blob (so an 86 GB
    /// model fits across N×32 GB cards — load-then-free is impossible since the
    /// experts are one packed blob too big for a single card). Non-owned expert
    /// pointers point at a shared zeroed gate_up buffer (→ 0 contribution). The
    /// non-expert weights (embed / lm_head / attention / norms) are always loaded
    /// in full (replicated per rank). `shard = None` loads everything (single-GPU).
    ///
    /// `tp_slice = Some(TpExpertSlice { tp, rank })` enables **TP-of-experts**:
    /// every rank owns ALL experts but each expert's weight matrix is column/row-split.
    /// gate‖up: column split `[2·inter, hidden]` → `[2·inter/tp, hidden]` (via
    /// `weight_store::expert_tp_column_pair`). down: row gather `[hidden, inter]` →
    /// `[hidden, inter/tp]` (via `weight_store::expert_tp_row_gather`). Mutually
    /// exclusive with EP `shard` (which sub-sets which experts are loaded).
    pub fn load(
        hfq: &mut HfqFile,
        cfg: &MiniMaxConfig,
        gpu: &mut Gpu,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
        tp_slice: Option<hipfire_runtime::tp_shard::TpExpertSlice>,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_exp = cfg.num_local_experts;

        // REAP keep-map and EP sharding are MUTUALLY EXCLUSIVE: REAP emulates a
        // pruned pool by partial-loading kept experts into the compact blob (so
        // `n_exp` is already the kept count and the pointer table spans only
        // kept slots), while EP-shard packs only rank-owned experts and routes
        // non-owned ones to a shared zeroed dummy. Combining them would
        // double-remap the slot space. Mirror deepseek4's guard and refuse.
        if cfg.reap_keep.is_some() && shard.is_some() {
            return Err("minimax: REAP keep-map + EP sharding are mutually exclusive".into());
        }
        // TP-of-experts and EP sharding are also mutually exclusive.
        if tp_slice.is_some() && shard.is_some() {
            return Err("minimax: TP expert slice + EP sharding are mutually exclusive".into());
        }
        // inter_local: intermediate dim per TP rank. Under TP this is inter/tp;
        // under no TP it equals inter (tp=1, inter_local==inter → byte-identical).
        let inter_local = tp_slice.map(|ts| ts.inter_local(inter)).unwrap_or(inter);

        // Globals.
        let (_qt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("minimax: upload embed: {e:?}"))?;
        let final_norm = load_norm(hfq, gpu, "model.norm.weight", &[hidden])?;
        let lm_head = load_wt(hfq, gpu, "lm_head.weight", cfg.vocab_size, hidden)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let attn_norm = load_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let ffn_norm = load_norm(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let q_norm = load_norm(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), &[q_dim])?;
            let k_norm = load_norm(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), &[kv_dim])?;
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
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
            )?;

            // REAP keep-map for this layer (None ⇒ no pruning / identity load).
            // When a keep is present, `n_exp == cfg.num_local_experts` is already
            // the KEPT count (overridden in apply_reap_plan); the router, routing
            // bias, and expert-pack loop below load only the kept original
            // experts, in compact slot order. (EP-shard is excluded above, so the
            // keep path never coexists with the non-owned-zero path.)
            let reap_ep = cfg.reap_keep.as_ref().map(|r| r.expert_plan(l));
            let keep_l = reap_ep.as_ref().and_then(|e| e.keep());

            // Router: hidden → n_exp. Under a keep, gather the router's expert
            // rows (`[orig_experts, hidden]`) down to the kept set so it emits
            // logits only for kept experts, in compact slot order. No keep ⇒ the
            // literal original full load (byte-identical to baseline).
            let router = match keep_l {
                Some(keep) => load_wt_keep(
                    hfq,
                    gpu,
                    &format!("{p}.block_sparse_moe.gate.weight"),
                    n_exp,
                    hidden,
                    keep,
                )?,
                None => load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.block_sparse_moe.gate.weight"),
                    n_exp,
                    hidden,
                )?,
            };
            // e_score_correction_bias: [n_exp] F16 → F32 (kept F16 in HFQ). This
            // is a PER-EXPERT routing bias added to the router logits for top-k
            // selection; under a keep it MUST be row-gathered to the kept set in
            // the same compact slot order as the router weight, else the bias
            // mis-aligns with the gathered logits. No keep ⇒ literal original.
            let routing_bias = match keep_l {
                Some(keep) => load_norm_keep(
                    hfq,
                    gpu,
                    &format!("{p}.block_sparse_moe.e_score_correction_bias"),
                    n_exp,
                    keep,
                )?,
                None => load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.block_sparse_moe.e_score_correction_bias"),
                    &[n_exp],
                )?,
            };

            // Routed experts: pack ALL experts of this layer into ONE gate_up
            // blob + ONE down blob (deepseek4 `upload_layer_routed_experts`
            // pattern). The old code did a separate `upload_raw`/hipMalloc per
            // expert per projection — 2*n_exp tiny allocs/layer, ~31.7k total,
            // each rounded up to HIP's allocation granularity. That fragmentation
            // wasted ~20GB of VRAM, inflating mq2-lloyd's 86GB file to a ~114GB
            // resident footprint that OOM'd gfx1151's 96GB carveout. The
            // `*_indexed` GEMV kernels index experts by device pointer, so one
            // packed blob + a base+e*stride pointer table is byte- and
            // result-identical to the per-expert layout (validated against the
            // tiny oracle: gfx1151 cosine unchanged).
            let mut gu_combined: Vec<u8> = Vec::new();
            let mut dn_combined: Vec<u8> = Vec::new();
            let mut gu_stride = 0usize; // raw per-expert blob size (for validation only)
            let mut dn_stride = 0usize; // raw per-expert blob size (for validation only)
                                        // Packed per-expert size in the combined blob (equals raw stride when no
                                        // TP slicing; equals sliced size when tp_slice is Some).
            let mut gu_packed_stride = 0usize;
            let mut dn_packed_stride = 0usize;
            let mut qt_gu = 0u8;
            let mut qt_dn = 0u8;
            // EP shard: only upload rank-owned experts into the compact blob.
            // `local_of_global[e]` maps a global expert id to its slot in the
            // compact (owned-only) blob, or usize::MAX if not owned by this rank.
            let owns = |e: usize| {
                shard
                    .map(|(s, rank)| s.owns_expert(rank, e))
                    .unwrap_or(true)
            };
            let mut local_of_global = vec![usize::MAX; n_exp];
            let mut n_owned = 0usize;
            // Iterate COMPACT slots `0..n_exp`. `e` = the ORIGINAL expert index
            // loaded into this slot: under a REAP keep it is `src(slot)` (read the
            // kept experts in compact order); with no keep it is `slot` itself, so
            // the loop is byte-identical to the original literal pack (and the
            // EP-shard path is unchanged — REAP excludes sharding, so when a keep
            // is active `shard` is None and `owns(e)` is always true). `owns` /
            // `local_of_global` stay indexed by the original id `e` (== slot on
            // the no-keep path), preserving shard semantics exactly.
            for slot in 0..n_exp {
                let e = reap_ep.as_ref().map(|ep| ep.src(slot)).unwrap_or(slot);
                let ep = format!("{p}.block_sparse_moe.experts.{e}");
                let (qt1, w1) = read_tensor(hfq, &format!("{ep}.w1.weight"))?;
                let (_qt3, w3) = read_tensor(hfq, &format!("{ep}.w3.weight"))?;
                let (qt2, w2) = read_tensor(hfq, &format!("{ep}.w2.weight"))?;
                let gu_len = w1.len() + w3.len();
                if slot == 0 {
                    gu_stride = gu_len;
                    dn_stride = w2.len();
                    qt_gu = qt1;
                    qt_dn = qt2;
                    let cap = shard
                        .map(|(s, _)| s.experts_per_rank(n_exp))
                        .unwrap_or(n_exp);
                    gu_combined.reserve(gu_len * cap);
                    dn_combined.reserve(w2.len() * cap);
                } else if gu_len != gu_stride || w2.len() != dn_stride {
                    return Err(format!(
                        "minimax L{l}E{e}: non-uniform expert stride (gate_up {gu_len}/{gu_stride}, down {}/{dn_stride}); packed layout requires equal-size experts",
                        w2.len()
                    ));
                }
                if owns(e) {
                    // `local_of_global` is indexed by the KERNEL ROUTING INDEX —
                    // the slot the router/topk emits, which the device pointer
                    // table is built over below. With EP-shard that index is the
                    // GLOBAL expert id `e`; with REAP (shard excluded) the router
                    // is gathered to compact slots, so the routing index is the
                    // COMPACT SLOT. On the plain path `e == slot`, so both agree
                    // and the table is identity (byte-identical to baseline).
                    let route = if shard.is_some() { e } else { slot };
                    local_of_global[route] = n_owned;
                    n_owned += 1;
                    if let Some(ts) = tp_slice {
                        // TP-of-experts: slice this expert's gate‖up blob (column
                        // split) and down blob (row gather) to rank-local inter/tp.
                        let bb_gu = block_bytes_for_qt(qt_gu)?;
                        let bb_dn = block_bytes_for_qt(qt_dn)?;
                        // gate‖up raw blob for this expert: w1‖w3 concatenated.
                        let mut raw_gu = Vec::with_capacity(w1.len() + w3.len());
                        raw_gu.extend_from_slice(&w1);
                        raw_gu.extend_from_slice(&w3);
                        let sliced_gu = hipfire_runtime::weight_store::expert_tp_column_pair(
                            &raw_gu, inter, hidden, bb_gu, ts.rank, ts.tp,
                        )
                        .map_err(|e2| format!("minimax L{l}E{e}: TP column slice gate_up: {e2}"))?;
                        let sliced_dn = hipfire_runtime::weight_store::expert_tp_row_gather(
                            &w2, hidden, inter, bb_dn, ts.rank, ts.tp,
                        )
                        .map_err(|e2| format!("minimax L{l}E{e}: TP row gather down: {e2}"))?;
                        // Record the sliced per-expert size for the pointer table.
                        if n_owned == 1 {
                            gu_packed_stride = sliced_gu.len();
                            dn_packed_stride = sliced_dn.len();
                        }
                        gu_combined.extend_from_slice(&sliced_gu);
                        dn_combined.extend_from_slice(&sliced_dn);
                    } else {
                        if n_owned == 1 {
                            gu_packed_stride = gu_stride;
                            dn_packed_stride = dn_stride;
                        }
                        gu_combined.extend_from_slice(&w1);
                        gu_combined.extend_from_slice(&w3);
                        dn_combined.extend_from_slice(&w2);
                    }
                }
                // Non-owned: w1/w3/w2 read from the file (for stride validation)
                // then dropped — never uploaded. That is the EP memory win.
            }
            if n_owned == 0 {
                return Err(format!("minimax L{l}: shard rank owns no experts"));
            }
            // One allocation per projection. The representative `WeightTensor`'s
            // buffer IS the packed blob; its m/k describe a SINGLE expert's shape
            // (the forward's rotate_x_mq / silu_mul_rotate / dtype dispatch read
            // those + the AWQ scale, never the buffer's full extent — per-expert
            // data is reached through the pointer table below).
            // Under TP: inter_local = inter/tp, so gate_up m=2*inter_local, down k=inter_local.
            let mut gate_up = wt_from_raw(gpu, qt_gu, &gu_combined, 2 * inter_local, hidden)
                .map_err(|e2| format!("minimax: pack gate_up L{l}: {e2}"))?;
            let mut down = wt_from_raw(gpu, qt_dn, &dn_combined, hidden, inter_local)
                .map_err(|e2| format!("minimax: pack down L{l}: {e2}"))?;
            drop(gu_combined);
            drop(dn_combined);
            gate_up.awq_scale = load_mm_awq_scale(
                hfq,
                gpu,
                &format!("{p}.block_sparse_moe.awq_scale_gate_up.weight"),
                hidden,
            );
            if hipfire_config::developer_var_os("HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ").is_some() {
                // down-AWQ harmful (shared s_down bad approx); opt-in
                down.awq_scale = load_mm_awq_scale(
                    hfq,
                    gpu,
                    &format!("{p}.block_sparse_moe.awq_scale_down.weight"),
                    inter,
                );
            }
            if gate_up.awq_scale.is_some() {
                eprintln!("minimax: AWQ scales attached at L{l} (shared per-layer)");
            }
            let gu_base = gate_up.buf.buf.as_ptr() as u64;
            let dn_base = down.buf.buf.as_ptr() as u64;
            let experts = vec![MiniMaxExpertWeights { gate_up, down }];

            // Device pointer tables: n_exp u64 device addresses, stored as
            // [2*n_exp] F32 (8 bytes/ptr). Single-GPU: base + e*stride into the
            // full packed blob. EP shard: owned e → compact-blob slot
            // (base + local*stride); non-owned e → a shared ZEROED gate_up buffer
            // (→ 0 output ⇒ 0 contribution; down ptr is irrelevant since its rot
            // input is 0, so it reuses the compact down base).
            // Owned in `dummy_gate_up` on the layer below so `free_gpu` reclaims
            // it on a successful EP unload. GpuTensor has no Drop, so leaving it
            // on the stack here would leak its buffer; we must thread it into
            // the layer struct.
            let dummy_slot = if shard.is_some() && n_owned < n_exp {
                let slot = gpu
                    .zeros(&[gu_packed_stride / 4], DType::F32)
                    .map_err(|e| format!("minimax L{l}: zero gate_up dummy: {e:?}"))?;
                Some(slot)
            } else {
                None
            };
            let dummy_gu = dummy_slot
                .as_ref()
                .map(|z| z.buf.as_ptr() as u64)
                .unwrap_or(gu_base);
            let gu_bytes: Vec<u8> = (0..n_exp)
                .flat_map(|e| {
                    let ptr = if owns(e) {
                        gu_base + (local_of_global[e] * gu_packed_stride) as u64
                    } else {
                        dummy_gu
                    };
                    ptr.to_ne_bytes()
                })
                .collect();
            let dn_bytes: Vec<u8> = (0..n_exp)
                .flat_map(|e| {
                    let ptr = if owns(e) {
                        dn_base + (local_of_global[e] * dn_packed_stride) as u64
                    } else {
                        dn_base // rot input is 0 for non-owned ⇒ output 0 regardless
                    };
                    ptr.to_ne_bytes()
                })
                .collect();
            let expert_gate_up_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("minimax: alloc gu_ptrs: {e:?}"))?;
            let expert_down_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("minimax: alloc dn_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                .map_err(|e| format!("minimax: htod gu_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                .map_err(|e| format!("minimax: htod dn_ptrs: {e:?}"))?;
            let dummy_gate_up = dummy_slot;

            layers.push(MiniMaxLayerWeights {
                attn_norm,
                ffn_norm,
                q_norm,
                k_norm,
                wq,
                wk,
                wv,
                wo,
                router,
                routing_bias,
                experts,
                expert_gate_up_ptrs,
                expert_down_ptrs,
                dummy_gate_up,
            });
        }

        Ok(MiniMaxWeights {
            embed,
            final_norm,
            lm_head,
            layers,
            expert_manifest_cache: MiniMaxExpertManifestCache::new(
                MiniMaxManifestConfigIdentity::from_cfg(cfg),
            ),
        })
    }
}

// ──────────────────────────── Teardown ────────────────────────────
// Exhaustive-destructure frees so a future added GPU-bearing field is a
// compile error, not a silent VRAM leak. WeightTensors use `free_all`
// (buf + non-aliased ParoQuant rotation + AWQ sidecar).
//
// Every load layout owns ONE packed gate_up/down pair per layer (the indexed
// kernels reach all experts through the device-pointer tables into those
// blobs):
// - shard=None (single-GPU) and the safetensors path: the layer's single
//   `MiniMaxExpertWeights` IS the full packed pair over all experts.
// - shard=Some (EP packed blob, load_model_ep_minimax → LoadedModel →
//   unload_model): the layer's pair packs only the rank-owned expert subset;
//   non-owned experts alias the shared zeroed `dummy_gate_up` buffer, which
//   the layer additionally owns and frees (None otherwise).
// Per-expert `free_gpu` therefore releases each packed blob exactly once in
// every layout — no double free, no leak — and the dummy, when present, is
// reclaimed with its layer.

impl MiniMaxExpertWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MiniMaxExpertWeights { gate_up, down } = self;
        gate_up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl MiniMaxLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MiniMaxLayerWeights {
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            wq,
            wk,
            wv,
            wo,
            router,
            routing_bias,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            dummy_gate_up,
        } = self;
        let _ = gpu.free_tensor(attn_norm);
        let _ = gpu.free_tensor(ffn_norm);
        let _ = gpu.free_tensor(q_norm);
        let _ = gpu.free_tensor(k_norm);
        wq.free_all(gpu);
        wk.free_all(gpu);
        wv.free_all(gpu);
        wo.free_all(gpu);
        router.free_all(gpu);
        let _ = gpu.free_tensor(routing_bias);
        for e in experts {
            e.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(expert_gate_up_ptrs);
        let _ = gpu.free_tensor(expert_down_ptrs);
        if let Some(dummy) = dummy_gate_up {
            let _ = gpu.free_tensor(dummy);
        }
    }
}

impl MiniMaxWeights {
    /// Return all weight GPU buffers to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MiniMaxWeights {
            embed,
            final_norm,
            lm_head,
            layers,
            expert_manifest_cache: _, // pure CPU state — nothing to free
        } = self;
        let _ = gpu.free_tensor(embed);
        let _ = gpu.free_tensor(final_norm);
        lm_head.free_all(gpu);
        for layer in layers {
            layer.free_gpu(gpu);
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV cache. Buffers are eager-allocated (the model
/// is dense in its per-token working set); the KV cache is Q8.
pub struct MiniMaxState {
    pub kv: KvCache,
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
    /// captures. Survives turn resets (the graph stays valid for the same
    /// model — only weight pointers + device buffers are baked, and those are
    /// stable across turns).
    pub ar_warmed_up: bool,

    // attention scratch
    pub tmp: GpuTensor,         // [hidden] rmsnorm(h)
    pub x_rot: GpuTensor,       // [hidden] FWHT scratch (unused for Q8 attn)
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]
    pub flash_partials: GpuTensor,

    // residual + embedding
    pub h: GpuTensor, // [hidden] residual stream

    // moe scratch
    pub ffn_tmp: GpuTensor,       // [hidden] rmsnorm(h)
    pub ffn_x_rot: GpuTensor,     // [hidden] FWHT(rmsnorm(h)) for MQ4 experts
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k] i32-in-F32
    pub topk_weights: GpuTensor,  // [k]
    pub gate_batch: GpuTensor,    // [k*inter]
    pub up_batch: GpuTensor,      // [k*inter]
    pub rot_batch: GpuTensor,     // [k*inter]
    pub down_expanded: GpuTensor, // [k*hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub final_rot: GpuTensor,      // [hidden]
    pub logits: GpuTensor,         // [vocab]
}

impl MiniMaxState {
    /// Return all decode-scratch + KV-cache GPU buffers to the pool. Consumes
    /// self. Exhaustive destructure: a future added buffer field fails to
    /// compile here. `pos_host` is host memory (Box) — no GPU free.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MiniMaxState {
            kv,
            pos_buf,
            pos_host: _,
            max_seq: _,
            n_tokens: _,
            ar_warmed_up: _,
            tmp,
            x_rot,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            flash_partials,
            h,
            ffn_tmp,
            ffn_x_rot,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            final_norm_buf,
            final_rot,
            logits,
        } = self;
        let _ = kv.free_gpu(gpu);
        // pos_buf is a raw DeviceBuffer (no Drop impl) — free explicitly.
        let _ = gpu.hip.free(pos_buf);
        for t in [
            tmp,
            x_rot,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            flash_partials,
            h,
            ffn_tmp,
            ffn_x_rot,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            final_norm_buf,
            final_rot,
            logits,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    pub fn new(gpu: &mut Gpu, cfg: &MiniMaxConfig) -> Result<Self, String> {
        // Cap the KV cache so the real 204800-ctx config doesn't OOM; callers
        // that need a specific window use `new_with_max_seq`.
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &MiniMaxConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        // `attention_q8_0_kv` (single-token decode) stages its per-head score
        // buffer in LDS sized by `max_seq`: `(max_seq + block + head_dim) * 4`
        // bytes must fit the 64 KB per-block shared-memory limit on every RDNA
        // arch, so the single-token attention launch is hard-bounded near 16K
        // context. A larger requested window blows the launch
        // (`hipModuleLaunchKernel: invalid argument` — observed serving the
        // 86 GB mq2-lloyd on gfx1151 with the daemon's default window: prefill
        // via the batched kernel succeeds, then the first decode token dies).
        // Clamp the served window here so the cache, the geometry hint, and the
        // flash-partial sizing all stay launch-valid. Proper fix = tile the
        // scores out of LDS (flash-style); tracked as a follow-up.
        const MINIMAX_ATTN_LDS_MAX_SEQ: usize = 12288;
        let max_seq = if max_seq > MINIMAX_ATTN_LDS_MAX_SEQ {
            eprintln!(
                "[minimax] requested max_seq {max_seq} exceeds the single-token \
                 attention LDS bound; clamping to {MINIMAX_ATTN_LDS_MAX_SEQ} \
                 (decode scores must fit the 64 KB per-block shared-mem limit)"
            );
            MINIMAX_ATTN_LDS_MAX_SEQ
        } else {
            max_seq
        };
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_exp = cfg.num_local_experts;
        let k = cfg.num_experts_per_tok;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("minimax: ensure_mq_signs: {e:?}"))?;

        let dims = hipfire_runtime::llama::KvDims {
            layers: hipfire_runtime::llama::KvLayers::Flat(cfg.num_hidden_layers),
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            max_seq, // already clamped to MINIMAX_ATTN_LDS_MAX_SEQ above
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
        .map_err(|e| format!("minimax: kv cache: {e:?}"))?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("minimax: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("minimax: alloc {label}: {e:?}"))
        };
        // Flash-attn partials: [n_heads * max_tiles * (2+head_dim)]; max_tiles
        // bounded by ceil(max_seq/tile). Use a generous tile bound of 64.
        let max_tiles = (max_seq / 256).max(1) + 1;
        let flash_partials = alloc(
            gpu,
            cfg.num_attention_heads * max_tiles * (2 + cfg.head_dim),
            "flash_partials",
        )?;

        Ok(MiniMaxState {
            kv,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            ar_warmed_up: false,
            tmp: alloc(gpu, hidden, "tmp")?,
            x_rot: alloc(gpu, hidden, "x_rot")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            flash_partials,
            h: alloc(gpu, hidden, "h")?,
            ffn_tmp: alloc(gpu, hidden, "ffn_tmp")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * inter, "gate_batch")?,
            up_batch: alloc(gpu, k * inter, "up_batch")?,
            rot_batch: alloc(gpu, k * inter, "rot_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            final_rot: alloc(gpu, hidden, "final_rot")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
    }
}

// ──────────────── ModelSource (safetensors) load helpers ────────────────

/// Determine whether a tensor's bytes represent F16 values or a quantized
/// format by comparing the byte count against the expected sizes.
fn classify_tensor_bytes(bytes: &[u8], numel: usize, dtype: &str) -> (bool, bool) {
    // BF16 has 2 bytes/element same as F16, so it must be explicitly excluded.
    if dtype == "BF16" {
        return (false, false);
    }
    let is_f16 = bytes.len() == numel * 2;
    // Q8_0: 34 bytes per block of 32 elements:
    //   [f16 scale (2 bytes)] [32 × i8 (32 bytes)]
    let q8_0_expected = ((numel + 31) / 32) * 34;
    let is_q8_0 = !is_f16 && bytes.len() == q8_0_expected;
    (is_f16, is_q8_0)
}

/// Load a 1D norm vector from a ModelSource (F16) → F32 GpuTensor.
/// Mirrors `load_norm` but sources from `&dyn ModelSource`.
fn load_norm_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (info, bytes) = source
        .tensor_data(name)
        .ok_or_else(|| format!("minimax: norm '{name}' missing in source"))?;
    let numel: usize = info.shape.iter().product();
    let (is_f16, _) = classify_tensor_bytes(bytes, numel, info.dtype.as_str());
    let f32_data: Vec<f32> = if is_f16 {
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else if info.dtype == "F16" {
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else if info.dtype == "BF16" {
        hipfire_runtime::safetensors_source::bf16_bytes_to_f32(bytes)
    } else {
        return Err(format!(
            "minimax: expected F16 norm for {name}, got {} bytes (numel={})",
            bytes.len(),
            numel
        ));
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("minimax: upload norm {name}: {e:?}"))
}

/// Load a MiniMax AWQ shared-scale sidecar (1D F16, length k) from
/// a ModelSource → F32 GpuTensor. Mirrors `load_mm_awq_scale`.
fn load_mm_awq_scale_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    k: usize,
) -> Option<GpuTensor> {
    let (info, bytes) = source.tensor_data(name)?;
    let numel: usize = info.shape.iter().product();
    let (is_f16, _) = classify_tensor_bytes(bytes, numel, info.dtype.as_str());
    if !is_f16 {
        return None;
    }
    if bytes.len() != k * 2 {
        eprintln!(
            "minimax AWQ sidecar {name}: {} bytes != {} (k*2); skipping",
            bytes.len(),
            k * 2
        );
        return None;
    }
    let f32_data: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_data.len()]).ok()
}

/// Load a quantized 2D weight from a ModelSource → WeightTensor.
/// Detects F16 vs Q8_0 vs quantized (Raw) from byte count.
fn load_wt_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, bytes) = source
        .tensor_data(name)
        .ok_or_else(|| format!("minimax: tensor '{name}' missing in source"))?;
    let numel: usize = info.shape.iter().product();
    let (is_f16, is_q8_0) = classify_tensor_bytes(bytes, numel, info.dtype.as_str());

    let dtype = if is_f16 {
        DType::F16
    } else if is_q8_0 {
        DType::Q8_0
    } else {
        // Quantized format (MQ4G256, MQ3G256Lloyd, etc.) — store as Raw.
        // The forward dispatcher handles Raw as quantized via gemv_auto.
        DType::Raw
    };

    let buf = gpu
        .upload_raw(bytes, &[bytes.len()])
        .map_err(|e| format!("minimax: upload '{name}': {e:?}"))?;
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

/// Load MiniMax weights from a `&dyn ModelSource` (safetensors or HFQ wrapper).
/// Mirrors `MiniMaxWeights::load` but reads tensor data via
/// `ModelSource::tensor_data()` instead of `HfqFile::tensor_data_vec()`.
/// Tensor names match those used in the HFQ path.
pub fn load_weights_from_safetensors(
    source: &dyn ModelSource,
    cfg: &MiniMaxConfig,
    gpu: &mut Gpu,
) -> Result<MiniMaxWeights, String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;

    // Globals.
    let (_, embed_bytes) = source
        .tensor_data("model.embed_tokens.weight")
        .ok_or_else(|| "minimax: model.embed_tokens.weight missing in source".to_string())?;
    let embed = gpu
        .upload_raw(embed_bytes, &[embed_bytes.len()])
        .map_err(|e| format!("minimax: upload embed: {e:?}"))?;
    let final_norm = load_norm_from_source(source, gpu, "model.norm.weight", &[hidden])?;
    let lm_head = load_wt_from_source(source, gpu, "lm_head.weight", cfg.vocab_size, hidden)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let attn_norm = load_norm_from_source(
            source,
            gpu,
            &format!("{p}.input_layernorm.weight"),
            &[hidden],
        )?;
        let ffn_norm = load_norm_from_source(
            source,
            gpu,
            &format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
        )?;
        let q_norm = load_norm_from_source(
            source,
            gpu,
            &format!("{p}.self_attn.q_norm.weight"),
            &[q_dim],
        )?;
        let k_norm = load_norm_from_source(
            source,
            gpu,
            &format!("{p}.self_attn.k_norm.weight"),
            &[kv_dim],
        )?;
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
            &format!("{p}.self_attn.o_proj.weight"),
            hidden,
            q_dim,
        )?;

        let router = load_wt_from_source(
            source,
            gpu,
            &format!("{p}.block_sparse_moe.gate.weight"),
            n_exp,
            hidden,
        )?;
        let routing_bias = load_norm_from_source(
            source,
            gpu,
            &format!("{p}.block_sparse_moe.e_score_correction_bias"),
            &[n_exp],
        )?;

        // Routed experts: pack ALL experts of this layer into ONE gate_up
        // blob + ONE down blob (same pattern as the HFQ path).
        let mut gu_combined: Vec<u8> = Vec::new();
        let mut dn_combined: Vec<u8> = Vec::new();
        let mut gu_stride = 0usize;
        let mut dn_stride = 0usize;

        for e in 0..n_exp {
            let ep = format!("{p}.block_sparse_moe.experts.{e}");
            let w1_name = format!("{ep}.w1.weight");
            let w3_name = format!("{ep}.w3.weight");
            let w2_name = format!("{ep}.w2.weight");

            let (_, w1_data) = source
                .tensor_data(&w1_name)
                .ok_or_else(|| format!("minimax: missing {w1_name}"))?;
            let (_, w3_data) = source
                .tensor_data(&w3_name)
                .ok_or_else(|| format!("minimax: missing {w3_name}"))?;
            let (_, w2_data) = source
                .tensor_data(&w2_name)
                .ok_or_else(|| format!("minimax: missing {w2_name}"))?;

            let gu_len = w1_data.len() + w3_data.len();
            if e == 0 {
                gu_stride = gu_len;
                dn_stride = w2_data.len();
                let cap = n_exp;
                gu_combined.reserve(gu_len * cap);
                dn_combined.reserve(w2_data.len() * cap);
            } else if gu_len != gu_stride || w2_data.len() != dn_stride {
                return Err(format!(
                    "minimax L{l}E{e}: non-uniform expert stride \
                     (gate_up {gu_len}/{gu_stride}, down {}/{}); \
                     packed layout requires equal-size experts",
                    w2_data.len(),
                    dn_stride
                ));
            }
            gu_combined.extend_from_slice(w1_data);
            gu_combined.extend_from_slice(w3_data);
            dn_combined.extend_from_slice(w2_data);
        }

        let qt = DType::Raw; // safetensors quantized weights stored as raw bytes
        let mut gate_up = {
            let buf = gpu
                .upload_raw(&gu_combined, &[gu_combined.len()])
                .map_err(|e| format!("minimax: pack gate_up L{l}: {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: qt,
                m: 2 * inter,
                k: hidden,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        };
        let mut down = {
            let buf = gpu
                .upload_raw(&dn_combined, &[dn_combined.len()])
                .map_err(|e| format!("minimax: pack down L{l}: {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: qt,
                m: hidden,
                k: inter,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        };
        drop(gu_combined);
        drop(dn_combined);

        gate_up.awq_scale = load_mm_awq_scale_from_source(
            source,
            gpu,
            &format!("{p}.block_sparse_moe.awq_scale_gate_up.weight"),
            hidden,
        );
        if hipfire_config::developer_var_os("HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ").is_some() {
            down.awq_scale = load_mm_awq_scale_from_source(
                source,
                gpu,
                &format!("{p}.block_sparse_moe.awq_scale_down.weight"),
                inter,
            );
        }
        if gate_up.awq_scale.is_some() {
            eprintln!("minimax: AWQ scales attached at L{l} (shared per-layer)");
        }

        let gu_base = gate_up.buf.buf.as_ptr() as u64;
        let dn_base = down.buf.buf.as_ptr() as u64;
        let experts = vec![MiniMaxExpertWeights { gate_up, down }];

        // Device pointer tables: n_exp u64 device addresses, stored as
        // [2*n_exp] F32 (8 bytes/ptr). Single-GPU: base + e*stride into the
        // full packed blob.
        let gu_bytes: Vec<u8> = (0..n_exp)
            .flat_map(|e| {
                let ptr = gu_base + (e * gu_stride) as u64;
                ptr.to_ne_bytes()
            })
            .collect();
        let dn_bytes: Vec<u8> = (0..n_exp)
            .flat_map(|e| {
                let ptr = dn_base + (e * dn_stride) as u64;
                ptr.to_ne_bytes()
            })
            .collect();

        let expert_gate_up_ptrs = gpu
            .alloc_tensor(&[2 * n_exp], DType::F32)
            .map_err(|e| format!("minimax: alloc gu_ptrs: {e:?}"))?;
        let expert_down_ptrs = gpu
            .alloc_tensor(&[2 * n_exp], DType::F32)
            .map_err(|e| format!("minimax: alloc dn_ptrs: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
            .map_err(|e| format!("minimax: htod gu_ptrs: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
            .map_err(|e| format!("minimax: htod dn_ptrs: {e:?}"))?;

        layers.push(MiniMaxLayerWeights {
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            wq,
            wk,
            wv,
            wo,
            router,
            routing_bias,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            dummy_gate_up: None,
        });
    }

    Ok(MiniMaxWeights {
        embed,
        final_norm,
        lm_head,
        layers,
        expert_manifest_cache: MiniMaxExpertManifestCache::new(
            MiniMaxManifestConfigIdentity::from_cfg(cfg),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::MiniMaxM2;
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::moe_plan::{MoEExecutionKind, MoEExecutionPolicy};
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
    use hipfire_runtime::weight_manifest::{
        resolve_expert_manifest_for_policy, ExpertExecutionIdentity,
    };

    /// TP-valid fixture: even attention head counts (Tp=2 divisibility) and
    /// inter=512 so every projected TP local slice is 256-aligned.
    fn test_config_512() -> MiniMaxConfig {
        MiniMaxConfig {
            vocab_size: 16,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 512,
            num_local_experts: 4,
            num_experts_per_tok: 2,
            rotary_dim: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            use_qk_norm: true,
            use_routing_bias: true,
            scoring_func: "sigmoid".into(),
            num_mtp_modules: 0,
            reap_keep: None,
        }
    }

    fn test_config_512_layers(n: usize) -> MiniMaxConfig {
        let mut cfg = test_config_512();
        cfg.num_hidden_layers = n;
        cfg
    }

    /// `test_config_512` with an intermediate dim the TP projection path must
    /// refuse (8/2 = 4 is not a multiple of 256) — a genuine resolution
    /// failure used to pin the cached-failure semantics.
    fn test_config_8() -> MiniMaxConfig {
        let mut cfg = test_config_512();
        cfg.intermediate_size = 8;
        cfg
    }

    fn projection_policy(kind: MoEExecutionKind, ranks: usize) -> MoEExecutionPolicy {
        let mesh = match kind {
            MoEExecutionKind::Single => DeviceMesh::single(),
            MoEExecutionKind::Tp => DeviceMesh::rect(&[(DimKind::Tp, ranks)]),
            MoEExecutionKind::Ep => DeviceMesh::rect(&[(DimKind::Ep, ranks)]),
        };
        MoEExecutionPolicy::new(kind, mesh).unwrap()
    }

    #[test]
    fn expert_group_spec_declares_truthful_f16_footprint() {
        // One expert's resident F16 footprint = gate (w1) + up (w3) + down
        // (w2) matrices, each hidden x inter elements x 2 bytes.
        let cfg = test_config_512();
        let specs = MiniMaxM2::expert_group_manifest(&cfg, &MoEExecutionPolicy::single());
        assert_eq!(
            specs[0].resources.bytes_per_expert,
            3 * cfg.hidden_size * cfg.intermediate_size * 2,
            "truthful declared F16 footprint is 3 * hidden * inter * 2"
        );
    }

    #[test]
    fn authority_plan_borrow_rejects_router_identity_mismatch() {
        // Batched semantic admission: the direct batched indexed sigmoid
        // kernels require the exact `sigmoid_topk` router identity. A plan
        // declaring any other router identity must be refused before kernels
        // use that layer.
        let cfg = test_config_512();
        let single = MoEExecutionPolicy::single();
        let specs = MiniMaxM2::expert_group_manifest(&cfg, &single);
        let manifest = MiniMaxM2::weight_manifest(&cfg);
        let resolution = resolve_expert_manifest_for_policy(&specs, &manifest, &single).unwrap();
        let mut wrong_router = resolution.clone();
        wrong_router.plans[0].router_identity = "bias_aware_topk".into();
        let authority = SingleMoeAuthority::new(&single, &wrong_router);
        let err = authority.plan_for_layer(0).unwrap_err();
        assert!(err.contains("sigmoid_topk"), "got: {err}");
    }

    /// Canned seam output: a valid resolution tagged with the requested
    /// policy kind + rank count, so a winner-tagged result can be proven to
    /// belong to the requesting thread's OWN key.
    fn canned_resolution_tagged(
        policy: &MoEExecutionPolicy,
        tag: &str,
    ) -> ExpertManifestResolution {
        use hipfire_runtime::tp_shard::ExpertAssign;
        use hipfire_runtime::weight_manifest::{
            ExpertGroupPlan, ExpertParallelism, ExpertResourceRequirements, ExpertSourceLayout,
        };
        ExpertManifestResolution {
            plans: vec![ExpertGroupPlan {
                group: format!("{tag}:{:?}:{}", policy.kind(), policy.rank_count()),
                layer: Some(0),
                n_experts: 4,
                group_size: policy.rank_count(),
                parallelism: match policy.kind() {
                    MoEExecutionKind::Single => ExpertParallelism::Single,
                    MoEExecutionKind::Tp => ExpertParallelism::TensorParallel,
                    MoEExecutionKind::Ep => ExpertParallelism::ExpertParallel,
                },
                assignment: ExpertAssign::Stride,
                experts: Vec::new(),
                source_layout: ExpertSourceLayout::PackedSeparate {
                    gate: "experts_gate".into(),
                    up: "experts_up".into(),
                    down: "experts_down".into(),
                    sidecars: Vec::new(),
                },
                resources: ExpertResourceRequirements {
                    bytes_per_expert: 1,
                    alignment: 256,
                },
                router: "router".into(),
                router_identity: "sigmoid_topk".into(),
                allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
                collective: None,
            }],
            layer_collectives: Vec::new(),
        }
    }

    #[test]
    fn first_call_wrong_config_refuses_and_leaves_cache_unseeded() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let calls = Arc::new(AtomicUsize::new(0));
        let cfg_a = test_config_512();
        // Success-capable mismatch: cfg_b would resolve fine, but it is NOT
        // the config the cache was bound to at load.
        let cfg_b = test_config_512_layers(2);
        let cache = MiniMaxExpertManifestCache::with_seams(
            MiniMaxManifestConfigIdentity::from_cfg(&cfg_a),
            Box::new({
                let calls = Arc::clone(&calls);
                move |_cfg: &MiniMaxConfig, policy: &MoEExecutionPolicy| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(canned_resolution_tagged(policy, "a"))
                }
            }),
            None,
        );
        let single = MoEExecutionPolicy::single();
        // WRONG FIRST request: refused explicitly, and the cache stays
        // unseeded — the resolver (launch-capable seam) never runs.
        let err = cache.get_or_resolve(&cfg_b, &single).unwrap_err();
        assert!(err.contains("refusing reuse"), "got: {err}");
        assert!(err.contains("config"), "got: {err}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a wrong first request must leave the cache empty/unbound"
        );
        // The load-bound config then resolves fresh.
        let first = cache.get_or_resolve(&cfg_a, &single).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Repeated load-bound requests reuse the identical cached object.
        let second = cache.get_or_resolve(&cfg_a, &single).unwrap();
        assert!(std::ptr::eq(first, second));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_reuses_identical_object_and_rejects_policy_mismatch() {
        let cfg = test_config_512();
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        let single = MoEExecutionPolicy::single();
        let first = cache.get_or_resolve(&cfg, &single).unwrap();
        let second = cache.get_or_resolve(&cfg, &single).unwrap();
        assert!(std::ptr::eq(first, second));
        assert_eq!(first, second);
        assert_eq!(first.plans[0].collective, None);
        // Same kind (Tp) but a different mesh — both constructible — refuses
        // reuse instead of re-resolving.
        let tp2 = projection_policy(MoEExecutionKind::Tp, 2);
        let tp1 = projection_policy(MoEExecutionKind::Tp, 1);
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        assert_eq!(
            cache.get_or_resolve(&cfg, &tp2).unwrap().plans[0].group_size,
            2
        );
        let err = cache.get_or_resolve(&cfg, &tp1).unwrap_err();
        assert!(err.contains("exact policy"), "got: {err}");
        assert!(err.contains("Tp"), "got: {err}");
        let ep2 = projection_policy(MoEExecutionKind::Ep, 2);
        let err = cache.get_or_resolve(&cfg, &ep2).unwrap_err();
        assert!(err.contains("exact policy"), "got: {err}");
    }

    #[test]
    fn cache_caches_failure_and_stays_bound() {
        // TP projection of the invalid fixture is refused (local slice 8/2=4
        // is not a multiple of 256) — a genuine resolution failure cached as
        // failure: no fallback, no re-resolution, no weakening.
        let cfg = test_config_8();
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        let tp2 = projection_policy(MoEExecutionKind::Tp, 2);
        let err1 = cache.get_or_resolve(&cfg, &tp2).unwrap_err();
        assert!(err1.contains("multiple of 256"), "got: {err1}");
        let err2 = cache.get_or_resolve(&cfg, &tp2).unwrap_err();
        assert_eq!(err1, err2, "cached failure must stay the same failure");
        // The cache stays bound to the FAILED policy: a policy that would
        // otherwise resolve (Single) is refused — proof the failure was
        // cached, not re-resolved.
        let err3 = cache
            .get_or_resolve(&cfg, &MoEExecutionPolicy::single())
            .unwrap_err();
        assert!(err3.contains("exact policy"), "got: {err3}");
    }

    #[test]
    fn single_policy_object_is_stable() {
        let cfg = test_config_512();
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        let p1 = cache.single_policy();
        let p2 = cache.single_policy();
        assert!(
            std::ptr::eq(p1, p2),
            "single policy must be one stable object"
        );
        assert_eq!(p1.rank_count(), 1);
        assert_eq!(p1.kind(), MoEExecutionKind::Single);
        let first = cache.get_or_resolve(&cfg, p1).unwrap();
        let second = cache.get_or_resolve(&cfg, p2).unwrap();
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn initialized_tp_lookup_constructs_no_owned_key() {
        use std::sync::atomic::Ordering;
        // The cached fast path must compare the BORROWED request against the
        // stored key — no owned MoEExecutionPolicy/mesh-axis clone on
        // repeated initialized lookups. The cfg(test) construction counter
        // counts cold-path owned-key constructions.
        let cfg = test_config_512();
        let policy = projection_policy(MoEExecutionKind::Tp, 2);
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        let first = cache.get_or_resolve(&cfg, &policy).unwrap();
        assert_eq!(cache.cold_constructs.load(Ordering::SeqCst), 1);
        for _ in 0..100 {
            let again = cache.get_or_resolve(&cfg, &policy).unwrap();
            assert!(std::ptr::eq(first, again));
        }
        assert_eq!(
            cache.cold_constructs.load(Ordering::SeqCst),
            1,
            "repeated initialized TP/EP lookup must perform zero owned-key/policy constructions"
        );
    }

    #[test]
    fn concurrent_same_key_resolves_once_via_race_seam() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};
        const N: usize = 8;
        let calls = Arc::new(AtomicUsize::new(0));
        let rendezvous = Arc::new(Barrier::new(N));
        let cfg = test_config_512();
        let cache = MiniMaxExpertManifestCache::with_seams(
            MiniMaxManifestConfigIdentity::from_cfg(&cfg),
            Box::new({
                let calls = Arc::clone(&calls);
                move |_cfg: &MiniMaxConfig, policy: &MoEExecutionPolicy| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(canned_resolution_tagged(policy, "same"))
                }
            }),
            Some(Arc::new(move || {
                rendezvous.wait();
            })),
        );
        let policy = MoEExecutionPolicy::single();
        // Shared references: the spawn closures move copies of these.
        let cache = &cache;
        let cfg = &cfg;
        let policy = &policy;
        let start = Arc::new(Barrier::new(N));
        let results: Vec<Result<usize, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..N)
                .map(|_| {
                    let start = Arc::clone(&start);
                    scope.spawn(move || {
                        start.wait();
                        cache
                            .get_or_resolve(&cfg, &policy)
                            .map(|r| r as *const ExpertManifestResolution as usize)
                            .map_err(|e| e)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // Every contender observed the empty cell, rendezvoused at the
        // cfg(test) hook, and then all entered get_or_init: exactly one
        // resolver invocation, one cached object.
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "same-key resolution must run exactly once"
        );
        assert_eq!(
            cache.cold_constructs.load(Ordering::SeqCst),
            1,
            "only the winning initializer may construct the owned key/policy"
        );
        assert!(results.iter().all(|r| r.is_ok()), "{results:?}");
        let addr = results[0].as_ref().unwrap();
        assert!(
            results.iter().all(|r| r.as_ref().unwrap() == addr),
            "all callers must observe the IDENTICAL cached object"
        );
    }

    #[test]
    fn concurrent_distinct_policy_first_calls_one_winner_via_race_seam() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};
        const N: usize = 8;
        let calls = Arc::new(AtomicUsize::new(0));
        let rendezvous = Arc::new(Barrier::new(N));
        let cfg = test_config_512();
        let cache = MiniMaxExpertManifestCache::with_seams(
            MiniMaxManifestConfigIdentity::from_cfg(&cfg),
            Box::new({
                let calls = Arc::clone(&calls);
                move |_cfg: &MiniMaxConfig, policy: &MoEExecutionPolicy| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(canned_resolution_tagged(policy, "policy"))
                }
            }),
            Some(Arc::new(move || {
                rendezvous.wait();
            })),
        );
        // Shared references: the spawn closures move copies of these.
        let cache = &cache;
        let cfg = &cfg;
        let start = Arc::new(Barrier::new(N));
        let policies: Vec<MoEExecutionPolicy> = (1..=N)
            .map(|r| projection_policy(MoEExecutionKind::Tp, r))
            .collect();
        let results: Vec<(MoEExecutionPolicy, Result<String, String>)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = policies
                    .iter()
                    .map(|policy| {
                        let start = Arc::clone(&start);
                        scope.spawn(move || {
                            start.wait();
                            let out = cache
                                .get_or_resolve(&cfg, policy)
                                .map(|r| r.plans[0].group.clone());
                            (policy.clone(), out)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "distinct first calls: exactly one resolver invocation"
        );
        assert_eq!(
            cache.cold_constructs.load(Ordering::SeqCst),
            1,
            "only the winning initializer may construct the owned key/policy"
        );
        let oks: Vec<&(MoEExecutionPolicy, Result<String, String>)> =
            results.iter().filter(|(_, r)| r.is_ok()).collect();
        let mismatches: Vec<&(MoEExecutionPolicy, Result<String, String>)> =
            results.iter().filter(|(_, r)| r.is_err()).collect();
        assert_eq!(oks.len(), 1, "exactly one distinct-key caller may win");
        assert_eq!(
            mismatches.len(),
            N - 1,
            "every other caller gets a mismatch"
        );
        let (winner_policy, winner_group) = (oks[0].0.clone(), oks[0].1.as_ref().unwrap());
        assert_eq!(
            winner_group,
            &format!(
                "policy:{:?}:{}",
                winner_policy.kind(),
                winner_policy.rank_count()
            ),
            "winner must observe its OWN key's resolution"
        );
        for (policy, r) in &mismatches {
            let err = r.as_ref().unwrap_err();
            assert!(err.contains("refusing reuse"), "policy {policy:?}: {err}");
        }
    }

    #[test]
    fn authority_sequential_and_batched_share_identical_objects() {
        let cfg = test_config_512();
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        let seq = SingleMoeAuthority::from_cache(&cache, &cfg).unwrap();
        let batch = SingleMoeAuthority::from_cache(&cache, &cfg).unwrap();
        assert!(std::ptr::eq(seq.policy(), batch.policy()));
        assert!(std::ptr::eq(seq.resolution(), batch.resolution()));
        let again = SingleMoeAuthority::from_cache(&cache, &cfg).unwrap();
        assert!(std::ptr::eq(batch.policy(), again.policy()));
        assert!(std::ptr::eq(batch.resolution(), again.resolution()));
    }

    #[test]
    fn authority_refuses_mismatched_config_before_admission() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let calls = Arc::new(AtomicUsize::new(0));
        let cfg_a = test_config_512();
        let cache = MiniMaxExpertManifestCache::with_seams(
            MiniMaxManifestConfigIdentity::from_cfg(&cfg_a),
            Box::new({
                let calls = Arc::clone(&calls);
                move |_cfg: &MiniMaxConfig, policy: &MoEExecutionPolicy| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(canned_resolution_tagged(policy, "authority"))
                }
            }),
            None,
        );
        let a = SingleMoeAuthority::from_cache(&cache, &cfg_a).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let mut cfg_b = test_config_512();
        cfg_b.hidden_size = 128;
        let err = SingleMoeAuthority::from_cache(&cache, &cfg_b).unwrap_err();
        assert!(err.contains("refusing reuse"), "got: {err}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a mismatched config must never reach the resolver (launch-capable) callback"
        );
        assert!(a.plan_for_layer(0).is_ok());
    }

    #[test]
    fn authority_plan_borrow_validates_layer_and_single_admission() {
        let cfg = test_config_512_layers(3);
        let cache = MiniMaxExpertManifestCache::new(MiniMaxManifestConfigIdentity::from_cfg(&cfg));
        let authority = SingleMoeAuthority::from_cache(&cache, &cfg).unwrap();
        for l in 0..3 {
            let plan = authority.plan_for_layer(l).unwrap();
            assert_eq!(plan.layer, Some(l));
            // Exact batched semantic admission on the borrowed plan.
            assert_eq!(plan.router_identity, "sigmoid_topk");
            assert!(plan
                .allowed_executions
                .contains(&ExpertExecutionIdentity::IndexedQuantized));
        }
        // One residual dense row-parallel (wo) collective per layer; the
        // claimed expert sources are excluded exactly once per layer.
        assert_eq!(authority.resolution().layer_collectives.len(), 3);
        assert!(authority
            .resolution()
            .layer_collectives
            .iter()
            .all(|(l, c)| *l < 3
                && *c
                    == hipfire_runtime::multi_gpu::CollectiveHint::AllReduce {
                        kind: hipfire_runtime::multi_gpu::DimKind::Tp
                    }));
        let err = authority.plan_for_layer(3).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
        let mut mis_scoped = authority.resolution().clone();
        mis_scoped.plans[0].layer = Some(7);
        let bad = SingleMoeAuthority::new(authority.policy(), &mis_scoped);
        let err = bad.plan_for_layer(0).unwrap_err();
        assert!(err.contains("Some(0)"), "got: {err}");
        let tp_policy = projection_policy(MoEExecutionKind::Tp, 2);
        let specs = MiniMaxM2::expert_group_manifest(&cfg, &tp_policy);
        let manifest = MiniMaxM2::weight_manifest(&cfg);
        let tp_resolution =
            resolve_expert_manifest_for_policy(&specs, &manifest, &tp_policy).unwrap();
        let single = MoEExecutionPolicy::single();
        let tp_authority = SingleMoeAuthority::new(&single, &tp_resolution);
        let err = tp_authority.plan_for_layer(0).unwrap_err();
        assert!(err.contains("not Single-admitted"), "got: {err}");
        let empty = ExpertManifestResolution {
            plans: Vec::new(),
            layer_collectives: Vec::new(),
        };
        let empty_authority = SingleMoeAuthority::new(&single, &empty);
        assert!(empty_authority.plan_for_layer(0).is_err());
    }

    #[test]
    fn authority_plan_borrow_rejects_non_indexed_execution() {
        // Batched semantic admission: the direct batched kernels are the
        // indexed quantized family; a plan that does not admit
        // `IndexedQuantized` must be refused before kernels use that layer.
        let cfg = test_config_512();
        let single = MoEExecutionPolicy::single();
        let specs = MiniMaxM2::expert_group_manifest(&cfg, &single);
        let manifest = MiniMaxM2::weight_manifest(&cfg);
        let resolution = resolve_expert_manifest_for_policy(&specs, &manifest, &single).unwrap();
        let mut grouped = resolution.clone();
        grouped.plans[0].allowed_executions = vec![ExpertExecutionIdentity::GroupedQuantized];
        let authority = SingleMoeAuthority::new(&single, &grouped);
        let err = authority.plan_for_layer(0).unwrap_err();
        assert!(err.contains("IndexedQuantized"), "got: {err}");
    }
}
