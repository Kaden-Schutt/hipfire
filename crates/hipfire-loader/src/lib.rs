// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Top-of-DAG model loader. Owns `LoadedModel`, the carrier registry,
//! and `load_model` — the single arch-dispatch point for the daemon.

mod carriers;
pub use carriers::*;

/// Speculative-decode build/glue (RAII slot guard now; `DflashSpeculator` +
/// `build_speculator` at Stages 1-2). Lives here at the top of the DAG where
/// both `LoadedModel`/`ModelState` and the arch crates are in scope.
pub mod model_parallel;
pub mod session_state;
pub mod spec_build;
pub use model_parallel::{ModelParallel, ModelParallelKind, PipelineImpl};

use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind, Gpus};
use hipfire_runtime::spec::{SpecEmit, SpecEmitCtx, SpecTargetGuard, Speculator};
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use rdna_compute::Gpu;
use std::path::Path;

// ─── Object-safe Carrier trait ──────────────────────────────────────

/// One arch's complete load contract. Object-safe → usable as `&dyn Carrier`.
pub trait Carrier: Send + Sync {
    fn name(&self) -> &'static str;
    /// Whether this carrier claims a given `arch_id`. `is_dir` distinguishes
    /// the two namespaces: HFQ-header ids (`HfqFile::arch_id`) vs the
    /// `derive_arch_id` ids emitted for safetensors directories. Kept as a
    /// pure `(u32, bool) -> bool` fn so the registry's disjointness can be
    /// unit-tested without constructing a real `ModelSource`.
    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool;
    /// Default probe delegates to [`Carrier::claims_arch_id`]; carriers only
    /// implement the pure id predicate.
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src.arch_id(), Some(id) if self.claims_arch_id(id, src.is_dir()))
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String>;

    /// Borrow this model's spec-decode target out of `state`, arch-erased as a
    /// [`SpecTargetGuard`]. This is the daemon's single dispatch for the
    /// spec-decode path — it then only ever sees `&mut dyn SpecTarget`, never an
    /// arch type. Default (AR-only carriers): `Err` WITHOUT touching `state` —
    /// only an override may `state.take()`.
    fn spec_target_guard<'m>(
        &self,
        _state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        Err(format!("{}: spec-decode target unsupported", self.name()))
    }

    /// Construct this model's per-token spec-decode emitter from the
    /// model-independent [`SpecEmitCtx`]. The arch's emitter extracts its own
    /// grammar schema from `ctx.tools` (raw JSON) internally. Default: `Err`
    /// (arch has no spec emitter).
    fn make_spec_emitter<'a>(
        &self,
        _ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Err(format!("{}: spec emitter unsupported", self.name()))
    }
}

/// The single registry lookup the daemon's spec path routes through: resolve the
/// carrier that claims `arch_id`, so the daemon never arch-matches for the
/// spec-decode guard / emitter. `is_dir` is `false` here because every
/// spec-capable arch is disjoint on the bare HFQ `arch_id` (qwen35 5|6, llama
/// 0|1, qwen2 7, deepseek4 9) and all carriers ignore the dir flag; if a future
/// arch needs HFQ-vs-dir disambiguation in the spec path, thread a retained
/// `is_dir` from load time rather than re-deriving it here.
pub fn carrier_for(arch_id: u32) -> Option<&'static dyn Carrier> {
    REGISTRY
        .iter()
        .copied()
        .find(|c| c.claims_arch_id(arch_id, false))
}

// ─── Registry ─────────────────────────────────────────────────────────

const REGISTRY: &[&dyn Carrier] = &[
    &Qwen2Carrier,
    &Qwen35Carrier,
    &LlamaCarrier,
    &DotsOcrCarrier,
    &Deepseek4Carrier,
    &MinimaxCarrier,
    &Lfm2MoeCarrier,
    &Cohere2MoeCarrier,
];

// ─── Constants ────────────────────────────────────────────────────────

/// Built-in Qwen3.5/3.6 chat template (froggeric/Qwen at HF).
/// Used when no per-model or env-override template is available.
const FROGGERIC_QWEN35_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/qwen35-froggeric-v20.jinja");

/// Built-in LFM2.5 chat template.
const LFM2_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/lfm2-liquidai.jinja");

// ─── Eviction policy wrapper ──────────────────────────────────────────

/// Eviction policy wrapper — dispatches to plain TriAttention or CASK m-folding.
pub enum Eviction {
    Plain(EvictionCtx),
    Cask(CaskCtx),
}

impl Eviction {
    pub fn maybe_evict(
        &self,
        gpu: &mut rdna_compute::Gpu,
        kv: &mut llama::KvCache,
        physical: usize,
    ) -> hip_bridge::HipResult<Option<hipfire_runtime::triattn::EvictionResult>> {
        match self {
            Eviction::Plain(c) => c.maybe_evict(gpu, kv, physical),
            Eviction::Cask(c) => c.maybe_evict(gpu, kv, physical),
        }
    }
    pub fn budget(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.budget,
            Eviction::Cask(c) => c.base.budget,
        }
    }
    pub fn beta(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.beta,
            Eviction::Cask(c) => c.base.beta,
        }
    }
    pub fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        match self {
            Eviction::Plain(c) => c.free_gpu(gpu),
            Eviction::Cask(c) => c.free_gpu(gpu),
        }
    }
}

fn effective_cask_m_folding(cask_m_folding: bool, draft_path: Option<&str>) -> bool {
    cask_m_folding && draft_path.is_none()
}

// `DdtreeState`, `DflashState`, `load_dflash_state`, and the `DflashSpeculator`
// impl now live in `hipfire_arch_qwen35::dflash_spec` — all qwen35 + runtime
// types, so the loader only constructs and routes them, never owns the DFlash
// mechanics.

// ─── AsstTurnCache ────────────────────────────────────────────────────

/// Per-turn token cache for V4F prefix-cache stability.
pub struct AsstTurnCache {
    cap: Option<usize>,
    map: std::collections::HashMap<u64, Vec<u32>>,
    order: std::collections::VecDeque<u64>,
}

impl AsstTurnCache {
    pub fn new_from_env() -> Self {
        let unbounded = std::env::var("HIPFIRE_PROMPT_CACHE_UNBOUNDED")
            .ok()
            .as_deref()
            == Some("1");
        let cap = if unbounded {
            None
        } else {
            Some(
                std::env::var("HIPFIRE_PROMPT_CACHE_CAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32),
            )
        };
        Self {
            cap,
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    pub fn touch_mru(&mut self, fp: u64) {
        if let Some(pos) = self.order.iter().position(|k| *k == fp) {
            self.order.remove(pos);
        }
        self.order.push_back(fp);
    }

    pub fn contains_key(&self, fp: &u64) -> bool {
        self.map.contains_key(fp)
    }

    pub fn get(&mut self, fp: &u64) -> Option<&Vec<u32>> {
        if self.map.contains_key(fp) {
            self.touch_mru(*fp);
            self.map.get(fp)
        } else {
            None
        }
    }

    pub fn insert(&mut self, fp: u64, tokens: Vec<u32>) {
        if self.map.contains_key(&fp) {
            self.map.insert(fp, tokens);
            self.touch_mru(fp);
            return;
        }
        if let Some(c) = self.cap {
            while self.order.len() >= c {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                } else {
                    break;
                }
            }
        }
        self.map.insert(fp, tokens);
        self.order.push_back(fp);
    }
}

impl Default for AsstTurnCache {
    fn default() -> Self {
        Self::new_from_env()
    }
}

// ─── ModelState ────────────────────────────────────────────────────────

/// Arch-specific core state, dispatched in `LoadedModel.state`.
///
/// `unload_model` matches this exhaustively with NO wildcard: adding a variant
/// without a teardown arm is a compile error, which is the whole point of
/// folding self-contained arch state in here rather than leaving it as loose
/// `Option<…>` fields that a reload can silently leak.
pub enum ModelState {
    Qwen2(hipfire_arch_qwen2::Qwen2Bundle),
    Qwen35(hipfire_arch_qwen35::Qwen35Bundle),
    Llama(hipfire_arch_llama::LlamaBundle),
    Lfm2Moe(Lfm2MoeBundle),
    Minimax(MiniMaxBundle),
    Cohere2Moe(Cohere2MoeBundle),
    Deepseek4(hipfire_arch_deepseek4::Deepseek4Bundle),
    DotsOcr(hipfire_arch_dots_ocr::DotsOcrBundle),
}

/// LFM2.5-MoE (arch_id=11) GPU bundle. Re-exported from the arch crate, which
/// owns it so `impl SpecTarget for Lfm2MoeBundle` (the n-gram verify seam, incl.
/// the conv-state snapshot/rollback) can live next to the forward it drives
/// (orphan rule). Field-identical to the prior loader-local struct. `eos_tok` is
/// resolved at load time and rides along so the generate path doesn't re-tokenize.
pub use lfm2moe::Lfm2MoeBundle;

/// MiniMax-M2 (arch_id=10) GPU bundle. Re-exported from the arch crate, which
/// owns it so `impl SpecTarget for MiniMaxBundle` (the n-gram verify seam) can
/// live next to the forward it drives (orphan rule). Field-identical to the
/// prior loader-local struct (`config`/`weights`/`state`/`eos_tok`).
pub use minimax::MiniMaxBundle;

/// Cohere2-MoE / North-Mini-Code (arch_id=12) GPU bundle. Re-exported from the
/// arch crate, which owns it so `impl SpecTarget for Cohere2MoeBundle` (the
/// n-gram verify seam) lives next to the forward it drives (orphan rule).
/// Field-identical to the prior loader-local struct.
pub use cohere2moe::Cohere2MoeBundle;

/// The qwen35-VL vision tower state (config + weights), grouped so `LoadedModel`
/// carries ONE optional field instead of two. Loader-side by design: the vision
/// types live in the `hipfire-arch-qwen35-vl` extension crate, so this is NOT
/// folded into the base `Qwen35Bundle` (whose crate must not depend on its own
/// extension). `None` for non-VL qwen35 models.
pub struct Qwen35Vl {
    pub config: qwen35_vl::VisionConfig,
    pub weights: qwen35_vl::VisionWeights,
}

// ─── SessionState / PersistState ─────────────────────────────────────

/// Per-request state that a context reset wipes. Owns every resettable field, so
/// `SessionState::reset` is the single, total reset for request-scoped state
/// (the #462 lever). Fields migrate in over Increment 1; kept minimal here.
#[derive(Default)]
pub struct SessionState {
    pub seq_pos: usize,
    pub conversation_tokens: Vec<u32>,
    pub prefill_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub dflash_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub kv_adaptive: Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
}

impl SessionState {
    /// Total reset of request-scoped state. Frees GPU checkpoint snapshots
    /// (DeltaNetSnapshot has no Drop) before clearing, then resets scalars.
    pub fn reset(&mut self, gpu: &mut rdna_compute::Gpu) {
        for (_, snap) in self.prefill_checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
        for (_, snap) in self.dflash_checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
        if let Some(ad) = self.kv_adaptive.as_mut() {
            ad.reset();
        }
        self.seq_pos = 0;
        self.conversation_tokens.clear();
    }
}

/// Per-turn state that SURVIVES a context reset: the assistant-turn LCP cache
/// (keeps multi-turn prefix matches byte-exact) and the lazily-built decoded
/// vocab (expensive Arc). `reset` must never touch these.
#[derive(Default)]
pub struct PersistState {
    pub asst_turn_cache: AsstTurnCache,
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
}

// ─── LoadedModel ──────────────────────────────────────────────────────

/// Per-model config, set during load and read-only afterward (NOT reset). Written at
/// three load-time sites — carrier/`load_model_*` construction, `load_model`'s rec_*
/// finalization, and the daemon load-message handler (mtp_*) — never per-request. Pub
/// fields: it crosses into the daemon crate, so immutability is by convention, not
/// enforced (hence `ModelMeta`, not `ImmutableMeta`).
pub struct ModelMeta {
    pub arch_id: u32,
    pub model_path: String,
    pub chat_template: Option<String>,
    pub max_seq: usize,
    pub physical_cap: usize,
    /// Resolved EOS for the EP serve path (ds4 OR minimax — mutually exclusive; `0` if
    /// neither). Unifies the old `deepseek4_eos_tok` + `minimax_eos_tok`.
    pub eos_tok: u32,
    pub mtp_mode: String,
    pub mtp_k: usize,
    pub rec_temperature: Option<f32>,
    pub rec_top_p: Option<f32>,
    pub rec_top_k: Option<f32>,
    pub rec_min_p: Option<f32>,
    pub rec_presence_penalty: Option<f32>,
}

pub struct LoadedModel {
    /// Owning parallelism enum — the single-value answer to "which axis?".
    /// `Tp` owns the TpModel (migrated Task 3). `Pp(Dense)` owns the PpModel (migrated Task 4).
    /// `Pp(ArchResident)` carries the qwen35-PP mesh (migrated Task 6).
    pub parallel: ModelParallel,
    // Shared arch state
    pub state: Option<ModelState>,
    // LFM2.5-8B-A1B (arch_id=11) and MiniMax-M2 (arch_id=10) live in
    // `state` as ModelState::{Lfm2Moe,Minimax} so unload teardown is
    // compiler-enforced (see ModelState).
    // Vision state (qwen35-VL tower), grouped into one optional field.
    pub vision: Option<Qwen35Vl>,
    // Shared
    pub tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer>,
    /// Model-owned eviction policy. Its calibrated sidecar data and reusable GPU
    /// scratch survive request resets and are released only by `unload_model`.
    /// Request-owned cursors, KV compaction state, target state, and the DFlash
    /// mirror are reset separately by the daemon's fresh-context path.
    pub eviction: Option<Eviction>,
    pub session: SessionState,
    pub persist: PersistState,
    /// The model's speculative-decode drafter+verifier, when a draft model is
    /// loaded (`Box<dyn Speculator>` so the daemon's decode loop is agnostic to
    /// DFlash chain / DDTree tree / future MTP). Replaces the old
    /// `dflash: Option<DflashState>` field — the `DflashState` now lives inside
    /// the `DflashSpeculator` impl behind this trait object.
    pub speculator: Option<Box<dyn Speculator>>,
    /// Immutable per-model config (populated at load, never per-request).
    /// Single source of truth for arch_id, eos_tok, mtp_mode/mtp_k,
    /// max_seq, physical_cap, model_path, chat_template, rec_* sampling defaults.
    pub meta: ModelMeta,
}

impl LoadedModel {
    /// Shared-field skeleton: arch state None, pp = 1, all non-core arch slots
    /// None, collections empty, mtp defaults, asst cache from env. Callers set
    /// only the fields they own via struct-update (`..LoadedModel::skeleton(..)`).
    pub fn skeleton(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        mtp_mode: &str,
        mtp_k: usize,
    ) -> Self {
        LoadedModel {
            parallel: ModelParallel::Single,
            state: None,
            vision: None,
            tokenizer: Some(tokenizer),
            eviction: None,
            session: SessionState::default(),
            persist: PersistState {
                asst_turn_cache: AsstTurnCache::new_from_env(),
                decoded_vocab: None,
            },
            speculator: None,
            meta: ModelMeta {
                arch_id,
                model_path,
                chat_template,
                max_seq,
                physical_cap,
                eos_tok: 0,
                mtp_mode: mtp_mode.to_string(),
                mtp_k,
                rec_temperature: None,
                rec_top_p: None,
                rec_top_k: None,
                rec_min_p: None,
                rec_presence_penalty: None,
            },
        }
    }

    /// LFM2.5-MoE bundle if this model is arch_id=11, else None.
    pub fn lfm2moe(&self) -> Option<&Lfm2MoeBundle> {
        match &self.state {
            Some(ModelState::Lfm2Moe(b)) => Some(b),
            _ => None,
        }
    }

    pub fn lfm2moe_mut(&mut self) -> Option<&mut Lfm2MoeBundle> {
        match &mut self.state {
            Some(ModelState::Lfm2Moe(b)) => Some(b),
            _ => None,
        }
    }

    /// MiniMax-M2 bundle if this model is arch_id=10, else None.
    pub fn minimax(&self) -> Option<&MiniMaxBundle> {
        match &self.state {
            Some(ModelState::Minimax(b)) => Some(b),
            _ => None,
        }
    }

    pub fn minimax_mut(&mut self) -> Option<&mut MiniMaxBundle> {
        match &mut self.state {
            Some(ModelState::Minimax(b)) => Some(b),
            _ => None,
        }
    }

    /// Qwen2 bundle if this model is arch_id=7 (plain qwen2 via `Qwen2Carrier`),
    /// else None. The live `Qwen2State` is at `.state`. NOTE: dots-ocr (arch_id=8)
    /// uses `ModelState::DotsOcr`, not this variant — see `dots_ocr_mut()`.
    pub fn qwen2_mut(&mut self) -> Option<&mut hipfire_arch_qwen2::Qwen2Bundle> {
        match &mut self.state {
            Some(ModelState::Qwen2(b)) => Some(b),
            _ => None,
        }
    }

    /// Cohere2-MoE bundle if this model is arch_id=12, else None.
    pub fn cohere2moe(&self) -> Option<&Cohere2MoeBundle> {
        match &self.state {
            Some(ModelState::Cohere2Moe(b)) => Some(b),
            _ => None,
        }
    }

    pub fn cohere2moe_mut(&mut self) -> Option<&mut Cohere2MoeBundle> {
        match &mut self.state {
            Some(ModelState::Cohere2Moe(b)) => Some(b),
            _ => None,
        }
    }

    /// dots.ocr bundle if this model is arch_id=8, else None.
    pub fn dots_ocr_mut(&mut self) -> Option<&mut hipfire_arch_dots_ocr::DotsOcrBundle> {
        match &mut self.state {
            Some(ModelState::DotsOcr(b)) => Some(b),
            _ => None,
        }
    }

    /// DeepSeek V4 bundle if this model is a single-GPU arch_id=9, else None.
    /// (EP/pp ds4 keeps its state in `ep` (EpArch::Ds4), so this is None there.)
    pub fn deepseek4(&self) -> Option<&hipfire_arch_deepseek4::Deepseek4Bundle> {
        match &self.state {
            Some(ModelState::Deepseek4(b)) => Some(b),
            _ => None,
        }
    }

    pub fn deepseek4_mut(&mut self) -> Option<&mut hipfire_arch_deepseek4::Deepseek4Bundle> {
        match &mut self.state {
            Some(ModelState::Deepseek4(b)) => Some(b),
            _ => None,
        }
    }

    /// Single-arch Qwen3.5/3.6 bundle if loaded (also present under TP/PP — the
    /// base bundle stays in `ModelState::Qwen35`), else None.
    pub fn qwen35(&self) -> Option<&hipfire_arch_qwen35::Qwen35Bundle> {
        match &self.state {
            Some(ModelState::Qwen35(b)) => Some(b),
            _ => None,
        }
    }

    pub fn qwen35_mut(&mut self) -> Option<&mut hipfire_arch_qwen35::Qwen35Bundle> {
        match &mut self.state {
            Some(ModelState::Qwen35(b)) => Some(b),
            _ => None,
        }
    }

    /// Whether the loaded DeepSeek model carries MTP/spec weights that
    /// `mtp_mode=auto` should treat as spec-eligible.
    pub fn mtp_weights_present(&self) -> bool {
        self.deepseek4()
            .map(|b| b.weights.mtp_layer.is_some() || b.weights.dspark.is_some())
            .unwrap_or(false)
    }

    /// Disjoint-field borrow of the request-scoped sub-structs. Native
    /// disjoint-field borrow (no unsafe): the compiler proves these point at
    /// distinct fields. Grows to also yield `arch`/`parallel` in later
    /// increments. Call at a method body, do not store alongside `&mut self`.
    pub fn session_parts_mut(&mut self) -> (&mut SessionState, &mut PersistState, &ModelMeta) {
        (&mut self.session, &mut self.persist, &self.meta)
    }

    /// pp>1 skeleton — sets the load-bearing multi-GPU fields together so they
    /// cannot be set piecemeal. The qwen35 PP scratch is carried inside
    /// `Qwen35Bundle.pipeline`; the mesh is carried in `parallel` as
    /// `Pp(ArchResident(gpus))`.
    pub fn skeleton_pp(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        mtp_mode: &str,
        mtp_k: usize,
        gpus: Gpus,
    ) -> Self {
        LoadedModel {
            parallel: ModelParallel::Pp(crate::model_parallel::PipelineImpl::ArchResident(gpus)),
            ..LoadedModel::skeleton(
                arch_id,
                tokenizer,
                max_seq,
                physical_cap,
                model_path,
                chat_template,
                mtp_mode,
                mtp_k,
            )
        }
    }
}

/// Expert-parallel serving state.
pub struct EpState {
    pub gpus: Gpus,
    pub inner: EpArch,
}

pub enum EpArch {
    Ds4 {
        config: hipfire_arch_deepseek4::DeepseekV4Config,
        weights: Vec<hipfire_arch_deepseek4::DeepseekV4Weights>,
        state: Vec<hipfire_arch_deepseek4::DeepseekV4State>,
        partials: Vec<rdna_compute::GpuTensor>,
        /// Per-rank int64 scratch for the reproducible EP i64 down path.
        /// Each buffer holds `hidden_size * 8` raw bytes; pre-zeroed per step.
        partials_i64: Vec<rdna_compute::GpuTensor>,
    },
    Minimax {
        config: minimax::MiniMaxConfig,
        weights: Vec<minimax::MiniMaxWeights>,
        state: Vec<minimax::MiniMaxState>,
        partials: Vec<rdna_compute::GpuTensor>,
        /// Per-rank int64 scratch for the reproducible EP i64 down path.
        /// Each buffer holds `hidden_size * 8` raw bytes; pre-zeroed per step.
        partials_i64: Vec<rdna_compute::GpuTensor>,
    },
}

// ─── Helper functions ─────────────────────────────────────────────────

/// Layer 1 (env var) + Layer 2 (per-model ~/.hipfire/templates) — source-agnostic.
fn resolve_chat_template_overrides(model_path: &str) -> Option<String> {
    if let Ok(env_path) = std::env::var("HIPFIRE_CHAT_TEMPLATE_FILE") {
        if !env_path.is_empty() {
            match std::fs::read_to_string(&env_path) {
                Ok(s) => {
                    eprintln!(
                        "[chat_template] using HIPFIRE_CHAT_TEMPLATE_FILE={}",
                        env_path
                    );
                    return Some(s);
                }
                Err(e) => eprintln!(
                    "[chat_template] HIPFIRE_CHAT_TEMPLATE_FILE={env_path} failed to read ({e}); falling through"
                ),
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let basename = std::path::Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !basename.is_empty() {
            let per_model = std::path::Path::new(&home)
                .join(".hipfire")
                .join("templates")
                .join(format!("{basename}.j2"));
            if per_model.is_file() {
                match std::fs::read_to_string(&per_model) {
                    Ok(s) => {
                        eprintln!(
                            "[chat_template] using per-model override {}",
                            per_model.display()
                        );
                        return Some(s);
                    }
                    Err(e) => eprintln!(
                        "[chat_template] per-model file {} failed to read ({e}); falling through",
                        per_model.display()
                    ),
                }
            }
        }
    }
    None
}

fn resolve_chat_template(hfq: &HfqFile, model_path: &str) -> Option<String> {
    if let Some(s) = resolve_chat_template_overrides(model_path) {
        return Some(s);
    }
    match hfq.arch_id {
        5 | 6 => return Some(FROGGERIC_QWEN35_TEMPLATE.to_string()),
        11 => {
            if let Some(t) = hfq.chat_template() {
                return Some(t);
            }
            return Some(LFM2_TEMPLATE.to_string());
        }
        12 => {
            if let Some(t) = hfq.chat_template_named("tool_use") {
                return Some(
                    t.replace("<|START_RESPONSE|>", "<|START_TEXT|>")
                        .replace("<|END_RESPONSE|>", "<|END_TEXT|>")
                        .replace("{{message.tool_plan}}", "{{ message.tool_plan or '' }}")
                        .replace("{{ tc['function']['name'] }}", "{{ tc.name }}")
                        .replace(
                            "{{ tc['function']['arguments']|tojson }}",
                            "{{ tc.arguments|tojson }}",
                        ),
                );
            }
        }
        _ => {}
    }
    hfq.chat_template()
}

pub(crate) fn parse_state_quant(
    mode: Option<&str>,
) -> Result<hipfire_arch_qwen35::qwen35::StateQuant, String> {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)"
        )),
    }
}

// ─── Load functions ───────────────────────────────────────────────────

// ─── Core arch carrier load ─────────────────────────────────────────────

/// Build a `LoadedModel` from a carrier `Bundle`, shared fields, and
/// eviction/DFlash state. This is the common body for qwen35 dispatch
/// where eviction and DFlash need per-arch type info.
fn free_unpublished_qwen35_bundle(bundle: Qwen35Bundle, gpu: &mut Gpu) {
    // `finish_qwen35_load` is only called on the pp=1 carrier path, so the
    // pipeline payload cannot be present here (it needs its owning `Gpus`).
    debug_assert!(
        bundle.pipeline.is_none(),
        "single-GPU Qwen35 finish received pipeline-owned state"
    );
    if let Some(head) = bundle.mtp_head {
        head.free_gpu(gpu);
    }
    bundle.kv_cache.free_gpu(gpu);
    bundle.scratch.free_gpu(gpu);
    bundle.weights.free_gpu(gpu);
    bundle.dn_state.free_gpu(gpu);
}

fn finish_qwen35_load(
    bundle: Qwen35Bundle,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    physical_cap: usize,
    arch_id: u32,
    chat_template: Option<String>,
    ctx: &mut LoadCtx,
    vision_config: Option<qwen35_vl::VisionConfig>,
    vision_weights: Option<qwen35_vl::VisionWeights>,
) -> Result<LoadedModel, String> {
    use hipfire_arch_qwen35::qwen35::LayerType;
    let mut bundle = Some(bundle);
    let mut eviction = None;
    let mut vision_weights = vision_weights;
    let result = (|| -> Result<LoadedModel, String> {
        // Extract references for eviction/DFlash setup (borrow, don't move).
        let bundle_ref = bundle.as_ref().expect("Qwen35 bundle is still staged");
        let config = &bundle_ref.config;
        let dn_state = &bundle_ref.dn_state;
        // ── Eviction ───────────────────────────────────────────────────
        let cask_m_folding = effective_cask_m_folding(ctx.cask.cask_m_folding, ctx.draft_path);
        if ctx.cask.cask_m_folding && !cask_m_folding {
            eprintln!(
                "[hipfire-daemon] cask:true + draft: both set — downgrading to plain TriAttention drop-eviction (CASK m-fold + DFlash is a known-broken combo; see feedback_cask_mfold_dflash_broken.md)",
            );
        }
        let configured_eviction = if let Some(ref sidecar_path) = ctx.cask.sidecar {
            let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
                use std::io::ErrorKind;
                let p = Path::new(sidecar_path);
                let why = match e.kind() {
                    ErrorKind::NotFound if p.symlink_metadata().is_ok() =>
                        format!("dangling symlink (target absent): {sidecar_path}"),
                    ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
                    ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
                    ErrorKind::UnexpectedEof => format!("truncated/corrupt sidecar: {sidecar_path}"),
                    _ => format!("read error ({e}): {sidecar_path}"),
                };
                format!("cask sidecar load failed — {why} (regen: hipfire sidecar-gen, or HIPFIRE_CASK_OFF=1)")
            })?;
            let fa_layer_ids: Vec<usize> = config
                .layer_types
                .iter()
                .enumerate()
                .filter_map(|(i, t)| {
                    if *t == LayerType::FullAttention {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            if fa_layer_ids.is_empty() {
                eprintln!("  cask_sidecar set but model has no FullAttention layers — ignoring");
                None
            } else {
                // Validate before moving a GPU-backed base into CaskCtx::new;
                // an invalid CASK request must return Err, not panic after
                // `EvictionCtx` has acquired reusable scratch.
                if cask_m_folding && !(0.0..=1.0).contains(&ctx.cask.core_frac) {
                    return Err(format!(
                        "invalid CASK core_frac {} (expected 0.0..=1.0)",
                        ctx.cask.core_frac
                    ));
                }
                if cask_m_folding && ctx.cask.fold_m < 2 {
                    return Err(format!(
                        "invalid CASK fold_m {} (expected >= 2)",
                        ctx.cask.fold_m
                    ));
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let base = EvictionCtx::new(
                    ctx.gpu,
                    &centers,
                    fa_layer_ids,
                    ctx.cask.budget,
                    ctx.cask.beta,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    physical_cap,
                )
                .map_err(|e| format!("build EvictionCtx: {e}"))?;
                if cask_m_folding {
                    eprintln!(
                        "  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                        ctx.cask.core_frac,
                        ctx.cask.fold_m,
                        ctx.cask.budget,
                        ctx.cask.beta,
                        physical_cap
                    );
                    Some(Eviction::Cask(CaskCtx::new(
                        base,
                        ctx.cask.core_frac,
                        ctx.cask.fold_m,
                    )))
                } else {
                    eprintln!(
                        "  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                        ctx.cask.budget, ctx.cask.beta, physical_cap
                    );
                    Some(Eviction::Plain(base))
                }
            }
        } else {
            None
        };
        eviction = configured_eviction;

        // ── DSpark sidecar (wins over DFlash/MTP/n-gram) ───────────────
        // The drafter is a dense-qwen3 body (llama crate); it drives the qwen35
        // ModelSlot target via the SpecTarget DSpark capture hooks. Discovered as
        // `<stem>-dspark.<ext>` next to the trunk, independent of ctx.draft_path.
        let dspark_speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> = if ctx
            .spec
            .dspark
            != Some(false)
        {
            let base = std::path::Path::new(ctx.path);
            let sidecar_path = match (base.parent(), base.file_stem(), base.extension()) {
                (Some(parent), Some(stem), Some(ext)) => Some(parent.join(format!(
                    "{}-dspark.{}",
                    stem.to_string_lossy(),
                    ext.to_string_lossy()
                ))),
                _ => None,
            };
            match sidecar_path.filter(|p| p.exists()) {
                Some(p) => {
                    eprintln!("  qwen35: opening DSpark sidecar HFQ {p:?}");
                    match hipfire_runtime::hfq::HfqFile::open(&p) {
                        Ok(mut sidecar) => {
                            sidecar.drop_mmap();
                            match hipfire_arch_llama::dspark_body::load_qwen3_dspark(
                                &sidecar, ctx.gpu,
                            ) {
                                Ok(Some((dspark_weights, assets))) => {
                                    let block = dspark_weights.cfg.block_size;
                                    // Reduced-vocab drafters (ORNITH) ship a compressed
                                    // lm_head; run_heads reads vocab from lm_head.shape[0].
                                    let vocab = if dspark_weights.cfg.draft_vocab_size > 0 {
                                        dspark_weights.cfg.draft_vocab_size
                                    } else {
                                        assets.config.vocab_size
                                    };
                                    let stage_norm = assets.weights.output_norm.shallow_clone();
                                    // upload_raw sets dtype=Raw; the data is F16.
                                    let mut lm_head = assets.weights.output.buf.shallow_clone();
                                    lm_head.dtype = rdna_compute::DType::F16;
                                    lm_head.shape = vec![vocab];
                                    let conf_threshold =
                                        std::env::var("HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD")
                                            .ok()
                                            .and_then(|s| s.parse().ok())
                                            .or(ctx.spec.dspark_conf_threshold)
                                            .unwrap_or(0.1f32);
                                    eprintln!(
                                        "  qwen35 DSpark enabled (block={}, target_layers={:?}, draft_vocab={}, conf={:.2})",
                                        block,
                                        dspark_weights.cfg.target_layer_ids,
                                        vocab,
                                        conf_threshold
                                    );
                                    match hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
                                        assets,
                                        &dspark_weights.cfg,
                                        ctx.gpu,
                                    ) {
                                        Ok(body) => {
                                            Some(hipfire_runtime::dspark_core::build_dspark_speculator(
                                            body,
                                            dspark_weights,
                                            stage_norm,
                                            lm_head,
                                            block,
                                            physical_cap,
                                            conf_threshold,
                                            true, // sampled verify (temp>0) supported
                                        ))
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "  qwen35: DSpark body build failed: {e} — AR/other"
                                            );
                                            None
                                        }
                                    }
                                }
                                Ok(None) => {
                                    eprintln!(
                                        "  qwen35: DSpark sidecar {p:?} has no dspark_* metadata — skipping"
                                    );
                                    None
                                }
                                Err(e) => {
                                    eprintln!("  qwen35: WARNING DSpark sidecar load failed: {e}");
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("  qwen35: WARNING cannot open DSpark sidecar {p:?}: {e}");
                            None
                        }
                    }
                }
                None => None,
            }
        } else {
            None
        };

        // ── DFlash (skipped when a DSpark sidecar won) ─────────────────
        let dflash = if dspark_speculator.is_some() {
            None
        } else if let Some(dp) = ctx.draft_path {
            match hipfire_arch_qwen35::dflash_spec::load_dflash_state(
                dp,
                physical_cap,
                config,
                dn_state,
                ctx.gpu,
                ctx.spec.ddtree_budget,
                ctx.spec.ddtree_topk,
            ) {
                Ok(s) => {
                    eprintln!(
                        "  DFlash draft loaded: {} (layers={}, hidden={}, block={})",
                        dp,
                        s.draft_config.n_layers,
                        s.draft_config.hidden,
                        s.draft_config.block_size
                    );
                    Some(s)
                }
                Err(e) => {
                    eprintln!(
                        "  DFlash draft load failed ({}): {} — falling back to AR only",
                        dp, e
                    );
                    None
                }
            }
        } else {
            None
        };
        // Pick the arch-generic speculator: a loaded DFlash draft, else the opt-in
        // model-free n-gram drafter. `eviction` is borrowed (not moved) here, so
        // it is still available for the struct literal below; `config`/`dn_state` are
        // borrowed only for the n-gram arm's scratch construction (snapshot copied to
        // GPU), released before `bundle` moves into `state`. `None` ⇒ AR-only model.
        // DSpark wins over DFlash/MTP/n-gram when its sidecar loaded.
        let speculator = dspark_speculator.or_else(|| {
            crate::spec_build::build_speculator(
                arch_id,
                dflash,
                eviction.is_none(),
                physical_cap,
                ctx.spec,
            )
        });

        Ok(LoadedModel {
            state: Some(ModelState::Qwen35(
                bundle.take().expect("Qwen35 bundle is still staged"),
            )),
            eviction: eviction.take(),
            speculator,
            vision: vision_config
                .zip(vision_weights.take())
                .map(|(config, weights)| Qwen35Vl { config, weights }),
            ..LoadedModel::skeleton(
                arch_id,
                tokenizer,
                ctx.max_seq,
                physical_cap,
                ctx.path.to_string(),
                chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
            )
        })
    })();

    match result {
        Ok(model) => Ok(model),
        Err(error) => {
            if let Some(eviction) = eviction.take() {
                eviction.free_gpu(ctx.gpu);
            }
            if let Some(weights) = vision_weights.take() {
                weights.free_gpu(ctx.gpu);
            }
            if let Some(bundle) = bundle.take() {
                free_unpublished_qwen35_bundle(bundle, ctx.gpu);
            }
            ctx.gpu.drain_pool();
            Err(error)
        }
    }
}

// ─── Main public API ──────────────────────────────────────────────────

fn normalize_mtp_k(arch_id: u32, mtp_k: Option<usize>) -> Result<usize, String> {
    let _ = arch_id;
    let value = mtp_k.unwrap_or(3);
    if (1..=8).contains(&value) {
        Ok(value)
    } else {
        Err(format!("MTP K must be in 1..=8, got {value}"))
    }
}

pub(crate) fn reject_qwen_native_mtp(mtp_mode: &str) -> Result<(), String> {
    if mtp_mode == "on" {
        Err("Qwen native MTP is disabled pending SPEC-003".into())
    } else {
        Ok(())
    }
}

fn mtp_mode_from_spec(spec: SpecLoadCfg) -> &'static str {
    match spec.mtp_mode {
        Some(true) => "on",
        Some(false) => "off",
        None => "auto",
    }
}

/// Load a model from an HFQ file (or safetensors directory). This is the
/// single arch-dispatch point via the carrier registry.
#[allow(clippy::too_many_arguments)]
pub fn load_model(
    path: &str,
    max_seq: usize,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    mesh: &DeviceMesh,
    pp_bands: Option<&[usize]>,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    let src = ModelSource::from_path(path)?;
    let source_arch_id = match &src {
        ModelSource::Hfq(hfq) => hfq.arch_id,
        ModelSource::Dir(source) => source.arch_id(),
    };

    // Author-recommended sampling defaults (temp/top_p/top_k from the .hfq's baked
    // `generation_config`). Extract HERE, from the already-open source, BEFORE the
    // carrier allocates any GPU buffers. The `metadata_json` parse churns the host
    // heap; doing it AFTER allocation but BEFORE the first-warmup AR hipGraph
    // capture perturbs buffer placement and — on gfx12 / ROCm 7.2, which snapshots
    // kernarg/buffer addresses at graph-instantiate — makes the captured graph
    // replay ~2× slower (gfx12 MoE A3B 99→50; bisected to config-inheritance commit
    // 2a7a1c8b). Parsing pre-allocation lets the heap settle. HFQ sources only;
    // raw-safetensors PP carries no generation_config.
    let rec_sampling = match &src {
        ModelSource::Hfq(hfq) => hfq.recommended_sampling(),
        _ => None,
    };

    // DFlash lm_head quant check — only for HFQ sources
    if draft_path.is_some() {
        if let ModelSource::Hfq(ref hfq) = src {
            let lm_qt = hfq
                .tensor_data("lm_head.weight")
                .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
                .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
                .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
                .map(|(info, _)| info.quant_type);
            let arch_is_gfx11 = matches!(
                gpu.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
            );
            let supported = match lm_qt {
                Some(3 | 6 | 13) => true,
                Some(17) => arch_is_gfx11,
                _ => false,
            };
            if !supported {
                let qt_desc = match lm_qt {
                    Some(qt) => format!("quant_type={qt}"),
                    None => "no lm_head/embed_tokens tensor found at any known name".to_string(),
                };
                return Err(format!(
                    "DFlash draft requested but target lm_head {} is not \
                     supported by speculative.rs's batched GEMM paths on this arch \
                     ({}). Supported: Q8_0 (qt=3), HFQ4G256 (qt=6), MQ4G256 (qt=13) \
                     always; MQ3G256 (qt=17) on gfx11 only. Other dtypes \
                     (MQ2 qt=18, MQ6/MQ8, HFQ3/HFQ2, HFQ4G128, HFQ6, F16, …) fall \
                     through to a per-row GEMV that hangs verify. Reload without a \
                     draft, or use an MQ4 / HFQ4 / Q8 target.",
                    qt_desc, gpu.arch
                ));
            }
            let arch_is_dense_qwen35 = hfq.arch_id == 5;
            let mq3_supported = arch_is_gfx11 && arch_is_dense_qwen35;
            let mq_unsupported = hfq
                .first_tensor_with_quant_type(18)
                .map(|n| ("MQ2 (qt=18)", n));
            let mq_unsupported = mq_unsupported.or_else(|| {
                if !mq3_supported {
                    hfq.first_tensor_with_quant_type(17)
                        .map(|n| ("MQ3 (qt=17)", n))
                } else {
                    None
                }
            });
            if let Some((qt_label, name)) = mq_unsupported {
                let arch_reason = if !arch_is_dense_qwen35 && qt_label.starts_with("MQ3") {
                    format!(
                        "arch_id={} (MoE/A3B-class) has no MQ3 MoE kernels",
                        hfq.arch_id
                    )
                } else {
                    format!(
                        "arch={} lacks the corresponding batched WMMA prefill family",
                        gpu.arch
                    )
                };
                return Err(format!(
                    "DFlash draft requested but model contains {qt_label} weight \
                     `{name}` and {arch_reason}. The prefill fast-path falls back \
                     to per-token `forward_scratch` for every spec verify cycle \
                     (or worse, a kernel-stride mismatch on MoE) — defeating \
                     DFlash's speedup. Reload without a draft, or use an MQ4 / \
                     HFQ4 / Q8 target.",
                ));
            }
        }
    }

    let mtp_mode = mtp_mode_from_spec(spec);
    let mut ctx = LoadCtx {
        path,
        max_seq,
        draft_path,
        kv_mode_override,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp: mesh.size_of(DimKind::Pp),
        pp_bands,
        mtp_mode,
        mtp_k: normalize_mtp_k(source_arch_id, spec.mtp_k)?,
        spec,
        gpu,
    };

    // Carrier registry dispatch. Collect all matches so an overlap between
    // two carriers' `claims_arch_id` fails loudly here instead of silently
    // resolving to whichever was registered first.
    let mut matches = REGISTRY.iter().filter(|c| c.probe(&src));
    let carrier = matches
        .next()
        .ok_or_else(|| format!("no carrier for {}", src.describe()))?;
    if let Some(other) = matches.next() {
        return Err(format!(
            "ambiguous carrier dispatch for {}: '{}' and '{}' both claim it",
            src.describe(),
            carrier.name(),
            other.name()
        ));
    }
    let mut result = carrier.load(src, &mut ctx)?;
    // Apply the author-recommended sampling extracted pre-allocation (see above).
    // Do NOT reparse the .hfq metadata here: a post-allocation / pre-capture parse
    // is the gfx12 hipGraph-replay regression root-caused above.
    if let Some(rec) = rec_sampling {
        result.meta.rec_temperature = rec.temperature;
        result.meta.rec_top_p = rec.top_p;
        result.meta.rec_top_k = rec.top_k.map(|k| k as f32);
    }
    Ok(result)
}

fn load_cohere2moe(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
    mtp_mode: &str,
    mtp_k: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let config = <cohere2moe::Cohere2Moe as Architecture>::config_from_hfq(&hfq)?;
    let weights = <cohere2moe::Cohere2Moe as Architecture>::load_weights(&mut hfq, &config, gpu)?;
    let state = cohere2moe::Cohere2MoeState::new_with_max_seq(gpu, &config, max_seq)
        .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        };
        try_one("<|END_OF_TURN_TOKEN|>")
            .or_else(|| try_one("</s>"))
            .or_else(|| try_one("<|endoftext|>"))
            .unwrap_or(255001)
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        state: Some(ModelState::Cohere2Moe(Cohere2MoeBundle {
            config,
            weights,
            state,
            eos_tok,
        })),
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            mtp_mode,
            mtp_k,
        )
    })
}

// ─── MMQ screening ────────────────────────────────────────────────────

// ─── EP load functions ────────────────────────────────────────────────

/// EP partial-load fault injector. Reads `HIPFIRE_EP_FAIL_RANK` so the GPU
/// cleanup test can force a deterministic mid-load failure after a given rank and
/// assert the staging guard reclaimed every loaded rank's VRAM. Gated behind the
/// `ep-fault-inject` feature: production/default builds compile the `None` stub
/// below, so a stray `HIPFIRE_EP_FAIL_RANK` in the environment can NEVER fail a
/// real EP load.
#[cfg(feature = "ep-fault-inject")]
fn ep_fail_rank() -> Option<usize> {
    match std::env::var("HIPFIRE_EP_FAIL_RANK").ok() {
        Some(s) if !s.is_empty() => s.parse::<usize>().ok(),
        _ => None,
    }
}

#[cfg(not(feature = "ep-fault-inject"))]
fn ep_fail_rank() -> Option<usize> {
    None
}

/// Staging guard for the ds4 EP load (transactional partial-load cleanup). Owns
/// the `Gpus` orchestrator plus the per-rank weights / state / partials as they
/// are built up. If the load fails mid-way (a `?` early return, or the
/// `HIPFIRE_EP_FAIL_RANK` fault), `Drop` explicitly frees every rank's VRAM
/// (weights → state → partial) and drains each device's pool, so a failed EP load
/// leaks NO VRAM. On success the caller calls `into_parts()` to disarm the guard
/// and move ownership into the `LoadedModel`.
struct Ds4EpStaging {
    /// `Option` so `into_parts` can move the `Gpus` out on success without a
    /// placeholder. `None` after a successful disarm.
    gpus: Option<Gpus>,
    weights: Vec<deepseek4::DeepseekV4Weights>,
    state: Vec<deepseek4::DeepseekV4State>,
    partials: Vec<rdna_compute::GpuTensor>,
    partials_i64: Vec<rdna_compute::GpuTensor>,
}

impl Ds4EpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            state: Vec::new(),
            partials: Vec::new(),
            partials_i64: Vec::new(),
        }
    }
    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staging gpus taken")
    }
    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<deepseek4::DeepseekV4Weights>,
        Vec<deepseek4::DeepseekV4State>,
        Vec<rdna_compute::GpuTensor>,
        Vec<rdna_compute::GpuTensor>,
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let state = std::mem::take(&mut self.state);
        let partials = std::mem::take(&mut self.partials);
        let partials_i64 = std::mem::take(&mut self.partials_i64);
        (gpus, weights, state, partials, partials_i64)
    }
}

impl Drop for Ds4EpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        eprintln!(
            "[loader] EP ds4 load failed — freeing {} partially-loaded rank(s) (no VRAM leak)",
            self.weights.len()
        );
        for (r, w) in self.weights.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                w.free_gpu(dev);
            }
        }
        for (r, s) in self.state.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                s.free_gpu(dev);
            }
        }
        for (r, p) in self.partials.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for (r, p) in self.partials_i64.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
    }
}

/// Staging guard for the MiniMax EP load — mirror of `Ds4EpStaging` with the
/// MiniMax weight/state types.
struct MinimaxEpStaging {
    gpus: Option<Gpus>,
    weights: Vec<minimax::MiniMaxWeights>,
    state: Vec<minimax::MiniMaxState>,
    partials: Vec<rdna_compute::GpuTensor>,
    partials_i64: Vec<rdna_compute::GpuTensor>,
}

impl MinimaxEpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            state: Vec::new(),
            partials: Vec::new(),
            partials_i64: Vec::new(),
        }
    }
    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staging gpus taken")
    }
    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<minimax::MiniMaxWeights>,
        Vec<minimax::MiniMaxState>,
        Vec<rdna_compute::GpuTensor>,
        Vec<rdna_compute::GpuTensor>,
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let state = std::mem::take(&mut self.state);
        let partials = std::mem::take(&mut self.partials);
        let partials_i64 = std::mem::take(&mut self.partials_i64);
        (gpus, weights, state, partials, partials_i64)
    }
}

impl Drop for MinimaxEpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        eprintln!(
            "[loader] EP minimax load failed — freeing {} partially-loaded rank(s) (no VRAM leak)",
            self.weights.len()
        );
        for (r, w) in self.weights.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                w.free_gpu(dev);
            }
        }
        for (r, s) in self.state.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                s.free_gpu(dev);
            }
        }
        for (r, p) in self.partials.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for (r, p) in self.partials_i64.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
    }
}

/// Expert-parallel (EP) model load — shards the routed experts across `tp` ranks
/// (`Gpus::init_tp` + per-arch sharded weight load), wrapped in a staging guard so
/// a mid-load failure frees every already-loaded rank's VRAM (no leak, prior model
/// at the call site left intact). ds4 (arch_id 9) and MiniMax (arch_id 10) only.
///
/// KNOWN RESIDUAL — constructor-mid-failure leak (scoped follow-up, NOT fixed):
/// the staging guard frees every rank that has been COMPLETED and `push`ed, so a
/// failure BETWEEN ranks leaks no VRAM. But a failure INSIDE a single rank's
/// constructor — after it uploaded some tensors but before it returns `Ok` —
/// leaks those partial allocations (`GpuTensor` has no `Drop`). The fault injector
/// (`HIPFIRE_EP_FAIL_RANK`) fires AFTER a rank's constructor returns `Ok`, so it
/// tests the completed-rank cleanup path (which IS fixed), not this inner window.
/// The proper fix is an unwind-safe allocation-tracking loader refactor. Deferred.
pub fn load_model_ep(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    spec: SpecLoadCfg,
) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let mtp_mode = mtp_mode_from_spec(spec);
    let mtp_k = normalize_mtp_k(hfq.arch_id, spec.mtp_k)?;
    match hfq.arch_id {
        9 => load_model_ep_ds4(path, max_seq, mesh, mtp_mode, mtp_k),
        10 => load_model_ep_minimax(path, max_seq, mesh, mtp_mode, mtp_k),
        id => Err(format!(
            "EP not supported for arch_id={id} (expected 9 for DeepSeek V4 or 10 for MiniMax)"
        )),
    }
}

/// Load a model for **real tensor-parallel (TP)** serving — dense row/col
/// sharding across `tp` GPUs (the `Tp` axis), distinct from expert-parallel
/// ([`load_model_ep`], the `Ep` axis). This is the daemon entry the EP↔TP
/// disentanglement reserves for the dense TP serve path.
///
/// The TP *forward* is validated end-to-end (see the `tp_*_parity` examples:
/// `execute_steps_tp` + the store→forward bridge reproduce single-GPU logits at
/// Tp-2). Wiring it into a served [`LoadedModel`] — per-rank sharded
/// `LlamaWeights` from a `WeightStore`, per-rank scratch/KV, a `Gpus`-threaded
/// decode loop, and `tp_decode_parity` — is **PB-TP5** and not yet done, so this
/// returns a clear error rather than silently falling back to single-GPU.
pub fn load_model_tp(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    spec: SpecLoadCfg,
) -> Result<LoadedModel, String> {
    // Host-side metadata BEFORE GPU allocation (chat template + recommended
    // sampling), matching the ds4/minimax EP loaders.
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let arch_id = hfq.arch_id;
    let mtp_mode = mtp_mode_from_spec(spec);
    let mtp_k = normalize_mtp_k(arch_id, spec.mtp_k)?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let chat_template = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();
    drop(hfq); // TpModel::load reopens; free this handle before GPU work.

    let tp_model = hipfire_runtime::tp_serve::TpModel::load(path, mesh, max_seq)?;
    let eos_tok = tp_model.eos_token();

    Ok(LoadedModel {
        parallel: ModelParallel::Tp(tp_model), // Task 3: TP axis migrated from m.tp
        meta: ModelMeta {
            arch_id,
            model_path: path.to_string(),
            chat_template: chat_template.clone(),
            max_seq,
            physical_cap: max_seq,
            eos_tok,
            mtp_mode: mtp_mode.to_string(),
            mtp_k,
            rec_temperature: rec.and_then(|r| r.temperature),
            rec_top_p: rec.and_then(|r| r.top_p),
            rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
            rec_min_p: None,
            rec_presence_penalty: None,
        },
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            mtp_mode,
            mtp_k,
        )
    })
}

/// Load a dense llama-family HFQ for **pipeline-parallel (PP)** serving — layers
/// banded across `pp` stages, residual handed across each seam via
/// `boundary_copy` (P-C, the `Pp` axis). Distinct from the qwen35 hand-coded PP
/// path (`load_qwen35_pp`); this is the arch-generic driver-owned loop for
/// llama-family models. Served via the daemon's `generate_pp` (the shared
/// `generate_dense` loop).
pub fn load_model_pp(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    spec: SpecLoadCfg,
) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let arch_id = hfq.arch_id;
    let mtp_mode = mtp_mode_from_spec(spec);
    let mtp_k = normalize_mtp_k(arch_id, spec.mtp_k)?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let chat_template = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();
    drop(hfq); // PpModel::load reopens; free this handle before GPU work.

    let pp_model = hipfire_runtime::pp_serve::PpModel::load(path, mesh, max_seq)?;
    let eos_tok = pp_model.eos_token();

    Ok(LoadedModel {
        parallel: ModelParallel::Pp(crate::model_parallel::PipelineImpl::Dense(pp_model)),
        meta: ModelMeta {
            arch_id,
            model_path: path.to_string(),
            chat_template: chat_template.clone(),
            max_seq,
            physical_cap: max_seq,
            eos_tok,
            mtp_mode: mtp_mode.to_string(),
            mtp_k,
            rec_temperature: rec.and_then(|r| r.temperature),
            rec_top_p: rec.and_then(|r| r.top_p),
            rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
            rec_min_p: None,
            rec_presence_penalty: None,
        },
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            mtp_mode,
            mtp_k,
        )
    })
}

fn load_model_ep_ds4(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    mtp_mode: &str,
    mtp_k: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let n_exp = config.n_routed_experts;

    // Host-side metadata work (chat template + author-recommended sampling) BEFORE
    // any GPU allocation / EP hipGraph capture. `recommended_sampling()` reparses
    // the .hfq metadata_json (serde_json::from_str); doing that post-allocation but
    // pre-capture churns the host heap and — on gfx12 / ROCm 7.2, which snapshots
    // buffer addresses at graph-instantiate — slows the captured EP-decode graph
    // replay. Same regression as load_model (gfx12 A3B 99→50), mirrored here for the
    // ds4 EP path; see project_gfx12_hipgraph_late_host_alloc_clobber. The EP graph
    // itself (deepseek4 forward.rs begin_graph_capture) is untouched — it still
    // captures + engages; this only settles the heap before it instantiates.
    let chat_template = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();

    let ep = mesh.size_of(DimKind::Ep);
    let gpus =
        Gpus::from_mesh(mesh, config.num_hidden_layers).map_err(|e| format!("from_mesh: {e:?}"))?;
    let n = gpus.devices.len();
    if n != ep {
        return Err(format!(
            "from_mesh gave {n} devices, expected ep={ep} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    eprintln!("[loader] EP load: ep={ep} arch=ds4 experts={n_exp} (rank r owns e%{ep}==r)");
    let shard = ShardConfig::new(
        ep,
        /*tp_kv_replicate=*/ true,
        n_exp,
        ExpertAssign::Stride,
    )
    .map_err(|e| format!("ShardConfig: {e:?}"))?;
    // Transactional partial-load: build per-rank weights/state/partials INTO the
    // staging guard. Every `?` below early-returns while `staging` is alive, so
    // its `Drop` frees the ranks already loaded.
    let fail_rank = ep_fail_rank();
    let _ = fail_rank;
    let mut staging = Ds4EpStaging::new(gpus);
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut h = HfqFile::open(Path::new(path)).map_err(|e| format!("reopen rank {r}: {e}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = deepseek4::DeepseekV4::load_weights_sharded(&mut h, &config, dev, &shard, r)
            .map_err(|e| format!("shard load rank {r}: {e:?}"))?;
        staging.weights.push(w);
        // Deterministic partial-load fault for testing the cleanup path. Fires
        // AFTER ranks 0..=r loaded; the guard's Drop frees them all.
        if fail_rank == Some(r) {
            return Err(format!(
                "HIPFIRE_EP_FAIL_RANK={r}: synthetic ds4 EP load failure after rank {r} (testing partial-load cleanup)"
            ));
        }
    }
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let st =
            deepseek4::DeepseekV4State::new(&config).map_err(|e| format!("state {r}: {e:?}"))?;
        staging.state.push(st);
        let p = staging.gpus_mut().devices[r]
            .zeros(&[config.hidden_size], rdna_compute::DType::F32)
            .map_err(|e| format!("partial {r}: {e:?}"))?;
        staging.partials.push(p);
        let pi = staging.gpus_mut().devices[r]
            .zeros(&[config.hidden_size * 8], rdna_compute::DType::Raw)
            .map_err(|e| format!("partial_i64 {r}: {e:?}"))?;
        staging.partials_i64.push(pi);
    }
    let peer = staging
        .gpus_mut()
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e:?}"))?;
    staging
        .gpus_mut()
        .ensure_rank_streams()
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!("[loader] EP load complete: {n} ranks, peer_access={peer}");
    let (gpus, weights, state, partials, partials_i64) = staging.into_parts();

    let eos_tok: u32 = {
        let ids = tokenizer.encode("<｜end▁of▁sentence｜>");
        if ids.len() == 1 {
            ids[0]
        } else {
            1
        }
    };
    // chat_template + rec extracted pre-allocation above (gfx12 hipGraph hazard).
    Ok(LoadedModel {
        parallel: ModelParallel::Ep(EpState {
            gpus,
            inner: EpArch::Ds4 {
                config,
                weights,
                state,
                partials,
                partials_i64,
            },
        }),
        meta: ModelMeta {
            arch_id,
            model_path: path.to_string(),
            chat_template: chat_template.clone(),
            max_seq,
            physical_cap: max_seq,
            eos_tok,
            mtp_mode: mtp_mode.to_string(),
            mtp_k,
            rec_temperature: rec.and_then(|r| r.temperature),
            rec_top_p: rec.and_then(|r| r.top_p),
            rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
            rec_min_p: None,
            rec_presence_penalty: None,
        },
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            mtp_mode,
            mtp_k,
        )
    })
}

fn load_model_ep_minimax(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    mtp_mode: &str,
    mtp_k: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let n_exp = config.num_local_experts;

    // Host-side metadata work (chat template + author-recommended sampling) BEFORE
    // any GPU allocation / EP hipGraph capture. `recommended_sampling()` reparses
    // the .hfq metadata_json (serde_json::from_str); doing that post-allocation but
    // pre-capture churns the host heap and — on gfx12 / ROCm 7.2, which snapshots
    // buffer addresses at graph-instantiate — slows the captured EP-decode graph
    // replay. Same regression as load_model (gfx12 A3B 99→50), mirrored here for the
    // minimax EP path; see project_gfx12_hipgraph_late_host_alloc_clobber. The EP
    // graph itself (minimax forward.rs begin_graph_capture) is untouched — it still
    // captures + engages; this only settles the heap before it instantiates.
    let chat_template = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();

    let ep = mesh.size_of(DimKind::Ep);
    let gpus =
        Gpus::from_mesh(mesh, config.num_hidden_layers).map_err(|e| format!("from_mesh: {e:?}"))?;
    let n = gpus.devices.len();
    if n != ep {
        return Err(format!(
            "from_mesh gave {n} devices, expected ep={ep} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    eprintln!("[loader] EP load: ep={ep} arch=minimax experts={n_exp} (rank r owns e%{ep}==r)");
    let shard = ShardConfig::new(
        ep,
        /*tp_kv_replicate=*/ true,
        n_exp,
        ExpertAssign::Stride,
    )
    .map_err(|e| format!("ShardConfig: {e:?}"))?;
    let fail_rank = ep_fail_rank();
    let _ = fail_rank;
    let mut staging = MinimaxEpStaging::new(gpus);
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut h = HfqFile::open(Path::new(path)).map_err(|e| format!("reopen rank {r}: {e}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = minimax::MiniMaxWeights::load(&mut h, &config, dev, Some((&shard, r)), None)
            .map_err(|e| format!("shard load rank {r}: {e:?}"))?;
        staging.weights.push(w);
        if fail_rank == Some(r) {
            return Err(format!(
                "HIPFIRE_EP_FAIL_RANK={r}: synthetic minimax EP load failure after rank {r} (testing partial-load cleanup)"
            ));
        }
    }
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let st = {
            let dev = &mut staging.gpus_mut().devices[r];
            minimax::MiniMaxState::new_with_max_seq(dev, &config, max_seq)
                .map_err(|e| format!("state {r}: {e:?}"))?
        };
        staging.state.push(st);
        let p = staging.gpus_mut().devices[r]
            .zeros(&[config.hidden_size], rdna_compute::DType::F32)
            .map_err(|e| format!("partial {r}: {e:?}"))?;
        staging.partials.push(p);
        let pi = staging.gpus_mut().devices[r]
            .zeros(&[config.hidden_size * 8], rdna_compute::DType::Raw)
            .map_err(|e| format!("partial_i64 {r}: {e:?}"))?;
        staging.partials_i64.push(pi);
    }
    let peer = staging
        .gpus_mut()
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e:?}"))?;
    staging
        .gpus_mut()
        .ensure_rank_streams()
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!("[loader] EP load complete: {n} ranks, peer_access={peer}");
    let (gpus, weights, state, partials, partials_i64) = staging.into_parts();

    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        };
        try_one("[e~[")
            .or_else(|| try_one("<|im_end|>"))
            .or_else(|| try_one("</s>"))
            .or_else(|| try_one("<|endoftext|>"))
            .unwrap_or(1)
    };
    // chat_template + rec extracted pre-allocation above (gfx12 hipGraph hazard).
    Ok(LoadedModel {
        parallel: ModelParallel::Ep(EpState {
            gpus,
            inner: EpArch::Minimax {
                config,
                weights,
                state,
                partials,
                partials_i64,
            },
        }),
        meta: ModelMeta {
            arch_id,
            model_path: path.to_string(),
            chat_template: chat_template.clone(),
            max_seq,
            physical_cap: max_seq,
            eos_tok,
            mtp_mode: mtp_mode.to_string(),
            mtp_k,
            rec_temperature: rec.and_then(|r| r.temperature),
            rec_top_p: rec.and_then(|r| r.top_p),
            rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
            rec_min_p: None,
            rec_presence_penalty: None,
        },
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            mtp_mode,
            mtp_k,
        )
    })
}

// ─── Unload ───────────────────────────────────────────────────────────

pub fn unload_model(mut m: LoadedModel, gpu: &mut rdna_compute::Gpu) {
    // Dense-TP / dense-PP unload: both axes are owned by `m.parallel`.
    // Replace with Single to move out the owned model, then dispatch by variant.
    // The single replace is correct because TP and PP are mutually exclusive.
    //
    // TP arm (PB-TP5, Task 3): TpModel::free frees every owned tensor + drains
    // each device pool (bare drop() leaks — no freeing Drop). Daemon `gpu` untouched.
    //
    // PP arm (P-C, Task 4): PpModel owns its own Gpus + per-stage scratch/KV.
    // PpModel::free frees them + drains each stage pool. Return here: dense-PP owns
    // its entire mesh inside PpModel, so this is the whole teardown. Without the
    // return a dense-PP unload would fall through into the qwen35-PP ArchResident arm.
    // See .agent-memory/notes/dense-pp-unload-panic-pp-gpus-expect.md.
    match std::mem::replace(&mut m.parallel, ModelParallel::Single) {
        ModelParallel::Tp(tp) => {
            tp.free();
        }
        ModelParallel::Pp(crate::model_parallel::PipelineImpl::Dense(pp)) => {
            pp.free();
            let _ = gpu;
            return;
        }
        // EP unload-free (Task 5: EP now owned by m.parallel). An EP model owns its
        // own `Gpus` (the daemon's single `gpu` is unused for ep>1). Without this
        // arm a SUCCESSFUL EP unload leaked every per-rank weight / state / partial.
        // Free per-rank weights → state → partials on each owning device, invalidate
        // caches + graph state, drain each pool, then drop the `Gpus` (tears down
        // comms + devices). The daemon's `gpu` is untouched.
        // (The `partials` free here is what reclaims the ds4/minimax per-rank dummy
        // all-reduce buffer that would otherwise leak per load/unload cycle.)
        ModelParallel::Ep(ep) => {
            let EpState { mut gpus, inner } = ep;
            match inner {
                EpArch::Ds4 {
                    weights,
                    state,
                    partials,
                    ..
                } => {
                    for (r, w) in weights.into_iter().enumerate() {
                        if let Some(dev) = gpus.devices.get_mut(r) {
                            let _ = dev.bind_thread();
                            w.free_gpu(dev);
                        }
                    }
                    for (r, s) in state.into_iter().enumerate() {
                        if let Some(dev) = gpus.devices.get_mut(r) {
                            let _ = dev.bind_thread();
                            s.free_gpu(dev);
                        }
                    }
                    for (r, p) in partials.into_iter().enumerate() {
                        if let Some(dev) = gpus.devices.get_mut(r) {
                            let _ = dev.bind_thread();
                            let _ = dev.free_tensor(p);
                        }
                    }
                }
                EpArch::Minimax {
                    weights,
                    state,
                    partials,
                    ..
                } => {
                    for (r, w) in weights.into_iter().enumerate() {
                        if let Some(dev) = gpus.devices.get_mut(r) {
                            let _ = dev.bind_thread();
                            w.free_gpu(dev);
                        }
                    }
                    for (r, s) in state.into_iter().enumerate() {
                        if let Some(dev) = gpus.devices.get_mut(r) {
                            let _ = dev.bind_thread();
                            s.free_gpu(dev);
                        }
                    }
                    for (r, p) in partials.into_iter().enumerate() {
                        if let Some(dev) = gpus.devices.get_mut(r) {
                            let _ = dev.bind_thread();
                            let _ = dev.free_tensor(p);
                        }
                    }
                }
            }
            for dev in gpus.devices.iter_mut() {
                let _ = dev.bind_thread();
                dev.invalidate_weight_caches();
                dev.invalidate_graph_state();
                dev.drain_pool();
            }
            let _ = gpu;
            // `gpus` drops here, tearing down comms + devices.
            return;
        }
        _ => {}
    }
    if let ModelParallel::Pp(crate::model_parallel::PipelineImpl::ArchResident(mut gpus)) =
        m.parallel
    {
        match m.state.take() {
            Some(ModelState::Qwen35(b)) => {
                if let Some(pl) = b.pipeline {
                    pl.scratch_set.free_gpu_multi(&mut gpus);
                    b.dn_state.free_gpu_multi(&mut gpus, &pl.dn_la_to_device);
                }
                b.kv_cache.free_gpu_multi(&mut gpus);
                b.weights.free_gpu_multi(&mut gpus);
            }
            // Only Qwen35 supports pp>1 today, so the other carriers can never
            // reach this arm with multi-GPU state to free — dropping is correct.
            // Listing them explicitly (rather than `_`) makes that a
            // compiler-enforced invariant: adding a pp>1-capable carrier without
            // a teardown arm here is a build error, not a silent VRAM leak.
            Some(ModelState::Qwen2(_))
            | Some(ModelState::Llama(_))
            | Some(ModelState::Lfm2Moe(_))
            | Some(ModelState::Minimax(_))
            | Some(ModelState::Cohere2Moe(_))
            | Some(ModelState::Deepseek4(_))
            | Some(ModelState::DotsOcr(_))
            | None => {}
        }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return;
    }
    if let Some(spec) = m.speculator {
        // Frees the drafter's GPU buffers (draft weights + scratch) AND its
        // checkpoint ring — a drafter that forgets is a compile error, not a
        // silent VRAM leak. The vestigial `m.session.dflash_checkpoints` (now always
        // empty) is still drained below for defense-in-depth.
        spec.free(gpu);
    }
    if let Some(ev) = m.eviction {
        ev.free_gpu(gpu);
    }
    for (_, snap) in m.session.prefill_checkpoints {
        snap.free_gpu(gpu);
    }
    for (_, snap) in m.session.dflash_checkpoints {
        snap.free_gpu(gpu);
    }
    // Free arch-specific GPU state from the carrier bundle
    if let Some(state) = m.state {
        match state {
            ModelState::Qwen2(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Qwen35(b) => {
                if let Some(head) = b.mtp_head {
                    head.free_gpu(gpu);
                }
                b.kv_cache.free_gpu(gpu);
                b.scratch.free_gpu(gpu);
                b.weights.free_gpu(gpu);
                b.dn_state.free_gpu(gpu);
            }
            ModelState::Llama(b) => {
                b.scratch.free_gpu(gpu);
                b.weights.free_gpu(gpu);
                b.kv.free_gpu(gpu);
            }
            ModelState::Lfm2Moe(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Minimax(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Cohere2Moe(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Deepseek4(b) => {
                b.pbs.free_gpu(gpu);
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::DotsOcr(b) => {
                b.weights.free_gpu(gpu);
                b.state.free_gpu(gpu);
                // config is host-side — no GPU free
            }
        }
    }
    // Non-core arch weights
    if let Some(v) = m.vision {
        v.weights.free_gpu(gpu);
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
}

#[cfg(test)]
mod registry_tests {
    use super::{LoadedModel, REGISTRY};

    #[test]
    fn cask_m_folding_is_disabled_for_drafts() {
        assert!(super::effective_cask_m_folding(true, None));
        assert!(!super::effective_cask_m_folding(true, Some("draft.hfq")));
        assert!(!super::effective_cask_m_folding(false, Some("draft.hfq")));
    }

    fn test_tokenizer() -> hipfire_runtime::tokenizer::Tokenizer {
        hipfire_runtime::tokenizer::Tokenizer::from_hf_json(
            r#"{"model":{"vocab":{"a":0},"merges":[]}}"#,
        )
        .expect("test tokenizer")
    }

    #[test]
    fn skeleton_stores_supplied_mtp_k() {
        let model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            None,
            "auto",
            6,
        );

        assert_eq!(model.meta.mtp_k, 6);
    }

    #[test]
    fn skeleton_stores_supplied_mtp_mode() {
        let model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            None,
            "on",
            6,
        );

        assert_eq!(model.meta.mtp_mode, "on");
    }

    #[test]
    fn skeleton_stores_default_mtp_k() {
        let model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            None,
            "auto",
            3,
        );

        assert_eq!(model.meta.mtp_k, 3);
    }

    #[test]
    fn normalize_mtp_k_bounds_mtp_arches() {
        let cases = [(1, 1), (8, 8)];
        for arch_id in [5, 6, 9] {
            for (input, expected) in cases {
                assert_eq!(
                    super::normalize_mtp_k(arch_id, Some(input)),
                    Ok(expected),
                    "arch_id={arch_id}, input={input}"
                );
            }
        }
    }

    #[test]
    fn normalize_mtp_k_rejects_out_of_range_values() {
        for arch_id in [5, 6, 9] {
            for input in [0, 9, 10] {
                assert!(
                    super::normalize_mtp_k(arch_id, Some(input)).is_err(),
                    "arch_id={arch_id}, input={input}"
                );
            }
        }
    }

    /// Every known arch_id must be claimed by AT MOST one carrier, for both
    /// source namespaces (HFQ header ids and `derive_arch_id` dir ids). This
    /// guards the otherwise-silent first-match overlap in `load_model`: add a
    /// carrier whose `claims_arch_id` collides with an existing one and this
    /// fails in CI instead of mis-routing weights at runtime.
    #[test]
    fn carriers_are_disjoint() {
        // Sweep well past the assigned range, plus the reserved sentinels
        // (20 = DFlash draft, 0xFF = toy/template — neither should dispatch).
        let ids = (0u32..=64).chain([20, 0xFF]);
        for id in ids {
            for is_dir in [false, true] {
                let claimers: Vec<&str> = REGISTRY
                    .iter()
                    .filter(|c| c.claims_arch_id(id, is_dir))
                    .map(|c| c.name())
                    .collect();
                assert!(
                    claimers.len() <= 1,
                    "arch_id={id} is_dir={is_dir} claimed by multiple carriers: {claimers:?}"
                );
            }
        }
    }

    /// Pin the intended routing so a future probe edit can't silently move an
    /// existing model to the wrong carrier. `is_dir` matters in general, but
    /// Qwen2 routes to the qwen2 carrier in BOTH forms (HFQ id 7 and dir, which
    /// derives to id 7) so its Q/K/V attention biases load — the llama-family
    /// dir loader (id 1) drops them.
    #[test]
    fn known_ids_route_as_expected() {
        let cases: &[(u32, bool, &str)] = &[
            (7, false, "qwen2"),
            (7, true, "qwen2"),
            (5, false, "qwen35"),
            (6, false, "qwen35"),
            (5, true, "qwen35"),
            (6, true, "qwen35"),
            (0, false, "llama"),
            (1, false, "llama"),
            (0, true, "llama"),
            (1, true, "llama"),
            (8, false, "dots_ocr"),
            (9, false, "deepseek4"),
            (10, false, "minimax"),
            (11, false, "lfm2moe"),
            (12, false, "cohere2moe"),
        ];
        for &(id, is_dir, want) in cases {
            let got: Vec<&str> = REGISTRY
                .iter()
                .filter(|c| c.claims_arch_id(id, is_dir))
                .map(|c| c.name())
                .collect();
            assert_eq!(
                got,
                vec![want],
                "arch_id={id} is_dir={is_dir} should route to exactly [{want}]"
            );
        }
    }

    /// The unassigned HFQ ids 2..=4 must reach NO carrier — this is the
    /// regression guard for fix B (the old `arch_id < 5` open range silently
    /// loaded them as llama).
    #[test]
    fn unassigned_low_ids_match_nothing() {
        for id in [2u32, 3, 4] {
            let n = REGISTRY
                .iter()
                .filter(|c| c.claims_arch_id(id, false))
                .count();
            assert_eq!(n, 0, "arch_id={id} (unassigned) should match no carrier");
        }
    }

    #[test]
    fn qwen_mtp_on_is_rejected_before_native_head_loading() {
        let error = super::reject_qwen_native_mtp("on")
            .expect_err("Qwen native MTP must remain disabled pending SPEC-003");

        assert!(error.contains("SPEC-003"), "{error}");
        assert!(super::reject_qwen_native_mtp("auto").is_ok());
        assert!(super::reject_qwen_native_mtp("off").is_ok());
    }

    /// Exercises the published loader owner, rather than `Speculator::free` in
    /// isolation: `unload_model` must release DFlash, its DDTree snapshots, the
    /// Qwen35 target, and the pool before the next model lifetime begins.
    #[test]
    #[ignore = "requires an AMD GPU plus qwen3.6-27b.mq4 and qwen36-27b-dflash-mq4.hfq"]
    fn unload_model_reclaims_published_qwen35_dflash_state() {
        use std::path::Path;

        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_DFLASH_FREE_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        let draft_path = std::env::var("HIPFIRE_DFLASH_FREE_DRAFT")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen36-27b-dflash-mq4.hfq"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_DFLASH_FREE_TARGET"
        );
        assert!(
            Path::new(&draft_path).is_file(),
            "draft fixture not found: {draft_path}; set HIPFIRE_DFLASH_FREE_DRAFT"
        );

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let cask = super::CaskConfig::default();
        let mesh = super::DeviceMesh::single();
        let load = |gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                Some(&draft_path),
                Some("q8"),
                None,
                None,
                &cask,
                &mesh,
                None,
                super::SpecLoadCfg {
                    ddtree_budget: Some(1),
                    ddtree_topk: Some(1),
                    ..Default::default()
                },
                gpu,
            )
        };

        // Warm the fixed loader/kernel allocations once; the second lifetime is
        // the leak check, so only model-owned resources affect its baseline.
        let warmup = load(&mut gpu).expect("warmup load_model with DFlash");
        super::unload_model(warmup, &mut gpu);
        let baseline = free_vram(&gpu, "after warmup unload").expect("baseline VRAM");

        let model = load(&mut gpu).expect("measured load_model with DFlash");
        let result = (|| -> Result<(), String> {
            if model.speculator.is_none() {
                return Err(
                    "draft was supplied but DFlash was not published in LoadedModel".into(),
                );
            }
            let loaded = free_vram(&gpu, "after DFlash model load")?;
            if loaded >= baseline {
                return Err(format!(
                    "published DFlash model did not allocate observable VRAM: baseline={baseline}, loaded={loaded}"
                ));
            }
            Ok(())
        })();

        // This is deliberately outside `result`: every assertion/error after a
        // successful publication reaches the production `unload_model` path.
        super::unload_model(model, &mut gpu);
        let after = free_vram(&gpu, "after measured unload").expect("post-unload VRAM");
        if let Err(error) = result {
            panic!("{error}");
        }
        assert_eq!(
            after, baseline,
            "unload_model leaked published Qwen35 DFlash state: baseline={baseline}, after={after}"
        );
    }

    /// `finish_qwen35_load` receives a fully GPU-backed bundle before opening a
    /// requested TriAttention sidecar. A sidecar read failure must free that
    /// unpublished bundle rather than leaving its weights/KV/scratch resident.
    #[test]
    #[ignore = "requires an AMD GPU and qwen3.6-27b.mq4"]
    fn rejected_qwen35_sidecar_load_reclaims_unpublished_bundle() {
        use std::path::Path;

        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_QWEN35_SIDECAR_FAILURE_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_QWEN35_SIDECAR_FAILURE_TARGET"
        );

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let load = |cask: &super::CaskConfig, gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                None,
                Some("q8"),
                None,
                None,
                cask,
                &mesh,
                None,
                super::SpecLoadCfg::default(),
                gpu,
            )
        };

        let no_eviction = super::CaskConfig::default();
        let warmup = load(&no_eviction, &mut gpu).expect("warmup Qwen35 load");
        super::unload_model(warmup, &mut gpu);
        let baseline = free_vram(&gpu, "after warmup unload").expect("baseline VRAM");

        let rejected = super::CaskConfig {
            sidecar: Some(format!("{target_path}.missing-triattn-sidecar")),
            cask_m_folding: false,
            budget: 32,
            beta: 8,
            core_frac: 0.5,
            fold_m: 2,
        };
        let error = match load(&rejected, &mut gpu) {
            Ok(model) => {
                super::unload_model(model, &mut gpu);
                panic!("missing sidecar must reject the load");
            }
            Err(error) => error,
        };
        assert!(error.contains("cask sidecar load failed"), "{error}");
        gpu.drain_pool();
        let after = free_vram(&gpu, "after rejected sidecar load").expect("post-error VRAM");
        assert_eq!(
            after, baseline,
            "sidecar read failure leaked unpublished Qwen35 bundle: baseline={baseline}, after={after}"
        );
    }
}
