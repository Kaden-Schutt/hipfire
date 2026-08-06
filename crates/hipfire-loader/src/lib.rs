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
pub mod parallel_capability;
pub mod spec_build;
pub use model_parallel::{ModelParallel, ModelParallelKind, PipelineImpl};

use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen35::mtp_head::Qwen35MtpHead;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, Qwen35Scratch, Qwen35ScratchSet, Qwen35Weights};
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::moe_plan::{MoEExecutionKind, MoEExecutionPolicy};
use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind, Gpus};
use hipfire_runtime::spec::{SpecEmit, SpecEmitCtx, SpecTargetGuard, Speculator};
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use parallel_capability::{
    resolve, ModelVariant, ParallelAdmission, ParallelAxis, RawParallelRequest,
};
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

    /// Source-aware parallel variant classification (CAP-001 Task 2).
    ///
    /// Each carrier inspects the concrete [`ModelSource`] (HFQ vs safetensors
    /// Dir) to produce the fine-grained [`ModelVariant`] needed for the
    /// parallel capability policy table. Default rejects — carriers that want
    /// classification override this method.
    ///
    /// # Classification scope
    ///
    /// Carriers classify **family and source facts only**: arch_id (HFQ header
    /// vs `derive_arch_id` namespace), tensor existence (VL tower, QK-norm),
    /// and config fields (expert count). They do NOT encode parallelism
    /// policy, axis choice, or normalisation.
    fn classify_parallel_variant(
        &self,
        src: &ModelSource,
    ) -> Result<crate::parallel_capability::ModelVariant, String> {
        Err(format!(
            "{}: CAP-001 variant classification unsupported for {}",
            self.name(),
            src.describe()
        ))
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

/// Source-aware carrier + variant resolution (CAP-001 Task 2).
///
/// Resolves a concrete [`ModelSource`] to the matching [`Carrier`] (via
/// [`Carrier::probe`], which respects HFQ vs safetensors namespaces) and
/// then calls [`Carrier::classify_parallel_variant`] to produce the fine-
/// grained [`ModelVariant`] for the parallel capability policy table.
///
/// # Errors
///
/// - `"no carrier for ..."` when zero carriers claim the source's arch_id.
/// - `"ambiguous carrier ..."` when multiple carriers match (arch_id
///   collision in the registry).
/// - Carries through any error from [`Carrier::classify_parallel_variant`].
pub fn classify_source(
    src: &ModelSource,
) -> Result<
    (
        &'static dyn Carrier,
        crate::parallel_capability::ModelVariant,
    ),
    String,
> {
    let arch_id = src
        .arch_id()
        .ok_or_else(|| format!("no arch_id in source: {}", src.describe()))?;
    let matching: Vec<&&dyn Carrier> = REGISTRY.iter().filter(|c| c.probe(src)).collect();
    if matching.is_empty() {
        return Err(format!(
            "no carrier for arch_id {} ({})",
            arch_id,
            src.describe()
        ));
    }
    if matching.len() > 1 {
        let names: Vec<&str> = matching.iter().map(|c| c.name()).collect();
        return Err(format!(
            "ambiguous carrier for arch_id {} ({}): carriers {:?} match",
            arch_id,
            src.describe(),
            names,
        ));
    }
    let carrier = matching[0];
    let variant = carrier.classify_parallel_variant(src)?;
    Ok((*carrier, variant))
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
        let unbounded = hipfire_config::developer_var("HIPFIRE_PROMPT_CACHE_UNBOUNDED")
            .ok()
            .as_deref()
            == Some("1");
        let cap = if unbounded {
            None
        } else {
            Some(
                hipfire_config::developer_var("HIPFIRE_PROMPT_CACHE_CAP")
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

    /// Drop all cached assistant-turn token sequences (authoritative cold reset).
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// Read-only emptiness probe for test/snapshot surfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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

/// A typed discriminant for the architecture state used by reset ownership
/// validation. Keeping this separate from the GPU-backed bundles makes the
/// ownership contract checkable without constructing a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStateKind {
    Qwen2,
    Qwen35,
    Llama,
    Lfm2Moe,
    Minimax,
    Cohere2Moe,
    Deepseek4,
    DotsOcr,
}

impl From<&ModelState> for ModelStateKind {
    fn from(state: &ModelState) -> Self {
        match state {
            ModelState::Qwen2(_) => Self::Qwen2,
            ModelState::Qwen35(_) => Self::Qwen35,
            ModelState::Llama(_) => Self::Llama,
            ModelState::Lfm2Moe(_) => Self::Lfm2Moe,
            ModelState::Minimax(_) => Self::Minimax,
            ModelState::Cohere2Moe(_) => Self::Cohere2Moe,
            ModelState::Deepseek4(_) => Self::Deepseek4,
            ModelState::DotsOcr(_) => Self::DotsOcr,
        }
    }
}

/// A reset can only proceed when the architecture state is owned by the
/// selected parallelism axis and the model metadata agrees with that state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResetError {
    InvalidOwnership {
        parallel: ModelParallelKind,
        state: Option<ModelStateKind>,
    },
    Architecture {
        parallel: ModelParallelKind,
        state: Option<ModelStateKind>,
        arch_id: u32,
        message: String,
    },
    Session {
        message: String,
    },
    Speculator {
        message: String,
    },
}

impl ResetError {
    /// A failed reset means the daemon cannot prove that device state is
    /// request-local.  Session/speculator failures are GPU-buffer ownership
    /// failures; architecture failures are also fatal when they originate in
    /// a device operation.  The daemon uses this classification to stop
    /// serving instead of attempting another request on possibly dirty state.
    pub fn is_gpu_fatal(&self) -> bool {
        matches!(self, Self::Session { .. } | Self::Speculator { .. })
            || matches!(self, Self::Architecture { message, .. } if is_gpu_reset_message(message))
    }
}

fn is_gpu_reset_message(message: &str) -> bool {
    [
        "bind ",
        "reset ",
        "synchronize ",
        "invalidate ",
        "memset",
        "free",
    ]
    .iter()
    .any(|prefix| message.contains(prefix))
        || message.contains("GPU")
        || message.contains("hip")
}

/// Validate which object owns the state that a total reset must clear.
///
/// TP, dense PP, and EP carry their architecture state inside the parallel
/// owner, so `LoadedModel.state` must be `None`. Qwen35 PP keeps Qwen35 state
/// in `LoadedModel.state`, and every single model keeps architecture state
/// there. Parallel-owned paths validate their supported architecture family
/// against `arch_id`; single-owned state validates the concrete state kind.
pub(crate) fn reset_ownership_kind(
    parallel: ModelParallelKind,
    state: Option<ModelStateKind>,
    arch_id: u32,
) -> Result<ModelParallelKind, ResetError> {
    use ModelParallelKind::*;

    let valid = match parallel {
        Tp | PpDense | Ep => state.is_none(),
        PpQwen35 => state == Some(ModelStateKind::Qwen35),
        Single => state.is_some(),
    };

    if !valid {
        return Err(ResetError::InvalidOwnership { parallel, state });
    }

    let architecture_valid = match parallel {
        Tp | PpDense => state.is_none() && matches!(arch_id, 0 | 1),
        Ep => state.is_none() && matches!(arch_id, 9 | 10),
        PpQwen35 => state == Some(ModelStateKind::Qwen35) && matches!(arch_id, 5 | 6),
        Single => match state {
            Some(state) => state_matches_arch_id(state, arch_id),
            None => false,
        },
    };
    if !architecture_valid {
        return Err(ResetError::Architecture {
            parallel,
            state,
            arch_id,
            message: format!(
                "arch_id={arch_id} is incompatible with parallel={parallel:?} state={state:?}"
            ),
        });
    }

    Ok(parallel)
}

fn state_matches_arch_id(state: ModelStateKind, arch_id: u32) -> bool {
    match state {
        ModelStateKind::Qwen2 => arch_id == 7,
        ModelStateKind::Qwen35 => matches!(arch_id, 5 | 6),
        ModelStateKind::Llama => matches!(arch_id, 0 | 1),
        ModelStateKind::Lfm2Moe => arch_id == 11,
        ModelStateKind::Minimax => arch_id == 10,
        ModelStateKind::Cohere2Moe => arch_id == 12,
        ModelStateKind::Deepseek4 => arch_id == 9,
        ModelStateKind::DotsOcr => arch_id == 8,
    }
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

// ─── LoadedModel ──────────────────────────────────────────────────────

/// Immutable load-time admission snapshot.  Its fields cannot be fabricated by
/// callers; only the resolver-owned [`AdmittedLoad`] can create one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmissionRecord {
    admission: ParallelAdmission,
}

impl AdmissionRecord {
    fn from_admission(admission: ParallelAdmission) -> Self {
        Self { admission }
    }

    fn variant(self) -> ModelVariant {
        self.admission.variant()
    }

    fn requested(self) -> RawParallelRequest {
        self.admission.requested()
    }

    fn effective(self) -> RawParallelRequest {
        self.admission.effective()
    }
}

pub struct LoadedModel {
    pub arch_id: u32,
    pub pp: usize,
    pub pp_gpus: Option<Gpus>,
    pub pp_scratch_set: Option<hipfire_arch_qwen35::qwen35::Qwen35ScratchSet>,
    pub pp_dn_la_to_device: Option<Vec<u8>>,
    pub ep: Option<EpState>,
    // Shared arch state
    pub state: Option<ModelState>,
    pub kv_cache: Option<llama::KvCache>,
    pub dn_state: Option<DeltaNetState>,
    // Reusable Qwen2 recurrent state (used by dots_ocr and Qwen2 non-core falcon)
    pub qwen2_state: Option<hipfire_arch_qwen2::qwen2::Qwen2State>,
    // DeepSeek V4 Flash (arch_id=9) single-GPU config/weights/state/eos now live
    // in `state` as ModelState::Deepseek4(Deepseek4Bundle) so unload teardown is
    // compiler-enforced and the bundle can be borrowed as a `SpecTarget`.
    pub deepseek4_pbs: Option<hipfire_arch_deepseek4::forward::PrefillBatchScratch>,
    // DeepSeek V4 (arch_id=9) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Ds4), NOT in `state`, so there is no Deepseek4Bundle for EP
    // models — the eos must be carried here (mirrors `minimax_eos_tok`).
    pub deepseek4_eos_tok: u32,
    // MiniMax-M2 (arch_id=10) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Minimax), NOT in `state`, so `minimax()` is None for EP
    // models — the eos must be carried here (mirrors `deepseek4_eos_tok`).
    pub minimax_eos_tok: u32,
    // LFM2.5-8B-A1B (arch_id=11) and MiniMax-M2 (arch_id=10) live in
    // `state` as ModelState::{Lfm2Moe,Minimax} so unload teardown is
    // compiler-enforced (see ModelState).
    // MTP config
    pub mtp_mode: String,
    pub mtp_k: usize,
    pub mtp_weights_present: bool,
    // Qwen3.5/3.6 native MTP (NextN) head (arch_id=21). Loaded once at model
    // load when a bundled `.mq4-mtp` trailer OR a separate `.mtp` sidecar is
    // present alongside the trunk. Persistent for the life of the model;
    // `generate_qwen35_mtp` allocates a fresh per-request `MtpSpecState`
    // against it (so the recurrent MTP-KV never bleeds across requests). None
    // for every other arch and for qwen35 trunks without an MTP head.
    pub qwen35_mtp_head: Option<hipfire_arch_qwen35::mtp_head::Qwen35MtpHead>,
    // dots.ocr state
    pub dots_ocr_config: Option<hipfire_arch_dots_ocr::dots_ocr::DotsOcrConfig>,
    pub dots_ocr_weights: Option<hipfire_arch_dots_ocr::dots_ocr::DotsOcrWeights>,
    // Vision state
    pub vision_config: Option<qwen35_vl::VisionConfig>,
    pub vision_weights: Option<qwen35_vl::VisionWeights>,
    // Shared
    pub tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer>,
    pub seq_pos: usize,
    pub max_seq: usize,
    pub physical_cap: usize,
    pub eviction: Option<Eviction>,
    pub kv_adaptive: Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    pub conversation_tokens: Vec<u32>,
    pub prefill_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub dflash_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub asst_turn_cache: AsstTurnCache,
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
    pub model_path: String,
    /// The model's speculative-decode drafter+verifier, when a draft model is
    /// loaded (`Box<dyn Speculator>` so the daemon's decode loop is agnostic to
    /// DFlash chain / DDTree tree / future MTP). Replaces the old
    /// `dflash: Option<DflashState>` field — the `DflashState` now lives inside
    /// the `DflashSpeculator` impl behind this trait object.
    pub speculator: Option<Box<dyn Speculator>>,
    pub chat_template: Option<String>,
    // Author-recommended sampling defaults, baked into the .hfq's
    // `generation_config` metadata and read at load time on the HFQ source
    // path (raw-safetensors PP path leaves them `None`). The generate handler
    // falls back to these when the request omits the matching knob, before the
    // arch-ladder defaults. `rec_min_p` / `rec_presence_penalty` are NOT carried
    // in generation_config (they reach the daemon only via the request), so they
    // stay `None` on the load path.
    pub rec_temperature: Option<f32>,
    pub rec_top_p: Option<f32>,
    pub rec_top_k: Option<f32>,
    pub rec_min_p: Option<f32>,
    pub rec_presence_penalty: Option<f32>,
    // ── Device-mesh additions (no beta equivalent) ────────────────────
    /// Dense tensor-parallel model (CAP-001/device-mesh axis). `Some` only
    /// when a TP load was admitted; beta's flat shape has no TP carrier.
    pub tp_model: Option<hipfire_runtime::tp_serve::TpModel>,
    /// Dense pipeline-parallel model (device-mesh axis). `Some` only for the
    /// dense PP load path; beta's flat `pp`/`pp_gpus` fields carry the qwen35
    /// arch-resident mesh instead.
    pub pp_model: Option<hipfire_runtime::pp_serve::PpModel>,
    /// Fingerprint of the preprocessed image for the active VL model turn.
    /// Request state, not a model cache: a reset clears it, and the daemon
    /// updates it only after image preprocessing succeeds.
    pub vl_image_state: Option<u64>,
}

impl LoadedModel {
    /// Reset CPU-owned request state without requiring a GPU. Checkpoint
    /// snapshots are deliberately not touched here; [`Self::reset`] drains
    /// and frees those before calling this CPU-only helper.
    pub fn reset_cpu(&mut self) {
        self.seq_pos = 0;
        self.conversation_tokens.clear();
        self.vl_image_state = None;
        if let Some(ad) = self.kv_adaptive.as_mut() {
            ad.reset();
        }
    }

    /// Total reset of request-scoped state. Frees GPU checkpoint snapshots
    /// (DeltaNetSnapshot has no Drop) before clearing, then resets scalars.
    pub fn reset(&mut self, gpu: &mut rdna_compute::Gpu) {
        let _ = self.reset_checked(gpu);
    }

    /// Fallible reset for the model owner. Checkpoint GPU frees are attempted
    /// before the request state is cleared, and any failure is returned. A
    /// checkpoint whose buffer free fails remains owned by the model for a
    /// later poison/unload retry.
    pub fn reset_checked(&mut self, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        let mut first_error = None;
        let mut prefill_remaining = Vec::new();
        for (position, mut snap) in self.prefill_checkpoints.drain(..) {
            if let Err(error) = snap.free_gpu_checked(gpu) {
                first_error.get_or_insert(error);
                prefill_remaining.push((position, snap));
            }
        }
        self.prefill_checkpoints = prefill_remaining;
        let mut dflash_remaining = Vec::new();
        for (position, mut snap) in self.dflash_checkpoints.drain(..) {
            if let Err(error) = snap.free_gpu_checked(gpu) {
                first_error.get_or_insert(error);
                dflash_remaining.push((position, snap));
            }
        }
        self.dflash_checkpoints = dflash_remaining;
        self.reset_cpu();
        first_error.map_or(Ok(()), Err)
    }

    /// Total reset authority for the model owner (daemon `model_reset_context`).
    /// Validates the load layout before touching state (silent-corruption
    /// guard), frees GPU checkpoint snapshots + CPU/VL request state, and
    /// resets the speculative drafter. Architecture/device recurrent reset is
    /// composed by the caller (daemon-side `reset_qwen35_recurrent` /
    /// `slot.reset_recurrent`), which owns the live arch surface.
    pub fn reset_context(&mut self, gpu: &mut rdna_compute::Gpu) -> Result<(), ResetError> {
        validate_reset_layout(
            self.arch_id,
            self.pp,
            self.pp_gpus.as_ref(),
            self.pp_dn_la_to_device.as_deref(),
            self.ep.as_ref(),
            &self.state,
        )?;
        self.reset_checked(gpu).map_err(|message| ResetError::Session { message })?;
        if let Some(spec) = self.speculator.as_mut() {
            spec.reset(gpu)
                .map_err(|message| ResetError::Speculator { message })?;
        }
        reset_owned_arch_state(self, gpu)
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
    ) -> Self {
        LoadedModel {
            arch_id,
            pp: 1,
            ep: None,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            state: None,
            kv_cache: None,
            dn_state: None,
            qwen2_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            minimax_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            qwen35_mtp_head: None,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0,
            max_seq,
            physical_cap,
            eviction: None,
            kv_adaptive: None,
            conversation_tokens: Vec::new(),
            asst_turn_cache: AsstTurnCache::new_from_env(),
            prefill_checkpoints: Vec::new(),
            dflash_checkpoints: Vec::new(),
            decoded_vocab: None,
            model_path,
            speculator: None,
            chat_template,
            rec_temperature: None,
            rec_top_p: None,
            rec_top_k: None,
            rec_min_p: None,
            rec_presence_penalty: None,
            tp_model: None,
            pp_model: None,
            vl_image_state: None,
        }
    }

    /// Qwen35 arch-resident pipeline-parallel (pp>1) skeleton: the flat PP
    /// fields set together so they cannot be set piecemeal (a dropped
    /// `pp_scratch_set` is a silent VRAM leak; `pp_gpus`/`pp_dn_la_to_device`
    /// are `.expect()`ed in unload).
    pub fn skeleton_pp(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        pp: usize,
        pp_gpus: Gpus,
        pp_scratch_set: Qwen35ScratchSet,
        pp_dn_la_to_device: Vec<u8>,
    ) -> Self {
        LoadedModel {
            pp,
            pp_gpus: Some(pp_gpus),
            pp_scratch_set: Some(pp_scratch_set),
            pp_dn_la_to_device: Some(pp_dn_la_to_device),
            ..LoadedModel::skeleton(
                arch_id,
                tokenizer,
                max_seq,
                physical_cap,
                model_path,
                chat_template,
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpArchKind {
    Ds4,
    Minimax,
}

fn validate_ep_layout(
    arch_id: u32,
    arch_kind: EpArchKind,
    state_len: usize,
    device_len: usize,
) -> Result<(), ResetError> {
    let expected_kind = match arch_id {
        9 => EpArchKind::Ds4,
        10 => EpArchKind::Minimax,
        _ => {
            return Err(architecture_error(
                ModelParallelKind::Ep,
                None,
                arch_id,
                "unsupported EP architecture",
            ));
        }
    };
    if arch_kind != expected_kind {
        return Err(architecture_error(
            ModelParallelKind::Ep,
            None,
            arch_id,
            format!("EP arch_id={arch_id} does not match {arch_kind:?} state"),
        ));
    }
    ep_owner_visitation(arch_id, state_len, device_len).map(|_| ())
}

/// Return the EP owners in reset order.  Keeping this as a pure layout helper
/// makes the rank-visitation contract testable without constructing GPU state;
/// the reset loop consumes the same order for the actual device reset.
fn ep_owner_visitation(
    arch_id: u32,
    state_len: usize,
    device_len: usize,
) -> Result<Vec<usize>, ResetError> {
    if state_len != device_len {
        return Err(architecture_error(
            ModelParallelKind::Ep,
            None,
            arch_id,
            format!("EP state/device cardinality mismatch: state={state_len} devices={device_len}"),
        ));
    }
    Ok((0..state_len).collect())
}

fn require_qwen35_pipeline_metadata(
    pipeline_present: bool,
    arch_id: u32,
) -> Result<(), ResetError> {
    if pipeline_present {
        Ok(())
    } else {
        Err(architecture_error(
            ModelParallelKind::PpQwen35,
            Some(ModelStateKind::Qwen35),
            arch_id,
            "Qwen35 ArchResident PP is missing pipeline metadata",
        ))
    }
}

fn reject_qwen35_single_pipeline_metadata(
    pipeline_present: bool,
    arch_id: u32,
) -> Result<(), ResetError> {
    if pipeline_present {
        Err(architecture_error(
            ModelParallelKind::Single,
            Some(ModelStateKind::Qwen35),
            arch_id,
            "single Qwen35 model unexpectedly carries pipeline metadata",
        ))
    } else {
        Ok(())
    }
}

fn validate_qwen35_recurrent_cardinality(
    expected: usize,
    s_scales: usize,
    conv_states: usize,
    s_ef_residual: usize,
    dn_la_to_device: usize,
    arch_id: u32,
) -> Result<(), ResetError> {
    for (name, actual) in [
        ("s_scales", s_scales),
        ("conv_states", conv_states),
        ("dn_la_to_device", dn_la_to_device),
    ] {
        if actual != expected {
            return Err(architecture_error(
                ModelParallelKind::PpQwen35,
                Some(ModelStateKind::Qwen35),
                arch_id,
                format!("Qwen35 PP {name} cardinality={actual}, expected={expected}"),
            ));
        }
    }
    // Error feedback is optional for Qwen35. An enabled vector must still be
    // one entry per DeltaNet layer; an empty vector means the feature is off.
    if s_ef_residual != 0 && s_ef_residual != expected {
        return Err(architecture_error(
            ModelParallelKind::PpQwen35,
            Some(ModelStateKind::Qwen35),
            arch_id,
            format!("Qwen35 PP s_ef_residual cardinality={s_ef_residual}, expected={expected}"),
        ));
    }
    Ok(())
}

/// Validate a banded qwen35 PP load against the flat canonical fields: the
/// recurrent-state cardinalities must agree with the layer→device map
/// (`LoadedModel.pp_dn_la_to_device`), and every mapped device must exist in
/// the PP mesh (`LoadedModel.pp_gpus`). Callers pass the layer→device map and
/// device count directly so the pure helper stays testable without a GPU.
fn validate_qwen35_pipeline_layout(
    bundle: &hipfire_arch_qwen35::Qwen35Bundle,
    dn_la_to_device: &[u8],
    device_len: usize,
    arch_id: u32,
) -> Result<(), ResetError> {
    let expected = bundle.dn_state.s_matrices.len();
    validate_qwen35_recurrent_cardinality(
        expected,
        bundle.dn_state.s_scales.len(),
        bundle.dn_state.conv_states.len(),
        bundle.dn_state.s_ef_residual.len(),
        dn_la_to_device.len(),
        arch_id,
    )?;
    qwen35_pp_owner_visitation(dn_la_to_device, expected, device_len, arch_id)?;
    Ok(())
}

/// Return the PP owner for each recurrent layer in visitation order.  The
/// reset loop below uses this exact vector, so emulated meshes exercise the
/// same owner routing as physical PP meshes without needing a GPU in tests.
fn qwen35_pp_owner_visitation(
    dn_la_to_device: &[u8],
    expected_layers: usize,
    device_len: usize,
    arch_id: u32,
) -> Result<Vec<usize>, ResetError> {
    if dn_la_to_device.len() != expected_layers {
        return Err(architecture_error(
            ModelParallelKind::PpQwen35,
            Some(ModelStateKind::Qwen35),
            arch_id,
            format!(
                "Qwen35 PP dn_la_to_device cardinality={}, expected={expected_layers}",
                dn_la_to_device.len()
            ),
        ));
    }
    if let Some((layer, device_id)) = dn_la_to_device
        .iter()
        .enumerate()
        .find(|(_, device_id)| usize::from(**device_id) >= device_len)
    {
        return Err(architecture_error(
            ModelParallelKind::PpQwen35,
            Some(ModelStateKind::Qwen35),
            arch_id,
            format!("Qwen35 PP layer {layer} maps to device {device_id}"),
        ));
    }
    Ok(dn_la_to_device
        .iter()
        .map(|&device| device as usize)
        .collect())
}

/// Invoke every physical reset owner in the order selected by the production
/// layout. Keeping the invocation seam injectable makes owner coverage testable
/// without pretending that booleans are GPU state.
fn reset_each_owner<F>(owners: &[usize], mut reset: F) -> Result<(), ResetError>
where
    F: FnMut(usize, usize) -> Result<(), ResetError>,
{
    for (index, &owner) in owners.iter().enumerate() {
        reset(index, owner)?;
    }
    Ok(())
}

fn architecture_error(
    parallel: ModelParallelKind,
    state: Option<ModelStateKind>,
    arch_id: u32,
    message: impl Into<String>,
) -> ResetError {
    ResetError::Architecture {
        parallel,
        state,
        arch_id,
        message: message.into(),
    }
}

/// Validate the flat parallel layout against the concrete arch state before a
/// context reset / teardown. Single qwen35 models must NOT carry PP metadata;
/// pp>1 qwen35 models MUST carry `pp_gpus` + `pp_dn_la_to_device` and satisfy
/// the banded-PP invariants; EP models must match their arch's state/device
/// cardinality. Called from `unload_model` so a corrupted layout fails with a
/// descriptive error instead of a partial free or an `.expect()` panic.
fn qwen35_recurrent_groups<'a, T>(
    s_matrices: &'a [T],
    s_scales: &'a [T],
    conv_states: &'a [T],
    s_ef_residual: &'a [T],
) -> [&'a [T]; 4] {
    [s_matrices, s_scales, conv_states, s_ef_residual]
}

/// Total architecture/device reset for the flat LoadedModel: PP recurrent
/// memset (per LA device), EP per-rank reset, or single-arch bundle reset.
fn reset_owned_arch_state(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), ResetError> {
    let arch_id = m.arch_id;
    if m.tp_model.is_some() || m.pp_model.is_some() {
        // Dense TP/dense-PP carriers load with `state=None` (their weights/KV
        // live inside the carrier). Their `prefill` writes from position 0 and
        // the dense-reset contract intentionally does not zero physical KV, so
        // a reset needs no arch-specific work here.
        return Ok(());
    }
    if m.pp > 1 {
        match (&mut m.state, &mut m.pp_gpus, m.pp_dn_la_to_device.as_ref()) {
            (Some(ModelState::Qwen35(bundle)), Some(gpus), Some(la)) => {
                reset_qwen35_pipeline_recurrent(&bundle.dn_state, la, gpus, arch_id)?;
                bundle.kv_cache.compact_offset = 0;
                Ok(())
            }
            _ => require_qwen35_pipeline_metadata(false, arch_id),
        }
    } else if m.ep.is_some() {
        reset_ep_state(m.ep.as_mut().expect("checked is_some"), arch_id)
    } else {
        match m.state.as_mut() {
            Some(state) => reset_single_arch_state(state, arch_id, gpu),
            None => Err(ResetError::InvalidOwnership {
                parallel: ModelParallelKind::Single,
                state: None,
            }),
        }
    }
}

fn reset_single_arch_state(
    state: &mut ModelState,
    arch_id: u32,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), ResetError> {
    match state {
        ModelState::Qwen2(b) => b.state.reset(),
        ModelState::Qwen35(b) => {
            b.dn_state.reset(gpu).map_err(|error| {
                architecture_error(
                    ModelParallelKind::Single,
                    Some(ModelStateKind::Qwen35),
                    arch_id,
                    format!("reset Qwen35 recurrent state: {error:?}"),
                )
            })?;
            b.kv_cache.compact_offset = 0;
        }
        ModelState::Llama(b) => b.kv.compact_offset = 0,
        ModelState::Lfm2Moe(b) => {
            b.state.reset(gpu).map_err(|error| {
                architecture_error(
                    ModelParallelKind::Single,
                    Some(ModelStateKind::Lfm2Moe),
                    arch_id,
                    error,
                )
            })?;
            b.state.kv.compact_offset = 0;
        }
        ModelState::Minimax(b) => b.state.reset(),
        ModelState::Cohere2Moe(b) => b.state.reset(gpu).map_err(|error| {
            architecture_error(
                ModelParallelKind::Single,
                Some(ModelStateKind::Cohere2Moe),
                arch_id,
                error,
            )
        })?,
        ModelState::Deepseek4(b) => {
            b.state.reset_with_gpu(gpu).map_err(|error| {
                architecture_error(
                    ModelParallelKind::Single,
                    Some(ModelStateKind::Deepseek4),
                    arch_id,
                    format!("reset DeepSeek MTP state: {error:?}"),
                )
            })?;
            b.state.zero_decode_caches_checked(gpu).map_err(|error| {
                architecture_error(
                    ModelParallelKind::Single,
                    Some(ModelStateKind::Deepseek4),
                    arch_id,
                    error.to_string(),
                )
            })?;
            gpu.invalidate_graph_state_checked().map_err(|error| {
                architecture_error(
                    ModelParallelKind::Single,
                    Some(ModelStateKind::Deepseek4),
                    arch_id,
                    format!("invalidate DeepSeek graphs: {error}"),
                )
            })?;
            gpu.hip.device_synchronize().map_err(|error| {
                architecture_error(
                    ModelParallelKind::Single,
                    Some(ModelStateKind::Deepseek4),
                    arch_id,
                    format!("synchronize DeepSeek graph invalidation: {error:?}"),
                )
            })?;
        }
        ModelState::DotsOcr(b) => b.state.reset(),
    }
    Ok(())
}

fn reset_ep_state(ep: &mut EpState, arch_id: u32) -> Result<(), ResetError> {
    let (arch_kind, state_len) = match &ep.inner {
        EpArch::Ds4 { state, .. } => (EpArchKind::Ds4, state.len()),
        EpArch::Minimax { state, .. } => (EpArchKind::Minimax, state.len()),
    };
    validate_ep_layout(arch_id, arch_kind, state_len, ep.gpus.devices.len())?;
    let owner_ranks = ep_owner_visitation(arch_id, state_len, ep.gpus.devices.len())?;

    match &mut ep.inner {
        EpArch::Ds4 { state, .. } => {
            reset_each_owner(&owner_ranks, |_index, rank| {
                let state = &mut state[rank];
                let gpu = &mut ep.gpus.devices[rank];
                gpu.bind_thread().map_err(|error| {
                    architecture_error(
                        ModelParallelKind::Ep,
                        None,
                        arch_id,
                        format!("bind EP device {rank}: {error:?}"),
                    )
                })?;
                state.reset_with_gpu(gpu).map_err(|error| {
                    architecture_error(
                        ModelParallelKind::Ep,
                        None,
                        arch_id,
                        format!("reset DeepSeek MTP state on rank {rank}: {error:?}"),
                    )
                })?;
                state.zero_decode_caches_checked(gpu).map_err(|error| {
                    architecture_error(ModelParallelKind::Ep, None, arch_id, error.to_string())
                })?;
                gpu.invalidate_graph_state_checked().map_err(|error| {
                    architecture_error(
                        ModelParallelKind::Ep,
                        None,
                        arch_id,
                        format!("invalidate DeepSeek graphs on rank {rank}: {error}"),
                    )
                })?;
                gpu.hip.device_synchronize().map_err(|error| {
                    architecture_error(
                        ModelParallelKind::Ep,
                        None,
                        arch_id,
                        format!(
                            "synchronize DeepSeek graph invalidation on rank {rank}: {error:?}"
                        ),
                    )
                })?;
                Ok(())
            })?;
        }
        EpArch::Minimax { state, .. } => {
            reset_each_owner(&owner_ranks, |_index, rank| {
                state[rank].reset();
                Ok(())
            })?;
        }
    }
    Ok(())
}

fn reset_qwen35_pipeline_recurrent(
    dn: &hipfire_arch_qwen35::qwen35::DeltaNetState,
    dn_la_to_device: &[u8],
    gpus: &mut Gpus,
    arch_id: u32,
) -> Result<(), ResetError> {
    let owner_devices = qwen35_pp_owner_visitation(
        dn_la_to_device,
        dn.s_matrices.len(),
        gpus.devices.len(),
        arch_id,
    )?;
    for buffers in qwen35_recurrent_groups(
        &dn.s_matrices,
        &dn.s_scales,
        &dn.conv_states,
        &dn.s_ef_residual,
    ) {
        if buffers.is_empty() {
            continue;
        }
        reset_each_owner(&owner_devices, |layer_index, device_id| {
            let s = &buffers[layer_index];
            let g = gpus.devices.get_mut(device_id).ok_or_else(|| {
                architecture_error(
                    ModelParallelKind::PpQwen35,
                    Some(ModelStateKind::Qwen35),
                    arch_id,
                    format!("PP device index {device_id} is out of range"),
                )
            })?;
            g.bind_thread().map_err(|error| {
                architecture_error(
                    ModelParallelKind::PpQwen35,
                    Some(ModelStateKind::Qwen35),
                    arch_id,
                    format!("bind PP device {device_id}: {error:?}"),
                )
            })?;
            match g.active_stream.as_ref() {
                Some(stream) => {
                    g.hip
                        .memset_async(&s.buf, 0, s.buf.size(), stream)
                        .map_err(|error| {
                            architecture_error(
                                ModelParallelKind::PpQwen35,
                                Some(ModelStateKind::Qwen35),
                                arch_id,
                                format!("reset PP recurrent state: {error:?}"),
                            )
                        })?;
                }
                None => {
                    g.hip.memset(&s.buf, 0, s.buf.size()).map_err(|error| {
                        architecture_error(
                            ModelParallelKind::PpQwen35,
                            Some(ModelStateKind::Qwen35),
                            arch_id,
                            format!("reset PP recurrent state: {error:?}"),
                        )
                    })?;
                }
            }
            Ok(())
        })?;
    }
    for (device_id, g) in gpus.devices.iter_mut().enumerate() {
        g.bind_thread().map_err(|error| {
            architecture_error(
                ModelParallelKind::PpQwen35,
                Some(ModelStateKind::Qwen35),
                arch_id,
                format!("bind PP device {device_id}: {error:?}"),
            )
        })?;
        g.hip.device_synchronize().map_err(|error| {
            architecture_error(
                ModelParallelKind::PpQwen35,
                Some(ModelStateKind::Qwen35),
                arch_id,
                format!("synchronize PP recurrent reset on device {device_id}: {error:?}"),
            )
        })?;
    }
    Ok(())
}

fn validate_reset_layout(
    arch_id: u32,
    pp: usize,
    pp_gpus: Option<&Gpus>,
    pp_dn_la_to_device: Option<&[u8]>,
    ep: Option<&EpState>,
    state: &Option<ModelState>,
) -> Result<(), ResetError> {
    if let Some(ep) = ep {
        let (arch_kind, state_len) = match &ep.inner {
            EpArch::Ds4 { state, .. } => (EpArchKind::Ds4, state.len()),
            EpArch::Minimax { state, .. } => (EpArchKind::Minimax, state.len()),
        };
        return validate_ep_layout(arch_id, arch_kind, state_len, ep.gpus.devices.len());
    }
    match state {
        Some(ModelState::Qwen35(bundle)) if pp > 1 => {
            let gpus = pp_gpus.ok_or_else(|| {
                architecture_error(
                    ModelParallelKind::PpQwen35,
                    Some(ModelStateKind::Qwen35),
                    arch_id,
                    "pp>1 without pp_gpus".to_string(),
                )
            })?;
            let dn_la_to_device = pp_dn_la_to_device.ok_or_else(|| {
                architecture_error(
                    ModelParallelKind::PpQwen35,
                    Some(ModelStateKind::Qwen35),
                    arch_id,
                    "pp>1 without pp_dn_la_to_device".to_string(),
                )
            })?;
            validate_qwen35_pipeline_layout(bundle, dn_la_to_device, gpus.devices.len(), arch_id)
        }
        Some(ModelState::Qwen35(_)) => {
            reject_qwen35_single_pipeline_metadata(pp_dn_la_to_device.is_some(), arch_id)
        }
        _ => Ok(()),
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
        /// The exact EP execution policy of this model, built ONCE from the
        /// same admitted mesh the `Gpus` are bound to (same mesh epoch). The
        /// daemon threads `&policy` into every `forward_ep` call — never
        /// reconstructed per token.
        policy: MoEExecutionPolicy,
    },
    Minimax {
        config: minimax::MiniMaxConfig,
        weights: Vec<minimax::MiniMaxWeights>,
        state: Vec<minimax::MiniMaxState>,
        partials: Vec<rdna_compute::GpuTensor>,
        /// Per-rank int64 scratch for the reproducible EP i64 down path.
        /// Each buffer holds `hidden_size * 8` raw bytes; pre-zeroed per step.
        partials_i64: Vec<rdna_compute::GpuTensor>,
        /// The exact EP execution policy of this model, built ONCE from the
        /// same admitted mesh the `Gpus` are bound to (same mesh epoch). The
        /// daemon threads `&policy` into every `forward_ep` call — never
        /// reconstructed per token.
        policy: MoEExecutionPolicy,
    },
}

// ─── Helper functions ─────────────────────────────────────────────────

/// Layer 1 (resolved config) + Layer 2 (per-model ~/.hipfire/templates).
fn resolve_chat_template_overrides(model_path: &str) -> Option<String> {
    if let Some(config_path) = hipfire_runtime::config::get().chat_template_file.as_deref() {
        if !config_path.is_empty() {
            match std::fs::read_to_string(config_path) {
                Ok(s) => {
                    return Some(s);
                }
                Err(e) => eprintln!(
                    "[chat_template] configured template {config_path} failed to read ({e}); falling through"
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

/// Hard-error free for unfinished qwen35 finish path: bundle + optional VL.
fn rollback_unfinished_qwen35(
    err: String,
    bundle: Qwen35Bundle,
    vision_weights: Option<qwen35_vl::VisionWeights>,
    gpu: &mut Gpu,
) -> String {
    let mut notes = Vec::new();
    if let Err(c) = hipfire_arch_qwen35::free_qwen35_bundle(bundle, gpu) {
        notes.push(c);
    }
    if let Some(vw) = vision_weights {
        vw.free_gpu(gpu);
    }
    if notes.is_empty() {
        err
    } else {
        format!("{err}; cleanup also failed: {}", notes.join("; "))
    }
}

/// CASK / plain eviction setup. Only hard-error stage in finish_qwen35_load
/// before the bundle is published into LoadedModel.
fn build_qwen35_eviction(
    config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
    physical_cap: usize,
    ctx: &mut LoadCtx,
) -> Result<Option<Eviction>, String> {
    use hipfire_arch_qwen35::qwen35::LayerType;
    let Some(ref sidecar_path) = ctx.cask.sidecar else {
        return Ok(None);
    };
    let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
        use std::io::ErrorKind;
        let p = Path::new(sidecar_path);
        let why = match e.kind() {
            ErrorKind::NotFound if p.symlink_metadata().is_ok() => {
                format!("dangling symlink (target absent): {sidecar_path}")
            }
            ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
            ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
            ErrorKind::UnexpectedEof => format!("truncated/corrupt sidecar: {sidecar_path}"),
            _ => format!("read error ({e}): {sidecar_path}"),
        };
        format!(
            "cask sidecar load failed — {why} (regen: hipfire sidecar-gen, or HIPFIRE_CASK_OFF=1)"
        )
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
        return Ok(None);
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
    if ctx.cask.cask_m_folding {
        eprintln!(
            "  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
            ctx.cask.core_frac, ctx.cask.fold_m, ctx.cask.budget, ctx.cask.beta, physical_cap
        );
        Ok(Some(Eviction::Cask(CaskCtx::new(
            base,
            ctx.cask.core_frac,
            ctx.cask.fold_m,
        ))))
    } else {
        eprintln!(
            "  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
            ctx.cask.budget, ctx.cask.beta, physical_cap
        );
        Ok(Some(Eviction::Plain(base)))
    }
}

/// Build a `LoadedModel` from a carrier `Bundle`, shared fields, and
/// eviction/DFlash state. This is the common body for qwen35 dispatch
/// where eviction and DFlash need per-arch type info.
///
/// Hard errors before publish free the bundle and any preloaded VL weights.
fn finish_qwen35_load(
    bundle: Qwen35Bundle,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    physical_cap: usize,
    arch_id: u32,
    chat_template: Option<String>,
    ctx: &mut LoadCtx,
    vision_config: Option<qwen35_vl::VisionConfig>,
    mut vision_weights: Option<qwen35_vl::VisionWeights>,
) -> Result<LoadedModel, String> {
    // ── Eviction (only hard-error stage before publish) ────────────
    // Built before long-lived borrows so rollback can move `bundle`.
    let eviction = match build_qwen35_eviction(&bundle.config, physical_cap, ctx) {
        Ok(e) => e,
        Err(e) => {
            return Err(rollback_unfinished_qwen35(
                e,
                bundle,
                vision_weights,
                ctx.gpu,
            ));
        }
    };

    // Extract references for DFlash/spec setup (borrow, don't move)
    let config = &bundle.config;
    let dn_state = &bundle.dn_state;

    // Adaptive KV cannot combine with generic (DSpark/DFlash/n-gram/bundled-MTP
    // build_speculator) drafters. Suppress before any GPU drafter alloc; native
    // qwen35_mtp_head load below is intentionally left alone.
    let adaptive_blocks_generic_spec = bundle.kv_adaptive.is_some();
    if adaptive_blocks_generic_spec {
        eprintln!(
            "  kv_adaptive engaged — suppressing generic speculator path (DSpark/DFlash/n-gram/bundled MTP-spec)"
        );
    }

    // ── DSpark sidecar (wins over DFlash/MTP/n-gram) ───────────────
    // The drafter is a dense-qwen3 body (llama crate); it drives the qwen35
    // ModelSlot target via the SpecTarget DSpark capture hooks. Discovered as
    // `<stem>-dspark.<ext>` next to the trunk, independent of ctx.draft_path.
    let dspark_speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> =
        if adaptive_blocks_generic_spec {
            None
        } else if ctx.spec.dspark != Some(false) {
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
                                    let conf_threshold = hipfire_config::developer_var(
                                        "HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD",
                                    )
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
                                            dspark_weights.free_gpu(ctx.gpu);
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
    let dflash = if adaptive_blocks_generic_spec || dspark_speculator.is_some() {
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
            eviction.is_some(),
        ) {
            Ok(s) => {
                eprintln!(
                    "  DFlash draft loaded: {} (layers={}, hidden={}, block={})",
                    dp, s.draft_config.n_layers, s.draft_config.hidden, s.draft_config.block_size
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
    // Pick the arch-generic speculator: a loaded DFlash draft → DflashSpeculator,
    // else (opt-in) the model-free n-gram drafter. `eviction` is borrowed (not
    // moved) here, so it is still available for the struct literal below;
    // `config`/`dn_state` are borrowed only for the n-gram arm's scratch
    // construction (snapshot copied to GPU), released before `bundle` moves into
    // `state`. `None` ⇒ AR-only model. DSpark wins over DFlash/n-gram when its
    // sidecar loaded. The native MTP head is NOT a speculator here — it is
    // published on `model.qwen35_mtp_head` (block below) and driven by the
    // daemon's MtpSpeculator. When adaptive, upstream gates left dspark/dflash
    // as None — no free needed.
    let speculator = if adaptive_blocks_generic_spec {
        None
    } else {
        dspark_speculator.or_else(|| {
            crate::spec_build::build_speculator(
                arch_id,
                dflash,
                eviction.is_none(),
                physical_cap,
                ctx.spec,
            )
        })
    };

    // ── Qwen3.5/3.6 native MTP (NextN) head ────────────────────────
    //
    // Load the arch_id=21 MTP head when it is present either bundled in the
    // trunk file (a `.mq4-mtp` trailer, magic HFBNDMTP) or as a sibling `.mtp`
    // sidecar (`<trunk>.mtp` next to the model path). The head is OPTIONAL:
    // `Ok(None)` / a missing sidecar just leaves MTP serving unavailable and
    // the model serves via the unchanged DFlash/AR path. Failures here are
    // non-fatal — log and continue with `qwen35_mtp_head = None`.
    //
    // max_seq mirrors the trunk's KV capacity (the MTP head's KV is a single
    // F32 layer, so even a 100K window is only a few hundred MB at dim=5120).
    let qwen35_mtp_head: Option<hipfire_arch_qwen35::mtp_head::Qwen35MtpHead> = {
        use hipfire_arch_qwen35::mtp_head;
        let trunk_path = Path::new(ctx.path);
        // 1. Bundled trailer inside the trunk file?
        let bundled = match mtp_head::load_mtp_head_bundled(trunk_path, ctx.gpu, physical_cap) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("  MTP head (bundled) load failed: {e} — MTP serving disabled");
                None
            }
        };
        match bundled {
            Some(h) => {
                eprintln!(
                    "  MTP head loaded (bundled .mq4-mtp): n_embd={} vocab={} K-default=3",
                    h.config.n_embd, h.config.vocab_size
                );
                Some(h)
            }
            None => {
                // 2. Sidecar `<trunk>.mtp` next to the model path?
                let sidecar = trunk_path.with_extension("mtp");
                if sidecar.exists() {
                    match mtp_head::load_mtp_head(&sidecar, ctx.gpu, physical_cap) {
                        Ok(h) => {
                            eprintln!(
                                "  MTP head loaded (sidecar {}): n_embd={} vocab={} K-default=3",
                                sidecar.display(),
                                h.config.n_embd,
                                h.config.vocab_size
                            );
                            Some(h)
                        }
                        Err(e) => {
                            eprintln!(
                                "  MTP head (sidecar {}) load failed: {e} — MTP serving disabled",
                                sidecar.display()
                            );
                            None
                        }
                    }
                } else {
                    None
                }
            }
        }
    };

    // Move adaptive controller out of the bundle before parking the rest in
    // ModelState. LoadedModel.kv_adaptive is the runtime home for downshift hooks.
    let mut bundle = bundle;
    let kv_adaptive = bundle.kv_adaptive.take();
    let state = Some(ModelState::Qwen35(bundle));
    let mut model = LoadedModel {
        state,
        eviction,
        speculator,
        vision_config,
        vision_weights,
        max_seq: ctx.max_seq,
        kv_adaptive,
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            ctx.max_seq,
            physical_cap,
            ctx.path.to_string(),
            chat_template,
        )
    };
    // `mtp_weights_present` drives the mtp_mode=auto serve decision (mirrors the
    // dspark probe set in the daemon's load handler). For qwen35 the presence of a
    // loaded MTP head IS the signal.
    model.mtp_weights_present = qwen35_mtp_head.is_some();
    model.qwen35_mtp_head = qwen35_mtp_head;
    Ok(model)
}

// ─── CAP-001 Task 3: host-only admission guard (loader boundary) ──────
//
// `admit_path` opens a model source and classifies its parallelism
// capability before any GPU/tokenizer/allocation work.  The returned
// `AdmittedLoad` token is consumed by `load_admitted`, which routes to
// the appropriate private axis-specific constructor.
//
// This is a **conventional high-level loader boundary**: external
// callers interact only with `admit_path` and `load_admitted`.  The
// raw TP/PP constructors are gated behind the `loader-internal`
// feature and `#[doc(hidden)]` — internal/unsupported by convention.
// Cargo feature gating / `#[doc(hidden)]` is not access control.

/// Admission result that retains the opened model path, source, and
/// resolved parallelism.  Public but **opaque**: no public access to
/// the underlying path, source, or carrier — only the resolved
/// admission token ([`Self::admission`]).
///
/// The source is consumed by [`load_admitted`] via the crate-private
/// [`Self::into_parts`].  Callers call [`admit_path`] for the token,
/// then pass it to [`load_admitted`] with a matching mesh.
pub struct AdmittedLoad {
    /// Original model path (file or directory).  Used by the private
    /// load helpers to populate `model_path` in [`LoadedModel`] metadata
    /// and for sidecar discovery.
    path: String,
    source: ModelSource,
    carrier: &'static dyn Carrier,
    admission: ParallelAdmission,
}

impl std::fmt::Debug for AdmittedLoad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmittedLoad")
            .field("admission", &self.admission)
            .finish_non_exhaustive()
    }
}

impl AdmittedLoad {
    /// The resolved parallelism admission — variant, effective degrees,
    /// and whether normalisation was applied.  This is the only public
    /// accessor; the path, source, and carrier remain private.
    pub fn admission(&self) -> ParallelAdmission {
        self.admission
    }

    fn record(&self) -> AdmissionRecord {
        AdmissionRecord::from_admission(self.admission)
    }

    /// The model path that was admitted.  Crate-private — used by
    /// `load_admitted` dispatch.
    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    /// Borrow the opened model source (HFQ or safetensors directory).
    pub(crate) fn source_ref(&self) -> &ModelSource {
        &self.source
    }

    /// Consume the `AdmittedLoad`, returning its owned parts.
    pub(crate) fn into_parts(
        self,
    ) -> (String, ModelSource, &'static dyn Carrier, ParallelAdmission) {
        (self.path, self.source, self.carrier, self.admission)
    }
}

/// Open a model source, classify its architecture, and resolve the
/// parallelism request.  Returns a source-owning [`AdmittedLoad`]
/// token.  No GPU / tokenizer / allocation work is performed.
///
/// Errors are prefixed with `[CAP-001]` and include the exact path.
pub fn admit_path(
    path: impl AsRef<std::path::Path>,
    request: RawParallelRequest,
) -> Result<AdmittedLoad, String> {
    let path = path.as_ref();
    // Reject non-UTF-8 paths at the boundary — a lossy replacement would
    // obscure the real filesystem path in diagnostics.  The canonical
    // loader path (HFQ + safetensors) is always valid UTF-8 on Linux.
    let path_str = path
        .to_str()
        .ok_or_else(|| format!("[CAP-001] non-UTF-8 model path: {}", path.display()))?;
    let src = ModelSource::from_path(path_str)
        .map_err(|e| format!("[CAP-001] failed to open model source '{path_str}': {e}"))?;
    let (carrier, variant) = classify_source(&src)
        .map_err(|e| format!("[CAP-001] failed to classify '{path_str}': {e}"))?;
    let admission = resolve(variant, request)
        .map_err(|e| format!("[CAP-001] admission refused '{path_str}': {e}"))?;
    Ok(AdmittedLoad {
        path: path_str.to_string(),
        source: src,
        carrier,
        admission,
    })
}

/// Internal admission of an already-opened [`ModelSource`] with a known
/// path.  Like [`admit_path`] but skips the open step.  Used only in
/// tests (the production wrappers use [`admit_path`] directly).
#[cfg(test)]
fn admit_source(
    path: &str,
    src: ModelSource,
    request: RawParallelRequest,
) -> Result<AdmittedLoad, String> {
    let (carrier, variant) = classify_source(&src)?;
    let admission = resolve(variant, request).map_err(|e| e.to_string())?;
    Ok(AdmittedLoad {
        path: path.to_string(),
        source: src,
        carrier,
        admission,
    })
}

/// Derive a [`RawParallelRequest`] from a [`DeviceMesh`] — the three
/// parallelism degrees (pp, tp, ep) that the mesh requests.
fn raw_request(mesh: &DeviceMesh) -> RawParallelRequest {
    RawParallelRequest::new(
        mesh.size_of(DimKind::Pp),
        mesh.size_of(DimKind::Tp),
        mesh.size_of(DimKind::Ep),
    )
}

fn effective_load_mesh(admitted: &AdmittedLoad) -> DeviceMesh {
    let effective = admitted.admission().effective();
    DeviceMesh::rect(&[
        (DimKind::Pp, effective.pp),
        (DimKind::Tp, effective.tp),
        (DimKind::Ep, effective.ep),
    ])
}

fn select_load_mesh<'a>(
    admitted: &AdmittedLoad,
    mesh: &'a DeviceMesh,
) -> std::borrow::Cow<'a, DeviceMesh> {
    if admitted.admission().was_normalized() {
        std::borrow::Cow::Owned(effective_load_mesh(admitted))
    } else {
        std::borrow::Cow::Borrowed(mesh)
    }
}

// ─── ModelLoadOptions ──────────────────────────────────────────────────

/// Aggregated load options passed to [`load_admitted`] alongside the
/// admitted token and mesh.
///
/// Construct via [`ModelLoadOptions::new`] and optional setters.
/// This struct is `#[non_exhaustive]` — new fields may be added in
/// minor releases without a breaking change.
#[derive(Clone)]
#[non_exhaustive]
pub struct ModelLoadOptions<'a> {
    /// Maximum sequence length for KV cache allocation.  Required.
    max_seq: usize,
    /// Optional DFlash drafter path.  Default: `None`.
    draft_path: Option<&'a str>,
    /// Override KV cache quantization mode.  Default: `None` (auto).
    kv_mode_override: Option<&'a str>,
    /// Override adaptive KV mode.  Default: `None` (auto).
    kv_adaptive_override: Option<&'a str>,
    /// Override DeltaNet state quantization.  Default: `None` (auto).
    state_quant_override: Option<&'a str>,
    /// CASK eviction config.  Default: a no-eviction config.
    cask: &'a CaskConfig,
    /// PP layer bands (Qwen35 PP).  Default: `None` (uniform split).
    pp_bands: Option<&'a [usize]>,
    /// Speculative decoding config.  Default: `SpecLoadCfg::default()`.
    spec: SpecLoadCfg,
    /// KV allocation backend.  Default: `Contiguous` (beta daemon compat).
    kv_backend: hipfire_runtime::kv_backend::KvBackend,
}

impl<'a> ModelLoadOptions<'a> {
    /// Create options with the required `max_seq`.  All optional fields
    /// use their defaults:
    /// - `draft_path`: `None`
    /// - `kv_mode_override`: `None`
    /// - `kv_adaptive_override`: `None`
    /// - `state_quant_override`: `None`
    /// - `cask`: a no-eviction config (`budget=0`, `beta=0`)
    /// - `pp_bands`: `None`
    /// - `spec`: `SpecLoadCfg::default()`
    pub fn new(max_seq: usize) -> Self {
        Self {
            max_seq,
            draft_path: None,
            kv_mode_override: None,
            kv_adaptive_override: None,
            state_quant_override: None,
            cask: &DEFAULT_CASK,
            pp_bands: None,
            spec: SpecLoadCfg::default(),
            kv_backend: hipfire_runtime::kv_backend::KvBackend::default(),
        }
    }

    // ── Accessors for internal use ──
    pub(crate) fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub(crate) fn draft_path(&self) -> Option<&'a str> {
        self.draft_path
    }
    pub(crate) fn kv_mode_override(&self) -> Option<&'a str> {
        self.kv_mode_override
    }
    pub(crate) fn kv_adaptive_override(&self) -> Option<&'a str> {
        self.kv_adaptive_override
    }
    pub(crate) fn state_quant_override(&self) -> Option<&'a str> {
        self.state_quant_override
    }
    pub(crate) fn cask(&self) -> &'a CaskConfig {
        self.cask
    }
    pub(crate) fn pp_bands(&self) -> Option<&'a [usize]> {
        self.pp_bands
    }
    pub(crate) fn spec(&self) -> SpecLoadCfg {
        self.spec
    }

    // ── Fluent setters ──
    /// Set the DFlash drafter path.
    pub fn with_draft_path(mut self, draft_path: &'a str) -> Self {
        self.draft_path = Some(draft_path);
        self
    }
    /// Override KV cache quantization mode.
    pub fn with_kv_mode_override(mut self, mode: &'a str) -> Self {
        self.kv_mode_override = Some(mode);
        self
    }
    /// Override adaptive KV mode.
    pub fn with_kv_adaptive_override(mut self, mode: &'a str) -> Self {
        self.kv_adaptive_override = Some(mode);
        self
    }
    /// Override DeltaNet state quantization.
    pub fn with_state_quant_override(mut self, quant: &'a str) -> Self {
        self.state_quant_override = Some(quant);
        self
    }
    /// Set CASK eviction config.
    pub fn with_cask(mut self, cask: &'a CaskConfig) -> Self {
        self.cask = cask;
        self
    }
    /// Set PP layer bands.
    pub fn with_pp_bands(mut self, bands: &'a [usize]) -> Self {
        self.pp_bands = Some(bands);
        self
    }
    /// Set the KV allocation backend (beta daemon compat; VMM opt-in).
    pub fn with_kv_backend(mut self, kv_backend: hipfire_runtime::kv_backend::KvBackend) -> Self {
        self.kv_backend = kv_backend;
        self
    }

    pub fn with_kv_mode_override_opt(mut self, mode: Option<&'a str>) -> Self {
        self.kv_mode_override = mode;
        self
    }

    pub fn with_kv_adaptive_override_opt(mut self, mode: Option<&'a str>) -> Self {
        self.kv_adaptive_override = mode;
        self
    }

    pub fn with_state_quant_override_opt(mut self, quant: Option<&'a str>) -> Self {
        self.state_quant_override = quant;
        self
    }

    pub fn with_spec(mut self, spec: SpecLoadCfg) -> Self {
        self.spec = spec;
        self
    }
}

// Static default for wrappers that don't need eviction config.
static DEFAULT_CASK: CaskConfig = CaskConfig {
    sidecar: None,
    cask_m_folding: false,
    budget: 0,
    beta: 0,
    core_frac: 0.0,
    fold_m: 0,
};

// ─── Single public consumer ────────────────────────────────────────────

/// Load a model from an already-admitted source token.  This is the
/// single public entry point for all parallelism axes.
///
/// `gpu` is required for Single-axis loads and unused for TP/PP/EP
/// (those axes manage their own GPU mesh via `Gpus::from_mesh`).
///
/// # Errors
///
/// - Topology mismatch: `raw_request(mesh) != admitted.admission().effective()`
/// - Single axis with `None` GPU: `[CAP-001] Single load requires a GPU`
/// - Axis routing: the effective axis is dispatched to the appropriate
///   private `load_*_admitted` helper.
pub fn load_admitted(
    admitted: AdmittedLoad,
    mesh: &DeviceMesh,
    opts: ModelLoadOptions,
    gpu: Option<&mut rdna_compute::Gpu>,
) -> Result<LoadedModel, String> {
    let raw = raw_request(mesh);
    let effective = admitted.admission().effective();

    // Topology equality: the mesh must match the admission's effective degrees.
    if effective != raw {
        return Err(format!(
            "[CAP-001] load_admitted: mesh request ({:?}) does not match \
             admission effective ({:?})",
            raw, effective,
        ));
    }

    let axis = effective.axis();

    // CAP-001 Task 5: record dispatch decision at the load_admitted boundary
    // (cfg(test) only — zero cost in production).
    #[cfg(test)]
    crate::construction_recorder::record(crate::construction_recorder::RecordedEvent::RoutedTo {
        axis,
        variant: admitted.admission().variant(),
        effective,
    });

    // Route by effective axis.
    match axis {
        ParallelAxis::Single => load_single_admitted(admitted, mesh, opts, gpu),
        ParallelAxis::Tp => load_tp_admitted(admitted, mesh, opts),
        ParallelAxis::Pp => load_pp_admitted(admitted, mesh, opts),
        ParallelAxis::Ep => load_ep_admitted(admitted, mesh, opts),
    }
}

// ─── Private axis-specific consumers ───────────────────────────────────

/// Single-GPU load from an admitted source.  Every existing argument in
/// `load_model` is threaded through [`ModelLoadOptions`].
fn load_single_admitted(
    admitted: AdmittedLoad,
    mesh: &DeviceMesh,
    opts: ModelLoadOptions,
    gpu: Option<&mut rdna_compute::Gpu>,
) -> Result<LoadedModel, String> {
    #[cfg(test)]
    crate::construction_recorder::record(
        crate::construction_recorder::RecordedEvent::EnteredConstructor {
            axis: crate::parallel_capability::ParallelAxis::Single,
        },
    );
    let gpu = gpu.ok_or_else(|| "[CAP-001] Single load requires a GPU (got None)".to_string())?;
    let (path, source, carrier, _admission) = admitted.into_parts();

    let source_arch_id = match &source {
        ModelSource::Hfq(hfq) => hfq.arch_id,
        ModelSource::Dir(source) => source.arch_id(),
    };

    // Author-recommended sampling defaults (temp/top_p/top_k from the .hfq's baked
    // `generation_config`). Extract HERE, from the already-open source, BEFORE the
    // carrier allocates any GPU buffers.
    let rec_sampling = match &source {
        ModelSource::Hfq(hfq) => hfq.recommended_sampling(),
        _ => None,
    };

    // DFlash lm_head quant check — only for HFQ sources
    if let ModelSource::Hfq(hfq) = &source {
        preflight_dflash_draft(hfq, opts.draft_path(), gpu)?;
    }

    let mtp_mode = mtp_mode_from_spec(opts.spec());
    let mut ctx = LoadCtx {
        path: &path,
        max_seq: opts.max_seq(),
        draft_path: opts.draft_path(),
        kv_mode_override: opts.kv_mode_override(),
        kv_adaptive_override: opts.kv_adaptive_override(),
        state_quant_override: opts.state_quant_override(),
        cask: opts.cask(),
        kv_backend: opts.kv_backend,
        pp: mesh.size_of(DimKind::Pp),
        pp_bands: opts.pp_bands(),
        mtp_mode,
        mtp_k: normalize_mtp_k(source_arch_id, opts.spec().mtp_k)?,
        spec: opts.spec(),
        kv_physical_cap: None,
        gpu,
    };

    let mut result = carrier.load(source, &mut ctx)?;
    if let Some(rec) = rec_sampling {
        result.rec_temperature = rec.temperature;
        result.rec_top_p = rec.top_p;
        result.rec_top_k = rec.top_k.map(|k| k as f32);
    }
    Ok(result)
}

/// TP load from an admitted source.  Topology/axis enforcement is in
/// [`load_admitted`] — this helper assumes a valid TP admission.
fn load_tp_admitted(
    admitted: AdmittedLoad,
    mesh: &DeviceMesh,
    opts: ModelLoadOptions,
) -> Result<LoadedModel, String> {
    #[cfg(test)]
    crate::construction_recorder::record(
        crate::construction_recorder::RecordedEvent::EnteredConstructor {
            axis: crate::parallel_capability::ParallelAxis::Tp,
        },
    );
    let path = admitted.path();
    let hfq_ref = match admitted.source_ref() {
        ModelSource::Hfq(hfq) => hfq,
        _ => return Err("load_model_tp: TP load requires HFQ source".into()),
    };
    let arch_id = hfq_ref.arch_id;
    let mtp_mode = mtp_mode_from_spec(opts.spec());
    let mtp_k = normalize_mtp_k(arch_id, opts.spec().mtp_k)?;
    #[cfg(feature = "arch-qwen35")]
    if matches!(arch_id, 5 | 6) {
        let config =
            <hipfire_arch_qwen35::Qwen35 as hipfire_runtime::arch::Architecture>::config_from_hfq(
                hfq_ref,
            )?;
        hipfire_arch_qwen35::arch::qwen35_tp_preflight(&config, mesh.size_of(DimKind::Tp))?;
    }
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_ref.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
    let chat_template = resolve_chat_template(hfq_ref, path);
    let rec = hfq_ref.recommended_sampling();
    let tp_model =
        hipfire_runtime::tp_serve::TpModel::load_from_hfq(hfq_ref, mesh, opts.max_seq())?;

    Ok(LoadedModel {
        tp_model: Some(tp_model),
        mtp_mode: mtp_mode.to_string(),
        mtp_k,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            opts.max_seq(),
            opts.max_seq(),
            path.to_string(),
            chat_template,
        )
    })
}

/// PP load from an admitted source.  Topology/axis enforcement is in
/// [`load_admitted`].
fn load_pp_admitted(
    admitted: AdmittedLoad,
    mesh: &DeviceMesh,
    opts: ModelLoadOptions,
) -> Result<LoadedModel, String> {
    #[cfg(test)]
    crate::construction_recorder::record(
        crate::construction_recorder::RecordedEvent::EnteredConstructor {
            axis: crate::parallel_capability::ParallelAxis::Pp,
        },
    );
    let path = admitted.path();
    let hfq_ref = match admitted.source_ref() {
        ModelSource::Hfq(hfq) => hfq,
        _ => return Err("load_model_pp: PP load requires HFQ source".into()),
    };
    let arch_id = hfq_ref.arch_id;
    let mtp_mode = mtp_mode_from_spec(opts.spec());
    let mtp_k = normalize_mtp_k(arch_id, opts.spec().mtp_k)?;
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_ref.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
    let chat_template = resolve_chat_template(hfq_ref, path);
    let rec = hfq_ref.recommended_sampling();
    let pp_model =
        hipfire_runtime::pp_serve::PpModel::load_from_hfq(hfq_ref, mesh, opts.max_seq())?;

    Ok(LoadedModel {
        pp_model: Some(pp_model),
        mtp_mode: mtp_mode.to_string(),
        mtp_k,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            opts.max_seq(),
            opts.max_seq(),
            path.to_string(),
            chat_template,
        )
    })
}

/// EP load from an admitted source.  Consumes the source so every rank
/// weight load uses the SAME HfqFile.  Topology/axis enforcement is in
/// [`load_admitted`].
fn load_ep_admitted(
    admitted: AdmittedLoad,
    mesh: &DeviceMesh,
    opts: ModelLoadOptions,
) -> Result<LoadedModel, String> {
    #[cfg(test)]
    crate::construction_recorder::record(
        crate::construction_recorder::RecordedEvent::EnteredConstructor {
            axis: crate::parallel_capability::ParallelAxis::Ep,
        },
    );
    let mtp_mode = mtp_mode_from_spec(opts.spec());
    let mtp_k = normalize_mtp_k(0, opts.spec().mtp_k)?;
    let variant = admitted.record().variant();
    let (path, source, _carrier, _) = admitted.into_parts();
    let mut hfq = match source {
        ModelSource::Hfq(hfq) => hfq,
        _ => return Err("load_model_ep: EP load requires HFQ source".into()),
    };
    let model = match variant {
        ModelVariant::Deepseek4 => {
            load_model_ep_ds4(&mut hfq, &path, opts.max_seq(), mesh, mtp_mode, mtp_k)
        }
        ModelVariant::Minimax => {
            load_model_ep_minimax(&mut hfq, &path, opts.max_seq(), mesh, mtp_mode, mtp_k)
        }
        v => Err(format!(
            "EP not supported for {v:?} (expected Deepseek4 or Minimax)"
        )),
    }?;
    Ok(model)
}

/// Host-only admission routing seam: the generic `load_model` entry point
/// accepts only Single-axis loads (no TP/PP/EP parallelism).  Checks the
/// effective axis and returns the [`AdmittedLoad`] on success.  Every
/// continuation inside `load_model` is preceded by this seam — no GPU or
/// tokenizer work happens before it passes.
pub(crate) fn route_generic_single(admitted: AdmittedLoad) -> Result<AdmittedLoad, String> {
    if admitted.admission().effective().axis() != ParallelAxis::Single {
        return Err(format!(
            "load_model: effective axis is {:?}, expected Single; use \
             load_model_tp (TP), load_model_pp (PP), or load_model_ep (EP)",
            admitted.admission().effective().axis(),
        ));
    }
    Ok(admitted)
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

/// Load a model from an HFQ file (or safetensors directory).  Self-admits
/// via [`admit_path`] then delegates to [`load_admitted`].
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
    let raw = raw_request(mesh);
    let admitted = route_generic_single(admit_path(path, raw)?)?;
    let load_mesh = select_load_mesh(&admitted, mesh);
    let mut opts = ModelLoadOptions::new(max_seq)
        .with_cask(cask)
        .with_spec(spec);
    if let Some(dp) = draft_path {
        opts = opts.with_draft_path(dp);
    }
    if let Some(kv) = kv_mode_override {
        opts = opts.with_kv_mode_override(kv);
    }
    if let Some(kv) = kv_adaptive_override {
        opts = opts.with_kv_adaptive_override(kv);
    }
    if let Some(sq) = state_quant_override {
        opts = opts.with_state_quant_override(sq);
    }
    if let Some(bands) = pp_bands {
        opts = opts.with_pp_bands(bands);
    }
    load_admitted(admitted, load_mesh.as_ref(), opts, Some(gpu))
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
        mtp_mode: mtp_mode.to_string(),
        mtp_k,
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
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
    match hipfire_config::developer_var("HIPFIRE_EP_FAIL_RANK").ok() {
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
/// Expert-parallel (EP) load, beta-compatible 3-arg entry: shards the routed
/// experts across `tp` ranks. Wraps the device-mesh admission path with a mesh
/// built from the TP degree.
pub fn load_model_ep(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, tp.max(1))]);
    load_model_ep_with_mesh(path, max_seq, &mesh, hipfire_runtime::loader_api::SpecLoadCfg::default())
}

/// EP load with an explicit mesh + spec (device-mesh admission path).
pub fn load_model_ep_with_mesh(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    spec: SpecLoadCfg,
) -> Result<LoadedModel, String> {
    let admitted = admit_path(path, raw_request(mesh))?;
    let opts = ModelLoadOptions::new(max_seq).with_spec(spec);
    load_admitted(admitted, mesh, opts, None)
}

/// DFlash draft preflight shared by the single and legacy pp>1 registry
/// loads: lm_head quant compatibility + MQ2/MQ3 refusal. Beta ran these
/// before `carrier.load`; the legacy pp>1 path must too, or an opted-in
/// PP draft (`HIPFIRE_PP_DFLASH=1`) can reach the documented per-row GEMV
/// verify hang.
fn preflight_dflash_draft(
    hfq: &HfqFile,
    draft_path: Option<&str>,
    gpu: &rdna_compute::Gpu,
) -> Result<(), String> {
    if draft_path.is_none() {
        return Ok(());
    }
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
    Ok(())
}

/// Beta-compat pp>1 dispatch: qwen35 hand-coded pipeline-parallel through the
/// carrier registry (`load_qwen35_pp` → `skeleton_pp`/`pp_gpus`, served by the
/// daemon's `generate_multi`). The device-mesh policy table marks qwen35 Pp
/// "Planned" (GEN-001) and `load_admitted` cannot recreate beta's carrier, so
/// pp>1 loads keep beta's carrier-registry dispatch instead of admission.
/// Non-qwen35 carriers keep their own pp behavior (LlamaCarrier refuses pp>1;
/// the daemon routes dense llama pp>1 through admission to the PpModel carrier
/// instead).
fn load_pp_via_carrier_registry(
    path: &str,
    max_seq: usize,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    kv_backend: hipfire_runtime::kv_backend::KvBackend,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    let src = ModelSource::from_path(path)?;
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
    if let ModelSource::Hfq(hfq) = &src {
        preflight_dflash_draft(hfq, draft_path, gpu)?;
    }
    if kv_backend == hipfire_runtime::kv_backend::KvBackend::Vmm && carrier.name() != "qwen35" {
        return Err(format!(
            "KV backend 'vmm' currently supports qwen3.5 only (selected carrier: {})",
            carrier.name()
        ));
    }
    let rec_sampling = match &src {
        ModelSource::Hfq(hfq) => hfq.recommended_sampling(),
        _ => None,
    };
    let arch_id = src
        .arch_id()
        .ok_or_else(|| format!("no arch_id in source: {}", src.describe()))?;
    let mut ctx = LoadCtx {
        path,
        max_seq,
        draft_path,
        kv_mode_override,
        kv_adaptive_override,
        state_quant_override,
        cask,
        kv_backend,
        pp,
        pp_bands: None,
        mtp_mode: mtp_mode_from_spec(spec),
        mtp_k: normalize_mtp_k(arch_id, spec.mtp_k)?,
        spec,
        kv_physical_cap: None,
        gpu,
    };
    let mut result = carrier.load(src, &mut ctx)?;
    if result.pp > 1 && result.pp_gpus.is_none() {
        return Err("pp>1 LoadedModel missing pp_gpus — carrier bug".into());
    }
    if let Some(rec) = rec_sampling {
        result.rec_temperature = rec.temperature;
        result.rec_top_p = rec.top_p;
        result.rec_top_k = rec.top_k.map(|k| k as f32);
    }
    Ok(result)
}

/// Single-GPU load with the full override surface (beta daemon entry point).
/// Routes through the device-mesh admission path; `pp` maps to the PP degree.
pub fn load_model_with_kv_backend(
    path: &str,
    max_seq: usize,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    kv_backend_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    // Retry any arenas left by a prior failed teardown; refuse the load if
    // ownership is still live so a new model cannot stack on pending VMM state.
    ensure_vmm_ready_for_load(gpu)?;
    let kv_backend: hipfire_runtime::kv_backend::KvBackend = kv_backend_override
        .unwrap_or("contiguous")
        .parse()
        .map_err(|err| format!("kv_backend parse: {err}"))?;
    // pp>1 keeps beta's carrier-registry dispatch (qwen35 hand-coded PP —
    // load_qwen35_pp/pp_gpus, served by generate_multi); admission marks
    // qwen35 Pp "Planned" (GEN-001) and cannot recreate beta's carrier.
    // pp<=1 routes through the device-mesh admission path.
    if pp > 1 {
        return load_pp_via_carrier_registry(
            path,
            max_seq,
            draft_path,
            kv_mode_override,
            kv_backend,
            kv_adaptive_override,
            state_quant_override,
            cask,
            pp,
            spec,
            gpu,
        );
    }
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, pp.max(1))]);
    let mut opts = ModelLoadOptions::new(max_seq)
        .with_kv_backend(kv_backend)
        .with_kv_mode_override_opt(kv_mode_override)
        .with_kv_adaptive_override_opt(kv_adaptive_override)
        .with_state_quant_override_opt(state_quant_override)
        .with_cask(cask)
        .with_spec(spec);
    if let Some(draft) = draft_path {
        opts = opts.with_draft_path(draft);
    }
    let admitted = admit_path(path, RawParallelRequest::new(pp, 1, 1))?;
    load_admitted(admitted, &mesh, opts, Some(gpu))
}

/// Load a model for **real tensor-parallel (TP)** serving — dense row/col
/// sharding across `tp` GPUs (the `Tp` axis), distinct from expert-parallel
/// ([`load_model_ep`], the `Ep` axis). This is the daemon entry the EP↔TP
/// disentanglement reserves for the dense TP serve path.
pub fn load_model_tp(
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    spec: SpecLoadCfg,
) -> Result<LoadedModel, String> {
    let admitted = admit_path(path, raw_request(mesh))?;
    let opts = ModelLoadOptions::new(max_seq).with_spec(spec);
    load_admitted(admitted, mesh, opts, None)
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
    let admitted = admit_path(path, raw_request(mesh))?;
    let opts = ModelLoadOptions::new(max_seq).with_spec(spec);
    load_admitted(admitted, mesh, opts, None)
}

fn load_model_ep_ds4(
    hfq: &mut HfqFile,
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    mtp_mode: &str,
    mtp_k: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(hfq)?;
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
    let chat_template = resolve_chat_template(hfq, path);
    let rec = hfq.recommended_sampling();

    let ep = mesh.size_of(DimKind::Ep);
    // The exact EP execution policy of this model, built ONCE from the SAME
    // admitted mesh the `Gpus` below are bound to — the policy's mesh is a
    // clone of `mesh`, so it carries the exact same mesh epoch (the sealed
    // executor's mesh-identity check compares `policy.mesh()` against the
    // Gpus-bound epoch). The daemon threads `&policy` into every
    // `forward_ep` call; nothing reconstructs mesh/policy per token.
    let policy = MoEExecutionPolicy::new(MoEExecutionKind::Ep, mesh.clone())
        .map_err(|e| format!("ds4 EP policy from admitted mesh: {e:?}"))?;
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
    // Weight-load loop (admitted source, not reopened path).
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = deepseek4::DeepseekV4::load_weights_sharded(hfq, &config, dev, &shard, r)
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
        ep: Some(EpState {
            gpus,
            inner: EpArch::Ds4 {
                config,
                weights,
                state,
                partials,
                partials_i64,
                policy,
            },
        }),
        deepseek4_eos_tok: eos_tok,
        mtp_mode: mtp_mode.to_string(),
        mtp_k,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
    })
}

fn load_model_ep_minimax(
    hfq: &mut HfqFile,
    path: &str,
    max_seq: usize,
    mesh: &DeviceMesh,
    mtp_mode: &str,
    mtp_k: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(hfq)?;
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
    let chat_template = resolve_chat_template(hfq, path);
    let rec = hfq.recommended_sampling();

    let ep = mesh.size_of(DimKind::Ep);
    // The exact EP execution policy of this model, built ONCE from the SAME
    // admitted mesh the `Gpus` below are bound to — the policy's mesh is a
    // clone of `mesh`, so it carries the exact same mesh epoch (the sealed
    // executor's mesh-identity check compares `policy.mesh()` against the
    // Gpus-bound epoch). The daemon threads `&model.policy` into every
    // `forward_ep` call; nothing reconstructs mesh/policy per token.
    let policy = MoEExecutionPolicy::new(MoEExecutionKind::Ep, mesh.clone())
        .map_err(|e| format!("minimax EP policy from admitted mesh: {e:?}"))?;
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
        let dev = &mut staging.gpus_mut().devices[r];
        let w = minimax::MiniMaxWeights::load(hfq, &config, dev, Some((&shard, r)), None)
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
        ep: Some(EpState {
            gpus,
            inner: EpArch::Minimax {
                config,
                weights,
                state,
                partials,
                partials_i64,
                policy,
            },
        }),
        minimax_eos_tok: eos_tok,
        mtp_mode: mtp_mode.to_string(),
        mtp_k,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
    })
}

/// Retry pending VMM arenas and refuse progress while any remain registered.
pub fn ensure_vmm_ready_for_load(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    gpu.ensure_vmm_cleaned().map_err(|err| {
        format!(
            "refusing load: prior VMM teardown still pending ({err}); retry unload or restart the process"
        )
    })
}

// ─── Unload ───────────────────────────────────────────────────────────

pub fn unload_model(mut m: LoadedModel, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    // Silent-corruption guard: validate the flat parallel layout BEFORE any
    // teardown, so a corrupted PP/EP/qwen35 layout fails with a descriptive
    // error instead of a partial free or an `.expect()` panic mid-free.
    validate_reset_layout(
        m.arch_id,
        m.pp,
        m.pp_gpus.as_ref(),
        m.pp_dn_la_to_device.as_deref(),
        m.ep.as_ref(),
        &m.state,
    )
    .map_err(|e| format!("unload aborted: {e:?}"))?;

    // EP unload-free. An EP model owns its own `Gpus` (the daemon's single `gpu`
    // is unused for tp>1). Without this branch a SUCCESSFUL EP unload leaked every
    // per-rank weight / state / partial. Free per-rank weights → state → partials
    // on each owning device, invalidate caches + graph state, drain each pool, then
    // drop the `Gpus` (tears down comms + devices). The daemon's `gpu` is untouched.
    // (The `partials` free here is what reclaims the ds4/minimax per-rank dummy
    // all-reduce buffer that would otherwise leak per load/unload cycle.)
    if let Some(ep) = m.ep.take() {
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
            return Ok(());
        }
    if m.pp > 1 {
        let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");
        if let Some(scratch_set) = m.pp_scratch_set {
            scratch_set.free_gpu_multi(&mut gpus);
        }
        match m.state.take() {
            Some(ModelState::Qwen35(b)) => {
                b.kv_cache.free_gpu_multi(&mut gpus);
                let la_to_device =
                    m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
                b.dn_state.free_gpu_multi(&mut gpus, &la_to_device);
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
        return Ok(());
    }

    // ── Generic path (single-device; incl. Qwen35 single) ─────────────
    if let Some(spec) = m.speculator {
        // Frees the drafter's GPU buffers (draft weights + scratch) AND its
        // checkpoint ring — a drafter that forgets is a compile error, not a
        // silent VRAM leak.
        spec.free(gpu);
    }
    // Native MTP (NextN) head — loaded once at load time and GPU-resident for
    // the life of the model; freed here so an unload without a reset never
    // leaks the head's buffers.
    if let Some(head) = m.qwen35_mtp_head {
        head.free_gpu(gpu);
    }
    if let Some(ev) = m.eviction {
        ev.free_gpu(gpu);
    }
    let mut first_err: Option<String> = None;
    let mut note = |r: Result<(), String>| {
        if let Err(err) = r {
            if first_err.is_none() {
                first_err = Some(err);
            }
        }
    };
    if let Some(kv) = m.kv_cache {
        note(kv.free_gpu(gpu).map_err(|e| e.to_string()));
    }
    if let Some(dn) = m.dn_state {
        dn.free_gpu(gpu);
    }
    for (_, snap) in m.prefill_checkpoints {
        snap.free_gpu(gpu);
    }
    for (_, snap) in m.dflash_checkpoints {
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
                note(b.kv_cache.free_gpu(gpu).map_err(|e| e.to_string()));
                b.scratch.free_gpu(gpu);
                b.weights.free_gpu(gpu);
                b.dn_state.free_gpu(gpu);
            }
            ModelState::Llama(b) => {
                b.scratch.free_gpu(gpu);
                b.weights.free_gpu(gpu);
                note(b.kv.free_gpu(gpu).map_err(|e| e.to_string()));
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
    if let Some(w) = m.vision_weights {
        w.free_gpu(gpu);
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    // After ordinary frees/pool drain, retry any VMM arenas retained by a
    // failed free_tensor. Success is reported only when none remain.
    note(gpu.ensure_vmm_cleaned().map_err(|e| e.to_string()));
    // Device-mesh carriers: bare drop reclaims nothing (TpModel/PpModel
    // document explicit `free()` as the only reclaim path — no freeing `Drop`
    // on GpuTensor/DeviceBuffer/GpuPool). Without this every TP/dense-PP
    // load/unload cycle leaked the whole model.
    if let Some(tp) = m.tp_model.take() {
        tp.free();
    }
    if let Some(pp) = m.pp_model.take() {
        pp.free();
    }
    match first_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
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
    fn skeleton_defaults_mtp_k() {
        let model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            None,
        );

        assert_eq!(model.mtp_k, 3);
    }

    #[test]
    fn skeleton_defaults_mtp_mode() {
        let model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            None,
        );

        assert_eq!(model.mtp_mode, "auto");
    }

    #[test]
    fn skeleton_pp_carries_flat_pipeline_fields() {
        let model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            None,
        );
        assert_eq!(model.mtp_k, 3);
        assert_eq!(model.mtp_mode, "auto");
        assert!(model.pp_gpus.is_none());
        assert!(model.pp_scratch_set.is_none());
        assert!(model.pp_dn_la_to_device.is_none());
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

    /// When a CASK sidecar is present, the KV cache allocation must be sized by
    /// the resolved `physical_cap` (~296 slots for budget=32, beta=8) rather
    /// than the full `max_seq` (2048). The VRAM difference between a CASK-bound
    /// load and a non-CASK load at the same max_seq confirms the KV buffers are
    /// physically smaller.
    #[test]
    #[ignore = "requires an AMD GPU and qwen3.6-27b.mq4 plus TriAttention sidecar"]
    fn qwen35_cask_physical_cap_reduces_kv_allocation() {
        use std::path::Path;

        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_QWEN35_CASK_TEST_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        let sidecar_path = std::env::var("HIPFIRE_TRIATTN_SIDECAR")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/triattn-centers.bin"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_QWEN35_CASK_TEST_TARGET"
        );
        if !Path::new(&sidecar_path).is_file() {
            eprintln!(
                "  SKIP: TriAttention sidecar not found at {sidecar_path}; \
                 set HIPFIRE_TRIATTN_SIDECAR to enable VRAM comparison"
            );
            return;
        }

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let no_cask = super::CaskConfig::default();

        let cask_config = super::CaskConfig {
            sidecar: Some(sidecar_path.clone()),
            cask_m_folding: false,
            budget: 32,
            beta: 8,
            core_frac: 0.5,
            fold_m: 2,
        };

        // Use max_seq large enough that budget+beta+safety (32+8+256=296) is <
        // max_seq, so physical_cap clamping leaves a measurable VRAM gap.
        const KV_MAX_SEQ: usize = 2048;
        const EXPECTED_PHYSICAL_CAP: usize = 296; // budget+beta+safety = 32+8+256

        // ── Warmup (max_seq=64, no CASK) ─────────────────────────────
        let warmup = super::load_model(
            &target_path,
            64,
            None,
            Some("q8"),
            None,
            None,
            &no_cask,
            &mesh,
            None,
            super::SpecLoadCfg::default(),
            &mut gpu,
        )
        .expect("warmup load");
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = free_vram(&gpu, "after warmup unload").expect("baseline VRAM");

        // ── CASK load (max_seq=2048, budget=32, beta=8) ──────────────
        let cask_model = super::load_model(
            &target_path,
            KV_MAX_SEQ,
            None,
            Some("q8"),
            None,
            None,
            &cask_config,
            &mesh,
            None,
            super::SpecLoadCfg::default(),
            &mut gpu,
        )
        .expect("CASK load");
        let cask_free = free_vram(&gpu, "after CASK load").expect("CASK VRAM");
        assert!(
            cask_free < baseline,
            "CASK load must allocate VRAM: baseline={baseline}, free={cask_free}"
        );
        // Check that physical_cap is reflected both in the metadata and the
        // underlying KvCache.
        assert_eq!(
            cask_model.physical_cap, EXPECTED_PHYSICAL_CAP,
            "CASK load physical_cap = budget+beta+safety, not max_seq"
        );
        if let Some(super::ModelState::Qwen35(ref bundle)) = cask_model.state {
            assert_eq!(
                bundle.kv_cache.physical_cap, EXPECTED_PHYSICAL_CAP,
                "CASK load KvCache.physical_cap must match resolved cap"
            );
        } else {
            panic!("CASK load did not produce a Qwen35 bundle");
        }

        super::unload_model(cask_model, &mut gpu);
        gpu.drain_pool();
        let after_cask = free_vram(&gpu, "after CASK unload").expect("post-CASK VRAM");
        assert_eq!(
            after_cask, baseline,
            "CASK unload leaked: baseline={baseline}, after={after_cask}"
        );

        // ── Non-CASK load (max_seq=2048, no sidecar) ─────────────────
        let full_model = super::load_model(
            &target_path,
            KV_MAX_SEQ,
            None,
            Some("q8"),
            None,
            None,
            &no_cask,
            &mesh,
            None,
            super::SpecLoadCfg::default(),
            &mut gpu,
        )
        .expect("non-CASK load");
        let full_free = free_vram(&gpu, "after non-CASK load").expect("non-CASK VRAM");
        assert!(
            full_free < baseline,
            "non-CASK load must allocate VRAM: baseline={baseline}, free={full_free}"
        );
        assert_eq!(
            full_model.physical_cap, KV_MAX_SEQ,
            "non-CASK load physical_cap = max_seq"
        );
        if let Some(super::ModelState::Qwen35(ref bundle)) = full_model.state {
            assert_eq!(
                bundle.kv_cache.physical_cap, KV_MAX_SEQ,
                "non-CASK load KvCache.physical_cap must be full max_seq"
            );
        } else {
            panic!("non-CASK load did not produce a Qwen35 bundle");
        }

        super::unload_model(full_model, &mut gpu);
        gpu.drain_pool();
        let after_full = free_vram(&gpu, "after non-CASK unload").expect("post-non-CASK VRAM");
        assert_eq!(
            after_full, baseline,
            "non-CASK unload leaked: baseline={baseline}, after={after_full}"
        );

        // ── CASK must leave more free VRAM than non-CASK ─────────────
        assert!(
            cask_free > full_free,
            "CASK load must leave more free VRAM than non-CASK load: \
             CASK free={cask_free}, non-CASK free={full_free} \
             (CASK physical_cap={EXPECTED_PHYSICAL_CAP} < max_seq={KV_MAX_SEQ})"
        );
    }

    /// The loader must reject budget/beta/cap combinations where eviction can
    /// never fire, before any GPU allocation. The budget=0 and budget+beta+4 >
    /// max_seq checks happen before the sidecar file is opened, so they're
    /// exercisable with a dummy path.
    #[test]
    #[ignore = "requires an AMD GPU and qwen3.6-27b.mq4"]
    fn qwen35_cask_rejects_impossible_budget_beta() {
        use std::path::Path;

        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_QWEN35_CASK_TEST_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_QWEN35_CASK_TEST_TARGET"
        );

        // A dummy path that exists; the budget=0 and impossible-config checks
        // fire BEFORE any sidecar read attempt.
        let dummy_sidecar = Some("/dev/null".to_string());

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let no_cask = super::CaskConfig::default();

        // Warmup: settle the driver so repeated load_model calls have a
        // consistent starting state.
        let warmup = super::load_model(
            &target_path,
            64,
            None,
            Some("q8"),
            None,
            None,
            &no_cask,
            &mesh,
            None,
            super::SpecLoadCfg::default(),
            &mut gpu,
        )
        .expect("warmup load");
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();

        // Manual `match` avoids `expect_err`'s `LoadedModel: Debug` bound.

        // ── budget=0 → must reject ─────────────────────────────────
        let zero_budget = super::CaskConfig {
            sidecar: dummy_sidecar.clone(),
            budget: 0,
            beta: 8,
            ..Default::default()
        };
        let error = match super::load_model(
            &target_path,
            64,
            None,
            Some("q8"),
            None,
            None,
            &zero_budget,
            &mesh,
            None,
            super::SpecLoadCfg::default(),
            &mut gpu,
        ) {
            Err(e) => e,
            Ok(_) => panic!("budget=0 must be rejected"),
        };
        assert!(
            error.contains("cask budget must be >0"),
            "expected budget=0 error, got: {error}"
        );
        gpu.drain_pool();

        // ── budget+beta+4 > max_seq → must reject ───────────────────
        // budget=32, beta=8 → budget+beta+4=44 > max_seq=40
        let impossible = super::CaskConfig {
            sidecar: dummy_sidecar.clone(),
            budget: 32,
            beta: 8,
            ..Default::default()
        };
        let error = match super::load_model(
            &target_path,
            40,
            None,
            Some("q8"),
            None,
            None,
            &impossible,
            &mesh,
            None,
            super::SpecLoadCfg::default(),
            &mut gpu,
        ) {
            Err(e) => e,
            Ok(_) => panic!("impossible budget+beta must be rejected"),
        };
        assert!(
            error.contains("eviction can never fire"),
            "expected 'eviction can never fire' error, got: {error}"
        );
        gpu.drain_pool();
    }

    /// Non-eviction (no CASK sidecar) loading must remain byte-identical:
    /// two loads with the same config must produce the same VRAM footprint.
    /// This verifies the `kv_physical_cap: None` path leaves allocation
    /// untouched.
    #[test]
    #[ignore = "requires an AMD GPU and qwen3.6-27b.mq4"]
    fn qwen35_no_eviction_load_unchanged() {
        use std::path::Path;

        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_QWEN35_CASK_TEST_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_QWEN35_CASK_TEST_TARGET"
        );

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let no_cask = super::CaskConfig::default();
        let load = |gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                None,
                Some("q8"),
                None,
                None,
                &no_cask,
                &mesh,
                None,
                super::SpecLoadCfg::default(),
                gpu,
            )
        };

        // ── Warmup ───────────────────────────────────────────────────
        let warmup = load(&mut gpu).expect("warmup load");
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = free_vram(&gpu, "after warmup unload").expect("baseline VRAM");

        // ── First load ───────────────────────────────────────────────
        let model_a = load(&mut gpu).expect("first non-CASK load");
        let free_a = free_vram(&gpu, "after first non-CASK load").expect("first VRAM");
        assert_eq!(
            model_a.physical_cap, 64,
            "non-CASK load must have physical_cap == max_seq"
        );
        super::unload_model(model_a, &mut gpu);
        gpu.drain_pool();
        let after_a = free_vram(&gpu, "after first non-CASK unload").expect("post-first VRAM");
        assert_eq!(
            after_a, baseline,
            "first non-CASK load leaked: baseline={baseline}, after={after_a}"
        );

        // ── Second load (identical config) ───────────────────────────
        let model_b = load(&mut gpu).expect("second non-CASK load");
        let free_b = free_vram(&gpu, "after second non-CASK load").expect("second VRAM");
        assert_eq!(
            model_b.physical_cap, 64,
            "second non-CASK load must also have physical_cap == max_seq"
        );

        // VRAM footprint must match the first load within the ROCm driver
        // accounting drift envelope.
        const DRIVER_VRAM_DRIFT_BYTES: usize = 64 * 1024 * 1024; // 64 MB
        assert!(
            free_a.abs_diff(free_b) <= DRIVER_VRAM_DRIFT_BYTES,
            "second non-CASK load has different VRAM footprint than first: \
             first={free_a}, second={free_b} (drift tolerance={} MB)",
            DRIVER_VRAM_DRIFT_BYTES / (1024 * 1024)
        );

        super::unload_model(model_b, &mut gpu);
        gpu.drain_pool();
        let after_b = free_vram(&gpu, "after second non-CASK unload").expect("post-second VRAM");
        assert_eq!(
            after_b, baseline,
            "second non-CASK load leaked: baseline={baseline}, after={after_b}"
        );
    }

    #[cfg(feature = "dflash-fault-inject")]
    #[test]
    #[ignore = "requires an AMD GPU plus generic DFlash target and draft fixtures"]
    fn generic_dflash_load_rolls_back_each_completed_resource() {
        use std::path::Path;

        let target_path = std::env::var("HIPFIRE_GENERIC_DFLASH_TARGET")
            .expect("HIPFIRE_GENERIC_DFLASH_TARGET is required");
        let draft_path = std::env::var("HIPFIRE_GENERIC_DFLASH_DRAFT")
            .expect("HIPFIRE_GENERIC_DFLASH_DRAFT is required");
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}"
        );
        assert!(
            Path::new(&draft_path).is_file(),
            "draft fixture not found: {draft_path}"
        );

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("synchronize before {context}: {e}"))?;
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        fn assert_free_vram_not_below_baseline(
            gpu: &rdna_compute::Gpu,
            baseline: usize,
            context: &str,
        ) {
            const DRIVER_VRAM_DRIFT_BYTES: usize = 64 * 1024 * 1024;
            let after = free_vram(gpu, context).expect("post-cleanup VRAM");
            assert!(
                after.saturating_add(DRIVER_VRAM_DRIFT_BYTES) >= baseline,
                "post-cleanup free VRAM exceeded the {DRIVER_VRAM_DRIFT_BYTES}-byte ROCm driver-accounting envelope for {context}: baseline={baseline}, after={after}"
            );
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let cask = super::CaskConfig::default();
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
                    dspark: Some(false),
                    ..Default::default()
                },
                gpu,
            )
        };

        let mut warmup = load(&mut gpu).expect("warmup generic DFlash load");
        assert!(warmup.speculator.is_some());
        {
            use hipfire_runtime::spec::PrefillOutcome;

            let bos = match warmup.state.as_ref().expect("loaded model state") {
                super::ModelState::Llama(bundle) => bundle.config.bos_token,
                _ => panic!("generic DFlash test loaded a non-LLaMA target"),
            };
            let bundle = match warmup.state.as_mut().expect("loaded model state") {
                super::ModelState::Llama(bundle) => bundle,
                _ => unreachable!(),
            };
            let speculator = warmup.speculator.as_mut().expect("published DFlash");
            let block_size = speculator.block_size();
            assert!(
                bundle.kv.physical_cap >= 1 + block_size,
                "generic DFlash block does not fit target KV capacity: block={block_size}, cap={}",
                bundle.kv.physical_cap
            );
            let prefill = speculator
                .prefill(&mut gpu, bundle, &[bos], &[bos], 0, false, None, &|| false)
                .expect("generic DFlash warmup prefill");
            let seed = match prefill {
                PrefillOutcome::Ready { first_token } => first_token,
                PrefillOutcome::Aborted => panic!("generic DFlash warmup prefill aborted"),
            };
            speculator
                .step(&mut gpu, bundle, 1, seed, &[bos], None, 0.0)
                .expect("generic DFlash warmup generation step");
        }
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = free_vram(&gpu, "after warmup unload").expect("baseline VRAM");

        let has_awq_scale = [target_path.as_str(), draft_path.as_str()]
            .iter()
            .any(|path| {
                hipfire_runtime::hfq::HfqFile::open(Path::new(path))
                    .map(|hfq| {
                        hfq.tensors()
                            .iter()
                            .any(|t| t.name.ends_with(".awq_scale.weight"))
                    })
                    .unwrap_or(false)
            });
        if has_awq_scale {
            let error =
                match hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
                    hipfire_runtime::dflash_generic::GenericDflashConstructionStage::AwqScaleUpload(
                        0,
                    ),
                    || load(&mut gpu),
                ) {
                    Ok(model) => {
                        super::unload_model(model, &mut gpu);
                        panic!("faulted AWQ scale upload unexpectedly succeeded");
                    }
                    Err(error) => error,
                };
            assert!(error.contains("test fault after generic DFlash"), "{error}");
            gpu.drain_pool();
            assert_free_vram_not_below_baseline(&gpu, baseline, "after AWQ scale rollback");
        }

        for stage in [
            hipfire_runtime::dflash_generic::GenericDflashConstructionStage::DraftWeights,
            hipfire_runtime::dflash_generic::GenericDflashConstructionStage::DraftScratch,
            hipfire_runtime::dflash_generic::GenericDflashConstructionStage::VerifyScratch,
            hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetWeights,
            hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetKv,
        ] {
            let error =
                match hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
                    stage,
                    || load(&mut gpu),
                ) {
                    Ok(model) => {
                        super::unload_model(model, &mut gpu);
                        panic!("faulted generic DFlash load unexpectedly succeeded");
                    }
                    Err(error) => error,
                };
            assert!(error.contains("test fault after generic DFlash"), "{error}");
            gpu.drain_pool();
            assert_free_vram_not_below_baseline(
                &gpu,
                baseline,
                &format!("after generic DFlash rollback ({})", stage.label()),
            );
        }

        const MAX_ALLOCATION_FAULT_DEPTH: usize = 4096;
        macro_rules! exercise_allocation_faults {
            ($variant:ident, $limit:expr, $label:literal) => {{
                let mut success = false;
                for allocation in 0..=$limit {
                    let stage =
                        hipfire_runtime::dflash_generic::GenericDflashConstructionStage::$variant(
                            allocation,
                        );
                    match hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
                        stage,
                        || load(&mut gpu),
                    ) {
                        Ok(model) => {
                            super::unload_model(model, &mut gpu);
                            gpu.drain_pool();
                            assert_free_vram_not_below_baseline(
                                &gpu,
                                baseline,
                                &format!(
                                    "after {} success sentinel at allocation {allocation}",
                                    $label
                                ),
                            );
                            success = true;
                            break;
                        }
                        Err(error) => {
                            assert!(error.contains("test fault after generic DFlash"), "{error}");
                            gpu.drain_pool();
                            assert_free_vram_not_below_baseline(
                                &gpu,
                                baseline,
                                &format!("after {} rollback at allocation {allocation}", $label),
                            );
                        }
                    }
                }
                assert!(
                    success,
                    "{} allocation fault loop did not reach a constructor-success sentinel",
                    $label
                );
            }};
        }

        exercise_allocation_faults!(
            VerifyScratchAllocation,
            hipfire_runtime::llama::GENERIC_DFLASH_VERIFY_SCRATCH_ALLOCATION_COUNT,
            "verify-scratch"
        );
        exercise_allocation_faults!(
            TargetWeightsAllocation,
            MAX_ALLOCATION_FAULT_DEPTH,
            "target-weights"
        );
        exercise_allocation_faults!(TargetKvAllocation, MAX_ALLOCATION_FAULT_DEPTH, "target-kv");

        for cycle in 0..3 {
            let mut model = load(&mut gpu).expect("successful generic DFlash load");
            assert!(
                model.speculator.is_some(),
                "cycle {cycle} did not publish DFlash"
            );
            if cycle == 0 {
                use hipfire_runtime::spec::PrefillOutcome;

                let bos = match model.state.as_ref().expect("loaded model state") {
                    super::ModelState::Llama(bundle) => bundle.config.bos_token,
                    _ => panic!("generic DFlash test loaded a non-LLaMA target"),
                };
                let bundle = match model.state.as_mut().expect("loaded model state") {
                    super::ModelState::Llama(bundle) => bundle,
                    _ => unreachable!(),
                };
                let speculator = model.speculator.as_mut().expect("published DFlash");
                let block_size = speculator.block_size();
                assert!(
                    bundle.kv.physical_cap >= 1 + block_size,
                    "generic DFlash block does not fit target KV capacity: block={block_size}, cap={}",
                    bundle.kv.physical_cap
                );
                let prefill = speculator
                    .prefill(&mut gpu, bundle, &[bos], &[bos], 0, false, None, &|| false)
                    .expect("generic DFlash prefill");
                let seed = match prefill {
                    PrefillOutcome::Ready { first_token } => first_token,
                    PrefillOutcome::Aborted => panic!("generic DFlash prefill aborted"),
                };
                speculator
                    .step(&mut gpu, bundle, 1, seed, &[bos], None, 0.0)
                    .expect("generic DFlash generation step");
            }
            super::unload_model(model, &mut gpu);
            gpu.drain_pool();
            assert_free_vram_not_below_baseline(
                &gpu,
                baseline,
                &format!("after generic DFlash unload (cycle {cycle})"),
            );
        }
    }

    #[cfg(feature = "dflash-fault-inject")]
    #[test]
    #[ignore = "requires an AMD GPU and a directory LLaMA fixture supporting q8/asym3/asym4"]
    fn directory_llama_kv_modes_roll_back_each_allocation() {
        use std::path::Path;

        let target_path = std::env::var("HIPFIRE_LLAMA_KV_ROLLBACK_TARGET")
            .expect("HIPFIRE_LLAMA_KV_ROLLBACK_TARGET is required");
        assert!(
            Path::new(&target_path).is_dir(),
            "directory fixture not found: {target_path}"
        );

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let cask = super::CaskConfig::default();
        let load = |mode: &str, gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                None,
                Some(mode),
                None,
                None,
                &cask,
                &mesh,
                None,
                super::SpecLoadCfg::default(),
                gpu,
            )
        };

        const MAX_ALLOCATION_FAULT_DEPTH: usize = 4096;
        for mode in ["q8", "asym3", "asym4"] {
            let warmup = load(mode, &mut gpu).expect("warmup directory LLaMA KV load");
            super::unload_model(warmup, &mut gpu);
            gpu.drain_pool();
            let baseline = free_vram(&gpu, "after KV warmup unload").expect("baseline VRAM");
            let mut success = false;
            for allocation in 0..=MAX_ALLOCATION_FAULT_DEPTH {
                let result =
                    hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
                        hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetKvAllocation(
                            allocation,
                        ),
                        || load(mode, &mut gpu),
                    );
                match result {
                    Ok(model) => {
                        super::unload_model(model, &mut gpu);
                        gpu.drain_pool();
                        assert_eq!(
                            free_vram(&gpu, "after KV success sentinel").expect("sentinel VRAM"),
                            baseline,
                            "{mode} KV success sentinel leaked at allocation {allocation}"
                        );
                        success = true;
                        break;
                    }
                    Err(error) => {
                        assert!(
                            error.contains("test fault after generic DFlash"),
                            "{mode}: {error}"
                        );
                        gpu.drain_pool();
                        assert_eq!(
                            free_vram(&gpu, "after KV rollback").expect("post-error VRAM"),
                            baseline,
                            "{mode} KV rollback leaked at allocation {allocation}"
                        );
                    }
                }
            }
            assert!(
                success,
                "{mode} KV allocation loop did not reach a success sentinel"
            );
        }
    }

    #[cfg(feature = "dflash-fault-inject")]
    #[test]
    #[ignore = "requires an AMD GPU and a ParoQuant directory fixture"]
    fn paro_weight_uploads_roll_back_each_upload() {
        use std::path::Path;

        let target_path = std::env::var("HIPFIRE_PARO_ROLLBACK_TARGET")
            .expect("HIPFIRE_PARO_ROLLBACK_TARGET is required");
        assert!(
            Path::new(&target_path).is_dir(),
            "ParoQuant directory fixture not found: {target_path}"
        );

        fn free_vram(gpu: &rdna_compute::Gpu, context: &str) -> Result<usize, String> {
            gpu.hip
                .get_vram_info()
                .map(|(free, _)| free)
                .map_err(|e| format!("measure {context}: {e}"))
        }

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let cask = super::CaskConfig::default();
        let load = |gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                None,
                Some("q8"),
                None,
                None,
                &cask,
                &mesh,
                None,
                super::SpecLoadCfg::default(),
                gpu,
            )
        };

        let warmup = load(&mut gpu).expect("warmup ParoQuant load");
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = free_vram(&gpu, "after Paro warmup unload").expect("baseline VRAM");

        const MAX_ALLOCATION_FAULT_DEPTH: usize = 4096;
        let mut success = false;
        for upload in 0..=MAX_ALLOCATION_FAULT_DEPTH {
            let result = hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
                hipfire_runtime::dflash_generic::GenericDflashConstructionStage::ParoWeightUpload(
                    upload,
                ),
                || load(&mut gpu),
            );
            match result {
                Ok(model) => {
                    super::unload_model(model, &mut gpu);
                    gpu.drain_pool();
                    assert_eq!(
                        free_vram(&gpu, "after Paro success sentinel").expect("sentinel VRAM"),
                        baseline,
                        "Paro success sentinel leaked at upload {upload}"
                    );
                    success = true;
                    break;
                }
                Err(error) => {
                    assert!(error.contains("test fault after generic DFlash"), "{error}");
                    gpu.drain_pool();
                    assert_eq!(
                        free_vram(&gpu, "after Paro rollback").expect("post-error VRAM"),
                        baseline,
                        "Paro rollback leaked at upload {upload}"
                    );
                }
            }
        }
        assert!(success, "Paro upload loop did not reach a success sentinel");
    }

    #[test]
    #[ignore = "requires an AMD GPU plus a LLaMA target and adjacent DSpark sidecar"]
    fn dspark_load_unload_reclaims_baseline() {
        use std::path::Path;

        let target_path =
            std::env::var("HIPFIRE_DSPARK_TARGET").expect("HIPFIRE_DSPARK_TARGET is required");
        let sidecar_path =
            std::env::var("HIPFIRE_DSPARK_SIDECAR").expect("HIPFIRE_DSPARK_SIDECAR is required");
        assert!(
            Path::new(&target_path).is_file(),
            "DSpark target fixture not found: {target_path}"
        );
        assert!(
            Path::new(&sidecar_path).is_file(),
            "DSpark sidecar fixture not found: {sidecar_path}"
        );

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let cask = super::CaskConfig::default();
        let load = |gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                None,
                Some("q8"),
                None,
                None,
                &cask,
                &mesh,
                None,
                super::SpecLoadCfg {
                    dspark: Some(true),
                    ..Default::default()
                },
                gpu,
            )
        };

        fn exercise_dspark_window(model: &mut super::LoadedModel, gpu: &mut rdna_compute::Gpu) {
            use hipfire_runtime::spec::PrefillOutcome;

            let bos = match model.state.as_ref().expect("loaded model state") {
                super::ModelState::Llama(bundle) => bundle.config.bos_token,
                _ => panic!("DSpark test loaded a non-LLaMA target"),
            };
            let bundle = match model.state.as_mut().expect("loaded model state") {
                super::ModelState::Llama(bundle) => bundle,
                _ => unreachable!(),
            };
            let speculator = model.speculator.as_mut().expect("published DSpark");
            let block_size = speculator.block_size();
            assert!(
                bundle.kv.physical_cap >= 1 + block_size,
                "DSpark block does not fit target KV capacity: block={block_size}, cap={}",
                bundle.kv.physical_cap
            );
            let prefill = speculator
                .prefill(gpu, bundle, &[bos], &[bos], 0, false, None, &|| false)
                .expect("DSpark prefill");
            let seed = match prefill {
                PrefillOutcome::Ready { first_token } => first_token,
                PrefillOutcome::Aborted => panic!("DSpark prefill aborted"),
            };
            speculator
                .step(gpu, bundle, 1, seed, &[bos], None, 0.0, 64)
                .expect("DSpark generation step");
        }

        let mut warmup = load(&mut gpu).expect("warmup DSpark load");
        assert!(warmup.speculator.is_some(), "DSpark was not published");
        exercise_dspark_window(&mut warmup, &mut gpu);
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = gpu.hip.get_vram_info().expect("baseline VRAM");

        for cycle in 0..3 {
            let mut model = load(&mut gpu).expect("successful DSpark load");
            assert!(
                model.speculator.is_some(),
                "cycle {cycle} did not publish DSpark"
            );
            exercise_dspark_window(&mut model, &mut gpu);
            super::unload_model(model, &mut gpu);
            gpu.drain_pool();
            assert_eq!(
                gpu.hip.get_vram_info().expect("post-unload VRAM"),
                baseline,
                "DSpark load/prefill/step/unload leaked on cycle {cycle}"
            );
        }
    }

    #[cfg(feature = "dflash-fault-inject")]
    #[test]
    #[ignore = "requires an AMD GPU plus a LLaMA target and adjacent DSpark sidecar"]
    fn dspark_sidecar_rolls_back_each_staging_milestone() {
        use std::path::Path;

        let target_path =
            std::env::var("HIPFIRE_DSPARK_TARGET").expect("HIPFIRE_DSPARK_TARGET is required");
        assert!(Path::new(&target_path).is_file());
        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let mesh = super::DeviceMesh::single();
        let cask = super::CaskConfig::default();
        let load = |gpu: &mut rdna_compute::Gpu| {
            super::load_model(
                &target_path,
                64,
                None,
                Some("q8"),
                None,
                None,
                &cask,
                &mesh,
                None,
                super::SpecLoadCfg {
                    dspark: Some(true),
                    ..Default::default()
                },
                gpu,
            )
        };

        let warmup = load(&mut gpu).expect("warmup DSpark load");
        super::unload_model(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = gpu.hip.get_vram_info().expect("baseline VRAM");

        let error = match hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
            hipfire_runtime::dflash_generic::GenericDflashConstructionStage::F32KvAllocation(2),
            || load(&mut gpu),
        ) {
            Ok(model) => {
                super::unload_model(model, &mut gpu);
                panic!("faulted DSpark F32 KV construction unexpectedly succeeded");
            }
            Err(error) => error,
        };
        assert!(error.contains("F32 KV allocation"), "{error}");
        gpu.drain_pool();
        assert_eq!(
            gpu.hip.get_vram_info().expect("F32 KV rollback VRAM"),
            baseline,
            "F32 KV construction leaked after a later allocation failed"
        );

        const MAX_ALLOCATION_FAULT_DEPTH: usize = 4096;
        let mut success = false;
        for allocation in 0..=MAX_ALLOCATION_FAULT_DEPTH {
            let result = hipfire_runtime::dflash_generic::with_generic_dflash_construction_fault(
                hipfire_runtime::dflash_generic::GenericDflashConstructionStage::DsparkAllocation(
                    allocation,
                ),
                || load(&mut gpu),
            );
            match result {
                Ok(model) => {
                    super::unload_model(model, &mut gpu);
                    gpu.drain_pool();
                    assert_eq!(
                        gpu.hip.get_vram_info().expect("DSpark success VRAM"),
                        baseline,
                        "DSpark success sentinel leaked at allocation {allocation}"
                    );
                    success = true;
                    break;
                }
                Err(error) => {
                    assert!(error.contains("DSpark allocation"), "{error}");
                    gpu.drain_pool();
                    assert_eq!(
                        gpu.hip.get_vram_info().expect("DSpark rollback VRAM"),
                        baseline,
                        "DSpark rollback leaked at allocation {allocation}"
                    );
                }
            }
        }
        assert!(success, "DSpark allocation loop did not reach success");
    }

    #[test]
    #[ignore = "requires an AMD GPU plus valid and malformed DSpark sidecar fixtures"]
    fn malformed_dspark_norm_rolls_back_after_sidecar_allocations() {
        use std::path::Path;

        let valid_sidecar =
            std::env::var("HIPFIRE_DSPARK_SIDECAR").expect("HIPFIRE_DSPARK_SIDECAR is required");
        let malformed_sidecar = std::env::var("HIPFIRE_DSPARK_MALFORMED_NORM_SIDECAR")
            .expect("HIPFIRE_DSPARK_MALFORMED_NORM_SIDECAR is required");
        assert!(Path::new(&valid_sidecar).is_file());
        assert!(Path::new(&malformed_sidecar).is_file());

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let load = |path: &str, gpu: &mut rdna_compute::Gpu| {
            let mut sidecar = hipfire_runtime::hfq::HfqFile::open(Path::new(path))
                .map_err(|e| format!("open DSpark sidecar {path}: {e}"))?;
            sidecar.drop_mmap();
            hipfire_arch_llama::dspark_body::load_qwen3_dspark(&sidecar, gpu)
        };
        let free_loaded = |(weights, assets): (
            hipfire_runtime::dspark_core::DsparkWeights,
            hipfire_arch_llama::dspark_body::Qwen3DrafterAssets,
        ),
                           gpu: &mut rdna_compute::Gpu| {
            assets.weights.free_gpu(gpu);
            assets.kv.free_gpu(gpu);
            assets.scratch.free_gpu(gpu);
            assets.pbs.free_gpu(gpu);
            weights.free_gpu(gpu);
        };

        let warmup = load(&valid_sidecar, &mut gpu)
            .expect("valid DSpark warmup")
            .expect("valid sidecar metadata");
        free_loaded(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = gpu.hip.get_vram_info().expect("baseline VRAM");

        let error = match load(&malformed_sidecar, &mut gpu) {
            Ok(Some(loaded)) => {
                free_loaded(loaded, &mut gpu);
                panic!("malformed DSpark norm unexpectedly loaded");
            }
            Ok(None) => panic!("malformed DSpark sidecar was not recognized"),
            Err(error) => error,
        };
        assert!(error.contains("norm"), "{error}");
        gpu.drain_pool();
        assert_eq!(
            gpu.hip.get_vram_info().expect("post-malformed VRAM"),
            baseline,
            "malformed DSpark norm leaked sidecar allocations"
        );
    }

    #[test]
    #[ignore = "requires an AMD GPU plus valid and malformed dequant_f32 DSpark sidecar fixtures"]
    fn malformed_dspark_dequant_f32_rolls_back_after_sidecar_allocations() {
        use std::path::Path;

        let valid_sidecar =
            std::env::var("HIPFIRE_DSPARK_SIDECAR").expect("HIPFIRE_DSPARK_SIDECAR is required");
        let malformed_sidecar = std::env::var("HIPFIRE_DSPARK_MALFORMED_DEQUANT_SIDECAR")
            .expect("HIPFIRE_DSPARK_MALFORMED_DEQUANT_SIDECAR is required");
        assert!(Path::new(&valid_sidecar).is_file());
        assert!(Path::new(&malformed_sidecar).is_file());

        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let load = |path: &str, gpu: &mut rdna_compute::Gpu| {
            let mut sidecar = hipfire_runtime::hfq::HfqFile::open(Path::new(path))
                .map_err(|e| format!("open DSpark sidecar {path}: {e}"))?;
            sidecar.drop_mmap();
            hipfire_arch_llama::dspark_body::load_qwen3_dspark(&sidecar, gpu)
        };
        let free_loaded = |(weights, assets): (
            hipfire_runtime::dspark_core::DsparkWeights,
            hipfire_arch_llama::dspark_body::Qwen3DrafterAssets,
        ),
                           gpu: &mut rdna_compute::Gpu| {
            assets.weights.free_gpu(gpu);
            assets.kv.free_gpu(gpu);
            assets.scratch.free_gpu(gpu);
            assets.pbs.free_gpu(gpu);
            weights.free_gpu(gpu);
        };

        let warmup = load(&valid_sidecar, &mut gpu)
            .expect("valid DSpark warmup")
            .expect("valid sidecar metadata");
        free_loaded(warmup, &mut gpu);
        gpu.drain_pool();
        let baseline = gpu.hip.get_vram_info().expect("baseline VRAM");

        let error = match load(&malformed_sidecar, &mut gpu) {
            Ok(Some(loaded)) => {
                free_loaded(loaded, &mut gpu);
                panic!("malformed DSpark dequant_f32 tensor unexpectedly loaded");
            }
            Ok(None) => panic!("malformed DSpark sidecar was not recognized"),
            Err(error) => error,
        };
        assert!(error.contains("dequant_f32"), "{error}");
        gpu.drain_pool();
        assert_eq!(
            gpu.hip.get_vram_info().expect("post-malformed VRAM"),
            baseline,
            "malformed DSpark dequant_f32 leaked prior sidecar allocations"
        );
    }

    #[test]
    #[ignore = "requires an AMD GPU plus a malformed HFQ target with a broken final norm"]
    fn malformed_hfq_final_norm_rolls_back_after_embedding_allocation() {
        use std::path::Path;

        let target_path = std::env::var("HIPFIRE_MALFORMED_HFQ_FINAL_NORM_TARGET")
            .expect("HIPFIRE_MALFORMED_HFQ_FINAL_NORM_TARGET is required");
        assert!(Path::new(&target_path).is_file());

        let mut hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&target_path))
            .expect("open malformed HFQ target");
        let config = hipfire_runtime::hfq::config_from_hfq(&hfq)
            .expect("malformed final norm fixture has valid config metadata");
        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        let baseline = gpu.hip.get_vram_info().expect("baseline VRAM");

        let error = match hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu) {
            Ok(weights) => {
                weights.free_gpu(&mut gpu);
                panic!("malformed HFQ final norm unexpectedly loaded");
            }
            Err(error) => error,
        };
        assert!(
            error.message.contains("model.norm.weight")
                || error.message.contains("expected F16/F32"),
            "{error:?}"
        );
        hfq.drop_mmap();
        gpu.drain_pool();
        assert_eq!(
            gpu.hip.get_vram_info().expect("post-malformed VRAM"),
            baseline,
            "malformed HFQ final norm leaked the prior embedding allocation"
        );
    }
}

#[cfg(test)]
mod reset_ownership_tests {
    use super::{
        architecture_error, ep_owner_visitation, qwen35_pp_owner_visitation,
        reject_qwen35_single_pipeline_metadata, require_qwen35_pipeline_metadata, reset_each_owner,
        reset_ownership_kind, validate_ep_layout, validate_qwen35_recurrent_cardinality,
        EpArchKind, ModelParallelKind, ModelStateKind, ResetError,
    };

    #[test]
    fn ownership_truth_table() {
        use ModelParallelKind::*;
        use ModelStateKind::*;

        #[derive(Clone, Copy)]
        enum Expected {
            Valid,
            Ownership,
            Architecture,
        }

        let cases = [
            ("single Qwen2", Single, Some(Qwen2), 7, Expected::Valid),
            ("single Qwen35", Single, Some(Qwen35), 5, Expected::Valid),
            ("single Llama", Single, Some(Llama), 0, Expected::Valid),
            ("single Lfm2Moe", Single, Some(Lfm2Moe), 11, Expected::Valid),
            ("single Minimax", Single, Some(Minimax), 10, Expected::Valid),
            (
                "single Cohere2Moe",
                Single,
                Some(Cohere2Moe),
                12,
                Expected::Valid,
            ),
            (
                "single Deepseek4",
                Single,
                Some(Deepseek4),
                9,
                Expected::Valid,
            ),
            ("single DotsOcr", Single, Some(DotsOcr), 8, Expected::Valid),
            ("Qwen35 PP", PpQwen35, Some(Qwen35), 5, Expected::Valid),
            ("parked Qwen35 PP", PpQwen35, None, 5, Expected::Ownership),
            (
                "mismatched Qwen35 PP",
                PpQwen35,
                Some(Qwen2),
                7,
                Expected::Ownership,
            ),
            ("TP with state", Tp, Some(Qwen35), 5, Expected::Ownership),
            (
                "dense PP with state",
                PpDense,
                Some(Qwen35),
                5,
                Expected::Ownership,
            ),
            ("EP with state", Ep, Some(Qwen35), 5, Expected::Ownership),
            ("TP owns state", Tp, None, 0, Expected::Valid),
            ("dense PP owns state", PpDense, None, 1, Expected::Valid),
            ("EP owns state", Ep, None, 9, Expected::Valid),
            ("single parked", Single, None, 0, Expected::Ownership),
            (
                "single architecture mismatch",
                Single,
                Some(Qwen2),
                5,
                Expected::Architecture,
            ),
            (
                "TP architecture mismatch",
                Tp,
                None,
                9,
                Expected::Architecture,
            ),
            (
                "dense PP architecture mismatch",
                PpDense,
                None,
                5,
                Expected::Architecture,
            ),
            (
                "EP architecture mismatch",
                Ep,
                None,
                0,
                Expected::Architecture,
            ),
            (
                "Qwen35 PP architecture mismatch",
                PpQwen35,
                Some(Qwen35),
                9,
                Expected::Architecture,
            ),
        ];

        for (name, parallel, state, arch_id, expected) in cases {
            let result = reset_ownership_kind(parallel, state, arch_id);
            match expected {
                Expected::Valid => assert_eq!(result, Ok(parallel), "{name}"),
                Expected::Ownership => assert!(
                    matches!(result, Err(ResetError::InvalidOwnership { .. })),
                    "{name}: result={result:?}"
                ),
                Expected::Architecture => assert!(
                    matches!(result, Err(ResetError::Architecture { .. })),
                    "{name}: result={result:?}"
                ),
            }
        }
    }

    #[test]
    // Characterizes post-load reset ownership; CAP-001 owns unified loader/daemon admission enforcement.
    fn parallel_reset_ownership_table() {
        use ModelParallelKind::*;
        use ModelStateKind::*;

        enum Expected {
            AcceptedNow,
            Refused,
        }

        let cases = [
            ("arch 0 TP", Tp, None, 0, Expected::AcceptedNow),
            ("arch 0 dense PP", PpDense, None, 0, Expected::AcceptedNow),
            ("arch 0 EP", Ep, None, 0, Expected::Refused),
            ("arch 1 TP", Tp, None, 1, Expected::AcceptedNow),
            ("arch 1 dense PP", PpDense, None, 1, Expected::AcceptedNow),
            ("arch 1 EP", Ep, None, 1, Expected::Refused),
            (
                "arch 5 Qwen35 PP",
                PpQwen35,
                Some(Qwen35),
                5,
                Expected::AcceptedNow,
            ),
            ("arch 5 TP", Tp, None, 5, Expected::Refused),
            ("arch 5 EP", Ep, None, 5, Expected::Refused),
            (
                "arch 6 Qwen35 PP",
                PpQwen35,
                Some(Qwen35),
                6,
                Expected::AcceptedNow,
            ),
            ("arch 6 TP", Tp, None, 6, Expected::Refused),
            ("arch 6 EP", Ep, None, 6, Expected::Refused),
            ("arch 7 dense PP", PpDense, None, 7, Expected::Refused),
            ("arch 7 TP", Tp, None, 7, Expected::Refused),
            ("arch 7 EP", Ep, None, 7, Expected::Refused),
            ("arch 8 dense PP", PpDense, None, 8, Expected::Refused),
            ("arch 8 TP", Tp, None, 8, Expected::Refused),
            ("arch 8 EP", Ep, None, 8, Expected::Refused),
            ("arch 9 EP", Ep, None, 9, Expected::AcceptedNow),
            ("arch 9 dense PP", PpDense, None, 9, Expected::Refused),
            ("arch 9 TP", Tp, None, 9, Expected::Refused),
            ("arch 10 EP", Ep, None, 10, Expected::AcceptedNow),
            ("arch 10 dense PP", PpDense, None, 10, Expected::Refused),
            ("arch 10 TP", Tp, None, 10, Expected::Refused),
            ("arch 11 dense PP", PpDense, None, 11, Expected::Refused),
            ("arch 11 TP", Tp, None, 11, Expected::Refused),
            ("arch 11 EP", Ep, None, 11, Expected::Refused),
            ("arch 12 dense PP", PpDense, None, 12, Expected::Refused),
            ("arch 12 TP", Tp, None, 12, Expected::Refused),
            ("arch 12 EP", Ep, None, 12, Expected::Refused),
        ];

        for (name, parallel, state, arch_id, expected) in cases {
            let result = reset_ownership_kind(parallel, state, arch_id);
            match expected {
                Expected::AcceptedNow => assert_eq!(result, Ok(parallel), "{name}"),
                Expected::Refused => assert!(
                    matches!(result, Err(ResetError::Architecture { .. })),
                    "{name}: result={result:?}"
                ),
            }
        }
    }

    #[test]
    fn ep_rejects_arch_variant_mismatch() {
        let result = validate_ep_layout(9, EpArchKind::Minimax, 2, 2);
        assert!(matches!(result, Err(ResetError::Architecture { .. })));
    }

    #[test]
    fn ep_rejects_state_device_cardinality_mismatch() {
        let result = validate_ep_layout(10, EpArchKind::Minimax, 2, 1);
        assert!(matches!(result, Err(ResetError::Architecture { .. })));
    }

    #[test]
    fn qwen35_pp_rejects_missing_pipeline_metadata() {
        let result = require_qwen35_pipeline_metadata(false, 5);
        assert!(matches!(result, Err(ResetError::Architecture { .. })));
    }

    #[test]
    fn single_qwen35_rejects_pipeline_metadata() {
        assert!(reject_qwen35_single_pipeline_metadata(false, 5).is_ok());
        let result = reject_qwen35_single_pipeline_metadata(true, 5);
        assert!(matches!(result, Err(ResetError::Architecture { .. })));
    }

    #[test]
    fn qwen35_pp_accepts_disabled_error_feedback() {
        assert!(validate_qwen35_recurrent_cardinality(3, 3, 3, 0, 3, 5).is_ok());
        assert!(validate_qwen35_recurrent_cardinality(3, 3, 3, 3, 3, 5).is_ok());
    }

    #[test]
    fn qwen35_pp_rejects_nonempty_error_feedback_cardinality_mismatch() {
        let result = validate_qwen35_recurrent_cardinality(3, 3, 3, 2, 3, 5);
        assert!(matches!(result, Err(ResetError::Architecture { .. })));
    }

    #[test]
    fn production_pp_reset_owner_invocations_are_observable() {
        let owners =
            qwen35_pp_owner_visitation(&[0, 1, 1, 0], 4, 2, 5).expect("valid emulated PP mapping");
        let mut invoked = Vec::new();
        reset_each_owner(&owners, |_index, owner| {
            invoked.push(owner);
            Ok(())
        })
        .unwrap();
        assert_eq!(invoked, owners);
        assert!(qwen35_pp_owner_visitation(&[0, 2], 2, 2, 5).is_err());
        assert!(qwen35_pp_owner_visitation(&[0], 2, 2, 5).is_err());
    }

    #[test]
    fn production_ep_reset_owner_invocations_are_observable() {
        let owners = ep_owner_visitation(9, 3, 3).unwrap();
        let mut invoked = Vec::new();
        reset_each_owner(&owners, |_index, owner| {
            invoked.push(owner);
            Ok(())
        })
        .unwrap();
        assert_eq!(invoked, owners);
        assert!(ep_owner_visitation(9, 3, 2).is_err());
    }

    #[test]
    fn reset_failure_classifies_gpu_owned_failures_as_fatal() {
        assert!(ResetError::Session {
            message: "checkpoint free failed".into(),
        }
        .is_gpu_fatal());
        assert!(ResetError::Speculator {
            message: "drafter reset failed".into(),
        }
        .is_gpu_fatal());
        assert!(architecture_error(
            ModelParallelKind::Single,
            Some(ModelStateKind::Qwen35),
            5,
            "reset recurrent state: hipErrorIllegalAddress"
        )
        .is_gpu_fatal());
        assert!(!architecture_error(
            ModelParallelKind::Single,
            Some(ModelStateKind::Qwen35),
            5,
            "single Qwen35 model unexpectedly carries pipeline metadata"
        )
        .is_gpu_fatal());
    }
}

#[cfg(all(test, feature = "arch-qwen35"))]
mod qwen35_tp_loader_tests {
    use super::load_model_tp;
    use hipfire_runtime::loader_api::SpecLoadCfg;
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
    use std::io::Write;

    fn write_metadata_only_hfq(path: &std::path::Path) {
        let metadata = serde_json::json!({
            "architecture": "qwen35",
            "config": {
                "hidden_size": 1024,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "vocab_size": 1000,
                "layer_types": ["linear_attention"]
            }
        })
        .to_string();
        let metadata_offset = 32u64;
        let index_offset = metadata_offset + metadata.len() as u64;
        let data_offset = (index_offset + 4 + 4095) & !4095;
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"HFQM").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&5u32.to_le_bytes()).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&metadata_offset.to_le_bytes()).unwrap();
        file.write_all(&data_offset.to_le_bytes()).unwrap();
        file.write_all(metadata.as_bytes()).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(vec![0; (data_offset - index_offset - 4) as usize].as_slice())
            .unwrap();
    }

    #[test]
    fn load_model_tp_rejects_qwen35_before_tokenizer_or_gpu_work() {
        let path = std::env::temp_dir().join(format!(
            "hipfire-qwen35-tp-preflight-{}-{}.hfq",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        write_metadata_only_hfq(&path);
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);

        let result = load_model_tp(path.to_str().unwrap(), 8, &mesh, SpecLoadCfg::default());
        let error = match result {
            Ok(_) => panic!("Qwen35 Tp=2 must be rejected before GPU work"),
            Err(error) => error,
        };
        // CAP-001 Task 3: admission guard now catches TP for Qwen35 dense
        // (AXIS-002 Planned) before any TP preflight check runs.
        assert!(
            error.contains("AXIS-002") || error.contains("planned"),
            "expected AXIS-002 / planned error from admission guard, got: {error}"
        );
        std::fs::remove_file(path).unwrap();
    }
}

// ─── CAP-001 Task 5: construction-order recorder ─────────────────────
//
// The cfg(test) recorder below observes the load_admitted → private-helper
// boundary.  It is NOT a test-only policy duplicate: it sits in production
// code (gated by #[cfg(test)]), records real dispatch decisions, and proves
// that admission refusals (Planned, Unsupported, InvalidDegree) never enter
// mesh / GPU / collective / peer / allocation / stream construction.
//
// Lifecycle:
//   - `record()` is called from `load_admitted` (RoutedTo) and the private
//     `load_*_admitted` helpers (EnteredConstructor).
//   - Tests call `take()` to drain recorded events for assertion.
//   - `clear()` resets state between test cases.

#[cfg(test)]
pub(crate) mod construction_recorder {
    use crate::parallel_capability::{ModelVariant, ParallelAxis, RawParallelRequest};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum RecordedEvent {
        /// The dispatch decision in `load_admitted` — the axis that was
        /// selected for routing.  Fires BEFORE the GPU check for Single
        /// axis, so it is observable with `gpu=None`.
        RoutedTo {
            axis: ParallelAxis,
            variant: ModelVariant,
            effective: RawParallelRequest,
        },
        /// The private constructor was entered.  Fires at the top of each
        /// `load_*_admitted` helper, before any mesh/GPU/tokenizer work.
        EnteredConstructor { axis: ParallelAxis },
    }

    // Thread-local storage so parallel test execution does not mix events.
    // Each test thread gets its own independent event Vec.
    use std::cell::RefCell;
    thread_local! {
        static RECORDED: RefCell<Vec<RecordedEvent>> = const { RefCell::new(Vec::new()) };
    }

    pub fn record(event: RecordedEvent) {
        RECORDED.with(|d| d.borrow_mut().push(event));
    }

    /// Drain all recorded events for the current thread.
    pub fn take() -> Vec<RecordedEvent> {
        RECORDED.with(|d| d.borrow_mut().drain(..).collect())
    }

    /// Clear all recorded events for the current thread.
    pub fn clear() {
        RECORDED.with(|d| d.borrow_mut().clear());
    }
}

// ─── CAP-001 Task 3 + Task 5 admission tests ────────────────────────

#[cfg(test)]
mod cap_001_loader_tests {
    use super::*;
    use crate::parallel_capability::{ModelVariant, ParallelAxis, RawParallelRequest};
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
    use std::io::Write;
    use std::path::Path;

    // ── Minimal HFQ fixture writer ───────────────────────────────────

    fn write_hfq(
        dir: &Path,
        name: &str,
        arch_id: u32,
        metadata_json: &str,
        tensor_payload_sizes: &[(&str, u64)],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        write_hfq_at(&path, arch_id, metadata_json, tensor_payload_sizes);
        path
    }

    fn tmp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn qwen35_vl_cfg() -> String {
        serde_json::json!({
            "model_type": "qwen3.5", "hidden_size": 2048,
            "num_hidden_layers": 24, "num_attention_heads": 16,
            "vocab_size": 152064,
            "vision_config": {"hidden_size": 1024, "num_layers": 6},
        })
        .to_string()
    }

    fn lfm2_cfg(num_experts: u64, model_type: &str) -> String {
        serde_json::json!({
            "model_type": model_type, "hidden_size": 2048,
            "num_hidden_layers": 4, "num_attention_heads": 16,
            "num_key_value_heads": 16, "intermediate_size": 8192,
            "vocab_size": 32000, "num_experts": num_experts,
        })
        .to_string()
    }

    fn vl_tensors() -> Vec<(&'static str, u64)> {
        vec![("model.visual.patch_embed.proj.weight", 12)]
    }

    // ── admit_path direct tests (public API) ─────────────────────────

    #[test]
    fn rejects_qwen35_vl_pp_with_axis004() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "q35vl.hfq", 5, &qwen35_vl_cfg(), &vl_tensors());
        let err = admit_path(path.to_str().unwrap(), RawParallelRequest::new(2, 1, 1)).unwrap_err();
        assert!(err.contains("AXIS-004"), "expected AXIS-004, got: {err}");
        assert!(err.contains("PP"), "expected PP mention, got: {err}");
    }

    #[test]
    fn admits_qwen35_vl_single() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "q35vl.hfq", 5, &qwen35_vl_cfg(), &vl_tensors());
        let a = admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 1)).unwrap();
        assert_eq!(a.admission().variant(), ModelVariant::Qwen35Vl);
        assert_eq!(a.admission().effective(), RawParallelRequest::new(1, 1, 1));
    }

    /// LFM2 dense EP normalises to Single → same Planned error as explicit Single.
    #[test]
    fn dense_ep_normalises_to_single_same_error() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "lfm2_d.hfq", 11, &lfm2_cfg(0, "lfm2"), &[]);
        let ep_err =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 2)).unwrap_err();
        let single_err =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 1)).unwrap_err();
        assert!(ep_err.contains("AXIS-003"), "EP: {ep_err}");
        assert!(single_err.contains("AXIS-003"), "Single: {single_err}");
        // Both fail at the same policy cell — the normalised-to-Single path
        // evaluates the same (Lfm2Dense, Single) cell as an explicit Single request.
        assert!(ep_err.contains("planned admission"), "EP: {ep_err}");
        assert!(
            single_err.contains("planned admission"),
            "Single: {single_err}"
        );
    }

    #[test]
    fn rejects_zero_degree() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "zero.hfq",
            5,
            &r#"{"model_type":"qwen3.5"}"#,
            &[],
        );
        let err = admit_path(path.to_str().unwrap(), RawParallelRequest::new(0, 1, 1)).unwrap_err();
        assert!(err.contains("degree"), "expected degree error, got: {err}");
    }

    // ── Axis enforcement: effective axis must match loader entry axis ──

    /// `load_model_ep` rejects when the effective axis is not EP.
    /// DeepSeek4 TP→EP remap preserves the EP effective axis, so it passes
    /// axis enforcement.  A non-EP request (e.g. LlamaNoQkNorm Single) fails.
    #[test]
    fn load_model_ep_requires_effective_ep() {
        let dir = tmp_dir();
        // Complete LLaMA config (no QK-norm → LlamaNoQkNorm, Single Admitted)
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "llama.hfq", 0, meta, &[]);
        // LlamaNoQkNorm with tp=1, ep=1 → effective axis = Single → fails
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 1), (DimKind::Ep, 1)]);
        let err = match load_model_ep_with_mesh(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected axis enforcement error, got Ok"),
        };
        assert!(
            err.contains("GPU") || err.contains("Single"),
            "expected Single/GPU error from load_admitted, got: {err}"
        );
    }

    /// `load_model_tp` rejects DeepSeek4 TP→EP remap: topology
    /// equality (effective != raw) centralized in [`load_admitted`].
    #[test]
    fn load_model_tp_rejects_deepseek4_remap() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        // tp=2, ep=1 → DeepSeek4 remap → effective (1,1,2) != raw (1,2,1)
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 1)]);
        let err = match load_model_tp(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected topology mismatch error, got Ok"),
        };
        assert!(
            err.contains("does not match"),
            "expected topology mismatch error from load_admitted, got: {err}"
        );
    }

    /// `load_model_pp` rejects normalised dense EP (resolved to Single)
    /// before reaching PpModel construction.
    #[test]
    fn load_model_pp_rejects_normalised_dense_ep_via_axis() {
        let dir = tmp_dir();
        // Complete LLaMA config → LlamaNoQkNorm, (NoQkNorm, Ep) → NormalizeToSingle
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "llama_ep.hfq", 0, meta, &[]);
        // LlamaNoQkNorm with ep=2 → NormalizeToSingle → effective (1,1,1) != raw (1,1,2)
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let err = match load_model_pp(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected topology mismatch error, got Ok"),
        };
        assert!(
            err.contains("does not match"),
            "expected topology mismatch error from load_admitted, got: {err}"
        );
    }

    // ── Real entry-point guards (Qwen35-VL PP→AXIS-004) ──────────────

    /// `load_model_tp` with Qwen35-VL + Tp=2 → AXIS-004 before tokenizer.
    #[test]
    fn tp_rejects_qwen35_vl_before_tokenizer() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "tpvl.hfq", 5, &qwen35_vl_cfg(), &vl_tensors());
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let err = match load_model_tp(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected AXIS-004 error, got Ok"),
        };
        assert!(err.contains("AXIS-004"), "expected AXIS-004, got: {err}");
    }

    /// `load_model_pp` with Qwen35-VL + Pp=2 → AXIS-004 before tokenizer.
    #[test]
    fn pp_rejects_qwen35_vl_before_tokenizer() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "ppvl.hfq", 5, &qwen35_vl_cfg(), &vl_tensors());
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]);
        let err = match load_model_pp(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected AXIS-004 error, got Ok"),
        };
        assert!(err.contains("AXIS-004"), "expected AXIS-004, got: {err}");
    }

    /// `load_model_ep` with LFM2 dense + Ep=2 → AXIS-003 (Planned) via
    /// admission (not axis enforcement), before any GPU work.
    #[test]
    fn ep_rejects_lfm2_dense_before_gpu() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "l2ep.hfq", 11, &lfm2_cfg(0, "lfm2"), &[]);
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let err = match load_model_ep_with_mesh(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected AXIS-003 error, got Ok"),
        };
        assert!(err.contains("AXIS-003"), "expected AXIS-003, got: {err}");
    }

    // ── Admitted source consumed ─────────────────────────────────────
    //
    // Verify that the admission result's variant matches expectations for
    // each loader's dispatching axis.  These use `admit_source` directly
    // (the same private function the loaders call), proving the source is
    // inspected once and the carrier/variant are correctly resolved.

    #[test]
    fn deepseek4_ep_admission_has_correct_variant() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        let path_str = path.to_str().unwrap();
        let src = ModelSource::from_path(path_str).unwrap();
        let admitted = admit_source(path_str, src, RawParallelRequest::new(1, 1, 2)).unwrap();
        assert_eq!(admitted.admission().variant(), ModelVariant::Deepseek4);
        assert_eq!(admitted.admission().effective().axis(), ParallelAxis::Ep);
    }

    // ── A→B pathname replacement proof ───────────────────────────────
    //
    // Prove that the admitted source carries artifact A's data even after
    // the file at the path is replaced with artifact B.  Every loader entry
    // point must consume the already-admitted source, not reopen the path.

    /// Write an HFQ fixture to an exact path (not dir+name).
    fn write_hfq_at(
        path: &Path,
        arch_id: u32,
        metadata_json: &str,
        tensor_payload_sizes: &[(&str, u64)],
    ) {
        let mut f = std::fs::File::create(path).unwrap();
        let meta_val: serde_json::Value =
            serde_json::from_str(metadata_json).expect("metadata must be valid JSON");
        let wrapped = if meta_val.get("config").is_some() {
            metadata_json.to_string()
        } else {
            format!(r#"{{"architecture":"test","config":{metadata_json}}}"#)
        };
        let meta_bytes = wrapped.as_bytes();
        let n_tensors = tensor_payload_sizes.len() as u32;
        let mut idx = Vec::new();
        idx.extend_from_slice(&n_tensors.to_le_bytes());
        for (tname, data_size) in tensor_payload_sizes {
            let nb = tname.as_bytes();
            idx.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            idx.extend_from_slice(nb);
            idx.push(1);
            idx.push(1);
            idx.extend_from_slice(&4u32.to_le_bytes());
            idx.extend_from_slice(&0u32.to_le_bytes());
            idx.extend_from_slice(&data_size.to_le_bytes());
        }
        let metadata_offset: u64 = 32;
        let data_offset: u64 = metadata_offset + meta_bytes.len() as u64 + idx.len() as u64;
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&n_tensors.to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(meta_bytes).unwrap();
        f.write_all(&idx).unwrap();
        for &(_, data_size) in tensor_payload_sizes {
            let payload = vec![0u8; data_size as usize];
            f.write_all(&payload).unwrap();
        }
        f.flush().unwrap();
    }

    /// Admitted source carries artifact A's arch_id after the on-disk
    /// file at path is replaced with artifact B (different arch_id).
    /// Proves the admission mechanism preserves the originally opened
    /// source and does NOT reopen the path.
    #[test]
    fn admitted_source_arch_id_survives_path_swap() {
        let dir = tmp_dir();
        let path = dir.path().join("swap.hfq");

        // Write artifact A (arch_id=9 = DeepSeek4)
        write_hfq_at(&path, 9, &r#"{"model_type":"deepseek_v4"}"#, &[]);

        // Admit source from A — opens and classifies the file
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 1, 2)).unwrap();
        assert_eq!(
            admitted.admission().variant(),
            ModelVariant::Deepseek4,
            "pre-swap: variant must be Deepseek4"
        );

        // Replace the on-disk file with artifact B (arch_id=5 = Qwen35Dense)
        write_hfq_at(
            &path,
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );

        // THE PROOF: consume the admitted source and verify it still
        // carries A's arch_id (9), NOT the replaced file's arch_id (5).
        let (_path, source, _carrier, admission) = admitted.into_parts();
        assert_eq!(
            admission.variant(),
            ModelVariant::Deepseek4,
            "post-swap: admission variant must still be Deepseek4 (artifact A)"
        );
        match source {
            ModelSource::Hfq(hfq) => {
                assert_eq!(
                    hfq.arch_id, 9,
                    "post-swap: HfqFile.arch_id must be 9 (artifact A, not 5 from replaced file)"
                );
            }
            _ => panic!("expected Hfq source"),
        }
    }

    /// Admitted source carries artifact A's data (via `admit_path`) after
    /// the on-disk path is replaced with artifact B. Proves the admitted
    /// source is consumed from the admission token, not reopened.
    #[test]
    fn continuation_sees_artifact_a_after_path_swap() {
        let dir = tmp_dir();
        let path = dir.path().join("swap2.hfq");

        // Write artifact A (arch_id=9, DeepSeek4)
        write_hfq_at(&path, 9, &r#"{"model_type":"deepseek_v4"}"#, &[]);

        let raw = RawParallelRequest::new(1, 1, 2);
        let path_str = path.to_str().unwrap();
        let admitted = admit_path(path_str, raw).unwrap();

        // Replace the file at the SAME path with artifact B (arch_id=5)
        write_hfq_at(
            &path,
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );

        // The admitted source was opened from A; check its arch_id NOT
        // the replaced file's arch_id.
        assert_eq!(
            admitted.source_ref().arch_id(),
            Some(9),
            "admitted source arch_id must be 9 (A), not 5 (B)"
        );
        assert_eq!(
            admitted.admission().variant(),
            ModelVariant::Deepseek4,
            "variant must still be Deepseek4 (A)"
        );
    }

    // ── load_model_ep topology equality (effective EP != raw) ──────────

    /// `load_model_ep` rejects DeepSeek4 with a raw TP mesh (tp=2, ep=1)
    /// where the legacy TP→EP remap makes effective (1,1,2) != raw (1,2,1).
    /// The topology equality check fires BEFORE any GPU work.
    #[test]
    fn load_model_ep_rejects_deepseek4_tp_remap_topology_mismatch() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4_tp.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        // tp=2, ep=1 → DeepSeek4 remap → effective (1,1,2) != raw (1,2,1)
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 1)]);
        let err = match load_model_ep_with_mesh(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected topology mismatch error, got Ok"),
        };
        assert!(
            err.contains("does not match"),
            "expected topology mismatch error, got: {err}"
        );
    }

    /// `load_model_ep` rejects Minimax TP→EP remap the same way: raw
    /// tp=2 after remap makes effective != raw, topology equality fires.
    #[test]
    fn load_model_ep_rejects_minimax_tp_remap_topology_mismatch() {
        let dir = tmp_dir();
        // Minimal Minimax config (arch_id=10)
        let meta = r#"{"config":{"model_type":"minimax","vocab_size":32000,"hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":16,"intermediate_size":8192,"num_local_experts":8,"num_experts_per_tok":2}}"#;
        let path = write_hfq(dir.path(), "mm_tp.hfq", 10, meta, &[]);
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 1)]);
        let err = match load_model_ep_with_mesh(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected topology mismatch error, got Ok"),
        };
        assert!(
            err.contains("does not match"),
            "expected topology mismatch from load_admitted for Minimax TP→EP remap, got: {err}"
        );
    }

    // ── Malformed-tokenizer CAP error before continuation/GPU ────────

    // ── Generic load routing seam (route_generic_single) ──────────

    /// Normalized dense EP is accepted as effective Single, retaining source.
    #[test]
    fn route_generic_single_accepts_normalized_dense_ep() {
        let dir = tmp_dir();
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "dense_ep.hfq", 0, meta, &[]);
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        // LlamaNoQkNorm + ep=2 → NormalizeToSingle → effective (1,1,1) Single
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 1, 2)).unwrap();
        assert_eq!(
            admitted.admission().effective().axis(),
            ParallelAxis::Single
        );
        let readmitted = route_generic_single(admitted).unwrap();
        // Source retained, variant preserved
        assert_eq!(
            readmitted.admission().variant(),
            ModelVariant::LlamaNoQkNorm
        );
        assert_eq!(
            readmitted.admission().effective(),
            RawParallelRequest::new(1, 1, 1)
        );
        assert!(readmitted.source_ref().arch_id().is_some());
    }

    /// The public `load_model` continuation must replace a normalized raw EP
    /// mesh with the admitted effective Single mesh before `load_admitted`.
    #[test]
    fn load_model_uses_effective_mesh_for_qwen35_dense_ep() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "qwen35_dense_ep.hfq",
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );
        let admitted = route_generic_single(
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 2)).unwrap(),
        )
        .unwrap();
        assert_eq!(admitted.admission().variant(), ModelVariant::Qwen35Dense);
        assert_eq!(
            admitted.admission().effective(),
            RawParallelRequest::new(1, 1, 1)
        );

        let effective_mesh = effective_load_mesh(&admitted);
        assert_eq!(
            raw_request(&effective_mesh),
            admitted.admission().effective()
        );
        assert_eq!(effective_mesh.size_of(DimKind::Pp), 1);
        assert_eq!(effective_mesh.size_of(DimKind::Tp), 1);
        assert_eq!(effective_mesh.size_of(DimKind::Ep), 1);
    }

    /// A normal Single admission must borrow the caller's mesh, preserving its
    /// exact axis representation rather than replacing it with a canonical mesh.
    #[test]
    fn load_model_preserves_original_mesh_for_normal_admission() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "qwen35_single.hfq",
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );

        for mesh in [
            DeviceMesh::single(),
            DeviceMesh::rect(&[(DimKind::Ep, 1), (DimKind::Tp, 1)]),
        ] {
            let admitted = route_generic_single(
                admit_path(path.to_str().unwrap(), raw_request(&mesh)).unwrap(),
            )
            .unwrap();
            assert!(!admitted.admission().was_normalized());

            let selected = select_load_mesh(&admitted, &mesh);
            assert!(std::ptr::eq(selected.as_ref(), &mesh));
            assert_eq!(selected.axes(), mesh.axes());
        }
    }

    /// Admitted Qwen3 TP is rejected (route_generic_single expects Single).
    #[test]
    fn route_generic_single_rejects_qwen3_tp() {
        let dir = tmp_dir();
        // PlainQwen3 (arch_id=1) with TP=2 is Admitted
        let meta = r#"{"model_type":"qwen3","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "q3tp.hfq", 1, meta, &[]);
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 2, 1)).unwrap();
        assert_eq!(admitted.admission().effective().axis(), ParallelAxis::Tp);
        let err = match route_generic_single(admitted) {
            Err(e) => e,
            Ok(_) => panic!("route_generic_single must reject Qwen3 TP"),
        };
        assert!(
            err.contains("Single"),
            "expected Single rejection for TP, got: {err}"
        );
        assert!(
            err.contains("load_model_tp"),
            "expected load_model_tp hint, got: {err}"
        );
    }

    /// Admitted DeepSeek EP is rejected (route_generic_single expects Single).
    #[test]
    fn route_generic_single_rejects_deepseek4_ep() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4ep.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 1, 2)).unwrap();
        assert_eq!(admitted.admission().effective().axis(), ParallelAxis::Ep);
        let err = match route_generic_single(admitted) {
            Err(e) => e,
            Ok(_) => panic!("route_generic_single must reject DeepSeek EP"),
        };
        assert!(
            err.contains("Single"),
            "expected Single rejection for EP, got: {err}"
        );
        assert!(
            err.contains("load_model_ep"),
            "expected load_model_ep hint, got: {err}"
        );
    }

    /// Effective topology mismatch: DeepSeek4 TP with raw (1,2,1) and
    /// effective (1,1,2) fails the topology-equality check in load_model_ep.
    /// The error must identify the mismatch, not fire a generic admission
    /// refusal.
    #[test]
    fn load_model_ep_topology_mismatch_error_is_explicit() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4_tp2.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 1)]);
        let err = match load_model_ep_with_mesh(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected topology mismatch error, got Ok"),
        };
        assert!(
            err.contains("does not match"),
            "expected 'topology mismatch', got: {err}"
        );
    }

    // ── Public load_admitted with Option<Gpu> ─────────────────────────

    /// Mismatched mesh with `None` GPU: topology mismatch before GPU.
    #[test]
    fn load_admitted_mesh_mismatch_rejected_before_gpu() {
        let dir = tmp_dir();
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "llama_single.hfq", 0, meta, &[]);
        // Admit with Single request — effective (1,1,1).
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 1)).unwrap();
        // Deliberately mismatched mesh: Tp=2 → raw=(1,2,1) != effective=(1,1,1)
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let err = match load_admitted(
            admitted,
            &mesh,
            ModelLoadOptions::new(64).with_spec(SpecLoadCfg::default()),
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected mesh/effective mismatch error, got Ok"),
        };
        assert!(
            err.contains("mesh request") && err.contains("does not match"),
            "expected mesh/effective mismatch error, got: {err}"
        );
    }

    /// Matching Single admission with `None` GPU: GPU-required before
    /// source/GPU construction.
    #[test]
    fn load_admitted_single_requires_gpu() {
        let dir = tmp_dir();
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "llama_single_gpu.hfq", 0, meta, &[]);
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 1)).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 1)]);
        let err = match load_admitted(
            admitted,
            &mesh,
            ModelLoadOptions::new(64).with_spec(SpecLoadCfg::default()),
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected GPU-required error, got Ok"),
        };
        assert!(
            err.contains("Single load requires a GPU"),
            "expected GPU-required error, got: {err}"
        );
    }

    // ── Admitted source carry-forward (direct rank loop proof) ───────

    /// Admit/open A, write B to a temp then `rename(B, A_path)`, invoke
    /// a direct `for` loop (the EP rank pattern) for 3 iterations, and
    /// assert each iteration observes A-specific metadata.  No
    /// `File::create` truncation of A.
    #[test]
    fn admitted_source_survives_path_swap_in_direct_rank_loop() {
        let dir = tmp_dir();
        let path_a = dir.path().join("ep_iter.hfq");

        // Artifact A: DeepSeek4 (arch_id=9)
        write_hfq_at(&path_a, 9, &r#"{"model_type":"deepseek_v4"}"#, &[]);

        // Open/admit A
        let src_path = path_a.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 1, 1)).unwrap();
        let (_path, source, _carrier, _) = admitted.into_parts();
        let hfq = match source {
            ModelSource::Hfq(hfq) => hfq,
            _ => panic!("expected Hfq source"),
        };

        // Write B to a temp path (arch_id=5, Qwen35Dense — different
        // from A's arch_id=9).  Do NOT open B — only write it to disk.
        let path_b = dir.path().join("ep_iter_b.hfq");
        write_hfq_at(
            &path_b,
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );

        // Atomically replace A with B (rename, not File::create).
        std::fs::rename(&path_b, &path_a).unwrap();

        // Invoke a direct `for` loop (mirroring the EP rank-weight load
        // pattern in `load_model_ep_ds4`).  Each iteration must observe
        // A-specific metadata (arch_id=9), not B.
        let mut call_count = 0usize;
        for r in 0..3 {
            call_count += 1;
            assert_eq!(
                hfq.arch_id, 9,
                "rank {r}: expected arch_id=9 (artifact A), not B"
            );
        }
        assert_eq!(call_count, 3, "expected 3 rank iterations");
    }

    // ── Atomic A→B TP constructor proof ──────────────────────────────
    //
    // Admit A (PlainQwen3, no q_norm tensor), replace on-disk path with
    // artifact B (arch_id=9), call TpModel::load_from_hfq(&A_hfq, ...),
    // assert A-specific qk-norm preflight error — NOT B's arch-id error.

    #[test]
    fn tp_load_from_hfq_sees_artifact_a_after_path_swap() {
        use hipfire_runtime::tp_serve::TpModel;
        let dir = tmp_dir();
        let path_a = dir.path().join("tp_swap.hfq");

        // Artifact A: PlainQwen3 (arch_id=1), valid Qwen3 config,
        // NO q_norm tensor.  This config is full enough to reach the
        // qk-norm check inside `TpModel::load_from_hfq_inner`.
        write_hfq_at(
            &path_a,
            1,
            &r#"{"config":{"model_type":"qwen3","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}}"#,
            &[],
        );

        // Open/admit A
        let src_path = path_a.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 2, 1)).unwrap();
        let hfq_ref = match admitted.source_ref() {
            ModelSource::Hfq(hfq) => hfq,
            _ => panic!("expected Hfq source"),
        };

        // Write B to temp path (arch_id=9 DeepSeek4, different artifact)
        let path_b = dir.path().join("tp_swap_b.hfq");
        write_hfq_at(
            &path_b,
            9,
            &r#"{"config":{"model_type":"deepseek_v4","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}}"#,
            &[],
        );

        // Atomically replace A with B
        std::fs::rename(&path_b, &path_a).unwrap();

        // Call TpModel::load_from_hfq with the already-opened A HfqFile.
        // A has arch_id=1 (in range 0|1) but no q_norm tensor.
        // Expected: qk-norm error (A-specific), NOT arch_id=9 (B's error).
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let err = match TpModel::load_from_hfq(hfq_ref, &mesh, 64) {
            Err(e) => e,
            Ok(_) => panic!("expected qk-norm error, got Ok"),
        };
        assert!(
            err.contains("qk-norm"),
            "expected A-specific qk-norm error, got: {err}"
        );
        assert!(
            !err.contains("arch_id=9"),
            "error must NOT reference B's arch_id=9, got: {err}"
        );
    }

    /// A MiniMax EP fixture with valid config but NO "tokenizer" in the
    /// metadata must fail at tokenizer creation inside the EP constructor,
    /// BEFORE any GPU allocation. The error must mention "tokenizer".
    #[test]
    fn malformed_tokenizer_cap_error_before_ep_gpu() {
        let dir = tmp_dir();
        // Valid Minimax config but NO tokenizer key in metadata.
        // config is the MINIMAL set required by RawMiniMaxConfig.
        let meta = r#"{"config":{"model_type":"minimax","vocab_size":32000,"hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":16,"intermediate_size":8192,"num_local_experts":8,"num_experts_per_tok":2}}"#;
        let path = write_hfq(dir.path(), "mm_notok.hfq", 10, meta, &[]);
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let err = match load_model_ep_with_mesh(
            path.to_str().unwrap(),
            64,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected tokenizer error before GPU, got Ok"),
        };
        assert!(
            err.contains("tokenizer"),
            "expected tokenizer error (no 'tokenizer' key in metadata), got: {err}"
        );
        // Confirm the error is NOT a GPU-related message (no "gpu",
        // "hipMalloc", "VRAM", "bind", "graph capture" etc.)
        let gpu_hints = ["gpu", "hipMalloc", "VRAM", "bind", "graph"];
        for hint in &gpu_hints {
            assert!(
                !err.to_lowercase().contains(hint),
                "tokenizer error must NOT reference GPU ({hint}), got: {err}"
            );
        }
    }

    #[test]
    fn llama_tp_admission_has_correct_variant() {
        let dir = tmp_dir();
        // LlamaQkNorm (arch_id=0) with Tp=2 - need QK-norm tensor for LlamaQkNorm
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(
            dir.path(),
            "llama.hfq",
            0,
            meta,
            &[("model.layers.0.self_attn.q_norm.weight", 4)],
        );
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 2, 1)).unwrap();
        assert_eq!(admitted.admission().variant(), ModelVariant::LlamaQkNorm);
        assert_eq!(admitted.admission().effective().axis(), ParallelAxis::Tp);
    }

    // ── admit_path admission tests ────────────────────────────────────

    /// `admit_path` returns correct admission state (variant + effective).
    #[test]
    fn admit_path_returns_correct_admission() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 2, 1)).unwrap();
        assert_eq!(admitted.admission().variant(), ModelVariant::Deepseek4);
        // DeepSeek4 TP→EP remap effective (1,1,2)
        assert_eq!(
            admitted.admission().effective(),
            RawParallelRequest::new(1, 1, 2)
        );
    }

    /// Dense EP normalisation via `admit_path` — effective (1,1,1).
    #[test]
    fn admit_path_dense_ep_normalises_to_single() {
        let dir = tmp_dir();
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "llama_ep.hfq", 0, meta, &[]);
        // LlamaNoQkNorm + ep=2 → NormalizeToSingle → effective (1,1,1), axis=Single
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 2)).unwrap();
        assert_eq!(
            admitted.admission().effective(),
            RawParallelRequest::new(1, 1, 1)
        );
        assert_eq!(admitted.admission().variant(), ModelVariant::LlamaNoQkNorm);
        assert_eq!(
            admitted.admission().effective().axis(),
            ParallelAxis::Single
        );
    }

    /// Qwen35Dense EP=2 normalizes to Single before construction. The integrated
    /// path must retain the normalized admission and enter the Single boundary.
    #[test]
    fn qwen35_dense_ep_normalizes_to_single_and_records_single_route() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "qwen35_ep.hfq",
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );

        let admitted = admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 2))
            .expect("dense Qwen35 EP admission");
        assert_eq!(admitted.admission().variant(), ModelVariant::Qwen35Dense);
        assert_eq!(
            admitted.admission().effective(),
            RawParallelRequest::new(1, 1, 1)
        );

        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 1)]);
        let _err = load_admitted(admitted, &mesh, ModelLoadOptions::new(64), None);
        assert_eq!(
            crate::construction_recorder::take(),
            vec![
                crate::construction_recorder::RecordedEvent::RoutedTo {
                    axis: ParallelAxis::Single,
                    variant: ModelVariant::Qwen35Dense,
                    effective: RawParallelRequest::new(1, 1, 1),
                },
                crate::construction_recorder::RecordedEvent::EnteredConstructor {
                    axis: ParallelAxis::Single,
                },
            ]
        );
    }

    /// Qwen35-VL PP → AXIS-004 error from `admit_path`.
    #[test]
    fn admit_path_rejects_qwen35_vl_pp() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "q35vl.hfq", 5, &qwen35_vl_cfg(), &vl_tensors());
        let err = admit_path(path.to_str().unwrap(), RawParallelRequest::new(2, 1, 1)).unwrap_err();
        assert!(err.contains("AXIS-004"), "expected AXIS-004, got: {err}");
    }

    /// Source consumption proof: the admitted source carries the expected
    /// arch_id.  Verifies the source is consumed from AdmittedLoad, not
    /// reopened.
    #[test]
    fn admitted_source_consumed() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "ds4_src.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(1, 1, 2)).unwrap();
        // The source's arch_id must match the admitted variant's expectation
        assert_eq!(admitted.source_ref().arch_id(), Some(9));
        assert_eq!(admitted.admission().variant(), ModelVariant::Deepseek4);
        // consume via into_parts to prove ownership transfer
        let (_path, _source, _carrier, _admission) = admitted.into_parts();
    }

    #[test]
    fn llama_pp_admission_has_correct_variant() {
        let dir = tmp_dir();
        // No QK-norm tensor → LlamaNoQkNorm variant
        let meta = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "llama.hfq", 0, meta, &[]);
        let src_path = path.to_str().unwrap();
        let src = ModelSource::from_path(src_path).unwrap();
        let admitted = admit_source(src_path, src, RawParallelRequest::new(2, 1, 1)).unwrap();
        assert_eq!(admitted.admission().variant(), ModelVariant::LlamaNoQkNorm);
        assert_eq!(admitted.admission().effective().axis(), ParallelAxis::Pp);
    }

    // ── CAP-001 Task 5: construction recorder ────────────────────────

    /// Admitted Single via `load_admitted(gpu=None)` records the dispatch and
    /// constructor boundary in order, including the constructor event before
    /// the GPU-required failure.
    #[test]
    fn construction_recorder_single_gpu_none_fires_routed_to_only() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        // Qwen35Dense (arch_id=5) Single is Admitted — no QK-norm, no VL config needed
        let path = write_hfq(
            dir.path(),
            "rec_single.hfq",
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 1)).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 1)]);
        let _err = load_admitted(admitted, &mesh, ModelLoadOptions::new(64), None);

        let events = crate::construction_recorder::take();
        assert_eq!(
            events,
            vec![
                crate::construction_recorder::RecordedEvent::RoutedTo {
                    axis: ParallelAxis::Single,
                    variant: ModelVariant::Qwen35Dense,
                    effective: RawParallelRequest::new(1, 1, 1),
                },
                crate::construction_recorder::RecordedEvent::EnteredConstructor {
                    axis: ParallelAxis::Single,
                },
            ]
        );
    }

    /// Admitted TP via `load_admitted(gpu=None)` records BOTH `RoutedTo Tp`
    /// and `EnteredConstructor Tp` — TP doesn't need a GPU, so the dispatch
    /// reaches the private constructor (which then fails at tokenizer).
    #[test]
    fn construction_recorder_tp_fires_routed_to_and_entered() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        // PlainQwen3 (arch_id=1) with Tp=2 is Admitted.
        let meta = r#"{"model_type":"qwen3","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "rec_tp.hfq", 1, meta, &[]);
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 2, 1)).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let _err = load_admitted(admitted, &mesh, ModelLoadOptions::new(64), None);

        let events = crate::construction_recorder::take();
        assert_eq!(
            events,
            vec![
                crate::construction_recorder::RecordedEvent::RoutedTo {
                    axis: ParallelAxis::Tp,
                    variant: ModelVariant::PlainQwen3,
                    effective: RawParallelRequest::new(1, 2, 1),
                },
                crate::construction_recorder::RecordedEvent::EnteredConstructor {
                    axis: ParallelAxis::Tp,
                },
            ]
        );
    }

    /// Admitted PP via `load_admitted(gpu=None)` records BOTH `RoutedTo Pp`
    /// and `EnteredConstructor Pp`.
    #[test]
    fn construction_recorder_pp_fires_routed_to_and_entered() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        // PlainQwen3 (arch_id=1) with Pp=2 is Admitted.
        let meta = r#"{"model_type":"qwen3","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}"#;
        let path = write_hfq(dir.path(), "rec_pp.hfq", 1, meta, &[]);
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(2, 1, 1)).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]);
        let _err = load_admitted(admitted, &mesh, ModelLoadOptions::new(64), None);

        let events = crate::construction_recorder::take();
        assert_eq!(
            events,
            vec![
                crate::construction_recorder::RecordedEvent::RoutedTo {
                    axis: ParallelAxis::Pp,
                    variant: ModelVariant::PlainQwen3,
                    effective: RawParallelRequest::new(2, 1, 1),
                },
                crate::construction_recorder::RecordedEvent::EnteredConstructor {
                    axis: ParallelAxis::Pp,
                },
            ]
        );
    }

    /// Admitted EP via `load_admitted` records both `RoutedTo Ep` and
    /// `EnteredConstructor Ep`.
    #[test]
    fn construction_recorder_ep_fires_routed_to_and_entered() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        // DeepSeek4 (arch_id=9) with Ep=2 is Admitted.
        let path = write_hfq(
            dir.path(),
            "rec_ep.hfq",
            9,
            &r#"{"model_type":"deepseek_v4"}"#,
            &[],
        );
        let admitted =
            admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 2)).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let _err = load_admitted(admitted, &mesh, ModelLoadOptions::new(64), None);

        let events = crate::construction_recorder::take();
        assert_eq!(
            events,
            vec![
                crate::construction_recorder::RecordedEvent::RoutedTo {
                    axis: ParallelAxis::Ep,
                    variant: ModelVariant::Deepseek4,
                    effective: RawParallelRequest::new(1, 1, 2),
                },
                crate::construction_recorder::RecordedEvent::EnteredConstructor {
                    axis: ParallelAxis::Ep,
                },
            ]
        );
    }

    /// Dense EP (LFM2Dense + Ep=2) normalises to Single, but Single for
    /// LFM2Dense is Planned — `admit_path` refuses before `load_admitted`.
    /// The recorder must be EMPTY — proof that no construction boundary
    /// was ever entered.
    #[test]
    fn dense_ep_refusal_never_enters_construction() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "rec_lfm2_ep.hfq", 11, &lfm2_cfg(0, "lfm2"), &[]);
        let err = admit_path(path.to_str().unwrap(), RawParallelRequest::new(1, 1, 2)).unwrap_err();
        assert!(
            err.contains("AXIS-003"),
            "expected AXIS-003 (Planned), got: {err}"
        );

        let events = crate::construction_recorder::take();
        assert_eq!(
            events,
            Vec::<crate::construction_recorder::RecordedEvent>::new()
        );
    }

    /// Qwen35-VL PP (Planned Axis-004) — `admit_path` refuses before
    /// `load_admitted`. The recorder must be EMPTY.
    #[test]
    fn planned_vl_pp_refusal_never_enters_construction() {
        crate::construction_recorder::clear();
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "rec_vl_pp.hfq",
            5,
            &qwen35_vl_cfg(),
            &vl_tensors(),
        );
        let err = admit_path(path.to_str().unwrap(), RawParallelRequest::new(2, 1, 1)).unwrap_err();
        assert!(
            err.contains("AXIS-004"),
            "expected AXIS-004 (Planned), got: {err}"
        );

        let events = crate::construction_recorder::take();
        assert_eq!(
            events,
            Vec::<crate::construction_recorder::RecordedEvent>::new()
        );
    }


}  // end mod cap_001_loader_tests
#[cfg(test)]
mod session_state_tests {
    use super::LoadedModel;

    fn test_tokenizer() -> hipfire_runtime::tokenizer::Tokenizer {
        hipfire_runtime::tokenizer::Tokenizer::from_hf_json(
            r#"{"model":{"vocab":{"a":0},"merges":[]}}"#,
        )
        .expect("test tokenizer")
    }

    #[test]
    fn reset_cpu_clears_active_request_state() {
        let mut model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            Some("template".to_string()),
        );
        model.seq_pos = 23;
        model.conversation_tokens.extend_from_slice(&[1, 2, 3]);
        model.kv_adaptive = Some(hipfire_runtime::kv_adaptive::KvAdaptive::from_preset(
            hipfire_runtime::kv_adaptive::Preset::Balanced,
            128,
            2,
            64,
        ));
        model
            .kv_adaptive
            .as_mut()
            .expect("adaptive state")
            .next_step = 2;

        model.reset_cpu();

        assert_eq!(model.seq_pos, 0);
        assert!(model.conversation_tokens.is_empty());
        assert!(model.prefill_checkpoints.is_empty());
        assert!(model.dflash_checkpoints.is_empty());
        assert_eq!(model.kv_adaptive.as_ref().unwrap().next_step, 0);
    }

    #[test]
    fn cpu_reset_preserves_cache_and_metadata() {
        let mut model = LoadedModel::skeleton(
            5,
            test_tokenizer(),
            128,
            128,
            "model.mq4".to_string(),
            Some("template".to_string()),
        );
        model.seq_pos = 9;
        model.asst_turn_cache.insert(7, vec![11, 12]);
        model.decoded_vocab = Some(std::sync::Arc::new(vec!["token".to_string()]));
        let meta = (
            model.arch_id,
            model.model_path.clone(),
            model.chat_template.clone(),
        );

        model.reset_cpu();

        assert_eq!(model.seq_pos, 0);
        assert!(model.asst_turn_cache.contains_key(&7));
        assert_eq!(model.decoded_vocab.as_deref().map(Vec::len), Some(1));
        assert_eq!(
            (model.arch_id, model.model_path, model.chat_template),
            meta
        );
    }
}
