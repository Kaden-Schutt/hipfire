// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Top-of-DAG model loader. Owns `LoadedModel`, the carrier registry,
//! and `load_model` — the single arch-dispatch point for the daemon.

pub mod batch_staging;
mod carriers;
pub use carriers::*;

/// Speculative-decode build/glue (RAII slot guard now; `DflashSpeculator` +
/// `build_speculator` at Stages 1-2). Lives here at the top of the DAG where
/// both `LoadedModel` and the arch crates are in scope.
pub mod spec_build;

use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_gemma4 as gemma4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_muse_glimmer as glimmer;
use hipfire_arch_qwen35::qwen35::{self};
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode;
use hipfire_runtime::llama;
use hipfire_runtime::llama::{KvCacheExt, KvDims, KvLayers, KvTarget};
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::{SpecEmit, SpecEmitCtx, SpecTargetGuard, Speculator};
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use rdna_compute::Gpu;
use std::any::Any;
use std::path::Path;

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

    /// Declared capabilities for this arch. Default is the conservative
    /// “no capability” set — carriers override to declare what they support.
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps::default()
    }

    /// Per-arch sampling defaults (`temp`, `top_p`, `repeat_penalty`).
    /// Default is the `else` arm of the daemon ladder (`0.3/0.8/1.0`).
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::default()
    }

    /// Grammar `Config` for constrained tool-call decoding.
    ///
    /// The default resolves the two `HIPFIRE_QWEN35_NGRAM_*` operator
    /// tunables and applies to **every** arch, because that is what the
    /// daemon's generic AR path did when it called
    /// `hipfire_arch_qwen35::grammar_config::resolve_qwen35_grammar_config()`
    /// unconditionally. The variable names are historically qwen-scoped; the
    /// behaviour never was. Preserved verbatim rather than "fixed", since
    /// narrowing it to Qwen would silently change decoding for anyone who
    /// sets those variables against another model.
    ///
    /// Resolution lives in `hipfire-arch-qwen35` (its `spec_emit` is the other
    /// caller) and is *called* here rather than duplicated: the loader already
    /// depends on every arch crate, so this is a downward call, and it leaves
    /// exactly one definition of the env contract in the tree.
    fn grammar_config(&self) -> saddle_core::grammar::json::Config {
        hipfire_arch_qwen35::grammar_config::resolve_qwen35_grammar_config()
    }

    /// Borrow this model's spec-decode target out of `state`, arch-erased as a
    /// [`SpecTargetGuard`]. This is the daemon's single dispatch for the
    /// spec-decode path — it then only ever sees `&mut dyn SpecTarget`, never an
    /// arch type. Default (AR-only carriers): `Err` WITHOUT touching `state` —
    /// only an override may `state.take()`.
    fn spec_target_guard<'m>(
        &self,
        _state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
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

    // ── GenDispatch (wave2) — arch-erased generation/bench hooks ──────────
    // These are ADDITIVE ONLY; CarrierPolicy is concurrently adding caps() and
    // sampling_defaults() in the same trait. Do not restructure.

    /// Bench-prefill dispatch for `bench_prefill` (daemon.rs:15793-16008).
    /// The ~215-line `if m.arch_id == 5 { ... } else if ...` selecting a
    /// per-arch prefill path. Each carrier implements its arch's body
    /// verbatim; the daemon resolves `carrier_for(arch_id)` and calls this
    /// once. `None` = not handled (caller falls through), `Some(run_ok)`
    /// = handled. `prefill_err` is set only by the Glimmer path, mirroring
    /// the daemon's mutable capture.
    fn bench_prefill(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        None
    }

    /// Bench-decode prime for the generic Qwen/Glimmer path
    /// (daemon.rs:16238 `prime_error` if arch_id==14 else Qwen). Return
    /// `Some(prime_error)` if handled, `None` if not this carrier's arch.
    /// The outer `Option` is dispatch; the inner `Option<String>` is the
    /// prime error (None = prime succeeded).
    fn bench_decode_prime(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _synthetic: &[u32],
    ) -> Option<Option<String>> {
        None
    }

    /// Bench-decode run for the generic Qwen/Glimmer path
    /// (daemon.rs:16325 `run_ok` if arch_id==14 else Qwen). Return
    /// `Some((run_ok, decode_err))` if handled, `None` otherwise.
    /// `decode_err` captures the first failing iteration's diagnostic for
    /// Glimmer.
    fn bench_decode_run(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _context: usize,
        _iterations: usize,
        _decode_err: &mut Option<String>,
    ) -> Option<bool> {
        None
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

// ─── Typed routing (replaces stringly `c.name() == "..."` predicates) ──────
// Each route is an exact `arch_id` match — no carrier `name()` or broader
// `claims_arch_id` set leaks through. Gemma 22 is deliberately excluded from
// `GenerationEarlyRoute::Gemma4` (arch 22 is a sidecar drafter, never a
// primary generate target). See blocker 1.

/// Continuous-batch staging route. `None` = not batch-capable (fallback to sequential).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousBatchRoute {
    Qwen35,
    Lfm2Moe,
}
/// Exact arch_id -> continuous-batch route. Only qwen35 5|6 admits: LFM2 (11)
/// has no servable batch path (see the lfm2moe carrier caps), so the route
/// refuses it and no batch state is ever allocated. No carrier probing — pure
/// id match. `ContinuousBatchRoute::Lfm2Moe` stays for the staging body, which
/// is now unreachable.
pub fn continuous_batch_route(arch_id: u32) -> Option<ContinuousBatchRoute> {
    match arch_id {
        5 | 6 => Some(ContinuousBatchRoute::Qwen35),
        _ => None,
    }
}

/// Bench-decode redline route. Exhaustive — every arch maps to exactly one
/// variant, so a `match` without wildcard is a compile error when a new
/// bench arch is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchDecodeRoute {
    Deepseek4,
    Lfm2Moe,
    Qwen35,
    MuseGlimmer,
    Unsupported,
}
pub fn bench_decode_route(arch_id: u32) -> BenchDecodeRoute {
    match arch_id {
        9 => BenchDecodeRoute::Deepseek4,
        11 => BenchDecodeRoute::Lfm2Moe,
        5 | 6 => BenchDecodeRoute::Qwen35,
        14 => BenchDecodeRoute::MuseGlimmer,
        _ => BenchDecodeRoute::Unsupported,
    }
}

/// Vision route. `None` = no vision encoder (text-only). The daemon still
/// gates on `has_image`/`has_vl`; this route only selects the per-arch
/// vision implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionRoute {
    DotsOcr,
    QwenVl,
    Lfm2Vl,
    None,
}
pub fn vision_route(arch_id: u32) -> VisionRoute {
    // Declared-capability gate: text-only arches declare `supports_images == false`
    // and must return `None` even if the arch_id table would say otherwise.
    // The table itself cannot be removed: it discriminates *which* vision
    // implementation to run (QwenVl vs DotsOcr vs Lfm2Vl have distinct generate
    // bodies in `hipfire_generate::vision`), not just whether vision is present.
    // Consulting caps here makes the gate declarative without changing behaviour.
    let caps = carrier_for(arch_id).map(|c| c.caps()).unwrap_or_default();
    if !caps.supports_images {
        return VisionRoute::None;
    }
    match arch_id {
        8 => VisionRoute::DotsOcr,
        5 | 6 => VisionRoute::QwenVl,
        11 => VisionRoute::Lfm2Vl,
        _ => VisionRoute::None,
    }
}

/// EP prompt construction route. DeepSeek4 (arch 9) uses the DSML prompt
/// builder; all other EP arches (10 MiniMax, etc.) use Jinja.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpPromptRoute {
    Dsml,
    Jinja,
}
pub fn ep_prompt_route(arch_id: u32) -> EpPromptRoute {
    match arch_id {
        9 => EpPromptRoute::Dsml,
        _ => EpPromptRoute::Jinja,
    }
}

/// EP EOS token selection route. MiniMax (10) carries its own EOS; all others
/// (including DeepSeek4) use `deepseek4_eos_tok`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpEosRoute {
    Deepseek4,
    Minimax,
    Qwen35,
}
pub fn ep_eos_route(arch_id: u32) -> EpEosRoute {
    match arch_id {
        10 => EpEosRoute::Minimax,
        5 | 6 => EpEosRoute::Qwen35,
        _ => EpEosRoute::Deepseek4,
    }
}

/// Early generation short-circuit route (Gemma4 eager, Glimmer eager).
/// `None` = no early short-circuit (fall through to the generic
/// `select_generation_route` / DFlash/spec dispatch). Critically, arch 22
/// (Gemma4 EAGLE drafter sidecar) maps to `None` — only arch 13 is a
/// primary Gemma4 generate target. See blocker 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationEarlyRoute {
    Gemma4,
    MuseGlimmer,
}
pub fn generation_early_route(arch_id: u32) -> Option<GenerationEarlyRoute> {
    match arch_id {
        13 => Some(GenerationEarlyRoute::Gemma4),
        14 => Some(GenerationEarlyRoute::MuseGlimmer),
        _ => None,
    }
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
    &MapleCarrier,
    &Gemma4Carrier,
    &MuseGlimmerCarrier,
];

// ─── Constants ────────────────────────────────────────────────────────

/// Built-in Qwen3.5/3.6 chat template (froggeric/Qwen at HF).
/// Fallback when arch 5/6 has no configured/per-model override and the HFQ
/// lacks an embedded `tokenizer_config.chat_template` (older Qwen3.5/3.6 files).
/// Qwen3.8 ships an official template with `reasoning_effort` and uses that.
const FROGGERIC_QWEN35_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/qwen35-froggeric-v20.jinja");

/// Built-in LFM2.5 chat template.
const LFM2_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/lfm2-liquidai.jinja");

/// Built-in Gemma 4 IT chat template (arch_id=13).
const GEMMA4_TEMPLATE: &str = include_str!("../../hipfire-runtime/templates/gemma-4-it.jinja");

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

// `DdtreeState`, `DflashState`, `load_dflash_state`, and the `DflashSpeculator`
// impl now live in `hipfire_arch_qwen35::dflash_spec` — all qwen35 + runtime
// types, so the loader only constructs and routes them, never owns the DFlash
// mechanics.

// ─── AsstTurnCache ────────────────────────────────────────────────────

/// Per-turn token cache for V4F prefix-cache stability.
pub struct AsstTurnCache {
    cap: Option<usize>,
    map: std::collections::HashMap<u64, hipfire_runtime::prompt_frame::CachedAssistantTurn>,
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

    pub fn get(&mut self, fp: &u64) -> Option<&hipfire_runtime::prompt_frame::CachedAssistantTurn> {
        if self.map.contains_key(fp) {
            self.touch_mru(*fp);
            self.map.get(fp)
        } else {
            None
        }
    }

    pub fn insert(&mut self, fp: u64, turn: hipfire_runtime::prompt_frame::CachedAssistantTurn) {
        if self.map.contains_key(&fp) {
            self.map.insert(fp, turn);
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
        self.map.insert(fp, turn);
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

// `ModelState` was the closed 12-variant enum that stored `LoadedModel.state`.
// It has been replaced by `Option<Box<dyn ArchModel>>` (see `LoadedModel.state`
// below). All per-architecture teardown now lives in each bundle's
// `ArchModel::free_gpu`; there is no central match any more.
/// `ArchModel` for the one bundle the loader defines itself.
///
/// Every other architecture implements the trait in its own crate. Glimmer's
/// bundle type lives here rather than in `hipfire-arch-muse-glimmer`, so the
/// impl has to live here too — the orphan rule leaves no choice. Moving the
/// type into its crate is Phase 2's job; until then this is the honest place.
impl hipfire_runtime::arch_model::ArchModel for MuseGlimmerBundle {
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "muse_glimmer"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        // Glimmer keeps its KV inside GlimmerState rather than owning a
        // runtime `KvCache` directly, so there is nothing to hand back.
        None
    }
    fn reset_session_state(&mut self, _gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        // Mirrors daemon reset arms (main.rs:3339 / 3390): `bundle.reset_session_state()`.
        MuseGlimmerBundle::reset_session_state(self);
        Ok(())
    }
    fn free_gpu(self: Box<Self>, gpu: &mut rdna_compute::Gpu) {
        // This WAS the `ModelState::MuseGlimmer` arm of `unload_model`, moved
        // verbatim, drafter first. Per PR #566: freeing only one of
        // state/weights leaks ~1.3 GB over five load cycles, so both sides
        // are mandatory.
        let b = *self;
        if let Some(drafter) = b.drafter {
            drafter.scratch.free_gpu(gpu);
            drafter.weights.free_gpu(gpu);
        }
        b.state.free_gpu(gpu);
        b.weights.free_gpu(gpu);
    }
}

/// `ArchModel` for the three bundles the loader defines itself.
///
/// `Gemma4Bundle`, `Gemma4LoweredBundle` and `Deepseek4HeterogeneousBundle` are
/// declared here rather than in their arch crates, so — orphan rule — the impls
/// must be here too. Each `free_gpu` mirrors its `unload_model` arm exactly;
/// freeing less leaks, freeing more double-frees.
impl hipfire_runtime::arch_model::ArchModel for Gemma4Bundle {
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "gemma4"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }
    fn reset_session_state(&mut self, _gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        // Mirrors daemon reset arms (main.rs:3336 / 3387): `bundle.state.reset()`.
        self.state.reset();
        Ok(())
    }
    fn free_gpu(self: Box<Self>, gpu: &mut rdna_compute::Gpu) {
        let b = *self;
        if let Some(eagle) = b.eagle {
            eagle.spec_scratch.free(gpu);
            eagle.drafter_scratch.free_gpu(gpu);
            eagle.drafter_weights.free_gpu(gpu);
        }
        b.state.free_gpu(gpu);
        b.weights.free_gpu(gpu);
    }
}

impl hipfire_runtime::arch_model::ArchModel for Gemma4LoweredBundle {
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "gemma4"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        // Two caches (q8 sliding + asym3 full) and no basis for preferring
        // one, so expose neither rather than silently picking.
        None
    }
    fn free_gpu(self: Box<Self>, gpu: &mut rdna_compute::Gpu) {
        let b = *self;
        b.scratch.free_gpu(gpu);
        let _ = b.kv_sliding.free_gpu(gpu);
        let _ = b.kv_full.free_gpu(gpu);
        b.weights.free_gpu(gpu);
    }
}

impl hipfire_runtime::arch_model::ArchModel for Deepseek4HeterogeneousBundle {
    fn dim(&self) -> usize {
        self.model.config.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.model.config.num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "deepseek4"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }
    fn free_gpu(self: Box<Self>, _gpu: &mut rdna_compute::Gpu) {
        // Self-owned two-device transaction: `Drop` frees each resource on its
        // exact owner and drains both pools. Calling anything else here would
        // double-free, which is why this body is a drop and not a sequence.
        drop(self);
    }
}

/// Self-owned gfx1100+dense / gfx1151+routed DeepSeek V4 state. The model
/// owns both HIP devices and tears them down in its `Drop` implementation;
/// the daemon's ordinary single-device `Gpu` must never free these buffers.
pub struct Deepseek4HeterogeneousBundle {
    pub model: hipfire_arch_deepseek4::DeepseekV4HeterogeneousModel,
    pub eos_tok: u32,
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

/// Gemma 4 EAGLE speculative-decode scratch state (arch-22 drafter riding an
/// arch-13 target). Populated only when `LoadCtx::gemma4_drafter_path` is
/// `Some`. The drafter has NO KV cache of its own — it queries the target's
/// KV at a constant position, so the only per-session state is the target's
/// `Gemma4State` plus this scratch. Mirrors the PR's `Gemma4EagleState`.
pub struct Gemma4EagleState {
    pub drafter_config: gemma4::drafter::Gemma4DrafterConfig,
    pub drafter_weights: gemma4::drafter::Gemma4DrafterWeights,
    pub drafter_scratch: gemma4::drafter::Gemma4DrafterScratch,
    /// Reusable seed/draft/verify hidden buffers for `spec_step_gemma4_eagle`,
    /// sized for `draft_len` at load time.
    pub spec_scratch: gemma4::speculative::Gemma4SpecScratch,
    /// Drafts per round (verify block = draft_len + 1).
    pub draft_len: usize,
}

/// Gemma 4 dense text (arch_id=13) GPU bundle — eager dense path. Re-exported
/// from the arch crate, which owns config/weights/state so `impl SpecTarget`
/// can live there when needed. The `eagle` field holds the optional EAGLE
/// drafter state (arch-22) when `gemma4_drafter_path` was supplied; `None`
/// is AR-only. This keeps one `ModelState::Gemma4` variant for the dense
/// path, matching beta's pattern for other arches.
pub struct Gemma4Bundle {
    pub config: gemma4::config::Gemma4Config,
    pub weights: gemma4::gemma4::Gemma4Weights,
    pub state: gemma4::gemma4::Gemma4State,
    pub eos_tok: u32,
    pub eagle: Option<Gemma4EagleState>,
}

/// Lowered / MoE Gemma 4 execution bundle (arch_id=13 26B-A4B + opt-in
/// batched/WMMA dense prefill). Uses `lowered::{Gemma4Config, Gemma4Weights,
/// Gemma4Scratch}` plus TWO `hipfire_runtime::llama::KvCache`s (q8
/// ring-buffered sliding + asym3 full) and `eos_tok`. Mutually exclusive
/// with `Gemma4Bundle` (eager) via the `ModelState` enum — a given
/// `LoadedModel` populates exactly one of the two variants.
/// Chose second `ModelState` variant over enum-inside-bundle to keep
/// `Gemma4Bundle`'s struct shape stable for the existing eager AR path that
/// `Gemma4Generate` already matches as `Some(ModelState::Gemma4(bundle))`.
pub struct Gemma4LoweredBundle {
    pub config: gemma4::lowered::Gemma4Config,
    pub weights: gemma4::lowered::Gemma4Weights,
    pub scratch: gemma4::lowered::Gemma4Scratch,
    pub kv_sliding: llama::KvCache,
    pub kv_full: llama::KvCache,
    pub eos_tok: u32,
}

/// Muse Glimmer 30B dense text (arch_id=14) GPU bundle — eager dense path.
/// Mirrors Gemma4Bundle exactly (per cross-agent contract).
pub struct MuseGlimmerBundle {
    pub config: glimmer::config::GlimmerConfig,
    pub weights: glimmer::glimmer::GlimmerWeights,
    pub state: glimmer::glimmer::GlimmerState,
    pub eos_tok: u32,
    /// Optional DFlash drafter (arch 23, 5-layer diffusion) loaded when
    /// `HIPFIRE_DFLASH_DRAFT` (or `params.draft`) is set. OFF by default
    /// (`dflash_mode=off` forces `ctx.draft_path=None`). When `None`, the
    /// daemon runs AR-only; when `Some`, the target can be driven via the
    /// generic DFlash `Speculator` (see `carriers.rs`).
    pub drafter: Option<GlimmerDrafterBundle>,
    /// Host-side target hidden history for DFlash: per-position concat of
    /// residual hidden at target_layer_ids [1,13,25,37,49] (5*6656=33280 f32/pos)
    /// in that order. Grows by 1 row per committed token; used as
    /// `target_hidden` input to the drafter's encoder.fc. Empty until first
    /// prefill completes. Order matches the concat the drafter's fc expects —
    /// mixing layers silently degrades tau.
    ///
    /// Authoritative only when device hidden capture is disabled. When the
    /// device log is enabled, that log is the source of truth and this vector
    /// is retained solely as an exact host fallback (session rewind must not
    /// mutate it in device mode).
    pub target_hidden_host: Vec<f32>,
}

impl MuseGlimmerBundle {
    /// f32 elements per captured position (`num_extract * hidden`), or `None`
    /// when no DFlash drafter is loaded.
    #[inline]
    pub fn capture_row_elems(&self) -> Option<usize> {
        self.drafter
            .as_ref()
            .map(|d| d.config.num_extract() * d.config.hidden)
    }

    /// Full session reset: target KV cursor + device log metadata, host capture
    /// vector, and drafter absolute history watermarks.
    pub fn reset_session_state(&mut self) {
        self.state.reset();
        self.target_hidden_host.clear();
        if let Some(d) = self.drafter.as_mut() {
            d.scratch.reset_history();
        }
    }

    /// Whether [`Self::rewind_session_to`] can safely land on `target`.
    ///
    /// Rejects `target > state.n_tokens`. Device capture requires an Idle log
    /// that still retains `target` in its ring. Host-backed DFlash requires the
    /// host vector to be an exact whole-row packing of the current
    /// `state.n_tokens` (no gap/partial row) with at least `target` rows. With
    /// no drafter, rewind is target-KV-only and always allowed within range.
    pub fn can_rewind_session_to(&self, target: usize) -> bool {
        if target > self.state.n_tokens {
            return false;
        }
        if self.state.device_hidden_capture_enabled() {
            return self
                .state
                .target_hidden_log()
                .map(|log| log.can_rewind_to(target))
                .unwrap_or(false);
        }
        match self.capture_row_elems() {
            None => true,
            Some(row_elems) => {
                if row_elems == 0 {
                    return false;
                }
                let host_len = self.target_hidden_host.len();
                if host_len % row_elems != 0 {
                    return false;
                }
                let host_rows = host_len / row_elems;
                host_rows >= target && host_rows == self.state.n_tokens
            }
        }
    }

    /// Rewind target KV (+ device log when enabled), host capture (host-backed
    /// DFlash only), and drafter history to `target`.
    ///
    /// All fallible checks run before any mutation. Device mode never touches
    /// `target_hidden_host`. On success: `state.n_tokens == target`, device log
    /// end == target when enabled, host rows == target when host-backed+dflash,
    /// and drafter `kv_abs_end <= target`.
    pub fn rewind_session_to(&mut self, target: usize) -> Result<(), String> {
        if !self.can_rewind_session_to(target) {
            return Err(format!(
                "muse glimmer: cannot rewind session to {target} (n_tokens={}, device_capture={})",
                self.state.n_tokens,
                self.state.device_hidden_capture_enabled()
            ));
        }

        // Compute host truncate length before mutating anything.
        let host_new_len = if self.state.device_hidden_capture_enabled() {
            None
        } else if let Some(row_elems) = self.capture_row_elems() {
            let new_len = target.checked_mul(row_elems).ok_or_else(|| {
                format!(
                    "muse glimmer: host capture truncate overflow target={target} row_elems={row_elems}"
                )
            })?;
            Some(new_len)
        } else {
            None
        };

        glimmer::forward::rollback_to(&mut self.state, target)?;

        if let Some(new_len) = host_new_len {
            self.target_hidden_host.truncate(new_len);
        }
        if let Some(d) = self.drafter.as_mut() {
            d.scratch.rewind_history(target);
        }
        Ok(())
    }

    /// Current target KV cursor (`state.n_tokens`).
    #[inline]
    pub fn n_tokens(&self) -> usize {
        self.state.n_tokens
    }

    /// Whether the device hidden-capture log is installed on the target state.
    #[inline]
    pub fn device_hidden_capture_enabled(&self) -> bool {
        self.state.device_hidden_capture_enabled()
    }

    /// Committed absolute end of the device hidden log, when installed.
    #[inline]
    pub fn device_hidden_log_end(&self) -> Option<usize> {
        self.state
            .target_hidden_log()
            .map(|log| log.committed_abs_end())
    }

    /// Drafter scratch ctx capacity (`None` without a drafter).
    #[inline]
    pub fn drafter_ctx_capacity(&self) -> Option<usize> {
        self.drafter.as_ref().map(|d| d.scratch.ctx_capacity())
    }

    /// Host-backed capture row count when `capture_row_elems` is known and the
    /// host vector is an exact whole-row packing; otherwise `None`.
    #[inline]
    pub fn host_capture_rows(&self) -> Option<usize> {
        let row_elems = self.capture_row_elems()?;
        if row_elems == 0 || self.target_hidden_host.len() % row_elems != 0 {
            return None;
        }
        Some(self.target_hidden_host.len() / row_elems)
    }
}

/// Muse Glimmer DFlash drafter (arch 23) — `muse_glimmer_assistant` 5-layer
/// diffusion draft head (encoder.fc + output_norm_enc, block 16, mask 201818,
/// target_layer_ids [1,13,25,37,49]). No embed/lm_head (uses target's).
/// Stored alongside the arch-14 target when `HIPFIRE_DFLASH_DRAFT` is set.
pub struct GlimmerDrafterBundle {
    pub config: glimmer::drafter::GlimmerDrafterConfig,
    pub weights: glimmer::drafter::GlimmerDrafterWeights,
    /// Scratch sized to `max_seq` (target's ctx capacity). Allocated once at
    /// load time, freed on unload.
    pub scratch: glimmer::drafter::GlimmerDrafterScratch,
}

/// v1 draft length for gemma4 EAGLE (`params.spec`). dl=3 is the validated
/// config.
pub const GEMMA4_EAGLE_DRAFT_LEN: usize = 3;

/// Gate for the `params.spec` knob: draft_len 1..=5 accepted; absent means
/// the validated default (`GEMMA4_EAGLE_DRAFT_LEN` = 3). Mirrors the PR's
/// `gemma4_eagle_spec_len` — refuse-don't-degrade for unvalidated lengths.
pub fn gemma4_eagle_spec_len(spec: Option<u64>) -> Result<usize, String> {
    match spec {
        None => Ok(GEMMA4_EAGLE_DRAFT_LEN),
        Some(n) if (1..=5).contains(&n) => Ok(n as usize),
        Some(n) => Err(format!(
            "gemma4 EAGLE supports spec=1..=5 (draft_len; default 3, dl <= 5              parity-validated on gfx1201); got spec={n} — reload with a              supported spec or drop params.drafter."
        )),
    }
}

/// Env opt-in for the gemma4 batched/WMMA prefill
/// (`HIPFIRE_BATCHED_PREFILL=1` / `HIPFIRE_WMMA_PREFILL=1`).
pub fn gemma4_batched_prefill_optin(_gpu: &Gpu) -> bool {
    gemma4::lowered::batched_prefill_enabled() || gemma4::lowered::wmma_prefill_enabled()
}

// ─── LoadedModel ──────────────────────────────────────────────────────

pub struct LoadedModel {
    pub arch_id: u32,
    pub pp: usize,
    pub pp_gpus: Option<Gpus>,
    pub pp_dn_la_to_device: Option<Vec<u8>>,
    pub ep: Option<EpState>,
    // Shared arch state — every arch lives here as `Box<dyn ArchModel>`.
    // Previously `ModelState::DotsOcr` / `ModelState::Qwen35` etc; now a
    // trait object so adding an architecture does not edit a closed enum.
    // All consumers go through the typed accessors below (`dots_ocr()`,
    // `qwen35()`, etc.) which downcast via `Any`.
    pub state: Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
    // DeepSeek V4 (arch_id=9) single-GPU prefill scratch now lives inside
    // `Deepseek4Bundle.pbs` (moved from here so `LoadedModel` can become arch-free).
    // DeepSeek V4 (arch_id=9) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Ds4), NOT in `state`, so there is no Deepseek4Bundle for EP
    pub deepseek4_eos_tok: u32,
    // MiniMax-M2 (arch_id=10) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Minimax), NOT in `state`, so `minimax()` is None for EP
    // models — the eos must be carried here (mirrors `deepseek4_eos_tok`).
    pub minimax_eos_tok: u32,
    // Qwen3.5/3.6 (arch_id=5|6) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Qwen35), NOT in `state`, so the eos must be carried here
    // (mirrors `deepseek4_eos_tok` / `minimax_eos_tok`).
    pub qwen35_eos_tok: u32,
    // LFM2.5-8B-A1B (arch_id=11) and MiniMax-M2 (arch_id=10) live in
    // `state` as `Box<dyn ArchModel>` so unload teardown is
    // via `ArchModel::free_gpu` (no central match any more).
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
    ) -> Self {
        LoadedModel {
            arch_id,
            pp: 1,
            ep: None,
            pp_gpus: None,
            pp_dn_la_to_device: None,
            state: None,
            deepseek4_eos_tok: 0,
            minimax_eos_tok: 0,
            qwen35_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
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
        }
    }

    /// LFM2.5-MoE bundle if this model is arch_id=11, else None.
    /// Typed accessors for the remaining bundles.
    ///
    /// Six of these already existed; these six close the set so EVERY bundle is
    /// reachable the same way. That uniformity is the point: once every
    /// consumer goes through an accessor instead of destructuring
    /// `ModelState` itself, swapping the storage to `Box<dyn ArchModel>`
    /// changes these bodies and nothing else. Without them, that swap would
    /// have to rewrite ~143 call sites in `hipfire-generate` at the same time
    /// as changing the type — one unreviewable change instead of two safe ones.
    pub fn qwen35(&self) -> Option<&hipfire_arch_qwen35::Qwen35Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>())
    }

    pub fn qwen35_mut(&mut self) -> Option<&mut hipfire_arch_qwen35::Qwen35Bundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>())
    }

    pub fn llama(&self) -> Option<&hipfire_arch_llama::LlamaBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_llama::LlamaBundle>())
    }

    pub fn llama_mut(&mut self) -> Option<&mut hipfire_arch_llama::LlamaBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>())
    }

    pub fn gemma4(&self) -> Option<&Gemma4Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<Gemma4Bundle>())
    }

    pub fn gemma4_mut(&mut self) -> Option<&mut Gemma4Bundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Gemma4Bundle>())
    }

    pub fn gemma4_lowered_mut(&mut self) -> Option<&mut Gemma4LoweredBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Gemma4LoweredBundle>())
    }

    pub fn muse_glimmer(&self) -> Option<&MuseGlimmerBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<MuseGlimmerBundle>())
    }

    pub fn muse_glimmer_mut(&mut self) -> Option<&mut MuseGlimmerBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<MuseGlimmerBundle>())
    }

    pub fn deepseek4_heterogeneous_mut(&mut self) -> Option<&mut Deepseek4HeterogeneousBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Deepseek4HeterogeneousBundle>())
    }

    pub fn lfm2moe(&self) -> Option<&Lfm2MoeBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<Lfm2MoeBundle>())
    }

    pub fn lfm2moe_mut(&mut self) -> Option<&mut Lfm2MoeBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Lfm2MoeBundle>())
    }

    /// MiniMax-M2 bundle if this model is arch_id=10, else None.
    pub fn minimax(&self) -> Option<&MiniMaxBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<MiniMaxBundle>())
    }

    pub fn minimax_mut(&mut self) -> Option<&mut MiniMaxBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<MiniMaxBundle>())
    }

    /// Qwen2 bundle if this model is arch_id=7 (plain qwen2 via `Qwen2Carrier`),
    /// else None.
    pub fn qwen2_mut(&mut self) -> Option<&mut hipfire_arch_qwen2::Qwen2Bundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen2::Qwen2Bundle>())
    }

    /// Cohere2-MoE bundle if this model is arch_id=12, else None.
    pub fn cohere2moe(&self) -> Option<&Cohere2MoeBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<Cohere2MoeBundle>())
    }

    pub fn cohere2moe_mut(&mut self) -> Option<&mut Cohere2MoeBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Cohere2MoeBundle>())
    }

    /// Maple-Preview bundle if this model is arch_id=15, else None.
    pub fn maple(&self) -> Option<&hipfire_arch_maple::MapleBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_maple::MapleBundle>())
    }

    pub fn maple_mut(&mut self) -> Option<&mut hipfire_arch_maple::MapleBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_maple::MapleBundle>())
    }

    /// DeepSeek V4 bundle if this model is a single-GPU arch_id=9, else None.
    /// (EP/pp ds4 keeps its state in `ep` (EpArch::Ds4), so this is None there.)
    pub fn deepseek4(&self) -> Option<&hipfire_arch_deepseek4::Deepseek4Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_deepseek4::Deepseek4Bundle>())
    }

    pub fn deepseek4_mut(&mut self) -> Option<&mut hipfire_arch_deepseek4::Deepseek4Bundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        })
    }

    /// Qwen35-VL vision config if this model is arch_id=5|6 and was loaded
    /// with a vision tower (`model.visual.patch_embed.proj.weight` present),
    /// else None. Text-only Qwen35 returns None.
    pub fn vision_config(&self) -> Option<&qwen35_vl::VisionConfig> {
        self.qwen35().and_then(|b| b.vision_config.as_ref())
    }

    pub fn ack_dims(&self) -> (usize, usize, usize) {
        if let Some(st) = self.state.as_ref() {
            let arch = st.as_ref() as &dyn hipfire_runtime::arch_model::ArchModel;
            return (arch.dim(), arch.n_layers(), arch.vocab_size());
        }
        // EP and dense-TP loads keep the model in `ep`, leaving `state` empty.
        self.ep
            .as_ref()
            .map(|ep| ep.inner.model_dims())
            .unwrap_or((0, 0, 0))
    }

    pub fn vision_weights(&self) -> Option<&qwen35_vl::VisionWeights> {
        self.qwen35().and_then(|b| b.vision_weights.as_ref())
    }

    pub fn vision_config_mut(&mut self) -> Option<&mut qwen35_vl::VisionConfig> {
        self.qwen35_mut().and_then(|b| b.vision_config.as_mut())
    }

    pub fn vision_weights_mut(&mut self) -> Option<&mut qwen35_vl::VisionWeights> {
        self.qwen35_mut().and_then(|b| b.vision_weights.as_mut())
    }

    /// Declared vision capability across ALL carriers that can carry a
    /// tower: qwen35-vl (arch 5/6 bundle field), dots-ocr (arch 8), and
    /// lfm2-vl (arch-11 bundle field). The daemon's image gate consults
    /// this instead of the qwen-typed accessors above.
    pub fn has_vision_encoder(&self) -> bool {
        if self.dots_ocr().is_some() {
            return true;
        }
        if let Some(b) = self.lfm2moe() {
            if b.vision_config.is_some() {
                return true;
            }
        }
        self.qwen35().is_some_and(|b| b.vision_config.is_some())
    }

    /// LFM2-VL (arch 11) projected-vision config + weights, when loaded
    /// from an artifact quantized with `--include-vision`.
    pub fn lfm2_vision(
        &self,
    ) -> Option<(
        &hipfire_arch_lfm2_vl::VisionConfig,
        &hipfire_arch_lfm2_vl::VisionWeights,
    )> {
        let b = self.lfm2moe()?;
        Some((b.vision_config.as_ref()?, b.vision_weights.as_ref()?))
    }

    /// DotsOcr bundle if this model is arch_id=8, else None.
    pub fn dots_ocr(&self) -> Option<&hipfire_arch_dots_ocr::DotsOcrBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_dots_ocr::DotsOcrBundle>())
    }

    pub fn dots_ocr_mut(&mut self) -> Option<&mut hipfire_arch_dots_ocr::DotsOcrBundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        })
    }
    /// Arch-agnostic view of the loaded model, when any is loaded.
    pub fn as_arch_model(&self) -> Option<&dyn hipfire_runtime::arch_model::ArchModel> {
        self.state
            .as_deref()
            .map(|b| b as &dyn hipfire_runtime::arch_model::ArchModel)
    }
    pub fn as_arch_model_mut(&mut self) -> Option<&mut dyn hipfire_runtime::arch_model::ArchModel> {
        self.state
            .as_deref_mut()
            .map(|b| b as &mut dyn hipfire_runtime::arch_model::ArchModel)
    }
    /// pp>1 skeleton — sets the load-bearing multi-GPU fields together so
    /// they cannot be set piecemeal (`pp_gpus`/`pp_dn_la_to_device` are
    /// `.expect()`ed in unload). The per-device `Qwen35ScratchSet` that was
    /// previously the fourth field here now lives inside `Qwen35Bundle`
    /// (`b.pp_scratch_set`) and is freed via `free_gpu_multi` in the same
    /// pp>1 unload arm — co-locating it with the bundle's own `scratch`
    /// guarantees the set cannot be dropped without the bundle (and vice
    /// versa), eliminating the silent-VRAM-leak window a standalone
    /// `LoadedModel.pp_scratch_set: Option<Qwen35ScratchSet>` created.
    pub fn skeleton_pp(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        pp: usize,
        pp_gpus: Gpus,
        pp_dn_la_to_device: Vec<u8>,
    ) -> Self {
        LoadedModel {
            pp,
            pp_gpus: Some(pp_gpus),
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
        /// Exact gfx1201 MQ2R TP3/TP4 batched-prefill scratch, one per rank.
        /// Empty for every other EP route.
        prefill: Vec<hipfire_arch_deepseek4::forward::PrefillBatchScratch>,
    },
    Minimax {
        config: minimax::MiniMaxConfig,
        weights: Vec<minimax::MiniMaxWeights>,
        state: Vec<minimax::MiniMaxState>,
        partials: Vec<rdna_compute::GpuTensor>,
    },
    Qwen35 {
        config: hipfire_arch_qwen35::qwen35::Qwen35Config,
        weights: Vec<hipfire_arch_qwen35::qwen35::Qwen35Weights>,
        batch: Option<hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchEpState>,
    },
    /// Dense Qwen tensor parallelism. Kept separate from `Qwen35`, whose
    /// ownership and scheduling contract is four-rank routed-expert EP.
    Qwen35DenseTp {
        shard: hipfire_runtime::tp_shard::ShardConfig,
        configs: Vec<hipfire_arch_qwen35::qwen35::Qwen35Config>,
        weights: Vec<hipfire_arch_qwen35::qwen35::Qwen35Weights>,
        kv_caches: Vec<llama::KvCache>,
        dn_states: Vec<hipfire_arch_qwen35::qwen35::DeltaNetState>,
        scratches: Vec<hipfire_arch_qwen35::qwen35::Qwen35Scratch>,
    },
}

impl EpArch {
    // A dense-TP rank config keeps global dim/n_layers/vocab: `local_dense_tp_config` narrows only head counts and `hidden_dim`.
    pub fn model_dims(&self) -> (usize, usize, usize) {
        match self {
            EpArch::Ds4 { config, .. } => (
                config.hidden_size,
                config.num_hidden_layers,
                config.vocab_size,
            ),
            EpArch::Minimax { config, .. } => (
                config.hidden_size,
                config.num_hidden_layers,
                config.vocab_size,
            ),
            EpArch::Qwen35 { config, .. } => (config.dim, config.n_layers, config.vocab_size),
            EpArch::Qwen35DenseTp { configs, .. } => configs
                .first()
                .map(|c| (c.dim, c.n_layers, c.vocab_size))
                .unwrap_or((0, 0, 0)),
        }
    }
}

// ─── Helper functions ─────────────────────────────────────────────────

/// Layer 1 (resolved config) + Layer 2 (per-model ~/.hipfire/templates).
fn resolve_chat_template_overrides(model_path: &str) -> Option<String> {
    if let Some(config_path) = hipfire_runtime::config::get().chat_template_file.as_deref() {
        if !config_path.is_empty() {
            match std::fs::read_to_string(config_path) {
                Ok(s) => {
                    eprintln!("[chat_template] using configured template {config_path}");
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
        // Prefer HFQ-embedded tokenizer_config.chat_template (Qwen3.8 official
        // low/medium/xhigh reasoning_effort). Fall back to froggeric for older
        // Qwen3.5/3.6 files that lack one.
        5 | 6 => {
            return Some(qwen35_template_from_embedded(
                hfq.chat_template(),
                model_path,
            ))
        }
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

/// Arch 5/6 template after configured/per-model overrides: embedded HFQ
/// `tokenizer_config.chat_template` when present, else froggeric fallback.
/// Ornith 1.5 basenames get a load-time Qwen3.8-shaped `reasoning_effort`
/// adaptation of the embedded (non-native) parent template; other models and
/// the froggeric missing-template path are left alone.
fn qwen35_template_from_embedded(embedded: Option<String>, model_path: &str) -> String {
    match embedded {
        Some(t) if is_ornith15_artifact(model_path) => adapt_ornith15_embedded_chat_template(t),
        Some(t) => t,
        None => FROGGERIC_QWEN35_TEMPLATE.to_string(),
    }
}

/// Case-insensitive basename match for Ornith 1.5 MQ4/MQ4R artifacts.
/// Scope is exact: filename prefix `ornith-1.5-` or legacy `ornith1.5-`,
/// AND final extension case-insensitively exactly `.mq4` or `.mq4r`.
/// Other extensions (`.mq6`, `.hfq`, `.json`, sidecars) and bare names
/// are out of scope even when the Ornith 1.5 prefix matches.
fn is_ornith15_artifact(model_path: &str) -> bool {
    let basename = std::path::Path::new(model_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let lower = basename.to_ascii_lowercase();
    let prefix_ok = lower.starts_with("ornith-1.5-") || lower.starts_with("ornith1.5-");
    if !prefix_ok {
        return false;
    }
    // Path::extension is the final suffix after the last `.`.
    let ext = std::path::Path::new(basename)
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    ext == "mq4" || ext == "mq4r"
}

/// Qwen3.8 reasoning_effort system-prompt block (official chat_template.jinja).
const QWEN38_REASONING_BLOCK: &str = "\
{%- set reasoning_instructions = '' %}\n\
{%- if enable_thinking is undefined or enable_thinking is true %}\n\
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n\
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n\
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\n\
    {%- endif %}\n\
    {%- if resolved_reasoning_effort == 'xhigh' %}\n\
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}\n\
    {%- elif resolved_reasoning_effort == 'low' %}\n\
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}\n\
    {%- endif %}\n\
{%- endif %}\n";

/// Unique marker: tools gate opening the system tools block.
const ORNITH_TOOLS_GATE: &str = "{%- if tools and tools is iterable and tools is not mapping %}";

/// Unique compound marker: tools system open immediately followed by the
/// `# Tools` header (no reasoning_instructions yet).
const ORNITH_TOOLS_SYSTEM_OPEN: &str = concat!(
    "{%- if tools and tools is iterable and tools is not mapping %}\n",
    "    {{- '<|im_start|>system\\n' }}\n",
    "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}"
);

/// Replacement: same tools open, with non-empty reasoning_instructions
/// prepended into the system turn (Qwen3.8 structure).
const ORNITH_TOOLS_SYSTEM_OPEN_ADAPTED: &str = concat!(
    "{%- if tools and tools is iterable and tools is not mapping %}\n",
    "    {{- '<|im_start|>system\\n' }}\n",
    "    {%- if reasoning_instructions %}\n",
    "        {{- reasoning_instructions + '\\n\\n' }}\n",
    "    {%- endif %}\n",
    "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}"
);

/// Unique marker: no-tools branch that only emits a system turn when a
/// system message exists (no effort injection, no no-system path).
const ORNITH_NO_TOOLS_SYSTEM_BRANCH: &str = concat!(
    "{%- else %}\n",
    "    {%- if messages[0].role == 'system' %}\n",
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
    "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n",
    "    {%- endif %}\n",
    "{%- endif %}"
);

/// Replacement: Qwen3.8 system/no-system logic with reasoning_instructions.
/// Double space before `+ content` matches the official Qwen3.8 template.
const ORNITH_NO_TOOLS_SYSTEM_BRANCH_ADAPTED: &str = concat!(
    "{%- else %}\n",
    "    {%- if messages[0].role == 'system' %}\n",
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
    "        {%- if content %}\n",
    "            {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n' }}\n",
    "        {%- elif reasoning_instructions %}\n",
    "            {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n",
    "        {%- endif %}\n",
    "    {%- elif reasoning_instructions %}\n",
    "        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n",
    "    {%- endif %}\n",
    "{%- endif %}"
);

/// Load-time adapter for Ornith 1.5 embedded Qwen3.5/3.6 templates: inject the
/// official Qwen3.8 `reasoning_effort` low/medium/xhigh system-prompt contract
/// via exact-marker rewrite. Already-native templates and marker drift are
/// returned unchanged so `probe_effort_capability` stays honest.
fn adapt_ornith15_embedded_chat_template(template: String) -> String {
    if template.contains("reasoning_effort") {
        return template;
    }
    match try_adapt_ornith15_qwen35_to_effort_native(&template) {
        Ok(adapted) => adapted,
        Err(reason) => {
            eprintln!(
                "[chat_template] ornith-1.5 reasoning_effort adapter skipped ({reason}); keeping embedded template"
            );
            template
        }
    }
}

/// Atomic exact-marker rewrite. Preflights every required unique marker, then
/// applies all three substitutions. Any missing/duplicate marker → Err so the
/// caller returns the original string.
fn try_adapt_ornith15_qwen35_to_effort_native(template: &str) -> Result<String, String> {
    let gate_n = template.matches(ORNITH_TOOLS_GATE).count();
    if gate_n != 1 {
        return Err(format!("expected 1 tools gate marker, found {gate_n}"));
    }
    let tools_open_n = template.matches(ORNITH_TOOLS_SYSTEM_OPEN).count();
    if tools_open_n != 1 {
        return Err(format!(
            "expected 1 tools system-open marker, found {tools_open_n}"
        ));
    }
    let no_tools_n = template.matches(ORNITH_NO_TOOLS_SYSTEM_BRANCH).count();
    if no_tools_n != 1 {
        return Err(format!(
            "expected 1 no-tools system branch marker, found {no_tools_n}"
        ));
    }

    // Insert the reasoning block immediately before the tools gate.
    let mut out = template.replacen(
        ORNITH_TOOLS_GATE,
        &(QWEN38_REASONING_BLOCK.to_string() + ORNITH_TOOLS_GATE),
        1,
    );
    // Prepend non-empty instruction into the tools system turn.
    out = out.replacen(
        ORNITH_TOOLS_SYSTEM_OPEN,
        ORNITH_TOOLS_SYSTEM_OPEN_ADAPTED,
        1,
    );
    // Replace the no-tools system/no-system branch.
    out = out.replacen(
        ORNITH_NO_TOOLS_SYSTEM_BRANCH,
        ORNITH_NO_TOOLS_SYSTEM_BRANCH_ADAPTED,
        1,
    );

    // Honesty post-conditions: effort vocabulary + both instruction literals +
    // tools/system/no-system insertion branches present; original no-tools
    // emission gone.
    let has_effort_default = out.contains("reasoning_effort|default('xhigh')");
    let has_rungs = out.contains("('xhigh', 'medium', 'low')");
    let has_low = out.contains(
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
    );
    let has_xhigh = out.contains(
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    );
    // Jinja source stores `\n` as backslash+n inside quotes.
    let has_tools_prepend = out.contains("reasoning_instructions + '\\n\\n'");
    let has_no_system = out.contains("{%- elif reasoning_instructions %}");
    let original_no_tools_gone = !out.contains(ORNITH_NO_TOOLS_SYSTEM_BRANCH);
    if !(has_effort_default
        && has_rungs
        && has_low
        && has_xhigh
        && has_tools_prepend
        && has_no_system
        && original_no_tools_gone)
    {
        return Err("post-condition failed after marker rewrite".into());
    }
    Ok(out)
}

/// Rewrite the Onyx/Harmony chat template for Muse Glimmer (arch 14) so
/// its tool accessors match the flat `prompt_frame::ToolCall` (`tc.name`,
/// `tc.arguments`) and so a spliced cached tool body (`tc.rendered_body`)
/// replaces the regenerated ATEM text. Validates substitution counts and
/// fails loudly on upstream drift (a silent miss would revive the present
/// bug where the whole render raises and falls back to a bare unframed
/// prompt).
pub fn rewrite_muse_glimmer_onyx_template(template: &str) -> Result<String, String> {
    // Validate and count upstream accessor forms before mutating.
    let name_count = template.matches("tc.function.name").count();
    if name_count != 3 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 3 occurrences of `tc.function.name`, found {name_count}; upstream template drift?"
        ));
    }
    let args_count = template.matches("tc.function.arguments").count();
    if args_count != 1 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 1 occurrence of `tc.function.arguments`, found {args_count}; upstream template drift?"
        ));
    }
    let header = "{%- macro render_atem(tc) -%}";
    let header_count = template.matches(header).count();
    if header_count != 1 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 1 `render_atem` header, found {header_count}; upstream template drift?"
        ));
    }
    let tail = "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endmacro -%}";
    let tail_count = template.matches(tail).count();
    if tail_count != 2 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 2 ATEM tails, found {tail_count}; upstream template drift?"
        ));
    }

    // Flat accessors.
    let mut out = template
        .replace("tc.function.name", "tc.name")
        .replace("tc.function.arguments", "tc.arguments");

    // Verbatim splice branch: if a cached body was spliced as `rendered_body`,
    // emit it directly, otherwise render the ATEM XML.
    out = out.replacen(
        header,
        &format!(
            "{}{}",
            header,
            "{%- if tc.rendered_body is defined and tc.rendered_body -%}{{- tc.rendered_body -}}{%- else -%}"
        ),
        1,
    );
    out = out.replacen(
        tail,
        "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endif -%}{%- endmacro -%}",
        1,
    );

    // Post-condition: no leftover nested accessor, branch present.
    if out.contains("tc.function.") {
        return Err(
            "muse_glimmer Onyx template rewrite: leftover `tc.function.` after rewrite".into(),
        );
    }
    if !out.contains("tc.rendered_body") {
        return Err(
            "muse_glimmer Onyx template rewrite: missing `tc.rendered_body` branch after rewrite"
                .into(),
        );
    }
    Ok(out)
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
    activation_gate: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
    let mut base = EvictionCtx::new(
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
    if let Some(gate) = activation_gate {
        base.set_activation_gate(gate);
    }
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
    vision_weights: Option<qwen35_vl::VisionWeights>,
) -> Result<LoadedModel, String> {
    // ── Eviction (only hard-error stage before publish) ────────────
    // Built before long-lived borrows so rollback can move `bundle`.
    let activation_gate = bundle
        .kv_adaptive
        .as_ref()
        .and_then(|adaptive| adaptive.eviction_gate());
    let eviction = match build_qwen35_eviction(&bundle.config, physical_cap, activation_gate, ctx) {
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

    // Adaptive KV cannot combine with generic (DSpark/DFlash/n-gram/MTP via
    // build_speculator) drafters. Suppress before any GPU drafter alloc.
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
                                            0.5,
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
                                    eprintln!("  qwen35: DSpark sidecar {p:?} has no dspark_* metadata — skipping");
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
            &bundle.weights,
            // Exact plain Q8 KV (not asym/fwht variants). Matches design:
            // flags on the already-built cache — never re-resolve the string.
            {
                let kv = &bundle.kv_cache;
                kv.quant_q8
                    && !kv.quant_fwht
                    && !kv.quant_asym2
                    && !kv.quant_asym3
                    && !kv.quant_asym4
                    && matches!(kv.v_mode, llama::VMode::Q8)
            },
            // finish_qwen35_load is the single-GPU carrier path.
            true,
            // Fail-closed: adaptive KV starts FWHT4 and tier-switches at runtime.
            // Upstream also suppresses DFlash when adaptive is Some; this keeps
            // admission honest if a future path reaches load_dflash_state.
            bundle.kv_adaptive.is_some(),
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
    // ── qwen35 MTP head (single resolver: bundled .mq4-mtp trailer then sibling .mtp sidecar) ──
    // Precedence: DSpark > DFlash > MTP > n-gram. Gate: only arch 5/6, no adaptive/eviction,
    // and typed mtp != off. Bundled first then sidecar `.mtp`; physical_cap is the KV
    // window. On missing/failure errors when forced on (mtp=on), auto logs/falls back.
    let mtp: Option<hipfire_arch_qwen35::mtp_head::Qwen35MtpHead> = if adaptive_blocks_generic_spec
        || eviction.is_some()
        || !matches!(arch_id, 5 | 6)
        || ctx.spec.mtp == Some(false)
        || dflash.is_some()
        || dspark_speculator.is_some()
    {
        None
    } else {
        let trunk_path = Path::new(ctx.path);
        let mut head_opt: Option<hipfire_arch_qwen35::mtp_head::Qwen35MtpHead> = None;
        let mut load_err: Option<String> = None;
        match hipfire_arch_qwen35::mtp_head::load_mtp_head_bundled(
            trunk_path,
            ctx.gpu,
            physical_cap,
        ) {
            Ok(Some(h)) => {
                eprintln!(
                    "  MTP head loaded (bundled .mq4-mtp): n_embd={} vocab={}",
                    h.config.n_embd, h.config.vocab_size
                );
                head_opt = Some(h);
            }
            Ok(None) => {
                let sidecar = trunk_path.with_extension("mtp");
                if sidecar.exists() {
                    match hipfire_arch_qwen35::mtp_head::load_mtp_head(
                        &sidecar,
                        ctx.gpu,
                        physical_cap,
                    ) {
                        Ok(h) => {
                            eprintln!(
                                "  MTP head loaded (sidecar {}): n_embd={} vocab={}",
                                sidecar.display(),
                                h.config.n_embd,
                                h.config.vocab_size
                            );
                            head_opt = Some(h);
                        }
                        Err(e) => {
                            load_err =
                                Some(format!("sidecar {} load failed: {e}", sidecar.display()));
                        }
                    }
                }
            }
            Err(e) => {
                load_err = Some(format!("bundled trailer load failed: {e}"));
                let sidecar = trunk_path.with_extension("mtp");
                if sidecar.exists() {
                    match hipfire_arch_qwen35::mtp_head::load_mtp_head(
                        &sidecar,
                        ctx.gpu,
                        physical_cap,
                    ) {
                        Ok(h) => {
                            eprintln!(
                                "  MTP head loaded (sidecar {} after bundled error): n_embd={} vocab={}",
                                sidecar.display(),
                                h.config.n_embd,
                                h.config.vocab_size
                            );
                            head_opt = Some(h);
                            load_err = None;
                        }
                        Err(e2) => {
                            load_err =
                                Some(format!("bundled: {e}; sidecar {}: {e2}", sidecar.display()));
                        }
                    }
                }
            }
        }
        if head_opt.is_none() {
            if ctx.spec.mtp == Some(true) {
                return Err(rollback_unfinished_qwen35(
                    format!(
                        "MTP head required (mtp=on) but not found: {}",
                        load_err.unwrap_or_else(
                            || "no bundled trailer or .mtp sidecar found".to_string()
                        )
                    ),
                    bundle,
                    vision_weights,
                    ctx.gpu,
                ));
            }
            if let Some(err) = load_err {
                eprintln!("  MTP head load failed: {err} — falling back to AR/n-gram");
            }
        }
        head_opt
    };
    let mtp_present = mtp.is_some();
    // Pick the arch-generic speculator: a loaded DFlash draft → DflashSpeculator,
    // else an MTP head → MtpSpeculator<Qwen35MtpDrafter>, else (opt-in) the
    // model-free n-gram drafter. `eviction` is borrowed (not moved) here so it is
    // still available for the struct literal below; `config`/`dn_state` are
    // borrowed only for the n-gram arm's scratch construction. `None` ⇒ AR-only.
    // DSpark wins over DFlash/MTP/n-gram when its sidecar loaded.
    let speculator = if adaptive_blocks_generic_spec {
        None
    } else {
        dspark_speculator.or_else(|| {
            crate::spec_build::build_speculator(
                arch_id,
                dflash,
                mtp,
                eviction.is_none(),
                physical_cap,
                ctx.spec,
            )
        })
    };

    // Move adaptive controller out of the bundle before parking the rest in
    // `Box<dyn ArchModel>`. LoadedModel.kv_adaptive is the runtime home for downshift hooks.
    let mut bundle = bundle;
    bundle.vision_config = vision_config;
    bundle.vision_weights = vision_weights;
    let kv_adaptive = bundle.kv_adaptive.take();
    let state: Option<Box<dyn hipfire_runtime::arch_model::ArchModel>> = Some(Box::new(bundle));
    let mut model = LoadedModel {
        state,
        eviction,
        speculator,
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
    model.mtp_weights_present = mtp_present;
    Ok(model)
}

// ─── Main public API ──────────────────────────────────────────────────

/// gfx11 + gfx12 targets with WMMA-backed DFlash batched lm_head GEMM paths.
fn is_dflash_lm_head_wmma_arch(gpu_arch: &str) -> bool {
    matches!(
        gpu_arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
    )
}

/// Pre-allocation predicate: whether a target lm_head quant_type is admitted
/// for DFlash on `gpu_arch`. Always admits qt 3/6/13; admits legacy qt17 and
/// V2 qt44/47/48/49/50 on the gfx11+gfx12 WMMA set only.
fn dflash_lm_head_quant_supported(lm_qt: Option<u8>, gpu_arch: &str) -> bool {
    match lm_qt {
        Some(3 | 6 | 13) => true,
        Some(17 | 44 | 47 | 48 | 49 | 50) => is_dflash_lm_head_wmma_arch(gpu_arch),
        _ => false,
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
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    load_model_with_kv_backend(
        path,
        max_seq,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        draft_path,
        kv_mode_override,
        None,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        spec,
        gpu,
    )
}

/// Load a model with an optional per-load KV storage backend override.
#[allow(clippy::too_many_arguments)]
pub fn load_model_with_kv_backend(
    path: &str,
    max_seq: usize,
    deepseek4_experts_per_token: Option<usize>,
    deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
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
    let src = ModelSource::from_path(path)?;
    let kv_backend_raw = kv_backend_override.unwrap_or("contiguous");
    let kv_backend: KvBackend = kv_backend_raw.parse().map_err(|err| format!("{err}"))?;

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
            let arch_is_gfx11 = is_dflash_lm_head_wmma_arch(gpu.arch.as_str());
            let supported = dflash_lm_head_quant_supported(lm_qt, gpu.arch.as_str());
            if !supported {
                let qt_desc = match lm_qt {
                    Some(qt) => format!("quant_type={qt}"),
                    None => "no lm_head/embed_tokens tensor found at any known name".to_string(),
                };
                return Err(format!(
                    "DFlash draft requested but target lm_head {} is not \
                     supported by speculative.rs's batched GEMM paths on this arch \
                     ({}). Supported: Q8_0 (qt=3), HFQ4G256 (qt=6), MQ4G256 \
                     (qt=13) always; MQ3G256 (qt=17) and MQ4/6/5/3/2G256V2 \
                     (qt=44/47/48/49/50) on gfx11+gfx12 WMMA. Other dtypes fall \
                     through to unsupported per-row verification. Reload without \
                     a draft or use a supported target.",
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

    let mut ctx = LoadCtx {
        path,
        max_seq,
        deepseek4_compute_placement,
        deepseek4_experts_per_token,
        draft_path,
        kv_mode_override,
        kv_backend,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        spec,
        gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: GEMMA4_EAGLE_DRAFT_LEN,
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
    if kv_backend == KvBackend::Vmm
        && !matches!(carrier.name(), "qwen35" | "deepseek4" | "muse_glimmer")
    {
        return Err(format!(
            "KV backend 'vmm' currently supports qwen3.5, deepseek4, and Muse Glimmer only (selected carrier: {})",
            carrier.name()
        ));
    }
    // The allowlist above gates on CARRIER, which let `vmm` + `pp>1` through:
    // qwen35 is allowlisted, so a pipeline-parallel Qwen3.5 load passed it. But
    // VMM is strictly per-device — `ensure_vmm_ready_for_load` takes a single
    // `&mut Gpu`, `multi_gpu.rs` has no VMM path at all, and the pp>1 load tail
    // never mentions it. Refusing here, BEFORE any allocation, beats letting a
    // single-device KV backend be half-applied to a model spread across devices.
    if kv_backend == KvBackend::Vmm && ctx.pp > 1 {
        return Err(
            "KV backend 'vmm' is single-device and does not support pipeline parallelism (pp>1); \
             use a different kv_cache backend or load with pp=1"
                .to_string(),
        );
    }
    let mut result = carrier.load(src, &mut ctx)?;
    if result.pp > 1 && result.pp_gpus.is_none() {
        return Err("pp>1 LoadedModel missing pp_gpus — carrier bug".into());
    }
    // Apply the author-recommended sampling extracted pre-allocation (see above).
    // Do NOT reparse the .hfq metadata here: a post-allocation / pre-capture parse
    // is the gfx12 hipGraph-replay regression root-caused above.
    if let Some(rec) = rec_sampling {
        result.rec_temperature = rec.temperature;
        result.rec_top_p = rec.top_p;
        result.rec_top_k = rec.top_k.map(|k| k as f32);
    }
    Ok(result)
}

/// Load a model with Gemma4 EAGLE drafter support. Mirrors
/// `load_model_with_kv_backend` but threads `gemma4_drafter_path` /
/// `gemma4_draft_len` (params.drafter / params.spec) separately from
/// `draft_path` (Qwen DFlash) so a DFlash .hfq can never be routed into the
/// EAGLE loader by accident. When `gemma4_drafter_path` is `None` the Gemma4
/// eager path is AR-only.
#[allow(clippy::too_many_arguments)]
pub fn load_model_with_gemma4_drafter(
    path: &str,
    max_seq: usize,
    deepseek4_experts_per_token: Option<usize>,
    deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
    draft_path: Option<&str>,
    gemma4_drafter_path: Option<&str>,
    gemma4_draft_len: usize,
    kv_mode_override: Option<&str>,
    kv_backend_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    // Validate draft_len early (refuse-don't-degrade, same rule as daemon).
    let _ = gemma4_eagle_spec_len(Some(gemma4_draft_len as u64))
        .map_err(|e| format!("gemma4 drafter: {e}"))?;
    ensure_vmm_ready_for_load(gpu)?;
    let src = ModelSource::from_path(path)?;
    let kv_backend_raw = kv_backend_override.unwrap_or("contiguous");
    let kv_backend: KvBackend = kv_backend_raw.parse().map_err(|err| format!("{err}"))?;
    let rec_sampling = match &src {
        ModelSource::Hfq(hfq) => hfq.recommended_sampling(),
        _ => None,
    };
    // Reuse DFlash quant checks for draft_path (unchanged)
    if draft_path.is_some() {
        if let ModelSource::Hfq(ref hfq) = src {
            let lm_qt = hfq
                .tensor_data("lm_head.weight")
                .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
                .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
                .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
                .map(|(info, _)| info.quant_type);
            let supported = dflash_lm_head_quant_supported(lm_qt, gpu.arch.as_str());
            if !supported {
                let qt_desc = match lm_qt {
                    Some(qt) => format!("quant_type={qt}"),
                    None => "no lm_head/embed_tokens tensor found".to_string(),
                };
                return Err(format!(
                    "DFlash draft requested but target lm_head {} is not supported \
                     on gfx11+gfx12 WMMA ({}).",
                    qt_desc, gpu.arch
                ));
            }
        }
    }
    let mut ctx = LoadCtx {
        path,
        max_seq,
        deepseek4_compute_placement,
        deepseek4_experts_per_token,
        draft_path,
        kv_mode_override,
        kv_backend,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        spec,
        gpu,
        gemma4_drafter_path,
        gemma4_draft_len,
    };
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
    if kv_backend == KvBackend::Vmm
        && !matches!(carrier.name(), "qwen35" | "deepseek4" | "muse_glimmer")
    {
        return Err(format!(
            "KV backend 'vmm' currently supports qwen3.5, deepseek4, and Muse Glimmer only (selected carrier: {})",
            carrier.name()
        ));
    }
    // The allowlist above gates on CARRIER, which let `vmm` + `pp>1` through:
    // qwen35 is allowlisted, so a pipeline-parallel Qwen3.5 load passed it. But
    // VMM is strictly per-device — `ensure_vmm_ready_for_load` takes a single
    // `&mut Gpu`, `multi_gpu.rs` has no VMM path at all, and the pp>1 load tail
    // never mentions it. Refusing here, BEFORE any allocation, beats letting a
    // single-device KV backend be half-applied to a model spread across devices.
    if kv_backend == KvBackend::Vmm && ctx.pp > 1 {
        return Err(
            "KV backend 'vmm' is single-device and does not support pipeline parallelism (pp>1); \
             use a different kv_cache backend or load with pp=1"
                .to_string(),
        );
    }
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

fn load_cohere2moe(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
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
        state: Some(Box::new(Cohere2MoeBundle {
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
    prefill: Vec<deepseek4::forward::PrefillBatchScratch>,
}

impl Ds4EpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            state: Vec::new(),
            partials: Vec::new(),
            prefill: Vec::new(),
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
        Vec<deepseek4::forward::PrefillBatchScratch>,
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let state = std::mem::take(&mut self.state);
        let partials = std::mem::take(&mut self.partials);
        let prefill = std::mem::take(&mut self.prefill);
        (gpus, weights, state, partials, prefill)
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
        for (r, pbs) in self.prefill.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                pbs.free_gpu(dev);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Staging guard for the MiniMax EP load — mirror of `Ds4EpStaging` with the
/// MiniMax weight/state types.
struct MinimaxEpStaging {
    gpus: Option<Gpus>,
    weights: Vec<minimax::MiniMaxWeights>,
    state: Vec<minimax::MiniMaxState>,
    partials: Vec<rdna_compute::GpuTensor>,
}

impl MinimaxEpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            state: Vec::new(),
            partials: Vec::new(),
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
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let state = std::mem::take(&mut self.state);
        let partials = std::mem::take(&mut self.partials);
        (gpus, weights, state, partials)
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
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
    }
}
/// Staging guard for the Qwen3.5 EP load — mirror of `MinimaxEpStaging` with
/// the Qwen weight types. Owns the `Gpus` orchestrator plus per-rank weights
/// as they are built up. If the load fails mid-way (`?` early return, or the
/// `HIPFIRE_EP_FAIL_RANK` fault), `Drop` explicitly frees every rank's VRAM
/// on its owning device and drains each device's pool, so a failed EP load
/// leaks NO VRAM and publishes NO partial object. On success the caller calls
/// `into_parts()` to disarm the guard and move ownership into the `LoadedModel`.
struct Qwen35EpStaging {
    gpus: Option<Gpus>,
    weights: Vec<qwen35::Qwen35Weights>,
}

impl Qwen35EpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
        }
    }
    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staging gpus taken")
    }
    fn into_parts(mut self) -> (Gpus, Vec<qwen35::Qwen35Weights>) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        (gpus, weights)
    }
}

impl Drop for Qwen35EpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        eprintln!(
            "[loader] EP qwen35 load failed — freeing {} partially-loaded rank(s) (no VRAM leak)",
            self.weights.len()
        );
        for (r, w) in self.weights.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                w.free_gpu(dev);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Transactional owner for dense Qwen TP construction. Every vector is kept
/// rank-aligned; an early return frees only the objects that were published
/// into the guard, on their owning device.
struct Qwen35DenseTpStaging {
    gpus: Option<Gpus>,
    weights: Vec<qwen35::Qwen35Weights>,
    kv_caches: Vec<llama::KvCache>,
    dn_states: Vec<qwen35::DeltaNetState>,
    scratches: Vec<qwen35::Qwen35Scratch>,
}

impl Qwen35DenseTpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            kv_caches: Vec::new(),
            dn_states: Vec::new(),
            scratches: Vec::new(),
        }
    }

    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("dense TP staging gpus taken")
    }

    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<qwen35::Qwen35Weights>,
        Vec<llama::KvCache>,
        Vec<qwen35::DeltaNetState>,
        Vec<qwen35::Qwen35Scratch>,
    ) {
        (
            self.gpus.take().expect("dense TP into_parts called twice"),
            std::mem::take(&mut self.weights),
            std::mem::take(&mut self.kv_caches),
            std::mem::take(&mut self.dn_states),
            std::mem::take(&mut self.scratches),
        )
    }
}

impl Drop for Qwen35DenseTpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        for (rank, scratch) in self.scratches.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                let _ = scratch.free_gpu(gpu);
            }
        }
        for (rank, state) in self.dn_states.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                state.free_gpu(gpu);
            }
        }
        for (rank, kv) in self.kv_caches.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                let _ = kv.free_gpu(gpu);
            }
        }
        for (rank, weights) in self.weights.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                weights.free_gpu(gpu);
            }
        }
        // Best-effort: reclaim peer-rooted reduce scratch reserved pre-enable_peer_all
        // so a failed dense-TP load cannot strand VRAM after peer mapping.
        let _ = gpus.free_peer_reduce_scratch();
        for gpu in &mut gpus.devices {
            let _ = gpu.bind_thread();
            gpu.invalidate_weight_caches();
            gpu.invalidate_graph_state();
            gpu.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Admission refusal for Qwen3.5-MoE under expert-parallel load (#683 family).
/// Pure so the contract is unit-testable: any Qwen3.5 config with routed
/// experts (`num_experts > 0`, i.e. arch 6 and any mis-stamped arch 5) has no
/// EP serve path — `generate_ep` routes arch 6 at the dense-TP server, which
/// only accepts `EpArch::Qwen35DenseTp`. Refuse here, before `Gpus::init_tp`
/// and the per-rank weight upload, instead of after a full 4-rank load.
/// Dense Qwen3.5 (`num_experts == 0`) is unaffected and keeps its EP path.
pub fn qwen35_ep_moe_refusal(arch_id: u32, num_experts: usize) -> Option<String> {
    if num_experts > 0 {
        Some(format!(
            "Qwen3.5-MoE (arch_id={arch_id}) has no EP serve path; use TP or single-GPU"
        ))
    } else {
        None
    }
}

/// Message constructor shared by [`ep_admission`] and the per-entry match
/// backstops so the refusal text cannot drift between the two.
fn ep_unsupported_arch_message(arch_id: u32) -> String {
    format!(
        "EP not supported for arch_id={arch_id} (expected 5|6 for Qwen3.5, 9 for DeepSeek V4 or 10 for MiniMax)"
    )
}

/// EP load admission by arch_id. Only archs with an `EpArch` variant may enter
/// expert-parallel load (9/DeepSeek4, 10/MiniMax, 5|6/Qwen3.5); LFM2 (11),
/// Cohere2 (12) and anything else must fail here — right after the host-side
/// HFQ probe, before any device init — otherwise they would reach
/// `generate_ep` with no correct server. Pure so the contract is
/// unit-testable; both `load_model_ep_*` entries call it before dispatching.
pub fn ep_admission(arch_id: u32) -> Result<(), String> {
    match arch_id {
        5 | 6 | 9 | 10 => Ok(()),
        id => Err(ep_unsupported_arch_message(id)),
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
pub fn load_model_ep(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    load_model_ep_with_compressor_cache(
        path,
        max_seq,
        tp,
        hipfire_config::Deepseek4CompressorCache::F32,
    )
}

/// DeepSeek V4 stores its long-lived compressor cache as F32 or F16 only; it
/// has no block-quantised path. Rather than reject the quantised selectors,
/// map every sub-F16 request up to F16 — the nearest storage the model
/// actually implements, and the one whose intent ("smaller than F32") the
/// caller expressed. Redirecting rather than failing keeps `q8`, the historic
/// DS4 default, and the `asym`/`fwht`/`turbo` families usable, and it is a
/// widening in precision terms: F16 carries more of the value than any of
/// them would have.
///
/// This is deliberately not silent. A caller that asked for `q8` and received
/// F16 storage is running a different configuration from the F32 golden, so
/// the redirect is reported once at resolve time.
fn resolve_deepseek4_compressor_cache_kv_mode(
    kv_mode: Option<&str>,
) -> Result<hipfire_config::Deepseek4CompressorCache, String> {
    let raw = kv_mode.unwrap_or("f32").to_ascii_lowercase();
    match raw.as_str() {
        "" | "auto" | "f32" => Ok(hipfire_config::Deepseek4CompressorCache::F32),
        "f16" => Ok(hipfire_config::Deepseek4CompressorCache::F16),
        "q8" | "asym2" | "asym3" | "asym4" | "fwht2" | "fwht3" | "fwht4" | "turbo" | "turbo3"
        | "turbo4" => {
            eprintln!(
                "[deepseek4] kv_cache={raw} has no DeepSeek V4 implementation; \
                 using the F16 compressor cache instead (nearest supported storage \
                 below F32). Pass --kv f32 for the golden configuration."
            );
            Ok(hipfire_config::Deepseek4CompressorCache::F16)
        }
        other => Err(format!(
            "DeepSeek V4 kv_cache={other} is not recognised; use f32 (golden/default) or f16"
        )),
    }
}

/// EP load using the standard user-facing KV selector. DeepSeek V4 maps f32/f16
/// to its long-lived compressor-cache storage; MiniMax retains its historical
/// EP behavior and does not inherit DeepSeek-specific policy.
pub fn load_model_ep_with_kv_mode(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_backend: Option<&str>,
    state_quant: Option<&str>,
) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    // Admission: refuse archs with no `EpArch` before any per-arch device init.
    ep_admission(hfq.arch_id)?;
    let kv_backend_raw = kv_backend.unwrap_or("contiguous");
    let kv_backend_kind: KvBackend = kv_backend_raw.parse().map_err(|err| format!("{err}"))?;
    match hfq.arch_id {
        9 => load_model_ep_ds4(
            path,
            max_seq,
            tp,
            resolve_deepseek4_compressor_cache_kv_mode(kv_mode)?,
        ),
        10 if kv_backend_kind == KvBackend::Vmm => {
            Err(format!("KV backend '{kv_backend_raw}' requires tp=1"))
        }
        10 => load_model_ep_minimax(path, max_seq, tp),
        5 | 6 if kv_backend_kind == KvBackend::Vmm => {
            Err(format!("KV backend '{kv_backend_raw}' requires tp=1"))
        }
        5 | 6 => load_model_ep_qwen35(path, max_seq, tp, kv_mode, kv_backend, state_quant),
        // Backstop: `ep_admission` above already refused these; route through the
        // shared constructor (not `unreachable!`) so the refusal survives a
        // future edit that drops the early call.
        id => Err(ep_unsupported_arch_message(id)),
    }
}

/// Expert-parallel load with an explicit DeepSeek V4 compressor-cache storage
/// policy. The compatibility wrapper above deliberately retains the historical
/// F32 route.
pub fn load_model_ep_with_compressor_cache(
    path: &str,
    max_seq: usize,
    tp: usize,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    // Admission: refuse archs with no `EpArch` before any per-arch device init.
    ep_admission(hfq.arch_id)?;
    match hfq.arch_id {
        9 => load_model_ep_ds4(path, max_seq, tp, compressor_cache),
        10 if compressor_cache == hipfire_config::Deepseek4CompressorCache::F32 => {
            load_model_ep_minimax(path, max_seq, tp)
        }
        10 => Err("DeepSeek V4 compressor-cache storage cannot be applied to MiniMax".to_string()),
        5 | 6 if compressor_cache == hipfire_config::Deepseek4CompressorCache::F32 => {
            load_model_ep_qwen35(path, max_seq, tp, None, None, None)
        }
        5 | 6 => Err("DeepSeek V4 compressor-cache storage cannot be applied to Qwen3.5".to_string()),
        // Backstop: `ep_admission` above already refused these; route through the
        // shared constructor (not `unreachable!`) so the refusal survives a
        // future edit that drops the early call.
        id => Err(ep_unsupported_arch_message(id)),
    }
}

fn load_model_ep_ds4(
    path: &str,
    max_seq: usize,
    tp: usize,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let mut config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
    // The EP/TP serve path has no speculative verifier. Do not auto-discover
    // and replicate an adjacent DSpark sidecar onto every rank when it cannot
    // be selected; ordinary AR must retain that memory for per-rank KV state.
    config.load_dspark = false;
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

    let gpus =
        Gpus::init_tp(tp, config.num_hidden_layers).map_err(|e| format!("init_tp: {e:?}"))?;
    let n = gpus.devices.len();
    if n != tp {
        return Err(format!(
            "init_tp gave {n} devices, expected tp={tp} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    eprintln!("[loader] EP load: tp={tp} arch=ds4 experts={n_exp} (rank r owns e%{tp}==r)");
    let shard = ShardConfig::new_uneven_experts(
        tp,
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
    }
    // Exact-gated gfx1201 MQ2R TP3/TP4 graph + batched-prefill substrate.
    // Allocate every long-lived pointer before enabling peer access so peer
    // mappings and later graph capture observe the final allocation layout.
    let gfx1201_mq2r_tp = matches!(tp, 3 | 4)
        && config.mq2r
        && !config.mq2rxt
        && staging
            .gpus_mut()
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201());
    if gfx1201_mq2r_tp {
        // Keep compressor caches replicated and device-local on the normal
        // path. Peer-reading the block-cyclic experiment was exact but reduced
        // 21K NIAH decode from 41.15 to 11.10 tok/s. Capacity growth therefore
        // remains entirely device-local; F16 storage is what makes 1M feasible.
        // B=1024 is the certified short-context throughput schedule. At a
        // declared long-context ceiling, B=512 gives back about 1.34 GiB/rank
        // while retaining most grouped-MoE occupancy. Preserve that established
        // F32 schedule. An explicitly selected F16 cache may trade prefill
        // throughput for additional physical depth with B=128 at an extreme
        // declared ceiling. Replicated TP3 still tops out below 1M; the smaller
        // batch maximizes its measured F16 capacity without regressing F32.
        const SHORT_CONTEXT_BATCH: usize = 1024;
        const LONG_CONTEXT_BATCH: usize = 512;
        const EXTREME_CONTEXT_BATCH: usize = 128;
        const LONG_CONTEXT_THRESHOLD: usize = 32 * 1024;
        const EXTREME_CONTEXT_THRESHOLD: usize = 256 * 1024;
        let compressor_cache_dtype = match compressor_cache {
            hipfire_config::Deepseek4CompressorCache::F32 => rdna_compute::DType::F32,
            hipfire_config::Deepseek4CompressorCache::F16 => rdna_compute::DType::F16,
        };
        for state in &mut staging.state {
            state.compressor_cache_dtype = compressor_cache_dtype;
        }
        let ep_prefill_max_batch = if compressor_cache
            == hipfire_config::Deepseek4CompressorCache::F16
            && max_seq > EXTREME_CONTEXT_THRESHOLD
        {
            EXTREME_CONTEXT_BATCH
        } else if max_seq > LONG_CONTEXT_THRESHOLD {
            LONG_CONTEXT_BATCH
        } else {
            SHORT_CONTEXT_BATCH
        };
        let projected = deepseek4::forward::PrefillBatchScratch::projected_allocation_bytes(
            &config,
            ep_prefill_max_batch,
        )?;
        for rank in 0..n {
            let dev = &mut staging.gpus_mut().devices[rank];
            dev.bind_thread()
                .map_err(|e| format!("bind gfx1201 TP prefill rank {rank}: {e:?}"))?;
            let (free, _total) = dev
                .hip
                .get_vram_info()
                .map_err(|e| format!("query gfx1201 TP prefill VRAM rank {rank}: {e:?}"))?;
            if free < projected {
                return Err(format!(
                    "gfx1201 TP prefill rank {rank}: need {:.2} GiB scratch, only {:.2} GiB free",
                    projected as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64,
                ));
            }
            let pbs =
                deepseek4::forward::PrefillBatchScratch::new(dev, &config, ep_prefill_max_batch)
                    .map_err(|e| format!("allocate gfx1201 TP prefill rank {rank}: {e}"))?;
            staging.prefill.push(pbs);
        }
        eprintln!(
            "[loader] gfx1201 TP{tp} batched prefill: B={} scratch={:.2} GiB/rank compressor_cache={compressor_cache}",
            ep_prefill_max_batch,
            projected as f64 / (1u64 << 30) as f64,
        );
        staging
            .gpus_mut()
            .prepare_tp_graph_signals(config.num_hidden_layers * 2)
            .map_err(|e| format!("prepare gfx1201 TP graph signals: {e:?}"))?;
    } else if compressor_cache == hipfire_config::Deepseek4CompressorCache::F16 {
        // Single-device MQ2R also carries the F16 compressor cache on any
        // architecture whose kernels can compile it. Admission is the
        // capability predicate rather than a chip list: the two F16 sources
        // select their WMMA fragment layout in-source, so the set that can run
        // them is "wave32 WMMA on RDNA3 or RDNA4", and that fact belongs in
        // arch_caps next to the kernels it describes. Storage stays confined
        // to main_kv_cache and indexer_kv_cache; commit arithmetic remains F32
        // before the single F32-to-F16 store.
        let f16_single = tp <= 1
            && config.mq2r
            && !config.mq2rxt
            && staging
                .gpus_mut()
                .devices
                .iter()
                .all(|device| device.arch_caps.supports_ds4_f16_compressor_cache());
        if !f16_single {
            return Err(format!(
                "DeepSeek V4 compressor_cache=f16 requires MQ2R on TP3/TP4 gfx1201, or single-device MQ2R on an architecture with wave32 WMMA (RDNA3/RDNA4); got tp={tp}, mq2r={}, mq2rxt={}, devices={}",
                config.mq2r,
                config.mq2rxt,
                staging
                    .gpus_mut()
                    .devices
                    .iter()
                    .map(|device| device.arch.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }
    let peer = staging
        .gpus_mut()
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e:?}"))?;
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!("[loader] EP load complete: {n} ranks, peer_access={peer}");
    let (gpus, weights, state, partials, prefill) = staging.into_parts();

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
                prefill,
            },
        }),
        deepseek4_eos_tok: eos_tok,
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

fn load_model_ep_minimax(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
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

    let gpus =
        Gpus::init_tp(tp, config.num_hidden_layers).map_err(|e| format!("init_tp: {e:?}"))?;
    let n = gpus.devices.len();
    if n != tp {
        return Err(format!(
            "init_tp gave {n} devices, expected tp={tp} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    eprintln!("[loader] EP load: tp={tp} arch=minimax experts={n_exp} (rank r owns e%{tp}==r)");
    let shard = ShardConfig::new(
        tp,
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
        let w = minimax::MiniMaxWeights::load(&mut h, &config, dev, Some((&shard, r)))
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
    }
    let peer = staging
        .gpus_mut()
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e:?}"))?;
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!("[loader] EP load complete: {n} ranks, peer_access={peer}");
    let (gpus, weights, state, partials) = staging.into_parts();

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
            },
        }),
        minimax_eos_tok: eos_tok,
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
fn load_model_ep_qwen35(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_backend: Option<&str>,
    state_quant: Option<&str>,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq_probe = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    if hfq_probe.arch_id != 5 && hfq_probe.arch_id != 6 {
        return Err(format!(
            "EP qwen35 requires arch 5 or 6, got {}",
            hfq_probe.arch_id
        ));
    }
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_probe.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq_probe).map_err(|e| format!("qwen35 config: {e}"))?;
    if config.num_experts == 0 {
        drop(hfq_probe);
        return load_model_tp_qwen35_dense(path, max_seq, tp, kv_mode, state_quant);
    }
    // MoE EP: keep existing behavior; dense-only selectors are handled above. Silence unused.
    let _ = (kv_mode, kv_backend, state_quant);
    // Admission (#683): MoE has no EP serve path — refuse before `Gpus::init_tp`
    // (first device init) and the per-rank weight upload, not after a full load.
    if let Some(reason) = qwen35_ep_moe_refusal(hfq_probe.arch_id, config.num_experts) {
        return Err(reason);
    }
    if tp != 4 {
        return Err(format!(
            "EP qwen35 MoE requires tp=4, got tp={tp} (only 4×gfx1201 expert-parallel is supported)"
        ));
    }
    if config.paged_experts {
        return Err("EP qwen35: paged_experts must be false".to_string());
    }
    if config.reap_keep.is_some() {
        return Err("EP qwen35: REAP keep-map incompatible with EP".to_string());
    }
    let arch_id = hfq_probe.arch_id;
    let n_exp = config.num_experts;
    let chat_template = resolve_chat_template(&hfq_probe, path);
    let rec = hfq_probe.recommended_sampling();
    let gpus = Gpus::init_tp(tp, config.n_layers).map_err(|e| format!("init_tp: {e:?}"))?;
    let n = gpus.devices.len();
    if n != tp {
        return Err(format!(
            "init_tp gave {n} devices, expected tp={tp} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    for (idx, dev) in gpus.devices.iter().enumerate() {
        if !dev.arch_caps.is_gfx1201() {
            return Err(format!(
                "EP qwen35 requires all 4×gfx1201, rank {idx} is {}",
                dev.arch.as_str()
            ));
        }
    }
    eprintln!(
        "[loader] EP load: tp={tp} arch=qwen35 experts={n_exp} (rank r owns e%{tp}==r via Stride, replicated KV)"
    );
    let shard = ShardConfig::new(tp, true, n_exp, ExpertAssign::Stride)
        .map_err(|e| format!("ShardConfig: {e:?}"))?;
    let fail_rank = ep_fail_rank();
    let _ = fail_rank;
    let mut staging = Qwen35EpStaging::new(gpus);
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut h = HfqFile::open(Path::new(path)).map_err(|e| format!("reopen rank {r}: {e}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = qwen35::load_weights_ep_rank(&mut h, dev, &config, shard.clone(), r)
            .map_err(|e| format!("shard load rank {r}: {e:?}"))?;
        staging.weights.push(w);
        if fail_rank == Some(r) {
            return Err(format!(
                "HIPFIRE_EP_FAIL_RANK={r}: synthetic qwen35 EP load failure after rank {r} (testing partial-load cleanup)"
            ));
        }
    }
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!(
        "[loader] EP load complete: {n} ranks, peer access deferred until post-batch allocation"
    );
    let (gpus, weights) = staging.into_parts();
    let eos_tok: u32 = {
        let ids = tokenizer.encode("<|im_end|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            config.eos_token
        }
    };
    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Qwen35 {
                config,
                weights,
                batch: None,
            },
        }),
        qwen35_eos_tok: eos_tok,
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

fn load_model_tp_qwen35_dense(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    state_quant: Option<&str>,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("qwen35 config: {e}"))?;
    let shard = ShardConfig::new(tp, false, 0, ExpertAssign::Stride)
        .map_err(|e| format!("dense TP ShardConfig: {e}"))?;
    // Compute static per-rank whole-unit layouts CPU-only before any GPU allocation.
    // Validates GQA/G256 geometry, TP range 2..5, and global coverage contiguously.
    let layouts = qwen35::dense_tp_rank_layouts(&config, &shard)
        .map_err(|e| format!("dense TP layout: {e}"))?;
    // Resolve state quant via canonical parser; dense TP honors Q8/default, FP32, Q4.
    let state_quant_resolved = parse_state_quant(state_quant)?;
    // Resolve KV mode via Qwen policy (contiguous only). Explicit unsupported => fail before GPU init.
    let kv_raw = kv_mode.unwrap_or("");
    let kv_trim = kv_raw.trim();
    let kv_lower = kv_trim.to_ascii_lowercase();
    let kv_mode_resolved = if kv_lower.is_empty() {
        kv_mode::resolve("", &kv_mode::QWEN35_HFQ_POLICY, config.head_dim).mode
    } else {
        let rr = kv_mode::resolve(&kv_lower, &kv_mode::QWEN35_HFQ_POLICY, config.head_dim);
        if rr.warning.is_some() {
            return Err(format!(
                "unsupported kv_mode '{kv_trim}' (expected q8|asym2|asym3|asym4|fwht2|fwht3|fwht4)"
            ));
        }
        rr.mode
    };
    // Preflight weights before GPU allocation (validates qt geometry/blob/sidecar).
    qwen35::preflight_weights_dense_tp(&hfq, &config, &shard)?;
    let configs = layouts
        .iter()
        .map(|layout| qwen35::local_dense_tp_config(&config, layout))
        .collect::<Vec<_>>();
    let arch_id = hfq.arch_id;
    let chat_template = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();
    let eos_tok = {
        let ids = tokenizer.encode("<|im_end|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            config.eos_token
        }
    };
    drop(hfq);

    let gpus = Gpus::init_tp(tp, config.n_layers).map_err(|e| format!("init_tp: {e:?}"))?;
    if gpus.devices.len() != tp {
        return Err(format!(
            "init_tp gave {} devices, expected tp={tp}",
            gpus.devices.len()
        ));
    }
    let mut staging = Qwen35DenseTpStaging::new(gpus);
    for rank in 0..tp {
        staging.gpus_mut().devices[rank]
            .bind_thread()
            .map_err(|e| format!("dense TP bind rank {rank}: {e:?}"))?;
        let mut rank_hfq = HfqFile::open(Path::new(path))
            .map_err(|e| format!("dense TP reopen rank {rank}: {e}"))?;
        let weights = qwen35::load_weights_dense_tp_rank(
            &mut rank_hfq,
            &config,
            &mut staging.gpus_mut().devices[rank],
            &layouts[rank],
        )
        .map_err(|e| format!("dense TP weight load rank {rank}: {e:?}"))?;
        staging.weights.push(weights);

        let local = &configs[rank];
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();
        let dims = KvDims {
            layers: KvLayers::Mask(is_kv_layer),
            n_kv_heads: local.n_kv_heads,
            head_dim: local.head_dim,
            max_seq,
            physical_cap: Some(max_seq),
        };
        let kv = <llama::KvCache as KvCacheExt>::from_mode_with_backend(
            kv_mode_resolved,
            KvBackend::Contiguous,
            KvTarget::Single(&mut staging.gpus_mut().devices[rank]),
            &dims,
        )
        .map_err(|e| format!("dense TP KV rank {rank}: {e:?}"))?;
        staging.kv_caches.push(kv);
        let dn = qwen35::DeltaNetState::new_with_quant(
            &mut staging.gpus_mut().devices[rank],
            local,
            state_quant_resolved,
        )
        .map_err(|e| format!("dense TP DeltaNet rank {rank}: {e:?}"))?;
        staging.dn_states.push(dn);
        let scratch = qwen35::Qwen35Scratch::new_with_kv_max(
            &mut staging.gpus_mut().devices[rank],
            local,
            128,
            max_seq,
        )
        .map_err(|e| format!("dense TP scratch rank {rank}: {e:?}"))?;
        staging.scratches.push(scratch);
    }
    // Probe complete peer topology without mutation. Complete → enable + RCCL;
    // mixed → host-staged allreduce (no peer enable, no device reduce scratch).
    let peer = staging
        .gpus_mut()
        .can_access_peer_all()
        .map_err(|e| format!("dense TP can_access_peer_all: {e:?}"))?;
    if peer {
        let enabled = staging
            .gpus_mut()
            .enable_peer_all()
            .map_err(|e| format!("dense TP enable_peer_all: {e:?}"))?;
        if !enabled {
            return Err(
                "dense TP enable_peer_all returned false after can_access_peer_all".to_string(),
            );
        }
    } else {
        // Mixed P2P topology: select host-staged allreduce; do not enable peers
        // or reserve device reduction scratch.
        eprintln!(
            "[loader] dense qwen TP mixed topology: host-staged allreduce (no peer enable/scratch)"
        );
    }
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("dense TP ensure_rank_streams: {e:?}"))?;
    let (gpus, weights, kv_caches, dn_states, scratches) = staging.into_parts();
    eprintln!("[loader] dense qwen TP load complete: {tp} ranks, peer_access={peer}");

    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Qwen35DenseTp {
                shard,
                configs,
                weights,
                kv_caches,
                dn_states,
                scratches,
            },
        }),
        qwen35_eos_tok: eos_tok,
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
    // EP unload-free. An EP model owns its own `Gpus` (the daemon's single `gpu`
    // is unused for tp>1). Without this branch a SUCCESSFUL EP unload leaked every
    // per-rank weight / state / partial. Free per-rank weights → state → partials
    // on each owning device, invalidate caches + graph state, drain each pool, then
    // drop the `Gpus` (tears down comms + devices). The daemon's `gpu` is untouched.
    // (The `partials` free here is what reclaims the ds4/minimax per-rank dummy
    // all-reduce buffer that would otherwise leak per load/unload cycle.)
    if let Some(ep) = m.ep.take() {
        let EpState { mut gpus, inner } = ep;
        let mut ep_first_err: Option<String> = None;
        match inner {
            EpArch::Ds4 {
                weights,
                state,
                partials,
                prefill,
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
                for (r, pbs) in prefill.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        pbs.free_gpu(dev);
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
            EpArch::Qwen35 { weights, batch, .. } => {
                if let Some(b) = batch {
                    if let Err(e) = b.free_gpu(&mut gpus) {
                        if ep_first_err.is_none() {
                            ep_first_err = Some(e.to_string());
                        }
                    }
                }
                for (r, w) in weights.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        w.free_gpu(dev);
                    } else if ep_first_err.is_none() {
                        ep_first_err = Some(format!("unload qwen35: missing device for rank {r}"));
                    }
                }
            }
            EpArch::Qwen35DenseTp {
                weights,
                kv_caches,
                dn_states,
                scratches,
                ..
            } => {
                for (rank, scratch) in scratches.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        if let Err(e) = scratch.free_gpu(dev) {
                            if ep_first_err.is_none() {
                                ep_first_err = Some(format!(
                                    "unload dense qwen TP scratch rank {rank}: {e:?}"
                                ));
                            }
                        }
                    }
                }
                for (rank, state) in dn_states.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        state.free_gpu(dev);
                    }
                }
                for (rank, kv) in kv_caches.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        let _ = kv.free_gpu(dev);
                    }
                }
                for (rank, weights) in weights.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        weights.free_gpu(dev);
                    }
                }
            }
        }
        // Reclaim unleased peer-rooted collective scratch before device pool
        // teardown. Idempotent across EP variants; fold first error into
        // ep_first_err so unload reports cleanup failure.
        if let Err(e) = gpus.free_peer_reduce_scratch() {
            if ep_first_err.is_none() {
                ep_first_err = Some(format!("unload dense TP peer reduce scratch: {e:?}"));
            }
        }

        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
        if let Some(b) = m.qwen35_mut() {
            if let Some(batch_state) = b.qwen35_decode_batch.take() {
                // Single-GPU Qwen batch is not expected on EP, but free it on the
                // provided single gpu before multi-device teardown to avoid leaking
                // if staging ever leaves residual state.
                let _ = batch_state.free_gpu(gpu);
            }
        }
        let _ = gpu;
        if let Some(err) = ep_first_err {
            return Err(err);
        }
        return Ok(());
        // `gpus` drops here, tearing down comms + devices.
    }
    if m.pp > 1 {
        if let Some(b) = m.qwen35_mut() {
            if let Some(batch_state) = b.qwen35_decode_batch.take() {
                // Single-GPU batch state is not expected for pp>1, but free it on
                // the provided single gpu before multi-device teardown to avoid
                // leaking if a test ever stages it.
                let _ = batch_state.free_gpu(gpu);
            }
        }
        let mut gpus = m.pp_gpus.take().expect("pp>1 must carry pp_gpus");
        if let Some(state) = m.state.take() {
            // multi-GPU resources. Otherwise just drop the box (original match did
            // the same — non-qwen35 pp>1 is unreachable and had an empty arm).
            if let Ok(mut b) =
                (state as Box<dyn Any>).downcast::<hipfire_arch_qwen35::Qwen35Bundle>()
            {
                if let Some(scratch_set) = b.pp_scratch_set.take() {
                    scratch_set.free_gpu_multi(&mut gpus);
                }
                b.kv_cache.free_gpu_multi(&mut gpus);
                let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
                b.dn_state.free_gpu_multi(&mut gpus, &la_to_device);
                b.weights.free_gpu_multi(&mut gpus);
                b.scratch.free_gpu(&mut gpus.devices[0]);
            }
        }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return Ok(());
    }
    // Quiesce retained-PM4 (if any) before freeing any captured owner. Unknown
    // quiescence → keep model quarantined; daemon restart is containment.
    if let Some(spec) = &mut m.speculator {
        if let Err(reason) = spec.quiesce(gpu) {
            eprintln!("dflash verify PM4: unload refused — unknown quiescence: {reason}");
            std::mem::forget(m);
            return Err(format!(
                "dflash verify PM4: unload refused after unknown quiescence ({reason}); \
                 model remains quarantined until process restart"
            ));
        }
    }
    if let Some(spec) = m.speculator.take() {
        // Frees the drafter's GPU buffers (draft weights + scratch) AND its
        // checkpoint ring — a drafter that forgets is a compile error, not a
        // silent VRAM leak. The vestigial `m.dflash_checkpoints` (now always
        // empty) is still drained below for defense-in-depth.
        spec.free(gpu);
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
    for (_, snap) in m.prefill_checkpoints {
        snap.free_gpu(gpu);
    }
    for (_, snap) in m.dflash_checkpoints {
        snap.free_gpu(gpu);
    }
    if let Some(state) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) {
        if let Some(batch_state) = state.qwen35_decode_batch.take() {
            let _ = batch_state.free_gpu(gpu);
        }
    }
    // Free arch-specific GPU state from the carrier bundle via `ArchModel::free_gpu`.
    if let Some(state) = m.state {
        state.free_gpu(gpu);
    }
    // deepseek4 single-GPU scratch now lives inside `Deepseek4Bundle.pbs` and is
    // freed via `ArchModel::free_gpu` above — no separate `m.deepseek4_pbs` block.
    // Qwen35 vision tower is now inside the bundle and freed via
    // `ArchModel::free_gpu` above. Nothing to do here — the old
    // `m.vision_weights` field is gone.
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    // After ordinary frees/pool drain, retry any VMM arenas retained by a
    // failed free_tensor. Success is reported only when none remain.
    note(gpu.ensure_vmm_cleaned().map_err(|e| e.to_string()));
    match first_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
mod ep_admission_tests {
    use super::{ep_admission, qwen35_ep_moe_refusal};

    #[test]
    fn qwen35_moe_ep_refuses_before_load_but_dense_admits() {
        // Arch 6 MoE under EP: refused with the combination named.
        let err =
            qwen35_ep_moe_refusal(6, 128).expect("arch-6 MoE + EP must be refused at admission");
        assert!(err.contains("Qwen3.5-MoE"), "reason names the model: {err}");
        assert!(err.contains("no EP serve path"), "reason: {err}");
        assert!(err.contains('6'), "reason names the arch: {err}");
        // Mis-stamped arch 5 with routed experts: same missing serve path.
        assert!(qwen35_ep_moe_refusal(5, 128).is_some());
        // Adjacent supported: dense Qwen3.5 (either arch id) keeps its EP path.
        assert_eq!(qwen35_ep_moe_refusal(5, 0), None);
        assert_eq!(qwen35_ep_moe_refusal(6, 0), None);
    }

    #[test]
    fn ep_without_eparch_refuses_but_served_archs_admit() {
        // LFM2 (11), Cohere2 (12) and anything else with no `EpArch` variant:
        for arch in [11u32, 12, 13, 0, 99] {
            let err = match ep_admission(arch) {
                Ok(()) => panic!("arch {arch} + EP must refuse"),
                Err(e) => e,
            };
            assert!(err.contains("EP not supported"), "reason: {err}");
            assert!(err.contains(&arch.to_string()), "reason names the arch: {err}");
        }
        // Adjacent supported: DS4, MiniMax and Qwen3.5 keep their EP entries.
        for arch in [5u32, 6, 9, 10] {
            assert!(ep_admission(arch).is_ok(), "arch {arch} + EP must admit");
        }
    }
}

#[cfg(test)]
mod lfm2_batch_admission_tests {
    #[test]
    fn lfm2_continuous_batch_never_admits_but_qwen_still_does() {
        use super::{carrier_for, continuous_batch_route};
        // Arch 11: the route refuses, so staging takes the fallback arm and
        // no Lfm2DecodeBatchState is ever allocated; the caps gate in
        // `is_batch_request_eligible` (and the engine scheduler) agrees.
        assert_eq!(continuous_batch_route(11), None);
        let caps = carrier_for(11).expect("lfm2moe carrier").caps();
        assert!(
            !caps.supports_continuous_batch,
            "lfm2moe caps must stay false while no batch path is servable"
        );
        // Adjacent supported: qwen35 5|6 still admit continuous batching.
        assert!(continuous_batch_route(5).is_some());
        assert!(continuous_batch_route(6).is_some());
        assert!(carrier_for(5).expect("qwen35 carrier").caps().supports_continuous_batch);
    }
}

#[cfg(test)]
mod registry_tests {
    use super::{resolve_deepseek4_compressor_cache_kv_mode, REGISTRY};

    #[test]
    fn deepseek4_kv_mode_is_truthful_and_fail_closed() {
        use hipfire_config::Deepseek4CompressorCache::{F16, F32};

        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(None).unwrap(),
            F32
        );
        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(Some("auto")).unwrap(),
            F32
        );
        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(Some("f32")).unwrap(),
            F32
        );
        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(Some("f16")).unwrap(),
            F16
        );
        // Sub-F16 selectors redirect to F16 rather than failing: DS4 stores
        // its compressor cache as F32 or F16 only, and F16 is the nearest
        // implemented storage below F32 — a widening relative to what these
        // ask for. Only unrecognised strings still fail closed.
        for mode in [
            "q8", "asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4", "turbo", "turbo3", "turbo4",
        ] {
            assert_eq!(
                resolve_deepseek4_compressor_cache_kv_mode(Some(mode)).unwrap(),
                F16,
                "{mode} should redirect to F16"
            );
        }
        let error = resolve_deepseek4_compressor_cache_kv_mode(Some("nonsense")).unwrap_err();
        assert!(error.contains("not recognised"), "{error}");
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

    /// The Onyx/Harmony template exactly as Muse Glimmer's .hfq carries it, checked in so
    /// the rewrite's substitution-count contract is pinned against the real artifact rather
    /// than a hand-written approximation. If upstream republishes the template with a
    /// different accessor shape, THIS TEST fails before a user ever sees a bare unframed
    /// prompt at runtime.
    const ONYX_TEMPLATE: &str =
        include_str!("../../hipfire-runtime/templates/muse-glimmer-onyx.jinja");

    #[test]
    fn muse_glimmer_onyx_template_rewrite() {
        use super::rewrite_muse_glimmer_onyx_template;

        // The real carried template must rewrite cleanly.
        let out = rewrite_muse_glimmer_onyx_template(ONYX_TEMPLATE)
            .expect("the checked-in Onyx template must rewrite");

        // Flat accessors: the runtime's `prompt_frame::ToolCall` is flat, so every
        // `tc.function.*` dereference must be gone or the template's own
        // `raise_exception` fires and the whole render falls back to a bare prompt.
        assert_eq!(out.matches("tc.function.").count(), 0);
        assert_eq!(
            out.matches("tc.name").count(),
            ONYX_TEMPLATE.matches("tc.function.name").count()
        );
        assert_eq!(
            out.matches("tc.arguments").count(),
            ONYX_TEMPLATE.matches("tc.function.arguments").count()
        );

        // The verbatim-splice branch must wrap `render_atem`'s body exactly once, so a
        // cached tool body replaces the regenerated ATEM XML.
        // `{%- if tc.rendered_body is defined and tc.rendered_body -%}{{- tc.rendered_body -}}`
        assert_eq!(out.matches("tc.rendered_body").count(), 3);
        assert!(out.contains("{%- if tc.rendered_body is defined and tc.rendered_body -%}"));
        assert!(out.contains("{%- endif -%}{%- endmacro -%}"));

        // Only `render_atem` is wrapped. `render_tool_defs` ends with a BYTE-IDENTICAL ATEM
        // tail (it embeds a worked example of the call syntax), so a `replace`-all here
        // would inject a stray `{%- endif -%}` into the tool-definition preamble and break
        // every tools-bearing prompt. Pin both halves: exactly one tail stays bare, exactly
        // one is wrapped.
        let bare_tail = "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endmacro -%}";
        let wrapped_tail =
            "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endif -%}{%- endmacro -%}";
        assert_eq!(
            out.matches(bare_tail).count(),
            1,
            "render_tool_defs tail must stay bare"
        );
        assert_eq!(
            out.matches(wrapped_tail).count(),
            1,
            "render_atem tail must be wrapped"
        );

        // Upstream drift must fail LOUDLY rather than silently leaving nested accessors.
        let missing = "{%- macro render_atem(tc) -%}hello{%- endmacro -%}";
        let err = rewrite_muse_glimmer_onyx_template(missing).unwrap_err();
        assert!(
            err.contains("expected 3 occurrences"),
            "unexpected error for drifted template: {err}"
        );
    }

    #[test]
    fn vision_caps_declare_images() {
        // Text-only arches must not declare image support; the two VL
        // arches (Qwen3.5-VL 5/6 and dots.ocr 8) must.
        let qwen35 = REGISTRY
            .iter()
            .find(|c| c.name() == "qwen35")
            .expect("qwen35 carrier missing");
        let dots = REGISTRY
            .iter()
            .find(|c| c.name() == "dots_ocr")
            .expect("dots_ocr carrier missing");
        assert!(
            qwen35.caps().supports_images,
            "qwen35 (arch 5/6) must declare supports_images"
        );
        assert!(
            dots.caps().supports_images,
            "dots_ocr (arch 8) must declare supports_images"
        );
        // VisionRoute must be declarative via caps, not just arch_id.
        assert_eq!(super::vision_route(5), super::VisionRoute::QwenVl);
        assert_eq!(super::vision_route(6), super::VisionRoute::QwenVl);
        assert_eq!(super::vision_route(8), super::VisionRoute::DotsOcr);
        assert_eq!(super::vision_route(11), super::VisionRoute::Lfm2Vl);
        assert_eq!(super::vision_route(0), super::VisionRoute::None);
        assert_eq!(super::vision_route(7), super::VisionRoute::None);
        assert_eq!(super::vision_route(9), super::VisionRoute::None);
        // Text-only carriers must stay false.
        assert!(
            REGISTRY.iter().find(|c| c.name() == "lfm2moe").unwrap().caps().supports_images,
            "lfm2moe (arch 11, lfm2_vl artifacts) must declare supports_images —              tower-less checkpoints still refuse images via has_vision_encoder()"
        );
        for name in [
            "qwen2",
            "llama",
            "deepseek4",
            "minimax",
            "cohere2moe",
            "gemma4",
            "muse_glimmer",
        ] {
            let c = REGISTRY.iter().find(|c| c.name() == name).unwrap();
            assert!(
                !c.caps().supports_images,
                "{name} must not declare supports_images"
            );
        }
    }

    /// Pin every carrier's declared `ArchCaps` AND every arch_id route-table
    /// output. The route tables in `lib.rs` (continuous_batch / bench_decode /
    /// vision / ep_prompt / ep_eos / generation_early) are ROUTING
    /// discriminators — each variant selects a distinct function body — so
    /// they cannot be expressed as a boolean capability and must not drift
    /// silently when a carrier's caps row is edited. If you intentionally
    /// re-route an architecture, update this pin in the same commit.
    #[test]
    fn caps_and_route_tables_are_pinned() {
        use super::{
            bench_decode_route, continuous_batch_route, ep_eos_route, ep_prompt_route,
            generation_early_route, vision_route, BenchDecodeRoute, ContinuousBatchRoute,
            EpEosRoute, EpPromptRoute, GenerationEarlyRoute, VisionRoute,
        };
        use saddle_core::caps::{ArchCaps, DflashKind, ReasoningContract};

        let caps_of = |name: &str| -> ArchCaps {
            REGISTRY
                .iter()
                .find(|c| c.name() == name)
                .unwrap_or_else(|| panic!("{name} carrier missing"))
                .caps()
        };

        // ── Per-carrier declared capabilities (11 architectures) ──
        let text_only = ArchCaps::default();
        assert_eq!(caps_of("qwen2"), text_only);
        assert_eq!(
            caps_of("qwen35"),
            ArchCaps {
                reasoning_contract: ReasoningContract::QwenJinja,
                supports_continuous_batch: true,
                supports_ep_batch: true,
                dflash: Some(DflashKind::Qwen),
                supports_mtp: true,
                spec_excludes_adaptive: true,
                semantic_contract_version: Some(2),
                has_deltanet: true,
                supports_images: true,
            }
        );
        assert_eq!(
            caps_of("llama"),
            ArchCaps {
                dflash: Some(DflashKind::Llama),
                ..text_only
            }
        );
        assert_eq!(
            caps_of("dots_ocr"),
            ArchCaps {
                supports_images: true,
                ..text_only
            }
        );
        assert_eq!(
            caps_of("deepseek4"),
            ArchCaps {
                reasoning_contract: ReasoningContract::DeepSeek4,
                ..text_only
            }
        );
        assert_eq!(caps_of("minimax"), text_only);
        // lfm2moe declares supports_continuous_batch: false — the batch state
        // was allocated and never driven (eligibility always false), so the
        // capability is truthful only when false. Single-stream LFM unaffected.
        assert_eq!(
            caps_of("lfm2moe"),
            ArchCaps {
                supports_images: true,
                ..text_only
            }
        );
        assert_eq!(caps_of("cohere2moe"), text_only);
        assert_eq!(
            caps_of("gemma4"),
            ArchCaps {
                reasoning_contract: ReasoningContract::GemmaBoolean,
                ..text_only
            }
        );
        assert_eq!(
            caps_of("muse_glimmer"),
            ArchCaps {
                reasoning_contract: ReasoningContract::MuseGlimmer,
                semantic_contract_version: Some(2),
                ..text_only
            }
        );

        // ── continuous_batch_route: 5|6 -> Qwen35 only (11/LFM2 refuses) ──
        // The Some/None half duplicates caps().supports_continuous_batch; the
        // variant picks between two distinct staging bodies in batch_staging.
        for id in 0u32..=14 {
            let want = match id {
                5 | 6 => Some(ContinuousBatchRoute::Qwen35),
                _ => None,
            };
            assert_eq!(
                continuous_batch_route(id),
                want,
                "continuous_batch_route({id})"
            );
            // The capability half must stay consistent with the declared cap.
            let declared = super::carrier_for(id)
                .map(|c| c.caps().supports_continuous_batch)
                .unwrap_or(false);
            assert_eq!(
                want.is_some(),
                declared,
                "continuous_batch_route({id}).is_some() disagrees with carrier caps"
            );
        }

        // ── bench_decode_route: 9, 11, 5|6, 14; everything else Unsupported ──
        for id in 0u32..=14 {
            let want = match id {
                9 => BenchDecodeRoute::Deepseek4,
                11 => BenchDecodeRoute::Lfm2Moe,
                5 | 6 => BenchDecodeRoute::Qwen35,
                14 => BenchDecodeRoute::MuseGlimmer,
                _ => BenchDecodeRoute::Unsupported,
            };
            assert_eq!(bench_decode_route(id), want, "bench_decode_route({id})");
        }

        // ── vision_route: 8 -> DotsOcr, 5|6 -> QwenVl, 11 -> Lfm2Vl; gated by supports_images ──
        for id in 0u32..=14 {
            let want = match id {
                8 => VisionRoute::DotsOcr,
                5 | 6 => VisionRoute::QwenVl,
                11 => VisionRoute::Lfm2Vl,
                _ => VisionRoute::None,
            };
            assert_eq!(vision_route(id), want, "vision_route({id})");
        }

        // ── ep_prompt_route: 9 -> Dsml, everything else Jinja ──
        for id in 0u32..=14 {
            let want = if id == 9 {
                EpPromptRoute::Dsml
            } else {
                EpPromptRoute::Jinja
            };
            assert_eq!(ep_prompt_route(id), want, "ep_prompt_route({id})");
        }

        // ── ep_eos_route: Qwen, MiniMax and DeepSeek carry distinct EOS ──
        for id in 0u32..=14 {
            let want = match id {
                5 | 6 => EpEosRoute::Qwen35,
                10 => EpEosRoute::Minimax,
                _ => EpEosRoute::Deepseek4,
            };
            assert_eq!(ep_eos_route(id), want, "ep_eos_route({id})");
        }

        // ── generation_early_route: 13 -> Gemma4, 14 -> MuseGlimmer; arch 22
        // (Gemma4 EAGLE drafter sidecar) must stay None — it is never a
        // primary generate target.
        for id in (0u32..=14).chain([22]) {
            let want = match id {
                13 => Some(GenerationEarlyRoute::Gemma4),
                14 => Some(GenerationEarlyRoute::MuseGlimmer),
                _ => None,
            };
            assert_eq!(
                generation_early_route(id),
                want,
                "generation_early_route({id})"
            );
        }
    }

    #[test]
    fn dflash_lm_head_always_admits_qt3_6_13() {
        use super::dflash_lm_head_quant_supported;
        for qt in [3u8, 6, 13] {
            for arch in [
                "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx942",
            ] {
                assert!(
                    dflash_lm_head_quant_supported(Some(qt), arch),
                    "qt={qt} must always be admitted on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_qt17_on_wmma_set_only() {
        use super::dflash_lm_head_quant_supported;
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                dflash_lm_head_quant_supported(Some(17u8), arch),
                "qt=17 must be admitted on WMMA arch {arch}"
            );
        }
        for arch in ["gfx1030", "gfx942", "gfx1010"] {
            assert!(
                !dflash_lm_head_quant_supported(Some(17), arch),
                "qt=17 must be rejected on non-WMMA arch {arch}"
            );
        }
    }

    #[test]
    fn dflash_lm_head_v2_admitted_on_gfx11_gfx12_wmma() {
        use super::dflash_lm_head_quant_supported;
        let v2 = [44u8, 47, 48, 49, 50];
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201"] {
            for qt in v2 {
                assert!(
                    dflash_lm_head_quant_supported(Some(qt), arch),
                    "V2 qt={qt} must be admitted on {arch}"
                );
            }
        }
        // Full WMMA set beyond the focused four.
        for arch in ["gfx1101", "gfx1102", "gfx1150"] {
            for qt in v2 {
                assert!(
                    dflash_lm_head_quant_supported(Some(qt), arch),
                    "V2 qt={qt} must be admitted on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_v2_rejected_on_gfx1030_and_gfx942() {
        use super::dflash_lm_head_quant_supported;
        for arch in ["gfx1030", "gfx942"] {
            for qt in [44u8, 47, 48, 49, 50] {
                assert!(
                    !dflash_lm_head_quant_supported(Some(qt), arch),
                    "V2 qt={qt} must be rejected on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_rejects_unknown_qt_and_missing() {
        use super::dflash_lm_head_quant_supported;
        // qt45 MQ4CG256, legacy MQ2 qt18, F16, missing tensor — all rejected.
        for qt in [None, Some(18u8), Some(45), Some(16), Some(1)] {
            for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201"] {
                assert!(
                    !dflash_lm_head_quant_supported(qt, arch),
                    "qt={qt:?} must be rejected on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_wmma_arch_set() {
        use super::is_dflash_lm_head_wmma_arch;
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(is_dflash_lm_head_wmma_arch(arch), "{arch}");
        }
        for arch in ["gfx1030", "gfx942", "gfx1010", "gfx908"] {
            assert!(!is_dflash_lm_head_wmma_arch(arch), "{arch}");
        }
    }

    #[test]
    fn qwen_embedded_template_wins_over_froggeric() {
        // Qwen3.8-style embedded template carries official reasoning_effort.
        let embedded = "{% if reasoning_effort %}{{ reasoning_effort }}{% endif %}".to_string();
        let got = super::qwen35_template_from_embedded(Some(embedded.clone()), "qwen3.8-27b.mq4");
        assert_eq!(got, embedded);
        assert!(got.contains("reasoning_effort"));
        assert_ne!(got.as_str(), super::FROGGERIC_QWEN35_TEMPLATE);
    }

    #[test]
    fn qwen_missing_template_uses_froggeric_fallback() {
        // Legacy Qwen3.5/3.6 HFQs without tokenizer_config.chat_template.
        // Ornith path without embedded also keeps plain froggeric (no adapt).
        let got = super::qwen35_template_from_embedded(None, "ornith-1.5-35b-a3b.mq4");
        assert_eq!(got.as_str(), super::FROGGERIC_QWEN35_TEMPLATE);
        let got2 = super::qwen35_template_from_embedded(None, "qwen3.6-35b.mq4");
        assert_eq!(got2.as_str(), super::FROGGERIC_QWEN35_TEMPLATE);
    }

    /// Official Qwen3.8 low / xhigh instruction literals (exact contract).
    const ORNITH_LOW_INSTR: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    const ORNITH_XHIGH_INSTR: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

    /// Minimal Qwen3.5/3.6-shaped skeleton with the exact Ornith/official
    /// tools + no-tools system markers the load-time adapter rewrites, plus
    /// enough surrounding Jinja to actually render tools / system / user-only
    /// branches through `JinjaChatFrame::render_messages`.
    fn ornith_parent_template_skeleton() -> String {
        // This is an independent fixture copied from the embedded Ornith parent
        // template shape. Do not construct it from the production marker
        // constants: the fixture must catch whitespace drift in those markers.
        concat!(
            "{%- macro render_content(content, do_vision_count, is_system_content=false) %}{{- content }}{%- endmacro %}\n",
            "{%- if tools and tools is iterable and tools is not mapping %}\n",
            "    {{- '<|im_start|>system\\n' }}\n",
            "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}\n",
            "    {%- for tool in tools %}\n",
            "        {{- \"\\n\" }}\n",
            "        {{- tool | tojson }}\n",
            "    {%- endfor %}\n",
            "    {{- \"\\n</tools>\" }}\n",
            "    {%- if messages[0].role == 'system' %}\n",
            "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
            "        {%- if content %}\n",
            "            {{- '\\n\\n' + content }}\n",
            "        {%- endif %}\n",
            "    {%- endif %}\n",
            "    {{- '<|im_end|>\\n' }}\n",
            "{%- else %}\n",
            "    {%- if messages[0].role == 'system' %}\n",
            "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
            "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endif %}\n",
            "{%- for message in messages %}\n",
            "    {%- if message.role != 'system' %}\n",
            "        {%- set content = render_content(message.content, false)|trim %}\n",
            "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endfor %}",
        )
        .to_string()
    }

    /// Hermetic GPT-2-BPE tokenizer sufficient for Jinja string rendering
    /// (bos override is empty; encode path is unused by these tests).
    fn test_tokenizer() -> hipfire_runtime::tokenizer::Tokenizer {
        // GPT-2 mode trigger (`Ġ`) + full byte fallback so any short
        // ASCII content round-trips if a future test encodes.
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""system": 2"#.to_string());
        entries.push(r#""user": 3"#.to_string());
        entries.push(r#""assistant": 4"#.to_string());
        entries.push(r#""\n": 5"#.to_string());
        entries.push(r#""Ġ": 6"#.to_string());
        for b in 0u32..=255u32 {
            let ch = byte_to_gpt2_char_test(b as u8);
            let escaped = json_escape(&ch.to_string());
            entries.push(format!(r#""{escaped}": {}"#, 100 + b));
        }
        let vocab = entries.join(", ");
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                "added_tokens": [
                    {{"id": 0, "content": "<|im_start|>", "special": true}},
                    {{"id": 1, "content": "<|im_end|>", "special": true}}
                ]
            }}"#
        );
        hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn byte_to_gpt2_char_test(b: u8) -> char {
        let mut bs: Vec<u32> = Vec::new();
        bs.extend((b'!' as u32)..=(b'~' as u32));
        bs.extend((0xA1u32)..=(0xACu32));
        bs.extend((0xAEu32)..=(0xFFu32));
        let mut cs: Vec<u32> = bs.clone();
        let mut n = 0u32;
        for b2 in 0u32..=255u32 {
            if !bs.contains(&b2) {
                bs.push(b2);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (i, &bv) in bs.iter().enumerate() {
            if bv == b as u32 {
                return char::from_u32(cs[i]).unwrap();
            }
        }
        char::from_u32(b as u32).unwrap()
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    fn test_msg(
        role: hipfire_runtime::prompt_frame::Role,
        content: &str,
    ) -> hipfire_runtime::prompt_frame::Message {
        hipfire_runtime::prompt_frame::Message {
            role,
            content: content.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        }
    }

    fn adapted_ornith_template() -> String {
        super::qwen35_template_from_embedded(
            Some(ornith_parent_template_skeleton()),
            "ornith-1.5-35b-a3b.mq4",
        )
    }

    fn render_ornith(
        template: &str,
        messages: &[hipfire_runtime::prompt_frame::Message],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> String {
        let tok = test_tokenizer();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer: &tok,
            template,
            system: None,
            user: "",
            enable_thinking,
            bos_token: Some(""),
            reasoning_strength: None,
            reasoning_effort,
        };
        frame
            .render_messages(messages, tools, None)
            .unwrap_or_else(|e| panic!("render failed (effort={reasoning_effort:?}): {e}"))
    }

    /// Extract the first `<|im_start|>system ... <|im_end|>` body (no markers).
    fn first_system_body(rendered: &str) -> Option<&str> {
        const START: &str = "<|im_start|>system\n";
        const END: &str = "<|im_end|>";
        let i = rendered.find(START)?;
        let rest = &rendered[i + START.len()..];
        let j = rest.find(END)?;
        Some(&rest[..j])
    }

    #[test]
    fn ornith15_canonical_and_legacy_basenames_adapt() {
        let skeleton = ornith_parent_template_skeleton();
        for path in [
            "/models/ornith-1.5-35b-a3b.mq4",
            "/models/ornith-1.5-35b-a3b.mq4r",
            "/models/ORNITH-1.5-35B-A3B.MQ4",
            "/models/ORNITH-1.5-35B-A3B.MQ4R",
            "/models/ornith1.5-35b-a3b.mq4",
            "ornith1.5-35b-a3b.mq4r",
        ] {
            let got = super::qwen35_template_from_embedded(Some(skeleton.clone()), path);
            assert!(
                got.contains("reasoning_effort|default('xhigh')"),
                "path {path} must adapt: missing default"
            );
            assert!(
                got.contains("('xhigh', 'medium', 'low')"),
                "path {path} must accept exact rungs"
            );
            assert!(
                got.contains(ORNITH_LOW_INSTR),
                "path {path} missing low instruction"
            );
            assert!(
                got.contains(ORNITH_XHIGH_INSTR),
                "path {path} missing xhigh instruction"
            );
            // medium injects nothing: no medium instruction string.
            assert!(
                !got.contains("Reasoning effort is set to medium"),
                "path {path} must not steer medium"
            );
            // tools prepend + system/no-system branches.
            assert!(
                got.contains("reasoning_instructions + '\\n\\n'"),
                "path {path} missing tools/system prepend"
            );
            assert!(
                got.contains("{%- elif reasoning_instructions %}"),
                "path {path} missing no-system branch"
            );
            assert_ne!(got, skeleton, "path {path} must change the template");
        }
    }

    #[test]
    fn unrelated_qwen_basename_is_untouched() {
        let skeleton = ornith_parent_template_skeleton();
        for path in [
            "qwen3.6-35b-a3b.mq4",
            "qwen3.5-397b.mq4",
            "ornith-1.0-35b.mq4",
            "ornith-2.0-35b-a3b.mq4",
            "not-ornith-1.5-35b.mq4",
        ] {
            let got = super::qwen35_template_from_embedded(Some(skeleton.clone()), path);
            assert_eq!(got, skeleton, "path {path} must not adapt");
            assert!(!got.contains("reasoning_effort"), "path {path}");
        }
    }

    #[test]
    fn ornith15_wrong_extension_is_untouched() {
        let skeleton = ornith_parent_template_skeleton();
        for path in [
            "ornith-1.5-35b-a3b.mq6",
            "ornith-1.5-35b-a3b.hfq",
            "ornith-1.5-35b-a3b",
            "ornith-1.5-35b-a3b.json",
            "ornith-1.5-35b-a3b.mq4.sidecar",
            "ornith1.5-35b-a3b.MQ6",
            "/models/ORNITH-1.5-35B-A3B.hfq",
            "ornith-1.5-35b-a3b.mq4r.bak",
        ] {
            let got = super::qwen35_template_from_embedded(Some(skeleton.clone()), path);
            assert_eq!(got, skeleton, "path {path} must not adapt (extension gate)");
            assert!(
                !got.contains("reasoning_effort"),
                "path {path} must not gain effort"
            );
        }
    }

    #[test]
    fn already_native_ornith_template_stays_unchanged() {
        let native = format!(
            "{}{}",
            super::QWEN38_REASONING_BLOCK,
            ornith_parent_template_skeleton()
        );
        assert!(native.contains("reasoning_effort"));
        let got =
            super::qwen35_template_from_embedded(Some(native.clone()), "ornith-1.5-35b-a3b.mq4");
        assert_eq!(got, native, "already-native must not be rewritten");
    }

    #[test]
    fn marker_drift_returns_original_unadapted() {
        // Missing the no-tools branch marker → adapter must no-op.
        let drifted = format!(
            "{}\n{{%- set ns = namespace(x=1) %}}",
            super::ORNITH_TOOLS_SYSTEM_OPEN
        );
        let got =
            super::qwen35_template_from_embedded(Some(drifted.clone()), "ornith-1.5-35b-a3b.mq4");
        assert_eq!(
            got, drifted,
            "drift must keep original so probe stays honest"
        );
        assert!(!got.contains("reasoning_effort|default('xhigh')"));

        // Duplicate tools gate without the compound tools-open → no-op.
        let dup = format!(
            "{gate}\n{gate}\n{no_tools}",
            gate = super::ORNITH_TOOLS_GATE,
            no_tools = super::ORNITH_NO_TOOLS_SYSTEM_BRANCH
        );
        let got2 =
            super::qwen35_template_from_embedded(Some(dup.clone()), "ornith1.5-35b-a3b.mq4r");
        assert_eq!(got2, dup);
    }

    #[test]
    fn ornith15_adapted_jinja_renders_effort_contract() {
        use hipfire_runtime::prompt_frame::Role;

        let template = adapted_ornith_template();
        assert!(
            template.contains("reasoning_effort|default('xhigh')"),
            "adapter must inject effort block"
        );

        let tool = serde_json::json!({
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo",
                "parameters": {"type": "object", "properties": {}}
            }
        });
        let tools = [tool];

        let user_only = [test_msg(Role::User, "hi")];
        let with_system = [test_msg(Role::System, "SYS"), test_msg(Role::User, "hi")];

        // Four effort states × three structural branches.
        let efforts: [(Option<&str>, Option<&str>); 4] = [
            (None, Some(ORNITH_XHIGH_INSTR)),          // undefined → xhigh
            (Some("low"), Some(ORNITH_LOW_INSTR)),     // explicit low
            (Some("medium"), None),                    // medium: no instruction
            (Some("xhigh"), Some(ORNITH_XHIGH_INSTR)), // explicit xhigh
        ];

        for (effort, want_instr) in efforts {
            // 1) tools present
            let r = render_ornith(&template, &with_system, Some(&tools), true, effort);
            let sys = first_system_body(&r).expect("tools path emits system");
            assert!(
                sys.contains("# Tools"),
                "tools branch missing header (effort={effort:?}): {sys:?}"
            );
            match want_instr {
                Some(instr) => {
                    assert!(
                        sys.starts_with(instr),
                        "tools: instruction must lead system turn (effort={effort:?}): {sys:?}"
                    );
                    assert!(
                        sys.contains(&format!("{instr}\n\n# Tools")),
                        "tools: instruction must sit before # Tools (effort={effort:?})"
                    );
                    // system content still appended after tools block
                    assert!(sys.contains("SYS"), "tools: system content lost");
                }
                None => {
                    assert!(
                        !sys.contains("Reasoning effort is set to"),
                        "medium must not inject instruction (tools): {sys:?}"
                    );
                    assert!(
                        sys.starts_with("# Tools"),
                        "medium tools system starts at # Tools: {sys:?}"
                    );
                }
            }

            // 2) no tools + existing system message
            let r = render_ornith(&template, &with_system, None, true, effort);
            let sys = first_system_body(&r).expect("system path emits system");
            match want_instr {
                Some(instr) => {
                    assert_eq!(
                        sys,
                        format!("{instr}\n\nSYS"),
                        "no-tools+system placement (effort={effort:?})"
                    );
                }
                None => {
                    assert_eq!(sys, "SYS", "medium no-tools+system is bare content");
                }
            }
            assert!(
                r.contains("<|im_start|>user\nhi<|im_end|>"),
                "user turn must remain (effort={effort:?})"
            );

            // 3) no tools + user-only (no system message)
            let r = render_ornith(&template, &user_only, None, true, effort);
            match want_instr {
                Some(instr) => {
                    let sys = first_system_body(&r).expect("user-only effort injects system");
                    assert_eq!(
                        sys, instr,
                        "no-tools user-only system body (effort={effort:?})"
                    );
                }
                None => {
                    assert!(
                        first_system_body(&r).is_none(),
                        "medium user-only must not invent a system turn: {r:?}"
                    );
                }
            }
            assert!(
                r.contains("<|im_start|>user\nhi<|im_end|>"),
                "user turn must remain user-only (effort={effort:?})"
            );
        }

        // Undefined effort matches explicit xhigh on every branch.
        for tools_arg in [Some(tools.as_slice()), None] {
            for msgs in [with_system.as_slice(), user_only.as_slice()] {
                // Skip tools+user_only if tools need messages[0] system check —
                // still valid: tools path tolerates non-system first message.
                let undef = render_ornith(&template, msgs, tools_arg, true, None);
                let xhigh = render_ornith(&template, msgs, tools_arg, true, Some("xhigh"));
                assert_eq!(
                    undef,
                    xhigh,
                    "undefined must match xhigh (tools={})",
                    tools_arg.is_some()
                );
            }
        }

        // Thinking disabled: effort instructions absent on every branch.
        for effort in [None, Some("low"), Some("medium"), Some("xhigh")] {
            for (msgs, tools_arg) in [
                (with_system.as_slice(), Some(tools.as_slice())),
                (with_system.as_slice(), None),
                (user_only.as_slice(), None),
            ] {
                let r = render_ornith(&template, msgs, tools_arg, false, effort);
                assert!(
                    !r.contains("Reasoning effort is set to"),
                    "thinking-off must drop effort instr (effort={effort:?}): {r:?}"
                );
            }
        }
    }

    #[test]
    fn ornith15_adapted_template_probe_effort_capability() {
        let template = adapted_ornith_template();
        let tok = test_tokenizer();
        let cap = hipfire_runtime::prompt_frame::probe_effort_capability(&tok, &template);
        assert!(cap.native, "adapted Ornith template must be effort-native");
        assert_eq!(
            cap.supported,
            vec!["low", "medium", "xhigh"],
            "supported rungs must be exactly the Qwen3.8 contract"
        );
    }
}
