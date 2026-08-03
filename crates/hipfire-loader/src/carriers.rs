// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ + safetensors-dir).

use crate::parallel_capability::ModelVariant;
use crate::spec_build::Qwen35SlotGuard;
use crate::{
    finish_qwen35_load, reject_qwen_native_mtp, resolve_chat_template,
    resolve_chat_template_overrides, LoadedModel, ModelState,
};
use crate::{Carrier, CarrierLoadToken};
use hipfire_arch_minimax::{config_from_safetensors, load_weights_from_safetensors, MiniMaxState};
use hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::spec::{InPlaceGuard, SpecEmit, SpecEmitCtx, SpecTargetGuard};
use rdna_compute::Gpu;

fn dspark_lm_head_vocab(draft_vocab_size: usize, asset_vocab_size: usize) -> usize {
    if draft_vocab_size != 0 {
        draft_vocab_size
    } else {
        asset_vocab_size
    }
}

// The ChatML/Hermes per-token emitter (`Qwen35Emit`) is shared by every
// ChatML-family spec arm — qwen35 DFlash AND the llama/qwen2 n-gram paths all
// drive it (they already share qwen35's tool-call grammar). It physically lives
// in the qwen35 crate; the llama/qwen2 carriers wiring it here is composition-
// root glue, not an arch→arch dependency (those arch crates never name it). A
// future cleanup could hoist the emitter + grammar into the runtime.
use hipfire_arch_qwen35::spec_emit::Qwen35Emit;

// ─── Source-only metadata (tokenizer / chat_template / arch_id) ───────
//
// The single seam for the source-varying-but-arch-invariant axis. Adding a
// future source kind (e.g. GGUF) is one new `match` arm here plus the
// irreducible per-arch `(config, weights)` block in each carrier. Lives in
// `hipfire-loader` (not `loader_api`) because it calls `resolve_chat_template`,
// which reads the loader's built-in arch templates.
//
// NOTE: `arch_id` extraction is purely source-varying (`hfq.arch_id` vs
// `source.arch_id()`), so it belongs here — but the *values* live in two
// distinct namespaces (HFQ header ids vs `derive_arch_id` dir ids). A GGUF
// plug-in author must pick the correct namespace, not assume a single one.
struct SourceMeta {
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<String>,
    arch_id: u32,
}

fn resolve_source_meta(src: &ModelSource, path: &str) -> Result<SourceMeta, String> {
    match src {
        ModelSource::Hfq(hfq) => Ok(SourceMeta {
            tokenizer: hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|e| format!("tokenizer not found: {e}"))?,
            chat_template: resolve_chat_template(hfq, path),
            arch_id: hfq.arch_id,
        }),
        ModelSource::Dir(source) => {
            let arch_id = source.arch_id();
            Ok(SourceMeta {
                tokenizer: tokenizer_from_dir(source)?,
                chat_template: resolve_chat_template_overrides(path)
                    .or_else(|| source.chat_template())
                    .or_else(|| arch_default_template(arch_id)),
                arch_id,
            })
        }
    }
}

/// Folds the "no tokenizer.json / failed to parse" block duplicated verbatim
/// in every Dir arm today.
fn tokenizer_from_dir(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
) -> Result<hipfire_runtime::tokenizer::Tokenizer, String> {
    if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))
    } else {
        Err("no tokenizer.json found in model directory".into())
    }
}

/// Returns the first candidate string that tokenizes to exactly one token, or 1.
fn resolve_eos_tok(tokenizer: &hipfire_runtime::tokenizer::Tokenizer, candidates: &[&str]) -> u32 {
    for s in candidates {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            return ids[0];
        }
    }
    1
}

/// Dir-source diagnostic: arch_id + quant_method. One-line call at the top of
/// every Dir-capable carrier's load(). Qwen35 prints a richer variant inline.
fn dir_diag(src: &ModelSource) {
    if let ModelSource::Dir(s) = src {
        let qm = s
            .quant_config()
            .map(|q| q.method.as_str())
            .unwrap_or("none");
        eprintln!("  safetensors arch_id={}, quant_method={qm}", s.arch_id());
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Qwen35ParoLoaderKind {
    DenseManifest,
    LegacyMoe,
}

fn qwen35_paro_loader_kind(num_experts: usize) -> Qwen35ParoLoaderKind {
    if num_experts == 0 {
        Qwen35ParoLoaderKind::DenseManifest
    } else {
        Qwen35ParoLoaderKind::LegacyMoe
    }
}

// ─── Frozen-vs-Legacy selection routing ──────────────────────────────
//
// The carrier's decision after the no-GPU-allocation preflight.  Kept
// as a pure mapping so the selection semantics are testable without
// hardware: Ineligible models MUST route to the Legacy loader (never a
// new load failure); Invalid files MUST fail; an Eligible plan is the
// only input that authorizes Frozen allocation, and nothing after that
// point may fall back.

/// Route decision from a [`Qwen35FrozenPreflight`] selection.
#[expect(
    clippy::large_enum_variant,
    reason = "Legacy carries the ORIGINAL ModelSource to the Legacy loader; the route is a cold, \
             one-shot source-owning decision built before any GPU allocation and consumed \
             immediately by drive_route (no rollback owner, never copied or reused), so boxing \
             would only add a heap allocation and an indirection to a move-only value"
)]
pub(crate) enum Qwen35FrozenRoute {
    /// Frozen allocation authorized by the preflight plan; the plan owns
    /// the source.
    Frozen(hipfire_arch_qwen35::Qwen35FrozenPlan),
    /// Legacy loader fallback; the reason documents the selection and
    /// the ORIGINAL source is returned for the Legacy load.
    Legacy(String, ModelSource),
    /// Neither path can serve the file — the load must fail.
    Fail(String),
}

/// Pure routing of a Frozen preflight selection (no GPU, no I/O).
///
/// * [`Eligible`](hipfire_arch_qwen35::Qwen35FrozenPreflight::Eligible)
///   → [`Frozen`](Qwen35FrozenRoute::Frozen) — the ONLY input that
///   authorizes Frozen allocation; never routes to Legacy.
/// * [`Ineligible`](hipfire_arch_qwen35::Qwen35FrozenPreflight::Ineligible)
///   → [`Legacy`](Qwen35FrozenRoute::Legacy) — the model loads through
///   the existing Legacy path, reusing the exact admitted source; this
///   is a selection, not a failure.
/// * [`Invalid`](hipfire_arch_qwen35::Qwen35FrozenPreflight::Invalid)
///   → [`Fail`](Qwen35FrozenRoute::Fail).
impl std::fmt::Debug for Qwen35FrozenRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen35FrozenRoute::Frozen(_) => f.debug_struct("Frozen").finish(),
            Qwen35FrozenRoute::Legacy(reason, _) => {
                f.debug_struct("Legacy").field("reason", reason).finish()
            }
            Qwen35FrozenRoute::Fail(msg) => f.debug_struct("Fail").field("message", msg).finish(),
        }
    }
}

pub(crate) fn route_frozen_selection(
    selection: hipfire_arch_qwen35::Qwen35FrozenPreflight,
) -> Qwen35FrozenRoute {
    match selection {
        hipfire_arch_qwen35::Qwen35FrozenPreflight::Eligible(plan) => {
            Qwen35FrozenRoute::Frozen(plan)
        }
        hipfire_arch_qwen35::Qwen35FrozenPreflight::Ineligible(reason) => {
            Qwen35FrozenRoute::Legacy(reason.reason().to_string(), reason.into_source())
        }
        hipfire_arch_qwen35::Qwen35FrozenPreflight::Invalid(msg) => Qwen35FrozenRoute::Fail(msg),
    }
}

/// Drive one selected route to completion.  The route decides which ONE
/// of the two load arms runs — an Eligible route can NEVER invoke the
/// Legacy arm, even when the Frozen load fails operationally (the error
/// is surfaced, not re-examined).  Pure routing: the closures contain
/// all GPU work, so the fallback semantics are testable without
/// hardware.
pub(crate) fn drive_route<T>(
    route: Qwen35FrozenRoute,
    mut frozen: impl FnMut(hipfire_arch_qwen35::Qwen35FrozenPlan) -> Result<T, String>,
    mut legacy: impl FnMut(ModelSource) -> Result<T, String>,
) -> Result<T, String> {
    match route {
        Qwen35FrozenRoute::Frozen(plan) => frozen(plan),
        Qwen35FrozenRoute::Legacy(reason, source) => {
            eprintln!("  qwen35 Frozen preflight: {reason} — using Legacy loader");
            legacy(source)
        }
        Qwen35FrozenRoute::Fail(msg) => Err(format!(
            "qwen35: model cannot be loaded by either path: {msg}"
        )),
    }
}

// ─── Qwen2Carrier ────────────────────────────────────────────────────

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    fn name(&self) -> &'static str {
        "qwen2"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut() {
            Some(ModelState::Qwen2(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("qwen2: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // HFQ id 7 and qwen2 safetensors dirs (derive_arch_id → 7). Both route
        // here so the qwen2 Q/K/V `attention_bias=true` biases load (the
        // llama-family Dir loader drops them).
        arch_id == 7
    }
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::Qwen2)
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("qwen2: pipeline-parallel (pp>1) unsupported".into());
        }
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_qwen2::load_qwen2_bundle(src, ctx)?;
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). Qwen2
        // (arch_id=7, e.g. VibeThinker) impls `SpecTarget`, so it can be driven by
        // the arch-generic spec loop with no draft model. `None` ⇒ AR-only.
        let speculator =
            crate::spec_build::build_speculator(meta.arch_id, None, true, ctx.max_seq, ctx.spec);
        Ok(LoadedModel {
            state: Some(ModelState::Qwen2(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
                token.record(),
            )
        })
    }
}

// ─── Qwen35Carrier ───────────────────────────────────────────────────

fn kv_mode_from_ctx(ctx: &LoadCtx) -> String {
    ctx.kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default())
}

fn resolve_kv_mode(
    ctx: &LoadCtx,
    policy: &hipfire_runtime::kv_mode::KvModePolicy,
    head_dim: usize,
) -> hipfire_runtime::kv_mode::KvMode {
    let kv_mode = kv_mode_from_ctx(ctx);
    let hipfire_runtime::kv_mode::ResolveResult { mode, warning } =
        hipfire_runtime::kv_mode::resolve(&kv_mode, policy, head_dim);
    if let Some(w) = warning {
        eprintln!("  KV cache: {w} (site {})", policy.site);
    }
    mode
}

fn arch_default_template(arch_id: u32) -> Option<String> {
    match arch_id {
        5 | 6 => Some(super::FROGGERIC_QWEN35_TEMPLATE.to_string()),
        11 => Some(super::LFM2_TEMPLATE.to_string()),
        _ => None,
    }
}

/// Qwen3.5 pipeline-parallel (pp>1) load. Extracted from the carrier body so
/// the pp>1 multi-GPU tail (`skeleton_pp`) lives in one place; qwen35 is the
/// only carrier with a pp>1 path. KV policy (`QWEN35_PP_POLICY`), DeltaNet
/// quant, and scratch sizing are byte-identical to the previous inline block.
fn load_qwen35_pp(
    mut hfq_file: hipfire_runtime::hfq::HfqFile,
    meta: SourceMeta,
    ctx: &mut LoadCtx,
    token: &CarrierLoadToken,
) -> Result<LoadedModel, String> {
    let pp = ctx.pp;
    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq_file)
        .map_err(|e| format!("failed to read Qwen3.5 config: {e}"))?;
    let mut gpus = match ctx.pp_bands {
        Some(counts) => {
            let sum: usize = counts.iter().sum();
            if sum != config.n_layers {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
                    sum, config.n_layers
                ));
            }
            hipfire_runtime::multi_gpu::Gpus::init_layers(counts).map_err(|e| format!("{e}"))?
        }
        None => hipfire_runtime::multi_gpu::Gpus::init_uniform(pp, config.n_layers)
            .map_err(|e| format!("{e}"))?,
    };
    let layout = hipfire_arch_qwen35::qwen35::Layout::from_gpus(&gpus, config.n_layers);
    let mut hfq_source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq_file, &config);
    let weights =
        hipfire_arch_qwen35::qwen35::load_weights(&mut hfq_source, &mut gpus.devices, &layout)
            .map_err(|e| format!("{e}"))?;
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention)
        .collect();
    let mode = resolve_kv_mode(
        ctx,
        &hipfire_runtime::kv_mode::QWEN35_PP_POLICY,
        config.head_dim,
    );
    let dims = hipfire_runtime::llama::KvDims {
        layers: hipfire_runtime::llama::KvLayers::Mask(is_kv_layer),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: Some(ctx.max_seq),
    };
    let kv = hipfire_runtime::llama::KvCache::from_mode(
        mode,
        hipfire_runtime::llama::KvTarget::Multi(&mut gpus),
        &dims,
    )
    .map_err(|e| format!("{e}"))?;
    let dn_quant =
        crate::parse_state_quant(ctx.state_quant_override).map_err(|e| format!("{e}"))?;
    let (dn, la_to_device) = hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant_multi(
        &mut gpus, &config, dn_quant,
    )
    .map_err(|e| format!("{e}"))?;
    let scratch_set = hipfire_arch_qwen35::qwen35::Qwen35ScratchSet::new_with_kv_max_multi(
        &mut gpus,
        &config,
        2048,
        ctx.max_seq,
    )
    .map_err(|e| format!("{e}"))?;
    let gpu0 = &mut gpus.devices[0];
    let single_scratch = hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(
        gpu0,
        &config,
        2048,
        ctx.max_seq,
    )
    .map_err(|e| format!("{e}"))?;
    let bundle = hipfire_arch_qwen35::Qwen35Bundle {
        config,
        weights,
        scratch: single_scratch,
        kv_cache: kv,
        dn_state: dn,
        mtp_head: None,
        pipeline: Some(hipfire_arch_qwen35::carrier::Qwen35PipelineState {
            scratch_set,
            dn_la_to_device: la_to_device,
        }),
    };
    Ok(LoadedModel {
        state: Some(ModelState::Qwen35(bundle)),
        ..LoadedModel::skeleton_pp(
            meta.arch_id,
            meta.tokenizer,
            ctx.max_seq,
            ctx.max_seq,
            ctx.path.to_string(),
            meta.chat_template,
            ctx.mtp_mode,
            ctx.mtp_k,
            gpus,
            token.record(),
        )
    })
}

pub struct Qwen35Carrier;
impl Carrier for Qwen35Carrier {
    fn name(&self) -> &'static str {
        "qwen35"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        // qwen35 moves its bundle out of `state` into the RAII Qwen35SlotGuard
        // (lazy HfqFile reopen, bundle restored on Drop — the #462 guard).
        Ok(Box::new(Qwen35SlotGuard::take(state, model_path)?))
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 5 = dense (+VL), 6 = MoE — same ids in both namespaces.
        matches!(arch_id, 5 | 6)
    }
    fn classify_parallel_variant(&self, src: &ModelSource) -> Result<ModelVariant, String> {
        match src {
            ModelSource::Hfq(hfq) => match hfq.arch_id {
                6 => Ok(ModelVariant::Qwen35Moe),
                5 => {
                    // Match production loader: use tensor_data (not find_tensor_info)
                    // so that only tensors with physically present data count.
                    let has_vision_tower = hfq
                        .tensor_data("model.visual.patch_embed.proj.weight")
                        .is_some();
                    if has_vision_tower {
                        Ok(ModelVariant::Qwen35Vl)
                    } else {
                        Ok(ModelVariant::Qwen35Dense)
                    }
                }
                other => Err(format!("qwen35: unexpected HFQ arch_id {}", other)),
            },
            ModelSource::Dir(source) => {
                let arch_id = source.arch_id();
                match arch_id {
                    6 => {
                        let _facts = hipfire_arch_qwen35::qwen35::classify_vl(source)?;
                        Ok(ModelVariant::Qwen35Moe)
                    }
                    5 => {
                        // Use the architecture-owned VL helper (no raw metadata
                        // or substring checks in the carrier).
                        let facts = hipfire_arch_qwen35::qwen35::classify_vl(source)?;

                        if facts.is_vl_text {
                            return Ok(ModelVariant::Qwen35Vl);
                        }
                        if source
                            .tensor_info("model.visual.patch_embed.proj.weight")
                            .is_some()
                        {
                            return Ok(ModelVariant::Qwen35Vl);
                        }
                        // Unclassifiable: has VL indicator but composite + tensor
                        // both fail to confirm.
                        if facts.has_vision_config || facts.has_visual_key {
                            return Err("qwen35: VL indicator present but unclassifiable \
                                 (no text_config+vision_config composite and no VL tensor)"
                                .to_string());
                        }
                        Ok(ModelVariant::Qwen35Dense)
                    }
                    other => Err(format!("qwen35 safetensors: unexpected arch_id {}", other)),
                }
            }
        }
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        reject_qwen_native_mtp(ctx.mtp_mode)?;
        // Dir + pp>1: early return before any diagnostics/meta resolution,
        // preserving the original error string and preventing tokenizer work.
        if ctx.pp > 1 {
            if let ModelSource::Dir(..) = &src {
                return Err("qwen35: safetensors + pp>1 unsupported".into());
            }
        }
        // Per-source diagnostics stay at the call site, before resolve_source_meta.
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        match src {
            ModelSource::Hfq(hfq_file) => {
                // ── pp>1 path (pipeline-parallel) — extracted helper ──
                if ctx.pp > 1 {
                    return load_qwen35_pp(hfq_file, meta, ctx, token);
                }

                // ── pp=1 path (single-GPU) ────────────────────
                let physical_cap = if ctx.cask.sidecar.is_some() {
                    let env_override = std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok());
                    let safety = 256usize;
                    let floor = ctx.cask.budget + ctx.cask.beta + 4;
                    let derived = ctx.cask.budget + ctx.cask.beta + safety;
                    env_override.unwrap_or(derived).clamp(floor, ctx.max_seq)
                } else {
                    ctx.max_seq
                };

                // ── COR-006: pre-allocation eviction rejection ──────────
                // Reject impossible budget/beta/cap combinations before GPU
                // allocation so a fat-fingered config fails fast rather than
                // after allocating a full-context KV cache.
                if let Some(ref sidecar) = ctx.cask.sidecar {
                    if ctx.cask.budget == 0 {
                        return Err(format!(
                            "cask budget must be >0, got {} (sidecar={sidecar})",
                            ctx.cask.budget
                        ));
                    }
                    if ctx.cask.budget + ctx.cask.beta + 4 > ctx.max_seq {
                        return Err(format!(
                            "cask budget ({}) + beta ({}) + 4 = {} exceeds max_seq ({}) — eviction can never fire; reduce budget, increase max_seq, or override physical_cap via HIPFIRE_KV_PHYSICAL_CAP",
                            ctx.cask.budget,
                            ctx.cask.beta,
                            ctx.cask.budget + ctx.cask.beta + 4,
                            ctx.max_seq
                        ));
                    }
                }

                // ── COR-006: thread eviction physical_cap into KvCache alloc ──
                if physical_cap < ctx.max_seq {
                    ctx.kv_physical_cap = Some(physical_cap);
                }

                // ── VL probe FIRST (metadata-only) ─────────────────────
                // Index presence + config parse — NO payload read, NO GPU
                // allocation.  The vision WEIGHTS upload happens after the
                // route decision, in the selected path, from the same
                // artifact (the plan-bound source for Frozen, the returned
                // source for Legacy).
                let vision_config = {
                    use hipfire_arch_qwen35_vl::Qwen35Vl;
                    use hipfire_runtime::arch::Architecture;
                    let has_vision = hfq_file
                        .find_tensor_info("model.visual.patch_embed.proj.weight")
                        .is_some();
                    let vc = Qwen35Vl::config_from_hfq(&hfq_file).ok();
                    match vc {
                        Some(vc) if has_vision => {
                            eprintln!(
                                "  VL model: vision encoder (hidden={}, layers={})",
                                vc.hidden_size, vc.num_layers
                            );
                            Some(vc)
                        }
                        _ => None,
                    }
                };

                // ── Frozen-vs-Legacy selection (exact preflight) ────────
                // The no-GPU-allocation preflight consumes the source and
                // decides over the actual admitted Qwen35 MoE variant
                // (arch_id), config, HFQ manifest metadata, and target
                // arch.  Ineligible models fall back to the Legacy loader
                // with the SAME source (never a new load failure); Invalid
                // files fail; once Eligible the Frozen allocation begins
                // and never falls back.
                let flags = hipfire_arch_qwen35::Qwen35MoeLoadFlags::resolve();
                let selection = hipfire_arch_qwen35::preflight_qwen35_frozen(
                    ModelSource::Hfq(hfq_file),
                    &ctx.gpu.arch,
                    ctx.pp == 1,
                    flags,
                );

                // The two load arms share `ctx` through a cell so
                // `drive_route` can stay a pure CPU-testable seam; exactly
                // one arm runs per route.
                let ctx_cell = std::cell::RefCell::new(ctx);
                let result = drive_route(
                    route_frozen_selection(selection),
                    |plan| {
                        let ctx = &mut *ctx_cell.borrow_mut();
                        // The arch-owned planned operation performs the
                        // vision upload from the SEALED plan source
                        // (immutable borrow only) and then the Frozen
                        // load.  On bundle failure it aborts the vision
                        // owner through `vision_abort`; on success it
                        // returns bundle + vision owner.
                        let vision_config_ref = vision_config.as_ref();
                        let vision_closure = |hfq: &HfqFile, gpu: &mut Gpu| {
                            match vision_config_ref {
                                Some(vc) => {
                                    // The underlying loader takes only an
                                    // IMMUTABLE source borrow — the sealed
                                    // plan source cannot be mutated or
                                    // replaced by this closure.
                                    match hipfire_arch_qwen35_vl::qwen35_vl::load_vision_weights(
                                        hfq, vc, gpu,
                                    ) {
                                        Ok(vw) => Ok(Some(vw)),
                                        Err(e) => {
                                            // Preserve the historical
                                            // warn-and-continue semantics.
                                            eprintln!("  VL weight load failed: {e}");
                                            Ok(None)
                                        }
                                    }
                                }
                                None => Ok(None),
                            }
                        };
                        let vision_abort = |vision_owner: Option<VisionWeights>, gpu: &mut Gpu| {
                            if let Some(vw) = vision_owner {
                                vw.free_gpu(gpu);
                            }
                        };
                        // Frozen path: returns Qwen35LoadError on failure,
                        // which the carrier handles by freeing retained owners
                        // and enqueuing backlog entries for any that persist.
                        // No legacy fallback after this point.
                        match hipfire_arch_qwen35::load_qwen35_bundle_frozen_planned(
                            plan,
                            ctx,
                            vision_closure,
                            vision_abort,
                        ) {
                            Ok(outcome) => Ok((outcome.bundle, outcome.vision)),
                            Err(load_err) => {
                                let (msg, frozen_retained, common_cleanup) =
                                    load_err.try_free(ctx.gpu);
                                let n_frozen = frozen_retained.len();
                                let has_common = common_cleanup.is_some();
                                let domain = *ctx.gpu.allocation_domain_id();
                                for fail in frozen_retained {
                                    crate::retain_qwen_cleanup(
                                        domain,
                                        crate::QwenBacklogEntry::Cleanup(
                                            hipfire_arch_qwen35::qwen35::Qwen35CleanupFailure::from_frozen(
                                                fail,
                                            ),
                                        ),
                                    );
                                }
                                if let Some(cf) = common_cleanup {
                                    crate::retain_qwen_cleanup(
                                        domain,
                                        crate::QwenBacklogEntry::Cleanup(cf),
                                    );
                                }
                                let detail = if n_frozen > 0 || has_common {
                                    format!(
                                        " ({} frozen + {} common retained in backlog)",
                                        n_frozen,
                                        if has_common { 1 } else { 0 }
                                    )
                                } else {
                                    String::new()
                                };
                                Err(format!("qwen35 Frozen MoE load: {msg}{detail}"))
                            }
                        }
                    },
                    |source| {
                        let ctx = &mut *ctx_cell.borrow_mut();
                        let ModelSource::Hfq(mut hfq) = source else {
                            unreachable!("preflight routes only HFQ sources to Legacy")
                        };
                        // Vision upload AFTER the route decision, from the
                        // returned source.
                        let mut vision_weights = match &vision_config {
                            Some(vc) => {
                                use hipfire_runtime::arch::Architecture;
                                hipfire_arch_qwen35_vl::Qwen35Vl::load_weights(
                                    &mut hfq, vc, ctx.gpu,
                                )
                                .map_err(|e| eprintln!("  VL weight load failed: {e}"))
                                .ok()
                            }
                            None => None,
                        };
                        match hipfire_arch_qwen35::load_qwen35_bundle(ModelSource::Hfq(hfq), ctx) {
                            Ok(b) => Ok((b, vision_weights)),
                            Err(e) => {
                                if let Some(vw) = vision_weights.take() {
                                    vw.free_gpu(ctx.gpu);
                                }
                                Err(e)
                            }
                        }
                    },
                )?;
                let (bundle, vision_weights) = result;
                let ctx = ctx_cell.into_inner();
                finish_qwen35_load(
                    bundle,
                    meta.tokenizer,
                    physical_cap,
                    meta.arch_id,
                    meta.chat_template,
                    ctx,
                    vision_config,
                    vision_weights,
                    token,
                )
            }
            ModelSource::Dir(source) => {
                let config = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source)
                    .map_err(|e| format!("failed to parse Qwen3.5 config from config.json: {e}"))?;
                if ctx.draft_path.is_some() {
                    eprintln!(
                        "  warning: DFlash (speculative decoding) is not supported for safetensors Dir sources; draft_path ignored"
                    );
                }
                if ctx.cask.sidecar.is_some() {
                    eprintln!(
                        "  warning: CASK eviction is not supported for safetensors Dir sources; eviction sidecar ignored"
                    );
                }
                let weights = match qwen35_paro_loader_kind(config.num_experts) {
                    Qwen35ParoLoaderKind::DenseManifest => {
                        // Dense Paro is represented by the manifest resolver.
                        hipfire_arch_qwen35::load_qwen35_paro_weights(&source, &config, ctx.gpu)?
                    }
                    Qwen35ParoLoaderKind::LegacyMoe => {
                        // Paro MoE/A3B has separate gate_proj/up_proj tensors and
                        // layer-shared rotation sidecars.  The manifest currently
                        // models neither the fused gate_up payload nor paro_shared
                        // ownership, so preserve the production legacy route.
                        eprintln!(
                            "  Paro MoE/A3B: using legacy loader until manifest supports shared sidecars"
                        );
                        let mut paro =
                            hipfire_arch_qwen35::qwen35::ParoSource::new(&source, &config)
                                .map_err(|e| format!("legacy Paro source setup failed: {e:?}"))?;
                        hipfire_arch_qwen35::qwen35::load_weights(
                            &mut paro,
                            std::slice::from_mut(ctx.gpu),
                            &hipfire_arch_qwen35::qwen35::Layout::single(config.n_layers),
                        )
                        .map_err(|e| format!("legacy Paro MoE load failed: {e:?}"))?
                    }
                };
                let is_kv_layer: Vec<bool> = config
                    .layer_types
                    .iter()
                    .map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention)
                    .collect();
                let mode = resolve_kv_mode(
                    ctx,
                    &hipfire_runtime::kv_mode::QWEN35_PARO_POLICY,
                    config.head_dim,
                );
                let dims = hipfire_runtime::llama::KvDims {
                    layers: hipfire_runtime::llama::KvLayers::Mask(is_kv_layer),
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    max_seq: ctx.max_seq,
                    physical_cap: Some(ctx.max_seq),
                };
                let kv_cache = hipfire_runtime::llama::KvCache::from_mode(
                    mode,
                    hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                    &dims,
                )
                .map_err(|e| format!("KvCache: {e}"))?;

                let dn_quant = crate::parse_state_quant(ctx.state_quant_override)
                    .map_err(|e| format!("{e}"))?;
                eprintln!(
                    "  DeltaNet state quant: {}",
                    if dn_quant == hipfire_arch_qwen35::qwen35::StateQuant::FP32 {
                        "FP32"
                    } else if dn_quant == hipfire_arch_qwen35::qwen35::StateQuant::Q4 {
                        "Q4"
                    } else {
                        "Q8"
                    }
                );
                if config.dim < 2048 && dn_quant != hipfire_arch_qwen35::qwen35::StateQuant::FP32 {
                    eprintln!(
                        "  warning: model dim={} (<2048); FP32 DeltaNet state is recommended for small models (current: {})",
                        config.dim,
                        if dn_quant == hipfire_arch_qwen35::qwen35::StateQuant::Q4 {
                            "Q4"
                        } else {
                            "Q8"
                        }
                    );
                }
                let dn_state = hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant(
                    ctx.gpu, &config, dn_quant,
                )
                .map_err(|e| format!("DeltaNetState::new_with_quant: {e:?}"))?;
                let scratch = hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(
                    ctx.gpu,
                    &config,
                    2048,
                    ctx.max_seq,
                )
                .map_err(|e| format!("Qwen35Scratch::new_with_kv_max: {e:?}"))?;

                let bundle = hipfire_arch_qwen35::Qwen35Bundle {
                    config,
                    weights,
                    scratch,
                    kv_cache,
                    dn_state,
                    mtp_head: None,
                    pipeline: None,
                };
                Ok(LoadedModel {
                    state: Some(ModelState::Qwen35(bundle)),
                    ..LoadedModel::skeleton(
                        meta.arch_id,
                        meta.tokenizer,
                        ctx.max_seq,
                        ctx.max_seq,
                        ctx.path.to_string(),
                        meta.chat_template,
                        ctx.mtp_mode,
                        ctx.mtp_k,
                        token.record(),
                    )
                })
            }
        }
    }
}

// ─── LlamaCarrier ────────────────────────────────────────────────────

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    fn name(&self) -> &'static str {
        "llama"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut() {
            Some(ModelState::Llama(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("llama: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 0 = LLaMA/Mistral, 1 = plain Qwen3/Qwen2 (both namespaces).
        // Explicit allowlist (was an open `< 5` range that would silently
        // swallow any future HFQ id in 2..=4 into the llama path).
        matches!(arch_id, 0 | 1)
    }
    fn classify_parallel_variant(&self, src: &ModelSource) -> Result<ModelVariant, String> {
        match src {
            ModelSource::Hfq(hfq) => {
                // config_from_hfq already prefixes errors with "llama: " —
                // propagate without adding another prefix to avoid "llama: llama:".
                let config = hipfire_runtime::hfq::config_from_hfq(hfq)?;
                // Architecture check first: Qwen3 (arch_id=1) is always PlainQwen3
                // regardless of any QK-norm tensors that might be present.
                // has_qk_norm only applies to LLaMA/Mistral (arch_id=0).
                if config.arch == hipfire_runtime::llama::ModelArch::Qwen3 {
                    Ok(ModelVariant::PlainQwen3)
                } else if config.has_qk_norm {
                    Ok(ModelVariant::LlamaQkNorm)
                } else {
                    Ok(ModelVariant::LlamaNoQkNorm)
                }
            }
            ModelSource::Dir(source) => {
                let config = hipfire_runtime::hfq::config_from_safetensors_llama(source)?;
                if config.arch == hipfire_runtime::llama::ModelArch::Qwen3 {
                    Ok(ModelVariant::PlainQwen3)
                } else if config.has_qk_norm {
                    Ok(ModelVariant::LlamaQkNorm)
                } else {
                    Ok(ModelVariant::LlamaNoQkNorm)
                }
            }
        }
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "llama: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "llama: safetensors + pp>1 unsupported",
            }
            .into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        // ── source-varying seam: yields a LlamaBundle ──
        let mut bundle = match src {
            ModelSource::Hfq(hfq) => {
                match hipfire_arch_llama::load_llama_bundle(ModelSource::Hfq(hfq), ctx) {
                    Ok(bundle) => bundle,
                    Err(error) => return Err(error),
                }
            }
            ModelSource::Dir(source) => {
                let config =
                    hipfire_runtime::hfq::config_from_safetensors_llama(&source).map_err(|e| {
                        format!("failed to parse LLaMA/Qwen3 config from config.json: {e}")
                    })?;
                let weights = match hipfire_runtime::hfq::load_weights_paroquant_llama(
                    &source, &config, ctx.gpu,
                ) {
                    Ok(weights) => weights,
                    Err(error) => {
                        return Err(format!("load_weights_paroquant_llama: {error:?}"));
                    }
                };
                #[cfg(feature = "dflash-fault-inject")]
                if let Err(error) = hipfire_runtime::dflash_generic::generic_dflash_construction_boundary(
                    hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetWeights,
                ) {
                    weights.free_gpu(ctx.gpu);
                    return Err(error);
                }
                let mode = resolve_kv_mode(
                    ctx,
                    &hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY,
                    config.head_dim,
                );
                let dims = hipfire_runtime::llama::KvDims {
                    layers: hipfire_runtime::llama::KvLayers::Flat(config.n_layers),
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    max_seq: ctx.max_seq,
                    physical_cap: Some(ctx.max_seq),
                };
                let kv = match hipfire_runtime::llama::KvCache::from_mode(
                    mode,
                    hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                    &dims,
                ) {
                    Ok(kv) => kv,
                    Err(error) => {
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("KvCache: {error}"));
                    }
                };
                #[cfg(feature = "dflash-fault-inject")]
                if let Err(error) =
                    hipfire_runtime::dflash_generic::generic_dflash_construction_boundary(
                        hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetKv,
                    )
                {
                    kv.free_gpu(ctx.gpu);
                    weights.free_gpu(ctx.gpu);
                    return Err(error);
                }
                let scratch = match hipfire_runtime::llama::ForwardScratch::new_with_max_seq(
                    ctx.gpu,
                    &config,
                    ctx.max_seq,
                ) {
                    Ok(scratch) => scratch,
                    Err(error) => {
                        kv.free_gpu(ctx.gpu);
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("ForwardScratch::new_with_max_seq: {error:?}"));
                    }
                };
                hipfire_arch_llama::LlamaBundle {
                    config,
                    weights,
                    scratch,
                    kv,
                    dflash_extract_layers: Vec::new(),
                    dspark_weights: None,
                    dspark_assets: None,
                }
            }
        };

        // ── DSpark sidecar discovery ──────────────────────────────────────────
        // When a `<stem>-dspark.<ext>` sidecar exists alongside the main model
        // and speculation is not explicitly disabled (`ctx.spec.dspark != Some(false)`),
        // load the Qwen3-8B drafter body + DSpark globals into the bundle.
        //
        // The speculator BUILD arm (Task 10) reads bundle.dspark_weights +
        // bundle.dspark_assets to wire the DsparkDrafter into the serve path.
        // This block only does the load — no speculator is built here.
        if ctx.spec.dspark != Some(false) {
            let base_path = std::path::Path::new(ctx.path);
            let dspark_path: Option<std::path::PathBuf> = match (
                base_path.parent(),
                base_path.file_stem(),
                base_path.extension(),
            ) {
                (Some(parent), Some(stem), Some(ext)) => Some(parent.join(format!(
                    "{}-dspark.{}",
                    stem.to_string_lossy(),
                    ext.to_string_lossy()
                ))),
                _ => None,
            };
            if let Some(p) = dspark_path.filter(|p| p.exists()) {
                eprintln!("llama: opening DSpark sidecar HFQ {p:?}");
                match hipfire_runtime::hfq::HfqFile::open(&p) {
                    Ok(mut sidecar) => {
                        sidecar.drop_mmap();
                        match hipfire_arch_llama::dspark_body::load_qwen3_dspark(&sidecar, ctx.gpu)
                        {
                            Ok(Some((dspark_weights, dspark_assets))) => {
                                eprintln!(
                                    "  llama: DSpark sidecar loaded (block_size={}, target_layers={:?})",
                                    dspark_weights.cfg.block_size,
                                    dspark_weights.cfg.target_layer_ids,
                                );
                                bundle.dspark_weights = Some(dspark_weights);
                                bundle.dspark_assets = Some(dspark_assets);
                            }
                            Ok(None) => {
                                eprintln!(
                                    "  llama: DSpark sidecar {p:?} has no dspark_* metadata — skipping"
                                );
                            }
                            Err(e) => {
                                eprintln!("  llama: WARNING DSpark sidecar load failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  llama: WARNING cannot open DSpark sidecar {p:?}: {e}");
                    }
                }
            } else if ctx.spec.dspark == Some(true) {
                // Forced `--spec dspark` but the sidecar file is absent → we would
                // silently run AR. Warn (auto/`None` stays quiet — a missing sidecar
                // is the expected no-op there).
                eprintln!(
                    "  llama: WARNING `--spec dspark` requested but no `-dspark` sidecar found \
                     (expected `<stem>-dspark.<ext>` next to the model) — falling back to AR/other drafter"
                );
            }
        }

        // ── single shared tail ──
        // Precedence (arch_id=0/1): DSpark > DFlash > n-gram.
        //
        // DSpark sidecar speculator: present when the `-dspark` sidecar was loaded
        // (bundle.dspark_weights.is_some()) AND speculation is not explicitly disabled.
        // Consumes the assets from the bundle (moves them into the speculator body).
        //
        // If no DSpark sidecar is available, fall through to:
        // - DFlash generic speculator (arch_id=20 draft).
        // - Opt-in model-free n-gram (HIPFIRE_NGRAM_DRAFT=1).
        let speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> = if bundle
            .dspark_weights
            .is_some()
            && ctx.spec.dspark != Some(false)
        {
            let dspark_weights = bundle.dspark_weights.take().unwrap();
            let assets = bundle.dspark_assets.take().unwrap();
            let block = dspark_weights.cfg.block_size;
            let vocab = dspark_lm_head_vocab(
                dspark_weights.cfg.draft_vocab_size,
                assets.config.vocab_size,
            );

            // stage_norm = drafter's final `norm.weight` (output_norm in the sidecar).
            // Shallow-clone so the LlamaWeights (assets) owns the primary GpuTensor;
            // the speculator holds an alias that is freed before the weights on unload.
            let stage_norm = assets.weights.output_norm.shallow_clone();

            // lm_head fix: assets.weights.output.buf.dtype == Raw (upload_raw always
            // sets Raw), but the actual data layout is F16.  run_heads dispatches on
            // GpuTensor.dtype, so we shallow_clone and fix the dtype + shape here.
            // (The parity harness does the same at qwen3_dspark_parity.rs:215-217.)
            let mut lm_head = assets.weights.output.buf.shallow_clone();
            lm_head.dtype = rdna_compute::DType::F16;
            lm_head.shape = vec![vocab];

            // conf_threshold ladder: env > CLI arg > 0.1
            // Default 0.1 (sweep-tuned): 0.5 over-truncates (1.46/7 proposed);
            // 0.1 proposes ~6.94/7, +16.6% prose tok/s / +7.1% code tok/s.
            let conf_threshold = std::env::var("HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .or(ctx.spec.dspark_conf_threshold)
                .unwrap_or(0.1f32);

            eprintln!(
                "  llama DSpark speculator enabled (sidecar, block={}, conf_threshold={:.2})",
                block, conf_threshold
            );
            let body = match hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
                assets,
                &dspark_weights.cfg,
                ctx.gpu,
            ) {
                Ok(body) => body,
                Err(error) => {
                    dspark_weights.free_gpu(ctx.gpu);
                    bundle.free_gpu(ctx.gpu);
                    return Err(format!("llama DSpark body build failed: {error}"));
                }
            };
            Some(hipfire_runtime::dspark_core::build_dspark_speculator(
                body,
                dspark_weights,
                stage_norm,
                lm_head,
                block,
                ctx.max_seq,
                conf_threshold,
                // temp>0 sampled verify ENABLED: with lazy prefix sampling (only ~τ
                // lm_heads/window) qwen3 DSpark at temp>0 beats AR by ~+24% (29.6 vs
                // 23.8 tok/s on gfx1151 code) and stays distribution-identical to AR
                // (fused sample_top_p_pf, honors temp+top_p+top_k). The daemon routes
                // temp>0 llama through the chain path (requires_greedy()==false).
                true,
            ))
        } else if let Some(dp) = ctx.draft_path {
            // Peek at the draft's arch_id without consuming the path; the builder
            // opens it again internally.
            match hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(dp)) {
                Ok(draft_hfq) if draft_hfq.arch_id == 20 => {
                    // Parse DflashConfig to validate the cross-attention concat invariant
                    // (review finding L4): the drafter's hidden must equal the target dim.
                    let draft_cfg =
                        match hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq) {
                            Some(config) => config,
                            None => {
                                let error = format!(
                                    "DFlash draft '{}' has arch_id=20 but missing or malformed \
                                     'dflash' metadata block",
                                    dp
                                );
                                bundle.free_gpu(ctx.gpu);
                                return Err(error);
                            }
                        };
                    if bundle.config.dim != draft_cfg.hidden {
                        let error = format!(
                            "DFlash draft '{}' hidden={} != target dim={} \
                                 (cross-attention concat invariant L4: drafter hidden \
                                 must equal target residual dim)",
                            dp, draft_cfg.hidden, bundle.config.dim
                        );
                        bundle.free_gpu(ctx.gpu);
                        return Err(error);
                    }
                    // Drop the peek handle before the builder reopens it.
                    drop(draft_hfq);
                    let spec =
                        match hipfire_runtime::dflash_generic::build_generic_dflash_speculator(
                            ctx.gpu,
                            dp,
                            &mut bundle,
                            ctx.max_seq,
                        ) {
                            Ok(spec) => spec,
                            Err(error) => {
                                bundle.free_gpu(ctx.gpu);
                                return Err(format!(
                                    "DFlash generic speculator build failed: {error}"
                                ));
                            }
                        };
                    eprintln!(
                        "  DFlash generic speculator loaded for arch {} target: {}",
                        meta.arch_id, dp
                    );
                    Some(spec)
                }
                // Not a DFlash draft or unreadable — log why and fall through to n-gram.
                Err(e) => {
                    eprintln!(
                        "  [hipfire] draft '{}' unreadable ({e}); DFlash speculator not built, falling back to n-gram",
                        dp
                    );
                    crate::spec_build::build_speculator(
                        meta.arch_id,
                        None,
                        true,
                        ctx.max_seq,
                        ctx.spec,
                    )
                }
                Ok(draft_hfq) => {
                    eprintln!(
                        "  [hipfire] draft '{}' is arch_id={} (not 20 / DFlash); DFlash speculator not built, falling back to n-gram",
                        dp, draft_hfq.arch_id
                    );
                    crate::spec_build::build_speculator(
                        meta.arch_id,
                        None,
                        true,
                        ctx.max_seq,
                        ctx.spec,
                    )
                }
            }
        } else {
            // No draft configured: opt-in model-free n-gram (HIPFIRE_NGRAM_DRAFT=1) or None.
            crate::spec_build::build_speculator(meta.arch_id, None, true, ctx.max_seq, ctx.spec)
        };
        Ok(LoadedModel {
            state: Some(ModelState::Llama(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
                token.record(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        drive_route, dspark_lm_head_vocab, qwen35_paro_loader_kind, route_frozen_selection,
        Qwen35FrozenRoute, Qwen35ParoLoaderKind,
    };
    use hipfire_runtime::loader_api::ModelSource;
    use std::io::Write;

    #[test]
    fn dspark_lm_head_vocab_preserves_reduced_and_fallback_shapes() {
        assert_eq!(dspark_lm_head_vocab(151_936, 152_064), 151_936);
        assert_eq!(dspark_lm_head_vocab(0, 152_064), 152_064);
    }

    #[test]
    fn qwen35_paro_moe_keeps_legacy_shared_sidecar_route() {
        assert_eq!(
            qwen35_paro_loader_kind(0),
            Qwen35ParoLoaderKind::DenseManifest
        );
        assert_eq!(qwen35_paro_loader_kind(1), Qwen35ParoLoaderKind::LegacyMoe);
        assert_eq!(
            qwen35_paro_loader_kind(256),
            Qwen35ParoLoaderKind::LegacyMoe
        );
    }

    // ── Frozen selection routing (CPU-only, no GPU) ──────────────────
    //
    // The preflight + routing decision is fully metadata-driven: a real
    // on-disk HFQ index (no payloads needed for the selection) and an
    // arch string.  These tests prove the carrier's selection semantics:
    // Ineligible → Legacy fallback (never a new load failure), Eligible
    // → Frozen allocation with NO fallback path, Invalid → fail.

    fn moe_config_json() -> serde_json::Value {
        serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": 8,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "tie_word_embeddings": true,
        })
    }

    /// First-candidate physical name for a qwen35 manifest entry —
    /// mirrors the resolver's candidate list (test fixture only).
    fn qwen35_physical(
        config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
        name: &str,
        layer: Option<usize>,
    ) -> Option<String> {
        use hipfire_arch_qwen35::qwen35::LayerType;
        let stem = match (name, layer) {
            ("token_embd", None) => "embed_tokens.weight".to_string(),
            ("output_norm", None) => "norm.weight".to_string(),
            ("lm_head", None) => "lm_head.weight".to_string(),
            (name, Some(layer)) => {
                let rel = match name {
                    "attn_norm" => "input_layernorm.weight".to_string(),
                    "ffn_norm" => "post_attention_layernorm.weight".to_string(),
                    "wq" => "self_attn.q_proj.weight".to_string(),
                    "wk" => "self_attn.k_proj.weight".to_string(),
                    "wv" => "self_attn.v_proj.weight".to_string(),
                    "wo" => {
                        if config.layer_types[layer] == LayerType::LinearAttention {
                            "linear_attn.out_proj.weight".to_string()
                        } else {
                            "self_attn.o_proj.weight".to_string()
                        }
                    }
                    "q_norm" => "self_attn.q_norm.weight".to_string(),
                    "k_norm" => "self_attn.k_norm.weight".to_string(),
                    "wqkv" => "linear_attn.in_proj_qkv.weight".to_string(),
                    "wz" => "linear_attn.in_proj_z.weight".to_string(),
                    "w_alpha" => "linear_attn.in_proj_a.weight".to_string(),
                    "w_beta" => "linear_attn.in_proj_b.weight".to_string(),
                    "a_log" => "linear_attn.A_log".to_string(),
                    "dt_bias" => "linear_attn.dt_bias".to_string(),
                    "conv" => "linear_attn.conv1d.weight".to_string(),
                    "norm" => "linear_attn.norm.weight".to_string(),
                    "ffn_gate" => "mlp.gate_proj.weight".to_string(),
                    "ffn_up" => "mlp.up_proj.weight".to_string(),
                    "ffn_down" => "mlp.down_proj.weight".to_string(),
                    "router" => "mlp.gate.weight".to_string(),
                    "shared_expert_gate" => "mlp.shared_expert_gate.weight".to_string(),
                    "shared_gate" => "mlp.shared_expert.gate_proj.weight".to_string(),
                    "shared_up" => "mlp.shared_expert.up_proj.weight".to_string(),
                    "shared_down" => "mlp.shared_expert.down_proj.weight".to_string(),
                    name if name.starts_with("expert.") => {
                        let rest = name.strip_prefix("expert.").unwrap();
                        let (idx, proj) = rest.split_once('.').unwrap();
                        format!(
                            "mlp.experts.{idx}.{}.weight",
                            match proj {
                                "gate_up" => "gate_up_proj",
                                "down" => "down_proj",
                                _ => return None,
                            }
                        )
                    }
                    _ => return None,
                };
                format!("layers.{layer}.{rel}")
            }
            _ => return None,
        };
        Some(format!("model.language_model.{stem}"))
    }

    fn awq_physical(main_physical: &str) -> String {
        format!(
            "{}.awq_scale.weight",
            main_physical.strip_suffix(".weight").unwrap()
        )
    }

    /// Write an HFQ index with explicit per-tensor shapes.
    fn write_selection_fixture_shaped(
        path: &std::path::Path,
        arch_id: u32,
        config_json: &serde_json::Value,
        extra: &[(&str, Vec<u32>)],
    ) -> hipfire_arch_qwen35::qwen35::Qwen35Config {
        use hipfire_runtime::arch::Architecture;
        let config = hipfire_arch_qwen35::qwen35::config_from_metadata_json(
            &serde_json::json!({ "config": config_json }).to_string(),
        )
        .expect("fixture config must parse");
        let manifest = <hipfire_arch_qwen35::Qwen35 as Architecture>::weight_manifest(&config);

        let mut tensors: Vec<(String, Vec<u32>)> = Vec::new();
        for entry in &manifest {
            let physical = qwen35_physical(&config, &entry.name, entry.layer)
                .unwrap_or_else(|| panic!("no physical name for {}", entry.name));
            let shape: Vec<u32> = entry.logical_shape.iter().map(|&d| d as u32).collect();
            tensors.push((physical, shape));
        }
        tensors.extend(
            extra
                .iter()
                .map(|(n, shape)| (n.to_string(), shape.clone())),
        );

        let metadata = serde_json::json!({ "config": config_json }).to_string();
        let meta_bytes = metadata.as_bytes();
        let mut idx = Vec::new();
        idx.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, shape) in &tensors {
            idx.extend_from_slice(&(name.len() as u16).to_le_bytes());
            idx.extend_from_slice(name.as_bytes());
            idx.push(13); // MQ4G256
            idx.push(shape.len() as u8);
            for d in shape {
                idx.extend_from_slice(&d.to_le_bytes());
            }
            idx.extend_from_slice(&0u32.to_le_bytes()); // group_size
            let size: u64 = shape.iter().map(|&d| d as u64).product::<u64>().max(64);
            idx.extend_from_slice(&size.to_le_bytes());
        }
        let metadata_offset: u64 = 32;
        let data_offset: u64 = metadata_offset + meta_bytes.len() as u64 + idx.len() as u64;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&(tensors.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(meta_bytes).unwrap();
        f.write_all(&idx).unwrap();
        for _ in &tensors {
            f.write_all(&[0u8; 64]).unwrap();
        }
        f.flush().unwrap();
        config
    }

    fn selection_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire-qwen35-route-{tag}-{}-{}.hfq",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ))
    }

    fn default_flags() -> hipfire_arch_qwen35::Qwen35MoeLoadFlags {
        hipfire_arch_qwen35::Qwen35MoeLoadFlags {
            paged_experts: false,
            moe_awq_enabled: true,
        }
    }

    fn run_selection(
        path: &std::path::Path,
        arch: &str,
    ) -> hipfire_arch_qwen35::Qwen35FrozenPreflight {
        let hfq = hipfire_runtime::hfq::HfqFile::open(path).expect("fixture opens");
        hipfire_arch_qwen35::preflight_qwen35_frozen(
            ModelSource::Hfq(hfq),
            arch,
            true,
            default_flags(),
        )
    }

    #[test]
    fn frozen_selection_dense_routes_to_legacy_without_gpu() {
        // Dense qwen35 (arch_id 6, num_experts=0): the preflight must
        // select Legacy BEFORE any manifest work — the fixture carries
        // zero tensors, proving no source upload is required for the
        // decision.
        let path = selection_path("dense");
        let dense = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": 0,
        });
        write_selection_fixture_shaped(&path, 6, &dense, &[]);
        let selection = run_selection(&path, "gfx1100");
        match &selection {
            hipfire_arch_qwen35::Qwen35FrozenPreflight::Ineligible(reason) => {
                assert!(
                    reason.reason().contains("dense"),
                    "expected dense reason, got {}",
                    reason.reason()
                );
            }
            other => panic!("expected Ineligible for dense, got {other:?}"),
        }
        // Routing: Ineligible → Legacy fallback with the ORIGINAL source —
        // a selection, NOT a load failure.  drive_route must hand the
        // source to the Legacy arm and never touch the Frozen arm.
        let route = route_frozen_selection(selection);
        let legacy_called = std::cell::Cell::new(false);
        let frozen_called = std::cell::Cell::new(false);
        let result = drive_route(
            route,
            |_plan| {
                frozen_called.set(true);
                Err("frozen must not run".to_string())
            },
            |source| {
                legacy_called.set(true);
                assert!(
                    matches!(source, ModelSource::Hfq(_)),
                    "Legacy arm must receive the returned Hfq source"
                );
                Ok(42u32)
            },
        );
        assert_eq!(result, Ok(42));
        assert!(legacy_called.get(), "Legacy arm must run");
        assert!(
            !frozen_called.get(),
            "Frozen arm must not run for Ineligible"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn frozen_selection_eligible_routes_to_frozen_without_fallback() {
        // Full MoE fixture (8 experts x 2 layers, all MQ4) on gfx1100:
        // the preflight must authorize Frozen allocation.  The route from
        // an Eligible selection is Frozen ONLY — there is no Legacy
        // fallback path after this point.
        let path = selection_path("eligible");
        write_selection_fixture_shaped(&path, 6, &moe_config_json(), &[]);
        let selection = run_selection(&path, "gfx1100");
        assert!(
            matches!(
                selection,
                hipfire_arch_qwen35::Qwen35FrozenPreflight::Eligible(_)
            ),
            "expected Eligible, got {selection:?}"
        );
        let route = route_frozen_selection(selection);
        assert!(
            matches!(route, Qwen35FrozenRoute::Frozen(_)),
            "Eligible selection must route to Frozen, got {route:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn frozen_selection_eligible_operational_error_never_falls_back() {
        // Once Eligible, an OPERATIONAL failure of the Frozen load must
        // surface as a load error — the Legacy arm is never attempted.
        // This is the no-fallback contract, exercised through the exact
        // production seam (`drive_route` + `route_frozen_selection`)
        // without a GPU.
        let path = selection_path("eligible-err");
        write_selection_fixture_shaped(&path, 6, &moe_config_json(), &[]);
        let selection = run_selection(&path, "gfx1100");
        let route = route_frozen_selection(selection);
        let legacy_called = std::cell::Cell::new(false);
        let result: Result<(), String> = drive_route(
            route,
            |_plan| Err("frozen load failed: injected OOM".to_string()),
            |_source| {
                legacy_called.set(true);
                Ok(())
            },
        );
        assert!(
            result.is_err(),
            "the operational error must be surfaced, got {result:?}"
        );
        assert!(
            !legacy_called.get(),
            "Legacy must NEVER be attempted after an Eligible selection"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn frozen_selection_invalid_routes_to_fail() {
        // Routed gate-up AWQ companion: refused by the shared manifest
        // gate — neither path can serve the file, so the route fails.
        let path = selection_path("invalid");
        let config = write_selection_fixture_shaped(&path, 6, &moe_config_json(), &[]);
        let gate_up = qwen35_physical(&config, "expert.0.gate_up", Some(0)).unwrap();
        let companion: &'static str = Box::leak(awq_physical(&gate_up).into_boxed_str());
        let path2 = selection_path("invalid2");
        write_selection_fixture_shaped(&path2, 6, &moe_config_json(), &[(companion, vec![64])]);
        let selection = run_selection(&path2, "gfx1100");
        match &selection {
            hipfire_arch_qwen35::Qwen35FrozenPreflight::Invalid(msg) => {
                assert!(msg.contains("AWQ"), "message: {msg}");
            }
            other => panic!("expected Invalid for routed gate-up AWQ, got {other:?}"),
        }
        let route = route_frozen_selection(selection);
        match &route {
            Qwen35FrozenRoute::Fail(msg) => {
                assert!(msg.contains("AWQ"), "message: {msg}");
            }
            other => panic!("expected Fail route, got {other:?}"),
        }
        // Fail: neither arm runs.
        let legacy_called = std::cell::Cell::new(false);
        let frozen_called = std::cell::Cell::new(false);
        let result: Result<(), String> = drive_route(
            route,
            |_plan| {
                frozen_called.set(true);
                Err("frozen must not run".to_string())
            },
            |_source| {
                legacy_called.set(true);
                Err("legacy must not run".to_string())
            },
        );
        assert!(result.is_err());
        assert!(!legacy_called.get(), "Legacy arm must not run for Fail");
        assert!(!frozen_called.get(), "Frozen arm must not run for Fail");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&path2);
    }
}

// ─── Non-core carriers ───────────────────────────────────────────────

// ─── DotsOcrCarrier ──────────────────────────────────────────────────

pub struct DotsOcrCarrier;
impl Carrier for DotsOcrCarrier {
    fn name(&self) -> &'static str {
        "dots_ocr"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 8
    }
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::DotsOcr)
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "dots_ocr: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "dots_ocr: safetensors + pp>1 unsupported",
            }
            .into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        use hipfire_arch_dots_ocr::dots_ocr::{DotsOcrConfig, DotsOcrWeights};
        use hipfire_arch_dots_ocr::DotsOcr;
        use hipfire_runtime::arch::Architecture;
        // ── source-varying seam: (config, weights) only ──
        let (config, weights) = match src {
            ModelSource::Hfq(mut hfq) => {
                let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
                let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
                (config, weights)
            }
            ModelSource::Dir(source) => {
                let config = DotsOcrConfig::from_source(&source)?;
                let weights = DotsOcrWeights::load_weights_from_source(&source, &config, ctx.gpu)?;
                (config, weights)
            }
        };
        let state = hipfire_arch_qwen2::qwen2::Qwen2State::new_with_max_seq(
            ctx.gpu,
            &config.text,
            ctx.max_seq,
        )
        .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). dots.ocr's
        // text decoder IS Qwen2, so the n-gram arm drives it via the
        // `DotsOcrBundle: SpecTarget` impl — a strong fit because layout-JSON
        // output is densely self-repeating. The daemon's `generate_vl_dots_ocr`
        // routes to the spec decode loop when this is `Some` (vision prefill is
        // unchanged; only the decode phase becomes speculative).
        let speculator =
            crate::spec_build::build_speculator(meta.arch_id, None, true, ctx.max_seq, ctx.spec);
        Ok(LoadedModel {
            state: Some(crate::ModelState::DotsOcr(
                hipfire_arch_dots_ocr::DotsOcrBundle {
                    config,
                    weights,
                    state,
                },
            )),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
                token.record(),
            )
        })
    }
}

// ─── Deepseek4Carrier ────────────────────────────────────────────────

pub struct Deepseek4Carrier;
impl Carrier for Deepseek4Carrier {
    fn name(&self) -> &'static str {
        "deepseek4"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut() {
            Some(ModelState::Deepseek4(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("deepseek4: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(
            ctx,
        ))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 9
    }
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::Deepseek4)
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "deepseek4: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "deepseek4: safetensors + pp>1 unsupported",
            }
            .into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        use hipfire_arch_deepseek4 as deepseek4;
        use hipfire_runtime::arch::Architecture;
        // ── source-varying seam: (config, weights) only ──
        // NOTE: the Dir/safetensors arm is UNVALIDATED — no deepseek_v4
        // checkpoint was available locally to verify load fidelity. Reviewer-ask.
        // DSpark sidecar load gate: `speculation=dspark`/`auto` load the 3×MoE
        // sidecar; any other mechanism (`Some(false)`) skips it so it never pages
        // into VRAM. `None` (auto / directly-driven daemon) keeps default-on.
        let load_dspark = ctx.spec.dspark != Some(false);
        let (config, weights) = match src {
            ModelSource::Hfq(mut hfq) => {
                let mut config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
                config.load_dspark = load_dspark;
                let weights = <deepseek4::DeepseekV4 as Architecture>::load_weights(
                    &mut hfq, &config, ctx.gpu,
                )?;
                (config, weights)
            }
            ModelSource::Dir(source) => {
                let mut config = deepseek4::config_from_safetensors(&source).ok_or_else(|| {
                    "deepseek4: failed to parse config from safetensors".to_string()
                })?;
                config.load_dspark = load_dspark;
                let weights = deepseek4::DeepseekV4::load_weights_from_safetensors(
                    &source, &config, ctx.gpu,
                )?;
                (config, weights)
            }
        };
        let state = deepseek4::DeepseekV4State::new(&config)?;
        let pbs_max_batch: usize = std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let pbs = deepseek4::forward::PrefillBatchScratch::new(ctx.gpu, &config, pbs_max_batch)?;
        let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<｜end▁of▁sentence｜>"]);
        // deepseek4 MTP spec-decode capability: present iff the MTP addon weights loaded
        // (HIPFIRE_DEEPSEEK4_MTP_ADDON / .mtp-addon.hfq / HIPFIRE_DEEPSEEK4_LOAD_MTP). The
        // per-request spec gate (mtp_mode / HIPFIRE_DEEPSEEK4_SPEC_DECODE / temp<=eps) stays in
        // the generate path (T4 routing) — here we only build the capability. Undriven until T4:
        // the daemon's arch_id==9 branch still uses the bespoke generate_deepseek4 loop.
        // DSpark draft module (the `-dspark` sidecar) wins over the in-trunk MTP
        // layer when present. Built when the sidecar loaded AND the `speculation`
        // selector did not pick another mechanism (`ctx.spec.dspark != Some(false)`;
        // `None` = auto keeps the default-on behaviour). The threshold is the
        // CLI-forwarded `--dspark-conf-threshold` (env still wins in the builder).
        // `--spec dspark` (forced) but the sidecar was absent → we silently ran
        // AR before. Warn on the forced case only (auto/`None` legitimately falls
        // back without a sidecar and must stay quiet).
        if ctx.spec.dspark == Some(true) && weights.dspark.is_none() {
            eprintln!(
                "  deepseek4: WARNING `--spec dspark` requested but no `-dspark` sidecar was \
                 loaded (expected `<stem>-dspark.<ext>` next to the model) — falling back to MTP/AR"
            );
        }
        let dspark_enabled = weights.dspark.is_some() && ctx.spec.dspark != Some(false);
        let speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> = if dspark_enabled {
            let block = weights.dspark.as_ref().unwrap().cfg.block_size;
            let ctx_capacity = config.max_position_embeddings;
            eprintln!("  deepseek4 DSpark speculator enabled (sidecar, block={block})");
            Some(
                hipfire_arch_deepseek4::dspark_speculator::build_deepseek4_dspark_speculator(
                    &config,
                    &weights,
                    block,
                    ctx_capacity,
                    ctx.spec.dspark_conf_threshold,
                    // temp>0 sampled verify ENABLED in serving. The earlier "loses to
                    // AR → gate off" reasoning was a fixed-block measurement artifact;
                    // comprehensive temp=1.0 tests with the τ-adaptive block-depth
                    // controller show ds4 DSpark temp>0 BEATS AR, and the opt-in CACTUS
                    // acceptance-boost (request `cactus_delta`) adds more on top.
                    // Distribution-preserving at cactus_delta=0 (the default).
                    true,
                )
                .map_err(|e| format!("deepseek4 DSpark speculator build failed: {e}"))?,
            )
        } else if weights.mtp_layer.is_some() {
            let max_n = ctx.mtp_k;
            let ctx_capacity = config.max_position_embeddings;
            eprintln!("  deepseek4 MTP speculator enabled (in-weights, K={max_n})");
            Some(
                hipfire_arch_deepseek4::mtp_speculator::build_deepseek4_mtp_speculator(
                    max_n,
                    ctx_capacity,
                ),
            )
        } else {
            None
        };
        Ok(LoadedModel {
            state: Some(crate::ModelState::Deepseek4(deepseek4::Deepseek4Bundle {
                config,
                weights,
                state,
                eos_tok,
                pbs,
            })),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
                token.record(),
            )
        })
    }
}

// ─── MinimaxCarrier ──────────────────────────────────────────────────

pub struct MinimaxCarrier;
impl Carrier for MinimaxCarrier {
    fn name(&self) -> &'static str {
        "minimax"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut() {
            Some(ModelState::Minimax(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("minimax: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Preserve the generic emitter behavior here.  Carrier-specific
        // literal terminators are not part of the loader protocol.
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 10
    }
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::Minimax)
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            // Preserve the two per-source error strings byte-for-byte.
            return Err(match &src {
                ModelSource::Hfq(_) => "minimax: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "minimax: safetensors + pp>1 unsupported",
            }
            .into());
        }
        // Per-source diagnostic stays at the call site, before resolve_source_meta.
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        // ── source-varying seam: (config, weights) only ──
        use hipfire_runtime::arch::Architecture;
        let (config, weights) = match src {
            ModelSource::Hfq(mut hfq_file) => {
                let config =
                    <hipfire_arch_minimax::arch::MiniMaxM2 as Architecture>::config_from_hfq(
                        &hfq_file,
                    )?;
                let weights =
                    <hipfire_arch_minimax::arch::MiniMaxM2 as Architecture>::load_weights(
                        &mut hfq_file,
                        &config,
                        ctx.gpu,
                    )?;
                (config, weights)
            }
            ModelSource::Dir(source) => {
                let config = config_from_safetensors(&source)
                    .map_err(|e| format!("failed to parse MiniMax config from config.json: {e}"))?;
                let weights = load_weights_from_safetensors(&source, &config, ctx.gpu)?;
                (config, weights)
            }
        };

        // ── single shared tail (byte-identical to the previous per-arm tails) ──
        let state = MiniMaxState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
        let eos_tok = resolve_eos_tok(
            &meta.tokenizer,
            &["[e~[", "<|im_end|>", "</s>", "<|endoftext|>"],
        );
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). MiniMax-M2
        // (arch_id=10) impls `SpecTarget` (pure GQA, no recurrent state), so it
        // can be driven by the arch-generic spec loop with no draft model.
        // `None` ⇒ AR-only (the bespoke `generate_minimax` path).
        let speculator =
            crate::spec_build::build_speculator(meta.arch_id, None, true, ctx.max_seq, ctx.spec);
        Ok(LoadedModel {
            state: Some(ModelState::Minimax(crate::MiniMaxBundle {
                config,
                weights,
                state,
                eos_tok,
            })),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
                token.record(),
            )
        })
    }
}

// ─── Lfm2MoeCarrier ──────────────────────────────────────────────────

pub struct Lfm2MoeCarrier;
impl Carrier for Lfm2MoeCarrier {
    fn name(&self) -> &'static str {
        "lfm2moe"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut() {
            Some(ModelState::Lfm2Moe(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("lfm2moe: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Preserve the generic emitter behavior; carrier-owned literal marker
        // arrays are not part of the loader protocol.
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 11
    }
    fn classify_parallel_variant(&self, src: &ModelSource) -> Result<ModelVariant, String> {
        // Dense vs MoE is determined by `num_experts` in the config.
        // Extract the concrete source so we can pass it to the
        // architecture-owned helper (which expects &dyn model_source::ModelSource).
        match src {
            ModelSource::Hfq(hfq) => {
                let is_moe = hipfire_arch_lfm2moe::config::classify_is_moe(hfq)?;
                Ok(if is_moe {
                    ModelVariant::Lfm2Moe
                } else {
                    ModelVariant::Lfm2Dense
                })
            }
            ModelSource::Dir(s) => {
                let is_moe = hipfire_arch_lfm2moe::config::classify_is_moe(s)?;
                Ok(if is_moe {
                    ModelVariant::Lfm2Moe
                } else {
                    ModelVariant::Lfm2Dense
                })
            }
        }
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "lfm2moe: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "lfm2moe: safetensors + pp>1 unsupported",
            }
            .into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        use hipfire_arch_lfm2moe as lfm2moe;
        // ── source-varying seam: (config, weights) only ──
        let (config, weights) = match src {
            ModelSource::Hfq(mut hfq) => {
                let config = lfm2moe::config::Lfm2MoeConfig::from_hfq(&hfq)?;
                let weights = lfm2moe::lfm2moe::Lfm2MoeWeights::load(&mut hfq, &config, ctx.gpu)?;
                (config, weights)
            }
            ModelSource::Dir(source) => {
                let config = lfm2moe::config_from_source(&source).ok_or_else(|| {
                    "lfm2moe: failed to parse config from safetensors".to_string()
                })?;
                let weights = lfm2moe::load_weights_from_source(&source, &config, ctx.gpu)?;
                (config, weights)
            }
        };

        let state = lfm2moe::lfm2moe::Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
        let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<|im_end|>", "</s>", "<|endoftext|>"]);
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). LFM2.5-MoE
        // (arch_id=11) impls `SpecTarget` with conv-state snapshot/rollback in
        // `verify_block`/`commit_prefix`, so it can be driven by the arch-generic
        // spec loop with no draft model. `None` ⇒ AR-only (`generate_lfm2moe`).
        let speculator =
            crate::spec_build::build_speculator(meta.arch_id, None, true, ctx.max_seq, ctx.spec);
        Ok(LoadedModel {
            state: Some(ModelState::Lfm2Moe(crate::Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
            })),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                ctx.mtp_mode,
                ctx.mtp_k,
                token.record(),
            )
        })
    }
}

// ─── Cohere2MoeCarrier ───────────────────────────────────────────────
// cohere2moe (arch_id 12, HFQ-only) landed upstream via the generic
// `HfqCarrier` fn-pointer registry entry. Our dedicated-carrier refactor
// removed that generic struct, so this wraps the still-standalone
// `crate::load_cohere2moe` with the same HFQ-extraction glue the old
// `HfqCarrier::load` used — keeping cohere2moe's load path byte-identical
// to upstream while fitting the dedicated-carrier registry.
pub struct Cohere2MoeCarrier;
impl Carrier for Cohere2MoeCarrier {
    fn name(&self) -> &'static str {
        "cohere2moe"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<ModelState>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut() {
            Some(ModelState::Cohere2Moe(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("cohere2moe: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Arch-specific emitter: North's agentic-marker state machine (markers
        // never surfaced, reasoning channel, ACTION→tool_calls) + the empty-turn
        // and think-budget generation guards via `take_forced`.
        Ok(hipfire_arch_cohere2moe::spec_emit::Cohere2MoeEmit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 12 = Cohere2-MoE in both the HFQ and safetensors-Dir namespaces.
        arch_id == 12
    }
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::Cohere2Moe)
    }
    fn load(
        &self,
        src: ModelSource,
        ctx: &mut LoadCtx,
        token: &CarrierLoadToken,
    ) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("cohere2moe: pp>1 unsupported via registry".into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        match src {
            ModelSource::Hfq(hfq) => {
                let tokenizer =
                    hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                        .map_err(|e| format!("cohere2moe: tokenizer not found: {e}"))?;
                let mut lm = crate::load_cohere2moe(
                    hfq,
                    tokenizer,
                    ctx.gpu,
                    ctx.max_seq,
                    ctx.path,
                    ctx.mtp_mode,
                    ctx.mtp_k,
                    token.record(),
                )?;
                // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1).
                lm.speculator = crate::spec_build::build_speculator(
                    meta.arch_id,
                    None,
                    true,
                    ctx.max_seq,
                    ctx.spec,
                );
                Ok(lm)
            }
            ModelSource::Dir(source) => {
                // Transparent ParoQuant safetensors-Dir path (North-Mini-Code).
                let config = hipfire_arch_cohere2moe::Cohere2MoeConfig::from_safetensors(&source)
                    .map_err(|e| {
                    format!("failed to parse Cohere2-MoE config from config.json: {e}")
                })?;
                let weights =
                    hipfire_arch_cohere2moe::paro_dir::load_from_source(&source, &config, ctx.gpu)?;
                let state = hipfire_arch_cohere2moe::Cohere2MoeState::new_with_max_seq(
                    ctx.gpu,
                    &config,
                    ctx.max_seq,
                )
                .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
                let eos_tok = resolve_eos_tok(
                    &meta.tokenizer,
                    &["<|END_OF_TURN_TOKEN|>", "</s>", "<|endoftext|>"],
                );
                let speculator = crate::spec_build::build_speculator(
                    meta.arch_id,
                    None,
                    true,
                    ctx.max_seq,
                    ctx.spec,
                );
                Ok(LoadedModel {
                    state: Some(ModelState::Cohere2Moe(crate::Cohere2MoeBundle {
                        config,
                        weights,
                        state,
                        eos_tok,
                    })),
                    speculator,
                    ..LoadedModel::skeleton(
                        meta.arch_id,
                        meta.tokenizer,
                        ctx.max_seq,
                        ctx.max_seq,
                        ctx.path.to_string(),
                        meta.chat_template,
                        ctx.mtp_mode,
                        ctx.mtp_k,
                        token.record(),
                    )
                })
            }
        }
    }
}

// ─── Classification tests ──────────────────────────────────────────────

#[cfg(test)]
mod classification_tests {
    use super::*;
    use crate::parallel_capability::ModelVariant;
    use crate::{classify_source, Carrier};
    use hipfire_runtime::loader_api::ModelSource;
    use std::io::Write;

    // ── Fixture helpers ─────────────────────────────────────────────

    /// Build a complete LLaMA config JSON using `serde_json::Value` (no string
    /// splicing).  `overrides` are applied on top of the base — duplicate keys
    /// overwrite rather than producing malformed JSON.
    fn llama_cfg(overrides: &[(&str, &serde_json::Value)]) -> String {
        let mut cfg = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        });
        for (k, v) in overrides {
            cfg[k] = (*v).clone();
        }
        cfg.to_string()
    }

    /// Build a complete Qwen3.5 config JSON.
    fn qwen35_cfg(overrides: &[(&str, &serde_json::Value)]) -> String {
        let mut cfg = serde_json::json!({
            "model_type": "qwen3.5",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 152064,
        });
        for (k, v) in overrides {
            cfg[k] = (*v).clone();
        }
        cfg.to_string()
    }

    /// Build a complete Qwen3 config JSON.
    fn qwen3_cfg(overrides: &[(&str, &serde_json::Value)]) -> String {
        let mut cfg = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 152064,
        });
        for (k, v) in overrides {
            cfg[k] = (*v).clone();
        }
        cfg.to_string()
    }

    /// Build a complete LFM2 config JSON.
    fn lfm2_cfg(overrides: &[(&str, &serde_json::Value)]) -> String {
        let mut cfg = serde_json::json!({
            "model_type": "lfm2",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"],
        });
        for (k, v) in overrides {
            cfg[k] = (*v).clone();
        }
        cfg.to_string()
    }

    /// Wrap a config JSON body inside the `{"config":…}` envelope that
    /// `write_hfq_fixture` expects when its argument already has a `"config"` key.
    fn config_wrapped(cfg_body: &str) -> String {
        format!(r#"{{"config":{cfg_body}}}"#)
    }

    fn write_hfq_fixture(
        dir: &std::path::Path,
        name: &str,
        arch_id: u32,
        metadata_json: &str,
        tensor_names: &[&str],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();

        // Must be valid JSON with a "config" wrapper key. Panic early so
        // a misconstructed fixture fast-fails rather than creating a corrupt file.
        let meta_val: serde_json::Value =
            serde_json::from_str(metadata_json).expect("fixture metadata must be valid JSON");
        let wrapped = if meta_val.get("config").is_some() {
            metadata_json.to_string()
        } else {
            format!(r#"{{"architecture":"test","config":{}}}"#, metadata_json)
        };
        let meta_bytes = wrapped.as_bytes();
        let n_tensors = tensor_names.len() as u32;

        // Each indexed tensor must have nonzero data_size and matching
        // payload bytes so that tensor_data() returns a non-empty slice.
        // The payload itself is 12 zero bytes — enough to prove existence
        // without resembling any real weight data.
        const TENSOR_PAYLOAD_SIZE: u64 = 12;
        const TENSOR_PAYLOAD: [u8; TENSOR_PAYLOAD_SIZE as usize] = [0u8; 12];

        let mut idx_bytes: Vec<u8> = Vec::new();
        idx_bytes.extend_from_slice(&n_tensors.to_le_bytes());
        for tname in tensor_names {
            let name_bytes = tname.as_bytes();
            let name_len = name_bytes.len() as u16;
            idx_bytes.extend_from_slice(&name_len.to_le_bytes());
            idx_bytes.extend_from_slice(name_bytes);
            idx_bytes.push(1); // quant_type
            idx_bytes.push(1); // n_dims
            idx_bytes.extend_from_slice(&4u32.to_le_bytes()); // shape[0] = 4
            idx_bytes.extend_from_slice(&0u32.to_le_bytes()); // group_size
            idx_bytes.extend_from_slice(&TENSOR_PAYLOAD_SIZE.to_le_bytes());
        }

        let metadata_offset: u64 = 32;
        let data_offset: u64 = metadata_offset + meta_bytes.len() as u64 + idx_bytes.len() as u64;

        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&n_tensors.to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(meta_bytes).unwrap();
        f.write_all(&idx_bytes).unwrap();
        // Write nonzero payload data so tensor_data() returns non-empty slices
        for _ in tensor_names {
            f.write_all(&TENSOR_PAYLOAD).unwrap();
        }
        f.flush().unwrap();
        path
    }

    /// Write a safetensors directory with an optional single named F16 tensor
    /// (shape [4], 8 zero bytes of payload).  `None` writes an empty header `{}`;
    /// `Some(name)` writes exactly that tensor into the safetensors header so
    /// the fixture looks like a real dir-format model.
    fn write_safetensors_dir(
        dir: &std::path::Path,
        name: &str,
        config_json: &str,
        tensor_name: Option<&str>,
    ) -> std::path::PathBuf {
        let model_dir = dir.join(name);
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), config_json).unwrap();

        let mut st = std::fs::File::create(model_dir.join("model.safetensors")).unwrap();
        match tensor_name {
            None => {
                // Empty safetensors file with minimal header {}
                st.write_all(&2u64.to_le_bytes()).unwrap();
                st.write_all(b"{}").unwrap();
            }
            Some(tname) => {
                let header = serde_json::json!(
                    {tname: {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}}
                );
                let hdr = header.to_string();
                let hdr_bytes = hdr.as_bytes();
                st.write_all(&(hdr_bytes.len() as u64).to_le_bytes())
                    .unwrap();
                st.write_all(hdr_bytes).unwrap();
                st.write_all(&[0u8; 8]).unwrap();
            }
        }
        st.flush().unwrap();
        model_dir
    }

    fn tmp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    // ── classify_source error tests ──────────────────────────────────

    /// Zero carrier match → `classify_source` must return an error.
    #[test]
    fn classify_source_zero_carrier() {
        let dir = tmp_dir();
        let path = write_hfq_fixture(
            dir.path(),
            "unknown.hfq",
            99,
            &config_wrapped(&r#"{"model_type":"unknown"}"#),
            &[],
        );
        let src = ModelSource::from_path(path.to_str().unwrap()).unwrap();
        let err = match classify_source(&src) {
            Err(e) => e,
            Ok(_) => panic!("expected error for unknown arch_id, got Ok"),
        };
        assert!(
            err.contains("no carrier"),
            "expected 'no carrier' error, got: {err}"
        );
    }

    // ── Cross-namespace consistency ──────────────────────────────────

    #[test]
    fn source_namespace_behavior_hfq_vs_dir() {
        let dir = tmp_dir();

        let phfq = write_hfq_fixture(
            dir.path(),
            "hfq_qk.hfq",
            0,
            &config_wrapped(&llama_cfg(&[])),
            &["model.layers.0.self_attn.q_norm.weight"],
        );
        let shfq = ModelSource::from_path(phfq.to_str().unwrap()).unwrap();

        let pdir = write_safetensors_dir(
            dir.path(),
            "hfqvsdir_llama_qk",
            &llama_cfg(&[("architectures", &serde_json::json!(["LlamaForCausalLM"]))]),
            Some("model.layers.0.self_attn.q_norm.weight"),
        );
        let sdir = ModelSource::from_path(pdir.to_str().unwrap()).unwrap();

        let hfq_variant = classify_source(&shfq).unwrap().1;
        let dir_variant = classify_source(&sdir).unwrap().1;
        assert_eq!(
            hfq_variant, dir_variant,
            "HFQ and Dir QK-norm LLaMA must produce the same variant, got HFQ={hfq_variant:?} Dir={dir_variant:?}"
        );
        assert_eq!(hfq_variant, ModelVariant::LlamaQkNorm);
    }

    // ── Focused test: None vs Some tensor in safetensors dir ──────────

    #[test]
    fn safetensors_dir_none_vs_some_tensor() {
        let dir = tmp_dir();

        // None → empty safetensors header
        let none_dir = write_safetensors_dir(dir.path(), "none_tensor", "{}", None);
        let none_bytes = std::fs::read(none_dir.join("model.safetensors")).unwrap();
        let hdr_len = u64::from_le_bytes(none_bytes[0..8].try_into().unwrap());
        assert_eq!(
            &none_bytes[8..8 + hdr_len as usize],
            b"{}",
            "None must produce empty header {{}}"
        );

        // Some("test.weight") → single-entry header
        let some_dir = write_safetensors_dir(dir.path(), "some_tensor", "{}", Some("test.weight"));
        let some_bytes = std::fs::read(some_dir.join("model.safetensors")).unwrap();
        let hdr_len2 = u64::from_le_bytes(some_bytes[0..8].try_into().unwrap());
        let hdr_val: serde_json::Value =
            serde_json::from_slice(&some_bytes[8..8 + hdr_len2 as usize]).unwrap();
        assert!(
            hdr_val.get("test.weight").is_some(),
            "Some must produce header with 'test.weight' key, got: {hdr_val}"
        );
        // Exactly one entry — no extra tensors
        assert_eq!(
            hdr_val.as_object().map(|m| m.len()),
            Some(1),
            "Some must produce exactly one tensor entry"
        );
    }

    // ── Table-driven registry-seam classify_source tests ──────────────

    /// Each case builds a source through `from_path`, resolves it through
    /// `classify_source`, and asserts the expected carrier name + variant.
    #[test]
    fn classify_source_table() {
        struct Case {
            name: &'static str,
            /// Callback that writes fixture files and returns a path string.
            build: fn(&std::path::Path) -> String,
            expect_carrier: &'static str,
            expect_variant: ModelVariant,
        }

        fn hfq_path(dir: &std::path::Path, arch_id: u32, meta: &str, tensors: &[&str]) -> String {
            write_hfq_fixture(dir, "m.hfq", arch_id, meta, tensors)
                .to_string_lossy()
                .to_string()
        }

        fn dir_path(dir: &std::path::Path, name: &str, cfg: &str) -> String {
            write_safetensors_dir(dir, name, cfg, None)
                .to_string_lossy()
                .to_string()
        }

        let cases: &[Case] = &[
            // ── HFQ carriers (arch_id namespace) ──────────────────────
            Case {
                name: "qwen35-hfq-dense",
                build: |d| hfq_path(d, 5, &config_wrapped(&qwen35_cfg(&[])), &[]),
                expect_carrier: "qwen35",
                expect_variant: ModelVariant::Qwen35Dense,
            },
            Case {
                name: "qwen35-hfq-moe",
                build: |d| {
                    hfq_path(
                        d,
                        6,
                        &config_wrapped(&qwen35_cfg(&[("num_experts", &serde_json::json!(8))])),
                        &[],
                    )
                },
                expect_carrier: "qwen35",
                expect_variant: ModelVariant::Qwen35Moe,
            },
            Case {
                name: "qwen35-hfq-vl",
                build: |d| {
                    hfq_path(
                        d,
                        5,
                        &config_wrapped(&qwen35_cfg(&[(
                            "vision_config",
                            &serde_json::json!({"hidden_size": 1024}),
                        )])),
                        &["model.visual.patch_embed.proj.weight"],
                    )
                },
                expect_carrier: "qwen35",
                expect_variant: ModelVariant::Qwen35Vl,
            },
            Case {
                name: "llama-hfq-qknorm",
                build: |d| {
                    hfq_path(
                        d,
                        0,
                        &config_wrapped(&llama_cfg(&[])),
                        &["model.layers.0.self_attn.q_norm.weight"],
                    )
                },
                expect_carrier: "llama",
                expect_variant: ModelVariant::LlamaQkNorm,
            },
            Case {
                name: "llama-hfq-noqknorm",
                build: |d| hfq_path(d, 0, &config_wrapped(&llama_cfg(&[])), &[]),
                expect_carrier: "llama",
                expect_variant: ModelVariant::LlamaNoQkNorm,
            },
            Case {
                name: "llama-hfq-qwen3",
                build: |d| hfq_path(d, 1, &config_wrapped(&qwen3_cfg(&[])), &[]),
                expect_carrier: "llama",
                expect_variant: ModelVariant::PlainQwen3,
            },
            // Qwen3 (arch_id=1) with QK-norm tensor present must remain
            // PlainQwen3 — has_qk_norm only applies to LLaMA/Mistral.
            Case {
                name: "qwen3-hfq-qknorm-still-plain",
                build: |d| {
                    hfq_path(
                        d,
                        1,
                        &config_wrapped(&qwen3_cfg(&[])),
                        &["model.layers.0.self_attn.q_norm.weight"],
                    )
                },
                expect_carrier: "llama",
                expect_variant: ModelVariant::PlainQwen3,
            },
            Case {
                name: "lfm2-hfq-dense",
                build: |d| {
                    hfq_path(
                        d,
                        11,
                        &config_wrapped(&lfm2_cfg(&[
                            ("num_experts", &serde_json::json!(0)),
                            ("model_type", &serde_json::json!("lfm2")),
                        ])),
                        &[],
                    )
                },
                expect_carrier: "lfm2moe",
                expect_variant: ModelVariant::Lfm2Dense,
            },
            Case {
                name: "lfm2-hfq-moe",
                build: |d| {
                    hfq_path(
                        d,
                        11,
                        &config_wrapped(&lfm2_cfg(&[
                            ("model_type", &serde_json::json!("lfm2_moe")),
                            ("num_experts", &serde_json::json!(8)),
                            ("num_experts_per_tok", &serde_json::json!(2)),
                            ("moe_intermediate_size", &serde_json::json!(1024)),
                        ])),
                        &[],
                    )
                },
                expect_carrier: "lfm2moe",
                expect_variant: ModelVariant::Lfm2Moe,
            },
            Case {
                name: "deepseek4-hfq",
                build: |d| {
                    hfq_path(
                        d,
                        9,
                        &config_wrapped(&r#"{"model_type":"deepseek_v4"}"#),
                        &[],
                    )
                },
                expect_carrier: "deepseek4",
                expect_variant: ModelVariant::Deepseek4,
            },
            Case {
                name: "minimax-hfq",
                build: |d| {
                    hfq_path(
                        d,
                        10,
                        &config_wrapped(&r#"{"model_type":"minimax_m2"}"#),
                        &[],
                    )
                },
                expect_carrier: "minimax",
                expect_variant: ModelVariant::Minimax,
            },
            Case {
                name: "qwen2-hfq",
                build: |d| hfq_path(d, 7, &config_wrapped(&r#"{"model_type":"qwen2"}"#), &[]),
                expect_carrier: "qwen2",
                expect_variant: ModelVariant::Qwen2,
            },
            Case {
                name: "dots_ocr-hfq",
                build: |d| hfq_path(d, 8, &config_wrapped(&r#"{"model_type":"dots_ocr"}"#), &[]),
                expect_carrier: "dots_ocr",
                expect_variant: ModelVariant::DotsOcr,
            },
            Case {
                name: "cohere2moe-hfq",
                build: |d| {
                    hfq_path(
                        d,
                        12,
                        &config_wrapped(&r#"{"model_type":"cohere2_moe"}"#),
                        &[],
                    )
                },
                expect_carrier: "cohere2moe",
                expect_variant: ModelVariant::Cohere2Moe,
            },
            Case {
                name: "qwen2-dir",
                build: |d| dir_path(d, "qw2d", &r#"{"model_type":"qwen2"}"#),
                expect_carrier: "qwen2",
                expect_variant: ModelVariant::Qwen2,
            },
            // ── Dir carriers (namespace routing) ──────────────────────
            Case {
                name: "qwen35-dir-dense",
                build: |d| dir_path(d, "q35dd", &qwen35_cfg(&[])),
                expect_carrier: "qwen35",
                expect_variant: ModelVariant::Qwen35Dense,
            },
            Case {
                name: "qwen35-dir-moe",
                build: |d| {
                    dir_path(
                        d,
                        "q35dm",
                        &qwen35_cfg(&[("num_experts", &serde_json::json!(8))]),
                    )
                },
                expect_carrier: "qwen35",
                expect_variant: ModelVariant::Qwen35Moe,
            },
            Case {
                name: "qwen35-dir-vl",
                // Composite VL format: text_config + vision_config.
                // This is the canonical format where Qwen35Config.is_vl_text=true.
                build: |d| {
                    let cfg = r#"{"model_type":"qwen3.5","text_config":{"hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064},"vision_config":{"hidden_size":1024}}"#;
                    dir_path(d, "q35dv", cfg)
                },
                expect_carrier: "qwen35",
                expect_variant: ModelVariant::Qwen35Vl,
            },
            Case {
                name: "llama-dir-qknorm",
                build: |d| {
                    write_safetensors_dir(
                        d,
                        "llama_dqk",
                        &llama_cfg(&[]),
                        Some("model.layers.0.self_attn.q_norm.weight"),
                    )
                    .to_string_lossy()
                    .to_string()
                },
                expect_carrier: "llama",
                expect_variant: ModelVariant::LlamaQkNorm,
            },
            Case {
                name: "llama-dir-noqknorm",
                build: |d| dir_path(d, "llama_dnqk", &llama_cfg(&[])),
                expect_carrier: "llama",
                expect_variant: ModelVariant::LlamaNoQkNorm,
            },
            // Qwen3 directory (arch_id=1) with QK-norm tensor must remain
            // PlainQwen3 — has_qk_norm only applies to LLaMA/Mistral.
            Case {
                name: "qwen3-dir-qknorm-still-plain",
                build: |d| {
                    write_safetensors_dir(
                        d,
                        "qw3dqk",
                        &qwen3_cfg(&[]),
                        Some("model.layers.0.self_attn.q_norm.weight"),
                    )
                    .to_string_lossy()
                    .to_string()
                },
                expect_carrier: "llama",
                expect_variant: ModelVariant::PlainQwen3,
            },
            Case {
                name: "lfm2-dir-dense",
                build: |d| {
                    dir_path(
                        d,
                        "lfm2dd",
                        &lfm2_cfg(&[
                            ("num_experts", &serde_json::json!(0)),
                            ("model_type", &serde_json::json!("lfm2")),
                        ]),
                    )
                },
                expect_carrier: "lfm2moe",
                expect_variant: ModelVariant::Lfm2Dense,
            },
            Case {
                name: "lfm2-dir-moe",
                build: |d| {
                    dir_path(
                        d,
                        "lfm2dm",
                        &lfm2_cfg(&[
                            ("model_type", &serde_json::json!("lfm2_moe")),
                            ("num_experts", &serde_json::json!(8)),
                            ("num_experts_per_tok", &serde_json::json!(2)),
                            ("moe_intermediate_size", &serde_json::json!(1024)),
                        ]),
                    )
                },
                expect_carrier: "lfm2moe",
                expect_variant: ModelVariant::Lfm2Moe,
            },
        ];

        for case in cases {
            let dir = tmp_dir();
            let path_str = (case.build)(dir.path());
            let src = ModelSource::from_path(&path_str)
                .unwrap_or_else(|e| panic!("{}: from_path failed: {}", case.name, e));
            let result = classify_source(&src);
            let (carrier, variant) =
                result.unwrap_or_else(|e| panic!("{}: classify_source failed: {}", case.name, e));
            assert_eq!(
                carrier.name(),
                case.expect_carrier,
                "{}: carrier name mismatch",
                case.name
            );
            assert_eq!(
                variant, case.expect_variant,
                "{}: variant mismatch",
                case.name
            );
        }
    }

    /// HFQ fixture writer must produce nonzero data_size so that tensor_data
    /// returns a non-empty slice for each indexed tensor.  This test proves the
    /// VL tensor payload is physically present.
    #[test]
    fn hfq_vl_tensor_data_nonempty() {
        let dir = tmp_dir();
        let path = write_hfq_fixture(
            dir.path(),
            "vl_test.hfq",
            5,
            &config_wrapped(&qwen35_cfg(&[(
                "vision_config",
                &serde_json::json!({"hidden_size": 1024}),
            )])),
            &["model.visual.patch_embed.proj.weight"],
        );
        use hipfire_runtime::hfq::HfqFile;
        let hfq = HfqFile::open(&path).unwrap();
        let (info, data) = hfq
            .tensor_data("model.visual.patch_embed.proj.weight")
            .expect("VL tensor must be found");
        assert!(
            !data.is_empty(),
            "VL tensor payload must be non-empty, got {} bytes (data_size={})",
            data.len(),
            info.data_size,
        );
    }

    /// A flat VL directory (vision_config without text_config and without VL tensor)
    /// must be rejected as unclassifiable, not silently classified as dense.
    #[test]
    fn malformed_vl_directory_rejected() {
        let dir = tmp_dir();
        // Flat config: vision_config but NO text_config → unclassifiable VL
        let cfg = &qwen35_cfg(&[("vision_config", &serde_json::json!({"hidden_size": 1024}))]);
        let model_dir = write_safetensors_dir(dir.path(), "q35_bad_vl", cfg, None);
        let src = ModelSource::from_path(model_dir.to_str().unwrap()).unwrap();
        let err = match classify_source(&src) {
            Err(e) => e,
            Ok(_) => panic!("expected unclassifiable VL error, got Ok"),
        };
        assert!(
            err.contains("unclassifiable"),
            "must mention unclassifiable, got: {err}"
        );
    }

    /// A carrier using the DEFAULT `classify_parallel_variant` must produce an
    /// error that includes `CAP-001`, `self.name()`, and `src.describe()`.
    #[test]
    fn default_classify_rejection_includes_cap001_name_and_describe() {
        // Use a carrier with NO override (default trait method).
        // The test itself creates one via a carrier that doesn't override.
        // Actually, every carrier in the registry overrides classify_parallel_variant
        // (some return fixed Ok). We need a synthetic carrier that uses the default.
        struct DefaultCarrier;
        impl Carrier for DefaultCarrier {
            fn name(&self) -> &'static str {
                "test_default"
            }
            fn claims_arch_id(&self, _arch_id: u32, _is_dir: bool) -> bool {
                true
            }
            fn load(
                &self,
                _src: ModelSource,
                _ctx: &mut LoadCtx,
                _token: &CarrierLoadToken,
            ) -> Result<LoadedModel, String> {
                Err("unused".into())
            }
        }

        let dir = tmp_dir();
        let path = write_hfq_fixture(
            dir.path(),
            "dummy.hfq",
            42,
            &config_wrapped(&r#"{"model_type":"dummy"}"#),
            &[],
        );
        let src = ModelSource::from_path(path.to_str().unwrap()).unwrap();
        let result = DefaultCarrier.classify_parallel_variant(&src);
        let err = result.unwrap_err();
        let want = format!(
            "{}: CAP-001 variant classification unsupported for {}",
            DefaultCarrier.name(),
            src.describe()
        );
        assert_eq!(err, want, "default error must match exactly");
    }
}
