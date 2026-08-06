// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ + safetensors-dir).

use crate::spec_build::Qwen35SlotGuard;
use crate::Carrier;
use crate::{
    finish_qwen35_load, resolve_chat_template, resolve_chat_template_overrides, LoadedModel,
    ModelState,
};
use hipfire_arch_minimax::{config_from_safetensors, load_weights_from_safetensors, MiniMaxState};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::spec::{InPlaceGuard, SpecEmit, SpecEmitCtx, SpecTargetGuard};

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
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("qwen2: pipeline-parallel (pp>1) unsupported".into());
        }
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_qwen2::load_qwen2_bundle(src, ctx)?;
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). Qwen2
        // (arch_id=7, e.g. VibeThinker) impls `SpecTarget`, so it can be driven by
        // the arch-generic spec loop with no draft model. `None` ⇒ AR-only.
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
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
            )
        })
    }
}

// ─── Qwen35Carrier ───────────────────────────────────────────────────

fn kv_mode_from_ctx(ctx: &LoadCtx) -> String {
    ctx.kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone())
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
) -> Result<LoadedModel, String> {
    let pp = ctx.pp;
    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq_file)
        .map_err(|e| format!("failed to read Qwen3.5 config: {e}"))?;
    let mut gpus = match hipfire_config::developer_var("HIPFIRE_PP_LAYERS")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(spec) => {
            let counts: Result<Vec<usize>, _> =
                spec.split(',').map(|s| s.trim().parse::<usize>()).collect();
            let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
            if counts.len() != pp {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
                    counts.len(),
                    pp
                ));
            }
            let sum: usize = counts.iter().sum();
            if sum != config.n_layers {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
                    sum, config.n_layers
                ));
            }
            hipfire_runtime::multi_gpu::Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
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
        // Adaptive is single-GPU only; PP path never engages the controller.
        kv_adaptive: None,
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
            pp,
            gpus,
            scratch_set,
            la_to_device,
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
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.kv_backend == KvBackend::Vmm && ctx.pp > 1 {
            return Err(
                "qwen35: KV backend 'vmm' currently requires pp=1; use 'contiguous' for pipeline parallelism"
                    .into(),
            );
        }
        if ctx.kv_backend == KvBackend::Vmm && ctx.cask.sidecar.is_some() {
            return Err(
                "qwen35: KV backend 'vmm' does not yet support CASK/TriAttention eviction; disable the sidecar or use 'contiguous'"
                    .into(),
            );
        }
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
            ModelSource::Hfq(mut hfq_file) => {
                // ── pp>1 path (pipeline-parallel) — extracted helper ──
                if ctx.pp > 1 {
                    return load_qwen35_pp(hfq_file, meta, ctx);
                }

                // ── pp=1 path (single-GPU) ────────────────────
                let physical_cap = if ctx.cask.sidecar.is_some() {
                    let env_override = hipfire_config::developer_var("HIPFIRE_KV_PHYSICAL_CAP")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok());
                    let safety = 256usize;
                    let floor = ctx.cask.budget + ctx.cask.beta + 4;
                    let derived = ctx.cask.budget + ctx.cask.beta + safety;
                    env_override.unwrap_or(derived).clamp(floor, ctx.max_seq)
                } else {
                    ctx.max_seq
                };

                // VL detection — loads weights from hfq_file in-place
                let (vision_config, vision_weights) = {
                    use hipfire_arch_qwen35_vl::Qwen35Vl;
                    use hipfire_runtime::arch::Architecture;
                    let has_vision = hfq_file
                        .tensor_data("model.visual.patch_embed.proj.weight")
                        .is_some();
                    let vc = Qwen35Vl::config_from_hfq(&hfq_file).ok();
                    match vc {
                        Some(vc) if has_vision => {
                            let vw = Qwen35Vl::load_weights(&mut hfq_file, &vc, ctx.gpu)
                                .map_err(|e| eprintln!("  VL weight load failed: {e}"))
                                .ok();
                            eprintln!(
                                "  VL model: vision encoder (hidden={}, layers={})",
                                vc.hidden_size, vc.num_layers
                            );
                            (Some(vc), vw)
                        }
                        _ => (None, None),
                    }
                };

                // Trunk bundle after optional VL upload. On bundle failure, reclaim
                // any vision weights already on-device (HFQ is single-pass: VL must
                // load from the same file before the carrier consumes it).
                let bundle = match hipfire_arch_qwen35::load_qwen35_bundle(
                    ModelSource::Hfq(hfq_file),
                    ctx,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        if let Some(vw) = vision_weights {
                            vw.free_gpu(ctx.gpu);
                        }
                        return Err(e);
                    }
                };
                finish_qwen35_load(
                    bundle,
                    meta.tokenizer,
                    physical_cap,
                    meta.arch_id,
                    meta.chat_template,
                    ctx,
                    vision_config,
                    vision_weights,
                )
            }
            ModelSource::Dir(source) => {
                let config = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source)
                    .map_err(|e| format!("failed to parse Qwen3.5 config from config.json: {e}"))?;
                if ctx.draft_path.is_some() {
                    eprintln!("  warning: DFlash (speculative decoding) is not supported for safetensors Dir sources; draft_path ignored");
                }
                if ctx.cask.sidecar.is_some() {
                    eprintln!("  warning: CASK eviction is not supported for safetensors Dir sources; eviction sidecar ignored");
                }
                // CPU-only before any GPU ownership (parity with HFQ carrier).
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

                let mut paro_source =
                    hipfire_arch_qwen35::qwen35::ParoSource::new(&source, &config)
                        .map_err(|e| format!("ParoSource::new: {e:?}"))?;
                let paro_layout = hipfire_arch_qwen35::qwen35::Layout::single(config.n_layers);
                let weights = hipfire_arch_qwen35::qwen35::load_weights(
                    &mut paro_source,
                    std::slice::from_mut(ctx.gpu),
                    &paro_layout,
                )
                .map_err(|e| format!("load_weights: {e:?}"))?;
                hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);

                // Staged GPU free on every post-weight error (VMM arenas via free_gpu).
                let kv_cache = match hipfire_runtime::llama::KvCache::from_mode_with_backend(
                    mode,
                    ctx.kv_backend,
                    hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                    &dims,
                ) {
                    Ok(k) => k,
                    Err(e) => {
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("KvCache: {e}"));
                    }
                };

                let dn_state = match hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant(
                    ctx.gpu, &config, dn_quant,
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        let mut note = format!("DeltaNetState::new_with_quant: {e:?}");
                        if let Err(fe) = kv_cache.free_gpu(ctx.gpu) {
                            note = format!("{note}; cleanup also failed: {fe}");
                        }
                        weights.free_gpu(ctx.gpu);
                        return Err(note);
                    }
                };

                let scratch = match hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(
                    ctx.gpu,
                    &config,
                    2048,
                    ctx.max_seq,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let mut note = format!("Qwen35Scratch::new_with_kv_max: {e:?}");
                        if let Err(fe) = kv_cache.free_gpu(ctx.gpu) {
                            note = format!("{note}; cleanup also failed: {fe}");
                        }
                        dn_state.free_gpu(ctx.gpu);
                        weights.free_gpu(ctx.gpu);
                        return Err(note);
                    }
                };

                let bundle = hipfire_arch_qwen35::Qwen35Bundle {
                    config,
                    weights,
                    scratch,
                    kv_cache,
                    dn_state,
                    // Dir/safetensors path does not engage adaptive (HFQ carrier only).
                    kv_adaptive: None,
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
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
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
                hipfire_arch_llama::load_llama_bundle(ModelSource::Hfq(hfq), ctx)?
            }
            ModelSource::Dir(source) => {
                let config =
                    hipfire_runtime::hfq::config_from_safetensors_llama(&source).map_err(|e| {
                        format!("failed to parse LLaMA/Qwen3 config from config.json: {e}")
                    })?;
                let weights =
                    hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, ctx.gpu)
                        .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;
                hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
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
                let kv = hipfire_runtime::llama::KvCache::from_mode(
                    mode,
                    hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                    &dims,
                )
                .map_err(|e| format!("KvCache: {e}"))?;
                let scratch = hipfire_runtime::llama::ForwardScratch::new_with_max_seq(
                    ctx.gpu,
                    &config,
                    ctx.max_seq,
                )
                .map_err(|e| format!("ForwardScratch::new_with_max_seq: {e:?}"))?;
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
            let vocab = assets.config.vocab_size;

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
            let conf_threshold =
                hipfire_config::developer_var("HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .or(ctx.spec.dspark_conf_threshold)
                    .unwrap_or(0.1f32);

            eprintln!(
                "  llama DSpark speculator enabled (sidecar, block={}, conf_threshold={:.2})",
                block, conf_threshold
            );
            let body = hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
                assets,
                &dspark_weights.cfg,
                ctx.gpu,
            )
            .map_err(|e| format!("llama DSpark body build failed: {e}"))?;
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
                0.5,
            ))
        } else if let Some(dp) = ctx.draft_path {
            // Peek at the draft's arch_id without consuming the path; the builder
            // opens it again internally.
            match hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(dp)) {
                Ok(draft_hfq) if draft_hfq.arch_id == 20 => {
                    // Parse DflashConfig to validate the cross-attention concat invariant
                    // (review finding L4): the drafter's hidden must equal the target dim.
                    let draft_cfg = hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq)
                        .ok_or_else(|| {
                            format!(
                                "DFlash draft '{}' has arch_id=20 but missing or malformed \
                                 'dflash' metadata block",
                                dp
                            )
                        })?;
                    if bundle.config.dim != draft_cfg.hidden {
                        return Err(format!(
                            "DFlash draft '{}' hidden={} != target dim={} \
                                 (cross-attention concat invariant L4: drafter hidden \
                                 must equal target residual dim)",
                            dp, draft_cfg.hidden, bundle.config.dim
                        ));
                    }
                    // Drop the peek handle before the builder reopens it.
                    drop(draft_hfq);
                    let spec = hipfire_runtime::dflash_generic::build_generic_dflash_speculator(
                        ctx.gpu,
                        dp,
                        &mut bundle,
                        ctx.max_seq,
                    )
                    .map_err(|e| format!("DFlash generic speculator build failed: {e}"))?;
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
                        None,
                        true,
                        ctx.max_seq,
                        ctx.spec,
                    )
                }
            }
        } else {
            // No draft configured: opt-in model-free n-gram (HIPFIRE_NGRAM_DRAFT=1) or None.
            crate::spec_build::build_speculator(
                meta.arch_id,
                None,
                None,
                true,
                ctx.max_seq,
                ctx.spec,
            )
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
            )
        })
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
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
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
        hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
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
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            qwen2_state: Some(state),
            dots_ocr_config: Some(config),
            dots_ocr_weights: Some(weights),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
            )
        })
    }
}

// ─── Deepseek4Carrier ────────────────────────────────────────────────

fn apply_deepseek4_experts_per_token(
    config: &mut hipfire_arch_deepseek4::DeepseekV4Config,
    requested: Option<usize>,
) -> Result<(), String> {
    let Some(requested) = requested else {
        return Ok(());
    };
    let checkpoint = config.num_experts_per_tok;
    if requested == 0 || requested > checkpoint {
        return Err(format!(
            "deepseek4: experts-per-token override must be in 1..={checkpoint}, got {requested}"
        ));
    }
    if requested != checkpoint {
        eprintln!("deepseek4: runtime experts-per-token override {checkpoint} -> {requested}");
        config.num_experts_per_tok = requested;
    }
    Ok(())
}

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
            Some(ModelState::Deepseek4Heterogeneous(_)) => {
                Err("deepseek4 heterogeneous route is direct-AR only until G6".into())
            }
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
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "deepseek4: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "deepseek4: safetensors + pp>1 unsupported",
            }
            .into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        if !matches!(
            ctx.deepseek4_compute_placement,
            hipfire_config::Deepseek4ComputePlacement::Single
        ) {
            if !matches!(&src, ModelSource::Hfq(_)) {
                return Err(
                    "deepseek4 heterogeneous placement requires the frozen MQ2R HFQ artifact"
                        .into(),
                );
            }
            if ctx
                .deepseek4_experts_per_token
                .is_some_and(|value| value != 6)
            {
                return Err("deepseek4 heterogeneous placement requires checkpoint top-k 6".into());
            }
            if ctx.draft_path.is_some() || ctx.spec.dspark == Some(true) {
                return Err(
                    "deepseek4 heterogeneous placement is direct-AR only until G6/G7".into(),
                );
            }
            let artifact =
                hipfire_arch_deepseek4::DeepseekV4VerifiedArtifact::verify(ctx.path.as_ref())?;
            let plan = hipfire_arch_deepseek4::DeepseekV4HeterogeneousLoadPlan {
                placement: ctx.deepseek4_compute_placement.clone(),
                prefill_max_batch: 1024,
                ..Default::default()
            };
            let model = hipfire_arch_deepseek4::DeepseekV4HeterogeneousModel::load_verified(
                &artifact, plan,
            )?;
            let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<｜end▁of▁sentence｜>"]);
            let advertised_context = model.config.max_position_embeddings;
            return Ok(LoadedModel {
                state: Some(crate::ModelState::Deepseek4Heterogeneous(
                    crate::Deepseek4HeterogeneousBundle { model, eos_tok },
                )),
                ..LoadedModel::skeleton(
                    meta.arch_id,
                    meta.tokenizer,
                    advertised_context,
                    advertised_context,
                    ctx.path.to_string(),
                    meta.chat_template,
                )
            });
        }

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
                apply_deepseek4_experts_per_token(&mut config, ctx.deepseek4_experts_per_token)?;
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
                apply_deepseek4_experts_per_token(&mut config, ctx.deepseek4_experts_per_token)?;
                config.load_dspark = load_dspark;
                let weights = deepseek4::DeepseekV4::load_weights_from_safetensors(
                    &source, &config, ctx.gpu,
                )?;
                (config, weights)
            }
        };
        let state = deepseek4::DeepseekV4State::new(&config)?;
        let pbs_max_batch: usize = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_PP_BATCH")
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
            // spec_k resolution MUST mirror daemon.rs:9349 (HIPFIRE_DEEPSEEK4_SPEC_K →
            // HIPFIRE_MTP_K → default 2) so T4's spec.k() matches the bespoke loop's window.
            let max_n: usize = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_SPEC_K")
                .ok()
                .and_then(|s| s.parse().ok())
                .or_else(|| Some(hipfire_runtime::config::get().mtp_k))
                .unwrap_or(2);
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
        let advertised_context = config.max_position_embeddings;
        eprintln!(
            "  deepseek4 compressed cache: automatic VMM growth to advertised context {advertised_context}"
        );
        Ok(LoadedModel {
            state: Some(crate::ModelState::Deepseek4(deepseek4::Deepseek4Bundle {
                config,
                weights,
                state,
                eos_tok,
            })),
            speculator,
            deepseek4_pbs: Some(pbs),
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                advertised_context,
                advertised_context,
                ctx.path.to_string(),
                meta.chat_template,
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
        // Shared ChatML emitter (same one qwen2 reuses): MiniMax-M2 is ChatML
        // (`<|im_end|>`), so the generic think/tool-call/EOS scanning applies.
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 10
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
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
        hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);

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
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
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
        // Shared ChatML emitter (same one qwen2/minimax reuse): LFM2.5 is ChatML
        // (`<|im_end|>`), no bespoke marker state machine.
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 11
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
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
        hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);

        let state = lfm2moe::lfm2moe::Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
        let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<|im_end|>", "</s>", "<|endoftext|>"]);
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). LFM2.5-MoE
        // (arch_id=11) impls `SpecTarget` with conv-state snapshot/rollback in
        // `verify_block`/`commit_prefix`, so it can be driven by the arch-generic
        // spec loop with no draft model. `None` ⇒ AR-only (`generate_lfm2moe`).
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
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
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
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
                let mut lm =
                    crate::load_cohere2moe(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)?;
                // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1).
                lm.speculator = crate::spec_build::build_speculator(
                    meta.arch_id,
                    None,
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
                    )
                })
            }
        }
    }
}
