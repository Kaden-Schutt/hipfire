// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ + safetensors-dir).

use crate::parallel_capability::ModelVariant;
use crate::spec_build::Qwen35SlotGuard;
use crate::Carrier;
use crate::{
    finish_qwen35_load, resolve_chat_template, resolve_chat_template_overrides, LoadedModel,
};
use hipfire_arch_minimax::{config_from_safetensors, load_weights_from_safetensors, MiniMaxState};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::spec::{InPlaceGuard, SpecEmit, SpecEmitCtx, SpecTargetGuard};
use std::any::Any;

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
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen2::Qwen2Bundle>()
        }) {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
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
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.qwen2_mut().unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        for &tok in synthetic {
            if hipfire_arch_qwen2::qwen2::forward_step(gpu, weights, config, state, tok).is_err() {
                ok = false;
                break;
            }
        }
        Some(ok)
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
            state: Some(Box::new(bundle)),
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
        13 => Some(super::GEMMA4_TEMPLATE.to_string()),
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
    let mut gpus = match parse_pp_layer_bands(
        hipfire_config::developer_var("HIPFIRE_PP_LAYERS")
            .ok()
            .as_deref(),
        pp,
        config.n_layers,
    )? {
        Some(counts) => {
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
    let kv = <hipfire_runtime::llama::KvCache as hipfire_runtime::llama::KvCacheExt>::from_mode(
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
        pp_scratch_set: Some(scratch_set),
        vision_config: None,
        vision_weights: None,
        qwen35_decode_batch: None,
    };
    Ok(LoadedModel {
        state: Some(Box::new(bundle)),
        ..LoadedModel::skeleton_pp(
            meta.arch_id,
            meta.tokenizer,
            ctx.max_seq,
            ctx.max_seq,
            ctx.path.to_string(),
            meta.chat_template,
            pp,
            gpus,
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
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
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
                    // Match the production loader: use tensor_data (not
                    // find_tensor_info) so that only tensors with physically
                    // present data count.
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
            ModelSource::Dir(source) => match source.arch_id() {
                6 => Ok(ModelVariant::Qwen35Moe),
                5 => {
                    // VL tower presence is the classifier for dense (arch 5);
                    // the pre-merge composite text_config+vision_config check
                    // is a Phase 2 refinement (classify_vl removed from the
                    // arch crate during the mainline refactor).
                    if source
                        .tensor_info("model.visual.patch_embed.proj.weight")
                        .is_some()
                    {
                        Ok(ModelVariant::Qwen35Vl)
                    } else {
                        Ok(ModelVariant::Qwen35Dense)
                    }
                }
                other => Err(format!("qwen35 safetensors: unexpected arch_id {}", other)),
            },
        }
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: true,
            supports_ep_batch: true,
            dflash: Some(saddle_core::caps::DflashKind::Qwen),
            supports_mtp: true,
            spec_excludes_adaptive: true,
            semantic_contract_version: Some(2),
            has_deltanet: true,
            // Architectural capability: Qwen3.5-VL tower is optional per-model
            // (probed via `model.visual.patch_embed.proj.weight`); this flag
            // declares that the arch CAN accept images when that tower is
            // present. Per-instance gating still checks `LoadedModel::vision_config`.
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::QwenJinja,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.qwen35_mut().unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv_cache;
        let dn = &mut b.dn_state;
        Some(
            hipfire_arch_qwen35::qwen35::forward_prefill_batch(
                gpu, weights, config, synthetic, 0, kv, dn, scratch, None, None, None, None,
            )
            .is_ok(),
        )
    }
    fn bench_decode_prime(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
    ) -> Option<Option<String>> {
        let b = m.qwen35_mut().unwrap();
        Some(
            hipfire_arch_qwen35::qwen35::forward_prefill_batch(
                gpu,
                &b.weights,
                &b.config,
                synthetic,
                0,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
                None,
                None,
                None,
                None,
            )
            .err()
            .map(|e| format!("{e:?}")),
        )
    }
    fn bench_decode_run(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        context: usize,
        iterations: usize,
        _decode_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.qwen35_mut().unwrap();
        let mut ok = true;
        for i in 0..iterations {
            let token = 101 + (i as u32 % 1000);
            if hipfire_arch_qwen35::qwen35::forward_scratch(
                gpu,
                &b.weights,
                &b.config,
                token,
                context + i,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.kv_backend == KvBackend::Vmm && ctx.pp > 1 {
            return Err(
                "qwen35: KV backend 'vmm' currently requires pp=1; use 'contiguous' for pipeline parallelism"
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
                let physical_cap = ctx.cask.physical_cap(ctx.max_seq)?;

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
                        // The bundle error carries exact-retention cleanup
                        // owners; retry once before reducing to a String.
                        // Whatever still fails is logged — never dropped.
                        let hipfire_arch_qwen35::carrier::Qwen35BundleLoadError {
                            message,
                            cleanup,
                        } = e;
                        if let Some(cf) = cleanup {
                            let first_failures = cf.num_failed();
                            match cf.retry(ctx.gpu) {
                                Ok(()) => {
                                    if first_failures > 0 {
                                        eprintln!(
                                            "  qwen35 bundle load: first cleanup attempt failed \
                                             ({first_failures} owner(s)) but retry succeeded"
                                        );
                                    }
                                }
                                Err(remaining) => {
                                    let pending = remaining.num_failed();
                                    let summaries = remaining.error_summaries();
                                    hipfire_runtime::gpu_cleanup::enqueue_cleanup_failure(
                                        remaining,
                                    );
                                    eprintln!(
                                        "  qwen35 bundle load cleanup retry failed \
                                         ({pending} owner(s) enqueued to the retained-owner \
                                         backlog): {}",
                                        summaries.join("; ")
                                    );
                                }
                            }
                        }
                        return Err(message);
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
                let kv_cache = match <hipfire_runtime::llama::KvCache as hipfire_runtime::llama::KvCacheExt>::from_mode_with_backend(
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
                    pp_scratch_set: None,
                    vision_config: None,
                    vision_weights: None,
                    qwen35_decode_batch: None,
                };
                Ok(LoadedModel {
                    state: Some(Box::new(bundle)),
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
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        }) {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
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
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: Some(saddle_core::caps::DflashKind::Llama),
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.llama_mut().unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_runtime::llama::forward_scratch(
                gpu, weights, config, tok, i, kv, scratch, 0.0, 1.0, 42, 0, 1.0,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
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

        let mut bundle = hipfire_arch_llama::load_llama_bundle(src, ctx)?;

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
            state: Some(Box::new(bundle)),
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
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::DotsOcr)
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.dots_ocr_mut().unwrap();
        let state = &mut bundle.state;
        let config = &bundle.config;
        let weights = &bundle.weights;
        let mut ok = true;
        for &tok in synthetic {
            if hipfire_arch_qwen2::qwen2::forward_step(gpu, &weights.text, &config.text, state, tok)
                .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
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

        let bundle = hipfire_arch_dots_ocr::load_dots_ocr_bundle(src, ctx)?;
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
            state: Some(Box::new(bundle)),
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

pub struct Deepseek4Carrier;
impl Carrier for Deepseek4Carrier {
    fn name(&self) -> &'static str {
        "deepseek4"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        if state
            .as_ref()
            .is_some_and(|s| (s.as_ref() as &dyn Any).is::<crate::Deepseek4HeterogeneousBundle>())
        {
            Err("deepseek4 heterogeneous route is direct-AR only until G6".into())
        } else if let Some(b) = state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        }) {
            Ok(Box::new(InPlaceGuard { bundle: b }))
        } else {
            Err("deepseek4: spec target state mismatch".into())
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
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::DeepSeek4,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.0, 1.0, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m
            .state
            .as_mut()
            .and_then(|s| {
                (s.as_mut() as &mut dyn Any)
                    .downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
            })
            .unwrap();
        let pbs = b
            .pbs
            .as_mut()
            .expect("deepseek4_pbs missing on arch_id=9 bench_prefill");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let ok = hipfire_arch_deepseek4::forward::forward_prefill_batch_chunked(
            config, weights, state, gpu, synthetic, 0, pbs,
        )
        .is_ok();
        if ok {
            state.n_tokens = n as u64;
        }
        Some(ok)
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
        let compressor_cache =
            crate::resolve_deepseek4_compressor_cache_kv_mode(ctx.kv_mode_override)?;

        if !matches!(
            ctx.deepseek4_compute_placement,
            hipfire_config::Deepseek4ComputePlacement::Single
        ) {
            let model = hipfire_arch_deepseek4::load_deepseek4_heterogeneous_model(
                &src,
                ctx,
                compressor_cache,
            )?;
            let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<｜end▁of▁sentence｜>"]);
            let advertised_context = model.config.max_position_embeddings;
            return Ok(LoadedModel {
                state: Some(Box::new(crate::Deepseek4HeterogeneousBundle {
                    model,
                    eos_tok,
                })),
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
        let deepseek4::Deepseek4LoadParts {
            config,
            weights,
            state,
            pbs,
        } = deepseek4::load_deepseek4_bundle(src, ctx, compressor_cache)?;
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
            "  deepseek4 KV cache: automatic VMM growth to advertised context {advertised_context}"
        );
        Ok(LoadedModel {
            state: Some(Box::new(deepseek4::Deepseek4Bundle {
                config,
                weights,
                state,
                eos_tok,
                pbs: Some(pbs),
            })),
            speculator,
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
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::MiniMaxBundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
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
    fn classify_parallel_variant(&self, _src: &ModelSource) -> Result<ModelVariant, String> {
        Ok(ModelVariant::Minimax)
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 1.0, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.minimax_mut().expect("arch_id=10 requires minimax bundle");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_arch_minimax::forward::decode_step(
                config, weights, state, gpu, tok, i as u32,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
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
        let bundle = hipfire_arch_minimax::load_minimax_bundle(src, ctx)?;
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
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
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::Lfm2MoeBundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
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
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: true,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.1, 0.80, 1.05)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.lfm2moe_mut().expect("arch_id=11 requires lfm2moe bundle");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_arch_lfm2moe::forward::decode_step(
                config, weights, state, gpu, tok, i as u32,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
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
        let bundle = hipfire_arch_lfm2moe::load_lfm2moe_bundle(src, ctx)?;
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
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
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::Cohere2MoeBundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
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
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 0.95, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m
            .cohere2moe_mut()
            .expect("arch_id=12 requires cohere2moe bundle");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        if hipfire_arch_cohere2moe::forward::forward_batch_supported(weights) && synthetic.len() > 1
        {
            let mut i = 0;
            while i < synthetic.len() {
                let end = (i + 256).min(synthetic.len());
                let start_pos = state.n_tokens;
                if hipfire_arch_cohere2moe::forward::forward_batch(
                    config,
                    weights,
                    state,
                    gpu,
                    &synthetic[i..end],
                    start_pos,
                )
                .is_err()
                {
                    ok = false;
                    break;
                }
                i = end;
            }
        } else {
            for (i, &tok) in synthetic.iter().enumerate() {
                if hipfire_arch_cohere2moe::forward::decode_step(
                    config, weights, state, gpu, tok, i as u32,
                )
                .is_err()
                {
                    ok = false;
                    break;
                }
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("cohere2moe: pp>1 unsupported via registry".into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_cohere2moe::load_cohere2moe_bundle(src, ctx)?;
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
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

// ─── Gemma4Carrier ───────────────────────────────────────────────────

fn gemma4_use_lowered(
    enable_moe_block: bool,
    want_batched: bool,
    has_drafter: bool,
    is_e_series: bool,
) -> bool {
    enable_moe_block || (want_batched && !has_drafter && !is_e_series)
}

fn gemma4_validate_drafter_route(is_e_series: bool, has_drafter: bool) -> Result<(), String> {
    if is_e_series && has_drafter {
        return Err(
            "gemma4: E2B/E4B EAGLE spec-decode is not yet supported; load the E-series target without params.drafter"
                .into(),
        );
    }
    Ok(())
}

pub struct Gemma4Carrier;
impl Carrier for Gemma4Carrier {
    fn name(&self) -> &'static str {
        "gemma4"
    }
    fn spec_target_guard<'m>(
        &self,
        _state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        Err("gemma4: spec decode not yet wired (AR-only)".into())
    }
    fn make_spec_emitter<'a>(
        &self,
        _ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Err("gemma4: spec emitter not yet wired".into())
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 13 = gemma4_text (primary), 22 = gemma4_unified_assistant (EAGLE drafter sidecar).
        // The drafter file (22) is loaded via params.drafter path, not as a primary serve model,
        // but claiming it here lets the quantizer's 22-stamped draft file be probed without
        // "no carrier" error and keeps the two namespaces aligned. Primary serve of 22 alone
        // would still need a target model, so it naturally fails later in generate routing.
        matches!(arch_id, 13 | 22)
    }
    fn classify_parallel_variant(&self, src: &ModelSource) -> Result<ModelVariant, String> {
        match src {
            ModelSource::Hfq(hfq) => match hfq.arch_id {
                // 13 is the only primary Gemma4 topology id. arch 22 (EAGLE
                // drafter sidecar) is never a CAP-001 target — no row exists.
                13 => Ok(ModelVariant::Gemma4),
                other => Err(format!("gemma4: unexpected HFQ arch_id {}", other)),
            },
            ModelSource::Dir(source) => match source.arch_id() {
                13 => Ok(ModelVariant::Gemma4),
                other => Err(format!("gemma4 safetensors: unexpected arch_id {}", other)),
            },
        }
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: Some(2),
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::GemmaBoolean,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 0.95, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.gemma4_mut().unwrap();
        let config = &bundle.config;
        let weights = &bundle.weights;
        let state = &mut bundle.state;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_arch_gemma4::forward::decode_step(config, weights, state, gpu, tok, i as u32)
                .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("gemma4: pp>1 unsupported".into());
        }
        if ctx.draft_path.is_some() {
            return Err("params.draft (qwen3.5 DFlash) is not supported on arch_id=13 (Gemma 4). For Gemma 4 spec-decode pass the arch-22 EAGLE drafter via params.drafter instead.".to_string());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_gemma4::load_gemma4_bundle(src, ctx)?;
        match bundle {
            hipfire_arch_gemma4::Gemma4Bundle::Lowered(l) => {
                let eos_tok = resolve_eos_tok(
                    &meta.tokenizer,
                    &["<end_of_turn>", "<turn|>", "<eos>", "<|im_end|>"],
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
                    state: Some(Box::new(crate::Gemma4LoweredBundle {
                        config: l.config,
                        weights: l.weights,
                        scratch: l.scratch,
                        kv_sliding: l.kv_sliding,
                        kv_full: l.kv_full,
                        eos_tok,
                        cursor: hipfire_arch_gemma4::lowered::Gemma4Cursor::default(),
                        prefix_cache: crate::GemmaPrefixCache::default(),
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
            hipfire_arch_gemma4::Gemma4Bundle::Eager(e) => {
                let eos_tok = resolve_eos_tok(
                    &meta.tokenizer,
                    &["<end_of_turn>", "<turn|>", "<eos>", "<|im_end|>"],
                );
                let _ = &e.weights;
                // Optional EAGLE drafter (arch-22) — populated only when
                // `gemma4_drafter_path` is Some. Validates draft_len 1..=5,
                // arch_id 22, and backbone_hidden == target dim. On failure
                // logs and falls back to AR-only (mirrors PR's contract) to
                // avoid hard failing a valid target model due to a bad sidecar.
                let eagle = if let Some(dp) = ctx.gemma4_drafter_path {
                    let draft_len = crate::gemma4_eagle_spec_len(Some(ctx.gemma4_draft_len as u64))
                        .map_err(|e| format!("gemma4 drafter spec_len: {e}"))?;
                    match load_gemma4_eagle_state(dp, draft_len, &e.config, &e.weights, ctx.gpu) {
                        Ok(st) => {
                            eprintln!(
                                "  gemma4 EAGLE drafter loaded: {} (layers={}, hidden={}, draft_len={})",
                                dp, st.drafter_config.n_layers, st.drafter_config.hidden, st.draft_len,
                            );
                            Some(st)
                        }
                        Err(e) => {
                            eprintln!(
                                "  gemma4 EAGLE drafter load failed ({}): {} — falling back to AR only",
                                dp, e
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                let speculator = crate::spec_build::build_speculator(
                    meta.arch_id,
                    None,
                    None,
                    true,
                    ctx.max_seq,
                    ctx.spec,
                );
                Ok(LoadedModel {
                    state: Some(Box::new(crate::Gemma4Bundle {
                        config: e.config,
                        weights: e.weights,
                        state: e.state,
                        eos_tok,
                        eagle,
                        prefix_cache: crate::GemmaPrefixCache::default(),
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

fn load_gemma4_eagle_state(
    drafter_path: &str,
    draft_len: usize,
    target_cfg: &hipfire_arch_gemma4::config::Gemma4Config,
    target_weights: &hipfire_arch_gemma4::gemma4::Gemma4Weights,
    gpu: &mut rdna_compute::Gpu,
) -> Result<crate::Gemma4EagleState, String> {
    use std::path::Path;
    let dhfq = hipfire_runtime::hfq::HfqFile::open(Path::new(drafter_path))
        .map_err(|e| format!("open gemma4 drafter: {e}"))?;
    if dhfq.arch_id != 22 {
        return Err(format!(
            "gemma4 EAGLE drafter must be arch_id=22 (gemma4_unified_assistant); got arch_id={} — a DFlash draft goes in params.draft on qwen3.5 targets, not params.drafter",
            dhfq.arch_id
        ));
    }
    let dcfg = hipfire_arch_gemma4::drafter::Gemma4DrafterConfig::from_hfq(&dhfq)?;
    if dcfg.backbone_hidden != target_cfg.dim {
        return Err(format!(
            "drafter backbone_hidden ({}) != target hidden ({}) — this drafter was trained against a different target width",
            dcfg.backbone_hidden, target_cfg.dim
        ));
    }
    let drafter_weights =
        hipfire_arch_gemma4::drafter::Gemma4DrafterWeights::load(&dhfq, &dcfg, gpu)?;
    let drafter_scratch = hipfire_arch_gemma4::drafter::Gemma4DrafterScratch::new(gpu, &dcfg)
        .map_err(|e| format!("gemma4 drafter scratch: {e}"))?;
    let spec_scratch =
        hipfire_arch_gemma4::speculative::Gemma4SpecScratch::new(gpu, target_cfg, draft_len)
            .map_err(|e| format!("gemma4 spec scratch: {e}"))?;
    // Prime the batched verify path (b=1 then real block size) on disposable
    // throwaway states — mirrors PR's warmup to ensure kernels are compiled.
    let warm_b = draft_len + 1;
    {
        let mut warm =
            hipfire_arch_gemma4::gemma4::Gemma4State::new_with_max_seq(gpu, target_cfg, warm_b + 4)
                .map_err(|e| format!("gemma4-eagle: warm state: {e}"))?;
        let _ = hipfire_arch_gemma4::forward::forward_batch(
            target_cfg,
            target_weights,
            &mut warm,
            gpu,
            &[target_cfg.bos_token],
            0,
        );
        let _ = gpu.hip.device_synchronize();
        warm.free_gpu(gpu);
    }
    {
        let mut warm =
            hipfire_arch_gemma4::gemma4::Gemma4State::new_with_max_seq(gpu, target_cfg, warm_b + 4)
                .map_err(|e| format!("gemma4-eagle: warm state 2: {e}"))?;
        let _ = hipfire_arch_gemma4::forward::forward_batch(
            target_cfg,
            target_weights,
            &mut warm,
            gpu,
            &vec![target_cfg.bos_token; warm_b],
            0,
        );
        let _ = gpu.hip.device_synchronize();
        warm.free_gpu(gpu);
    }
    Ok(crate::Gemma4EagleState {
        drafter_config: dcfg,
        drafter_weights,
        drafter_scratch,
        spec_scratch,
        draft_len,
    })
}

// ─── MuseGlimmerCarrier ────────────────────────────────────────────────
/// HIPFIRE_PP_LAYERS band-spec validation — the same parse/count/sum
/// authority `load_qwen35_pp` uses, exposed so the daemon preflight can
/// refuse a bad band spec BEFORE the eager prior-model teardown. Returns
/// `None` when the env is unset/empty (uniform split).
pub fn parse_pp_layer_bands(
    spec: Option<&str>,
    pp: usize,
    n_layers: usize,
) -> Result<Option<Vec<usize>>, String> {
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
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
    if sum != n_layers {
        return Err(format!(
            "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
            sum, n_layers
        ));
    }
    Ok(Some(counts))
}

/// Muse Glimmer carrier preconditions — the single authority shared by the
/// carrier load and the daemon-side preflight (VMM+CASK, pp, source shape,
/// gemma4 drafter).
pub fn preflight_muse_glimmer(src: &ModelSource, ctx: &LoadCtx) -> Result<(), String> {
    if ctx.kv_backend == KvBackend::Vmm && ctx.cask.sidecar.is_some() {
        return Err(
            "muse_glimmer: KV backend 'vmm' does not support CASK/TriAttention eviction; disable the sidecar or use 'contiguous'"
                .into(),
        );
    }
    if ctx.pp > 1 {
        return Err(match src {
            ModelSource::Hfq(_) => "muse_glimmer: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "muse_glimmer: safetensors + pp>1 unsupported",
        }
        .into());
    }
    if matches!(src, ModelSource::Dir(_)) {
        return Err(
            "muse_glimmer: safetensors Dir load not yet wired — use HFQ (quantize with --arch-id 14) or add config_from_source to hipfire-arch-muse-glimmer"
                .into(),
        );
    }
    if ctx.gemma4_drafter_path.is_some() {
        return Err(
            "params.drafter (Gemma4 EAGLE) is not supported on arch_id=14 (Muse Glimmer).                  Remove params.drafter for AR-only decode."
                .to_string(),
        );
    }
    Ok(())
}

/// Shared tokenizer metadata resolution — the same authority every carrier
/// load uses (`resolve_source_meta`). Exposed so the daemon preflight can
/// refuse a source whose tokenizer metadata does not parse BEFORE any
/// prior-model teardown.
pub fn preflight_tokenizer(src: &ModelSource, path: &str) -> Result<(), String> {
    resolve_source_meta(src, path).map(|_| ())
}

pub struct MuseGlimmerCarrier;
impl Carrier for MuseGlimmerCarrier {
    fn name(&self) -> &'static str {
        "muse_glimmer"
    }
    fn spec_target_guard<'m>(
        &self,
        _state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        Err("muse_glimmer: spec decode not yet wired (AR-only)".into())
    }
    fn make_spec_emitter<'a>(
        &self,
        _ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Err("muse_glimmer: spec emitter not yet wired".into())
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 14
    }
    fn classify_parallel_variant(&self, src: &ModelSource) -> Result<ModelVariant, String> {
        match src {
            ModelSource::Hfq(hfq) => match hfq.arch_id {
                // 14 is the only primary Muse Glimmer id. arch 23 (DFlash
                // drafter sidecar) is never a CAP-001 target — no row exists.
                14 => Ok(ModelVariant::MuseGlimmer),
                other => Err(format!("muse_glimmer: unexpected HFQ arch_id {}", other)),
            },
            ModelSource::Dir(source) => match source.arch_id() {
                14 => Ok(ModelVariant::MuseGlimmer),
                other => Err(format!(
                    "muse_glimmer safetensors: unexpected arch_id {}",
                    other
                )),
            },
        }
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: Some(2),
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::MuseGlimmer,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.muse_glimmer_mut().unwrap();
        Some(if bundle.device_hidden_capture_enabled() {
            match hipfire_arch_muse_glimmer::forward::prefill_with_device_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
            ) {
                Ok(_) => true,
                Err(e) => {
                    *prefill_err = Some(e);
                    false
                }
            }
        } else {
            let mut hidden_out: Vec<f32> = Vec::new();
            match hipfire_arch_muse_glimmer::forward::prefill_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
                &[],
                &mut hidden_out,
            ) {
                Ok(_) => true,
                Err(e) => {
                    *prefill_err = Some(e);
                    false
                }
            }
        })
    }
    fn bench_decode_prime(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
    ) -> Option<Option<String>> {
        let bundle = m.muse_glimmer_mut().unwrap();
        bundle.reset_session_state();
        Some(if bundle.device_hidden_capture_enabled() {
            hipfire_arch_muse_glimmer::forward::prefill_with_device_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
            )
            .err()
        } else {
            let mut hidden_out: Vec<f32> = Vec::new();
            hipfire_arch_muse_glimmer::forward::prefill_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
                &[],
                &mut hidden_out,
            )
            .err()
        })
    }
    fn bench_decode_run(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        context: usize,
        iterations: usize,
        decode_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.muse_glimmer_mut().unwrap();
        let config = &bundle.config;
        let weights = &bundle.weights;
        let state = &mut bundle.state;
        let mut ok = true;
        for i in 0..iterations {
            let token = 101 + (i as u32 % 1000);
            match hipfire_arch_muse_glimmer::forward::decode_step(
                config,
                weights,
                state,
                gpu,
                token,
                (context + i) as u32,
            ) {
                Ok(_) => {}
                Err(e) => {
                    *decode_err = Some(format!("iter {i} pos {}: {e}", context + i));
                    ok = false;
                    break;
                }
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        // Shared authority with the daemon-side preflight.
        preflight_muse_glimmer(&src, ctx)?;
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        match src {
            ModelSource::Hfq(hfq) => {
                // HFQ load path via the crate API (contract: from_hfq / load /
                // new_with_max_seq_backend with ctx.kv_backend).
                let config = hipfire_arch_muse_glimmer::config::GlimmerConfig::from_hfq(&hfq)?;
                let weights = hipfire_arch_muse_glimmer::glimmer::GlimmerWeights::load(
                    &hfq, &config, ctx.gpu,
                )?;
                let mut state =
                    match hipfire_arch_muse_glimmer::glimmer::GlimmerState::new_with_max_seq_backend(
                        ctx.gpu,
                        &config,
                        ctx.max_seq,
                        ctx.kv_backend,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            weights.free_gpu(ctx.gpu);
                            return Err(format!(
                                "muse_glimmer: GlimmerState::new_with_max_seq_backend failed: {e}"
                            ));
                        }
                    };
                // eos resolution by name with 200001 as the documented fallback.
                // 200001 is Glimmer's eos_token_id (config.eos_token, bos 200000).
                // Try tokenizer encode of common EOS surface forms first; if none
                // tokenize to a single id, fall back to 200001 (never 1 — the
                // generic fallback in resolve_eos_tok would be silently wrong
                // for Glimmer and would cause runaway generation).
                let eos_tok = {
                    let tok = resolve_eos_tok(
                        &meta.tokenizer,
                        &[
                            "<end_of_turn>",
                            "<|im_end|>",
                            "</s>",
                            "<|endoftext|>",
                            "<eos>",
                        ],
                    );
                    if tok == 1 {
                        200001
                    } else {
                        tok
                    }
                };
                // Optional DFlash drafter (arch 23). OFF by default — daemon
                // only populates ctx.draft_path when HIPFIRE_DFLASH_DRAFT is set
                // (or params.draft) and dflash_mode != "off". When present, load
                // the 5-layer diffusion draft head (encoder.fc / layers.* / norm)
                // and stash it on the bundle for the speculator to consume.
                // On any failure, log and fall back to AR-only (never fail the
                // target load because the draft is auxiliary).
                let drafter: Option<crate::GlimmerDrafterBundle> = if let Some(dp) =
                    ctx.draft_path.clone()
                {
                    match (|| -> Result<crate::GlimmerDrafterBundle, String> {
                        let dhfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&dp))
                            .map_err(|e| format!("open glimmer drafter '{dp}': {e}"))?;
                        if dhfq.arch_id
                            != hipfire_arch_muse_glimmer::drafter::GLIMMER_DRAFTER_ARCH_ID
                        {
                            return Err(format!(
                                "glimmer drafter '{}' arch_id {} != {} (muse_glimmer_assistant); a DFlash draft (arch 20) or Gemma4 draft (22) does not match",
                                dp, dhfq.arch_id, hipfire_arch_muse_glimmer::drafter::GLIMMER_DRAFTER_ARCH_ID
                            ));
                        }
                        let dcfg =
                            hipfire_arch_muse_glimmer::drafter::GlimmerDrafterConfig::from_hfq(
                                &dhfq,
                            )?;
                        if dcfg.hidden != config.dim {
                            return Err(format!(
                                "glimmer drafter hidden {} != target dim {} (cross-attention concat invariant)",
                                dcfg.hidden, config.dim
                            ));
                        }
                        if dcfg.block_size != 16 {
                            eprintln!("glimmer drafter: WARNING block_size {} != 16 (expected diffusion recipe)", dcfg.block_size);
                        }
                        let dweights =
                            hipfire_arch_muse_glimmer::drafter::GlimmerDrafterWeights::load(
                                &dhfq, &dcfg, ctx.gpu,
                            )?;
                        // Freeze HIPFIRE_GLIMMER_CTX_CAP once at load (daemon/load default
                        // 256). Same value sizes drafter scratch and device hidden log.
                        let ctx_cap = {
                            let requested = std::env::var("HIPFIRE_GLIMMER_CTX_CAP")
                                .ok()
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .filter(|v| *v > 0)
                                .unwrap_or(
                                    hipfire_arch_muse_glimmer::drafter::GLIMMER_DRAFTER_CTX_CAP_DEFAULT,
                                );
                            requested.clamp(1, ctx.max_seq)
                        };
                        let dscratch =
                            match hipfire_arch_muse_glimmer::drafter::GlimmerDrafterScratch::new(
                                ctx.gpu,
                                &dcfg,
                                ctx.max_seq,
                                ctx_cap,
                            ) {
                                Ok(s) => s,
                                Err(e) => {
                                    dweights.free_gpu(ctx.gpu);
                                    return Err(format!("glimmer drafter scratch: {e}"));
                                }
                            };
                        eprintln!(
                            "  glimmer DFlash drafter loaded: {} (layers={}, hidden={}, block={}, mask={}, ctx_cap={})",
                            dp, dcfg.n_layers, dcfg.hidden, dcfg.block_size, dcfg.mask_token_id, ctx_cap
                        );
                        Ok(crate::GlimmerDrafterBundle {
                            config: dcfg,
                            weights: dweights,
                            scratch: dscratch,
                        })
                    })() {
                        Ok(b) => Some(b),
                        Err(e) => {
                            eprintln!("  glimmer DFlash drafter load failed ({}): {} — falling back to AR only", ctx.draft_path.as_deref().unwrap_or(""), e);
                            None
                        }
                    }
                } else {
                    None
                };
                // Device hidden capture is an opt-in experiment. The controlled
                // gfx1100 gate removed all capture D2Hs but regressed decode
                // throughput, so the unchanged host path remains the default.
                // Selection is frozen here — no runtime path flipping after load.
                if let Some(d) = &drafter {
                    let device_enabled =
                        hipfire_config::developer_var("HIPFIRE_GLIMMER_DEVICE_CAPTURE")
                            .ok()
                            .as_deref()
                            == Some("1");
                    if !device_enabled {
                        eprintln!(
                            "  glimmer hidden capture: backend=host (set HIPFIRE_GLIMMER_DEVICE_CAPTURE=1 to test device capture)"
                        );
                    } else {
                        // Same frozen cap used for scratch construction (no env re-read).
                        let ctx_cap = d.scratch.ctx_capacity();
                        let layers = &d.config.target_layer_ids;
                        match state.enable_device_hidden_capture(
                            ctx.gpu,
                            layers,
                            ctx_cap,
                            config.n_layers,
                            config.dim,
                        ) {
                            Ok(()) => {
                                eprintln!(
                                    "  glimmer hidden capture: backend=device capacity_rows={} target_layer_ids={:?} shape=[{}, {}, {}]",
                                    ctx_cap,
                                    layers,
                                    ctx_cap,
                                    layers.len(),
                                    config.dim,
                                );
                            }
                            Err(e) => {
                                eprintln!(
                                    "  glimmer hidden capture: device enable failed ({e}) — continuing with host fallback"
                                );
                            }
                        }
                    }
                }
                let speculator = crate::spec_build::build_speculator(
                    meta.arch_id,
                    None,
                    None,
                    true,
                    ctx.max_seq,
                    ctx.spec,
                );
                let chat_template = match meta.chat_template {
                    Some(t) => match crate::rewrite_muse_glimmer_onyx_template(&t) {
                        Ok(rewritten) => Some(rewritten),
                        Err(e) => {
                            state.free_gpu(ctx.gpu);
                            weights.free_gpu(ctx.gpu);
                            if let Some(d) = drafter {
                                d.scratch.free_gpu(ctx.gpu);
                                d.weights.free_gpu(ctx.gpu);
                            }
                            return Err(e);
                        }
                    },
                    None => None,
                };
                Ok(LoadedModel {
                    state: Some(Box::new(crate::MuseGlimmerBundle {
                        config,
                        weights,
                        state,
                        eos_tok,
                        drafter,
                        target_hidden_host: Vec::new(),
                    })),
                    speculator,
                    ..LoadedModel::skeleton(
                        meta.arch_id,
                        meta.tokenizer,
                        ctx.max_seq,
                        ctx.max_seq,
                        ctx.path.to_string(),
                        chat_template,
                    )
                })
            }
            ModelSource::Dir(source) => {
                let _ = source;
                return Err(
                    "muse_glimmer: safetensors Dir load not yet wired — use HFQ (quantize with --arch-id 14) or add config_from_source to hipfire-arch-muse-glimmer".into()
                );
            }
        }
    }
}

#[cfg(test)]
mod gemma4_route_tests {
    use super::{gemma4_use_lowered, gemma4_validate_drafter_route, Carrier, Gemma4Carrier};

    #[test]
    fn gemma4_advertises_staged_semantic_contract() {
        let caps = Gemma4Carrier.caps();
        assert_eq!(caps.semantic_contract_version, Some(2));
    }

    #[test]
    fn e_series_never_enters_dense_lowered_prefill() {
        assert!(!gemma4_use_lowered(false, true, false, true));
    }

    #[test]
    fn dense_opt_in_and_moe_keep_existing_routes() {
        assert!(gemma4_use_lowered(false, true, false, false));
        assert!(!gemma4_use_lowered(false, true, true, false));
        assert!(gemma4_use_lowered(true, false, false, false));
    }

    #[test]
    fn e_series_drafter_fails_closed() {
        assert!(gemma4_validate_drafter_route(true, true).is_err());
        assert!(gemma4_validate_drafter_route(true, false).is_ok());
        assert!(gemma4_validate_drafter_route(false, true).is_ok());
    }
}
