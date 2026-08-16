use crate::qwen35::{
    DeltaNetState, LayerType, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use crate::Qwen35;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_adaptive::{KvAdaptive, Preset};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode::{self, ResolveResult};
use hipfire_runtime::llama::{self, KvCache, KvDims, KvLayers, KvTarget};
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

pub struct Qwen35Bundle {
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub scratch: Qwen35Scratch,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    /// Adaptive KV controller when engaged at load. Moved into
    /// `LoadedModel.kv_adaptive` by `finish_qwen35_load`.
    pub kv_adaptive: Option<KvAdaptive>,
    /// Optional Qwen3.5-VL vision tower — `Some` when the HFQ contained
    /// `model.visual.patch_embed.proj.weight`. Remains `None` for pure
    /// text checkpoints; the bundle's text path is unaffected.
    pub vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    pub vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
}

/// Build the Qwen35 GPU bundle from an HFQ source.
///
/// CPU-only config/compat validation runs **before** weight upload. Every
/// fallible GPU stage after weights is transactional: on failure, owners
/// constructed so far are explicitly `free_gpu`'d (KvCache via
/// [`KvCache::free_gpu`]) and the original error is preserved with any
/// cleanup/VMM-pending context appended.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen35Bundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("qwen35: directory source unsupported".into());
    };

    let config = <Qwen35 as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;

    // ── CPU-only parse + compatibility (zero GPU allocation) ─────────
    let plan = plan_qwen35_gpu_stages(&config, ctx)?;
    let dn_quant = parse_state_quant(ctx.state_quant_override)?;
    eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
    warn_tiny_model_state(&hfq, dn_quant);

    // ── Weight upload (first GPU ownership) ──────────────────────────
    let weights = <Qwen35 as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
    hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);

    // ── KV (transactional: free weights on fail) ─────────────────────
    let (kv, kv_adaptive) = match construct_kv_cache(&config, ctx, plan) {
        Ok(v) => v,
        Err(e) => {
            weights.free_gpu(ctx.gpu);
            return Err(append_cleanup_context(e, note_vmm_after_free(ctx.gpu)));
        }
    };

    // ── DeltaNet state (free kv + weights on fail) ───────────────────
    let dn = match DeltaNetState::new_with_quant(ctx.gpu, &config, dn_quant) {
        Ok(v) => v,
        Err(e) => {
            let cleanup = free_kv_and_weights(kv, weights, ctx.gpu);
            return Err(append_cleanup_context(format!("{e}"), cleanup));
        }
    };

    // ── Scratch (free dn + kv + weights on fail) ─────────────────────
    let scratch = match Qwen35Scratch::new_with_kv_max(ctx.gpu, &config, 2048, ctx.max_seq) {
        Ok(v) => v,
        Err(e) => {
            let cleanup = free_dn_kv_and_weights(dn, kv, weights, ctx.gpu);
            return Err(append_cleanup_context(format!("{e}"), cleanup));
        }
    };

    Ok(Qwen35Bundle {
        config,
        weights,
        scratch,
        kv_cache: kv,
        dn_state: dn,
        kv_adaptive,
        vision_config: None,
        vision_weights: None,
    })
}

/// CPU-resolved KV construction inputs. Built before any weight upload so
/// malformed adaptive / V / compat configs never touch device memory.
struct Qwen35KvPlan {
    mode: hipfire_runtime::kv_mode::KvMode,
    is_kv_layer: Vec<bool>,
    dims: KvDims,
    /// `None` = static path. `Some` carries a fully-built controller (floors
    /// authoritative) ready for adaptive KV construction.
    adaptive: Option<KvAdaptive>,
    /// Static path only: final V encoding (adaptive always starts Q8).
    static_v: llama::VMode,
    /// Original HIPFIRE_KV_V string for logging (static path).
    kv_v_env: String,
}

/// All CPU-only validation that must precede `load_weights`.
fn plan_qwen35_gpu_stages(config: &Qwen35Config, ctx: &LoadCtx) -> Result<Qwen35KvPlan, String> {
    let kv_mode = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());

    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();

    let ResolveResult { mode, warning } =
        kv_mode::resolve(&kv_mode, &kv_mode::QWEN35_HFQ_POLICY, config.head_dim);
    if let Some(w) = warning {
        eprintln!("  KV cache: {w} (site {})", kv_mode::QWEN35_HFQ_POLICY.site);
    }

    let kv_v_env = hipfire_config::developer_var("HIPFIRE_KV_V").unwrap_or_default();
    let v_mode_override = match kv_v_env.as_str() {
        "lloyd2" => Some(llama::VMode::Lloyd2),
        "lloyd3" => Some(llama::VMode::Lloyd3),
        "lloyd4" => Some(llama::VMode::Lloyd4),
        "q8" | "" => None,
        other => {
            return Err(format!(
                "HIPFIRE_KV_V='{other}' unknown (expected q8|lloyd2|lloyd3|lloyd4)"
            ));
        }
    };

    let kv_adaptive_spec = ctx
        .kv_adaptive_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_adaptive.clone());
    let adaptive_req = parse_kv_adaptive(&kv_adaptive_spec)?;
    validate_adaptive_cask_handoff(
        adaptive_req.is_some(),
        ctx.cask.sidecar.is_some(),
        ctx.cask.handoff_tokens,
        ctx.kv_backend,
        ctx.cask.cask_m_folding,
        ctx.max_seq,
    )?;

    // Adaptive DFlash is out of scope: refuse explicit adaptive + DFlash draft.
    if adaptive_req.is_some() {
        if let Some(draft) = ctx.draft_path {
            if !draft.is_empty() {
                return Err(
                    "kv_adaptive is incompatible with DFlash (adaptive×DFlash is out of scope); \
                     disable one of HIPFIRE_KV_ADAPTIVE / draft path"
                        .into(),
                );
            }
        }
        if ctx.pp > 1 {
            return Err("kv_adaptive requires single-GPU (pp=1)".into());
        }
        if let Some(vm) = v_mode_override {
            return Err(format!(
                "kv_adaptive cannot combine with explicit HIPFIRE_KV_V={vm:?}; \
                 adaptive always starts V=q8 and downshifts via the controller"
            ));
        }
        if config.head_dim != 256 {
            return Err(format!(
                "kv_adaptive requires head_dim=256 (got {})",
                config.head_dim
            ));
        }
    }

    let physical_cap = ctx.cask.physical_cap(ctx.max_seq)?;
    let dims = KvDims {
        layers: KvLayers::Mask(is_kv_layer.clone()),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: Some(physical_cap),
    };

    if let Some((preset, k_floor, v_floor)) = adaptive_req {
        // Build controller first — floors and thresholds are authoritative.
        let mut ad = match preset {
            Some(p) => KvAdaptive::from_preset(p, ctx.max_seq, config.n_kv_heads, config.head_dim),
            None => KvAdaptive::new(
                ctx.max_seq,
                config.n_kv_heads,
                config.head_dim,
                k_floor,
                v_floor,
            ),
        };
        if ad.current_cap() < hipfire_runtime::llama::PREFILL_MAX_BATCH {
            return Err(format!(
                "kv_adaptive={kv_adaptive_spec}: max_seq={} too small: start-tier capacity {} < prefill chunk {}",
                ctx.max_seq,
                ad.current_cap(),
                hipfire_runtime::llama::PREFILL_MAX_BATCH,
            ));
        }
        if ctx.cask.sidecar.is_some() {
            ad.configure_eviction_handoff(ctx.cask.handoff_tokens);
        }
        Ok(Qwen35KvPlan {
            mode,
            is_kv_layer,
            dims,
            adaptive: Some(ad),
            static_v: llama::VMode::Q8,
            kv_v_env,
        })
    } else {
        // Static path: final K mode + optional Lloyd-V known before allocation.
        let static_v = v_mode_override.unwrap_or(llama::VMode::Q8);
        validate_cask_static_layout(ctx.cask.sidecar.is_some(), mode, static_v)?;
        if !matches!(static_v, llama::VMode::Q8) {
            let is_fwht = matches!(
                mode,
                hipfire_runtime::kv_mode::KvMode::Fwht2
                    | hipfire_runtime::kv_mode::KvMode::Fwht3
                    | hipfire_runtime::kv_mode::KvMode::Fwht4
            );
            if !is_fwht {
                return Err(format!(
                    "HIPFIRE_KV_V={kv_v_env} requires an FWHT K mode (fwht2/3/4); resolved mode is {mode:?}"
                ));
            }
        }
        Ok(Qwen35KvPlan {
            mode,
            is_kv_layer,
            dims,
            adaptive: None,
            static_v,
            kv_v_env,
        })
    }
}

fn validate_adaptive_cask_handoff(
    adaptive: bool,
    sidecar: bool,
    handoff_tokens: usize,
    backend: KvBackend,
    cask_m_folding: bool,
    max_seq: usize,
) -> Result<(), String> {
    if !adaptive {
        return if handoff_tokens == 0 {
            Ok(())
        } else {
            Err("memory.cask.handoff_tokens requires memory.kv_adaptive != off".into())
        };
    }
    if !sidecar {
        return if handoff_tokens == 0 {
            Ok(())
        } else {
            Err(
                "memory.cask.handoff_tokens requires memory.cask.sidecar (or auto-attached sidecar)"
                    .into(),
            )
        };
    }
    if handoff_tokens == 0 {
        return Err(
            "kv_adaptive and CASK require memory.cask.handoff_tokens > 0 for an explicit one-way handoff"
                .into(),
        );
    }
    if backend != KvBackend::Vmm {
        return Err("kv_adaptive -> CASK handoff currently requires memory.kv_backend=vmm".into());
    }
    if cask_m_folding {
        return Err(
            "kv_adaptive -> CASK handoff supports plain TriAttention eviction only; CASK m-folding needs FWHT/Lloyd fold kernels"
                .into(),
        );
    }
    if handoff_tokens > max_seq {
        return Err(format!(
            "memory.cask.handoff_tokens={handoff_tokens} exceeds max_seq={max_seq} and can never activate"
        ));
    }
    Ok(())
}

/// The static CASK path remains limited to Q8 V and Givens-asym K. Adaptive
/// FWHT/Lloyd layouts use the separately validated plain-TriAttention handoff;
/// m-folding still lacks the corresponding fold/requant kernels.
fn validate_cask_static_layout(
    cask_enabled: bool,
    mode: hipfire_runtime::kv_mode::KvMode,
    v_mode: llama::VMode,
) -> Result<(), String> {
    if !cask_enabled {
        return Ok(());
    }
    if !matches!(v_mode, llama::VMode::Q8) {
        return Err(format!(
            "CASK currently requires V=q8 (resolved V mode is {v_mode:?})"
        ));
    }
    if !matches!(
        mode,
        hipfire_runtime::kv_mode::KvMode::Q8
            | hipfire_runtime::kv_mode::KvMode::Asym2
            | hipfire_runtime::kv_mode::KvMode::Asym3
            | hipfire_runtime::kv_mode::KvMode::Asym4
    ) {
        return Err(format!(
            "CASK currently supports Q8/asym2/asym3/asym4 K only (resolved mode is {mode:?})"
        ));
    }
    Ok(())
}

fn construct_kv_cache(
    config: &Qwen35Config,
    ctx: &mut LoadCtx,
    plan: Qwen35KvPlan,
) -> Result<(KvCache, Option<KvAdaptive>), String> {
    if let Some(ad) = plan.adaptive {
        // Floors come from the controller, not separately parsed hints.
        let k_floor = ad.k_floor;
        let v_floor = ad.v_floor;
        // Adaptive always starts exactly FWHT4/Q8 regardless of resolved static mode.
        let start_mode = hipfire_runtime::kv_mode::KvMode::Fwht4;
        let k_floor_bph = k_floor.bytes_per_head(config.head_dim);

        let kv = match ctx.kv_backend {
            KvBackend::Vmm => {
                // Floor-reserved VMM arenas; current encoding FWHT4/Q8.
                KvCache::new_gpu_vmm_adaptive_filtered(
                    ctx.gpu,
                    &plan.is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    ctx.max_seq,
                    k_floor_bph,
                    v_floor,
                )
                .map_err(|e| format!("{e}"))?
            }
            KvBackend::Contiguous => {
                // Contiguous: allocate start-tier FWHT4/Q8 then floor-resize in place.
                // If floor-resize fails, free the start-tier cache explicitly.
                let mut kv = <KvCache as KvCacheExt>::from_mode_with_backend(
                    start_mode,
                    KvBackend::Contiguous,
                    KvTarget::Single(ctx.gpu),
                    &plan.dims,
                )
                .map_err(|e| format!("{e}"))?;
                if let Err(e) = kv.set_adaptive_floor_alloc(ctx.gpu, v_floor, k_floor_bph) {
                    let cleanup = kv.free_gpu(ctx.gpu).map_err(|fe| fe.to_string());
                    return Err(append_cleanup_context(format!("{e}"), cleanup));
                }
                kv
            }
        };

        eprintln!(
            "[adaptive-kv] engaged: backend={:?} pattern={:?} k_floor={:?} v_floor={:?} thresholds={:?} start_cap={} (max_seq={}, reserve at floor, start FWHT4/Q8)",
            ctx.kv_backend,
            ad.steps,
            ad.k_floor,
            ad.v_floor,
            ad.thresholds,
            ad.current_cap(),
            ctx.max_seq,
        );
        Ok((kv, Some(ad)))
    } else {
        let static_v = plan.static_v;
        let mode = plan.mode;
        let physical_cap = plan
            .dims
            .physical_cap
            .expect("Qwen3.5 KV plan always resolves physical_cap");
        let kv = match (ctx.kv_backend, static_v) {
            (KvBackend::Vmm, vm) => {
                // Unified VMM constructor: reserve == current; never post-alloc realloc.
                KvCache::new_gpu_vmm_capped_filtered(
                    ctx.gpu,
                    &plan.is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    ctx.max_seq,
                    physical_cap,
                    mode,
                    vm,
                )
                .map_err(|e| format!("{e}"))?
            }
            (KvBackend::Contiguous, llama::VMode::Q8) => <KvCache as KvCacheExt>::from_mode_with_backend(
                mode,
                KvBackend::Contiguous,
                KvTarget::Single(ctx.gpu),
                &plan.dims,
            )
            .map_err(|e| format!("{e}"))?,
            (KvBackend::Contiguous, vm) => {
                let mut kv = <KvCache as KvCacheExt>::from_mode_with_backend(
                    mode,
                    KvBackend::Contiguous,
                    KvTarget::Single(ctx.gpu),
                    &plan.dims,
                )
                .map_err(|e| format!("{e}"))?;
                if let Err(e) = kv.set_v_mode_realloc(ctx.gpu, vm) {
                    let cleanup = kv.free_gpu(ctx.gpu).map_err(|fe| fe.to_string());
                    return Err(append_cleanup_context(format!("{e}"), cleanup));
                }
                kv
            }
        };
        if !matches!(static_v, llama::VMode::Q8) {
            eprintln!(
                "[hipfire-arch-qwen35] V-cache mode override → {} (256-wide lloyd-V on fwht K)",
                plan.kv_v_env
            );
        }
        Ok((kv, None))
    }
}

pub fn free_qwen35_bundle(bundle: Qwen35Bundle, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let Qwen35Bundle {
        config: _,
        weights,
        scratch,
        kv_cache,
        dn_state,
        kv_adaptive: _,
        vision_config: _,
        vision_weights,
    } = bundle;
    // Match unload_model Qwen35 order: kv → scratch → weights → dn → vision.
    let mut first: Option<String> = None;
    if let Err(e) = kv_cache.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    scratch.free_gpu(gpu);
    weights.free_gpu(gpu);
    dn_state.free_gpu(gpu);
    if let Some(vw) = vision_weights {
        vw.free_gpu(gpu);
    }
    let vmm = note_vmm_after_free(gpu);
    match (first, vmm) {
        (None, Ok(())) => Ok(()),
        (Some(e), Ok(())) => Err(e),
        (None, Err(c)) => Err(c),
        (Some(e), Err(c)) => Err(format!("{e}; cleanup also failed: {c}")),
    }
}

fn free_kv_and_weights(
    kv: KvCache,
    weights: Qwen35Weights,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let mut first: Option<String> = None;
    if let Err(e) = kv.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    weights.free_gpu(gpu);
    match first {
        Some(e) => Err(append_cleanup_context(e, note_vmm_after_free(gpu))),
        None => note_vmm_after_free(gpu),
    }
}

fn free_dn_kv_and_weights(
    dn: DeltaNetState,
    kv: KvCache,
    weights: Qwen35Weights,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let mut first: Option<String> = None;
    if let Err(e) = kv.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    dn.free_gpu(gpu);
    weights.free_gpu(gpu);
    match first {
        Some(e) => Err(append_cleanup_context(e, note_vmm_after_free(gpu))),
        None => note_vmm_after_free(gpu),
    }
}

fn note_vmm_after_free(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    gpu.ensure_vmm_cleaned().map_err(|e| {
        format!("pending VMM teardown after free ({e}); retry unload or restart the process")
    })
}

/// Preserve the primary operation error; append cleanup failure context when present.
fn append_cleanup_context(op_err: String, cleanup: Result<(), String>) -> String {
    match cleanup {
        Ok(()) => op_err,
        Err(c) => format!("{op_err}; cleanup also failed: {c}"),
    }
}

// ─── Helper: StateQuant parsing ─────────────────────────────────────

fn parse_state_quant(mode: Option<&str>) -> Result<StateQuant, String> {
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)"
        )),
    }
}

fn state_quant_label(q: StateQuant) -> &'static str {
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::Q8 => "Q8",
        StateQuant::Q4 => "Q4",
    }
}

// ─── Helper: parameter count + tiny-model warning ─────────────────

fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

fn warn_tiny_model_state(hfq: &HfqFile, q: StateQuant) {
    const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
    let params = hfq_parameter_count(hfq);
    if params < TINY_MODEL_PARAMS && q != StateQuant::FP32 {
        eprintln!(
            "  warning: model has ~{:.2}B params; FP32 DeltaNet state is recommended below 2B for long-generation coherence (current: {})",
            params as f64 / 1.0e9,
            state_quant_label(q)
        );
    }
}

// ─── Helper: KV adaptive parsing ──────────────────────────────────

/// Parse adaptive policy. `Ok(None)` = off. Malformed/unsupported explicit
/// requests are hard errors (no silent ignore).
fn parse_kv_adaptive(
    s: &str,
) -> Result<
    Option<(
        Option<Preset>,
        hipfire_runtime::kv_adaptive::KMode,
        hipfire_runtime::llama::VMode,
    )>,
    String,
> {
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use hipfire_runtime::llama::VMode;
    match s {
        "" | "off" => Ok(None),
        // Floors agree with KvAdaptive::from_preset: C=F4/L4, B=F3/L3, A=F2/L2.
        "conservative" => Ok(Some((
            Some(Preset::Conservative),
            KMode::Fwht4,
            VMode::Lloyd4,
        ))),
        "balanced" => Ok(Some((Some(Preset::Balanced), KMode::Fwht3, VMode::Lloyd3))),
        "aggressive" => Ok(Some((
            Some(Preset::Aggressive),
            KMode::Fwht2,
            VMode::Lloyd2,
        ))),
        other if other.starts_with("advanced:") => {
            let spec = &other["advanced:".len()..];
            let mut k = None;
            let mut v = None;
            for kvp in spec.split(',') {
                let mut it = kvp.splitn(2, '=');
                match (it.next(), it.next()) {
                    (Some("k"), Some("fwht4")) => k = Some(KMode::Fwht4),
                    (Some("k"), Some("fwht3")) => k = Some(KMode::Fwht3),
                    (Some("k"), Some("fwht2")) => k = Some(KMode::Fwht2),
                    (Some("v"), Some("lloyd4")) => v = Some(VMode::Lloyd4),
                    (Some("v"), Some("lloyd3")) => v = Some(VMode::Lloyd3),
                    (Some("v"), Some("lloyd2")) => v = Some(VMode::Lloyd2),
                    (Some(key), Some(val)) => {
                        return Err(format!(
                            "kv_adaptive='{other}' unknown key/value {key}={val} \
                             (expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>)"
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "kv_adaptive='{other}' malformed \
                             (expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>)"
                        ));
                    }
                }
            }
            match (k, v) {
                (Some(k), Some(v)) => Ok(Some((None, k, v))),
                _ => Err(format!(
                    "kv_adaptive='{other}' incomplete \
                     (expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>)"
                )),
            }
        }
        other => Err(format!(
            "kv_adaptive='{other}' unknown \
             (expected off|conservative|balanced|aggressive|advanced:k=..,v=..)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use hipfire_runtime::llama::VMode;

    #[test]
    fn parse_presets_agree_with_controller_floors() {
        let cases = [
            (
                "conservative",
                Preset::Conservative,
                KMode::Fwht4,
                VMode::Lloyd4,
            ),
            ("balanced", Preset::Balanced, KMode::Fwht3, VMode::Lloyd3),
            (
                "aggressive",
                Preset::Aggressive,
                KMode::Fwht2,
                VMode::Lloyd2,
            ),
        ];
        for (spec, preset, k, v) in cases {
            let parsed = parse_kv_adaptive(spec).unwrap().unwrap();
            assert_eq!(parsed.0, Some(preset));
            assert_eq!(parsed.1, k);
            assert_eq!(parsed.2, v);
            let ad = KvAdaptive::from_preset(preset, 10_000, 4, 256);
            assert_eq!(
                (ad.k_floor, ad.v_floor),
                (k, v),
                "parse floors must match KvAdaptive::from_preset for {spec}"
            );
        }
    }

    #[test]
    fn parse_off_and_errors() {
        assert!(parse_kv_adaptive("off").unwrap().is_none());
        assert!(parse_kv_adaptive("").unwrap().is_none());
        assert!(parse_kv_adaptive("nope").is_err());
        assert!(parse_kv_adaptive("advanced:k=fwht2").is_err());
        assert!(parse_kv_adaptive("advanced:k=fwht9,v=lloyd2").is_err());
    }

    /// Malformed adaptive must fail in the CPU parse/plan seam — before any
    /// weight upload would run. Observable contract: pure helpers error with
    /// no GPU required.
    #[test]
    fn malformed_adaptive_fails_in_cpu_plan() {
        // Direct plan path: parse_kv_adaptive is the CPU seam for malformed specs.
        let err = parse_kv_adaptive("not-a-preset").unwrap_err();
        assert!(err.contains("unknown"), "{err}");
        let err = parse_kv_adaptive("advanced:k=fwht2").unwrap_err();
        assert!(
            err.contains("incomplete") || err.contains("malformed"),
            "{err}"
        );
    }

    #[test]
    fn adaptive_cask_handoff_compatibility_matrix_is_fail_closed() {
        let check = |adaptive, sidecar, handoff, backend, folding| {
            validate_adaptive_cask_handoff(adaptive, sidecar, handoff, backend, folding, 2048)
        };

        assert!(check(true, true, 128, KvBackend::Vmm, false).is_ok());
        assert!(check(true, true, 0, KvBackend::Vmm, false)
            .unwrap_err()
            .contains("handoff_tokens > 0"));
        assert!(check(true, true, 128, KvBackend::Contiguous, false)
            .unwrap_err()
            .contains("requires memory.kv_backend=vmm"));
        assert!(check(true, true, 128, KvBackend::Vmm, true)
            .unwrap_err()
            .contains("m-folding"));
        assert!(check(true, false, 128, KvBackend::Vmm, false)
            .unwrap_err()
            .contains("requires memory.cask.sidecar"));
        assert!(check(false, false, 128, KvBackend::Vmm, false)
            .unwrap_err()
            .contains("requires memory.kv_adaptive"));
        assert!(
            validate_adaptive_cask_handoff(true, true, 2049, KvBackend::Vmm, false, 2048)
                .unwrap_err()
                .contains("exceeds max_seq")
        );
    }

    #[test]
    fn append_cleanup_preserves_primary_error() {
        let s = append_cleanup_context("kv failed".into(), Ok(()));
        assert_eq!(s, "kv failed");
        let s = append_cleanup_context("kv failed".into(), Err("pending VMM teardown".into()));
        assert!(s.starts_with("kv failed; cleanup also failed:"), "{s}");
        assert!(s.contains("pending VMM"), "{s}");
    }

    #[test]
    fn cask_static_layout_accepts_only_current_compaction_formats() {
        use hipfire_runtime::kv_mode::KvMode;

        for mode in [KvMode::Q8, KvMode::Asym2, KvMode::Asym3, KvMode::Asym4] {
            validate_cask_static_layout(true, mode, VMode::Q8).unwrap();
        }
        for mode in [KvMode::Fwht2, KvMode::Fwht3, KvMode::Fwht4] {
            let err = validate_cask_static_layout(true, mode, VMode::Q8).unwrap_err();
            assert!(err.contains("Q8/asym2/asym3/asym4"), "{err}");
        }
        let err = validate_cask_static_layout(true, KvMode::Fwht3, VMode::Lloyd3).unwrap_err();
        assert!(err.contains("V=q8"), "{err}");
    }

    #[test]
    fn cask_layout_guard_is_inert_when_cask_is_disabled() {
        use hipfire_runtime::kv_mode::KvMode;

        validate_cask_static_layout(false, KvMode::Fwht3, VMode::Lloyd3).unwrap();
    }
}
