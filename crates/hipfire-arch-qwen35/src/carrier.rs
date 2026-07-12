use crate::qwen35::{
    DeltaNetState, LayerType, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use crate::Qwen35;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_adaptive::{KvAdaptive, Preset};
use hipfire_runtime::kv_mode::{self, ResolveResult};
use hipfire_runtime::llama::{self, KvCache, KvDims, KvLayers, KvTarget};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// qwen35 pipeline-parallel scratch, present iff this bundle is served PP (pp>1).
/// One payload so scratch + layer→device map are inseparable (no "one without the
/// other" illegal state).
pub struct Qwen35PipelineState {
    pub scratch_set: crate::qwen35::Qwen35ScratchSet,
    pub dn_la_to_device: Vec<u8>,
}

pub struct Qwen35Bundle {
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub scratch: Qwen35Scratch,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    /// Optional native MTP (NextN) head — present only when a `.mq4-mtp` trailer
    /// or `.mtp` sidecar was loaded. Lives here (not on `LoadedModel`) so it
    /// travels with the arch state through reset/reload; the loader folds it in
    /// after `load_bundle` returns. `None` on every non-MTP construction.
    pub mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
    /// PP scratch — `Some` only for the qwen35 pp>1 serve path; `None` single-GPU.
    /// Carried here (not on `LoadedModel`) so it travels with arch state and its
    /// teardown is arm-local to `ModelState::Qwen35`.
    pub pipeline: Option<Qwen35PipelineState>,
}

/// Build the Qwen35 GPU bundle from an HFQ source.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen35Bundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("qwen35: directory source unsupported".into());
    };

    let config = <Qwen35 as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
    let weights = <Qwen35 as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;

    // ── MMQ screening ────────────────────────────────────────
    if ctx.gpu.mmq_screen.enabled
        && matches!(
            ctx.gpu.arch.as_str(),
            "gfx906"
                | "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
        )
    {
        let t0 = std::time::Instant::now();
        let (n_safe, n_unsafe) = screen_weights_qwen35(&weights, ctx.gpu);
        let elapsed = t0.elapsed();
        eprintln!(
            "  MMQ screening: {n_safe} safe, {n_unsafe} unsafe (threshold={:.2}, {:.1}ms)",
            ctx.gpu.mmq_screen.threshold,
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    // ── KV mode ──────────────────────────────────────────────
    let kv_mode = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());

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
    let dims = KvDims {
        layers: KvLayers::Mask(is_kv_layer),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: Some(ctx.max_seq),
    };
    let mut kv =
        KvCache::from_mode(mode, KvTarget::Single(ctx.gpu), &dims).map_err(|e| format!("{e}"))?;

    // ── V-mode override via env ──────────────────────────────
    let kv_v_env = std::env::var("HIPFIRE_KV_V").unwrap_or_default();
    let v_mode_override = match kv_v_env.as_str() {
        "lloyd2" => Some(llama::VMode::Lloyd2),
        "lloyd3" => Some(llama::VMode::Lloyd3),
        "lloyd4" => Some(llama::VMode::Lloyd4),
        "q8" | "" => None,
        other => {
            eprintln!("[hipfire-arch-qwen35] HIPFIRE_KV_V='{other}' unknown — ignoring (expected q8|lloyd2|lloyd3|lloyd4)");
            None
        }
    };
    if let Some(vm) = v_mode_override {
        if (kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4) && kv.quant_fwht {
            kv.set_v_mode_realloc(ctx.gpu, vm)
                .map_err(|e| format!("{e}"))?;
            eprintln!(
                    "[hipfire-arch-qwen35] V-cache mode override → {kv_v_env} (256-wide lloyd-V on fwht K)"
                );
        } else {
            eprintln!("[hipfire-arch-qwen35] HIPFIRE_KV_V={kv_v_env} ignored — lloyd-V requires an FWHT K mode (fwht2/3/4); cache is a different mode");
        }
    }

    // ── KV adaptive ──────────────────────────────────────────
    let kv_adaptive_spec = ctx
        .kv_adaptive_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_ADAPTIVE").unwrap_or_default());

    let _kv_adaptive: Option<KvAdaptive> = {
        match parse_kv_adaptive(&kv_adaptive_spec) {
            None => None,
            Some((preset, k_floor, v_floor)) => {
                let ad = match preset {
                    Some(p) => {
                        KvAdaptive::from_preset(p, ctx.max_seq, config.n_kv_heads, config.head_dim)
                    }
                    None => KvAdaptive::new(
                        ctx.max_seq,
                        config.n_kv_heads,
                        config.head_dim,
                        k_floor,
                        v_floor,
                    ),
                };
                if !((kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4) && kv.quant_fwht) {
                    eprintln!("[hipfire-arch-qwen35] kv_adaptive={kv_adaptive_spec} ignored — adaptive KV requires an FWHT K mode (fwht2/3/4); cache is a different mode");
                    None
                } else if ctx.cask.sidecar.is_some() {
                    eprintln!("[hipfire-arch-qwen35] kv_adaptive={kv_adaptive_spec} ignored — adaptive KV is a no-eviction capacity strategy and CASK eviction is active (mutually exclusive)");
                    None
                } else if ad.current_cap() < hipfire_runtime::llama::PREFILL_MAX_BATCH {
                    eprintln!(
                            "[hipfire-arch-qwen35] kv_adaptive={kv_adaptive_spec} ignored — max_seq={} too small: start-tier capacity {} < prefill chunk {}",
                            ctx.max_seq, ad.current_cap(), hipfire_runtime::llama::PREFILL_MAX_BATCH,
                        );
                    None
                } else {
                    if !kv.quant_asym4 {
                        eprintln!("[hipfire-arch-qwen35] kv_adaptive: adaptive works best with kv_mode=fwht4 (K starts at fwht4); current K mode is not fwht4");
                    }
                    let k_floor_bph = k_floor.bytes_per_head(config.head_dim);
                    kv.set_adaptive_floor_alloc(ctx.gpu, v_floor, k_floor_bph)
                        .map_err(|e| format!("{e}"))?;
                    eprintln!(
                            "[adaptive-kv] engaged: pattern={:?} k_floor={:?} v_floor={:?} thresholds={:?} start_cap={} (max_seq={}, V buffer sized at floor)",
                            ad.steps, ad.k_floor, ad.v_floor, ad.thresholds, ad.current_cap(), ctx.max_seq,
                        );
                    Some(ad)
                }
            }
        }
    };

    // ── DeltaNet state ───────────────────────────────────────
    let dn_quant = parse_state_quant(ctx.state_quant_override)?;
    eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
    warn_tiny_model_state(&hfq, dn_quant);
    let dn =
        DeltaNetState::new_with_quant(ctx.gpu, &config, dn_quant).map_err(|e| format!("{e}"))?;

    // ── Scratch ──────────────────────────────────────────────
    let scratch = Qwen35Scratch::new_with_kv_max(ctx.gpu, &config, 2048, ctx.max_seq)
        .map_err(|e| format!("{e}"))?;

    Ok(Qwen35Bundle {
        config,
        weights,
        scratch,
        kv_cache: kv,
        dn_state: dn,
        mtp_head: None,
        pipeline: None,
    })
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

// ─── Helper: MMQ screening (inline from hipfire-loader) ───────────

fn screen_weights_qwen35(weights: &Qwen35Weights, gpu: &mut rdna_compute::Gpu) -> (usize, usize) {
    use crate::qwen35::LayerWeights;
    let mut n_safe = 0usize;
    let mut n_unsafe = 0usize;
    for layer in &weights.layers {
        let wts: Vec<&hipfire_runtime::llama::WeightTensor> = match layer {
            LayerWeights::DeltaNet(l) => {
                vec![
                    &l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.w_gate, &l.w_up, &l.wo,
                ]
            }
            LayerWeights::FullAttn(l) => {
                vec![&l.wq, &l.wk, &l.wv, &l.w_gate, &l.w_up, &l.wo]
            }
            LayerWeights::DeltaNetMoe(l) => {
                vec![&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.wo]
            }
            LayerWeights::FullAttnMoe(l) => {
                vec![&l.wq, &l.wk, &l.wv, &l.wo]
            }
        };
        for wt in wts {
            if !matches!(
                wt.gpu_dtype,
                rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256
            ) {
                continue;
            }
            if gpu.mmq_screen_weight(&wt.buf, wt.m, wt.k) {
                n_safe += 1;
            } else {
                n_unsafe += 1;
            }
        }
    }
    (n_safe, n_unsafe)
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

fn parse_kv_adaptive(
    s: &str,
) -> Option<(
    Option<Preset>,
    hipfire_runtime::kv_adaptive::KMode,
    hipfire_runtime::llama::VMode,
)> {
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use hipfire_runtime::llama::VMode;
    match s {
        "" | "off" => None,
        "conservative" => Some((Some(Preset::Conservative), KMode::Fwht4, VMode::Lloyd4)),
        "balanced" => Some((Some(Preset::Balanced), KMode::Fwht2, VMode::Lloyd2)),
        "aggressive" => Some((Some(Preset::Aggressive), KMode::Fwht2, VMode::Lloyd2)),
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
                    _ => {}
                }
            }
            match (k, v) {
                (Some(k), Some(v)) => Some((None, k, v)),
                _ => {
                    eprintln!("[hipfire-arch-qwen35] kv_adaptive='{other}' malformed — expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>; ignoring");
                    None
                }
            }
        }
        other => {
            eprintln!("[hipfire-arch-qwen35] kv_adaptive='{other}' unknown — expected off|conservative|balanced|aggressive|advanced:k=..,v=..; ignoring");
            None
        }
    }
}
