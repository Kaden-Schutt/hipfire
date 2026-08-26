// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 carrier load bodies, relocated from `hipfire-loader`'s
//! `carriers.rs` so the arch crate owns its model-loading work. The loader
//! keeps the `Carrier` trait boilerplate, tokenizer/EOS resolution
//! (`resolve_eos_tok`), the kv_mode→compressor-cache mapping
//! (`resolve_deepseek4_compressor_cache_kv_mode`, shared with the EP path
//! and covered by loader unit tests), the speculator wiring
//! (`LoadedModel.speculator` is a loader side-field), and `LoadedModel`
//! assembly (`ModelState::Deepseek4Heterogeneous` wraps the loader-local
//! `Deepseek4HeterogeneousBundle`).

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::PrefillBatchScratch;
use crate::{
    DeepseekV4, DeepseekV4HeterogeneousLoadPlan, DeepseekV4HeterogeneousModel,
    DeepseekV4VerifiedArtifact,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// The single-device load products. `eos_tok`, the speculator, and the
/// `LoadedModel` wrapper stay in the loader (tokenizer + spec wiring are
/// loader-owned); this struct is everything the model itself produces.
pub struct Deepseek4LoadParts {
    pub config: DeepseekV4Config,
    pub weights: DeepseekV4Weights,
    pub state: DeepseekV4State,
    pub pbs: PrefillBatchScratch,
}

/// Pure experts-per-token range check against the checkpoint default —
/// the single authority shared by the load path and daemon-side preflight.
pub fn validate_deepseek4_experts_per_token(
    requested: Option<usize>,
    checkpoint: usize,
) -> Result<(), String> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if requested == 0 || requested > checkpoint {
        return Err(format!(
            "deepseek4: experts-per-token override must be in 1..={checkpoint}, got {requested}"
        ));
    }
    Ok(())
}

fn apply_deepseek4_experts_per_token(
    config: &mut DeepseekV4Config,
    requested: Option<usize>,
) -> Result<(), String> {
    validate_deepseek4_experts_per_token(requested, config.num_experts_per_tok)?;
    if let Some(requested) = requested {
        if requested != config.num_experts_per_tok {
            eprintln!(
                "deepseek4: runtime experts-per-token override {} -> {requested}",
                config.num_experts_per_tok
            );
            config.num_experts_per_tok = requested;
        }
    }
    Ok(())
}

/// Heterogeneous-placement (gfx1201 MQ2R TP3/TP4) model load: admission
/// gates, frozen-artifact verification, and the heterogeneous load itself.
/// The caller (loader) has already checked the placement is not `Single`.
/// Pure heterogeneous-placement precondition gate — the single authority
/// shared by the load path and daemon-side preflight (placement != Single).
pub fn validate_heterogeneous_preconditions(
    src: &ModelSource,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
    experts_per_token: Option<usize>,
    draft_requested: bool,
    dspark: Option<bool>,
) -> Result<(), String> {
    if compressor_cache == hipfire_config::Deepseek4CompressorCache::F16 {
        return Err("deepseek4: kv_cache=f16 currently requires gfx1201 MQ2R TP3/TP4".into());
    }
    if !matches!(src, ModelSource::Hfq(_)) {
        return Err(
            "deepseek4 heterogeneous placement requires the frozen MQ2R HFQ artifact".into(),
        );
    }
    if experts_per_token.is_some_and(|value| value != 6) {
        return Err("deepseek4 heterogeneous placement requires checkpoint top-k 6".into());
    }
    if draft_requested || dspark == Some(true) {
        return Err("deepseek4 heterogeneous placement is direct-AR only until G6/G7".into());
    }
    Ok(())
}

pub fn load_heterogeneous_model(
    src: &ModelSource,
    ctx: &LoadCtx,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
) -> Result<DeepseekV4HeterogeneousModel, String> {
    validate_heterogeneous_preconditions(
        src,
        compressor_cache,
        ctx.deepseek4_experts_per_token,
        ctx.draft_path.is_some(),
        ctx.spec.dspark,
    )?;
    let artifact = DeepseekV4VerifiedArtifact::verify(ctx.path.as_ref())?;
    let plan = DeepseekV4HeterogeneousLoadPlan {
        placement: ctx.deepseek4_compute_placement.clone(),
        prefill_max_batch: 1024,
        ..Default::default()
    };
    DeepseekV4HeterogeneousModel::load_verified(&artifact, plan)
}

/// Build the single-device DeepSeek V4 load products from an HFQ or
/// safetensors-directory source.
pub fn load_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
) -> Result<Deepseek4LoadParts, String> {
    // ── source-varying seam: (config, weights) only ──
    // NOTE: the Dir/safetensors arm is UNVALIDATED — no deepseek_v4
    // checkpoint was available locally to verify load fidelity. Reviewer-ask.
    // DSpark sidecar load gate: `speculation=dspark`/`auto` load the 3×MoE
    // sidecar; any other mechanism (`Some(false)`) skips it so it never pages
    // into VRAM. `None` (auto / directly-driven daemon) keeps default-on.
    let load_dspark = ctx.spec.dspark != Some(false);
    let (config, weights) = match src {
        ModelSource::Hfq(mut hfq) => {
            let mut config = <DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
            apply_deepseek4_experts_per_token(&mut config, ctx.deepseek4_experts_per_token)?;
            config.load_dspark = load_dspark;
            let weights = <DeepseekV4 as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
            (config, weights)
        }
        ModelSource::Dir(source) => {
            let mut config = crate::config_from_safetensors(&source)
                .ok_or_else(|| "deepseek4: failed to parse config from safetensors".to_string())?;
            apply_deepseek4_experts_per_token(&mut config, ctx.deepseek4_experts_per_token)?;
            config.load_dspark = load_dspark;
            let weights = DeepseekV4::load_weights_from_safetensors(&source, &config, ctx.gpu)?;
            (config, weights)
        }
    };
    // F16 compressor cache on the single-device path: gfx1201 (certified)
    // and gfx1151 (ported — the indexer score kernel's WMMA is
    // generation-selected in the kernel source). Storage is confined to
    // main_kv_cache and indexer_kv_cache; every other compressor buffer
    // stays F32 and commit arithmetic completes in F32 before the single
    // F32-to-F16 store.
    let f16_ok =
        config.mq2r && !config.mq2rxt && ctx.gpu.arch_caps.supports_ds4_f16_compressor_cache();
    if compressor_cache == hipfire_config::Deepseek4CompressorCache::F16 && !f16_ok {
        return Err(format!(
            "deepseek4: kv_cache=f16 requires MQ2R on an architecture with wave32 WMMA (RDNA3/RDNA4); got arch={}, mq2r={}, mq2rxt={}",
            ctx.gpu.arch, config.mq2r, config.mq2rxt
        ));
    }
    let mut state = DeepseekV4State::new(&config)?;
    state.compressor_cache_dtype =
        if compressor_cache == hipfire_config::Deepseek4CompressorCache::F16 {
            rdna_compute::DType::F16
        } else {
            rdna_compute::DType::F32
        };
    let pbs_max_batch: usize = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let pbs = PrefillBatchScratch::new(ctx.gpu, &config, pbs_max_batch)?;
    Ok(Deepseek4LoadParts {
        config,
        weights,
        state,
        pbs,
    })
}
