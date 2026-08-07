use crate::qwen2::{Qwen2Config, Qwen2State, Qwen2Weights};
use crate::Qwen2;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::gpu_cleanup::{BundleTeardown, GpuCleanupFailure};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
// Trait in scope for `tensor_info`/`quant_config` on the Dir source.
use hipfire_runtime::model_source::ModelSource as _;
use rdna_compute::Gpu;

pub struct Qwen2Bundle {
    pub config: Qwen2Config,
    pub weights: Qwen2Weights,
    pub state: Qwen2State,
}

impl BundleTeardown for Qwen2Bundle {
    /// Exact-retention checked teardown: delegates to the nested checked
    /// frees ([`Qwen2Weights::free_checked`],
    /// [`Qwen2State::free_checked`]), merging every failure category whole
    /// via [`GpuCleanupFailure::merge`] — owners that survive are carried
    /// in the returned [`GpuCleanupFailure`] for exact-retention retry.
    fn free_checked(self, gpu: &mut Gpu) -> Result<(), GpuCleanupFailure> {
        let Qwen2Bundle {
            config: _,
            weights,
            state,
        } = self;
        let mut cf = GpuCleanupFailure::empty();
        if let Err(f) = weights.free_checked(gpu) {
            cf.merge(f);
        }
        if let Err(f) = state.free_checked(gpu) {
            cf.merge(f);
        }
        if cf.is_empty() {
            Ok(())
        } else {
            Err(cf)
        }
    }
}

/// Build the Qwen2 GPU bundle from an HFQ or safetensors-directory source.
/// Refusals owned here. The Dir arm loads the full-precision F16 `.weight`
/// tensors (and the Q/K/V `attention_bias=true` biases) via the source
/// loaders — Qwen2 needs those biases, which the llama-family Dir loader
/// drops, so Qwen2 dirs route here (arch_id=7) instead of to LlamaCarrier.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen2Bundle, String> {
    if ctx.draft_path.is_some() {
        return Err(
            "DFlash not supported on arch_id=7 (qwen2 bring-up). Reload without a draft.".into(),
        );
    }
    if ctx.cask.sidecar.is_some() {
        return Err("CASK eviction not supported on arch_id=7 (qwen2 bring-up). Reload without --cask-sidecar.".into());
    }
    let (config, weights) = match src {
        ModelSource::Hfq(mut hfq) => {
            let config = <Qwen2 as Architecture>::config_from_hfq(&hfq)?;
            let weights = <Qwen2 as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
            (config, weights)
        }
        ModelSource::Dir(source) => {
            let config = crate::qwen2::config_from_source(&source).ok_or_else(|| {
                "qwen2: failed to parse Qwen2Config from safetensors config.json".to_string()
            })?;
            // The Dir path loads the F16 `.weight` tensors; the ParoQuant 4-bit
            // qweight decode is NOT implemented for qwen2 dirs. Fail cleanly if
            // the F16 weights are absent (a paro-only / qweight-only dir) rather
            // than panicking deep in the tensor loader, and warn when we ignore
            // a present paro quant_config (loading F16 ≈ 2x the VRAM of 4-bit).
            let has_f16_weights = source
                .tensor_info("model.layers.0.self_attn.q_proj.weight")
                .is_some();
            let is_paro = source.quant_config().is_some();
            if !has_f16_weights {
                return Err(format!(
                    "qwen2: safetensors dir has no F16 `.weight` tensors{} — 4-bit \
                     qwen2 dir loading is not implemented; use the HFQ (arch_id=7) build",
                    if is_paro {
                        " (ParoQuant qweight-only)"
                    } else {
                        ""
                    }
                ));
            }
            if is_paro {
                eprintln!(
                    "  qwen2: loading F16 `.weight` (ParoQuant 4-bit qweight ignored — ~2x VRAM)"
                );
            }
            let weights = crate::qwen2::load_weights_from_source(&source, &config, ctx.gpu)
                .map_err(|e| format!("qwen2: load_weights_from_source: {e:?}"))?;
            (config, weights)
        }
    };
    hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
    let state = Qwen2State::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
        .map_err(|e| format!("qwen2: Qwen2State::new_with_max_seq failed: {e:?}"))?;
    Ok(Qwen2Bundle {
        config,
        weights,
        state,
    })
}
