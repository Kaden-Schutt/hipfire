// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::dspark_body::Qwen3DrafterAssets;
use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::dspark_core::DsparkWeights;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{
    ForwardScratch, KvCache, KvDims, KvLayers, KvTarget, LlamaConfig, LlamaWeights,
};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
    /// Decoder-layer indices whose residual hidden states a hidden-conditioned
    /// drafter (DFlash / EAGLE) wants captured, ascending order. Empty = no
    /// capture (the `SpecTarget::dflash_extract_layers` default of `None`). The
    /// speculator sets the real `target_layer_ids` via
    /// [`LlamaBundle::set_dflash_extract_layers`].
    pub dflash_extract_layers: Vec<usize>,
    /// Loaded DSpark drafter sidecar globals. `None` when no `-dspark` sidecar
    /// was found or speculation was disabled. Task-10 wires the speculator build.
    pub dspark_weights: Option<DsparkWeights>,
    /// Loaded DSpark drafter body assets (5-layer dense-GQA transformer +
    /// block-only KvCache/scratch).  `None` when `dspark_weights` is `None`.
    pub dspark_assets: Option<Qwen3DrafterAssets>,
}

/// Build the LLaMA GPU bundle from an HFQ or safetensors-directory source.
///
/// Verbatim relocation of the carrier's `(config, weights, kv, scratch)`
/// seam: HFQ via `Architecture` trait, Dir via ParoQuant loaders. Error
/// strings are byte-identical to the prior inline carrier block.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let (config, weights, kv, scratch) = match src {
        ModelSource::Hfq(mut hfq) => {
            let config =
                <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
            let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            // Every GPU-backed stage stays owned until the bundle is
            // published. Explicitly free earlier stages on each later error;
            // GpuTensor intentionally has no global Drop implementation.
            let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                Ok(scratch) => scratch,
                Err(error) => {
                    weights.free_gpu(ctx.gpu);
                    return Err(format!(
                        "llama: ForwardScratch::new_with_max_seq failed: {error:?}"
                    ));
                }
            };
            let dims = KvDims {
                layers: KvLayers::Flat(config.n_layers),
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                max_seq: ctx.max_seq,
                physical_cap: None,
            };
            let kv = match <KvCache as KvCacheExt>::from_mode(
                hipfire_runtime::kv_mode::resolve(
                    ctx.kv_mode_override.unwrap_or(""),
                    &hipfire_runtime::kv_mode::LLAMA_HFQ_POLICY,
                    config.head_dim,
                )
                .mode,
                KvTarget::Single(ctx.gpu),
                &dims,
            ) {
                Ok(kv) => kv,
                Err(error) => {
                    scratch.free_gpu(ctx.gpu);
                    weights.free_gpu(ctx.gpu);
                    return Err(format!(
                        "llama: <KvCache as KvCacheExt>::from_mode failed: {error}"
                    ));
                }
            };
            (config, weights, kv, scratch)
        }
        ModelSource::Dir(source) => {
            let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
                .map_err(|e| format!("failed to parse LLaMA/Qwen3 config from config.json: {e}"))?;
            let weights =
                hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, ctx.gpu)
                    .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            // Replicate carriers.rs `resolve_kv_mode` warning path verbatim.
            let kv_mode_str = ctx
                .kv_mode_override
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
            let rr = hipfire_runtime::kv_mode::resolve(
                &kv_mode_str,
                &hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY,
                config.head_dim,
            );
            if let Some(w) = rr.warning {
                eprintln!(
                    "  KV cache: {w} (site {})",
                    hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY.site
                );
            }
            let dims = KvDims {
                layers: KvLayers::Flat(config.n_layers),
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                max_seq: ctx.max_seq,
                physical_cap: Some(ctx.max_seq),
            };
            let kv =
                match <KvCache as KvCacheExt>::from_mode(rr.mode, KvTarget::Single(ctx.gpu), &dims)
                {
                    Ok(kv) => kv,
                    Err(error) => {
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("KvCache: {error}"));
                    }
                };
            let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                Ok(scratch) => scratch,
                Err(error) => {
                    let _ = kv.free_gpu(ctx.gpu);
                    weights.free_gpu(ctx.gpu);
                    return Err(format!("ForwardScratch::new_with_max_seq: {error:?}"));
                }
            };
            (config, weights, kv, scratch)
        }
    };
    Ok(LlamaBundle {
        config,
        weights,
        scratch,
        kv,
        dflash_extract_layers: Vec::new(),
        dspark_weights: None,
        dspark_assets: None,
    })
}

/// Alias matching the `load_<arch>_bundle` naming convention in the task.
pub use load_bundle as load_llama_bundle;

impl LlamaBundle {
    /// Set the decoder-layer indices whose residual hidden states the
    /// hidden-conditioned drafter wants captured (ascending order). The
    /// speculator calls this with `dflash::DflashConfig::target_layer_ids`.
    pub fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        debug_assert!(
            layers.windows(2).all(|w| w[0] < w[1]),
            "dflash extract layers must be strictly ascending: {layers:?}"
        );
        self.dflash_extract_layers = layers;
    }
}
