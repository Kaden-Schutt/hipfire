use crate::dspark_body::Qwen3DrafterAssets;
use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::dspark_core::DsparkWeights;
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

struct LlamaBundleStaging {
    config: LlamaConfig,
    weights: Option<LlamaWeights>,
    scratch: Option<ForwardScratch>,
    kv: Option<KvCache>,
}

impl LlamaBundleStaging {
    fn free_gpu(&mut self, gpu: &mut rdna_compute::Gpu) {
        if let Some(kv) = self.kv.take() {
            kv.free_gpu(gpu);
        }
        if let Some(scratch) = self.scratch.take() {
            scratch.free_gpu(gpu);
        }
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(gpu);
        }
    }

    fn into_bundle(mut self) -> LlamaBundle {
        LlamaBundle {
            config: self.config,
            weights: self.weights.take().expect("staged LLaMA weights"),
            scratch: self.scratch.take().expect("staged LLaMA scratch"),
            kv: self.kv.take().expect("staged LLaMA KV cache"),
            dflash_extract_layers: Vec::new(),
            dspark_weights: None,
            dspark_assets: None,
        }
    }
}

/// Build the LLaMA GPU bundle from an HFQ source.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("llama: directory source unsupported".into());
    };
    let config = <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
    let mut staged = LlamaBundleStaging {
        config,
        weights: None,
        scratch: None,
        kv: None,
    };
    let weights = match <Llama as Architecture>::load_weights(&mut hfq, &staged.config, ctx.gpu) {
        Ok(weights) => weights,
        Err(error) => return Err(error),
    };
    hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
    staged.weights = Some(weights);
    #[cfg(feature = "dflash-fault-inject")]
    if let Err(error) = hipfire_runtime::dflash_generic::generic_dflash_construction_boundary(
        hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetWeights,
    ) {
        staged.free_gpu(ctx.gpu);
        return Err(error);
    }
    // Size scratch (flash-attention partials) for the runtime KV cap so the
    // asym/flash attends, which index partials by ceil(physical_cap/128), don't
    // overflow it (the trait `new_state` only knows the model's declared max).
    let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &staged.config, ctx.max_seq) {
        Ok(scratch) => scratch,
        Err(error) => {
            staged.free_gpu(ctx.gpu);
            return Err(format!(
                "llama: ForwardScratch::new_with_max_seq failed: {error:?}"
            ));
        }
    };
    staged.scratch = Some(scratch);
    let dims = KvDims {
        layers: KvLayers::Flat(staged.config.n_layers),
        n_kv_heads: staged.config.n_kv_heads,
        head_dim: staged.config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: None,
    };
    let kv = match KvCache::from_mode(
        hipfire_runtime::kv_mode::resolve(
            ctx.kv_mode_override.unwrap_or(""),
            &hipfire_runtime::kv_mode::LLAMA_HFQ_POLICY,
            staged.config.head_dim,
        )
        .mode,
        KvTarget::Single(ctx.gpu),
        &dims,
    ) {
        Ok(kv) => kv,
        Err(error) => {
            staged.free_gpu(ctx.gpu);
            return Err(format!("llama: KvCache::from_mode failed: {error}"));
        }
    };
    staged.kv = Some(kv);
    #[cfg(feature = "dflash-fault-inject")]
    if let Err(error) = hipfire_runtime::dflash_generic::generic_dflash_construction_boundary(
        hipfire_runtime::dflash_generic::GenericDflashConstructionStage::TargetKv,
    ) {
        staged.free_gpu(ctx.gpu);
        return Err(error);
    }
    Ok(staged.into_bundle())
}

impl LlamaBundle {
    pub fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        let LlamaBundle {
            config: _,
            weights,
            scratch,
            kv,
            dflash_extract_layers: _,
            dspark_weights,
            dspark_assets,
        } = self;
        if let Some(assets) = dspark_assets {
            assets.weights.free_gpu(gpu);
            assets.kv.free_gpu(gpu);
            assets.scratch.free_gpu(gpu);
            assets.pbs.free_gpu(gpu);
        }
        if let Some(weights) = dspark_weights {
            weights.free_gpu(gpu);
        }
        kv.free_gpu(gpu);
        scratch.free_gpu(gpu);
        weights.free_gpu(gpu);
    }

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
