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

/// Build the LLaMA GPU bundle from an HFQ source.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("llama: directory source unsupported".into());
    };
    let config = <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
    hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
    // Size scratch (flash-attention partials) for the runtime KV cap so the
    // asym/flash attends, which index partials by ceil(physical_cap/128), don't
    // overflow it (the trait `new_state` only knows the model's declared max).
    let scratch = ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
        .map_err(|e| format!("llama: ForwardScratch::new_with_max_seq failed: {e:?}"))?;
    let dims = KvDims {
        layers: KvLayers::Flat(config.n_layers),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: None,
    };
    let kv = KvCache::from_mode(
        hipfire_runtime::kv_mode::resolve(
            ctx.kv_mode_override.unwrap_or(""),
            &hipfire_runtime::kv_mode::LLAMA_HFQ_POLICY,
            config.head_dim,
        )
        .mode,
        KvTarget::Single(ctx.gpu),
        &dims,
    )
    .map_err(|e| format!("llama: KvCache::from_mode failed: {e}"))?;
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
