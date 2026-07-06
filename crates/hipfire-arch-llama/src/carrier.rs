use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::llama::{
    ForwardScratch, KvCache, KvDims, KvLayers, KvTarget, LlamaConfig, LlamaWeights,
};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
}

/// Build the LLaMA GPU bundle from an HFQ source.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("llama: directory source unsupported".into());
    };
    let config = <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
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
            "",
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
    })
}
