// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::dspark_body::Qwen3DrafterAssets;
use crate::Llama;
use hipfire_hardware::DeviceMesh;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::dspark_core::DsparkWeights;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{
    EmbeddingFormat, ForwardScratch, KvCache, KvDims, KvLayers, KvTarget, LayerWeights,
    LlamaConfig, LlamaWeights, WeightTensor,
};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::weight_backend::hfq_weight_dtype;
use hipfire_runtime::weight_manifest::{plan_manifest, ManifestPlan, WeightEntry};
use hipfire_runtime::weight_store::{
    TakenWeight, WeightHandle, WeightStore, WeightStoreAssembly, WeightStoreAssemblyGuard,
    WeightOrigin,
};
use rdna_compute::{DType, GpuTensor};
use std::collections::HashMap;

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
    /// Pure declaration/placement plan captured at load time. The plan has no
    /// GPU handles and is immutable after publication.
    pub manifest_plan: ManifestPlan,
    /// A pilot store is attached only after its handles are assembled under
    /// this bundle. It is crate-visible so callers cannot create an independent
    /// unload owner; `ArchModel::free_gpu` is the sole release path.
    pub(crate) weight_store: Option<WeightStore>,
    /// Exact target identity captured when this bundle was admitted. The
    /// owner uses it to validate every store-origin component before teardown.
    pub(crate) weight_origin: WeightOrigin,
    pub(crate) mesh: DeviceMesh,
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

    /// block-only KvCache/scratch). `None` when `dspark_weights` is `None`.
    pub dspark_assets: Option<Qwen3DrafterAssets>,
}
fn plan_single(config: &LlamaConfig) -> Result<(DeviceMesh, ManifestPlan), String> {
    let mesh = DeviceMesh::single();
    let manifest = Llama::weight_manifest(config);
    let state = Llama::state_manifest(config);
    let plan = plan_manifest(&manifest, &state, &mesh, config.n_layers)
        .map_err(|e| format!("llama: manifest planning failed: {e}"))?;
    Ok((mesh, plan))
}

fn llama_kv_dims(config: &LlamaConfig, max_seq: usize, physical_cap: Option<usize>) -> KvDims {
    KvDims {
        layers: KvLayers::Flat(config.n_layers),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq,
        physical_cap,
    }
}

fn hfq_layer_names(layer: usize, relative: &str) -> Vec<String> {
    vec![
        format!("model.layers.{layer}.{relative}.weight"),
        format!("layers.{layer}.{relative}.weight"),
    ]
}

fn hfq_entry_names(entry: &WeightEntry) -> Result<Vec<String>, String> {
    let names = match (entry.name.as_str(), entry.layer) {
        ("token_embd", None) => vec!["model.embed_tokens.weight".to_string()],
        ("output_norm", None) => vec!["model.norm.weight".to_string()],
        ("lm_head", None) => vec![
            "lm_head.weight".to_string(),
            "model.lm_head.weight".to_string(),
            "model.language_model.lm_head.weight".to_string(),
        ],
        ("wq", Some(layer)) => hfq_layer_names(layer, "self_attn.q_proj"),
        ("wk", Some(layer)) => hfq_layer_names(layer, "self_attn.k_proj"),
        ("wv", Some(layer)) => hfq_layer_names(layer, "self_attn.v_proj"),
        ("wo", Some(layer)) => hfq_layer_names(layer, "self_attn.o_proj"),
        ("ffn_gate", Some(layer)) => hfq_layer_names(layer, "mlp.gate_proj"),
        ("ffn_up", Some(layer)) => hfq_layer_names(layer, "mlp.up_proj"),
        ("ffn_down", Some(layer)) => hfq_layer_names(layer, "mlp.down_proj"),
        ("attn_norm", Some(layer)) => hfq_layer_names(layer, "input_layernorm"),
        ("ffn_norm", Some(layer)) => hfq_layer_names(layer, "post_attention_layernorm"),
        ("q_norm", Some(layer)) => hfq_layer_names(layer, "self_attn.q_norm"),
        ("k_norm", Some(layer)) => hfq_layer_names(layer, "self_attn.k_norm"),
        (name, layer) => {
            return Err(format!(
                "llama: manifest entry {name}[layer {layer:?}] has no HFQ source mapping"
            ));
        }
    };
    Ok(names)
}

fn hfq_entry_data(hfq: &HfqFile, entry: &WeightEntry) -> Result<(Vec<u8>, u8), String> {
    for name in hfq_entry_names(entry)? {
        if let Some((info, data)) = hfq.tensor_data_vec(&name) {
            if !matches!(
                entry.name.as_str(),
                "token_embd" | "output_norm" | "attn_norm" | "ffn_norm" | "q_norm" | "k_norm"
            ) {
                let sidecar = match name.strip_suffix(".weight") {
                    Some(stem) => format!("{stem}.awq_scale.weight"),
                    None => format!("{name}.awq_scale.weight"),
                };
                if hfq.find_tensor_info(&sidecar).is_some() {
                    return Err(format!(
                        "llama: AWQ sidecar {sidecar} is not represented by the manifest pilot"
                    ));
                }
            }
            return Ok((data, info.quant_type));
        }
    }
    if entry.name == "lm_head" && entry.layer.is_none() {
        if let Some((info, data)) = hfq.tensor_data_vec("model.embed_tokens.weight") {
            return Ok((data, info.quant_type));
        }
    }
    Err(format!(
        "llama: source tensor for {}[layer {:?}] is missing",
        entry.name, entry.layer
    ))
}

fn f32_bytes_from_hfq(quant_type: u8, data: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(match quant_type {
        1 | 16 => data.len() * 2,
        2 => data.len(),
        _ => 0,
    });
    match quant_type {
        1 => {
            let chunks = data.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(format!("{name}: truncated F16 payload"));
            }
            for chunk in chunks {
                bytes.extend_from_slice(
                    &hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                        .to_le_bytes(),
                );
            }
        }
        2 => {
            if !data.len().is_multiple_of(4) {
                return Err(format!("{name}: truncated F32 payload"));
            }
            bytes.extend_from_slice(data);
        }
        16 => {
            let chunks = data.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(format!("{name}: truncated BF16 payload"));
            }
            for chunk in chunks {
                bytes.extend_from_slice(
                    &f32::from_bits(
                        u16::from_le_bytes([chunk[0], chunk[1]]) as u32 * (1 << 16),
                    )
                    .to_le_bytes(),
                );
            }
        }
        other => {
            return Err(format!(
                "{name}: quant_type={other} is not a host float payload"
            ));
        }
    }
    Ok(bytes)
}

fn hfq_source(hfq: &HfqFile, entry: &WeightEntry) -> Result<(Vec<u8>, DType), String> {
    let (data, quant_type) = hfq_entry_data(hfq, entry)?;
    let name = format!("{}[layer {:?}]", entry.name, entry.layer);
    if entry.name == "token_embd" {
        return match quant_type {
            1 | 2 | 16 => Ok((f32_bytes_from_hfq(quant_type, &data, &name)?, DType::F32)),
            3 => Ok((data, DType::Q8_0)),
            4 => Ok((data, DType::Q4K)),
            6 => Ok((data, DType::HFQ4G256)),
            7 => Ok((data, DType::HFQ4G128)),
            other => Err(format!(
                "{name}: quant_type={other} is unsupported for a LLaMA embedding"
            )),
        };
    }
    if matches!(entry.name.as_str(), "output_norm" | "attn_norm" | "ffn_norm" | "q_norm" | "k_norm")
    {
        return Ok((
            f32_bytes_from_hfq(quant_type, &data, &name)?,
            DType::F32,
        ));
    }
    match quant_type {
        1 | 2 | 16 => Ok((f32_bytes_from_hfq(quant_type, &data, &name)?, DType::F32)),
        other => hfq_weight_dtype(other)
            .map(|dtype| (data, dtype))
            .ok_or_else(|| format!("{name}: unsupported HFQ quant_type={other}")),
    }
}

fn take_slot(
    assembly: &mut WeightStoreAssembly<'_>,
    slots: &mut HashMap<(String, Option<usize>), usize>,
    name: &str,
    layer: Option<usize>,
) -> Result<(), String> {
    let slot = assembly
        .take(name, layer, 0)
        .ok_or_else(|| format!("llama: fulfilled store is missing {name}[layer {layer:?}]"))?;
    slots.insert((name.to_string(), layer), slot);
    Ok(())
}

fn require_resident(
    assembly: &WeightStoreAssemblyGuard<'_>,
    name: &str,
    layer: Option<usize>,
    slot: usize,
) -> Result<(), String> {
    if matches!(assembly.get(slot), Some(WeightHandle::Resident(_))) {
        Ok(())
    } else {
        Err(format!(
            "llama: {name}[layer {layer:?}] is an alias; typed LLaMA assembly requires a resident handle"
        ))
    }
}

fn resident_cell(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    name: &str,
    layer: Option<usize>,
) -> Result<GpuTensor, String> {
    let taken = cells
        .remove(&(name.to_string(), layer))
        .ok_or_else(|| format!("llama: assembled store is missing {name}[layer {layer:?}]"))?;
    match taken.handle {
        WeightHandle::Resident(tensor) => Ok(tensor),
        WeightHandle::Alias(source) => Err(format!(
            "llama: {name}[layer {layer:?}] aliases {source}, expected resident handle"
        )),
    }
}

fn resident_weight(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    name: &str,
    layer: Option<usize>,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let tensor = resident_cell(cells, name, layer)?;
    let dtype = tensor.dtype;
    Ok(WeightTensor {
        buf: tensor,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: dtype.row_stride(k),
        paro: None,
        awq_scale: None,
    })
}

fn embedding_format(dtype: DType) -> Result<EmbeddingFormat, String> {
    match dtype {
        DType::F32 => Ok(EmbeddingFormat::F32),
        DType::Q4K => Ok(EmbeddingFormat::Q4K),
        DType::HFQ4G256 => Ok(EmbeddingFormat::HFQ4G256),
        DType::HFQ4G128 => Ok(EmbeddingFormat::HFQ4G128),
        DType::Q8_0 => Ok(EmbeddingFormat::Q8_0),
        other => Err(format!(
            "llama: unsupported assembled embedding dtype {other:?}"
        )),
    }
}

fn assemble_llama_weights(
    config: &LlamaConfig,
    store: &mut WeightStore,
) -> Result<LlamaWeights, String> {
    let mut assembly = store.begin_assembly();
    let mut slots = HashMap::new();
    let mut take = |name: &str, layer: Option<usize>| take_slot(&mut assembly, &mut slots, name, layer);

    take("token_embd", None)?;
    take("output_norm", None)?;
    take("lm_head", None)?;
    for layer in 0..config.n_layers {
        for name in ["wq", "wk", "wv", "wo", "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm"] {
            take(name, Some(layer))?;
        }
        if config.has_qk_norm {
            take("q_norm", Some(layer))?;
            take("k_norm", Some(layer))?;
        }
    }

    drop(take);
    let guard = assembly.commit();
    for ((name, layer), slot) in &slots {
        require_resident(&guard, name, *layer, *slot)?;
    }
    let cells: HashMap<_, _> = guard
        .finalize()
        .into_iter()
        .map(|taken| ((taken.key.name.clone(), taken.key.layer), taken))
        .collect();
    let mut cells = cells;
    let token_embd = resident_cell(&mut cells, "token_embd", None)?;
    let embd_format = embedding_format(token_embd.dtype)?;
    let output_norm = resident_cell(&mut cells, "output_norm", None)?;
    let output = resident_weight(
        &mut cells,
        "lm_head",
        None,
        config.vocab_size,
        config.dim,
    )?;
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        let q_norm = if config.has_qk_norm {
            Some(resident_cell(&mut cells, "q_norm", Some(layer))?)
        } else {
            None
        };
        let k_norm = if config.has_qk_norm {
            Some(resident_cell(&mut cells, "k_norm", Some(layer))?)
        } else {
            None
        };
        layers.push(LayerWeights {
            attn_norm: resident_cell(&mut cells, "attn_norm", Some(layer))?,
            wq: resident_weight(
                &mut cells,
                "wq",
                Some(layer),
                config.n_heads * config.head_dim,
                config.dim,
            )?,
            wk: resident_weight(
                &mut cells,
                "wk",
                Some(layer),
                config.n_kv_heads * config.head_dim,
                config.dim,
            )?,
            wv: resident_weight(
                &mut cells,
                "wv",
                Some(layer),
                config.n_kv_heads * config.head_dim,
                config.dim,
            )?,
            wo: resident_weight(
                &mut cells,
                "wo",
                Some(layer),
                config.dim,
                config.n_heads * config.head_dim,
            )?,
            q_norm,
            k_norm,
            ffn_norm: resident_cell(&mut cells, "ffn_norm", Some(layer))?,
            w_gate: resident_weight(
                &mut cells,
                "ffn_gate",
                Some(layer),
                config.hidden_dim,
                config.dim,
            )?,
            w_up: resident_weight(
                &mut cells,
                "ffn_up",
                Some(layer),
                config.hidden_dim,
                config.dim,
            )?,
            w_down: resident_weight(
                &mut cells,
                "ffn_down",
                Some(layer),
                config.dim,
                config.hidden_dim,
            )?,
        });
    }
    Ok(LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd: false,
    })
}

/// Build the LLaMA GPU bundle from an HFQ or safetensors-directory source.
///
/// The HFQ plain-LLaMA Single path is the production manifest pilot: planning
/// and source admission happen first, fulfillment uploads transactionally, and
/// typed handles are moved into `LlamaWeights` before the committed remainder
/// is published beneath this bundle's owner. The directory path remains on its
/// existing ParoQuant loader until that source has an equivalent representation
/// resolver.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let (config, weights, kv, scratch, manifest_plan, weight_store, mesh, weight_origin) =
        match src {
            ModelSource::Hfq(hfq) => {
                let config =
                    <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
                let (mesh, manifest_plan) = plan_single(&config)?;
                let weight_origin = WeightOrigin::for_single(&mesh, ctx.gpu);
                let mut store = hipfire_runtime::weight_store::fulfill_manifest(
                    &Llama::weight_manifest(&config),
                    &mesh,
                    config.n_layers,
                    ctx.gpu,
                    |entry| hfq_source(&hfq, entry),
                )
                .map_err(|e| format!("llama: {e}"))?;
                let weights = match assemble_llama_weights(&config, &mut store) {
                    Ok(weights) => weights,
                    Err(error) => {
                        store.rollback_unpublished(ctx.gpu);
                        return Err(error);
                    }
                };
                hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
                // The plain LLaMA path has no independent cap resolver. PR
                // #661's physical-cap behavior is owned by the existing
                // upstream KV plan.
                let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                    Ok(scratch) => scratch,
                    Err(error) => {
                        store.rollback_unpublished(ctx.gpu);
                        weights.free_gpu(ctx.gpu);
                        return Err(format!(
                            "llama: ForwardScratch::new_with_max_seq failed: {error:?}"
                        ));
                    }
                };
                let dims = llama_kv_dims(&config, ctx.max_seq, None);
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
                        store.rollback_unpublished(ctx.gpu);
                        weights.free_gpu(ctx.gpu);
                        return Err(format!(
                            "llama: <KvCache as KvCacheExt>::from_mode failed: {error}"
                        ));
                    }
                };
                (
                    config,
                    weights,
                    kv,
                    scratch,
                    manifest_plan,
                    Some(store),
                    mesh,
                    weight_origin,
                )
            }
            ModelSource::Dir(source) => {
                let config =
                    hipfire_runtime::hfq::config_from_safetensors_llama(&source).map_err(|e| {
                        format!("failed to parse LLaMA/Qwen3 config from config.json: {e}")
                    })?;
                let (mesh, manifest_plan) = plan_single(&config)?;
                let weight_origin = WeightOrigin::for_single(&mesh, ctx.gpu);
                let weights =
                    hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, ctx.gpu)
                        .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;
                hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
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
                let dims = llama_kv_dims(&config, ctx.max_seq, Some(ctx.max_seq));
                let kv = match <KvCache as KvCacheExt>::from_mode(
                    rr.mode,
                    KvTarget::Single(ctx.gpu),
                    &dims,
                ) {
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
                        return Err(format!(
                            "ForwardScratch::new_with_max_seq: {error:?}"
                        ));
                    }
                };
                (
                    config,
                    weights,
                    kv,
                    scratch,
                    manifest_plan,
                    None,
                    mesh,
                    weight_origin,
                )
            }
        };

    let mut bundle = LlamaBundle {
        config,
        weights,
        scratch,
        kv,
        manifest_plan,
        weight_store: None,
        weight_origin,
        mesh,
        dflash_extract_layers: Vec::new(),
        dspark_weights: None,
        dspark_assets: None,
    };
    if let Some(store) = weight_store {
        if let Err((store, error)) = bundle.attach_weight_store(store) {
            let LlamaBundle {
                weights,
                scratch,
                kv,
                ..
            } = bundle;
            store.rollback_unpublished(ctx.gpu);
            scratch.free_gpu(ctx.gpu);
            weights.free_gpu(ctx.gpu);
            let _ = kv.free_gpu(ctx.gpu);
            return Err(error);
        }
    }
    Ok(bundle)
}

/// Alias matching the `load_<arch>_bundle` naming convention in the task.
pub use load_bundle as load_llama_bundle;

impl LlamaBundle {
    /// Attach a store whose resident handles have been assembled for this
    /// bundle. The complete target origin (mesh epoch, logical rank, and
    /// physical device) is checked before publication; teardown remains
    /// exclusively in `ArchModel::free_gpu`. On rejection the store is
    /// returned unchanged so the caller can retry against the right owner.
    pub fn attach_weight_store(
        &mut self,
        store: WeightStore,
    ) -> Result<(), (WeightStore, String)> {
        if self.weight_store.is_some() {
            return Err((store, "llama: weight store already attached".into()));
        }
        if let Err(error) = store.validate_origin_value(self.weight_origin) {
            return Err((store, format!("llama: weight store origin rejected: {error}")));
        }
        self.weight_store = Some(store);
        Ok(())
    }

    /// The immutable mesh identity used by this bundle's manifest plan.
    /// Callers that run the Single pilot must pass this exact mesh to
    /// `fulfill_manifest`; constructing a fresh `DeviceMesh::single()` would
    /// intentionally fail the origin check.
    pub fn manifest_mesh(&self) -> &DeviceMesh {
        &self.mesh
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

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::llama::ModelArch;
    use hipfire_runtime::weight_store::{WeightProjection, WeightProjectionKind};

    fn config() -> LlamaConfig {
        LlamaConfig {
            arch: ModelArch::Llama,
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 8,
            head_dim: 4,
            norm_eps: 1e-5,
            max_seq_len: 32,
            rope_freq_base: 10_000.0,
            bos_token: 1,
            eos_token: 2,
            has_qk_norm: false,
        }
    }

    fn alias_projection() -> WeightProjection {
        WeightProjection {
            kind: WeightProjectionKind::Static,
            axis: None,
            rank: 0,
            world_size: 1,
            logical_shape: vec![1],
            dtype: DType::F32,
        }
    }

    #[test]
    fn single_plan_covers_every_typed_llama_handle() {
        let (mesh, plan) = plan_single(&config()).unwrap();
        let manifest = Llama::weight_manifest(&config());
        assert_eq!(mesh.n_devices(), 1);
        assert_eq!(plan.weights.len(), 12);
        assert_eq!(plan.state.len(), 1);
        assert!(plan.collective_schedule.iter().any(|entry| entry.name == "wo"));
        assert!(manifest[0].dtype_constraint.accepts(DType::HFQ4G256));
        assert!(manifest[1].dtype_constraint.accepts(DType::MQ4G256));
        assert!(manifest[9].dtype_constraint.accepts(DType::F32));
        assert!(!manifest[9].dtype_constraint.accepts(DType::F16));
    }

    #[test]
    fn typed_assembly_rolls_back_when_a_cell_is_not_resident() {
        let mesh = DeviceMesh::single();
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, 0);
        let mut store = WeightStore::with_origin(origin);
        for name in ["token_embd", "output_norm", "lm_head"] {
            store
                .stage_alias(name, None, 0, "source", alias_projection())
                .unwrap();
        }
        let error = match assemble_llama_weights(
            &LlamaConfig {
                n_layers: 0,
                ..config()
            },
            &mut store,
        ) {
            Ok(_) => panic!("alias unexpectedly assembled as typed weights"),
            Err(error) => error,
        };
        assert!(error.contains("alias"));
        assert_eq!(store.len(), 3);
        assert!(store.contains("token_embd", None, 0));
        assert!(store.projection("lm_head", None, 0).is_some());
    }


    #[test]
    fn hfq_float_widening_matches_legacy_f32_representation() {
        let f16_one = [0x00, 0x3c, 0x00, 0xc0];
        let actual = f32_bytes_from_hfq(1, &f16_one, "test").unwrap();
        let expected = [1.0f32, -2.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn manifest_constraints_admit_every_pilot_representation() {
        let manifest = Llama::weight_manifest(&config());
        assert!(manifest[0].dtype_constraint.accepts(DType::HFQ4G256));
        assert!(manifest[1].dtype_constraint.accepts(DType::MQ4G256));
        assert!(manifest[9].dtype_constraint.accepts(DType::F32));
        assert!(!manifest[9].dtype_constraint.accepts(DType::F16));
    }
    #[test]
    fn physical_cap_remains_separate_from_configured_max_seq() {
        let dims = llama_kv_dims(&config(), 32_768, Some(4_096));
        assert_eq!(dims.max_seq, 32_768);
        assert_eq!(dims.physical_cap, Some(4_096));
    }
}
