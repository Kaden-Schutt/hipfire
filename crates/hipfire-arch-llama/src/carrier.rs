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
use hipfire_runtime::weight_store::{
    AttachedWeightStore, TakenWeight, WeightHandle, WeightLoadTransaction,
    WeightStoreAssembly, WeightStoreAssemblyGuard, WeightOrigin,
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
    pub(crate) weight_store: Option<AttachedWeightStore>,
    /// Exact target identity captured before publication. The attached store
    /// binds this identity into its private drain capability, so teardown
    /// cannot encounter an origin mismatch.
    pub(crate) weight_origin: WeightOrigin,
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
fn plan_single(
    config: &LlamaConfig,
    has_separate_lm_head: bool,
) -> Result<(DeviceMesh, ManifestPlan), String> {
    let mesh = DeviceMesh::single();
    let manifest = Llama::weight_manifest_for_hfq(config, has_separate_lm_head);
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

fn require_materialized(
    assembly: &WeightStoreAssemblyGuard<'_>,
    name: &str,
    layer: Option<usize>,
    slot: usize,
) -> Result<(), String> {
    match assembly.get(slot) {
        Some(WeightHandle::Resident(_)) => Ok(()),
        Some(WeightHandle::Alias(source))
            if name == "lm_head" && layer.is_none() && source == "token_embd" =>
        {
            Ok(())
        }
        Some(WeightHandle::Alias(source)) => Err(format!(
            "llama: {name}[layer {layer:?}] aliases {source}; only lm_head may tie token_embd"
        )),
        None => Err(format!(
            "llama: {name}[layer {layer:?}] assembly slot {slot} is missing"
        )),
    }
}

fn resident_cell(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    name: &str,
    layer: Option<usize>,
) -> GpuTensor {
    match cells.remove(&(name.to_string(), layer)) {
        Some(TakenWeight {
            handle: WeightHandle::Resident(tensor),
            ..
        }) => tensor,
        _ => unreachable!("validated LLaMA assembly lost resident {name}[layer {layer:?}]"),
    }
}

fn resident_weight(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    name: &str,
    layer: Option<usize>,
    m: usize,
    k: usize,
) -> WeightTensor {
    let tensor = resident_cell(cells, name, layer);
    let dtype = tensor.dtype;
    WeightTensor {
        buf: tensor,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: dtype.row_stride(k),
        paro: None,
        awq_scale: None,
    }
}

fn tied_weight(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    token_embd: &GpuTensor,
    embd_format: EmbeddingFormat,
    name: &str,
    layer: Option<usize>,
    m: usize,
    k: usize,
) -> WeightTensor {
    match cells.remove(&(name.to_string(), layer)) {
        Some(TakenWeight {
            handle: WeightHandle::Alias(source),
            ..
        }) if source == "token_embd" => {
            hipfire_runtime::weight_backend::tied_lm_head_alias(token_embd, embd_format, m, k)
        }
        _ => unreachable!("validated LLaMA assembly lost tied {name}[layer {layer:?}]"),
    }
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
    transaction: &mut WeightLoadTransaction,
) -> Result<LlamaWeights, String> {
    let mut assembly = transaction.begin_assembly();
    let mut slots = HashMap::new();
    let mut take = |name: &str, layer: Option<usize>| {
        take_slot(&mut assembly, &mut slots, name, layer)
    };

    take("token_embd", None)?;
    take("output_norm", None)?;
    take("lm_head", None)?;
    for layer in 0..config.n_layers {
        for name in [
            "wq",
            "wk",
            "wv",
            "wo",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "attn_norm",
            "ffn_norm",
        ] {
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
        require_materialized(&guard, name, *layer, *slot)?;
    }
    let token_slot = slots[&("token_embd".to_string(), None)];
    let token_dtype = match guard.get(token_slot) {
        Some(WeightHandle::Resident(tensor)) => tensor.dtype,
        _ => unreachable!("validated token_embd is not resident"),
    };
    let embd_format = embedding_format(token_dtype)?;
    let cells: HashMap<_, _> = guard
        .finalize()
        .into_iter()
        .map(|taken| ((taken.key.name.clone(), taken.key.layer), taken))
        .collect();
    let mut cells = cells;
    let token_embd = resident_cell(&mut cells, "token_embd", None);
    let output_norm = resident_cell(&mut cells, "output_norm", None);
    let lm_head_aliases_embd = matches!(
        cells.get(&("lm_head".to_string(), None)),
        Some(TakenWeight {
            handle: WeightHandle::Alias(_),
            ..
        })
    );
    let output = if lm_head_aliases_embd {
        tied_weight(
            &mut cells,
            &token_embd,
            embd_format,
            "lm_head",
            None,
            config.vocab_size,
            config.dim,
        )
    } else {
        resident_weight(
            &mut cells,
            "lm_head",
            None,
            config.vocab_size,
            config.dim,
        )
    };
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        let q_norm = if config.has_qk_norm {
            Some(resident_cell(&mut cells, "q_norm", Some(layer)))
        } else {
            None
        };
        let k_norm = if config.has_qk_norm {
            Some(resident_cell(&mut cells, "k_norm", Some(layer)))
        } else {
            None
        };
        layers.push(LayerWeights {
            attn_norm: resident_cell(&mut cells, "attn_norm", Some(layer)),
            wq: resident_weight(
                &mut cells,
                "wq",
                Some(layer),
                config.n_heads * config.head_dim,
                config.dim,
            ),
            wk: resident_weight(
                &mut cells,
                "wk",
                Some(layer),
                config.n_kv_heads * config.head_dim,
                config.dim,
            ),
            wv: resident_weight(
                &mut cells,
                "wv",
                Some(layer),
                config.n_kv_heads * config.head_dim,
                config.dim,
            ),
            wo: resident_weight(
                &mut cells,
                "wo",
                Some(layer),
                config.dim,
                config.n_heads * config.head_dim,
            ),
            q_norm,
            k_norm,
            ffn_norm: resident_cell(&mut cells, "ffn_norm", Some(layer)),
            w_gate: resident_weight(
                &mut cells,
                "ffn_gate",
                Some(layer),
                config.hidden_dim,
                config.dim,
            ),
            w_up: resident_weight(
                &mut cells,
                "ffn_up",
                Some(layer),
                config.hidden_dim,
                config.dim,
            ),
            w_down: resident_weight(
                &mut cells,
                "ffn_down",
                Some(layer),
                config.dim,
                config.hidden_dim,
            ),
        });
    }
    debug_assert!(cells.is_empty(), "validated LLaMA assembly left cells");
    Ok(LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HfqLoadRoute {
    /// Plain, non-AWQ files admitted to the manifest/typed-assembly pilot.
    ManifestPlainLlama,
    /// Files carrying AWQ scale sidecars retain the established loader until
    /// sidecar ownership is represented by the manifest transaction.
    LegacyAwq,
}

fn classify_hfq_route(hfq: &HfqFile) -> HfqLoadRoute {
    if hfq.has_awq_sidecars() {
        HfqLoadRoute::LegacyAwq
    } else {
        HfqLoadRoute::ManifestPlainLlama
    }
}

pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let (config, weights, kv, scratch, manifest_plan, weight_store, mesh, weight_origin) =
        match src {
            ModelSource::Hfq(hfq) => {
                let config =
                    <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
                // Admission and route classification are pure source checks.
                // They must run before any manifest fulfillment or GPU upload.
                hipfire_runtime::hfq::validate_llama_hfq_admission(&hfq)
                    .map_err(|e| e.to_string())?;
                let has_separate_lm_head =
                    hfq.find_tensor_info("lm_head.weight").is_some();
                let route = classify_hfq_route(&hfq);
                eprintln!("llama: HFQ source route = {route:?}");
                let (mesh, manifest_plan) = plan_single(&config, has_separate_lm_head)?;
                let weight_origin = WeightOrigin::for_single(&mesh, ctx.gpu);
                let (weights, mut weight_store) = match route {
                    HfqLoadRoute::LegacyAwq => {
                        let weights =
                            hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, ctx.gpu)
                                .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))?;
                        (weights, None)
                    }
                    HfqLoadRoute::ManifestPlainLlama => {
                        let manifest =
                            Llama::weight_manifest_for_hfq(&config, has_separate_lm_head);
                        let mut transaction = hipfire_runtime::weight_store::fulfill_manifest(
                            &manifest,
                            &mesh,
                            config.n_layers,
                            ctx.gpu,
                            |entry| hfq_source(&hfq, entry),
                        )
                        .map_err(|e| format!("llama: {e}"))?;
                        let weights = match assemble_llama_weights(&config, &mut transaction) {
                            Ok(weights) => weights,
                            Err(error) => {
                                transaction.rollback(ctx.gpu);
                                return Err(error);
                            }
                        };
                        (weights, Some(transaction))
                    }
                };
                hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
                // The plain LLaMA path has no independent cap resolver. PR
                // #661's physical-cap behavior is owned by the existing
                // upstream KV plan.
                let scratch =
                    match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                        Ok(scratch) => scratch,
                        Err(error) => {
                            if let Some(transaction) = weight_store.take() {
                                transaction.rollback(ctx.gpu);
                            }
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
                        if let Some(transaction) = weight_store.take() {
                            transaction.rollback(ctx.gpu);
                        }
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
                    weight_store,
                    mesh,
                    weight_origin,
                )
            }
            ModelSource::Dir(source) => {
                let config =
                    hipfire_runtime::hfq::config_from_safetensors_llama(&source).map_err(|e| {
                        format!("failed to parse LLaMA/Qwen3 config from config.json: {e}")
                    })?;
                let (mesh, manifest_plan) =
                    plan_single(&config, source.tensor_info("lm_head.weight").is_some())?;
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
    if let Some(transaction) = weight_store {
        if let Err((transaction, error)) = bundle.attach_weight_store(transaction) {
            let LlamaBundle {
                weights,
                scratch,
                kv,
                ..
            } = bundle;
            transaction.rollback(ctx.gpu);
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
    /// Attach an unpublished load transaction after validating the complete
    /// target identity. Publication creates the sole resident-store drain
    /// capability; a rejected transaction is returned unchanged.
    pub fn attach_weight_store(
        &mut self,
        transaction: WeightLoadTransaction,
    ) -> Result<(), (WeightLoadTransaction, String)> {
        if self.weight_store.is_some() {
            return Err((
                transaction,
                "llama: weight store already attached".into(),
            ));
        }
        let attached = match transaction.publish(self.weight_origin) {
            Ok(attached) => attached,
            Err((transaction, error)) => {
                return Err((transaction, format!("llama: weight store origin rejected: {error}")));
            }
        };
        self.weight_store = Some(attached);
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
    use hipfire_runtime::arch_model::ArchModel;
    use hipfire_runtime::hfq::{
        write_hfqm_package_mem, HfqFile, HfqMemTensor,
    };
    use hipfire_runtime::kv_backend::KvBackend;
    use hipfire_runtime::kv_mode::KvMode;
    use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
    use hipfire_runtime::llama::{KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
    use hipfire_runtime::weight_store::{
        WeightLoadTransaction, WeightOrigin, WeightProjection, WeightProjectionKind, WeightStore,
    };
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn hfq_tensor(name: &str, shape: &[u32], quant_type: u8, bytes: usize) -> HfqMemTensor {
        HfqMemTensor {
            name: name.into(),
            quant_type,
            shape: shape.to_vec(),
            group_size: 0,
            data: vec![0; bytes],
        }
    }

    fn f32_hfq_tensor(name: &str, shape: &[u32], malformed: bool) -> HfqMemTensor {
        let elements = shape.iter().map(|&dim| dim as usize).product::<usize>();
        hfq_tensor(name, shape, 2, if malformed { 4 } else { elements * 4 })
    }

    fn fixture_hfq(
        with_awq_sidecar: bool,
        with_q_proj_bias: bool,
        malformed_output_norm: bool,
        separate_lm_head: bool,
    ) -> (PathBuf, HfqFile) {
        let mut tensors = vec![
            f32_hfq_tensor("model.embed_tokens.weight", &[2, 32], false),
            f32_hfq_tensor("model.norm.weight", &[32], false),
            f32_hfq_tensor("model.layers.0.self_attn.q_proj.weight", &[32, 32], false),
            f32_hfq_tensor("model.layers.0.self_attn.k_proj.weight", &[32, 32], false),
            f32_hfq_tensor("model.layers.0.self_attn.v_proj.weight", &[32, 32], false),
            f32_hfq_tensor("model.layers.0.self_attn.o_proj.weight", &[32, 32], false),
            f32_hfq_tensor("model.layers.0.mlp.gate_proj.weight", &[64, 32], false),
            f32_hfq_tensor("model.layers.0.mlp.up_proj.weight", &[64, 32], false),
            f32_hfq_tensor("model.layers.0.mlp.down_proj.weight", &[32, 64], false),
            f32_hfq_tensor(
                "model.layers.0.input_layernorm.weight",
                &[32],
                false,
            ),
            f32_hfq_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                &[32],
                false,
            ),
        ];
        if malformed_output_norm {
            tensors[1] = f32_hfq_tensor("model.norm.weight", &[32], true);
        }
        if with_awq_sidecar {
            tensors.push(hfq_tensor(
                "model.layers.0.self_attn.q_proj.awq_scale.weight",
                &[32],
                1,
                32 * 2,
            ));
        }
        if with_q_proj_bias {
            tensors.push(hfq_tensor(
                "model.layers.0.self_attn.q_proj.bias",
                &[32],
                1,
                32 * 2,
            ));
        }
        if separate_lm_head {
            tensors.push(f32_hfq_tensor("lm_head.weight", &[2, 32], false));
        }
        let metadata = r#"{
            "config": {
                "model_type": "llama",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 64,
                "vocab_size": 2,
                "head_dim": 32,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 8,
                "rope_theta": 10000.0
            }
        }"#;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hipfire-g3-{}-{nonce}.hfq",
            std::process::id()
        ));
        write_hfqm_package_mem(&path, 0, metadata, &tensors).expect("write HFQ fixture");
        let hfq = HfqFile::open(&path).expect("open HFQ fixture");
        (path, hfq)
    }

    fn load_ctx<'a>(
        path: &'a Path,
        gpu: &'a mut rdna_compute::Gpu,
        cask: &'a CaskConfig,
    ) -> LoadCtx<'a> {
        LoadCtx {
            path: path.to_str().expect("fixture path is UTF-8"),
            max_seq: 8,
            deepseek4_compute_placement: Default::default(),
            deepseek4_experts_per_token: None,
            draft_path: None,
            kv_mode_override: Some("q8"),
            kv_backend: KvBackend::Contiguous,
            kv_adaptive_override: None,
            state_quant_override: None,
            cask,
            pp: 1,
            spec: SpecLoadCfg::default(),
            gpu,
            gemma4_drafter_path: None,
            gemma4_draft_len: 3,
        }
    }

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
        let (mesh, plan) = plan_single(&config(), true).unwrap();
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
        let mut transaction = WeightLoadTransaction::new(store);
        let error = match assemble_llama_weights(
            &LlamaConfig {
                n_layers: 0,
                ..config()
            },
            &mut transaction,
        ) {
            Ok(_) => panic!("alias unexpectedly assembled as typed weights"),
            Err(error) => error,
        };
        assert!(error.contains("alias"));
        assert_eq!(transaction.len(), 3);
        assert!(transaction.contains("token_embd", None, 0));
        assert!(transaction.projection("lm_head", None, 0).is_some());
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

    #[test]
    fn missing_lm_head_manifest_declares_a_tied_embedding_alias() {
        let manifest = Llama::weight_manifest_for_hfq(&config(), false);
        let token = &manifest[0];
        let output = manifest.last().expect("manifest has lm_head");
        assert!(matches!(
            output.policy,
            ShardPolicy::Tied { ref source } if source == "token_embd"
        ));
        assert!(token
            .dtype_constraint
            .same_source_set(&output.dtype_constraint));
    }

    #[test]
    fn production_hfq_single_route_aliases_missing_lm_head_without_second_allocation() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(false, false, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load plain HFQ fixture");
        drop(ctx);
        assert!(bundle.weights.lm_head_aliases_embd);
        assert_eq!(
            bundle.weights.output.buf.buf.as_ptr(),
            bundle.weights.token_embd.buf.as_ptr()
        );
        assert!(bundle.weight_store.is_some());
        Box::new(bundle).free_gpu(&mut gpu);
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn production_awq_route_preserves_legacy_loader() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(true, false, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx)
            .expect("AWQ fixture must use the legacy HFQ loader");
        drop(ctx);
        assert!(bundle.weight_store.is_none());
        assert!(bundle.weights.lm_head_aliases_embd);
        Box::new(bundle).free_gpu(&mut gpu);
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn production_biased_hfq_is_rejected_before_manifest_upload() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(false, true, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("biased HFQ unexpectedly loaded"),
            Err(error) => error,
        };
        drop(ctx);
        assert!(error.contains("q_proj.bias"));
        assert!(error.contains("refusing to load Qwen2"));
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn production_post_resident_failure_returns_clean_load_error() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(false, false, true, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("malformed output norm unexpectedly loaded"),
            Err(error) => error,
        };
        drop(ctx);
        assert!(error.contains("source payload") || error.contains("output_norm"));
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn production_manifest_matches_legacy_alias_contract() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(false, false, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load plain HFQ fixture");
        let manifest_alias = bundle.weights.lm_head_aliases_embd;
        drop(ctx);
        Box::new(bundle).free_gpu(&mut gpu);

        let hfq = HfqFile::open(&path).expect("reopen HFQ fixture");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("fixture config");
        let legacy = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
            .expect("legacy loader fixture");
        assert_eq!(manifest_alias, legacy.lm_head_aliases_embd);
        assert_eq!(legacy.embd_format, EmbeddingFormat::F32);
        legacy.free_gpu(&mut gpu);
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn physical_cap_is_honored_by_upstream_kv_constructor() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let dims = KvDims {
            layers: KvLayers::Flat(1),
            n_kv_heads: 1,
            head_dim: 32,
            max_seq: 8,
            physical_cap: Some(4),
        };
        let cache = <KvCache as KvCacheExt>::from_mode(
            KvMode::Q8,
            KvTarget::Single(&mut gpu),
            &dims,
        )
        .expect("upstream Q8 constructor");
        assert_eq!(cache.max_seq, 8);
        assert_eq!(cache.physical_cap, 4);
        let _ = cache.free_gpu(&mut gpu);
    }
}
