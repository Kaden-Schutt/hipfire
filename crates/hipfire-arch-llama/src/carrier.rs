// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::dspark_body::Qwen3DrafterAssets;
use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::dspark_core::DsparkWeights;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{
    EmbeddingFormat, ForwardScratch, KvCache, KvDims, KvLayers, KvTarget, LayerWeights,
    LlamaConfig, LlamaWeights, WeightTensor,
};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::weight_backend::hfq_weight_dtype;
use hipfire_runtime::weight_manifest::{plan_manifest, ManifestPlan, WeightEntry};
use hipfire_runtime::weight_store::{
    TakenWeight, WeightHandle, WeightLoadTransaction, WeightOrigin, WeightStoreAssembly,
    WeightStoreAssemblyGuard, WeightStoreError,
};
use rdna_compute::{DType, GpuTensor};
use std::collections::HashMap;

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
    /// The admitted mesh that owns this plan and the attached store origin.
    pub(crate) mesh: DeviceMesh,
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

/// Crate-private attached owner for the manifest transaction.
///
/// The runtime transaction stays public only long enough for the load carrier
/// to assemble or roll it back. Once wrapped here, the only consuming path is
/// the crate's [`hipfire_runtime::arch_model::ArchModel::free_gpu`] implementation.
pub(crate) struct AttachedWeightStore {
    transaction: WeightLoadTransaction,
}

impl AttachedWeightStore {
    fn from_transaction(
        transaction: WeightLoadTransaction,
        expected: WeightOrigin,
    ) -> Result<Self, (WeightLoadTransaction, WeightStoreError)> {
        if let Err(error) = transaction.validate_origin_value(expected) {
            return Err((transaction, error));
        }
        Ok(Self { transaction })
    }

    pub(crate) fn drain(self, gpu: &mut rdna_compute::Gpu) -> hip_bridge::HipResult<()> {
        self.transaction.rollback(gpu)
    }
}

fn with_weight_rollback_error(reason: String, rollback: hip_bridge::HipResult<()>) -> String {
    match rollback {
        Ok(()) => reason,
        Err(error) => format!("{reason}; resident rollback failed: {error}"),
    }
}

fn plan_single(
    config: &LlamaConfig,
    has_separate_lm_head: bool,
) -> Result<(DeviceMesh, ManifestPlan), String> {
    let mesh = DeviceMesh::single().map_err(|error| format!("llama: device mesh: {error}"))?;
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
const HFQ_LM_HEAD_NAMES: &[&str] = &[
    "lm_head.weight",
    "model.lm_head.weight",
    "model.language_model.lm_head.weight",
];

fn hfq_has_separate_lm_head(hfq: &HfqFile) -> bool {
    HFQ_LM_HEAD_NAMES
        .iter()
        .any(|name| hfq.find_tensor_info(name).is_some())
}

fn hfq_entry_names(entry: &WeightEntry) -> Result<Vec<String>, String> {
    let names = match (entry.name.as_str(), entry.layer) {
        ("token_embd", None) => vec!["model.embed_tokens.weight".to_string()],
        ("output_norm", None) => vec!["model.norm.weight".to_string()],
        ("lm_head", None) => HFQ_LM_HEAD_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
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
                    &f32::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]) as u32 * (1 << 16))
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
    if matches!(
        entry.name.as_str(),
        "output_norm" | "attn_norm" | "ffn_norm" | "q_norm" | "k_norm"
    ) {
        return Ok((f32_bytes_from_hfq(quant_type, &data, &name)?, DType::F32));
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
    let mut take =
        |name: &str, layer: Option<usize>| take_slot(&mut assembly, &mut slots, name, layer);

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
        resident_weight(&mut cells, "lm_head", None, config.vocab_size, config.dim)
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
    let (config, weights, kv, scratch, manifest_plan, weight_store, mesh, weight_origin) = match src
    {
        ModelSource::Hfq(hfq) => {
            let config =
                <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
            // Admission and route classification are pure source checks.
            // They must run before any manifest fulfillment or GPU upload.
            hipfire_runtime::hfq::validate_llama_hfq_admission(&hfq).map_err(|e| e.to_string())?;
            let has_separate_lm_head = hfq_has_separate_lm_head(&hfq);
            let route = classify_hfq_route(&hfq);
            eprintln!("llama: HFQ source route = {route:?}");
            let (mesh, manifest_plan) = plan_single(&config, has_separate_lm_head)?;
            let weight_origin = WeightOrigin::for_single(&mesh, ctx.gpu);
            let (weights, mut weight_store) = match route {
                HfqLoadRoute::LegacyAwq => {
                    let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, ctx.gpu)
                        .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))?;
                    (weights, None)
                }
                HfqLoadRoute::ManifestPlainLlama => {
                    let manifest = Llama::weight_manifest_for_hfq(&config, has_separate_lm_head);
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
                            return Err(with_weight_rollback_error(
                                error,
                                transaction.rollback(ctx.gpu),
                            ));
                        }
                    };
                    (weights, Some(transaction))
                }
            };
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            // The plain LLaMA path has no independent cap resolver. PR
            // #661's physical-cap behavior is owned by the existing
            // upstream KV plan.
            let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                Ok(scratch) => scratch,
                Err(error) => {
                    let rollback = if let Some(transaction) = weight_store.take() {
                        transaction.rollback(ctx.gpu)
                    } else {
                        Ok(())
                    };
                    weights.free_gpu(ctx.gpu);
                    return Err(with_weight_rollback_error(
                        format!("llama: ForwardScratch::new_with_max_seq failed: {error:?}"),
                        rollback,
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
                    let rollback = if let Some(transaction) = weight_store.take() {
                        transaction.rollback(ctx.gpu)
                    } else {
                        Ok(())
                    };
                    weights.free_gpu(ctx.gpu);
                    return Err(with_weight_rollback_error(
                        format!("llama: <KvCache as KvCacheExt>::from_mode failed: {error}"),
                        rollback,
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
            let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
                .map_err(|e| format!("failed to parse LLaMA/Qwen3 config from config.json: {e}"))?;
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
            let rollback = transaction.rollback(ctx.gpu);
            scratch.free_gpu(ctx.gpu);
            weights.free_gpu(ctx.gpu);
            let _ = kv.free_gpu(ctx.gpu);
            return Err(with_weight_rollback_error(error, rollback));
        }
    }
    Ok(bundle)
}

/// Alias matching the `load_<arch>_bundle` naming convention in the task.
pub use load_bundle as load_llama_bundle;

impl LlamaBundle {
    /// Attach an unpublished load transaction after validating the complete
    /// target identity. The resulting owner is crate-private and can only be
    /// consumed by `ArchModel::free_gpu`.
    fn attach_weight_store(
        &mut self,
        transaction: WeightLoadTransaction,
    ) -> Result<(), (WeightLoadTransaction, String)> {
        if self.weight_store.is_some() {
            return Err((transaction, "llama: weight store already attached".into()));
        }
        let attached = match AttachedWeightStore::from_transaction(transaction, self.weight_origin)
        {
            Ok(attached) => attached,
            Err((transaction, error)) => {
                return Err((
                    transaction,
                    format!("llama: weight store origin rejected: {error}"),
                ));
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
    use hipfire_runtime::arch_model::ArchModel;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
    use hipfire_runtime::kv_backend::KvBackend;
    use hipfire_runtime::kv_mode::KvMode;
    use hipfire_runtime::llama::ModelArch;
    use hipfire_runtime::llama::{
        forward_scratch_compute, forward_scratch_embed, KvCache, KvCacheExt, KvDims, KvLayers,
        KvTarget,
    };
    use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use hipfire_runtime::weight_store::test_support;
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
        let data = if malformed {
            vec![0; 4]
        } else {
            (0..elements)
                .flat_map(|value| ((value as f32) + 1.0).to_le_bytes())
                .collect()
        };
        HfqMemTensor {
            name: name.into(),
            quant_type: 2,
            shape: shape.to_vec(),
            group_size: 0,
            data,
        }
    }
    fn f16_hfq_tensor(name: &str, shape: &[u32]) -> HfqMemTensor {
        let elements = shape.iter().map(|&dim| dim as usize).product::<usize>();
        HfqMemTensor {
            name: name.into(),
            quant_type: 1,
            shape: shape.to_vec(),
            group_size: 0,
            data: (0..elements)
                .flat_map(|index| {
                    let bits = if index % 2 == 0 { 0x3c00u16 } else { 0x3800u16 };
                    bits.to_le_bytes()
                })
                .collect(),
        }
    }

    fn fixture_hfq(
        with_awq_sidecar: bool,
        with_q_proj_bias: bool,
        malformed_output_norm: bool,
        separate_lm_head: bool,
    ) -> (PathBuf, HfqFile) {
        fixture_hfq_with_lm_head(
            with_awq_sidecar,
            with_q_proj_bias,
            malformed_output_norm,
            separate_lm_head.then_some("lm_head.weight"),
        )
    }

    fn fixture_hfq_with_lm_head(
        with_awq_sidecar: bool,
        with_q_proj_bias: bool,
        malformed_output_norm: bool,
        lm_head_name: Option<&str>,
    ) -> (PathBuf, HfqFile) {
        let mut tensors = vec![
            f32_hfq_tensor("model.embed_tokens.weight", &[2, 32], false),
            f32_hfq_tensor("model.norm.weight", &[32], false),
            f16_hfq_tensor("model.layers.0.self_attn.q_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.self_attn.k_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.self_attn.v_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.self_attn.o_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.mlp.gate_proj.weight", &[64, 32]),
            f16_hfq_tensor("model.layers.0.mlp.up_proj.weight", &[64, 32]),
            f16_hfq_tensor("model.layers.0.mlp.down_proj.weight", &[32, 64]),
            f32_hfq_tensor("model.layers.0.input_layernorm.weight", &[32], false),
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
        if let Some(lm_head_name) = lm_head_name {
            tensors.push(f32_hfq_tensor(lm_head_name, &[2, 32], false));
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
        let path =
            std::env::temp_dir().join(format!("hipfire-g3-{}-{nonce}.hfq", std::process::id()));
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
        assert!(plan
            .collective_schedule
            .iter()
            .any(|entry| entry.name == "wo"));
        assert!(manifest[0].dtype_constraint.accepts(DType::HFQ4G256));
        assert!(manifest[1].dtype_constraint.accepts(DType::MQ4G256));
        assert!(manifest[9].dtype_constraint.accepts(DType::F32));
        assert!(!manifest[9].dtype_constraint.accepts(DType::F16));
    }

    #[test]
    fn typed_assembly_rolls_back_when_a_cell_is_not_resident() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
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
    fn production_awq_sidecar_selects_legacy_loader() {
        let (path, hfq) = fixture_hfq(true, false, false, false);
        assert_eq!(classify_hfq_route(&hfq), HfqLoadRoute::LegacyAwq);
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn alternate_explicit_lm_head_names_are_not_tied() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        for name in &HFQ_LM_HEAD_NAMES[1..] {
            let (path, hfq) = fixture_hfq_with_lm_head(false, false, false, Some(name));
            assert!(hfq_has_separate_lm_head(&hfq));
            let cask = CaskConfig::default();
            let mut ctx = load_ctx(&path, &mut gpu, &cask);
            let bundle =
                load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load explicit lm_head");
            drop(ctx);
            assert!(!bundle.weights.lm_head_aliases_embd);
            Box::new(bundle).free_gpu(&mut gpu);
            std::fs::remove_file(path).expect("remove HFQ fixture");
        }
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
    fn production_post_resident_failure_reclaims_every_uploaded_allocation() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(false, false, false, false);
        test_support::reset();
        test_support::arm_fail_after_upload(1);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("post-upload fault unexpectedly succeeded"),
            Err(error) => error,
        };
        drop(ctx);
        test_support::clear_faults();
        assert!(error.contains("test fault injected after resident upload"));
        let allocations = test_support::resident_allocations();
        assert!(allocations > 0, "fault must follow a resident upload");
        assert_eq!(
            allocations,
            test_support::resident_releases(),
            "every resident allocation must be reclaimed on load failure"
        );
        std::fs::remove_file(path).expect("remove HFQ fixture");
    }

    #[test]
    fn production_manifest_matches_legacy_forward_logits() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (path, hfq) = fixture_hfq(false, false, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(&path, &mut gpu, &cask);
        let mut bundle =
            load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load plain HFQ fixture");
        drop(ctx);

        let manifest_logits = {
            forward_scratch_embed(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                1,
                0,
                &bundle.scratch,
            )
            .expect("manifest embedding forward");
            forward_scratch_compute(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                0,
                &mut bundle.kv,
                &bundle.scratch,
            )
            .expect("manifest model forward");
            gpu.download_f32(&bundle.scratch.logits)
                .expect("download manifest logits")
        };
        Box::new(bundle).free_gpu(&mut gpu);

        let hfq = HfqFile::open(&path).expect("reopen HFQ fixture");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("fixture config");
        let legacy = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
            .expect("load legacy HFQ fixture");
        let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, 8)
            .expect("allocate legacy forward scratch");
        let dims = llama_kv_dims(&config, 8, None);
        let mut kv =
            <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                .expect("allocate legacy KV cache");
        forward_scratch_embed(&mut gpu, &legacy, &config, 1, 0, &scratch)
            .expect("legacy embedding forward");
        forward_scratch_compute(&mut gpu, &legacy, &config, 0, &mut kv, &scratch)
            .expect("legacy model forward");
        let legacy_logits = gpu
            .download_f32(&scratch.logits)
            .expect("download legacy logits");
        scratch.free_gpu(&mut gpu);
        let _ = kv.free_gpu(&mut gpu);
        legacy.free_gpu(&mut gpu);

        assert_eq!(manifest_logits.len(), legacy_logits.len());
        for (index, (manifest, legacy)) in manifest_logits.iter().zip(&legacy_logits).enumerate() {
            assert!(
                (manifest - legacy).abs() <= 1e-5,
                "logit mismatch at index {index}: manifest={manifest} legacy={legacy}"
            );
        }
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
        let cache =
            <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                .expect("upstream Q8 constructor");
        assert_eq!(cache.max_seq, 8);
        assert_eq!(cache.physical_cap, 4);
        let _ = cache.free_gpu(&mut gpu);
    }

    /// #666 G3 pinned-fixture parity oracle.
    ///
    /// Runs on the tracker fixture `qwen3:0.6b` (plain LLaMA-family HFQ,
    /// canonical local file `~/.hipfire/models/qwen3-0.6b-llama.mq4`). The
    /// path is taken from `HIPFIRE_G3_FIXTURE` when set, else the canonical
    /// `~/.hipfire/models` location. Skips silently when the file or a GPU is
    /// absent (no-GPU / no-fixture batteries stay green); fails loudly on a
    /// size or route-class mismatch so a substituted artifact cannot pass as
    /// the pinned fixture.
    ///
    /// Two routes are loaded from equivalent cloned state:
    ///   * production manifest route — `load_bundle` classifies this plain
    ///     file as `ManifestPlainLlama` and publishes through manifest
    ///     planning, transactional fulfillment, and typed assembly;
    ///   * validation-only reference — the legacy loader entry the manifest
    ///     route replaces (`hfq::load_weights_hfq` plus the same scratch and
    ///     KV constructors).
    ///
    /// Both then decode the same committed prompt greedily (argmax), and at
    /// every committed position the oracle records and asserts: token IDs,
    /// logits (max absolute difference), KV geometry / byte extents, the
    /// position counter, alias identity, and route identity. The evidence
    /// block is printed for the evidence run.
    #[test]
    fn pinned_fixture_manifest_legacy_parity_oracle() {
        let fixture = std::env::var("HIPFIRE_G3_FIXTURE").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.hipfire/models/qwen3-0.6b-llama.mq4")
        });
        // Fixture lock: the tracker artifact identity. A substituted file of
        // the wrong size cannot pass; route classification below additionally
        // refuses non-plain / mis-tagged artifacts.
        const PINNED_SIZE: u64 = 495_181_824;
        const PINNED_MD5: &str = "2579e10ba3a988818386f2b07632ee01";
        let Ok(meta) = std::fs::metadata(&fixture) else {
            eprintln!("g3-oracle: fixture absent ({fixture}); skipping");
            return;
        };
        assert_eq!(
            meta.len(),
            PINNED_SIZE,
            "g3-oracle: fixture size mismatch — not the pinned qwen3:0.6b artifact (md5 {PINNED_MD5})"
        );
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            eprintln!("g3-oracle: no GPU; skipping");
            return;
        };
        let prompt = "The capital of France is located in";
        let max_seq = 64usize;
        eprintln!(
            "g3-oracle: fixture={fixture} size={} md5={PINNED_MD5}",
            meta.len()
        );
        eprintln!("g3-oracle: prompt={prompt:?} (prompt md5 recorded by the evidence run)");

        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open pinned fixture");
        assert_eq!(
            classify_hfq_route(&hfq),
            HfqLoadRoute::ManifestPlainLlama,
            "pinned fixture must take the production manifest route"
        );
        assert!(
            !hfq.has_awq_sidecars(),
            "pinned fixture must be a plain (sidecar-free) LLaMA-family HFQ"
        );
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("pinned fixture tokenizer");
        let prompt_tokens = tokenizer.encode(prompt);
        eprintln!(
            "g3-oracle: prompt tokens = {prompt_tokens:?} ({} incl. BOS)",
            prompt_tokens.len()
        );
        let has_separate_lm_head = hfq_has_separate_lm_head(&hfq);
        eprintln!("g3-oracle: separate lm_head in fixture = {has_separate_lm_head}");

        let cask = CaskConfig::default();
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("production load");
        assert_eq!(
            ctx.kv_mode_override,
            Some("q8"),
            "oracle runs both routes in the same Q8 KV mode"
        );
        assert!(bundle.weight_store.is_some(), "production store attached");
        drop(ctx);

        // Validation-only reference: the legacy loader entry, same scratch/KV.
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen pinned fixture");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("reference config");
        let legacy = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
            .expect("legacy reference load");
        let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, max_seq)
            .expect("reference forward scratch");
        let dims = llama_kv_dims(&config, max_seq, None);
        let mut legacy_kv =
            <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                .expect("reference KV cache");

        eprintln!(
            "g3-oracle: route identity — production=ManifestPlainLlama, reference=legacy load_weights_hfq"
        );

        // Alias identity parity.
        assert_eq!(
            bundle.weights.lm_head_aliases_embd, legacy.lm_head_aliases_embd,
            "alias identity must match between routes"
        );
        eprintln!(
            "g3-oracle: lm_head aliases embed_tokens = {} (both routes)",
            legacy.lm_head_aliases_embd
        );

        // KV geometry parity (mode flags, dims, byte extents per layer).
        {
            let pkv = &bundle.kv;
            eprintln!(
                "g3-oracle: kv geometry — prod q8={} qint8={} kv_dim={} max_seq={} cap={} n_heads={} head_dim={}",
                pkv.quant_q8, pkv.quant_int8, pkv.kv_dim, pkv.max_seq, pkv.physical_cap,
                pkv.n_kv_heads, pkv.head_dim
            );
            assert_eq!(pkv.quant_q8, legacy_kv.quant_q8);
            assert_eq!(pkv.quant_int8, legacy_kv.quant_int8);
            assert_eq!(pkv.kv_dim, legacy_kv.kv_dim);
            assert_eq!(pkv.max_seq, legacy_kv.max_seq);
            assert_eq!(pkv.physical_cap, legacy_kv.physical_cap);
            assert_eq!(pkv.n_kv_heads, legacy_kv.n_kv_heads);
            assert_eq!(pkv.head_dim, legacy_kv.head_dim);
            assert_eq!(pkv.k_gpu.len(), legacy_kv.k_gpu.len());
            for layer in 0..pkv.k_gpu.len() {
                assert_eq!(
                    pkv.k_gpu[layer].byte_size(),
                    legacy_kv.k_gpu[layer].byte_size(),
                    "layer {layer} K byte extent parity"
                );
                assert_eq!(
                    pkv.v_gpu[layer].byte_size(),
                    legacy_kv.v_gpu[layer].byte_size(),
                    "layer {layer} V byte extent parity"
                );
            }
        }

        // Greedy decode in lockstep; compare at every committed position.
        let mut next_token: u32 = 0;
        let mut worst_logit_diff: f32 = 0.0;
        let mut tokens: Vec<u32> = Vec::new();
        let generated = 12usize;
        let total = prompt_tokens.len() + generated;
        assert!(total <= max_seq, "position budget vs KV max_seq");
        for pos in 0..total {
            let token = if pos < prompt_tokens.len() {
                prompt_tokens[pos]
            } else {
                next_token
            };
            forward_scratch_embed(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                token,
                pos,
                &bundle.scratch,
            )
            .expect("production embed");
            forward_scratch_compute(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                0,
                &mut bundle.kv,
                &bundle.scratch,
            )
            .expect("production compute");
            let prod_logits = gpu
                .download_f32(&bundle.scratch.logits)
                .expect("production logits");
            forward_scratch_embed(&mut gpu, &legacy, &config, token, pos, &scratch)
                .expect("reference embed");
            forward_scratch_compute(&mut gpu, &legacy, &config, 0, &mut legacy_kv, &scratch)
                .expect("reference compute");
            let legacy_logits = gpu.download_f32(&scratch.logits).expect("reference logits");
            assert_eq!(
                prod_logits.len(),
                legacy_logits.len(),
                "logit width parity at position {pos}"
            );
            let mut diff: f32 = 0.0;
            for (p, l) in prod_logits.iter().zip(&legacy_logits) {
                diff = diff.max((p - l).abs());
            }
            worst_logit_diff = worst_logit_diff.max(diff);
            assert!(
                diff <= 1e-5,
                "logit mismatch at committed position {pos}: max abs diff {diff}"
            );
            let prod_choice = argmax_index(&prod_logits) as u32;
            let legacy_choice = argmax_index(&legacy_logits) as u32;
            assert_eq!(
                prod_choice, legacy_choice,
                "token-id mismatch at committed position {pos}"
            );
            next_token = prod_choice;
            tokens.push(token);
            eprintln!(
                "g3-oracle: pos {pos:>2} token {token:>6} max-logit-diff {diff:.3e} (choice {prod_choice})"
            );
        }

        // End-state KV payload parity on layer 0 (full written extent).
        let mut prod_k = vec![0u8; bundle.kv.k_gpu[0].byte_size()];
        let mut ref_k = vec![0u8; legacy_kv.k_gpu[0].byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut prod_k, &bundle.kv.k_gpu[0].buf)
            .expect("download prod K");
        gpu.hip
            .memcpy_dtoh(&mut ref_k, &legacy_kv.k_gpu[0].buf)
            .expect("download ref K");
        let k_diffs = prod_k.iter().zip(&ref_k).filter(|(a, b)| a != b).count();
        eprintln!(
            "g3-oracle: layer-0 K payload — {} bytes compared, {k_diffs} byte diffs",
            prod_k.len()
        );
        assert_eq!(prod_k, ref_k, "layer-0 K payload must be byte-identical");
        let mut prod_v = vec![0u8; bundle.kv.v_gpu[0].byte_size()];
        let mut ref_v = vec![0u8; legacy_kv.v_gpu[0].byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut prod_v, &bundle.kv.v_gpu[0].buf)
            .expect("download prod V");
        gpu.hip
            .memcpy_dtoh(&mut ref_v, &legacy_kv.v_gpu[0].buf)
            .expect("download ref V");
        let v_diffs = prod_v.iter().zip(&ref_v).filter(|(a, b)| a != b).count();
        eprintln!(
            "g3-oracle: layer-0 V payload — {} bytes compared, {v_diffs} byte diffs",
            prod_v.len()
        );
        assert_eq!(prod_v, ref_v, "layer-0 V payload must be byte-identical");

        eprintln!(
            "g3-oracle: PASS — {total} committed positions, worst logit diff {worst_logit_diff:.3e}, tokens {tokens:?}"
        );
        Box::new(bundle).free_gpu(&mut gpu);
        scratch.free_gpu(&mut gpu);
        let _ = legacy_kv.free_gpu(&mut gpu);
        legacy.free_gpu(&mut gpu);
    }

    /// #666 G3 pinned-fixture lifecycle evidence.
    ///
    /// On the same tracker fixture as the parity oracle: production load
    /// through the manifest route, decode, existing-reset smoke (decode
    /// again after `reset_session_state` with identical output), unload via
    /// the sole consuming owner (`ArchModel::free_gpu`), immediate reload
    /// with identical decode, a deterministic post-upload fault whose
    /// rollback returns every resident store allocation (store accounting:
    /// allocations == releases on the failed path), and an immediate retry
    /// that decodes identically. Skips cleanly when the fixture or a GPU is
    /// absent.
    #[test]
    fn pinned_fixture_lifecycle_fault_retry_reload() {
        let Some(fixture) = pinned_fixture_path() else {
            return;
        };
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            eprintln!("g3-lifecycle: no GPU; skipping");
            return;
        };
        let prompt = "The capital of France is located in";
        let max_seq = 64usize;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open pinned fixture");
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("pinned fixture tokenizer");
        let prompt_tokens = tokenizer.encode(prompt);
        let cask = CaskConfig::default();

        // First production load + decode (warms store resident accounting).
        test_support::reset();
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for first load");
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("production load");
        drop(ctx);
        let allocations = test_support::resident_allocations();
        assert!(
            allocations > 0,
            "production load must publish resident store allocations"
        );
        eprintln!("g3-lifecycle: warm-baseline resident allocations = {allocations}");
        let baseline = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        eprintln!("g3-lifecycle: first decode = {baseline:?}");

        // Existing-reset smoke: reset_session_state leaves the model reusable
        // and the next decode is byte-identical.
        hipfire_runtime::arch_model::ArchModel::reset_session_state(&mut bundle, &mut gpu)
            .expect("existing reset smoke");
        let after_reset = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        assert_eq!(after_reset, baseline, "reset must not change decode output");

        // Unload through the sole consuming owner (ArchModel::free_gpu drains
        // the attached store and frees weights/scratch/KV).
        Box::new(bundle).free_gpu(&mut gpu);
        eprintln!("g3-lifecycle: unloaded via ArchModel::free_gpu");

        // Immediate reload decodes identically.
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for reload");
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("immediate reload");
        drop(ctx);
        let after_reload = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        assert_eq!(after_reload, baseline, "immediate reload decode parity");

        // Deterministic post-upload fault on the next load: it must fail and
        // roll back every resident allocation (no legacy fallback). Store
        // accounting is reset so the failed path alone is measured:
        // allocations == releases after the rollback.
        test_support::reset();
        test_support::arm_fail_after_upload(1);
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for fault");
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("post-upload fault unexpectedly succeeded"),
            Err(error) => error,
        };
        drop(ctx);
        test_support::clear_faults();
        assert!(
            error.contains("test fault injected after resident upload"),
            "unexpected error: {error}"
        );
        assert_eq!(
            test_support::resident_allocations(),
            test_support::resident_releases(),
            "fault rollback must return every resident allocation (zero-free)"
        );
        eprintln!("g3-lifecycle: deterministic fault rolled back — error {error:?}");

        // Immediate retry after the fault decodes identically.
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for retry");
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("immediate retry");
        drop(ctx);
        let after_retry = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        assert_eq!(after_retry, baseline, "immediate retry decode parity");
        Box::new(bundle).free_gpu(&mut gpu);
        eprintln!(
            "g3-lifecycle: PASS — load/reset/unload/reload/fault/retry all decode identically"
        );
    }

    fn pinned_fixture_path() -> Option<String> {
        let fixture = std::env::var("HIPFIRE_G3_FIXTURE").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.hipfire/models/qwen3-0.6b-llama.mq4")
        });
        const PINNED_SIZE: u64 = 495_181_824;
        const PINNED_MD5: &str = "2579e10ba3a988818386f2b07632ee01";
        let Ok(meta) = std::fs::metadata(&fixture) else {
            eprintln!("g3-lifecycle: fixture absent ({fixture}); skipping");
            return None;
        };
        assert_eq!(
            meta.len(),
            PINNED_SIZE,
            "fixture size mismatch — not the pinned qwen3:0.6b artifact (md5 {PINNED_MD5})"
        );
        Some(fixture)
    }

    fn hfq_clone(path: &str) -> HfqFile {
        HfqFile::open(std::path::Path::new(path)).expect("reopen pinned fixture")
    }

    /// Greedy argmax decode over `prompt_tokens` followed by `generated`
    /// self-generated tokens, one token per committed position.
    fn greedy_decode(
        gpu: &mut rdna_compute::Gpu,
        weights: &LlamaWeights,
        config: &LlamaConfig,
        kv: &mut KvCache,
        scratch: &ForwardScratch,
        prompt_tokens: &[u32],
        generated: usize,
    ) -> Vec<u32> {
        let mut next_token: u32 = 0;
        let mut out = Vec::new();
        let total = prompt_tokens.len() + generated;
        assert!(total <= kv.max_seq, "position budget vs KV max_seq");
        for pos in 0..total {
            let token = if pos < prompt_tokens.len() {
                prompt_tokens[pos]
            } else {
                next_token
            };
            forward_scratch_embed(gpu, weights, config, token, pos, scratch).expect("embed");
            forward_scratch_compute(gpu, weights, config, 0, kv, scratch).expect("compute");
            let logits = gpu.download_f32(&scratch.logits).expect("logits");
            next_token = argmax_index(&logits) as u32;
            out.push(token);
        }
        out
    }

    fn argmax_index(logits: &[f32]) -> usize {
        let mut best = 0usize;
        for (index, value) in logits.iter().enumerate() {
            if value > &logits[best] {
                best = index;
            }
        }
        best
    }
}
