// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5's Phase-3 store bridge.
//!
//! This module deliberately does not change any of the existing HFQ or
//! safetensors loader entry points.  It supplies the arch-owned seam needed by
//! the device-mesh loader: resolve a logical manifest entry to an HFQ tensor,
//! and assemble a completely validated `WeightStore` into the legacy typed
//! Qwen3.5 weight shape.

use crate::arch::Qwen35;
use crate::qwen35::{
    DeltaNetLayerWeights, DeltaNetMoeLayerWeights, ExpertWeights, FullAttnLayerWeights,
    FullAttnMoeLayerWeights, LayerType, LayerWeights, MoeFfnWeights, Qwen35Config, Qwen35Weights,
    SharedExpertWeights,
};
use hipfire_hardware::DeviceMesh;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{EmbeddingFormat, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::{paro_text_prefix, repack_awq_to_hfq4g128};
use hipfire_runtime::weight_backend::{dequant_f32, dequant_norm};
use hipfire_runtime::weight_manifest::placement_devices;
use hipfire_runtime::weight_manifest::{DTypeConstraint, ShardPolicy, SourceDType, WeightEntry};
use hipfire_runtime::weight_store::{TakenWeight, WeightHandle, WeightStore, WeightStoreTarget};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::RefCell;
use std::collections::HashMap;

const AWQ_SUFFIX: &str = ".awq_scale";
const PARO_SUFFIXES: [&str; 3] = [".paro_pairs", ".paro_theta", ".paro_channel_scales"];

fn is_paro_record(name: &str) -> bool {
    PARO_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
}

/// How the resolved bytes are laid out in the HFQ source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35SourceLayout {
    /// Bytes are a forward-ready quantized blob and must not be decoded.
    Raw,
    /// IEEE half precision source bytes.
    F16,
    /// IEEE single precision source bytes.
    F32,
    /// Brain floating point source bytes.
    BF16,
}

/// A logical manifest entry resolved to its actual HFQ source record.
/// `dtype` is the source dtype returned to `fulfill_manifest`; it is never a
/// guessed logical dtype.  The physical name is retained for diagnostics and
/// for companion lookup tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedQwen35Source {
    pub logical_name: String,
    pub physical_name: String,
    pub bytes: Vec<u8>,
    pub dtype: DType,
    pub layout: Qwen35SourceLayout,
    pub shape: Vec<usize>,
    pub companion: bool,
}

/// Qwen3.5's HFQ logical-name resolver.  It owns no bytes and does not upload
/// anything; callers can pass `resolve(...).bytes/dtype` directly to the
/// runtime `fulfill_manifest` closure.
pub struct Qwen35SourceResolver<'a> {
    hfq: &'a HfqFile,
    config: &'a Qwen35Config,
}

/// Resolver for the ParoQuant safetensors source.  It performs only the
/// source-format operation required before fulfillment (AWQ qweight/qzeros /
/// scales → HFQ4-G128 bytes); GPU upload and typed assembly remain shared with
/// the HFQ path.
pub struct Qwen35ParoSourceResolver<'a> {
    source: &'a dyn ModelSource,
    config: &'a Qwen35Config,
    prefix: &'static str,
    /// Logical Paro records discovered during metadata preflight.  Sidecars
    /// use these physical names directly; they must not resolve/repack their
    /// owner again during payload fulfillment.
    source_records: RefCell<HashMap<(String, Option<usize>), String>>,
}

impl<'a> Qwen35ParoSourceResolver<'a> {
    pub fn new(source: &'a dyn ModelSource, config: &'a Qwen35Config) -> Result<Self, String> {
        let prefix = paro_text_prefix(source).map_err(|e| format!("{e}"))?;
        Ok(Self {
            source,
            config,
            prefix,
            source_records: RefCell::new(HashMap::new()),
        })
    }

    /// Resolve names, source dtype, and shapes without touching tensor payloads.
    /// This is the mandatory preflight path used to discover Paro sidecars.
    pub fn resolve_metadata(&self, entry: &WeightEntry) -> Result<ResolvedQwen35Source, String> {
        let logical = entry.name.as_str();
        for (suffix, physical_suffix) in [
            (".paro_pairs", "pairs"),
            (".paro_theta", "theta"),
            (".paro_channel_scales", "channel_scales"),
        ] {
            if let Some(owner) = logical.strip_suffix(suffix) {
                let base = self.paro_quant_base(owner, entry.layer)?;
                let base = base.trim_end_matches(".qweight");
                let physical = format!("{base}.{physical_suffix}");
                let info = self
                    .source
                    .tensor_info(&physical)
                    .ok_or_else(|| format!("Paro source missing {physical}"))?;
                return Ok(ResolvedQwen35Source {
                    logical_name: entry.name.clone(),
                    physical_name: physical,
                    bytes: Vec::new(),
                    dtype: DType::Raw,
                    layout: Qwen35SourceLayout::Raw,
                    shape: info.shape.clone(),
                    companion: true,
                });
            }
        }
        let mut candidates = physical_candidates(self.config, logical, entry.layer);
        if logical == "token_embd" {
            candidates = vec![format!("{}.embed_tokens.weight", self.prefix)];
        }
        let base = candidates
            .into_iter()
            .find(|name| {
                self.source.tensor_info(name).is_some()
                    || self
                        .source
                        .tensor_info(&format!("{}.qweight", name.trim_end_matches(".weight")))
                        .is_some()
            })
            .ok_or_else(|| format!("qwen35 Paro source: no tensor for '{logical}'"))?;
        let quant_base = base.strip_suffix(".weight").unwrap_or(&base);
        if self
            .source
            .tensor_info(&format!("{quant_base}.qweight"))
            .is_some()
        {
            for suffix in ["qzeros", "scales"] {
                if self
                    .source
                    .tensor_info(&format!("{quant_base}.{suffix}"))
                    .is_none()
                {
                    return Err(format!("Paro source missing {quant_base}.{suffix}"));
                }
            }
            return Ok(ResolvedQwen35Source {
                logical_name: entry.name.clone(),
                physical_name: format!("{quant_base}.qweight"),
                bytes: Vec::new(),
                dtype: DType::ParoQ4G128,
                layout: Qwen35SourceLayout::Raw,
                shape: entry.logical_shape.clone(),
                companion: false,
            });
        }
        let info = self.source.tensor_info(&base).unwrap();
        let dtype = match info.dtype.as_str() {
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            other => {
                return Err(format!(
                    "Paro source tensor '{base}' has unsupported dtype {other}"
                ))
            }
        };
        let layout = match dtype {
            DType::F16 => Qwen35SourceLayout::F16,
            DType::BF16 => Qwen35SourceLayout::BF16,
            _ => Qwen35SourceLayout::F32,
        };
        if info.shape != entry.logical_shape {
            return Err(format!(
                "Paro source '{base}' shape {:?}, expected {:?}",
                info.shape, entry.logical_shape
            ));
        }
        Ok(ResolvedQwen35Source {
            logical_name: entry.name.clone(),
            physical_name: base,
            bytes: Vec::new(),
            dtype,
            layout,
            shape: info.shape.clone(),
            companion: false,
        })
    }

    pub fn resolve(&self, entry: &WeightEntry) -> Result<ResolvedQwen35Source, String> {
        let logical = entry.name.as_str();
        for (suffix, physical_suffix) in [
            (".paro_pairs", "pairs"),
            (".paro_theta", "theta"),
            (".paro_channel_scales", "channel_scales"),
        ] {
            if let Some(owner) = logical.strip_suffix(suffix) {
                let physical = self
                    .source_records
                    .borrow()
                    .get(&(entry.name.clone(), entry.layer))
                    .cloned()
                    .or_else(|| {
                        self.paro_quant_base(owner, entry.layer).ok().map(|base| {
                            format!("{}.{}", base.trim_end_matches(".qweight"), physical_suffix)
                        })
                    })
                    .ok_or_else(|| {
                        format!(
                            "Paro source record missing owner '{owner}[{:#?}]' for '{logical}'",
                            entry.layer
                        )
                    })?;
                let (info, data) = self
                    .source
                    .tensor_data(&physical)
                    .ok_or_else(|| format!("Paro source missing {physical}"))?;
                return Ok(ResolvedQwen35Source {
                    logical_name: entry.name.clone(),
                    physical_name: physical,
                    bytes: data.to_vec(),
                    dtype: DType::Raw,
                    layout: Qwen35SourceLayout::Raw,
                    shape: info.shape.clone(),
                    companion: true,
                });
            }
        }
        let candidates = physical_candidates(self.config, logical, entry.layer);
        let mut candidates = candidates;
        // Paro's raw/norm readers use the text prefix, while the logical
        // candidates already include it for the normal Qwen wrapper layout.
        if logical == "token_embd" {
            candidates = vec![format!(
                "{self_prefix}.embed_tokens.weight",
                self_prefix = self.prefix
            )];
        }

        let base = candidates
            .into_iter()
            .find(|name| {
                self.source.tensor_info(name).is_some()
                    || self
                        .source
                        .tensor_info(&format!(
                            "{}",
                            name.trim_end_matches(".weight").to_owned() + ".qweight"
                        ))
                        .is_some()
            })
            .ok_or_else(|| format!("qwen35 Paro source: no tensor for '{logical}'"))?;
        let quant_base = base.strip_suffix(".weight").unwrap_or(&base);
        let info = self.source.tensor_info(&format!("{quant_base}.qweight"));
        let (bytes, dtype, layout, physical_name, shape) = if info.is_some() {
            let qweight = self
                .source
                .tensor_data(&format!("{quant_base}.qweight"))
                .unwrap()
                .1;
            let qzeros = self
                .source
                .tensor_data(&format!("{quant_base}.qzeros"))
                .ok_or_else(|| format!("Paro source missing {quant_base}.qzeros"))?
                .1;
            let scales = self
                .source
                .tensor_data(&format!("{quant_base}.scales"))
                .ok_or_else(|| format!("Paro source missing {quant_base}.scales"))?
                .1;
            let group_size = self
                .source
                .quant_config()
                .map(|q| q.group_size as usize)
                .unwrap_or(128);
            (
                repack_awq_to_hfq4g128(
                    qweight,
                    qzeros,
                    scales,
                    entry.logical_shape[0],
                    entry.logical_shape.iter().skip(1).product(),
                    group_size,
                ),
                DType::ParoQ4G128,
                Qwen35SourceLayout::Raw,
                format!("{quant_base}.qweight"),
                entry.logical_shape.clone(),
            )
        } else {
            let (info, data) = self.source.tensor_data(&base).unwrap();
            let dtype = match info.dtype.as_str() {
                "F16" => DType::F16,
                "BF16" => DType::BF16,
                "F32" => DType::F32,
                other => {
                    return Err(format!(
                        "Paro source tensor '{base}' has unsupported dtype {other}"
                    ))
                }
            };
            let layout = match dtype {
                DType::F16 => Qwen35SourceLayout::F16,
                DType::BF16 => Qwen35SourceLayout::BF16,
                _ => Qwen35SourceLayout::F32,
            };
            (
                data.to_vec(),
                dtype,
                layout,
                base.clone(),
                info.shape.clone(),
            )
        };
        if dtype != DType::ParoQ4G128 && shape != entry.logical_shape {
            return Err(format!(
                "Paro source '{physical_name}' shape {shape:?}, expected {:?}",
                entry.logical_shape
            ));
        }
        Ok(ResolvedQwen35Source {
            logical_name: entry.name.clone(),
            physical_name,
            bytes,
            dtype,
            layout,
            shape,
            companion: false,
        })
    }

    /// Add the three rotation records needed by every quantized projection.
    /// Their names are logical, so the assembler can attach them without
    /// teaching the generic fulfillment layer about Paro.
    pub fn manifest_with_source_records(
        &self,
        manifest: &[WeightEntry],
    ) -> Result<Vec<WeightEntry>, String> {
        self.source_records.borrow_mut().clear();
        let manifest = paro_source_order(manifest);
        let mut records_by_owner: HashMap<(String, Option<usize>), Vec<WeightEntry>> =
            HashMap::new();
        for owner in manifest
            .iter()
            .filter(|e| !e.name.ends_with(AWQ_SUFFIX) && !is_paro_record(&e.name))
        {
            let source = self.resolve_metadata(owner)?;
            if source.dtype != DType::ParoQ4G128 {
                continue;
            }
            let base = owner.name.clone();
            let owner_physical = source.physical_name.trim_end_matches(".qweight");
            self.source_records.borrow_mut().insert(
                (owner.name.clone(), owner.layer),
                source.physical_name.clone(),
            );
            let records = [
                (".paro_pairs", "pairs"),
                (".paro_theta", "theta"),
                (".paro_channel_scales", "channel_scales"),
            ];
            let mut records_for_owner = Vec::new();
            for (suffix, physical_suffix) in records {
                let physical = format!("{owner_physical}.{physical_suffix}");
                let info = self.source.tensor_info(&physical).ok_or_else(|| {
                    format!(
                        "Paro source missing required sidecar {physical} for owner '{}'",
                        owner.name
                    )
                })?;
                self.source_records.borrow_mut().insert(
                    (format!("{}{suffix}", owner.name), owner.layer),
                    physical.clone(),
                );
                records_for_owner.push(WeightEntry {
                    name: format!("{base}{suffix}"),
                    layer: owner.layer,
                    logical_shape: info.shape.clone(),
                    dtype: DType::Raw,
                    dtype_constraint: DTypeConstraint::source_exact(DType::Raw),
                    placement: owner.placement,
                    policy: owner.policy.clone(),
                });
            }
            records_by_owner.insert((owner.name.clone(), owner.layer), records_for_owner);
        }
        let mut out = Vec::with_capacity(manifest.len() + records_by_owner.len() * 3);
        for entry in &manifest {
            out.push(entry.clone());
            if let Some(records) = records_by_owner.get(&(entry.name.clone(), entry.layer)) {
                out.extend(records.iter().cloned());
            }
        }
        Ok(out)
    }

    fn physical_candidates(&self, logical: &str, layer: Option<usize>) -> Vec<String> {
        physical_candidates(self.config, logical, layer)
    }

    fn paro_quant_base(&self, logical: &str, layer: Option<usize>) -> Result<String, String> {
        let base = self
            .physical_candidates(logical, layer)
            .into_iter()
            .map(|name| name.trim_end_matches(".weight").to_string())
            .find(|base| {
                self.source
                    .tensor_info(&format!("{base}.qweight"))
                    .is_some()
            })
            .ok_or_else(|| format!("qwen35 Paro source: no tensor for '{logical}'"))?;
        for suffix in ["qzeros", "scales"] {
            if self
                .source
                .tensor_info(&format!("{base}.{suffix}"))
                .is_none()
            {
                return Err(format!("Paro source missing {base}.{suffix}"));
            }
        }
        Ok(format!("{base}.qweight"))
    }
}

impl<'a> Qwen35SourceResolver<'a> {
    pub fn new(hfq: &'a HfqFile, config: &'a Qwen35Config) -> Self {
        Self { hfq, config }
    }

    /// Resolve one main or companion manifest entry.  This reports the
    /// *source* dtype/layout exactly as stored in HFQ.  Use
    /// [`Self::resolve_for_store`] for the forward-ready representation.
    pub fn resolve_metadata(&self, entry: &WeightEntry) -> Result<ResolvedQwen35Source, String> {
        let companion = entry.name.ends_with(AWQ_SUFFIX);
        let logical = entry.name.strip_suffix(AWQ_SUFFIX).unwrap_or(&entry.name);
        let candidates = self.physical_candidates(logical, entry.layer);
        let candidates = if companion {
            candidates
                .into_iter()
                .map(|name| awq_companion_physical(&name))
                .collect()
        } else {
            candidates
        };

        let (physical_name, info) = candidates
            .into_iter()
            .find_map(|name| self.hfq.find_tensor_info(&name).map(|info| (name, info)))
            .ok_or_else(|| {
                format!(
                    "qwen35 source: no HFQ tensor for logical '{}' (layer {:?})",
                    entry.name, entry.layer
                )
            })?;

        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        if !source_shape_matches(entry, &shape, info.quant_type) {
            return Err(format!(
                "qwen35 source: '{}' resolved to '{}' with shape {:?}, expected {:?}",
                entry.name, physical_name, shape, entry.logical_shape
            ));
        }
        let (dtype, layout) = qtype_dtype(info.quant_type).ok_or_else(|| {
            format!(
                "qwen35 source: '{}' has unsupported HFQ quant_type {}",
                physical_name, info.quant_type
            )
        })?;
        if companion && (dtype != DType::F16 || layout != Qwen35SourceLayout::F16) {
            return Err(format!(
                "qwen35 source: AWQ companion '{}' must be F16, got {dtype:?}",
                physical_name
            ));
        }
        Ok(ResolvedQwen35Source {
            logical_name: entry.name.clone(),
            physical_name,
            bytes: Vec::new(),
            dtype,
            layout,
            shape,
            companion,
        })
    }

    pub fn resolve(&self, entry: &WeightEntry) -> Result<ResolvedQwen35Source, String> {
        let mut source = self.resolve_metadata(entry)?;
        let (_, bytes) = self
            .hfq
            .tensor_data_pread(&source.physical_name)
            .ok_or_else(|| {
                format!(
                    "qwen35 source: payload disappeared for '{}'",
                    source.physical_name
                )
            })?;
        source.bytes = bytes.to_vec();
        Ok(source)
    }

    /// Resolve a record for store fulfillment.  The returned bytes and dtype
    /// remain the actual HFQ source representation.  Forward-ready widening
    /// and dequantization is performed by [`assemble_qwen35_weights`] through
    /// the same runtime conversion routines used by the legacy loader; doing
    /// it here would lose the source quant type and would make qt=13 conv1d
    /// tensors impossible to convert correctly.
    pub fn resolve_for_store(&self, entry: &WeightEntry) -> Result<(Vec<u8>, DType), String> {
        let source = self.resolve(entry)?;
        Ok((source.bytes, source.dtype))
    }

    /// Return optional AWQ companion entries that actually exist in this HFQ.
    /// Optional means absent sidecars are not manufactured into required store
    /// cells; present sidecars are explicit entries and are validated by the
    /// typed assembler.
    pub fn companion_entries(&self, manifest: &[WeightEntry]) -> Result<Vec<WeightEntry>, String> {
        let mut out = Vec::new();
        for entry in manifest
            .iter()
            .filter(|entry| !entry.name.ends_with(AWQ_SUFFIX))
        {
            let main = self.resolve_metadata(entry)?;
            if !main.dtype.supports_awq_sidecar() {
                continue;
            }
            let sidecar = expected_companion_entry(entry);
            let sidecar_candidates = self
                .physical_candidates(&entry.name, entry.layer)
                .into_iter()
                .map(|name| awq_companion_physical(&name));
            if sidecar_candidates
                .into_iter()
                .any(|name| self.hfq.find_tensor_info(&name).is_some())
            {
                self.resolve_metadata(&sidecar)?;
                out.push(sidecar);
            }
        }
        Ok(out)
    }

    /// Convenience helper for callers compiling a complete Qwen35 store plan.
    pub fn manifest_with_companions(
        &self,
        manifest: &[WeightEntry],
    ) -> Result<Vec<WeightEntry>, String> {
        let companions = self.companion_entries(manifest)?;
        let mut out = Vec::with_capacity(manifest.len() + companions.len());
        for entry in manifest {
            out.push(entry.clone());
            out.extend(
                companions
                    .iter()
                    .filter(|companion| {
                        companion.layer == entry.layer
                            && companion.name.strip_suffix(AWQ_SUFFIX) == Some(entry.name.as_str())
                    })
                    .cloned(),
            );
        }
        Ok(out)
    }

    fn physical_candidates(&self, logical: &str, layer: Option<usize>) -> Vec<String> {
        physical_candidates(self.config, logical, layer)
    }
}

fn physical_candidates(config: &Qwen35Config, logical: &str, layer: Option<usize>) -> Vec<String> {
    let stem = match (logical, layer) {
        ("token_embd", None) => "embed_tokens.weight".to_string(),
        ("output_norm", None) => "norm.weight".to_string(),
        ("lm_head", None) => "lm_head.weight".to_string(),
        (name, Some(layer)) => {
            let rel = match name {
                "attn_norm" => "input_layernorm.weight".to_string(),
                "ffn_norm" => "post_attention_layernorm.weight".to_string(),
                "wq" => "self_attn.q_proj.weight".to_string(),
                "wk" => "self_attn.k_proj.weight".to_string(),
                "wv" => "self_attn.v_proj.weight".to_string(),
                "wo" => {
                    if config.layer_types[layer] == LayerType::LinearAttention {
                        "linear_attn.out_proj.weight".to_string()
                    } else {
                        "self_attn.o_proj.weight".to_string()
                    }
                }
                "q_norm" => "self_attn.q_norm.weight".to_string(),
                "k_norm" => "self_attn.k_norm.weight".to_string(),
                "wqkv" => "linear_attn.in_proj_qkv.weight".to_string(),
                "wz" => "linear_attn.in_proj_z.weight".to_string(),
                "w_alpha" => "linear_attn.in_proj_a.weight".to_string(),
                "w_beta" => "linear_attn.in_proj_b.weight".to_string(),
                "a_log" => "linear_attn.A_log".to_string(),
                "dt_bias" => "linear_attn.dt_bias".to_string(),
                "conv" => "linear_attn.conv1d.weight".to_string(),
                "norm" => "linear_attn.norm.weight".to_string(),
                "ffn_gate" => "mlp.gate_proj.weight".to_string(),
                "ffn_up" => "mlp.up_proj.weight".to_string(),
                "ffn_down" => "mlp.down_proj.weight".to_string(),
                "router" => "mlp.gate.weight".to_string(),
                "shared_expert_gate" => "mlp.shared_expert_gate.weight".to_string(),
                "shared_gate" => "mlp.shared_expert.gate_proj.weight".to_string(),
                "shared_up" => "mlp.shared_expert.up_proj.weight".to_string(),
                "shared_down" => "mlp.shared_expert.down_proj.weight".to_string(),
                name if name.starts_with("expert.") => {
                    let rest = name.strip_prefix("expert.").unwrap();
                    let (idx, proj) = rest.split_once('.').ok_or(()).unwrap();
                    format!(
                        "mlp.experts.{idx}.{}.weight",
                        match proj {
                            "gate_up" => "gate_up_proj",
                            "down" => "down_proj",
                            _ => return Vec::new(),
                        }
                    )
                }
                _ => return Vec::new(),
            };
            format!("layers.{layer}.{rel}")
        }
        _ => return Vec::new(),
    };

    let mut out = Vec::with_capacity(3);
    let push = |out: &mut Vec<String>, name: String| {
        if !out.iter().any(|candidate| candidate == &name) {
            out.push(name);
        }
    };
    if logical == "lm_head" {
        push(&mut out, stem.clone());
        push(&mut out, "model.language_model.lm_head.weight".into());
        push(&mut out, "model.lm_head.weight".into());
        if config.tie_word_embeddings {
            push(&mut out, "model.language_model.embed_tokens.weight".into());
            push(&mut out, "model.embed_tokens.weight".into());
            push(&mut out, "embed_tokens.weight".into());
        }
        return out;
    }
    push(&mut out, format!("model.language_model.{stem}"));
    push(&mut out, format!("model.{stem}"));
    push(&mut out, stem);
    out
}

/// Return the Paro source-read order.  The legacy Paro orchestrator reads the
/// scalar shared-expert gate before the three quantized shared-expert
/// projections; HFQ's legacy order reads those four records in the opposite
/// order.  Keep the source-specific order at the manifest boundary rather
/// than forcing one common order onto both formats.
fn paro_source_order(manifest: &[WeightEntry]) -> Vec<WeightEntry> {
    const SHARED: [&str; 4] = [
        "shared_expert_gate",
        "shared_gate",
        "shared_up",
        "shared_down",
    ];
    let mut out = Vec::with_capacity(manifest.len());
    let mut emitted = std::collections::HashSet::new();
    for entry in manifest {
        if SHARED.contains(&entry.name.as_str()) {
            if emitted.insert(entry.layer) {
                for name in SHARED {
                    if let Some(shared) = manifest
                        .iter()
                        .find(|candidate| candidate.layer == entry.layer && candidate.name == name)
                    {
                        out.push(shared.clone());
                    }
                }
            }
        } else {
            out.push(entry.clone());
        }
    }
    out
}

/// Resolve the HFQ wire quant type to the dtype carried by a resident store
/// cell.  Host-decoded formats are still identified by their actual source
/// dtype; no logical F16/F32 promise is substituted here.
pub fn qtype_dtype(qt: u8) -> Option<(DType, Qwen35SourceLayout)> {
    let pair = match qt {
        0 => (DType::Q4F16G64, Qwen35SourceLayout::Raw),
        1 => (DType::F16, Qwen35SourceLayout::F16),
        2 => (DType::F32, Qwen35SourceLayout::F32),
        3 => (DType::Q8_0, Qwen35SourceLayout::Raw),
        4 => (DType::Q4K, Qwen35SourceLayout::Raw),
        5 => (DType::Q8HFQ, Qwen35SourceLayout::Raw),
        6 => (DType::HFQ4G256, Qwen35SourceLayout::Raw),
        7 => (DType::HFQ4G128, Qwen35SourceLayout::Raw),
        8 => (DType::HFQ6G256, Qwen35SourceLayout::Raw),
        9 => (DType::HFQ2G256, Qwen35SourceLayout::Raw),
        10 => (DType::HFQ2G128, Qwen35SourceLayout::Raw),
        11 => (DType::HFQ3G256, Qwen35SourceLayout::Raw),
        12 => (DType::HFQ3G128, Qwen35SourceLayout::Raw),
        13 => (DType::MQ4G256, Qwen35SourceLayout::Raw),
        14 => (DType::MQ8G256, Qwen35SourceLayout::Raw),
        15 => (DType::MQ6G256, Qwen35SourceLayout::Raw),
        16 => (DType::BF16, Qwen35SourceLayout::BF16),
        17 => (DType::MQ3G256, Qwen35SourceLayout::Raw),
        18 => (DType::MQ2G256, Qwen35SourceLayout::Raw),
        19 => (DType::MQ2G256Lloyd, Qwen35SourceLayout::Raw),
        20 => (DType::MQ3G256Lloyd, Qwen35SourceLayout::Raw),
        21 => (DType::HFP4G32, Qwen35SourceLayout::Raw),
        24 => (DType::MFP4G32, Qwen35SourceLayout::Raw),
        30 => (DType::MQ4G256Lloyd, Qwen35SourceLayout::Raw),
        31 => (DType::MQ5G256, Qwen35SourceLayout::Raw),
        32 => (DType::MFP4G32Lloyd, Qwen35SourceLayout::Raw),
        33 => (DType::MFP4G32P, Qwen35SourceLayout::Raw),
        34 => (DType::MFP4G32E8, Qwen35SourceLayout::Raw),
        35 => (DType::MFP4G32E8SOA, Qwen35SourceLayout::Raw),
        36 => (DType::MFP3G32E8, Qwen35SourceLayout::Raw),
        37 => (DType::MFP2G32E8, Qwen35SourceLayout::Raw),
        _ => return None,
    };
    Some(pair)
}

fn dtype_qtype(dtype: DType) -> Option<u8> {
    Some(match dtype {
        DType::Q4F16G64 => 0,
        DType::F16 => 1,
        DType::F32 => 2,
        DType::Q8_0 => 3,
        DType::Q4K => 4,
        DType::Q8HFQ => 5,
        DType::HFQ4G256 => 6,
        DType::HFQ4G128 => 7,
        DType::HFQ6G256 => 8,
        DType::HFQ2G256 => 9,
        DType::HFQ2G128 => 10,
        DType::HFQ3G256 => 11,
        DType::HFQ3G128 => 12,
        DType::MQ4G256 => 13,
        DType::MQ8G256 => 14,
        DType::MQ6G256 => 15,
        DType::BF16 => 16,
        DType::MQ3G256 => 17,
        DType::MQ2G256 => 18,
        DType::MQ2G256Lloyd => 19,
        DType::MQ3G256Lloyd => 20,
        DType::HFP4G32 => 21,
        DType::MFP4G32 => 24,
        DType::MQ4G256Lloyd => 30,
        DType::MQ5G256 => 31,
        DType::MFP4G32Lloyd => 32,
        DType::MFP4G32P => 33,
        DType::MFP4G32E8 => 34,
        DType::MFP4G32E8SOA => 35,
        DType::MFP3G32E8 => 36,
        DType::MFP2G32E8 => 37,
        _ => return None,
    })
}

fn source_allowed(constraint: &DTypeConstraint, dtype: DType) -> bool {
    match &constraint.source {
        SourceDType::Any => true,
        SourceDType::Exact(expected) => *expected == dtype,
        SourceDType::OneOf(allowed) => allowed.contains(&dtype),
    }
}

fn sidecar_name(name: &str) -> String {
    format!("{name}{AWQ_SUFFIX}")
}

fn expected_companion_entry(owner: &WeightEntry) -> WeightEntry {
    WeightEntry {
        name: sidecar_name(&owner.name),
        layer: owner.layer,
        logical_shape: vec![owner.logical_shape.last().copied().unwrap_or(0)],
        dtype: DType::F32,
        dtype_constraint: DTypeConstraint::source_exact(DType::F16),
        placement: owner.placement,
        policy: match &owner.policy {
            ShardPolicy::Tied { source } => ShardPolicy::Tied {
                source: sidecar_name(source),
            },
            policy => policy.clone(),
        },
    }
}

fn awq_companion_physical(name: &str) -> String {
    match name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{name}.awq_scale.weight"),
    }
}

fn source_shape_matches(entry: &WeightEntry, shape: &[usize], quant_type: u8) -> bool {
    if shape == entry.logical_shape {
        return true;
    }
    // HFQ preserves Conv1d's physical [channels, 1, kernel] shape while the
    // Qwen35 manifest exposes the legacy raw_f32 flattened element count.
    // qt=13 is still decoded by the legacy dequant_f32 path; only the metadata
    // representation differs.
    entry.name == "conv"
        && entry.layer.is_some()
        && quant_type == 13
        && shape.len() == 3
        && shape[1] == 1
        && shape[2] == 4
        && shape.iter().product::<usize>() == entry.logical_shape.iter().product::<usize>()
}

fn is_canonical_norm(entry: &WeightEntry) -> bool {
    matches!(
        entry.name.as_str(),
        "attn_norm" | "ffn_norm" | "output_norm" | "q_norm" | "k_norm"
    )
}

fn is_raw_deltanet(entry: &WeightEntry) -> bool {
    entry.layer.is_some() && matches!(entry.name.as_str(), "a_log" | "dt_bias" | "conv" | "norm")
}

fn resident<'a>(handle: &'a WeightHandle, entry: &WeightEntry) -> Result<&'a GpuTensor, String> {
    match handle {
        WeightHandle::Resident(t) => Ok(t),
        WeightHandle::Alias(_) => Err(format!(
            "qwen35 assembler: '{}' requires a resident tensor, got alias",
            entry.name
        )),
    }
}

fn resident_through_alias<'a>(
    store: &'a WeightStore,
    mut handle: &'a WeightHandle,
    layer: Option<usize>,
    entry: &WeightEntry,
) -> Result<&'a GpuTensor, String> {
    for _ in 0..4 {
        match handle {
            WeightHandle::Resident(tensor) => return Ok(tensor),
            WeightHandle::Alias(source) => {
                handle = store.get(source, layer, 0).ok_or_else(|| {
                    format!(
                        "qwen35 assembler: alias '{}' points to missing '{}', layer {:?}",
                        entry.name, source, layer
                    )
                })?;
            }
        }
    }
    Err(format!(
        "qwen35 assembler: alias chain for '{}' is too deep",
        entry.name
    ))
}

fn check_source_cell(
    store: &WeightStore,
    entry: &WeightEntry,
    device: usize,
) -> Result<(), String> {
    let handle = store.get(&entry.name, entry.layer, device).ok_or_else(|| {
        format!(
            "missing store cell {}[{:#?}] on device {device}",
            entry.name, entry.layer
        )
    })?;
    if let WeightHandle::Alias(source) = handle {
        let ShardPolicy::Tied { source: expected } = &entry.policy else {
            return Err(format!(
                "unexpected alias in non-tied cell '{}'",
                entry.name
            ));
        };
        if source != expected {
            return Err(format!(
                "alias '{}' points to '{}', expected '{}'",
                entry.name, source, expected
            ));
        }
        return Ok(());
    }
    let tensor = resident(handle, entry)?;
    if tensor.shape != entry.logical_shape {
        return Err(format!(
            "store cell '{}' shape {:?}, expected {:?}",
            entry.name, tensor.shape, entry.logical_shape
        ));
    }
    if !source_allowed(&entry.dtype_constraint, tensor.dtype) {
        return Err(format!(
            "store cell '{}' dtype {:?} violates source constraint {:?}",
            entry.name, tensor.dtype, entry.dtype_constraint.source
        ));
    }
    Ok(())
}

fn check_forward_handle(handle: &WeightHandle, entry: &WeightEntry) -> Result<(), String> {
    if let WeightHandle::Alias(source) = handle {
        if let ShardPolicy::Tied { source: expected } = &entry.policy {
            if source == expected {
                return Ok(());
            }
        }
        return Err(format!("unexpected alias in '{}'", entry.name));
    }
    let tensor = resident(handle, entry)?;
    if tensor.shape != entry.logical_shape {
        return Err(format!(
            "forward-ready '{}' shape {:?}, expected {:?}",
            entry.name, tensor.shape, entry.logical_shape
        ));
    }
    if let Some(expected) = canonical_store_dtype(entry) {
        if tensor.dtype != expected {
            return Err(format!(
                "forward-ready '{}' dtype {:?}, expected {:?}",
                entry.name, tensor.dtype, expected
            ));
        }
    }
    Ok(())
}

fn should_widen_to_f32(entry: &WeightEntry, dtype: DType) -> bool {
    entry.name.ends_with(AWQ_SUFFIX)
        || is_canonical_norm(entry)
        || is_raw_deltanet(entry)
        || (entry.name == "token_embd" && dtype == DType::MQ4G256)
        || matches!(dtype, DType::F16 | DType::BF16)
}

fn convert_handle_forward_ready(
    gpu: &mut Gpu,
    entry: &WeightEntry,
    handle: &WeightHandle,
) -> Result<Option<WeightHandle>, String> {
    let WeightHandle::Resident(source) = handle else {
        return Ok(None);
    };
    if !should_widen_to_f32(entry, source.dtype) {
        return Ok(None);
    }
    let quant_type = dtype_qtype(source.dtype).ok_or_else(|| {
        format!(
            "no legacy conversion path for {:?} '{}'",
            source.dtype, entry.name
        )
    })?;
    let mut bytes = vec![0u8; source.buf.size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &source.buf)
        .map_err(|e| format!("readback for '{}' failed: {e:?}", entry.name))?;
    let mut converted = if is_canonical_norm(entry) {
        dequant_norm(gpu, quant_type, &bytes, &entry.logical_shape, 1.0)
            .map_err(|e| format!("legacy norm conversion for '{}' failed: {e:?}", entry.name))?
    } else {
        let n = entry.logical_shape.iter().product();
        dequant_f32(gpu, quant_type, &bytes, n).map_err(|e| {
            format!(
                "legacy scalar conversion for '{}' failed: {e:?}",
                entry.name
            )
        })?
    };
    converted.shape = entry.logical_shape.clone();
    Ok(Some(WeightHandle::Resident(converted)))
}

fn free_resident_buffer_retaining_owner(gpu: &Gpu, tensor: &GpuTensor) -> Result<(), String> {
    let raw = unsafe { hip_bridge::DeviceBuffer::from_raw(tensor.buf.as_ptr(), tensor.buf.size()) };
    gpu.hip
        .free(raw)
        .map_err(|e| format!("source buffer free failed: {e:?}"))
}

struct ReplacementGuard {
    tensor: Option<GpuTensor>,
    free: Box<dyn FnMut(GpuTensor)>,
}

impl ReplacementGuard {
    fn new<F>(tensor: GpuTensor, free: F) -> Self
    where
        F: FnMut(GpuTensor) + 'static,
    {
        Self {
            tensor: Some(tensor),
            free: Box::new(free),
        }
    }

    fn take(&mut self) -> GpuTensor {
        self.tensor
            .take()
            .expect("replacement guard consumed twice")
    }
}

impl Drop for ReplacementGuard {
    fn drop(&mut self) {
        if let Some(tensor) = self.tensor.take() {
            (self.free)(tensor);
        }
    }
}

fn canonical_store_dtype(entry: &WeightEntry) -> Option<DType> {
    if entry.name.ends_with(AWQ_SUFFIX) || is_canonical_norm(entry) || is_raw_deltanet(entry) {
        Some(DType::F32)
    } else {
        None
    }
}

fn validate_typed_embedding_dtype(dtype: DType) -> Result<(), String> {
    if matches!(
        dtype,
        DType::HFQ4G256 | DType::HFQ4G128 | DType::Q8_0 | DType::F16 | DType::BF16 | DType::F32
    ) {
        Ok(())
    } else {
        Err(format!(
            "qwen35 assembler: unsupported typed embedding dtype {dtype:?}"
        ))
    }
}

fn validate_manifest_schema(config: &Qwen35Config, manifest: &[WeightEntry]) -> Result<(), String> {
    let expected_manifest = Qwen35::weight_manifest(config);
    let main_entries: Vec<&WeightEntry> = manifest
        .iter()
        .filter(|entry| !entry.name.ends_with(AWQ_SUFFIX) && !is_paro_record(&entry.name))
        .collect();
    for expected in expected_manifest.iter() {
        if !main_entries
            .iter()
            .any(|entry| entry.name == expected.name && entry.layer == expected.layer)
        {
            return Err(format!(
                "qwen35 assembler: manifest is missing {}[{:#?}]",
                expected.name, expected.layer
            ));
        }
    }
    for entry in &main_entries {
        let expected = expected_manifest
            .iter()
            .find(|expected| expected.name == entry.name && expected.layer == entry.layer)
            .ok_or_else(|| {
                format!(
                    "qwen35 assembler: unexpected manifest record {}[{:#?}]",
                    entry.name, entry.layer
                )
            })?;
        if entry.logical_shape != expected.logical_shape
            || entry.dtype != expected.dtype
            || entry.dtype_constraint != expected.dtype_constraint
            || entry.policy != expected.policy
            || entry.placement != expected.placement
        {
            return Err(format!(
                "qwen35 assembler: non-canonical manifest metadata for {}[{:#?}]",
                entry.name, entry.layer
            ));
        }
        if placement_devices(entry, &DeviceMesh::single(), config.n_layers) != vec![0] {
            return Err(format!(
                "qwen35 assembler: {}[{:#?}] is not placed on device 0",
                entry.name, entry.layer
            ));
        }
    }
    let mut seen_companions = std::collections::HashSet::new();
    for entry in manifest
        .iter()
        .filter(|entry| entry.name.ends_with(AWQ_SUFFIX))
    {
        let owner = entry.name.trim_end_matches(AWQ_SUFFIX);
        let owner = main_entries
            .iter()
            .find(|candidate| candidate.name == owner && candidate.layer == entry.layer)
            .ok_or_else(|| format!("sidecar '{}' has no owner", entry.name))?;
        if !seen_companions.insert((entry.name.clone(), entry.layer)) {
            return Err(format!(
                "qwen35 assembler: duplicate sidecar '{}[{:#?}]'",
                entry.name, entry.layer
            ));
        }
        let expected = expected_companion_entry(owner);
        if entry != &expected {
            return Err(format!(
                "qwen35 assembler: non-canonical companion metadata for {}[{:#?}]",
                entry.name, entry.layer
            ));
        }
    }
    for entry in manifest.iter().filter(|entry| is_paro_record(&entry.name)) {
        let suffix = PARO_SUFFIXES
            .iter()
            .find(|suffix| entry.name.ends_with(**suffix))
            .expect("is_paro_record checked");
        let owner = entry.name.trim_end_matches(suffix);
        let owner = main_entries
            .iter()
            .find(|candidate| candidate.name == owner && candidate.layer == entry.layer)
            .ok_or_else(|| format!("Paro record '{}' has no owner", entry.name))?;
        if entry.placement != owner.placement || entry.policy != owner.policy {
            return Err(format!("non-canonical Paro record '{}'", entry.name));
        }
    }
    Ok(())
}

fn tensor_from_handle(
    handle: WeightHandle,
    shape: &[usize],
    sidecar: Option<GpuTensor>,
    paro: Option<hipfire_runtime::llama::ParoRotation>,
) -> WeightTensor {
    let WeightHandle::Resident(buf) = handle else {
        panic!("validated qwen35 typed cell was not resident")
    };
    let m = shape.first().copied().unwrap_or(1);
    let k = shape.iter().skip(1).product::<usize>().max(1);
    let dtype = buf.dtype;
    WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: dtype.row_stride(k),
        paro,
        awq_scale: sidecar,
    }
}

fn tensor_handle(taken: &mut [Option<TakenWeight>], slot: usize) -> WeightHandle {
    taken[slot]
        .take()
        .expect("qwen35 assembler slot consumed twice")
        .handle
}

fn gpu_handle(taken: &mut [Option<TakenWeight>], slot: usize) -> GpuTensor {
    match tensor_handle(taken, slot) {
        WeightHandle::Resident(t) => t,
        WeightHandle::Alias(_) => panic!("validated qwen35 GPU cell was an alias"),
    }
}

fn typed_weight(
    taken: &mut [Option<TakenWeight>],
    slots: &HashMap<(String, Option<usize>), usize>,
    name: &str,
    layer: usize,
    shape: Vec<usize>,
) -> WeightTensor {
    let main_slot = *slots
        .get(&(name.to_string(), Some(layer)))
        .expect("preflighted Qwen35 weight key missing");
    let side_slot = slots.get(&(sidecar_name(name), Some(layer))).copied();
    let sidecar = side_slot.map(|slot| match tensor_handle(taken, slot) {
        WeightHandle::Resident(t) => t,
        WeightHandle::Alias(_) => panic!("validated Qwen35 sidecar was an alias"),
    });
    let paro = paro_rotation(taken, slots, name, layer);
    tensor_from_handle(tensor_handle(taken, main_slot), &shape, sidecar, paro)
}

fn paro_rotation(
    taken: &mut [Option<TakenWeight>],
    slots: &HashMap<(String, Option<usize>), usize>,
    name: &str,
    layer: usize,
) -> Option<hipfire_runtime::llama::ParoRotation> {
    let get = |suffix: &str| {
        slots
            .get(&(format!("{name}{suffix}"), Some(layer)))
            .copied()
    };
    let pairs = get(".paro_pairs")?;
    let theta = get(".paro_theta")?;
    let scales = get(".paro_channel_scales")?;
    let pairs = gpu_handle(taken, pairs);
    let theta = gpu_handle(taken, theta);
    let channel_scales = gpu_handle(taken, scales);
    Some(hipfire_runtime::llama::ParoRotation {
        krot: pairs.shape.first().copied().unwrap_or(8) as u32,
        group_size: 128,
        pairs,
        theta,
        channel_scales,
        is_alias: false,
    })
}

fn typed_moe_ffn(
    taken: &mut [Option<TakenWeight>],
    slots: &HashMap<(String, Option<usize>), usize>,
    config: &Qwen35Config,
    layer: usize,
    gate_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    down_awq_ptrs: Option<GpuTensor>,
    dtype_tags: Option<GpuTensor>,
) -> MoeFfnWeights {
    let d = config.dim;
    let router = typed_weight(taken, slots, "router", layer, vec![config.num_experts, d]);
    let shared_expert_gate = typed_weight(taken, slots, "shared_expert_gate", layer, vec![1, d]);
    let shared_expert = SharedExpertWeights {
        gate: typed_weight(
            taken,
            slots,
            "shared_gate",
            layer,
            vec![config.shared_expert_intermediate_size, d],
        ),
        up: typed_weight(
            taken,
            slots,
            "shared_up",
            layer,
            vec![config.shared_expert_intermediate_size, d],
        ),
        down: typed_weight(
            taken,
            slots,
            "shared_down",
            layer,
            vec![d, config.shared_expert_intermediate_size],
        ),
    };
    let mut experts = Vec::with_capacity(config.num_experts);
    for expert in 0..config.num_experts {
        experts.push(ExpertWeights {
            gate_up: typed_weight(
                taken,
                slots,
                &format!("expert.{expert}.gate_up"),
                layer,
                vec![2 * config.moe_intermediate_size, d],
            ),
            down: typed_weight(
                taken,
                slots,
                &format!("expert.{expert}.down"),
                layer,
                vec![d, config.moe_intermediate_size],
            ),
        });
    }
    MoeFfnWeights {
        router,
        experts,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs: gate_ptrs,
        expert_down_ptrs: down_ptrs,
        expert_down_awq_ptrs: down_awq_ptrs,
        expert_dtype_tags: dtype_tags,
        expert_gate_up_dummy: None,
        layer_idx: layer as u16,
        expert_shape: None,
        paro_shared: None,
    }
}

struct DerivedGuard {
    gpu: *const Gpu,
    tensors: Vec<GpuTensor>,
    active: bool,
}

impl Drop for DerivedGuard {
    fn drop(&mut self) {
        if self.active {
            for tensor in self.tensors.drain(..) {
                let _ = unsafe { (&*self.gpu).hip.free(tensor.buf) };
            }
        }
    }
}

fn alloc_derived(gpu: &mut Gpu, bytes: &[Vec<u8>]) -> Result<DerivedGuard, String> {
    let mut tensors = Vec::with_capacity(bytes.len());
    for payload in bytes {
        let tensor = gpu
            .alloc_tensor(&[payload.len()], DType::Raw)
            .map_err(|e| format!("derived record allocation failed: {e:?}"))?;
        if let Err(e) = gpu.hip.memcpy_htod(&tensor.buf, payload) {
            let _ = gpu.free_tensor(tensor);
            for prior in tensors.drain(..) {
                let _ = gpu.free_tensor(prior);
            }
            return Err(format!("derived record upload failed: {e:?}"));
        }
        tensors.push(tensor);
    }
    Ok(DerivedGuard {
        gpu: gpu as *mut Gpu as *const Gpu,
        tensors,
        active: true,
    })
}

fn ptr_bytes(ptrs: &[u64]) -> Vec<u8> {
    ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect()
}

fn dtype_tag(gate_up: DType, down: DType) -> u8 {
    match gate_up {
        DType::MQ6G256 => 0,
        DType::MQ2G256Lloyd => 1,
        DType::MQ3G256Lloyd => 3,
        DType::MFP4G32E8 => 4,
        DType::MFP3G32E8 => 5,
        DType::MFP2G32E8 => 6,
        DType::MQ4G256 => match down {
            DType::MQ6G256 => 0,
            DType::MQ2G256Lloyd => 1,
            DType::MFP2G32E8 => 6,
            DType::MQ3G256Lloyd => 3,
            DType::MFP3G32E8 => 5,
            _ => 2,
        },
        _ => 2,
    }
}

struct DerivedLayerPlan {
    has_down_awq: bool,
    dtype_tags: Option<Vec<u8>>,
}

/// Assemble a single-device Qwen35 store.  The function is intentionally not
/// called by the production HFQ/Dir loaders yet (that is Task 3c).
pub fn assemble_qwen35_weights(
    store: &mut WeightStore,
    config: &Qwen35Config,
    manifest: &[WeightEntry],
    gpu: &mut Gpu,
) -> Result<Qwen35Weights, String> {
    assemble_qwen35_weights_inner(store, config, manifest, gpu, false)
}

fn assemble_qwen35_weights_inner(
    store: &mut WeightStore,
    config: &Qwen35Config,
    manifest: &[WeightEntry],
    gpu: &mut Gpu,
    fail_after_commit: bool,
) -> Result<Qwen35Weights, String> {
    let device = 0;
    let main_entries: Vec<&WeightEntry> = manifest
        .iter()
        .filter(|entry| !entry.name.ends_with(AWQ_SUFFIX) && !is_paro_record(&entry.name))
        .collect();
    let companion_entries: Vec<&WeightEntry> = manifest
        .iter()
        .filter(|entry| entry.name.ends_with(AWQ_SUFFIX))
        .collect();

    if store.len() != manifest.len() {
        return Err(format!(
            "qwen35 assembler: store has {} cells, manifest expects {}",
            store.len(),
            manifest.len()
        ));
    }
    validate_manifest_schema(config, manifest)?;

    // Full preflight happens before the first take.  In particular this checks
    // every layer, every expert slot, aliases, shapes, source dtypes, and all
    // present sidecars.  Derived pointer records are computed only after the
    // source cells have been converted to their forward-ready residents.
    for entry in &main_entries {
        check_source_cell(store, entry, device)?;
    }
    for entry in &companion_entries {
        check_source_cell(store, entry, device)?;
        let owner = entry.name.trim_end_matches(AWQ_SUFFIX);
        let owner_entry = main_entries
            .iter()
            .find(|candidate| candidate.name == owner && candidate.layer == entry.layer)
            .ok_or_else(|| format!("sidecar '{}' has no owner", entry.name))?;
        let side = resident_through_alias(
            store,
            store.get(&entry.name, entry.layer, device).unwrap(),
            entry.layer,
            entry,
        )?;
        if side.dtype != DType::F16 || side.shape.len() != 1 {
            return Err(format!(
                "sidecar '{}' is not a source 1D F16 tensor",
                entry.name
            ));
        }
        let expected_k = owner_entry.logical_shape.last().copied().unwrap_or(0);
        if side.shape != [expected_k] {
            return Err(format!(
                "sidecar '{}' shape {:?}, expected [{expected_k}]",
                entry.name, side.shape
            ));
        }
        let owner_tensor = match &owner_entry.policy {
            ShardPolicy::Tied { source } => {
                let source_entry = main_entries
                    .iter()
                    .find(|candidate| {
                        candidate.name == *source && candidate.layer == owner_entry.layer
                    })
                    .ok_or_else(|| {
                        format!(
                            "tied sidecar '{}' source '{}' is missing",
                            entry.name, source
                        )
                    })?;
                resident(
                    store
                        .get(&source_entry.name, source_entry.layer, device)
                        .unwrap(),
                    source_entry,
                )?
            }
            _ => resident(
                store
                    .get(&owner_entry.name, owner_entry.layer, device)
                    .unwrap(),
                owner_entry,
            )?,
        };
        if !owner_tensor.dtype.supports_awq_sidecar() {
            return Err(format!(
                "sidecar '{}' attached to unsupported dtype {:?}",
                entry.name, owner_tensor.dtype
            ));
        }
    }
    for entry in manifest.iter().filter(|entry| is_paro_record(&entry.name)) {
        check_source_cell(store, entry, device)?;
    }

    let token_entry = main_entries
        .iter()
        .find(|entry| entry.name == "token_embd" && entry.layer.is_none())
        .ok_or("qwen35 assembler: manifest is missing token_embd")?;
    if let Some(WeightHandle::Alias(source)) = store.get("lm_head", None, device) {
        if source != "token_embd" {
            return Err(format!(
                "qwen35 assembler: lm_head alias points to '{source}'"
            ));
        }
    }

    let mut slots_by_key = HashMap::new();
    // Reservation is now infallible by construction.  The rollback guard owns
    // all taken and untaken residents.  The raw reborrow below is deliberate:
    // derived GPU records are fallible assembly work and must run while this
    // rollback guard is already active.
    let gpu_ptr = gpu as *mut Gpu;
    let mut tx = store.begin_assembly(WeightStoreTarget::Gpu(&*gpu));
    for entry in manifest {
        let slot = tx.take(&entry.name, entry.layer, 0).ok_or_else(|| {
            format!(
                "store cell disappeared while assembling {}[{:#?}]",
                entry.name, entry.layer
            )
        })?;
        slots_by_key.insert((entry.name.clone(), entry.layer), slot);
    }
    let mut guard = tx.commit();
    for entry in manifest {
        let slot = *slots_by_key
            .get(&(entry.name.clone(), entry.layer))
            .expect("preflighted Qwen35 store key missing");
        let converted = unsafe {
            convert_handle_forward_ready(&mut *gpu_ptr, entry, guard.get(slot).unwrap())?
        };
        if let Some(converted) = converted {
            let WeightHandle::Resident(converted) = converted else {
                return Err(format!(
                    "forward-ready conversion for '{}' returned an alias",
                    entry.name
                ));
            };
            let mut replacement = ReplacementGuard::new(converted, move |tensor| {
                let raw = unsafe {
                    hip_bridge::DeviceBuffer::from_raw(tensor.buf.as_ptr(), tensor.buf.size())
                };
                let _ = unsafe { (&*gpu_ptr).hip.free(raw) };
            });
            if let Some(WeightHandle::Resident(old)) = guard.get(slot) {
                if let Err(error) = free_resident_buffer_retaining_owner(unsafe { &*gpu_ptr }, old)
                {
                    return Err(error);
                }
            }
            let replacement_handle = WeightHandle::Resident(replacement.take());
            if let Err((handle, error)) = guard.replace_after_free(slot, replacement_handle) {
                if let WeightHandle::Resident(tensor) = handle {
                    let _ = ReplacementGuard::new(tensor, move |tensor| {
                        let raw = unsafe {
                            hip_bridge::DeviceBuffer::from_raw(
                                tensor.buf.as_ptr(),
                                tensor.buf.size(),
                            )
                        };
                        let _ = unsafe { (&*gpu_ptr).hip.free(raw) };
                    });
                }
                return Err(error);
            }
        }
    }
    let token_slot = *slots_by_key
        .get(&("token_embd".to_string(), None))
        .expect("preflighted token embedding slot missing");
    let token = resident(guard.get(token_slot).unwrap(), token_entry)?;
    validate_typed_embedding_dtype(token.dtype)?;
    for entry in manifest {
        let slot = *slots_by_key
            .get(&(entry.name.clone(), entry.layer))
            .expect("preflighted Qwen35 store key missing");
        check_forward_handle(guard.get(slot).unwrap(), entry)?;
    }

    let mut derived_payloads = Vec::new();
    let mut derived_plans = Vec::new();
    if config.num_experts > 0 {
        let mut gate_ptrs = Vec::with_capacity(config.num_experts);
        let mut down_ptrs = Vec::with_capacity(config.num_experts);
        let mut down_awq_ptrs = Vec::with_capacity(config.num_experts);
        let mut expert_tags = Vec::with_capacity(config.num_experts);
        let mut expert_dtype_pairs = Vec::with_capacity(config.num_experts);
        for layer in 0..config.n_layers {
            if !matches!(
                config.layer_types[layer],
                LayerType::LinearAttention | LayerType::FullAttention
            ) {
                return Err(format!("invalid Qwen35 layer type at {layer}"));
            }
            for expert in 0..config.num_experts {
                let mut expert_dtypes = [DType::Raw; 2];
                for (index, (suffix, ptrs)) in
                    [("gate_up", &mut gate_ptrs), ("down", &mut down_ptrs)]
                        .into_iter()
                        .enumerate()
                {
                    let name = format!("expert.{expert}.{suffix}");
                    let entry = main_entries
                        .iter()
                        .find(|entry| entry.name == name && entry.layer == Some(layer))
                        .ok_or_else(|| format!("missing expert mapping {name}[{layer}]"))?;
                    let slot = *slots_by_key
                        .get(&(entry.name.clone(), entry.layer))
                        .expect("preflighted expert slot missing");
                    let tensor = resident(guard.get(slot).unwrap(), entry)?;
                    expert_dtypes[index] = tensor.dtype;
                    ptrs.push(tensor.buf.as_ptr() as u64);
                }
                let down_name = sidecar_name(&format!("expert.{expert}.down"));
                if let Some(entry) = companion_entries
                    .iter()
                    .find(|entry| entry.name == down_name && entry.layer == Some(layer))
                {
                    let slot = *slots_by_key
                        .get(&(entry.name.clone(), entry.layer))
                        .expect("preflighted AWQ sidecar slot missing");
                    let sidecar = resident(guard.get(slot).unwrap(), entry)?;
                    down_awq_ptrs.push(sidecar.buf.as_ptr() as u64);
                }
                expert_tags.push(dtype_tag(expert_dtypes[0], expert_dtypes[1]));
                expert_dtype_pairs.push((expert_dtypes[0], expert_dtypes[1]));
            }
            let awq_count = down_awq_ptrs.len();
            if awq_count != 0 && awq_count != config.num_experts {
                return Err(format!(
                    "qwen35 assembler: partial MoE down AWQ coverage {awq_count}/{}",
                    config.num_experts
                ));
            }
            let first_pair = expert_dtype_pairs.first().copied();
            let mixed_tags = first_pair
                .is_some_and(|first| expert_dtype_pairs.iter().any(|&pair| pair != first));
            derived_payloads.push(ptr_bytes(&gate_ptrs));
            derived_payloads.push(ptr_bytes(&down_ptrs));
            if awq_count == config.num_experts {
                derived_payloads.push(ptr_bytes(&down_awq_ptrs));
            }
            let dtype_tags = mixed_tags.then(|| expert_tags.clone());
            if let Some(tags) = &dtype_tags {
                derived_payloads.push(tags.clone());
            }
            derived_plans.push(DerivedLayerPlan {
                has_down_awq: awq_count == config.num_experts,
                dtype_tags,
            });
            gate_ptrs.clear();
            down_ptrs.clear();
            down_awq_ptrs.clear();
            expert_tags.clear();
            expert_dtype_pairs.clear();
        }
    }
    let mut derived = unsafe { alloc_derived(&mut *gpu_ptr, &derived_payloads)? };
    let token_sidecar_slot = slots_by_key
        .get(&(sidecar_name("token_embd"), None))
        .copied();
    let output_sidecar_slot = slots_by_key.get(&(sidecar_name("lm_head"), None)).copied();
    let keep_token_sidecar = matches!(
        (
            guard.get(*slots_by_key.get(&("lm_head".into(), None)).unwrap()),
            output_sidecar_slot.and_then(|slot| guard.get(slot)),
        ),
        (Some(WeightHandle::Alias(_)), Some(WeightHandle::Alias(_)))
    );
    if let Some(slot) = token_sidecar_slot {
        if !keep_token_sidecar {
            guard.discard_resident(slot)?;
        }
    }
    if fail_after_commit {
        return Err("injected Qwen35 typed-assembly failure after commit".into());
    }
    let taken = guard.finalize();
    derived.active = false;
    let mut taken = taken.into_iter().map(Some).collect::<Vec<_>>();

    let slot = |name: &str, layer: Option<usize>| {
        *slots_by_key
            .get(&(name.to_string(), layer))
            .expect("preflighted Qwen35 store key missing")
    };
    let token_slot = slot("token_embd", None);
    let token_embd = gpu_handle(&mut taken, token_slot);
    let embd_format = match token_embd.dtype {
        DType::HFQ4G256 => EmbeddingFormat::HFQ4G256,
        DType::HFQ4G128 => EmbeddingFormat::HFQ4G128,
        DType::Q8_0 => EmbeddingFormat::Q8_0,
        DType::F32 => EmbeddingFormat::F32,
        other => unreachable!("preflighted embedding dtype is not forward-ready: {other:?}"),
    };

    let output_slot = slot("lm_head", None);
    let output_handle = tensor_handle(&mut taken, output_slot);
    let (output, lm_head_aliases_embd) = match output_handle {
        WeightHandle::Alias(source) => {
            debug_assert_eq!(source, "token_embd");
            let alias = GpuTensor {
                buf: unsafe { token_embd.buf.alias() },
                shape: token_embd.shape.clone(),
                dtype: token_embd.dtype,
            };
            let sidecar = output_sidecar_slot
                .map(|slot| tensor_handle(&mut taken, slot))
                .and_then(|handle| match handle {
                    WeightHandle::Resident(t) => Some(t),
                    WeightHandle::Alias(_) => token_sidecar_slot
                        .map(|slot| tensor_handle(&mut taken, slot))
                        .and_then(|handle| match handle {
                            WeightHandle::Resident(t) => Some(t),
                            WeightHandle::Alias(_) => None,
                        }),
                });
            (
                WeightTensor {
                    buf: alias,
                    gpu_dtype: hipfire_runtime::weight_backend::embedding_format_dtype(embd_format),
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: sidecar,
                },
                true,
            )
        }
        WeightHandle::Resident(buf) => {
            let shape = [config.vocab_size, config.dim];
            let sidecar = output_sidecar_slot.map(|slot| match tensor_handle(&mut taken, slot) {
                WeightHandle::Resident(t) => t,
                WeightHandle::Alias(_) => panic!("untied lm_head sidecar was an alias"),
            });
            (
                tensor_from_handle(WeightHandle::Resident(buf), &shape, sidecar, None),
                false,
            )
        }
    };
    if let Some(slot) = token_sidecar_slot {
        if !keep_token_sidecar {
            let _ = tensor_handle(&mut taken, slot);
        }
    }
    let output_norm = gpu_handle(&mut taken, slot("output_norm", None));

    let mut layers = Vec::with_capacity(config.n_layers);
    let mut derived_iter = derived.tensors.drain(..);
    for layer in 0..config.n_layers {
        let attn_norm = gpu_handle(&mut taken, slot("attn_norm", Some(layer)));
        let ffn_norm = gpu_handle(&mut taken, slot("ffn_norm", Some(layer)));
        let d = config.dim;
        let is_moe = config.num_experts > 0;
        let layer_value = match (config.layer_types[layer], is_moe) {
            (LayerType::LinearAttention, false) => LayerWeights::DeltaNet(DeltaNetLayerWeights {
                attn_norm,
                wqkv: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wqkv",
                    layer,
                    vec![
                        config.linear_num_key_heads * config.linear_key_head_dim * 2
                            + config.linear_num_value_heads * config.linear_value_head_dim,
                        d,
                    ],
                ),
                wz: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wz",
                    layer,
                    vec![
                        config.linear_num_value_heads * config.linear_value_head_dim,
                        d,
                    ],
                ),
                w_alpha: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "w_alpha",
                    layer,
                    vec![config.linear_num_value_heads, d],
                ),
                w_beta: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "w_beta",
                    layer,
                    vec![config.linear_num_value_heads, d],
                ),
                a_log: gpu_handle(&mut taken, slot("a_log", Some(layer))),
                dt_bias: gpu_handle(&mut taken, slot("dt_bias", Some(layer))),
                conv_weight: gpu_handle(&mut taken, slot("conv", Some(layer))),
                norm_weight: gpu_handle(&mut taken, slot("norm", Some(layer))),
                wo: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wo",
                    layer,
                    vec![
                        d,
                        config.linear_num_value_heads * config.linear_value_head_dim,
                    ],
                ),
                ffn_norm,
                w_gate: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "ffn_gate",
                    layer,
                    vec![config.hidden_dim, d],
                ),
                w_up: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "ffn_up",
                    layer,
                    vec![config.hidden_dim, d],
                ),
                w_down: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "ffn_down",
                    layer,
                    vec![d, config.hidden_dim],
                ),
            }),
            (LayerType::FullAttention, false) => LayerWeights::FullAttn(FullAttnLayerWeights {
                attn_norm,
                wq: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wq",
                    layer,
                    vec![2 * config.n_heads * config.head_dim, d],
                ),
                wk: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wk",
                    layer,
                    vec![config.n_kv_heads * config.head_dim, d],
                ),
                wv: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wv",
                    layer,
                    vec![config.n_kv_heads * config.head_dim, d],
                ),
                wo: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "wo",
                    layer,
                    vec![d, config.n_heads * config.head_dim],
                ),
                q_norm: gpu_handle(&mut taken, slot("q_norm", Some(layer))),
                k_norm: gpu_handle(&mut taken, slot("k_norm", Some(layer))),
                ffn_norm,
                w_gate: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "ffn_gate",
                    layer,
                    vec![config.hidden_dim, d],
                ),
                w_up: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "ffn_up",
                    layer,
                    vec![config.hidden_dim, d],
                ),
                w_down: typed_weight(
                    &mut taken,
                    &slots_by_key,
                    "ffn_down",
                    layer,
                    vec![d, config.hidden_dim],
                ),
            }),
            (LayerType::LinearAttention, true) => {
                let plan = &derived_plans[layer];
                LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                    attn_norm,
                    wqkv: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wqkv",
                        layer,
                        vec![
                            config.linear_num_key_heads * config.linear_key_head_dim * 2
                                + config.linear_num_value_heads * config.linear_value_head_dim,
                            d,
                        ],
                    ),
                    wz: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wz",
                        layer,
                        vec![
                            config.linear_num_value_heads * config.linear_value_head_dim,
                            d,
                        ],
                    ),
                    w_alpha: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "w_alpha",
                        layer,
                        vec![config.linear_num_value_heads, d],
                    ),
                    w_beta: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "w_beta",
                        layer,
                        vec![config.linear_num_value_heads, d],
                    ),
                    a_log: gpu_handle(&mut taken, slot("a_log", Some(layer))),
                    dt_bias: gpu_handle(&mut taken, slot("dt_bias", Some(layer))),
                    conv_weight: gpu_handle(&mut taken, slot("conv", Some(layer))),
                    norm_weight: gpu_handle(&mut taken, slot("norm", Some(layer))),
                    wo: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wo",
                        layer,
                        vec![
                            d,
                            config.linear_num_value_heads * config.linear_value_head_dim,
                        ],
                    ),
                    ffn_norm,
                    ffn: typed_moe_ffn(
                        &mut taken,
                        &slots_by_key,
                        config,
                        layer,
                        derived_iter.next().expect("gate pointer record"),
                        derived_iter.next().expect("down pointer record"),
                        plan.has_down_awq
                            .then(|| derived_iter.next().expect("AWQ pointer record")),
                        plan.dtype_tags
                            .as_ref()
                            .map(|_| derived_iter.next().expect("dtype tag record")),
                    ),
                })
            }
            (LayerType::FullAttention, true) => {
                let plan = &derived_plans[layer];
                LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                    attn_norm,
                    wq: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wq",
                        layer,
                        vec![2 * config.n_heads * config.head_dim, d],
                    ),
                    wk: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wk",
                        layer,
                        vec![config.n_kv_heads * config.head_dim, d],
                    ),
                    wv: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wv",
                        layer,
                        vec![config.n_kv_heads * config.head_dim, d],
                    ),
                    wo: typed_weight(
                        &mut taken,
                        &slots_by_key,
                        "wo",
                        layer,
                        vec![d, config.n_heads * config.head_dim],
                    ),
                    q_norm: gpu_handle(&mut taken, slot("q_norm", Some(layer))),
                    k_norm: gpu_handle(&mut taken, slot("k_norm", Some(layer))),
                    ffn_norm,
                    ffn: typed_moe_ffn(
                        &mut taken,
                        &slots_by_key,
                        config,
                        layer,
                        derived_iter.next().expect("gate pointer record"),
                        derived_iter.next().expect("down pointer record"),
                        plan.has_down_awq
                            .then(|| derived_iter.next().expect("AWQ pointer record")),
                        plan.dtype_tags
                            .as_ref()
                            .map(|_| derived_iter.next().expect("dtype tag record")),
                    ),
                })
            }
        };
        layers.push(layer_value);
    }
    debug_assert!(
        taken.iter().all(Option::is_none),
        "preflighted Qwen35 assembly left unconsumed records"
    );
    Ok(Qwen35Weights {
        token_embd,
        embd_format,
        output_norm,
        output,
        moe_has_mq6: layers.iter().any(|layer| match layer {
            LayerWeights::DeltaNetMoe(l) => l.ffn.experts.iter().any(|e| {
                e.gate_up.gpu_dtype == DType::MQ6G256 || e.down.gpu_dtype == DType::MQ6G256
            }),
            LayerWeights::FullAttnMoe(l) => l.ffn.experts.iter().any(|e| {
                e.gate_up.gpu_dtype == DType::MQ6G256 || e.down.gpu_dtype == DType::MQ6G256
            }),
            _ => false,
        }),
        layers,
        pager: None,
        lm_head_aliases_embd,
    })
}

/// Production HFQ loader: config/manifest validation happens before the first
/// payload read, fulfillment owns upload/rollback, and typed assembly is the
/// only bridge into the forward structs.
pub fn load_qwen35_hfq_weights(
    hfq: &HfqFile,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> Result<Qwen35Weights, String> {
    let resolver = Qwen35SourceResolver::new(hfq, config);
    let manifest = resolver.manifest_with_companions(&Qwen35::weight_manifest(config))?;
    let mut store = hipfire_runtime::weight_store::fulfill_manifest_gpu(
        &manifest,
        &DeviceMesh::single(),
        config.n_layers,
        gpu,
        |entry| {
            let (bytes, dtype) = resolver.resolve_for_store(entry)?;
            Ok((bytes, dtype))
        },
    )
    .map_err(|e| format!("qwen35 HFQ fulfillment: {e:?}"))?;
    assemble_qwen35_weights(&mut store, config, &manifest, gpu)
}

/// Production ParoQuant directory loader using the same resolver/fulfillment /
/// transactional assembler as HFQ.  The resolver is the only format-specific
/// part: it repacks AWQ payloads and exposes rotation records as manifest cells.
pub fn load_qwen35_paro_weights(
    source: &dyn ModelSource,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> Result<Qwen35Weights, String> {
    let resolver = Qwen35ParoSourceResolver::new(source, config)?;
    let manifest = resolver.manifest_with_source_records(&Qwen35::weight_manifest(config))?;
    let mut store = hipfire_runtime::weight_store::fulfill_manifest_gpu(
        &manifest,
        &DeviceMesh::single(),
        config.n_layers,
        gpu,
        |entry| {
            let resolved = resolver.resolve(entry)?;
            Ok((resolved.bytes, resolved.dtype))
        },
    )
    .map_err(|e| format!("qwen35 Paro fulfillment: {e:?}"))?;
    assemble_qwen35_weights(&mut store, config, &manifest, gpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Qwen35;
    use hipfire_hardware::DeviceMesh;
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::model_source::{ModelSource, QuantConfig, TensorInfo};
    use hipfire_runtime::weight_store::fulfill_manifest_gpu;
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct FakeParoTensor {
        info: TensorInfo,
        data: Vec<u8>,
    }

    struct FakeParoSource {
        tensors: HashMap<String, FakeParoTensor>,
        reads: RefCell<HashMap<String, usize>>,
        path: PathBuf,
    }

    impl FakeParoSource {
        fn new() -> Self {
            let mut source = Self {
                tensors: HashMap::new(),
                reads: RefCell::new(HashMap::new()),
                path: PathBuf::from("/fake-paro"),
            };
            source.add("model.embed_tokens.weight", "F16", vec![8, 8]);
            source
        }

        fn add(&mut self, name: &str, dtype: &str, shape: Vec<usize>) {
            self.tensors.insert(
                name.to_string(),
                FakeParoTensor {
                    info: TensorInfo {
                        name: name.to_string(),
                        dtype: dtype.to_string(),
                        shape,
                        quant_type: 0xff,
                        data_offset: 0,
                        data_size: 8,
                    },
                    data: vec![0u8; 8],
                },
            );
        }

        fn read_count(&self, name: &str) -> usize {
            self.reads.borrow().get(name).copied().unwrap_or(0)
        }
    }

    impl ModelSource for FakeParoSource {
        fn metadata_json(&self) -> &str {
            "{}"
        }

        fn arch_id(&self) -> u32 {
            5
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            static CONFIG: QuantConfig = QuantConfig {
                method: String::new(),
                bits: 4,
                group_size: 128,
                krot: 8,
                dynamic_excludes: Vec::new(),
            };
            Some(&CONFIG)
        }

        fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
            *self.reads.borrow_mut().entry(name.to_string()).or_default() += 1;
            let tensor = self.tensors.get(name)?;
            Some((&tensor.info, &tensor.data))
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name).map(|tensor| &tensor.info)
        }

        fn tensor_names(&self) -> Vec<&str> {
            self.tensors.keys().map(String::as_str).collect()
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    fn paro_owner_entry() -> WeightEntry {
        WeightEntry::layer(
            "shared_gate",
            0,
            vec![4, 8],
            DType::F16,
            ShardPolicy::Replicate,
        )
    }

    fn add_paro_owner(source: &mut FakeParoSource, sidecars: &[&str]) {
        let base = "model.layers.0.mlp.shared_expert.gate_proj";
        source.add(&format!("{base}.qweight"), "I32", vec![4, 8]);
        source.add(&format!("{base}.qzeros"), "I32", vec![1, 1]);
        source.add(&format!("{base}.scales"), "F16", vec![1, 4]);
        for suffix in sidecars {
            source.add(&format!("{base}.{suffix}"), "Raw", vec![1]);
        }
    }

    fn test_config(layer_types: &[&str], moe: bool) -> Qwen35Config {
        let mut value = serde_json::json!({
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": layer_types.len(),
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 8,
            "layer_types": layer_types,
            "tie_word_embeddings": true
        });
        if moe {
            value["num_experts"] = serde_json::json!(2);
            value["num_experts_per_tok"] = serde_json::json!(1);
            value["moe_intermediate_size"] = serde_json::json!(4);
            value["shared_expert_intermediate_size"] = serde_json::json!(4);
        }
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": value}).to_string())
            .unwrap()
    }

    #[test]
    fn qtype_mapping_keeps_source_dtype_and_layout_distinct() {
        assert_eq!(
            qtype_dtype(13),
            Some((DType::MQ4G256, Qwen35SourceLayout::Raw))
        );
        assert_eq!(qtype_dtype(1), Some((DType::F16, Qwen35SourceLayout::F16)));
        assert_eq!(
            qtype_dtype(16),
            Some((DType::BF16, Qwen35SourceLayout::BF16))
        );
        assert_eq!(qtype_dtype(0xfe), None);
    }

    #[test]
    fn source_constraint_is_checked_against_source_dtype() {
        let exact = DTypeConstraint::source_exact(DType::F16);
        assert!(source_allowed(&exact, DType::F16));
        assert!(!source_allowed(&exact, DType::F32));
        let one = DTypeConstraint::source_from_sources(vec![DType::F16, DType::Q8_0]);
        assert!(source_allowed(&one, DType::Q8_0));
        assert!(!source_allowed(&one, DType::MQ4G256));
    }

    #[test]
    fn companion_name_is_not_a_main_tensor() {
        assert_eq!(sidecar_name("expert.3.down"), "expert.3.down.awq_scale");
        assert_eq!(
            awq_companion_physical("model.language_model.layers.0.mlp.gate_proj.weight"),
            "model.language_model.layers.0.mlp.gate_proj.awq_scale.weight"
        );
        assert!(!awq_companion_physical("x.weight").contains("weight.awq_scale"));
    }

    #[test]
    fn paro_manifest_uses_legacy_scalar_gate_order() {
        let config = test_config(&["full_attention"], true);
        let hfq_order: Vec<_> = Qwen35::weight_manifest(&config)
            .into_iter()
            .filter(|entry| entry.layer == Some(0))
            .map(|entry| entry.name)
            .collect();
        let paro_order: Vec<_> = paro_source_order(&Qwen35::weight_manifest(&config))
            .into_iter()
            .filter(|entry| entry.layer == Some(0))
            .map(|entry| entry.name)
            .collect();
        let hfq_shared = hfq_order
            .iter()
            .position(|name| name == "shared_gate")
            .unwrap();
        let paro_shared = paro_order
            .iter()
            .position(|name| name == "shared_expert_gate")
            .unwrap();
        assert_eq!(
            &paro_order[paro_shared..paro_shared + 4],
            [
                "shared_expert_gate",
                "shared_gate",
                "shared_up",
                "shared_down"
            ]
        );
        assert_eq!(
            &hfq_order[hfq_shared..hfq_shared + 4],
            [
                "shared_gate",
                "shared_up",
                "shared_down",
                "shared_expert_gate"
            ]
        );
    }

    #[test]
    fn paro_manifest_propagates_missing_owner_metadata() {
        let source = FakeParoSource::new();
        let config = test_config(&["full_attention"], true);
        let resolver = Qwen35ParoSourceResolver::new(&source, &config).unwrap();
        let error = resolver
            .manifest_with_source_records(&[paro_owner_entry()])
            .unwrap_err();
        assert!(error.contains("no tensor for 'shared_gate'"), "{error}");
    }

    #[test]
    fn paro_manifest_propagates_missing_required_sidecar() {
        let mut source = FakeParoSource::new();
        add_paro_owner(&mut source, &["pairs", "theta"]);
        let config = test_config(&["full_attention"], true);
        let resolver = Qwen35ParoSourceResolver::new(&source, &config).unwrap();
        let error = resolver
            .manifest_with_source_records(&[paro_owner_entry()])
            .unwrap_err();
        assert!(
            error.contains("required sidecar") && error.contains("channel_scales"),
            "{error}"
        );
    }

    #[test]
    fn paro_sidecar_materialization_uses_cached_owner_record_once() {
        let mut source = FakeParoSource::new();
        add_paro_owner(&mut source, &["pairs", "theta", "channel_scales"]);
        let config = test_config(&["full_attention"], true);
        let resolver = Qwen35ParoSourceResolver::new(&source, &config).unwrap();
        let manifest = resolver
            .manifest_with_source_records(&[paro_owner_entry()])
            .unwrap();
        let owner = manifest
            .iter()
            .find(|entry| entry.name == "shared_gate")
            .unwrap();
        resolver.resolve(owner).unwrap();
        let sidecar = manifest
            .iter()
            .find(|entry| entry.name == "shared_gate.paro_pairs")
            .unwrap();
        let resolved = resolver.resolve(sidecar).unwrap();
        assert_eq!(
            resolved.physical_name,
            "model.layers.0.mlp.shared_expert.gate_proj.pairs"
        );
        assert_eq!(source.read_count(&resolved.physical_name), 1);
        assert_eq!(
            source.read_count("model.layers.0.mlp.shared_expert.gate_proj.qweight"),
            1
        );
        assert_eq!(
            source.read_count("model.layers.0.mlp.shared_expert.gate_proj.qzeros"),
            1
        );
        assert_eq!(
            source.read_count("model.layers.0.mlp.shared_expert.gate_proj.scales"),
            1
        );
    }

    #[test]
    fn source_layouts_are_preserved_until_legacy_conversion() {
        let norm = ResolvedQwen35Source {
            logical_name: "attn_norm".into(),
            physical_name: "x".into(),
            bytes: 1.0f32.to_le_bytes().to_vec(),
            dtype: DType::F32,
            layout: Qwen35SourceLayout::F32,
            shape: vec![1],
            companion: false,
        };
        let awq = ResolvedQwen35Source {
            logical_name: "ffn_gate.awq_scale".into(),
            physical_name: "ffn_gate.awq_scale.weight".into(),
            bytes: 0x3c00u16.to_le_bytes().to_vec(),
            dtype: DType::F16,
            layout: Qwen35SourceLayout::F16,
            shape: vec![1],
            companion: true,
        };
        assert_eq!(norm.layout, Qwen35SourceLayout::F32);
        assert_eq!(awq.layout, Qwen35SourceLayout::F16);
        assert_eq!(qtype_dtype(2).unwrap().1, norm.layout);
        assert_eq!(qtype_dtype(1).unwrap().1, awq.layout);
    }

    #[test]
    fn tied_awq_embedding_source_is_widened_before_forward_use() {
        let entry = WeightEntry::model(
            "token_embd",
            vec![8, 256],
            DType::F16,
            ShardPolicy::Pin(hipfire_runtime::weight_manifest::PinTarget::Embed),
        );
        assert!(should_widen_to_f32(&entry, DType::MQ4G256));
    }

    #[test]
    fn typed_embedding_validation_rejects_unsupported_forward_dtypes() {
        assert!(validate_typed_embedding_dtype(DType::F32).is_ok());
        assert!(validate_typed_embedding_dtype(DType::MQ4G256).is_err());
        assert!(validate_typed_embedding_dtype(DType::MQ3G256).is_err());
    }

    #[test]
    fn replacement_guard_frees_new_tensor_when_old_free_is_injected_to_fail() {
        use std::cell::Cell;
        use std::rc::Rc;

        let replacement_frees = Rc::new(Cell::new(0));
        let result: Result<(), &str> = {
            let _replacement = ReplacementGuard::new(GpuTensor::null_for_test(), {
                let replacement_frees = replacement_frees.clone();
                move |_tensor| replacement_frees.set(replacement_frees.get() + 1)
            });
            let injected_old_free: Result<(), &str> = Err("injected old-buffer free failure");
            injected_old_free
        };
        assert!(result.is_err());
        assert_eq!(replacement_frees.get(), 1);
    }

    #[test]
    fn manifest_validation_covers_dense_moe_and_mixed_topologies() {
        for (layers, moe) in [
            (&["full_attention"][..], false),
            (&["full_attention"][..], true),
            (&["full_attention", "linear_attention"][..], false),
        ] {
            let config = test_config(layers, moe);
            let manifest = Qwen35::weight_manifest(&config);
            validate_manifest_schema(&config, &manifest).unwrap();
        }
        let config = test_config(&["full_attention"], false);
        let mut malformed = Qwen35::weight_manifest(&config);
        let entry = malformed
            .iter_mut()
            .find(|entry| entry.name == "wq")
            .unwrap();
        entry.logical_shape[0] += 1;
        let error = validate_manifest_schema(&config, &malformed).unwrap_err();
        assert!(error.contains("non-canonical manifest metadata"));
        let mut wrong_policy = Qwen35::weight_manifest(&config);
        let entry = wrong_policy
            .iter_mut()
            .find(|entry| entry.name == "wq")
            .unwrap();
        entry.policy = ShardPolicy::ColumnShard { axis: 0 };
        assert!(validate_manifest_schema(&config, &wrong_policy).is_err());
    }

    #[test]
    fn mixed_moe_dtype_tags_match_legacy_dispatch_tags() {
        assert_eq!(dtype_tag(DType::MQ6G256, DType::MQ6G256), 0);
        assert_eq!(dtype_tag(DType::MQ4G256, DType::MQ2G256Lloyd), 1);
        assert_eq!(dtype_tag(DType::MQ4G256, DType::MQ3G256Lloyd), 3);
        assert_eq!(dtype_tag(DType::MFP4G32E8, DType::MFP4G32E8), 4);
    }

    #[test]
    fn canonical_conv_accepts_flattened_mq4_physical_shape() {
        let entry = WeightEntry::layer("conv", 0, vec![24], DType::F32, ShardPolicy::Replicate);
        assert!(source_shape_matches(&entry, &[6, 1, 4], 13));
        assert!(!source_shape_matches(&entry, &[6, 4, 1], 13));
        assert!(!source_shape_matches(&entry, &[6, 1, 4], 1));
    }

    #[test]
    fn generated_companions_preserve_tied_and_source_metadata() {
        let owner = WeightEntry::model(
            "lm_head",
            vec![32, 16],
            DType::F16,
            ShardPolicy::Tied {
                source: "token_embd".into(),
            },
        )
        .with_placement(hipfire_runtime::weight_manifest::PlacementHint::Pin(
            hipfire_runtime::weight_manifest::PinTarget::Output,
        ));
        let companion = expected_companion_entry(&owner);
        assert_eq!(companion.name, "lm_head.awq_scale");
        assert_eq!(companion.logical_shape, vec![16]);
        assert_eq!(companion.dtype, DType::F32);
        assert_eq!(
            companion.dtype_constraint,
            DTypeConstraint::source_exact(DType::F16)
        );
        assert_eq!(
            companion.policy,
            ShardPolicy::Tied {
                source: "token_embd.awq_scale".into()
            }
        );
        assert_eq!(companion.placement, owner.placement);
    }

    fn synthetic_store(config: &Qwen35Config, gpu: &mut Gpu) -> (Vec<WeightEntry>, WeightStore) {
        let manifest = Qwen35::weight_manifest(config);
        let store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            gpu,
            |entry| {
                let n = entry.logical_shape.iter().product::<usize>();
                Ok((vec![0u8; n * 4], DType::F32))
            },
        )
        .unwrap();
        (manifest, store)
    }

    #[test]
    #[ignore = "requires an AMD GPU; successful synthetic dense assembly"]
    fn synthetic_dense_assembly_is_forward_ready() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let config = test_config(&["full_attention", "linear_attention"], false);
        let mut gpu = Gpu::init().expect("GPU required for synthetic assembly");
        let (manifest, mut store) = synthetic_store(&config, &mut gpu);
        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.token_embd.dtype, DType::F32);
        assert!(weights.lm_head_aliases_embd);
        weights.free_gpu(&mut gpu);
    }

    #[test]
    #[ignore = "requires an AMD GPU; successful model-level AWQ assembly"]
    fn synthetic_dense_assembly_attaches_model_awq_companion() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let mut config = test_config(&["full_attention"], false);
        config.tie_word_embeddings = false;
        let mut gpu = Gpu::init().expect("GPU required for synthetic assembly");
        let mut manifest = Qwen35::weight_manifest(&config);
        let lm_head = manifest
            .iter()
            .find(|entry| entry.name == "lm_head")
            .cloned()
            .unwrap();
        manifest.push(expected_companion_entry(&lm_head));
        let mut store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |entry| {
                let n = entry.logical_shape.iter().product::<usize>();
                let dtype = if entry.name == "lm_head" {
                    DType::MQ4G256
                } else if entry.name.ends_with(AWQ_SUFFIX) {
                    DType::F16
                } else if entry.name == "token_embd" || entry.name == "wq" {
                    DType::F16
                } else if entry.name == "wk" {
                    DType::BF16
                } else {
                    DType::F32
                };
                let bytes = if dtype == DType::MQ4G256 {
                    vec![0u8; 136]
                } else {
                    vec![0u8; n * if dtype == DType::F16 { 2 } else { 4 }]
                };
                Ok((bytes, dtype))
            },
        )
        .unwrap();
        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        assert!(weights.output.awq_scale.is_some());
        assert!(!weights.lm_head_aliases_embd);
        assert_eq!(weights.token_embd.dtype, DType::F32);
        match &weights.layers[0] {
            LayerWeights::FullAttn(layer) => {
                assert_eq!(layer.wq.gpu_dtype, DType::F32);
                assert_eq!(layer.wk.gpu_dtype, DType::F32);
            }
            _ => panic!("expected full-attention layer"),
        }
        weights.free_gpu(&mut gpu);
    }

    #[test]
    #[ignore = "requires an AMD GPU; deterministic tied embedding/lm_head AWQ fixture"]
    fn synthetic_tied_embedding_lm_head_awq_assembly_owns_alias_and_sidecar_once() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let metadata = serde_json::json!({
            "config": {
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "vocab_size": 8,
                "layer_types": ["full_attention"],
                "tie_word_embeddings": true
            }
        });
        let config = crate::qwen35::config_from_metadata_json(&metadata.to_string()).unwrap();
        let mut gpu = Gpu::init().expect("GPU required for deterministic tied AWQ fixture");
        let mut manifest = Qwen35::weight_manifest(&config);
        let token = manifest
            .iter()
            .find(|entry| entry.name == "token_embd")
            .cloned()
            .unwrap();
        let lm_head = manifest
            .iter()
            .find(|entry| entry.name == "lm_head")
            .cloned()
            .unwrap();
        manifest.push(expected_companion_entry(&token));
        manifest.push(expected_companion_entry(&lm_head));

        let mut store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |entry| {
                let n = entry.logical_shape.iter().product::<usize>();
                if entry.name == "token_embd" {
                    assert_eq!(n % 256, 0);
                    return Ok((vec![0u8; (n / 256) * 136], DType::MQ4G256));
                }
                if entry.name.ends_with(AWQ_SUFFIX) {
                    return Ok((vec![0u8; n * 2], DType::F16));
                }
                Ok((vec![0u8; n * 4], DType::F32))
            },
        )
        .unwrap();

        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        assert!(
            store.is_empty(),
            "successful finalization must drain the store"
        );
        assert!(weights.lm_head_aliases_embd);
        assert_eq!(
            weights.output.buf.buf.as_ptr(),
            weights.token_embd.buf.as_ptr()
        );
        let sidecar = weights
            .output
            .awq_scale
            .as_ref()
            .expect("tied lm_head must retain its AWQ companion");
        assert_eq!(sidecar.shape, vec![config.dim]);
        assert_eq!(sidecar.dtype, DType::F32);

        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
    }

    #[test]
    #[ignore = "requires an AMD GPU; successful synthetic MoE assembly"]
    fn synthetic_moe_assembly_builds_pointer_tables_and_tags() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let config = test_config(&["linear_attention", "full_attention"], true);
        let mut gpu = Gpu::init().expect("GPU required for synthetic assembly");
        let (manifest, mut store) = synthetic_store(&config, &mut gpu);
        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        assert_eq!(weights.layers.len(), 2);
        for layer in &weights.layers {
            match layer {
                LayerWeights::DeltaNetMoe(layer) => {
                    assert_eq!(layer.ffn.expert_gate_up_ptrs.dtype, DType::Raw);
                    assert_eq!(
                        layer.ffn.expert_gate_up_ptrs.shape,
                        vec![config.num_experts * 8]
                    );
                    assert!(layer.ffn.expert_down_awq_ptrs.is_none());
                }
                LayerWeights::FullAttnMoe(layer) => {
                    assert_eq!(layer.ffn.expert_down_ptrs.dtype, DType::Raw);
                    assert_eq!(
                        layer.ffn.expert_down_ptrs.shape,
                        vec![config.num_experts * 8]
                    );
                    assert!(layer.ffn.expert_dtype_tags.is_none());
                }
                _ => panic!("expected MoE layer"),
            }
        }
        weights.free_gpu(&mut gpu);
    }

    #[test]
    #[ignore = "requires an AMD GPU; successful mixed-expert AWQ/tag assembly"]
    fn synthetic_mixed_moe_assembly_keeps_tags_and_awq_pointers_byte_exact() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let config = test_config(&["linear_attention", "full_attention"], true);
        let mut gpu = Gpu::init().expect("GPU required for synthetic assembly");
        let mut manifest = Qwen35::weight_manifest(&config);
        let down_entries: Vec<_> = manifest
            .iter()
            .filter(|entry| entry.name.starts_with("expert.") && entry.name.ends_with(".down"))
            .cloned()
            .collect();
        manifest.extend(down_entries.iter().map(expected_companion_entry));
        let mut store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |entry| {
                let dtype = match entry.name.as_str() {
                    "expert.0.gate_up" | "expert.1.gate_up" => DType::MQ4G256,
                    "expert.0.down" => DType::MQ2G256Lloyd,
                    "expert.1.down" => DType::MQ3G256Lloyd,
                    name if name.ends_with(AWQ_SUFFIX) => DType::F16,
                    _ => DType::F32,
                };
                let n = entry.logical_shape.iter().product::<usize>();
                let bytes = match dtype {
                    DType::MQ4G256 => vec![0; 136],
                    DType::MQ2G256Lloyd => vec![0; 72],
                    DType::MQ3G256Lloyd => vec![0; 112],
                    DType::F16 => vec![0; n * 2],
                    _ => vec![0; n * 4],
                };
                Ok((bytes, dtype))
            },
        )
        .unwrap();
        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        for layer in &weights.layers {
            let ffn = match layer {
                LayerWeights::DeltaNetMoe(layer) => &layer.ffn,
                LayerWeights::FullAttnMoe(layer) => &layer.ffn,
                _ => panic!("expected MoE layer"),
            };
            let tags = ffn.expert_dtype_tags.as_ref().expect("mixed tags");
            assert_eq!(tags.dtype, DType::Raw);
            assert_eq!(tags.shape, vec![config.num_experts]);
            assert_eq!(ffn.expert_gate_up_ptrs.dtype, DType::Raw);
            assert_eq!(ffn.expert_gate_up_ptrs.shape, vec![config.num_experts * 8]);
            assert_eq!(ffn.expert_down_ptrs.dtype, DType::Raw);
            assert_eq!(ffn.expert_down_ptrs.shape, vec![config.num_experts * 8]);
            let awq = ffn.expert_down_awq_ptrs.as_ref().expect("AWQ pointers");
            assert_eq!(awq.dtype, DType::Raw);
            assert_eq!(awq.shape, vec![config.num_experts * 8]);
        }
        weights.free_gpu(&mut gpu);
    }

    #[test]
    #[ignore = "requires an AMD GPU; verifies typed assembly rollback after commit"]
    fn typed_assembly_failure_after_commit_publishes_no_partial_model() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let metadata = serde_json::json!({
            "config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 8,
                "layer_types": ["full_attention"],
                "tie_word_embeddings": true
            }
        });
        let config = crate::qwen35::config_from_metadata_json(&metadata.to_string()).unwrap();
        let manifest = <Qwen35 as Architecture>::weight_manifest(&config);
        let mut gpu = Gpu::init().expect("GPU required for ignored rollback test");
        let mut store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |entry| {
                let elems = entry.logical_shape.iter().product::<usize>();
                Ok((vec![0; elems * 4], DType::F32))
            },
        )
        .unwrap();

        let result = assemble_qwen35_weights_inner(&mut store, &config, &manifest, &mut gpu, true);
        assert!(result.is_err());
        assert!(
            store.is_empty(),
            "rollback must free taken and untaken residents"
        );
    }

    #[test]
    #[ignore = "requires an AMD GPU and HIPFIRE_QWEN35_HFQ fixture path"]
    fn actual_hfq_source_bytes_and_dtype_survive_store_upload() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let path = match std::env::var("HIPFIRE_QWEN35_HFQ") {
            Ok(path) => path,
            Err(_) => return,
        };
        let mut hfq = HfqFile::open(std::path::Path::new(&path)).unwrap();
        let config = crate::qwen35::config_from_hfq(&hfq).unwrap();
        let manifest = <Qwen35 as Architecture>::weight_manifest(&config);
        let token_entry = manifest
            .iter()
            .find(|entry| entry.name == "token_embd" && entry.layer.is_none())
            .unwrap();
        let norm_entry = manifest
            .iter()
            .find(|entry| entry.name == "output_norm" && entry.layer.is_none())
            .unwrap();
        let raw_entry = manifest
            .iter()
            .find(|entry| entry.name == "a_log" && entry.layer.is_some());
        let conv_entry = manifest
            .iter()
            .find(|entry| entry.name == "conv" && entry.layer.is_some());
        let mut entries = vec![token_entry.clone(), norm_entry.clone()];
        if let Some(entry) = raw_entry {
            entries.push(entry.clone());
        }
        if let Some(entry) = conv_entry {
            entries.push(entry.clone());
        }
        let resolver = Qwen35SourceResolver::new(&hfq, &config);
        let expected: Vec<_> = entries
            .iter()
            .map(|entry| {
                let (bytes, dtype) = resolver.resolve_for_store(entry).unwrap();
                (entry.name.clone(), entry.layer, bytes, dtype)
            })
            .collect();
        let mut gpu = Gpu::init().expect("GPU required for ignored source test");
        let mut store = fulfill_manifest_gpu(
            &entries,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |candidate| {
                let (bytes, dtype) = resolver.resolve_for_store(candidate).unwrap();
                Ok((bytes, dtype))
            },
        )
        .unwrap();
        for (name, layer, expected_bytes, expected_dtype) in expected {
            let tensor = match store.take(&name, layer, 0).unwrap() {
                WeightHandle::Resident(tensor) => tensor,
                WeightHandle::Alias(_) => panic!("fixture probe selected an alias"),
            };
            assert_eq!(tensor.dtype, expected_dtype, "{name}");
            let mut actual = vec![0u8; expected_bytes.len()];
            gpu.hip.memcpy_dtoh(&mut actual, &tensor.buf).unwrap();
            assert_eq!(actual, expected_bytes, "{name}");
            gpu.free_tensor(tensor).unwrap();
        }
        if let Some(entry) = conv_entry {
            let resolved = resolver.resolve(entry).unwrap();
            assert_eq!(resolved.logical_name, "conv");
            if resolved.dtype == DType::MQ4G256 {
                assert_eq!(resolved.shape.len(), 3);
                assert_eq!(resolved.shape[1..], [1, 4]);
            }
        }
        hfq.drop_mmap();
    }

    #[test]
    #[ignore = "requires AMD GPU and HIPFIRE_QWEN35_HFQ; covers full HFQ assembly"]
    fn full_hfq_fixture_assembles_conv_awq_and_moe_derived_records() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let path = match std::env::var("HIPFIRE_QWEN35_HFQ") {
            Ok(path) => path,
            Err(_) => return,
        };
        let mut hfq = HfqFile::open(std::path::Path::new(&path)).unwrap();
        let config = crate::qwen35::config_from_hfq(&hfq).unwrap();
        let resolver = Qwen35SourceResolver::new(&hfq, &config);
        let manifest = resolver
            .manifest_with_companions(&Qwen35::weight_manifest(&config))
            .unwrap();
        let has_model_awq = manifest
            .iter()
            .any(|entry| entry.name == "token_embd.awq_scale" || entry.name == "lm_head.awq_scale");
        let has_moe_awq = manifest.iter().any(|entry| {
            entry.name.starts_with("expert.") && entry.name.ends_with("down.awq_scale")
        });
        let mut gpu = Gpu::init().expect("GPU required for full HFQ assembly");
        let mut store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |entry| {
                let (bytes, dtype) = resolver.resolve_for_store(entry).unwrap();
                Ok((bytes, dtype))
            },
        )
        .unwrap();
        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        assert_eq!(weights.layers.len(), config.n_layers);
        if has_model_awq {
            assert!(weights.output.awq_scale.is_some());
        }
        if config.num_experts > 0 && has_moe_awq {
            for layer in &weights.layers {
                match layer {
                    LayerWeights::DeltaNetMoe(layer) => {
                        assert!(layer.ffn.expert_down_awq_ptrs.is_some());
                    }
                    LayerWeights::FullAttnMoe(layer) => {
                        assert!(layer.ffn.expert_down_awq_ptrs.is_some());
                    }
                    _ => {}
                }
            }
        }
        weights.free_gpu(&mut gpu);
        hfq.drop_mmap();
    }

    #[test]
    #[ignore = "requires AMD GPU and HIPFIRE_QWEN35_HFQ; tied lm_head AWQ ownership"]
    fn real_fixture_tied_lm_head_awq_assembly_preserves_alias_ownership() {
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        let path = match std::env::var("HIPFIRE_QWEN35_HFQ") {
            Ok(path) => path,
            Err(_) => return,
        };
        let mut hfq = HfqFile::open(std::path::Path::new(&path)).unwrap();
        let config = crate::qwen35::config_from_hfq(&hfq).unwrap();
        if !config.tie_word_embeddings {
            return;
        }
        let resolver = Qwen35SourceResolver::new(&hfq, &config);
        let manifest = resolver
            .manifest_with_companions(&Qwen35::weight_manifest(&config))
            .unwrap();
        if !manifest
            .iter()
            .any(|entry| entry.name == "lm_head.awq_scale")
        {
            return;
        }
        let mut gpu = Gpu::init().expect("GPU required for tied AWQ fixture test");
        let mut store = fulfill_manifest_gpu(
            &manifest,
            &DeviceMesh::single(),
            config.n_layers,
            &mut gpu,
            |entry| {
                let (bytes, dtype) = resolver.resolve_for_store(entry).unwrap();
                Ok((bytes, dtype))
            },
        )
        .unwrap();
        let weights = assemble_qwen35_weights(&mut store, &config, &manifest, &mut gpu).unwrap();
        assert!(weights.lm_head_aliases_embd);
        if weights.output.gpu_dtype.supports_awq_sidecar() {
            assert!(weights.output.awq_scale.is_some());
        } else {
            assert!(weights.output.awq_scale.is_none());
        }
        weights.free_gpu(&mut gpu);
        hfq.drop_mmap();
    }
}
