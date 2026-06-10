// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Module metadata for HFQM v2 containers.
//!
//! HFQM v1 stores a flat tensor index followed by a flat tensor payload stream.
//! HFQM v2 keeps the flat tensor index for compatibility and adds a metadata
//! module table so very large MoE artifacts can group routed experts into
//! contiguous, slab-loadable units.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::hfq::HfqTensorInfo;

pub const HFQM_MODULE_TABLE_KEY: &str = "hfqm_modules";
pub const HFQM_MODULE_TABLE_FORMAT: &str = "hipfire.hfqm.modules.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HfqModuleKind {
    AlwaysResident,
    RoutedExpert,
    SharedExpert,
    Router,
    Attention,
    Norm,
    Embedding,
    LmHead,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfqModuleTensor {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub rel_offset: usize,
    pub data_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfqModuleRecord {
    pub module_id: String,
    pub kind: HfqModuleKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement_policy: Option<String>,
    pub data_offset: usize,
    pub data_size: usize,
    pub tensors: Vec<HfqModuleTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfqModuleTable {
    pub format: String,
    pub modules: Vec<HfqModuleRecord>,
}

impl HfqModuleRecord {
    pub fn data_end(&self) -> Option<usize> {
        self.data_offset.checked_add(self.data_size)
    }
}

pub fn parse_module_table(metadata_json: &str) -> std::io::Result<Option<Vec<HfqModuleRecord>>> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQM metadata is not valid JSON: {e}"),
        )
    })?;
    let Some(value) = meta.get(HFQM_MODULE_TABLE_KEY) else {
        return Ok(None);
    };
    let table: HfqModuleTable = serde_json::from_value(value.clone()).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid HFQM module table: {e}"),
        )
    })?;
    if table.format != HFQM_MODULE_TABLE_FORMAT {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported HFQM module table format {:?}", table.format),
        ));
    }
    validate_modules(&table.modules, usize::MAX)?;
    Ok(Some(table.modules))
}

pub fn module_table_json(modules: Vec<HfqModuleRecord>) -> HfqModuleTable {
    HfqModuleTable {
        format: HFQM_MODULE_TABLE_FORMAT.to_string(),
        modules,
    }
}

pub fn validate_modules(modules: &[HfqModuleRecord], file_len: usize) -> std::io::Result<()> {
    let mut ranges: Vec<(usize, usize, &str)> = Vec::with_capacity(modules.len());
    for module in modules {
        let Some(end) = module.data_end() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQM module {} byte range overflows", module.module_id),
            ));
        };
        if module.data_size == 0 || module.data_offset >= end || end > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQM module {} invalid range {}..{} for file_len {}",
                    module.module_id, module.data_offset, end, file_len
                ),
            ));
        }
        for tensor in &module.tensors {
            let Some(t_end) = tensor.rel_offset.checked_add(tensor.data_size) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "HFQM module {} tensor {} relative range overflows",
                        module.module_id, tensor.name
                    ),
                ));
            };
            if tensor.data_size == 0 || t_end > module.data_size {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "HFQM module {} tensor {} range {}..{} exceeds module size {}",
                        module.module_id, tensor.name, tensor.rel_offset, t_end, module.data_size
                    ),
                ));
            }
        }
        ranges.push((module.data_offset, end, module.module_id.as_str()));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    for pair in ranges.windows(2) {
        let (_, prev_end, prev_id) = pair[0];
        let (next_start, _, next_id) = pair[1];
        if next_start < prev_end {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQM modules {prev_id} and {next_id} overlap"),
            ));
        }
    }
    Ok(())
}

pub fn classify_always_resident_tensor(name: &str) -> HfqModuleKind {
    if name.ends_with("embed_tokens.weight") {
        HfqModuleKind::Embedding
    } else if name.ends_with("lm_head.weight") {
        HfqModuleKind::LmHead
    } else if name.contains(".mlp.gate.weight") {
        HfqModuleKind::Router
    } else if name.contains(".mlp.shared_expert.") || name.contains(".mlp.shared_expert_gate.") {
        HfqModuleKind::SharedExpert
    } else if name.contains("norm") {
        HfqModuleKind::Norm
    } else if name.contains(".self_attn.") || name.contains(".linear_attn.") {
        HfqModuleKind::Attention
    } else {
        HfqModuleKind::AlwaysResident
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertKey {
    layer: u16,
    expert: u16,
}

pub fn infer_qwen35_moe_expert_modules(tensors: &[HfqTensorInfo]) -> Vec<HfqModuleRecord> {
    let mut experts: BTreeMap<ExpertKey, Vec<&HfqTensorInfo>> = BTreeMap::new();
    for tensor in tensors {
        if let Some((layer, expert)) = parse_qwen35_expert_tensor_name(&tensor.name) {
            experts
                .entry(ExpertKey { layer, expert })
                .or_default()
                .push(tensor);
        }
    }

    let mut modules = Vec::with_capacity(experts.len());
    for (key, mut entries) in experts {
        entries.sort_by_key(|t| t.data_offset);
        let start = entries.iter().map(|t| t.data_offset).min().unwrap_or(0);
        let end = entries
            .iter()
            .map(|t| t.data_offset.saturating_add(t.data_size))
            .max()
            .unwrap_or(start);
        let tensors = entries
            .into_iter()
            .map(|t| HfqModuleTensor {
                name: t.name.clone(),
                quant_type: t.quant_type,
                shape: t.shape.clone(),
                group_size: t.group_size,
                rel_offset: t.data_offset.saturating_sub(start),
                data_size: t.data_size,
            })
            .collect();
        modules.push(HfqModuleRecord {
            module_id: format!("layers.{}.experts.{}", key.layer, key.expert),
            kind: HfqModuleKind::RoutedExpert,
            layer: Some(key.layer),
            expert: Some(key.expert),
            placement_policy: Some("lazy_lru".to_string()),
            data_offset: start,
            data_size: end.saturating_sub(start),
            tensors,
        });
    }
    modules
}

fn parse_qwen35_expert_tensor_name(name: &str) -> Option<(u16, u16)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layer_pos = parts.iter().position(|p| *p == "layers")?;
    let layer = parts.get(layer_pos + 1)?.parse::<u16>().ok()?;
    let expert_pos = parts.iter().position(|p| *p == "experts")?;
    let expert = parts.get(expert_pos + 1)?.parse::<u16>().ok()?;
    let role = parts.get(expert_pos + 2)?;
    if *role == "gate_up_proj" || *role == "gate_proj" || *role == "up_proj" || *role == "down_proj"
    {
        Some((layer, expert))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, offset: usize, size: usize) -> HfqTensorInfo {
        HfqTensorInfo {
            name: name.to_string(),
            quant_type: 15,
            shape: vec![4, 4],
            group_size: 256,
            data_offset: offset,
            data_size: size,
        }
    }

    #[test]
    fn infers_qwen35_routed_expert_modules() {
        let tensors = vec![
            tensor("model.layers.3.mlp.experts.7.gate_up_proj.weight", 100, 20),
            tensor("model.layers.3.mlp.experts.7.down_proj.weight", 120, 10),
            tensor("model.layers.3.mlp.gate.weight", 20, 5),
        ];
        let modules = infer_qwen35_moe_expert_modules(&tensors);
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].kind, HfqModuleKind::RoutedExpert);
        assert_eq!(modules[0].layer, Some(3));
        assert_eq!(modules[0].expert, Some(7));
        assert_eq!(modules[0].data_offset, 100);
        assert_eq!(modules[0].data_size, 30);
        assert_eq!(modules[0].tensors.len(), 2);
    }

    #[test]
    fn rejects_overlapping_modules() {
        let modules = vec![
            HfqModuleRecord {
                module_id: "a".to_string(),
                kind: HfqModuleKind::AlwaysResident,
                layer: None,
                expert: None,
                placement_policy: None,
                data_offset: 10,
                data_size: 10,
                tensors: vec![HfqModuleTensor {
                    name: "a.weight".to_string(),
                    quant_type: 15,
                    shape: vec![1],
                    group_size: 256,
                    rel_offset: 0,
                    data_size: 10,
                }],
            },
            HfqModuleRecord {
                module_id: "b".to_string(),
                kind: HfqModuleKind::AlwaysResident,
                layer: None,
                expert: None,
                placement_policy: None,
                data_offset: 19,
                data_size: 2,
                tensors: vec![HfqModuleTensor {
                    name: "b.weight".to_string(),
                    quant_type: 15,
                    shape: vec![1],
                    group_size: 256,
                    rel_offset: 0,
                    data_size: 2,
                }],
            },
        ];
        assert!(validate_modules(&modules, 100).is_err());
    }
}
