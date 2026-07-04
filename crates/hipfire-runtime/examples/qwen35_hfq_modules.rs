#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Probe and repack Qwen35-MoE HFQ expert modules.
//!
//! Usage:
//!   qwen35_hfq_modules probe MODEL.hfq
//!   qwen35_hfq_modules repack INPUT.hfq OUTPUT.hfq

#![allow(clippy::manual_checked_ops)]

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo, HFQM_MAGIC};
use hipfire_runtime::hfq_modules::{
    classify_always_resident_tensor, infer_qwen35_moe_expert_modules, module_table_json,
    HfqModuleKind, HfqModuleRecord, HfqModuleTensor, HFQM_MODULE_TABLE_KEY,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

const HFQM_V2: u32 = 2;
const DEFAULT_ALWAYS_BANK_BYTES: usize = 512 * 1024 * 1024;

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| usage());
    match cmd.as_str() {
        "probe" => {
            let model = args.next().unwrap_or_else(|| usage());
            probe(Path::new(&model));
        }
        "repack" => {
            let input = args.next().unwrap_or_else(|| usage());
            let output = args.next().unwrap_or_else(|| usage());
            repack(Path::new(&input), Path::new(&output));
        }
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: qwen35_hfq_modules probe MODEL.hfq");
    eprintln!("       qwen35_hfq_modules repack INPUT.hfq OUTPUT.hfq");
    std::process::exit(2);
}

fn probe(path: &Path) {
    let hfq = HfqFile::open_index_only(path).expect("open HFQ index");
    let file_bytes = std::fs::metadata(path).expect("stat model").len();
    let explicit_modules = !hfq.modules().is_empty();
    let modules = if explicit_modules {
        hfq.modules().to_vec()
    } else {
        infer_qwen35_moe_expert_modules(hfq.tensors())
    };
    let routed_expert_bytes: usize = modules
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
        .map(|m| module_logical_bytes(m, explicit_modules))
        .sum();
    let routed_modules = modules
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
        .count();
    let mut per_layer: BTreeMap<u16, usize> = BTreeMap::new();
    for module in modules
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
    {
        if let Some(layer) = module.layer {
            *per_layer.entry(layer).or_default() += 1;
        }
    }
    let largest_module_bytes = modules
        .iter()
        .map(|m| module_logical_bytes(m, explicit_modules))
        .max()
        .unwrap_or(0);
    let largest_routed_expert_module_bytes = modules
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
        .map(|m| module_logical_bytes(m, explicit_modules))
        .max()
        .unwrap_or(0);
    let mean_routed_expert_module_bytes = if routed_modules > 0 {
        routed_expert_bytes / routed_modules
    } else {
        0
    };
    let summary = json!({
        "path": path.display().to_string(),
        "hfqm_version": hfq.version,
        "module_table": if explicit_modules { "explicit_hfqm_v2" } else { "inferred_from_v1_tensor_names" },
        "modules_are_contiguous": explicit_modules,
        "file_bytes": file_bytes,
        "tensor_count": hfq.tensors().len(),
        "module_count": modules.len(),
        "routed_expert_modules": routed_modules,
        "routed_expert_bytes": routed_expert_bytes,
        "always_hot_or_other_bytes": file_bytes.saturating_sub(routed_expert_bytes as u64),
        "largest_module_bytes": largest_module_bytes,
        "largest_routed_expert_module_bytes": largest_routed_expert_module_bytes,
        "mean_routed_expert_module_bytes": mean_routed_expert_module_bytes,
        "per_layer_expert_modules": per_layer,
        "payload_read_bytes": 0,
        "gpu_allocated_bytes": 0,
        "full_payload_allocation_skipped": true,
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}

fn module_logical_bytes(module: &HfqModuleRecord, explicit_modules: bool) -> usize {
    if explicit_modules {
        module.data_size
    } else {
        module.tensors.iter().map(|t| t.data_size).sum()
    }
}

fn repack(input: &Path, output: &Path) {
    let hfq = HfqFile::open_index_only(input).expect("open source HFQ index");
    let source_meta: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).expect("source metadata JSON");
    let mut base_meta = source_meta;
    if let serde_json::Value::Object(ref mut map) = base_meta {
        map.remove(HFQM_MODULE_TABLE_KEY);
    }

    let (ordered, module_specs) = plan_tensor_order(hfq.tensors());
    let index_len = tensor_index_len(&ordered);
    let mut data_offset = 0usize;
    let mut metadata_json;
    let mut modules;
    for _ in 0..16 {
        modules = materialize_modules(&module_specs, data_offset);
        metadata_json = metadata_with_modules(base_meta.clone(), &modules);
        let unaligned = 32 + metadata_json.len() + index_len;
        let next = align_up(unaligned, 4096);
        if next == data_offset {
            break;
        }
        data_offset = next;
    }
    modules = materialize_modules(&module_specs, data_offset);
    metadata_json = metadata_with_modules(base_meta, &modules);
    data_offset = align_up(32 + metadata_json.len() + index_len, 4096);
    modules = materialize_modules(&module_specs, data_offset);
    metadata_json = metadata_with_modules(
        serde_json::from_str(&metadata_json).expect("metadata JSON"),
        &modules,
    );
    data_offset = align_up(32 + metadata_json.len() + index_len, 4096);

    write_hfqm_v2(
        input,
        output,
        hfq.arch_id,
        &metadata_json,
        &ordered,
        data_offset,
    )
    .expect("write modular HFQ");

    let routed_expert_bytes: usize = modules
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
        .map(|m| m.data_size)
        .sum();
    let summary = json!({
        "input": input.display().to_string(),
        "output": output.display().to_string(),
        "hfqm_version": HFQM_V2,
        "tensor_count": ordered.len(),
        "module_count": modules.len(),
        "routed_expert_modules": modules.iter().filter(|m| m.kind == HfqModuleKind::RoutedExpert).count(),
        "routed_expert_bytes": routed_expert_bytes,
    });
    eprintln!("{}", serde_json::to_string_pretty(&summary).unwrap());
}

#[derive(Clone)]
struct SourceSlice {
    offset: usize,
    len: usize,
}

#[derive(Clone)]
struct OrderedTensor {
    info: HfqTensorInfo,
    sources: Vec<SourceSlice>,
}

#[derive(Clone)]
struct PlannedModule {
    module_id: String,
    kind: HfqModuleKind,
    layer: Option<u16>,
    expert: Option<u16>,
    placement_policy: Option<String>,
    tensors: Vec<OrderedTensor>,
}

fn plan_tensor_order(tensors: &[HfqTensorInfo]) -> (Vec<OrderedTensor>, Vec<PlannedModule>) {
    let mut expert_groups: BTreeMap<(u16, u16), ExpertGroup> = BTreeMap::new();
    let mut always = Vec::new();
    for tensor in tensors {
        if let Some((layer, expert, role)) = parse_expert_name(&tensor.name) {
            expert_groups
                .entry((layer, expert))
                .or_default()
                .insert(role, tensor.clone());
        } else {
            always.push(tensor.clone());
        }
    }

    let mut ordered = Vec::new();
    let mut modules = Vec::new();
    for (bank_idx, bank) in group_always_resident(always, DEFAULT_ALWAYS_BANK_BYTES)
        .into_iter()
        .enumerate()
    {
        let planned: Vec<OrderedTensor> =
            bank.into_iter().map(OrderedTensor::from_source).collect();
        ordered.extend(planned.iter().cloned());
        modules.push(PlannedModule {
            module_id: format!("always_resident.bank.{bank_idx}"),
            kind: HfqModuleKind::AlwaysResident,
            layer: None,
            expert: None,
            placement_policy: Some("startup_slab".to_string()),
            tensors: planned,
        });
    }

    for ((layer, expert), group) in expert_groups {
        let planned = group.into_ordered_tensors(layer, expert);
        ordered.extend(planned.iter().cloned());
        modules.push(PlannedModule {
            module_id: format!("layers.{layer}.experts.{expert}"),
            kind: HfqModuleKind::RoutedExpert,
            layer: Some(layer),
            expert: Some(expert),
            placement_policy: Some("lazy_lru".to_string()),
            tensors: planned,
        });
    }

    (ordered, modules)
}

impl OrderedTensor {
    fn from_source(info: HfqTensorInfo) -> Self {
        let source = SourceSlice {
            offset: info.data_offset,
            len: info.data_size,
        };
        Self {
            info,
            sources: vec![source],
        }
    }
}

#[derive(Default)]
struct ExpertGroup {
    gate_up: Option<HfqTensorInfo>,
    gate: Option<HfqTensorInfo>,
    up: Option<HfqTensorInfo>,
    down: Option<HfqTensorInfo>,
}

impl ExpertGroup {
    fn insert(&mut self, role: ExpertTensorRole, tensor: HfqTensorInfo) {
        match role {
            ExpertTensorRole::GateUp => self.gate_up = Some(tensor),
            ExpertTensorRole::Gate => self.gate = Some(tensor),
            ExpertTensorRole::Up => self.up = Some(tensor),
            ExpertTensorRole::Down => self.down = Some(tensor),
        }
    }

    fn into_ordered_tensors(self, layer: u16, expert: u16) -> Vec<OrderedTensor> {
        let gate_up = if let Some(gate_up) = self.gate_up {
            OrderedTensor::from_source(gate_up)
        } else {
            let gate = self.gate.unwrap_or_else(|| {
                panic!("expert module layers.{layer}.experts.{expert} missing gate_proj")
            });
            let up = self.up.unwrap_or_else(|| {
                panic!("expert module layers.{layer}.experts.{expert} missing up_proj")
            });
            fused_gate_up_tensor(gate, up, layer, expert)
        };
        let down = self.down.unwrap_or_else(|| {
            panic!("expert module layers.{layer}.experts.{expert} missing down_proj")
        });
        vec![gate_up, OrderedTensor::from_source(down)]
    }
}

fn fused_gate_up_tensor(
    gate: HfqTensorInfo,
    up: HfqTensorInfo,
    layer: u16,
    expert: u16,
) -> OrderedTensor {
    if gate.quant_type != up.quant_type
        || gate.group_size != up.group_size
        || gate.shape.len() != up.shape.len()
        || gate.shape.len() < 2
        || gate.shape[1..] != up.shape[1..]
    {
        panic!(
            "cannot fuse layers.{layer}.experts.{expert} gate/up tensors: incompatible metadata"
        );
    }
    let mut shape = gate.shape.clone();
    shape[0] = shape[0].saturating_add(up.shape[0]);
    let name = if gate.name.contains(".gate_proj.") {
        gate.name.replace(".gate_proj.", ".gate_up_proj.")
    } else {
        format!("model.layers.{layer}.mlp.experts.{expert}.gate_up_proj.weight")
    };
    let data_size = gate.data_size.saturating_add(up.data_size);
    let info = HfqTensorInfo {
        name,
        quant_type: gate.quant_type,
        shape,
        group_size: gate.group_size,
        data_offset: gate.data_offset,
        data_size,
    };
    OrderedTensor {
        info,
        sources: vec![
            SourceSlice {
                offset: gate.data_offset,
                len: gate.data_size,
            },
            SourceSlice {
                offset: up.data_offset,
                len: up.data_size,
            },
        ],
    }
}

fn materialize_modules(planned: &[PlannedModule], data_offset: usize) -> Vec<HfqModuleRecord> {
    let mut cursor = data_offset;
    let mut out = Vec::with_capacity(planned.len());
    for module in planned {
        let start = cursor;
        let mut tensors = Vec::with_capacity(module.tensors.len());
        for tensor in &module.tensors {
            tensors.push(HfqModuleTensor {
                name: tensor.info.name.clone(),
                quant_type: tensor.info.quant_type,
                shape: tensor.info.shape.clone(),
                group_size: tensor.info.group_size,
                rel_offset: cursor - start,
                data_size: tensor.info.data_size,
            });
            cursor += tensor.info.data_size;
        }
        out.push(HfqModuleRecord {
            module_id: module.module_id.clone(),
            kind: module.kind,
            layer: module.layer,
            expert: module.expert,
            placement_policy: module.placement_policy.clone(),
            data_offset: start,
            data_size: cursor - start,
            tensors,
        });
    }
    out
}

fn metadata_with_modules(mut base: serde_json::Value, modules: &[HfqModuleRecord]) -> String {
    if let serde_json::Value::Object(ref mut map) = base {
        map.insert(
            HFQM_MODULE_TABLE_KEY.to_string(),
            serde_json::to_value(module_table_json(modules.to_vec())).unwrap(),
        );
    }
    serde_json::to_string(&base).unwrap()
}

fn write_hfqm_v2(
    input: &Path,
    output: &Path,
    arch_id: u32,
    metadata_json: &str,
    ordered: &[OrderedTensor],
    data_offset: usize,
) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(output)?);
    let metadata = metadata_json.as_bytes();
    let mut index = Vec::new();
    index.extend_from_slice(&(ordered.len() as u32).to_le_bytes());
    for tensor in ordered {
        let name = tensor.info.name.as_bytes();
        index.extend_from_slice(&(name.len() as u16).to_le_bytes());
        index.extend_from_slice(name);
        index.push(tensor.info.quant_type);
        index.push(tensor.info.shape.len() as u8);
        for dim in &tensor.info.shape {
            index.extend_from_slice(&dim.to_le_bytes());
        }
        index.extend_from_slice(&tensor.info.group_size.to_le_bytes());
        index.extend_from_slice(&(tensor.info.data_size as u64).to_le_bytes());
    }

    out.write_all(HFQM_MAGIC)?;
    out.write_all(&HFQM_V2.to_le_bytes())?;
    out.write_all(&arch_id.to_le_bytes())?;
    out.write_all(&(ordered.len() as u32).to_le_bytes())?;
    out.write_all(&(32u64).to_le_bytes())?;
    out.write_all(&(data_offset as u64).to_le_bytes())?;
    out.write_all(metadata)?;
    out.write_all(&index)?;
    let written = 32 + metadata.len() + index.len();
    if data_offset < written {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "computed HFQM data offset precedes index end",
        ));
    }
    out.write_all(&vec![0u8; data_offset - written])?;

    let mut src = File::open(input)?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    for tensor in ordered {
        let copied: usize = tensor.sources.iter().map(|s| s.len).sum();
        if copied != tensor.info.data_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ordered tensor {} has {} bytes of source slices for {} byte output",
                    tensor.info.name, copied, tensor.info.data_size
                ),
            ));
        }
        for source in &tensor.sources {
            copy_range(
                &mut src,
                &mut out,
                source.offset as u64,
                source.len,
                &mut buf,
            )?;
        }
    }
    out.flush()
}

fn copy_range(
    src: &mut File,
    dst: &mut BufWriter<File>,
    offset: u64,
    mut len: usize,
    buf: &mut [u8],
) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    src.seek(SeekFrom::Start(offset))?;
    while len > 0 {
        let n = len.min(buf.len());
        src.read_exact(&mut buf[..n])?;
        dst.write_all(&buf[..n])?;
        len -= n;
    }
    Ok(())
}

fn group_always_resident(
    mut tensors: Vec<HfqTensorInfo>,
    bank_size: usize,
) -> Vec<Vec<HfqTensorInfo>> {
    tensors.sort_by_key(|t| {
        let class = classify_always_resident_tensor(&t.name);
        (
            match class {
                HfqModuleKind::Embedding => 0,
                HfqModuleKind::Norm => 1,
                HfqModuleKind::Attention => 2,
                HfqModuleKind::Router => 3,
                HfqModuleKind::SharedExpert => 4,
                HfqModuleKind::LmHead => 5,
                _ => 6,
            },
            t.data_offset,
        )
    });
    let mut banks: Vec<Vec<HfqTensorInfo>> = Vec::new();
    let mut cur = Vec::new();
    let mut cur_bytes = 0usize;
    for tensor in tensors {
        if !cur.is_empty() && cur_bytes.saturating_add(tensor.data_size) > bank_size {
            banks.push(cur);
            cur = Vec::new();
            cur_bytes = 0;
        }
        cur_bytes = cur_bytes.saturating_add(tensor.data_size);
        cur.push(tensor);
    }
    if !cur.is_empty() {
        banks.push(cur);
    }
    banks
}

fn tensor_index_len(tensors: &[OrderedTensor]) -> usize {
    4 + tensors
        .iter()
        .map(|t| 2 + t.info.name.len() + 1 + 1 + t.info.shape.len() * 4 + 4 + 8)
        .sum::<usize>()
}

#[derive(Clone, Copy)]
enum ExpertTensorRole {
    GateUp,
    Gate,
    Up,
    Down,
}

fn parse_expert_name(name: &str) -> Option<(u16, u16, ExpertTensorRole)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layer_pos = parts.iter().position(|p| *p == "layers")?;
    let layer = parts.get(layer_pos + 1)?.parse::<u16>().ok()?;
    let expert_pos = parts.iter().position(|p| *p == "experts")?;
    let expert = parts.get(expert_pos + 1)?.parse::<u16>().ok()?;
    let role = parts.get(expert_pos + 2)?;
    let role = match *role {
        "gate_up_proj" => ExpertTensorRole::GateUp,
        "gate_proj" => ExpertTensorRole::Gate,
        "up_proj" => ExpertTensorRole::Up,
        "down_proj" => ExpertTensorRole::Down,
        _ => return None,
    };
    Some((layer, expert, role))
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}
