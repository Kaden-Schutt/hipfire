// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFQ output serialization + provenance.
//!
//! The `.hfq` writer (`write_hfq`), its streaming tensor spill, the
//! parameter-count / quantization-hash / git-provenance metadata builders, and
//! the small XXH64 helpers that back the quantization hash. Extracted from the
//! `hipfire-quantize` binary's `main.rs` so the GGUF import pipeline (now owned
//! by `hipfire-coexistence`) can produce byte-identical `.hfq` artifacts
//! through the same code path the native quantizer uses. See AGENTS.md: import
//! tooling lives outside the inference-adjacent quantize binary.

use hipfire_arch_api::{transformer_role, TensorRole};
use hipfire_quant_format::QuantType;
use std::fs::File;
use std::hash::Hasher;
use std::io::Write;
use std::path::{Path, PathBuf};
use twox_hash::XxHash64;

/// HFQ container magic + format version (the `.hfq` = "HFQM" v1 header).
pub const HFQ_MAGIC: &[u8; 4] = b"HFQM";
pub const HFQ_VERSION: u32 = 1;

pub struct HfqTensor {
    pub name: String,
    pub quant_type: QuantType,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    pub spilled_len: u64,
}

pub fn tensor_param_count(t: &HfqTensor) -> u64 {
    t.shape
        .iter()
        .fold(1u64, |acc, &dim| acc.saturating_mul(dim as u64))
}

pub fn config_u64_any(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    fn get_from_scope(scope: &serde_json::Value, keys: &[&str]) -> Option<u64> {
        keys.iter().find_map(|key| scope.get(*key)?.as_u64())
    }

    get_from_scope(config, keys)
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|scope| get_from_scope(scope, keys))
        })
        .or_else(|| {
            config
                .get("moe")
                .and_then(|scope| get_from_scope(scope, keys))
        })
        .or_else(|| {
            config
                .get("ffn_config")
                .and_then(|scope| get_from_scope(scope, keys))
        })
}

pub fn model_config_from_metadata(metadata: &serde_json::Value) -> &serde_json::Value {
    metadata.get("config").unwrap_or(metadata)
}

pub fn routed_moe_config(metadata: &serde_json::Value) -> Option<(u64, u64)> {
    let config = model_config_from_metadata(metadata);
    let num_experts = config_u64_any(
        config,
        &[
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "n_experts",
        ],
    )?;
    let top_k = config_u64_any(
        config,
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "n_experts_per_tok",
            "moe_top_k",
            "top_k",
            "num_selected_experts",
        ],
    )?;
    if num_experts == 0 || top_k == 0 {
        None
    } else {
        Some((num_experts, top_k))
    }
}

pub fn parameter_counts_metadata(
    metadata: &serde_json::Value,
    tensors: &[HfqTensor],
    total_params: u64,
    quantized_params: u64,
    skipped_params: u64,
) -> serde_json::Value {
    let mut routed_expert_params = 0u64;
    for t in tensors {
        if transformer_role(&t.name) == TensorRole::Expert {
            routed_expert_params = routed_expert_params.saturating_add(tensor_param_count(t));
        }
    }

    let (active_params, effective_params, moe) = if routed_expert_params > 0 {
        if let Some((num_experts, top_k)) = routed_moe_config(metadata) {
            let numerator = routed_expert_params.saturating_mul(top_k);
            let routed_active = numerator / num_experts;
            let active = total_params
                .saturating_sub(routed_expert_params)
                .saturating_add(routed_active);
            (
                active,
                active,
                Some(serde_json::json!({
                    "num_experts": num_experts,
                    "num_experts_per_tok": top_k,
                    "routed_expert_params": routed_expert_params,
                    "routed_expert_active_params": routed_active,
                    "active_rule": "dense_and_shared_full_plus_routed_top_k_over_num_experts",
                    "routed_active_fraction": {
                        "numerator": numerator,
                        "denominator": num_experts,
                    },
                })),
            )
        } else {
            (
                total_params,
                total_params,
                Some(serde_json::json!({
                    "routed_expert_params": routed_expert_params,
                    "active_rule": "unknown_top_k_or_num_experts",
                })),
            )
        }
    } else {
        (total_params, total_params, None)
    };

    let source_total_params = total_params.saturating_add(skipped_params);
    let mut counts = serde_json::json!({
        "schema": "hipfire.parameter_counts.v1",
        "total_params": total_params,
        "source_total_params": source_total_params,
        "active_params": active_params,
        "effective_params": effective_params,
        "quantized_params": quantized_params,
        "skipped_params": skipped_params,
    });
    if let Some(moe) = moe {
        if let serde_json::Value::Object(ref mut map) = counts {
            map.insert("moe".to_string(), moe);
        }
    }
    counts
}

pub fn insert_parameter_counts_metadata(
    metadata: &mut serde_json::Value,
    tensors: &[HfqTensor],
    total_params: u64,
    quantized_params: u64,
    skipped_params: u64,
) {
    let counts = parameter_counts_metadata(
        metadata,
        tensors,
        total_params,
        quantized_params,
        skipped_params,
    );
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("parameter_counts".to_string(), counts);
    }
}

pub struct Xxh64 {
    inner: XxHash64,
}

impl Xxh64 {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: XxHash64::with_seed(seed),
        }
    }

    pub fn update(&mut self, input: &[u8]) {
        self.inner.write(input);
    }

    pub fn digest(&self) -> u64 {
        self.inner.finish()
    }
}

pub fn xxh64_update_u8(h: &mut Xxh64, v: u8) {
    h.update(&[v]);
}

pub fn xxh64_update_u32(h: &mut Xxh64, v: u32) {
    h.update(&v.to_le_bytes());
}

pub fn xxh64_update_u64(h: &mut Xxh64, v: u64) {
    h.update(&v.to_le_bytes());
}

pub fn hfq_quantization_hash_metadata(
    tensors: &[HfqTensor],
    spill: Option<&TensorSpill>,
) -> std::io::Result<serde_json::Value> {
    let mut h = Xxh64::new(0);
    let mut payload_bytes = 0u64;
    h.update(b"hipfire-hfq-quantized-tensor-payload-v1");

    let mut spill_reader = if let Some(spill) = spill {
        Some(std::io::BufReader::new(File::open(&spill.path)?))
    } else {
        None
    };
    let mut buf = vec![0u8; 4 * 1024 * 1024];

    for t in tensors {
        let name_bytes = t.name.as_bytes();
        xxh64_update_u64(&mut h, name_bytes.len() as u64);
        h.update(name_bytes);
        xxh64_update_u8(&mut h, t.quant_type as u8);
        xxh64_update_u64(&mut h, t.shape.len() as u64);
        for &dim in &t.shape {
            xxh64_update_u32(&mut h, dim);
        }
        xxh64_update_u32(&mut h, t.group_size);
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        xxh64_update_u64(&mut h, data_len);
        payload_bytes += data_len;

        if t.spilled_len > 0 {
            let reader = spill_reader
                .as_mut()
                .expect("spilled tensor requires spill reader");
            let mut remaining = t.spilled_len as usize;
            while remaining > 0 {
                let chunk = remaining.min(buf.len());
                use std::io::Read;
                reader.read_exact(&mut buf[..chunk])?;
                h.update(&buf[..chunk]);
                remaining -= chunk;
            }
        } else {
            h.update(&t.data);
        }
    }

    Ok(serde_json::json!({
        "algorithm": "xxh64",
        "seed": 0,
        "scope": "hfq_tensor_index_and_payload_v1",
        "value": format!("{:016x}", h.digest()),
        "tensor_count": tensors.len(),
        "payload_bytes": payload_bytes,
        "producer": {
            "package": "hipfire-quantize",
            "hipfire_version": env!("CARGO_PKG_VERSION"),
            "git_commit": git_commit(),
            "git_branch": git_branch(),
            "git_describe": git_describe(),
            "git_dirty": git_dirty(),
        },
    }))
}

pub fn metadata_with_quantization_hash(
    mut metadata: serde_json::Value,
    tensors: &[HfqTensor],
    spill: Option<&TensorSpill>,
) -> std::io::Result<String> {
    let hash = hfq_quantization_hash_metadata(tensors, spill)?;
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("quantization_hash".to_string(), hash);
    }
    serde_json::to_string(&metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

pub fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn git_commit() -> Option<String> {
    command_stdout("git", &["rev-parse", "HEAD"])
}

pub fn git_branch() -> Option<String> {
    command_stdout("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

pub fn git_describe() -> Option<String> {
    command_stdout("git", &["describe", "--always", "--dirty", "--tags"])
}

pub fn git_dirty() -> Option<bool> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(!out.stdout.is_empty())
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
pub struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        // PID-unique so concurrent quantize runs in the same output dir don't
        // share a spill path (a sibling run's Drop would otherwise delete this
        // run's spill file → write_hfq NotFound panic).
        let path = dir.join(format!(".hipfire_quant_spill.{}.tmp", std::process::id()));
        let file = std::io::BufWriter::with_capacity(4 * 1024 * 1024, File::create(&path)?);
        Ok(Self {
            file,
            path,
            offset: 0,
        })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    pub fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    pub fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    let metadata_bytes = metadata_json.as_bytes();

    // Calculate offsets
    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    // Tensor index follows metadata
    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    // Write tensor count
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        // name length + name
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        // quant type
        index_bytes.push(t.quant_type as u8);
        // n_dims + shape
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        // group size
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        // data size (offset computed at read time from cumulative sizes)
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        index_bytes.extend_from_slice(&data_len.to_le_bytes());
    }

    // Data starts after index, aligned to 4096
    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    // Write header (32 bytes)
    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    // Write metadata
    f.write_all(metadata_bytes)?;

    // Write tensor index
    f.write_all(&index_bytes)?;

    // Pad to data alignment
    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    // Write tensor data — from spill file or from memory
    if let Some(spill) = spill {
        let _ = spill.flush();
        let mut spill_reader = std::io::BufReader::new(File::open(&spill.path)?);
        let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for t in tensors {
            if t.spilled_len > 0 {
                // Copy from spill file
                let mut remaining = t.spilled_len as usize;
                while remaining > 0 {
                    let chunk = remaining.min(buf.len());
                    use std::io::Read;
                    spill_reader.read_exact(&mut buf[..chunk])?;
                    f.write_all(&buf[..chunk])?;
                    remaining -= chunk;
                }
            } else {
                f.write_all(&t.data)?;
            }
        }
    } else {
        for t in tensors {
            f.write_all(&t.data)?;
        }
    }

    Ok(())
}
