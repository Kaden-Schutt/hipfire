// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    clippy::collapsible_if,
    clippy::if_same_then_else,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

//! dspark_convert: Convert a HuggingFace DSpark drafter (`Qwen3DSparkModel`
//! safetensors + config.json) into a hipfire `.dspark.hfq` sidecar the runtime
//! loads alongside its target model.
//!
//! Usage:
//!     dspark_convert --input <dir_or_hf_id> --output <target-quant>.dspark.hfq [--all-f16]
//!
//! This is the final port piece of the DSpark speculator: the runtime side
//! (`hipfire_arch_llama::dspark_body::load_qwen3_dspark` +
//! `hipfire_specdecode_dspark::dspark_core::DsparkConfig::from_metadata_json`)
//! already loads `<target>-<quant>.dspark.hfq` sidecars; this converter produces
//! them from the trained drafter export.
//!
//! ## AGENTS.md note
//! Per AGENTS.md, import/export & format-conversion tooling *ideally* lives in
//! the `hipfire-coexistence` binary, not `hipfire-quantize`. This bin is placed
//! here to match the existing DSpark-sibling converters (`dflash_convert`,
//! arch_id 20; `mtp_extract`, arch_id 21) which already live in
//! `hipfire-quantize/src/bin/`. TODO(dspark): fold all three draft/sidecar
//! converters into `hipfire-coexistence` in a future cleanup.
//!
//! ## Quant recipe (small trained drafter — preserve precision)
//!   - 2D matmul weights (attn q/k/v/o_proj, mlp gate/up/down_proj) → Q8F16
//!     (quant_type 3; the runtime `sidecar_weight` maps it to `DType::Q8_0`).
//!   - Everything else (norms, embed_tokens, main_proj/main_norm, markov heads,
//!     confidence head + bias, lm_head) → F16 (quant_type 1). The loader widens
//!     norm/embed payloads to F32 and keeps matrices as F16.
//!   `--all-f16` forces the body matmuls to F16 too (still runtime-loadable).
//!
//!   The MQ{3,4,6}-G256 FWHT codecs `dflash_convert` offers are intentionally
//!   NOT wired here: the DSpark drafter-body loader (`sidecar_weight`) only
//!   accepts quant_type ∈ {0,1,2,3,4,5,6,7}, so an MQ (13/15/17) sidecar would
//!   fail to load. Q8F16/F16 is the only faithful, loadable recipe today.
//!   TODO(dspark): teach `sidecar_weight` the MQ dtypes if a smaller drafter is
//!   ever wanted, then expose --mq{3,4,6} like dflash_convert.
//!
//! ## Tensor-name mapping (source safetensors → sidecar)
//!   fc.weight           → main_proj.weight   (`[dim, n_targets*dim]` concat proj)
//!   hidden_norm.weight  → main_norm.weight    (RMSNorm after fc)
//!   all others          → kept verbatim; the HF DSpark export already uses the
//!                         flat names the loader expects:
//!                           layers.{0..4}.self_attn.{q,k,v,o}_proj.weight
//!                           layers.{0..4}.self_attn.{q,k}_norm.weight
//!                           layers.{0..4}.{input,post_attention}_layernorm.weight
//!                           layers.{0..4}.mlp.{gate,up,down}_proj.weight
//!                           embed_tokens.weight
//!                           markov_head.markov_w1.weight / markov_w2.weight
//!                           confidence_head.proj.weight / confidence_head.proj.bias
//!                           norm.weight
//!                           lm_head.weight
//!
//! ## Metadata JSON layout (read by `DsparkConfig::from_metadata_json`)
//!
//! ```json
//! {
//!   "architecture": "qwen3",
//!   "config": {
//!     "dspark_block_size": 7,
//!     "dspark_target_layer_ids": [1, 9, 17, 25, 33],
//!     "dspark_markov_rank": 256,
//!     "dspark_noise_token_id": 151669,
//!     "dspark_enable_confidence": true,
//!     "dspark_confidence_uses_normed": true,
//!     "norm_eps": 1e-6
//!   }
//! }
//! ```
//!
//! arch_id for the DSpark drafter sidecar is `ARCH_ID_DSPARK_DRAFT` (22).

use hipfire_arch_api::ARCH_ID_DSPARK_DRAFT;
use hipfire_primitives::conv::{f32_slice_to_f16_bytes, f32_to_f16, plain_dtype_to_f32 as to_f32};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors Parser (mirrors dflash_convert) ───────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
            .expect("safetensors header is not valid utf8");
        let raw: serde_json::Value =
            serde_json::from_str(header_json).expect("safetensors header JSON parse failed");
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(v)
                    .unwrap_or_else(|e| panic!("tensor meta for {k}: {e}"));
                tensors.insert(k, meta);
            }
        }
        Ok(Self {
            _file: file,
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        let start = self.header_size + meta.data_offsets[0];
        let end = self.header_size + meta.data_offsets[1];
        Some((meta, &self.mmap[start..end]))
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }
}

// ─── Q8F16 quantization ────────────────────────────────────────────────────
//
// Group-of-32 symmetric int8 with an F16 per-group scale (34 bytes/group).
// quant_type 3 → the runtime maps it to `DType::Q8_0` in `sidecar_weight`.
// Lifted verbatim from the source `hipfire-quantize/main.rs::quantize_q8f16`.

fn quantize_q8f16(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 32;
    let block_bytes = 34;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-128.0).min(127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

// ─── HFQ File Format (mirrors dflash_convert) ──────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum QuantType {
    F16 = 1,
    F32 = 2,
    /// Group-32 int8 + F16 scale. Runtime `sidecar_weight` → `DType::Q8_0`.
    Q8F16 = 3,
}

struct HfqTensor {
    name: String,
    quant_type: QuantType,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();

    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type as u8);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }

    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    f.write_all(metadata_bytes)?;
    f.write_all(&index_bytes)?;

    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    for t in tensors {
        f.write_all(&t.data)?;
    }

    Ok(())
}

// ─── Model discovery (mirrors dflash_convert) ──────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);
    if path.join("config.json").exists() {
        return input.to_string();
    }
    if input.contains('/') {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_root =
                format!("{home}/.cache/huggingface/hub/models--{org}--{name}/snapshots");
            if let Ok(entries) = std::fs::read_dir(&cache_root) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.join("config.json").exists() {
                        return p.to_string_lossy().into_owned();
                    }
                }
            }
        }
    }
    input.to_string()
}

// ─── Tensor classification ─────────────────────────────────────────────────

/// True for the drafter-body 2D matmul weights that get Q8F16: attention
/// q/k/v/o projections and the MLP gate/up/down projections. Everything else
/// (norms, embeds, main_proj/main_norm, markov + confidence heads, lm_head)
/// stays F16. Mirrors the source `is_dspark_matmul_weight`.
fn is_dspark_matmul_weight(name: &str) -> bool {
    let is_attn = name.contains("self_attn.")
        && (name.ends_with("q_proj.weight")
            || name.ends_with("k_proj.weight")
            || name.ends_with("v_proj.weight")
            || name.ends_with("o_proj.weight"));
    let is_mlp = name.contains("mlp.")
        && (name.ends_with("gate_proj.weight")
            || name.ends_with("up_proj.weight")
            || name.ends_with("down_proj.weight"));
    is_attn || is_mlp
}

/// Map a source safetensors tensor name to its sidecar name. Only `fc.weight`
/// and `hidden_norm.weight` are renamed; the HF export already uses the flat
/// names the loader expects for everything else.
fn sidecar_tensor_name(name: &str) -> String {
    match name {
        "fc.weight" => "main_proj.weight".to_string(),
        "hidden_norm.weight" => "main_norm.weight".to_string(),
        other => other.to_string(),
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input_dir: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut all_f16 = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                input_dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--output" | "-o" => {
                output_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--all-f16" => {
                all_f16 = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: dspark_convert --input <dir_or_hf_id> --output <target-quant>.dspark.hfq [--all-f16]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    let input_dir = input_dir.expect("--input required");
    let output_path = output_path.expect("--output required");
    let input_dir = resolve_model_path(&input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(&output_path);

    eprintln!("dspark_convert");
    eprintln!("  input : {}", input_dir.display());
    eprintln!("  output: {}", output_path.display());
    eprintln!(
        "  dtype : {}",
        if all_f16 {
            "F16 (all tensors)"
        } else {
            "Q8F16 (body matmuls), F16 (norms/globals/embed/lm_head)"
        }
    );

    // ── config.json ────────────────────────────────────────────────────────
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_path.display()));
    let config: serde_json::Value =
        serde_json::from_str(&config_str).expect("config.json parse failed");

    // Verify architecture.
    let is_dspark = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().any(|v| v.as_str() == Some("Qwen3DSparkModel")))
        .unwrap_or(false);
    if !is_dspark {
        eprintln!(
            "warning: config.json architectures != [Qwen3DSparkModel] (got {:?}); \
             continuing anyway",
            config.get("architectures")
        );
    }

    // DSpark config fields. Defaults mirror the source emit path.
    // TODO(dspark): validate against a real drafter export — the exact
    // config.json key placement (top-level vs nested) is mirrored from the
    // source converter, which read these keys from the top level.
    let block_size = config
        .get("block_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(7) as usize;
    let target_layer_ids: Vec<u64> = config
        .get("target_layer_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_else(|| vec![1, 9, 17, 25, 33]);
    let markov_rank = config
        .get("markov_rank")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    let noise_token_id = config
        .get("mask_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(151669) as u32;
    let norm_eps = config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6);

    eprintln!(
        "  dspark: block_size={block_size} target_layer_ids={target_layer_ids:?} \
         markov_rank={markov_rank} noise_token_id={noise_token_id} norm_eps={norm_eps:e}"
    );

    // ── metadata JSON — keys read by DsparkConfig::from_metadata_json ───────
    // qwen3 drafter feeds once-normed hidden to the confidence head, so
    // dspark_confidence_uses_normed=true (the loader also pins this for qwen3,
    // but we emit it explicitly for a self-describing sidecar).
    let metadata = serde_json::json!({
        "architecture": "qwen3",
        "config": {
            "dspark_block_size": block_size,
            "dspark_target_layer_ids": target_layer_ids,
            "dspark_markov_rank": markov_rank,
            "dspark_noise_token_id": noise_token_id,
            "dspark_enable_confidence": true,
            "dspark_confidence_uses_normed": true,
            "norm_eps": norm_eps,
        },
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // ── safetensors ────────────────────────────────────────────────────────
    let st_paths = find_safetensors(input_dir);
    assert!(
        !st_paths.is_empty(),
        "no .safetensors files found in {}",
        input_dir.display()
    );
    let st_files: Vec<SafetensorsFile> = st_paths
        .iter()
        .inspect(|p| eprintln!("  loading: {}", p.display()))
        .map(|p| SafetensorsFile::open(p).expect("safetensors open failed"))
        .collect();

    let mut name_to_file: Vec<(String, usize)> = Vec::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            name_to_file.push((name, fi));
        }
    }
    name_to_file.sort_by_key(|(name, _)| name.clone());
    eprintln!("  tensors: {}", name_to_file.len());

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(name_to_file.len());
    let mut total_params = 0u64;
    let mut q8_params = 0u64;
    let mut f16_params = 0u64;

    for (name, fi) in &name_to_file {
        let (meta, raw) = st_files[*fi]
            .tensor_data(name)
            .expect("tensor lookup failed");
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        let sidecar_name = sidecar_tensor_name(name);
        let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
        let f32_data = to_f32(raw, &meta.dtype);

        // Body 2D matmul → Q8F16 (group 32) unless --all-f16; else F16.
        let (quant_type, group_size, data) =
            if !all_f16 && is_dspark_matmul_weight(name) && n_elements >= 32 {
                q8_params += n_elements as u64;
                (QuantType::Q8F16, 32u32, quantize_q8f16(&f32_data))
            } else {
                f16_params += n_elements as u64;
                (QuantType::F16, 0u32, f32_slice_to_f16_bytes(&f32_data))
            };

        hfq_tensors.push(HfqTensor {
            name: sidecar_name,
            quant_type,
            shape,
            group_size,
            data,
        });
    }

    eprintln!(
        "  summary: {} tensors, {:.3}B params (Q8F16 {:.1}%, F16 {:.1}%)",
        hfq_tensors.len(),
        total_params as f64 / 1e9,
        100.0 * q8_params as f64 / total_params.max(1) as f64,
        100.0 * f16_params as f64 / total_params.max(1) as f64,
    );

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir -p output parent");
        }
    }

    write_hfq(
        output_path,
        ARCH_ID_DSPARK_DRAFT,
        &metadata_json,
        &hfq_tensors,
    )
    .expect("write_hfq failed");

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  wrote: {} ({:.1} MB)",
        output_path.display(),
        file_size as f64 / 1e6
    );
}
