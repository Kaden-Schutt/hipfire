//! dflash_convert: Convert a HuggingFace DFlash draft safetensors + config.json
//! into a hipfire `.hfq` file with a dflash metadata section.
//!
//! Usage:
//!     dflash_convert --input <dir_or_hf_id> --output <file.hfq> [--keep-f32]
//!
//! Reads a single-file safetensors dump (the z-lab/Qwen3.5-*-DFlash draft
//! layout — no shards in practice at 1-4B params) and rewrites the tensors
//! into the hipfire HFQ container. All weights are cast BF16 → F16 by default
//! to halve the file size (pass `--keep-f32` to keep full F32 precision).
//! Per-layer norms (`input_layernorm`, `post_attention_layernorm`,
//! `q_norm`, `k_norm`, `hidden_norm`, `norm`) are always F32.
//!
//! Metadata JSON layout:
//!
//! ```json
//! {
//!   "architecture": "dflash",
//!   "config": {<full HF config.json>},
//!   "dflash": {
//!     "block_size": 16,
//!     "mask_token_id": 248070,
//!     "target_layer_ids": [1, 8, 15, 22, 29],
//!     "num_target_layers": 32,
//!     "draft_dtype": "f16"
//!   },
//!   "tokenizer": null
//! }
//! ```
//!
//! arch_id for the dflash draft is 20. The hipfire loader distinguishes
//! dflash drafts from Qwen3/Qwen3.5 by both arch_id and the presence of
//! the top-level `dflash` key in metadata.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors Parser (single-file only) ─────────────────────────────────

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
        let raw: serde_json::Value = serde_json::from_str(header_json)
            .expect("safetensors header JSON parse failed");
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

// ─── FP conversions ────────────────────────────────────────────────────────

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { (frac << 13) | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported input dtype: {other}"),
    }
}

fn f32_slice_to_f16_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 2);
    for &v in f32_data {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

fn f32_slice_to_f32_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 4);
    for &v in f32_data {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

// ─── FWHT + MQ4 quantization ──────────────────────────────────────────────

/// CPU-side FWHT on a 256-element group. Matches the GPU-side
/// fwht_forward_256 in rdna_compute: signs1 → butterfly → scale → signs2.
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}

/// Generate FWHT sign table matching the engine's gen_fwht_signs.
/// Standard MQ4 seeds are 42 (signs1) and 1042 (signs2).
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}

/// MagnumQuant MQ3-G256: FWHT-rotated 3-bit quantization.
/// 104 bytes per 256 weights (0.406 B/w). Same binary layout as HFQ3-G256.
/// Lifted verbatim from hipfire-quantize/main.rs's `quantize_mq3g256`.
fn quantize_mq3g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                q[j] = ((group[ci + j] - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }
    output
}

/// MagnumQuant MQ4-G256: FWHT-rotated 4-bit quantization.
/// 136 bytes per 256 weights (0.531 B/w). Same binary layout as HFQ4-G256;
/// the rotation is baked into the weights so the GEMM kernel just rotates
/// the input x instead of inverse-rotating W.
fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }
    output
}

// ─── HFQ File Format ──────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;
const ARCH_ID_DFLASH_DRAFT: u32 = 20;

#[repr(u8)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    MQ4G256 = 13,
    MQ3G256 = 17,
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

// ─── Model discovery ───────────────────────────────────────────────────────

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
            let cache_root = format!(
                "{home}/.cache/huggingface/hub/models--{org}--{name}/snapshots"
            );
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

// ─── Tensor classification ────────────────────────────────────────────────

/// Returns true for tensors that must stay in F32 for numerical fidelity:
/// any RMSNorm weight. The rest (Q/K/V/O/fc/gate/up/down projections) can
/// be cast to F16.
fn is_norm_tensor(name: &str) -> bool {
    name.contains("input_layernorm")
        || name.contains("post_attention_layernorm")
        || name.contains("q_norm")
        || name.contains("k_norm")
        || name == "hidden_norm.weight"
        || name == "norm.weight"
}

fn parse_int_array(json: &serde_json::Value) -> Vec<i64> {
    json.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_i64())
                .collect()
        })
        .unwrap_or_default()
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input_dir: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut keep_f32 = false;
    let mut use_mq4 = false;
    let mut use_mq3 = false;

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
            "--keep-f32" => {
                keep_f32 = true;
                i += 1;
            }
            "--mq4" => {
                use_mq4 = true;
                i += 1;
            }
            "--mq3" => {
                use_mq3 = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: dflash_convert --input <dir_or_hf_id> --output <file.hfq> [--keep-f32 | --mq4 | --mq3]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let n_format_flags = (keep_f32 as u8) + (use_mq4 as u8) + (use_mq3 as u8);
    if n_format_flags > 1 {
        eprintln!("--keep-f32, --mq4, and --mq3 are mutually exclusive");
        std::process::exit(1);
    }

    let input_dir = input_dir.expect("--input required");
    let output_path = output_path.expect("--output required");
    let input_dir = resolve_model_path(&input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(&output_path);

    eprintln!("dflash_convert");
    eprintln!("  input : {}", input_dir.display());
    eprintln!("  output: {}", output_path.display());
    let dtype_desc = if keep_f32 {
        "F32"
    } else if use_mq4 {
        "MQ4-G256 (weights), F32 (norms)"
    } else if use_mq3 {
        "MQ3-G256 (weights), F32 (norms)"
    } else {
        "F16 (weights), F32 (norms)"
    };
    eprintln!("  dtype : {}", dtype_desc);

    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_path.display()));
    let config: serde_json::Value =
        serde_json::from_str(&config_str).expect("config.json parse failed");

    let architectures = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let is_dflash = architectures
        .iter()
        .any(|v| v.as_str() == Some("DFlashDraftModel"));
    if !is_dflash {
        eprintln!(
            "warning: config.json architectures = {architectures:?}; expected DFlashDraftModel"
        );
    }

    let dflash_cfg = config
        .get("dflash_config")
        .expect("config.json missing dflash_config block");
    let block_size = config
        .get("block_size")
        .and_then(|v| v.as_u64())
        .expect("config.json missing block_size") as u32;
    let mask_token_id = dflash_cfg
        .get("mask_token_id")
        .and_then(|v| v.as_u64())
        .expect("dflash_config missing mask_token_id") as u32;
    let target_layer_ids = parse_int_array(
        dflash_cfg
            .get("target_layer_ids")
            .expect("dflash_config missing target_layer_ids"),
    );
    let num_target_layers = config
        .get("num_target_layers")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_target_layers");

    let num_hidden_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_hidden_layers") as usize;
    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .expect("config.json missing hidden_size") as usize;
    let num_attention_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_attention_heads") as usize;
    let num_key_value_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_key_value_heads") as usize;
    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = config
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .expect("config.json missing intermediate_size") as usize;

    eprintln!(
        "  dflash: block_size={}, mask_token_id={}, target_layers={:?}, hidden_layers={}, hidden={}",
        block_size, mask_token_id, target_layer_ids, num_hidden_layers, hidden_size,
    );

    // Metadata JSON for the HFQ file.
    let draft_dtype = if keep_f32 { "f32" } else if use_mq4 { "mq4" } else if use_mq3 { "mq3" } else { "f16" };
    // FWHT sign tables for MQ4 rotation. Seeds 42/1042 match the engine's
    // `rdna_compute::Gpu::ensure_mq_signs()` so quantized weights here can
    // be dequantized/used correctly on GPU at inference.
    let needs_fwht = use_mq4 || use_mq3;
    let signs1: Vec<f32> = if needs_fwht { gen_fwht_signs(42, 256) } else { Vec::new() };
    let signs2: Vec<f32> = if needs_fwht { gen_fwht_signs(1042, 256) } else { Vec::new() };
    let metadata = serde_json::json!({
        "architecture": "dflash",
        "config": config,
        "dflash": {
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids,
            "num_target_layers": num_target_layers,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "rms_norm_eps": config.get("rms_norm_eps").cloned().unwrap_or_else(|| serde_json::Value::from(1e-6)),
            "rope_theta": config.get("rope_theta").cloned().unwrap_or_else(|| serde_json::Value::from(10_000_000.0)),
            "vocab_size": config.get("vocab_size").cloned(),
            "draft_dtype": draft_dtype,
        },
        "tokenizer": serde_json::Value::Null,
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load + convert all safetensors files (draft is typically one file).
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .inspect(|p| eprintln!("  loading: {}", p.display()))
        .map(|p| SafetensorsFile::open(p).expect("safetensors open failed"))
        .collect();
    assert!(!st_files.is_empty(), "no .safetensors files found in input dir");

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
    let mut total_bytes_out = 0usize;

    for (name, fi) in &name_to_file {
        let (meta, raw) = st_files[*fi].tensor_data(name).expect("tensor lookup failed");
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        let f32_data = to_f32(raw, &meta.dtype);

        // Classification rules:
        //   norms → always F32 (small, precision-critical).
        //   other (projections) → F32 if --keep-f32,
        //                         MQ4-G256 if --mq4 (and innermost ≥ 256 divisible),
        //                         else F16.
        // MQ4 divisibility: quantize_mq4g256 pads the final partial group with
        // zeros. That's safe for weights since the padded lanes are never read
        // at inference. We still require N ≥ 256 to ensure a full first group
        // (per-group scale/min carries meaning).
        let (quant_type, group_size, bytes) = if is_norm_tensor(name) {
            (QuantType::F32, 0u32, f32_slice_to_f32_bytes(&f32_data))
        } else if keep_f32 {
            (QuantType::F32, 0u32, f32_slice_to_f32_bytes(&f32_data))
        } else if use_mq4 && n_elements >= 256 {
            let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
            (QuantType::MQ4G256, 256u32, q)
        } else if use_mq3 && n_elements >= 256 {
            let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
            (QuantType::MQ3G256, 256u32, q)
        } else {
            (QuantType::F16, 0u32, f32_slice_to_f16_bytes(&f32_data))
        };

        total_bytes_out += bytes.len();
        hfq_tensors.push(HfqTensor {
            name: name.clone(),
            quant_type,
            shape: meta.shape.iter().map(|d| *d as u32).collect(),
            group_size,
            data: bytes,
        });
    }

    eprintln!(
        "  total params: {:.3}B ({} tensors)",
        total_params as f64 / 1e9,
        hfq_tensors.len()
    );
    eprintln!("  total out  : {:.2} MiB", total_bytes_out as f64 / (1024.0 * 1024.0));

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir -p output parent");
        }
    }

    write_hfq(
        output_path,
        ARCH_ID_DFLASH_DRAFT,
        &metadata_json,
        &hfq_tensors,
    )
    .expect("write_hfq failed");

    eprintln!("  wrote: {}", output_path.display());
}
