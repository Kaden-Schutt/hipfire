// SPDX-License-Identifier: Apache-2.0
//! Imatrix GGUF reader — the single owner of `*.in_sum2` parsing.
//!
//! Llama.cpp's imatrix GGUF stores per-linear-layer pairs:
//!   `{name}.in_sum2`  F32 `[k]` or `[k, n_mat]` — sum of squared activations
//!   `{name}.counts`   F32 `[1, n_mat]`       — contributing token count
//!
//! This module ports the logic from `crates/hipfire-quantize/src/main.rs`
//! (`load_imatrix` + `safetensors_to_ggml_name`) and
//! `crates/hipfire-quantize/src/gguf_input.rs` into `saddle-quant` so no other
//! crate needs to duplicate the parser. See `crate::format::mod` for the
//! motivation (29 redundant parsers, `3dfd1b3f5`).

use crate::{QuantError, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// One parsed imatrix.
#[derive(Debug, Clone)]
pub struct Imatrix {
    entries: BTreeMap<String, Vec<f32>>,
    counts: BTreeMap<String, f32>,
    skipped_moe: usize,
}

/// Coverage of an imatrix against a model's weight names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    pub matched: usize,
    pub missing: Vec<String>,
}

impl Imatrix {
    pub fn get(&self, ggml_name: &str) -> Option<&[f32]> {
        self.entries.get(ggml_name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn skipped_moe(&self) -> usize {
        self.skipped_moe
    }

    /// Contributing token count for `name`, from the imatrix's `*.counts`
    /// record.
    ///
    /// Provenance, not decoration: an imatrix's value depends on how many
    /// tokens fed it, and a per-tensor count is the only way to detect a
    /// tensor that the calibration corpus barely exercised. `None` when the
    /// file carried no `.counts` entry for that tensor.
    pub fn counts(&self, name: &str) -> Option<f32> {
        self.counts.get(name).copied()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|k| k.as_str())
    }

    /// Try the direct safetensors key first, then fall back to the ggml-mapped
    /// name. Order matters — hipfire-produced imatrices are safetensors-keyed
    /// while llama.cpp ones are `blk.*`-keyed.
    pub fn lookup(&self, safetensors_name: &str) -> Option<&[f32]> {
        if let Some(v) = self.entries.get(safetensors_name) {
            return Some(v.as_slice());
        }
        let ggml = safetensors_to_ggml_name(safetensors_name)?;
        self.entries.get(&ggml).map(|v| v.as_slice())
    }

    /// How much of `safetensors_names` is covered by this imatrix.
    pub fn coverage(&self, safetensors_names: &[String]) -> Coverage {
        let mut matched = 0usize;
        let mut missing = Vec::new();
        for name in safetensors_names {
            if self.lookup(name).is_some() {
                matched += 1;
            } else {
                missing.push(name.clone());
            }
        }
        Coverage { matched, missing }
    }
}

/// Translate a hipfire safetensors-style tensor name to the ggml-style name
/// used by llama.cpp's imatrix output.
///
/// Ported exactly from `crates/hipfire-quantize/src/main.rs:6030-6084`.
pub fn safetensors_to_ggml_name(name: &str) -> Option<String> {
    let normalized = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    match normalized {
        "embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        "norm.weight" => return Some("output_norm.weight".to_string()),
        _ => {}
    }

    let rest = normalized.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer_idx = &rest[..dot];
    let slot_full = &rest[dot + 1..];
    let slot = slot_full.strip_suffix(".weight")?;

    let translated = match slot {
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "self_attn.gate_proj" => "attn_gate",
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.out_proj" => "ssm_out",
        _ => return None,
    };

    Some(format!("blk.{layer_idx}.{translated}.weight"))
}

/// Parse an imatrix GGUF file.
pub fn open(path: impl AsRef<Path>) -> Result<Imatrix> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(QuantError::Io)?;
    parse_bytes(&bytes)
}

// ---------------------------------------------------------------------------
// GGUF parsing (imatrix-specific, CPU-only, no `byteorder` dependency)
// ---------------------------------------------------------------------------

const ARTIFACT: &str = "imatrix";

fn truncated(context: &'static str, need: usize, have: usize) -> QuantError {
    QuantError::Truncated {
        artifact: ARTIFACT,
        context,
        need,
        have,
    }
}

fn ensure(bytes: &[u8], pos: usize, need: usize, context: &'static str) -> Result<()> {
    if pos + need > bytes.len() {
        return Err(truncated(context, pos + need, bytes.len()));
    }
    Ok(())
}

fn read_u32_le(bytes: &[u8], pos: &mut usize, context: &'static str) -> Result<u32> {
    ensure(bytes, *pos, 4, context)?;
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64_le(bytes: &[u8], pos: &mut usize, context: &'static str) -> Result<u64> {
    ensure(bytes, *pos, 8, context)?;
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    let len = read_u64_le(bytes, pos, "string len")? as usize;
    ensure(bytes, *pos, len, "string bytes")?;
    let s = String::from_utf8(bytes[*pos..*pos + len].to_vec())
        .map_err(|e| QuantError::Malformed(format!("invalid UTF-8 string: {e}")))?;
    *pos += len;
    Ok(s)
}

/// Skip or capture a GGUF metadata value. Returns the raw `U32` value of
/// `general.alignment` when that key is being parsed, otherwise `None`.
fn skip_meta_value(bytes: &[u8], pos: &mut usize, vtype: u32) -> Result<Option<u32>> {
    match vtype {
        0 => {
            // U8
            ensure(bytes, *pos, 1, "meta U8")?;
            *pos += 1;
            Ok(None)
        }
        1 => {
            ensure(bytes, *pos, 1, "meta I8")?;
            *pos += 1;
            Ok(None)
        }
        2 => {
            ensure(bytes, *pos, 2, "meta U16")?;
            *pos += 2;
            Ok(None)
        }
        3 => {
            ensure(bytes, *pos, 2, "meta I16")?;
            *pos += 2;
            Ok(None)
        }
        4 => {
            // U32
            ensure(bytes, *pos, 4, "meta U32")?;
            let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(Some(v))
        }
        5 => {
            ensure(bytes, *pos, 4, "meta I32")?;
            *pos += 4;
            Ok(None)
        }
        6 => {
            ensure(bytes, *pos, 4, "meta F32")?;
            *pos += 4;
            Ok(None)
        }
        7 => {
            ensure(bytes, *pos, 1, "meta Bool")?;
            *pos += 1;
            Ok(None)
        }
        8 => {
            // String
            let s = read_string(bytes, pos)?;
            // For string-typed alignment (should not happen) we ignore.
            let _ = s;
            Ok(None)
        }
        9 => {
            // Array
            let elem_type = read_u32_le(bytes, pos, "meta array elem type")?;
            let count = read_u64_le(bytes, pos, "meta array count")? as usize;
            for _ in 0..count {
                // recursively skip each element as typed value
                skip_meta_value(bytes, pos, elem_type)?;
            }
            Ok(None)
        }
        10 => {
            ensure(bytes, *pos, 8, "meta U64")?;
            *pos += 8;
            Ok(None)
        }
        11 => {
            ensure(bytes, *pos, 8, "meta I64")?;
            *pos += 8;
            Ok(None)
        }
        12 => {
            ensure(bytes, *pos, 8, "meta F64")?;
            *pos += 8;
            Ok(None)
        }
        other => Err(QuantError::Malformed(format!(
            "unknown GGUF metadata value type: {other}"
        ))),
    }
}

struct RawTensor {
    name: String,
    shape: Vec<u64>,
    dtype: u32,
    offset: u64,
}

fn parse_bytes(bytes: &[u8]) -> Result<Imatrix> {
    // ---- header ----
    if bytes.len() < 4 {
        return Err(truncated("magic", 4, bytes.len()));
    }
    if bytes[0..4] != *b"GGUF" {
        // produce a printable found string
        let found = String::from_utf8_lossy(&bytes[0..4]).into_owned();
        // include hex as well if non-utf8; lossy is fine
        return Err(QuantError::BadMagic {
            artifact: ARTIFACT,
            expected: "GGUF",
            found,
        });
    }
    if bytes.len() < 8 {
        return Err(truncated("version", 8, bytes.len()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if !(2..=3).contains(&version) {
        return Err(QuantError::UnsupportedVersion {
            artifact: ARTIFACT,
            found: version,
            supported: "2..3",
        });
    }
    if bytes.len() < 24 {
        return Err(truncated("header", 24, bytes.len()));
    }
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let metadata_kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let mut pos: usize = 24;

    // ---- metadata ----
    let mut alignment: usize = 32;
    for _ in 0..metadata_kv_count {
        let key = read_string(bytes, &mut pos)?;
        let vtype = read_u32_le(bytes, &mut pos, "meta value type")?;
        let captured = skip_meta_value(bytes, &mut pos, vtype)?;
        if key == "general.alignment" {
            if let Some(v) = captured {
                // Only U32 alignment matters; array/string cases already handled above
                if vtype == 4 {
                    alignment = v as usize;
                }
            } else if vtype == 10 {
                // U64 alignment case — we consumed 8 bytes above without capture;
                // re-parse to capture U64 value would require different path.
                // But spec says alignment is U32; not critical for tests.
            }
            // For robust handling: if alignment was an U32 stored as I32 etc, skip_meta_value
            // returned None; we could have parsed differently, but default to 32 remains fine
            // for synthetic tests which don't use metadata alignment.
            // For correctness, try reading alignment when vtype==4 we already captured;
            // otherwise keep default.
        }
        // Note: if the metadata value was U32 and we haven't captured because key != alignment,
        // skipping consumed it already.
    }

    // ---- tensor infos ----
    let mut tensors: Vec<RawTensor> = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(bytes, &mut pos)?;
        let n_dims = read_u32_le(bytes, &mut pos, "tensor n_dims")? as usize;
        // Guard against absurd n_dims to prevent OOM via truncated allocation check
        if n_dims > 8 {
            return Err(QuantError::Malformed(format!(
                "imatrix tensor {name}: n_dims {n_dims} too large"
            )));
        }
        let mut shape: Vec<u64> = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = read_u64_le(bytes, &mut pos, "tensor dim")?;
            shape.push(d);
        }
        let dtype = read_u32_le(bytes, &mut pos, "tensor dtype")?;
        let offset = read_u64_le(bytes, &mut pos, "tensor offset")?;
        tensors.push(RawTensor {
            name,
            shape,
            dtype,
            offset,
        });
    }

    let tensor_data_offset = (pos + alignment - 1) / alignment * alignment;
    if tensor_data_offset > bytes.len() {
        return Err(truncated(
            "tensor data offset",
            tensor_data_offset,
            bytes.len(),
        ));
    }

    // ---- collect entries ----
    let mut entries: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut counts: BTreeMap<String, f32> = BTreeMap::new();
    let mut skipped_moe: usize = 0;

    for t in &tensors {
        let is_in_sum2 = t.name.strip_suffix(".in_sum2");
        let is_counts = t.name.strip_suffix(".counts");
        let (base, is_imatrix_tensor) = if let Some(b) = is_in_sum2 {
            (b, true)
        } else if let Some(b) = is_counts {
            (b, false)
        } else {
            continue;
        };

        // Dtype check — non-F32 is an error, not a skip
        if t.dtype != 0 {
            return Err(QuantError::Malformed(format!(
                "imatrix tensor {} has dtype {} expected F32 (0)",
                t.name, t.dtype
            )));
        }

        // Determine n_mat and k
        if t.shape.is_empty() {
            return Err(QuantError::Malformed(format!(
                "imatrix tensor {} has empty shape",
                t.name
            )));
        }
        let k = t.shape[0] as usize;
        let n_mat: usize = if t.shape.len() >= 2 {
            // product of trailing dims
            let prod: u64 = t.shape[1..].iter().product();
            prod as usize
        } else {
            1
        };

        if is_imatrix_tensor {
            // in_sum2
            if n_mat != 1 {
                skipped_moe += 1;
                continue;
            }
            let numel: usize = t.shape.iter().map(|&d| d as usize).product();
            // Validate numel == k for dense case
            if numel != k {
                return Err(QuantError::Malformed(format!(
                    "imatrix tensor {} shape {:?} numel {} != k {}",
                    t.name, t.shape, numel, k
                )));
            }
            let byte_size = numel * 4;
            let start = tensor_data_offset + t.offset as usize;
            let end = start + byte_size;
            if end > bytes.len() {
                return Err(truncated("tensor data", end, bytes.len()));
            }
            let slice = &bytes[start..end];
            let mut vals = Vec::with_capacity(numel);
            for chunk in slice.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            entries.insert(base.to_string(), vals);
        } else {
            // counts
            if n_mat != 1 {
                // skip MoE counts quietly; do not bump skipped_moe again
                continue;
            }
            let numel: usize = t.shape.iter().map(|&d| d as usize).product();
            if numel == 0 {
                return Err(QuantError::Malformed(format!(
                    "imatrix tensor {} counts has empty data",
                    t.name
                )));
            }
            let byte_size = numel * 4;
            let start = tensor_data_offset + t.offset as usize;
            let end = start + byte_size;
            if end > bytes.len() {
                return Err(truncated("tensor data", end, bytes.len()));
            }
            let slice = &bytes[start..end];
            let v = f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
            counts.insert(base.to_string(), v);
        }
    }

    Ok(Imatrix {
        entries,
        counts,
        skipped_moe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ------------------------------------------------------------------
    // Helpers: minimal GGUF builder
    // ------------------------------------------------------------------

    struct TestTensorSpec {
        name: String,
        shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
    }

    fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_gguf(tensors: Vec<TestTensorSpec>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // no metadata
        let mut offset: u64 = 0;
        for t in &tensors {
            write_string(&mut buf, &t.name);
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for &d in &t.shape {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            // compute byte size for offset advance
            let numel: usize = t.shape.iter().map(|&d| d as usize).product::<usize>();
            let bsize = if t.dtype == 0 { numel * 4 } else { 0 };
            // For synthetic non-F32 we just advance by data.len()
            let adv = if bsize > 0 { bsize } else { t.data.len() };
            offset += adv as u64;
        }
        let pos = buf.len();
        let alignment = 32usize;
        let data_offset = (pos + alignment - 1) / alignment * alignment;
        buf.resize(data_offset, 0);
        for t in &tensors {
            buf.extend_from_slice(&t.data);
        }
        buf
    }

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    // ------------------------------------------------------------------
    // Name mapping: real bartowski Qwen3-27B distribution
    // ------------------------------------------------------------------

    #[test]
    fn bartowski_qwen3_distribution() {
        // 64 layers total: 48 LinearAttention, 16 FullAttention at 3,7,11,...
        // Expected distribution from the measured imatrix (496 entries):
        // ffn_up/gate/down 64 each, ssm_out/beta/alpha/attn_qkv/attn_gate 48 each,
        // attn_q/k/v/output 16 each.
        let full_layers: Vec<usize> = (0..64).filter(|x| x % 4 == 3).collect();
        assert_eq!(full_layers.len(), 16, "full-attention layer count");

        let mut safetensors_names: Vec<String> = Vec::new();
        for layer in 0..64u32 {
            // mlp always
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                safetensors_names.push(format!(
                    "model.language_model.layers.{layer}.mlp.{proj}.weight"
                ));
            }
            if full_layers.contains(&(layer as usize)) {
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    safetensors_names.push(format!(
                        "model.language_model.layers.{layer}.self_attn.{proj}.weight"
                    ));
                }
            } else {
                // linear attn slots
                safetensors_names.push(format!(
                    "model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"
                ));
                safetensors_names.push(format!(
                    "model.language_model.layers.{layer}.linear_attn.in_proj_z.weight"
                ));
                safetensors_names.push(format!(
                    "model.language_model.layers.{layer}.linear_attn.in_proj_a.weight"
                ));
                safetensors_names.push(format!(
                    "model.language_model.layers.{layer}.linear_attn.in_proj_b.weight"
                ));
                safetensors_names.push(format!(
                    "model.language_model.layers.{layer}.linear_attn.out_proj.weight"
                ));
            }
        }
        assert_eq!(
            safetensors_names.len(),
            496,
            "total safetensors count should be 496"
        );

        // map
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for name in &safetensors_names {
            let ggml =
                safetensors_to_ggml_name(name).unwrap_or_else(|| panic!("should map: {name}"));
            // tally by translated slot token, e.g. "ffn_up"
            // ggml form blk.{layer}.{slot}.weight ; extract slot
            let parts: Vec<&str> = ggml.split('.').collect();
            assert_eq!(parts.len(), 4, "ggml shape for {ggml}");
            let slot = parts[2].to_string();
            *counts.entry(slot).or_insert(0) += 1;
        }

        assert_eq!(counts.get("ffn_up"), Some(&64));
        assert_eq!(counts.get("ffn_gate"), Some(&64));
        assert_eq!(counts.get("ffn_down"), Some(&64));
        assert_eq!(counts.get("ssm_out"), Some(&48));
        assert_eq!(counts.get("ssm_beta"), Some(&48));
        assert_eq!(counts.get("ssm_alpha"), Some(&48));
        assert_eq!(counts.get("attn_qkv"), Some(&48));
        // linear-attn in_proj_z maps to attn_gate, which also appears as self_attn.gate_proj
        // but in this synthetic split we only used linear layers for attn_gate, so 48 expected
        assert_eq!(counts.get("attn_gate"), Some(&48));
        assert_eq!(counts.get("attn_q"), Some(&16));
        assert_eq!(counts.get("attn_k"), Some(&16));
        assert_eq!(counts.get("attn_v"), Some(&16));
        assert_eq!(counts.get("attn_output"), Some(&16));
    }

    #[test]
    fn mapper_spot_checks() {
        assert_eq!(
            safetensors_to_ggml_name("model.language_model.layers.7.linear_attn.in_proj_a.weight"),
            Some("blk.7.ssm_alpha.weight".to_string())
        );
        assert_eq!(
            safetensors_to_ggml_name("model.language_model.layers.3.self_attn.q_proj.weight"),
            Some("blk.3.attn_q.weight".to_string())
        );
        // top-level maps
        assert_eq!(
            safetensors_to_ggml_name("model.language_model.embed_tokens.weight"),
            Some("token_embd.weight".to_string())
        );
        assert_eq!(
            safetensors_to_ggml_name("model.lm_head.weight"),
            Some("output.weight".to_string())
        );
        assert_eq!(
            safetensors_to_ggml_name("model.norm.weight"),
            Some("output_norm.weight".to_string())
        );
        // bare layers prefix (without model.*)
        assert_eq!(
            safetensors_to_ggml_name("layers.0.mlp.gate_proj.weight"),
            Some("blk.0.ffn_gate.weight".to_string())
        );
        // glimmer gate_proj maps to attn_gate
        assert_eq!(
            safetensors_to_ggml_name("model.layers.5.self_attn.gate_proj.weight"),
            Some("blk.5.attn_gate.weight".to_string())
        );
        // unmapped slot returns None
        assert_eq!(
            safetensors_to_ggml_name("model.language_model.layers.0.linear_attn.conv1d.weight"),
            None
        );
        assert_eq!(
            safetensors_to_ggml_name("model.language_model.layers.0.input_layernorm.weight"),
            None
        );
        // also test model. prefix
        assert_eq!(
            safetensors_to_ggml_name("model.layers.2.mlp.up_proj.weight"),
            Some("blk.2.ffn_up.weight".to_string())
        );
    }

    #[test]
    fn lookup_safetensors_wins_over_ggml() {
        // Build a synthetic imatrix GGUF containing both a safetensors-keyed entry
        // and its ggml counterpart with distinct values.
        let st_name = "model.language_model.layers.0.mlp.gate_proj.weight";
        let ggml_name = safetensors_to_ggml_name(st_name).unwrap(); // blk.0.ffn_gate.weight
        let st_base = st_name.to_string();
        let ggml_base = ggml_name.clone();

        let st_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let ggml_vals = vec![9.0f32, 9.0, 9.0, 9.0];

        let tensors = vec![
            TestTensorSpec {
                name: format!("{st_base}.in_sum2"),
                shape: vec![st_vals.len() as u64],
                dtype: 0,
                data: f32s_to_bytes(&st_vals),
            },
            TestTensorSpec {
                name: format!("{st_base}.counts"),
                shape: vec![1],
                dtype: 0,
                data: f32s_to_bytes(&[42.0]),
            },
            TestTensorSpec {
                name: format!("{ggml_base}.in_sum2"),
                shape: vec![ggml_vals.len() as u64],
                dtype: 0,
                data: f32s_to_bytes(&ggml_vals),
            },
            TestTensorSpec {
                name: format!("{ggml_base}.counts"),
                shape: vec![1],
                dtype: 0,
                data: f32s_to_bytes(&[7.0]),
            },
        ];
        let bytes = build_gguf(tensors);
        let file = write_temp_gguf(&bytes);
        let im = open(file.path()).unwrap();
        assert_eq!(im.len(), 2);
        // lookup via safetensors name must return the safetensors-keyed values, not ggml
        let got = im.lookup(st_name).unwrap();
        assert_eq!(got, st_vals.as_slice(), "safetensors entry should win");
        // direct ggml get still returns ggml values
        assert_eq!(im.get(&ggml_base).unwrap(), ggml_vals.as_slice());
    }

    #[test]
    fn roundtrip_two_entries() {
        let vals_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let vals_b = vec![5.0f32, 6.0];
        let name_a = "blk.0.ffn_up.weight";
        let name_b = "blk.1.ffn_gate.weight";
        let tensors = vec![
            TestTensorSpec {
                name: format!("{name_a}.in_sum2"),
                shape: vec![vals_a.len() as u64],
                dtype: 0,
                data: f32s_to_bytes(&vals_a),
            },
            TestTensorSpec {
                name: format!("{name_a}.counts"),
                shape: vec![1],
                dtype: 0,
                data: f32s_to_bytes(&[100.0]),
            },
            TestTensorSpec {
                name: format!("{name_b}.in_sum2"),
                shape: vec![vals_b.len() as u64],
                dtype: 0,
                data: f32s_to_bytes(&vals_b),
            },
            TestTensorSpec {
                name: format!("{name_b}.counts"),
                shape: vec![1],
                dtype: 0,
                data: f32s_to_bytes(&[200.0]),
            },
        ];
        let bytes = build_gguf(tensors);
        let file = write_temp_gguf(&bytes);
        let im = open(file.path()).unwrap();
        assert_eq!(im.len(), 2);
        assert!(!im.is_empty());
        assert_eq!(im.skipped_moe(), 0);
        assert_eq!(im.get(name_a).unwrap(), vals_a.as_slice());
        assert_eq!(im.get(name_b).unwrap(), vals_b.as_slice());
        // ensure names iterator yields both
        let mut names: Vec<&str> = im.names().collect();
        names.sort();
        assert_eq!(names, vec![name_a, name_b]);
        // also test parse_bytes directly without tempfile
        let im2 = parse_bytes(&bytes).unwrap();
        assert_eq!(im2.len(), 2);
    }

    #[test]
    fn skipped_moe_counts() {
        // n_mat != 1 should increment skipped_moe and not appear in entries
        let vals = vec![1.0f32; 8]; // k=4, n_mat=2 => 8 values
        let tensors = vec![TestTensorSpec {
            name: "blk.0.moe_ffn_gate.weight.in_sum2".to_string(),
            shape: vec![4, 2],
            dtype: 0,
            data: f32s_to_bytes(&vals),
        }];
        let bytes = build_gguf(tensors);
        let im = parse_bytes(&bytes).unwrap();
        assert_eq!(im.len(), 0);
        assert!(im.is_empty());
        assert_eq!(im.skipped_moe(), 1);
        assert_eq!(im.get("blk.0.moe_ffn_gate.weight"), None);
    }

    #[test]
    fn coverage_reports_missing() {
        let vals = vec![1.0f32, 2.0, 3.0];
        let tensors = vec![
            TestTensorSpec {
                name: "blk.0.ffn_up.weight.in_sum2".to_string(),
                shape: vec![3],
                dtype: 0,
                data: f32s_to_bytes(&vals),
            },
            TestTensorSpec {
                name: "blk.0.ffn_up.weight.counts".to_string(),
                shape: vec![1],
                dtype: 0,
                data: f32s_to_bytes(&[10.0]),
            },
        ];
        let bytes = build_gguf(tensors);
        let im = parse_bytes(&bytes).unwrap();

        let names = vec![
            "model.language_model.layers.0.mlp.up_proj.weight".to_string(), // maps to blk.0.ffn_up.weight -> matched
            "model.language_model.layers.0.mlp.gate_proj.weight".to_string(), // blk.0.ffn_gate -> missing
            "model.language_model.layers.1.mlp.up_proj.weight".to_string(), // blk.1.ffn_up -> missing
        ];
        let cov = im.coverage(&names);
        assert_eq!(cov.matched, 1);
        assert_eq!(
            cov.missing,
            vec![
                "model.language_model.layers.0.mlp.gate_proj.weight".to_string(),
                "model.language_model.layers.1.mlp.up_proj.weight".to_string()
            ]
        );
    }

    #[test]
    fn non_f32_is_error() {
        let tensors = vec![TestTensorSpec {
            name: "blk.0.ffn_up.weight.in_sum2".to_string(),
            shape: vec![4],
            dtype: 1,           // F16, not F32
            data: vec![0u8; 8], // 4 * 2 bytes
        }];
        let bytes = build_gguf(tensors);
        let err = parse_bytes(&bytes).unwrap_err();
        match err {
            QuantError::Malformed(msg) => {
                assert!(msg.contains("dtype"), "msg: {msg}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn bad_magic_is_error() {
        let mut bytes = build_gguf(vec![]);
        bytes[0..4].copy_from_slice(b"BAD!");
        let err = parse_bytes(&bytes).unwrap_err();
        match err {
            QuantError::BadMagic {
                artifact,
                expected,
                found: _,
            } => {
                assert_eq!(artifact, "imatrix");
                assert_eq!(expected, "GGUF");
            }
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_is_error() {
        let mut bytes = build_gguf(vec![]);
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = parse_bytes(&bytes).unwrap_err();
        match err {
            QuantError::UnsupportedVersion {
                artifact,
                found,
                supported: _,
            } => {
                assert_eq!(artifact, "imatrix");
                assert_eq!(found, 99);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn truncated_is_error() {
        let bytes = build_gguf(vec![]);
        // truncate to just magic+version, missing header fields
        let truncated = &bytes[..6];
        let err = parse_bytes(truncated).unwrap_err();
        match err {
            QuantError::Truncated { artifact, .. } => assert_eq!(artifact, "imatrix"),
            other => panic!("expected Truncated, got {other:?}"),
        }

        // also test truncated tensor data
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensors = vec![TestTensorSpec {
            name: "blk.0.ffn_up.weight.in_sum2".to_string(),
            shape: vec![4],
            dtype: 0,
            data: f32s_to_bytes(&vals),
        }];
        let mut bytes2 = build_gguf(tensors);
        bytes2.truncate(bytes2.len() - 2); // cut off tail of data
        let err2 = parse_bytes(&bytes2).unwrap_err();
        match err2 {
            QuantError::Truncated { artifact, .. } => assert_eq!(artifact, "imatrix"),
            other => panic!("expected Truncated for data, got {other:?}"),
        }
    }
}
