// SPDX-License-Identifier: Apache-2.0
//! Minimal `.hfq` (HFQM container) reader + in-place norm patcher for Path-A
//! export. We don't re-serialize the container — recovery only changes the fp
//! RMSNorm weights, which are stored BF16 at fixed offsets/sizes, so we overwrite
//! those bytes in place (same size) and leave codes/index/header untouched.
//!
//! Format (from hipfire-quantize write_hfq / its reader):
//!   [0:4]   "HFQM"      [4:8] version u32   [8:12] arch u32
//!   [12:16] n_tensors u32  [16:24] metadata_offset u64  [24:32] data_offset u64
//!   metadata: brace-matched JSON at metadata_offset
//!   index @ (metadata_offset + json_end): u32 count, then per tensor:
//!     u16 name_len, name, u8 quant_type, u8 n_dims, n_dims×u32 shape,
//!     u32 group_size, u64 data_size
//!   data @ data_offset (4096-aligned): tensors concatenated in index order.

use std::collections::HashMap;

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const QT_BF16: u8 = 16; // QuantType::BF16 (=16; norms + down_proj are stored BF16)

#[derive(Debug, Clone)]
pub struct HfqEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub data_offset: usize, // absolute byte offset into the file
    pub data_size: usize,
}

/// Parse the HFQM header + index. Returns the tensor entries (with absolute data
/// offsets) and the metadata JSON string.
pub fn parse_hfq(bytes: &[u8]) -> Result<(Vec<HfqEntry>, String), String> {
    if bytes.len() < 32 || &bytes[0..4] != HFQ_MAGIC {
        return Err("not an HFQM container".into());
    }
    let n_tensors = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;

    // brace-match the metadata JSON
    let meta = &bytes[metadata_offset..data_offset];
    let (mut depth, mut in_str, mut esc, mut json_end) = (0i32, false, false, None);
    for (i, &b) in meta.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if b == b'\\' && in_str {
            esc = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = Some(i + 1);
                    break;
                }
            }
        }
    }
    let json_end = json_end.ok_or("HFQM metadata JSON did not end")?;
    let metadata_json = String::from_utf8(meta[..json_end].to_vec()).map_err(|e| e.to_string())?;

    let mut pos = metadata_offset + json_end;
    let idx_n = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    if idx_n != n_tensors {
        return Err(format!("index count {idx_n} != header count {n_tensors}"));
    }
    pos += 4;

    let mut entries = Vec::with_capacity(n_tensors);
    let mut cum = data_offset;
    for _ in 0..n_tensors {
        let name_len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8(bytes[pos..pos + name_len].to_vec()).map_err(|e| e.to_string())?;
        pos += name_len;
        let quant_type = bytes[pos];
        pos += 1;
        let n_dims = bytes[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let _group_size = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        entries.push(HfqEntry { name, quant_type, shape, data_offset: cum, data_size });
        cum += data_size;
    }
    Ok((entries, metadata_json))
}

/// Is this tensor one of the RMSNorm weights recovery tunes?
pub fn is_norm(name: &str) -> bool {
    name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name == "model.norm.weight"
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits >> 23) & 0xFF == 0xFF {
        return (bits >> 16) as u16; // inf/nan: truncate high half
    }
    let round_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round_bias) >> 16) as u16
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Patch a parsed HFQM byte buffer in place: overwrite each BF16 norm tensor
/// named in `tuned` (name → fp32 weights) with its tuned values. Same byte size,
/// so offsets/index/codes are untouched. Returns the number of tensors patched.
pub fn patch_norms_inplace(
    bytes: &mut [u8],
    entries: &[HfqEntry],
    tuned: &HashMap<String, Vec<f32>>,
) -> Result<usize, String> {
    let mut n = 0;
    for e in entries {
        let Some(vals) = tuned.get(&e.name) else { continue };
        if e.quant_type != QT_BF16 {
            return Err(format!("{}: expected BF16 norm (qt {}), refusing", e.name, e.quant_type));
        }
        if vals.len() * 2 != e.data_size {
            return Err(format!(
                "{}: tuned len {} (×2={}) != data_size {}",
                e.name, vals.len(), vals.len() * 2, e.data_size
            ));
        }
        for (i, &v) in vals.iter().enumerate() {
            let off = e.data_offset + i * 2;
            bytes[off..off + 2].copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        n += 1;
    }
    Ok(n)
}
