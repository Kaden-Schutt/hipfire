//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir> --output <output.hfq> [--format q4f16-g64]
//!
//! Reads safetensors files from a HuggingFace model directory and produces
//! a .hfq (HipFire Quantized) file with RDNA-native Q4_F16 quantized weights.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors Parser ─────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct SafetensorsMeta {
    #[serde(flatten)]
    tensors: HashMap<String, TensorMeta>,
}

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

        // First 8 bytes: u64 LE header size
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len]).unwrap();

        // Parse header, filtering out __metadata__ key
        let raw: serde_json::Value = serde_json::from_str(header_json).unwrap();
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(v).unwrap();
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

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

// ─── FP16/BF16 Conversion ───────────────────────────────────────────────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
    if new_exp >= 31 { return ((sign << 15) | (0x1F << 10)) as u16; }
    if new_exp <= 0 {
        if new_exp < -10 { return (sign << 15) as u16; }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Convert raw tensor bytes to F32 based on dtype string
fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => {
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "BF16" => {
            data.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "F32" => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        other => panic!("unsupported dtype: {other}"),
    }
}

// ─── Q4_F16_G64 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to Q4_F16_G64 format.
/// Group size 64: 36 bytes per 64 elements (0.5625 bytes/weight).
/// Block: f16 scale (2B) + f16 min (2B) + u8[32] packed nibbles (32B).
fn quantize_q4f16_g64(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 64;
    let block_bytes = 36;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(min_val).to_le_bytes());

        let actual_len = end - start;
        for i in 0..32 {
            let lo_val = if i < actual_len { group[i] } else { min_val };
            let hi_val = if 32 + i < actual_len { group[32 + i] } else { min_val };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 4 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

// ─── Q4_K Quantization (GGML-compatible) ─────────────────────────────────────

/// Quantize F32 weights to Q4_K format (144 bytes per 256 elements, 0.5625 B/w).
/// GGML-compatible block layout: f16 d + f16 dmin + 12B packed scales + 128B nibbles.
/// This produces blocks that work with the existing gemv_q4k kernel.
fn quantize_q4k(f32_data: &[f32]) -> Vec<u8> {
    let super_block_size = 256;
    let block_bytes = 144;
    let n = f32_data.len();
    let n_blocks = (n + super_block_size - 1) / super_block_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let sb_start = b * super_block_size;
        let sb_end = (sb_start + super_block_size).min(n);
        let out_off = b * block_bytes;

        // Compute per-sub-block scales and mins (8 sub-blocks of 32 elements)
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for sb in 0..8 {
            let start = sb_start + sb * 32;
            let end = (start + 32).min(sb_end);
            if start >= sb_end { break; }
            let group = &f32_data[start..end];

            let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;
            sub_scales[sb] = if range > 0.0 { range / 15.0 } else { 0.0 };
            sub_mins[sb] = min_val;
        }

        // Find super-block d and dmin that best represent the sub-block scales/mins
        // d * scale_int ≈ sub_scale, dmin * min_int ≈ -sub_min (where sub_min is negative offset)
        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().map(|m| -m).fold(0.0f32, f32::max); // mins are typically negative

        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 }; // 6-bit scale range
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

        // Quantize sub-block scales/mins to 6-bit integers
        let mut scale_ints = [0u8; 8];
        let mut min_ints = [0u8; 8];
        for sb in 0..8 {
            scale_ints[sb] = (sub_scales[sb] * inv_d + 0.5).min(63.0) as u8;
            min_ints[sb] = ((-sub_mins[sb]) * inv_dmin + 0.5).min(63.0) as u8;
        }

        // Write super-block header
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // Pack 6-bit scales/mins into 12 bytes (GGML encoding)
        let sc = &mut output[out_off + 4..out_off + 16];
        // First 4 sub-blocks: lower 6 bits in bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            sc[i] = (scale_ints[i] & 63) | ((scale_ints[4 + i] >> 4) << 6);
            sc[4 + i] = (min_ints[i] & 63) | ((min_ints[4 + i] >> 4) << 6);
        }
        // Remaining bits in bytes 8-11
        for i in 0..4 {
            sc[8 + i] = (scale_ints[4 + i] & 0xF) | ((min_ints[4 + i] & 0xF) << 4);
        }

        // Quantize and pack nibbles (128 bytes for 256 elements)
        // Layout: 4 groups of 32 bytes. Group g covers elements g*64..g*64+63.
        // Byte l in group g: low nibble = elem g*64+l, high nibble = elem g*64+32+l.
        let qs = &mut output[out_off + 16..out_off + 144];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;

            let eff_scale_e = d * scale_ints[sb_even] as f32;
            let eff_min_e = dmin * min_ints[sb_even] as f32;
            let inv_se = if eff_scale_e > 0.0 { 1.0 / eff_scale_e } else { 0.0 };

            let eff_scale_o = d * scale_ints[sb_odd] as f32;
            let eff_min_o = dmin * min_ints[sb_odd] as f32;
            let inv_so = if eff_scale_o > 0.0 { 1.0 / eff_scale_o } else { 0.0 };

            for l in 0..32 {
                let idx_e = sb_start + group * 64 + l;
                let idx_o = sb_start + group * 64 + 32 + l;

                let val_e = if idx_e < sb_end { f32_data[idx_e] } else { 0.0 };
                let val_o = if idx_o < sb_end { f32_data[idx_o] } else { 0.0 };

                let q_e = ((val_e + eff_min_e) * inv_se + 0.5).max(0.0).min(15.0) as u8;
                let q_o = ((val_o + eff_min_o) * inv_so + 0.5).max(0.0).min(15.0) as u8;

                qs[group * 32 + l] = q_e | (q_o << 4);
            }
        }
    }

    output
}

// ─── Q8_FP16 Quantization ────────────────────────────────────────────────────

/// Quantize to Q4-as-Q8: 4-bit precision (range [-8,7]) stored in Q8_0 format.
/// Same storage as Q8 (34 bytes per 32 elements, 1.0625 B/w) but values use only 4 bits.
/// Gets Q8 kernel speed (82% peak BW) with 4-bit quality. Best for VRAM-fitting models.
fn quantize_q4_as_q8(f32_data: &[f32]) -> Vec<u8> {
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
        let scale = max_abs / 7.0; // 4-bit symmetric: -8 to 7
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-8.0).min(7.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

/// Quantize F32 weights to Q8_0 format (compatible with GGML Q8_0).
/// Block: f16 scale (2B) + 32 × int8 = 34 bytes per 32 elements (1.0625 bytes/weight).
/// Symmetric quantization: scale = max(|w|) / 127, q = round(w / scale).
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

// ─── Q8_HFQ Quantization (Split-Metadata Row Layout) ─────────────────────────

/// Quantize F32 weights to Q8_HFQ format (split-metadata, 128B-aligned rows).
/// Row layout: [f16 scales × n_groups | int8 values × K | padding to 128B].
/// Returns (data, row_stride). Same 1.0625 B/w as Q8_0 for K=2048/4096 (zero padding waste).
fn quantize_q8hfq(f32_data: &[f32], m: usize, k: usize) -> (Vec<u8>, usize) {
    let group_size = 32;
    let n_groups = k / group_size;
    let scales_bytes = n_groups * 2;
    let raw_row = scales_bytes + k;
    let row_stride = (raw_row + 127) & !127; // pad to 128-byte boundary

    let mut output = vec![0u8; m * row_stride];

    for row in 0..m {
        let row_data = &f32_data[row * k..(row + 1) * k];
        let row_out = &mut output[row * row_stride..(row + 1) * row_stride];

        for g in 0..n_groups {
            let start = g * group_size;
            let group = &row_data[start..start + group_size];

            let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write f16 scale into scale array
            row_out[g * 2..g * 2 + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

            // Write int8 values into value array (after all scales)
            for i in 0..group_size {
                let q = (group[i] * inv_scale).round().max(-128.0).min(127.0) as i8;
                row_out[scales_bytes + start + i] = q as u8;
            }
        }
    }

    (output, row_stride)
}

// ─── HFQ File Format ────────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy)]
enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
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
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
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

    // Write tensor data
    for t in tensors {
        f.write_all(&t.data)?;
    }

    Ok(())
}

// ─── Model Discovery ────────────────────────────────────────────────────────

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

/// Determine which tensors to quantize (weight matrices) vs keep as F16 (norms, embeddings)
fn should_quantize(name: &str) -> bool {
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    // Quantize everything including embeddings (Q8 embedding saves ~2.3GB for 8B models)
    name.contains("weight")
}

/// For mixed quant: should this tensor be Q8 (fast) or Q4 (compressed)?
/// Q8: attention weights, embeddings, lm_head (need occupancy)
/// Q4: FFN weights (bulk of model, benefits from compression)
fn is_q8_tensor(name: &str) -> bool {
    name.contains("self_attn") || name.contains("attn_q") || name.contains("attn_k")
        || name.contains("attn_v") || name.contains("attn_output")
        || name.contains("q_proj") || name.contains("k_proj")
        || name.contains("v_proj") || name.contains("o_proj")
        || name.contains("embed") || name.contains("lm_head")
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_dir = args.iter().position(|a| a == "--input")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir> --output <output.hfq>"); std::process::exit(1); });

    let output_path = args.iter().position(|a| a == "--output")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir> --output <output.hfq> [--format q8f16|q4f16]"); std::process::exit(1); });

    let format = args.iter().position(|a| a == "--format")
        .map(|i| args[i + 1].as_str())
        .unwrap_or("q8f16");
    // q8f16 = all weights Q8 (interleaved blocks)
    // q4f16 = all weights Q4_F16_G64
    // q8-mixed = Q8 attn + Q4_K FFN (best tok/s for VRAM-constrained)
    // q8-fast = Q8 attn + Q4-as-Q8 FFN (all Q8 occupancy, most VRAM)
    // q8hfq = all weights Q8_HFQ (split-metadata, 128B-aligned rows)
    let use_q8 = format == "q8f16" || format == "q8";
    let use_mixed = format == "q8-mixed" || format == "mixed";
    let use_fast = format == "q8-fast" || format == "fast";
    let use_q8hfq = format == "q8hfq";
    let use_q4k_all = format == "q4k";
    let use_q4k_q8embed = format == "q4k-q8embed";

    let input_dir = Path::new(input_dir);
    let output_path = Path::new(output_path);

    // Read model config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|_| panic!("Cannot read {}", config_path.display()));
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let arch_str = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("llama");
    let arch_id = match arch_str {
        "llama" => 0u32,
        "qwen3" | "qwen2" => 1,
        other => { eprintln!("Warning: unknown architecture '{other}', treating as llama"); 0 }
    };
    eprintln!("Architecture: {arch_str} (id={arch_id})");

    // Read tokenizer if present
    let tokenizer_json = input_dir.join("tokenizer.json");
    let tokenizer_str = if tokenizer_json.exists() {
        std::fs::read_to_string(&tokenizer_json).ok()
    } else {
        None
    };

    // Build metadata JSON for .hfq
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load all safetensors files
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    // Collect all tensor names
    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!("Found {} tensors", all_tensors.len());

    // Quantize
    let mut hfq_tensors = Vec::new();
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f32;
    let mut _n_quant_groups = 0u64;

    for (name, file_idx) in &all_tensors {
        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        if should_quantize(name) && n_elements >= 32 {
            let f32_data = to_f32(raw_data, &meta.dtype);
            quantized_params += n_elements as u64;

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

            // Q8HFQ path: split-metadata per-row layout (needs M and K)
            // Exclude embeddings — they use a lookup kernel, not GEMV
            if use_q8hfq && meta.shape.len() == 2 && !name.contains("embed_tokens") {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let (quantized, row_stride) = quantize_q8hfq(&f32_data, m, k);

                // Compute quantization error for Q8HFQ
                let n_groups = k / 32;
                let scales_bytes = n_groups * 2;
                for row in 0..m {
                    let row_off = row * row_stride;
                    for g in 0..n_groups {
                        let scale = f16_to_f32(u16::from_le_bytes([
                            quantized[row_off + g * 2],
                            quantized[row_off + g * 2 + 1],
                        ]));
                        for i in 0..32 {
                            let qval = quantized[row_off + scales_bytes + g * 32 + i] as i8;
                            let dequant = scale * qval as f32;
                            let orig_idx = row * k + g * 32 + i;
                            let err = (dequant - f32_data[orig_idx]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                        _n_quant_groups += 1;
                    }
                }

                eprintln!("  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB, stride={})",
                    "Q8_HFQ", name, meta.shape, n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0,
                    row_stride);

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::Q8HFQ,
                    shape,
                    group_size: 32,
                    data: quantized,
                });
            } else {
            // Choose quant format per tensor
            let this_q8 = if use_q4k_all {
                false // everything Q4_K
            } else if use_q4k_q8embed {
                name.contains("embed") || name.contains("lm_head") // only embed/output Q8
            } else if use_mixed || use_fast {
                is_q8_tensor(name)
            } else {
                use_q8 || use_q8hfq // 1D Q8HFQ tensors fall back to Q8F16
            };
            let this_q4as8 = use_fast && !this_q8; // FFN tensors in q8-fast mode
            let this_q4k = use_q4k_all || use_q4k_q8embed || use_mixed;

            let (quantized, qt, gs, label) = if this_q8 {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_FP16")
            } else if this_q4as8 {
                let q = quantize_q4_as_q8(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q4asQ8")
            } else if this_q4k {
                let q = quantize_q4k(&f32_data);
                (q, QuantType::Q4K, 256u32, "Q4_K")
            } else {
                let q = quantize_q4f16_g64(&f32_data);
                (q, QuantType::Q4F16G64, 64u32, "Q4_F16")
            };

            // Compute quantization error
            let block_size = gs as usize;
            let n_blocks = (n_elements + block_size - 1) / block_size;
            for b in 0..n_blocks {
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                if this_q8 || this_q4as8 {
                    let off = b * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    for i in 0..(end - start) {
                        let qval = quantized[off + 2 + i] as i8;
                        let dequant = scale * qval as f32;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                } else {
                    let off = b * 36;
                    let scale = f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    let min_val = f16_to_f32(u16::from_le_bytes([quantized[off + 2], quantized[off + 3]]));
                    for i in 0..(end - start) {
                        let byte_idx = if i < 32 { i } else { i - 32 };
                        let nibble = if i < 32 {
                            quantized[off + 4 + byte_idx] & 0xF
                        } else {
                            quantized[off + 4 + byte_idx] >> 4
                        };
                        let dequant = nibble as f32 * scale + min_val;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                }
                _n_quant_groups += 1;
            }

            eprintln!("  {label:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB)",
                name, meta.shape, n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0);

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
            });
            } // end else (non-Q8HFQ path)
        } else {
            // Keep as F16 (convert BF16 → F16 if needed)
            let f16_data = match meta.dtype.as_str() {
                "F16" => raw_data.to_vec(),
                "BF16" => {
                    // BF16 → F32 → F16
                    let f32_vals = to_f32(raw_data, "BF16");
                    f32_vals.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                "F32" => {
                    let f32_vals = to_f32(raw_data, "F32");
                    f32_vals.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                other => panic!("unsupported dtype for norm/embd: {other}"),
            };

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  F16:        {} {:?} ({} elements, {:.1} KB)",
                name, meta.shape, n_elements, f16_data.len() as f64 / 1024.0);

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_data,
            });
        }
    }

    // Summary
    let total_bytes: usize = hfq_tensors.iter().map(|t| t.data.len()).sum();
    let mean_quant_error = if quantized_params > 0 {
        total_quant_error / quantized_params as f64
    } else { 0.0 };

    eprintln!("\n=== Quantization Summary ===");
    eprintln!("  Total params:     {total_params}");
    eprintln!("  Quantized params: {quantized_params} ({:.1}%)", 100.0 * quantized_params as f64 / total_params as f64);
    eprintln!("  Mean quant error: {mean_quant_error:.8}");
    eprintln!("  Max quant error:  {max_quant_error:.8}");
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);

    // Write .hfq file
    eprintln!("\nWriting: {}", output_path.display());
    write_hfq(output_path, arch_id, &metadata_json, &hfq_tensors).unwrap();

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}
