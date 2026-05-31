//! mtp_extract: Extract Qwen3.5/3.6 dense MTP head from a HuggingFace
//! safetensors directory and pack into a single hipfire `.hfq` file
//! (arch_id = 21, `QWEN35_MTP_HEAD`).
//!
//! Usage:
//!     mtp_extract --hf-dir <safetensors_dir> --output <out.mtp>
//!                 [--quant {mq4,q8}] [--verbose]
//!
//! Empirical: every released dense Qwen3.5 (0.8B / 2B / 4B / 9B / 27B)
//! and Qwen3.6-27B exposes the SAME 15 MTP-block tensors in safetensors.
//! MoE variants replace the dense FFN triple with a router, shared expert,
//! scalar shared gate, and 3D stacked routed experts. The extractor splits
//! those expert tensors into per-expert 2D weights so the runtime can reuse
//! the same indexed MoE decode kernels as A3B trunk inference.
//!
//! Output container:
//!   * arch_id = 21 (QWEN35_MTP_HEAD)
//!   * 15 tensors with hipfire-canonical names (see naming map below)
//!   * Norms (`*norm`) stay F32; weights default to MQ4 (or Q8 with
//!     `--quant q8` for the conservative first-integration path).
//!   * Embedding + LM head are intentionally NOT packed — the MTP head
//!     reuses the trunk's `embed_tokens` and `lm_head`. Metadata flags
//!     `shared_embed_with_trunk` + `shared_lm_head_with_trunk` make
//!     this explicit so consumers can wire them up at load time.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors parser ────────────────────────────────────────────────────

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
}

// ─── FP conversions (lifted from dflash_convert) ──────────────────────────

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

fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "F16" | "BF16" => 2,
        "F32" => 4,
        other => panic!("unsupported input dtype: {other}"),
    }
}

fn to_f32_range(data: &[u8], dtype: &str, elem_start: usize, elem_count: usize) -> Vec<f32> {
    let elem_size = dtype_size(dtype);
    let byte_start = elem_start * elem_size;
    let byte_end = byte_start + elem_count * elem_size;
    assert!(
        byte_end <= data.len(),
        "to_f32_range out of bounds: bytes {byte_start}..{byte_end} > {}",
        data.len()
    );
    to_f32(&data[byte_start..byte_end], dtype)
}

fn f32_slice_to_f32_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 4);
    for &v in f32_data {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

// ─── FWHT + MQ4 quantization (must match engine seeds 42/1042) ────────────

/// CPU-side FWHT on a 256-element group. Mirrors the GPU-side
/// `fwht_forward_256` in rdna_compute: signs1 → butterfly → 1/16 → signs2.
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
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
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// MagnumQuant MQ4-G256: FWHT-rotated 4-bit. 136 B / 256 weights
/// (0.531 B/w). Layout matches `quantize_mq4g256` in the trunk
/// quantizer at `crates/hipfire-quantize/src/main.rs`.
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

/// Q8_F16 quantization (group_size=32, 34 B per group, 1.0625 B/w).
/// Mirrors `quantize_q8f16` in main.rs so the resulting MTP weights can
/// be consumed by the existing GEMV kernels in hipfire-arch-qwen35.
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

// ─── HFQ container ────────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;
/// arch_id assignment: 0=llama, 1=qwen3/qwen2, 5=qwen3.5 dense,
/// 6=qwen3.5 MoE, 20=DFlash drafter, **21=Qwen3.5 MTP head** (this).
const ARCH_ID_QWEN35_MTP_HEAD: u32 = 21;

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum QuantType {
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    MQ4G256 = 13,
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

// ─── Naming map ──────────────────────────────────────────────────────────

/// (safetensors_name, hipfire_name, is_norm, is_2d_weight)
/// Order is the canonical pack order (norms first, then 2D weights —
/// matches the trunk extractor's natural dependency order at consume time).
const MTP_NAMING_MAP: &[(&str, &str, bool, bool)] = &[
    // Norms (F32, 1D)
    ("mtp.norm.weight", "shared_head_norm", true, false),
    ("mtp.pre_fc_norm_embedding.weight", "enorm", true, false),
    ("mtp.pre_fc_norm_hidden.weight", "hnorm", true, false),
    (
        "mtp.layers.0.input_layernorm.weight",
        "attn_norm",
        true,
        false,
    ),
    (
        "mtp.layers.0.post_attention_layernorm.weight",
        "attn_post_norm",
        true,
        false,
    ),
    (
        "mtp.layers.0.self_attn.q_norm.weight",
        "attn_q_norm",
        true,
        false,
    ),
    (
        "mtp.layers.0.self_attn.k_norm.weight",
        "attn_k_norm",
        true,
        false,
    ),
    // 2D weights (MQ4 / Q8)
    ("mtp.fc.weight", "eh_proj", false, true),
    ("mtp.layers.0.self_attn.q_proj.weight", "wq", false, true),
    ("mtp.layers.0.self_attn.k_proj.weight", "wk", false, true),
    ("mtp.layers.0.self_attn.v_proj.weight", "wv", false, true),
    ("mtp.layers.0.self_attn.o_proj.weight", "wo", false, true),
    ("mtp.layers.0.mlp.gate_proj.weight", "ffn_gate", false, true),
    ("mtp.layers.0.mlp.up_proj.weight", "ffn_up", false, true),
    ("mtp.layers.0.mlp.down_proj.weight", "ffn_down", false, true),
];

const MTP_COMMON_NAMING_MAP: &[(&str, &str, bool, bool)] = &[
    // Norms (F32, 1D)
    ("mtp.norm.weight", "shared_head_norm", true, false),
    ("mtp.pre_fc_norm_embedding.weight", "enorm", true, false),
    ("mtp.pre_fc_norm_hidden.weight", "hnorm", true, false),
    (
        "mtp.layers.0.input_layernorm.weight",
        "attn_norm",
        true,
        false,
    ),
    (
        "mtp.layers.0.post_attention_layernorm.weight",
        "attn_post_norm",
        true,
        false,
    ),
    (
        "mtp.layers.0.self_attn.q_norm.weight",
        "attn_q_norm",
        true,
        false,
    ),
    (
        "mtp.layers.0.self_attn.k_norm.weight",
        "attn_k_norm",
        true,
        false,
    ),
    // NextN projection + attention weights.
    ("mtp.fc.weight", "eh_proj", false, true),
    ("mtp.layers.0.self_attn.q_proj.weight", "wq", false, true),
    ("mtp.layers.0.self_attn.k_proj.weight", "wk", false, true),
    ("mtp.layers.0.self_attn.v_proj.weight", "wv", false, true),
    ("mtp.layers.0.self_attn.o_proj.weight", "wo", false, true),
];

const MTP_MOE_2D_NAMING_MAP: &[(&str, &str, bool, bool)] = &[
    ("mtp.layers.0.mlp.gate.weight", "moe_router", false, true),
    (
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        "moe_shared_expert_gate",
        false,
        true,
    ),
    (
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "moe_shared_gate",
        false,
        true,
    ),
    (
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "moe_shared_up",
        false,
        true,
    ),
    (
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "moe_shared_down",
        false,
        true,
    ),
];

// ─── Discovery ───────────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

fn find_tensor<'a>(
    st_files: &'a [SafetensorsFile],
    name: &str,
) -> Option<(&'a TensorMeta, &'a [u8])> {
    for st in st_files {
        if let Some((meta, data)) = st.tensor_data(name) {
            return Some((meta, data));
        }
    }
    None
}

// ─── CLI ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum QuantChoice {
    Mq4,
    Q8,
}

impl QuantChoice {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "mq4" | "MQ4" | "mq4g256" => Some(Self::Mq4),
            "q8" | "Q8" | "q8f16" => Some(Self::Q8),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Mq4 => "MQ4G256",
            Self::Q8 => "Q8_F16",
        }
    }
}

struct Args {
    hf_dir: PathBuf,
    output: PathBuf,
    quant: QuantChoice,
    verbose: bool,
    /// Optional FastMTP-style vocab-compression sidecar JSON. When set, we
    /// also pack a compressed `lm_head_draft.weight` (top-K rows of trunk
    /// lm_head, MQ4G256) and a `lm_head_draft.vocab_map` (u32 IDs packed
    /// as f32 bit-cast). Tensor count goes 15 -> 17 when present.
    vocab_sidecar: Option<PathBuf>,
}

fn quantize_mtp_tensor(
    hf_name: &str,
    shape: &[usize],
    f32_data: &[f32],
    is_norm: bool,
    is_2d: bool,
    quant: QuantChoice,
    force_q8: bool,
    signs1: &[f32],
    signs2: &[f32],
) -> (QuantType, u32, Vec<u8>, &'static str) {
    let n_elements: usize = shape.iter().product();
    assert_eq!(
        f32_data.len(),
        n_elements,
        "tensor {hf_name}: element count mismatch (got {}, expected {n_elements})",
        f32_data.len()
    );
    if is_norm || !is_2d {
        return (QuantType::F32, 0, f32_slice_to_f32_bytes(f32_data), "F32");
    }

    let k_dim = *shape.last().expect("2D weight has shape");
    if force_q8 {
        let q = quantize_q8f16(f32_data);
        return (QuantType::Q8F16, 32, q, "Q8_F16");
    }

    match quant {
        QuantChoice::Mq4 if k_dim % 256 == 0 && n_elements >= 256 => {
            let q = quantize_mq4g256(f32_data, signs1, signs2);
            (QuantType::MQ4G256, 256, q, "MQ4G256")
        }
        QuantChoice::Mq4 => {
            eprintln!("  note: {hf_name} K={k_dim} not 256-aligned — falling back to Q8_F16");
            let q = quantize_q8f16(f32_data);
            (QuantType::Q8F16, 32, q, "Q8_F16")
        }
        QuantChoice::Q8 => {
            let q = quantize_q8f16(f32_data);
            (QuantType::Q8F16, 32, q, "Q8_F16")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_safetensor_2d_or_norm(
    st_files: &[SafetensorsFile],
    st_name: &str,
    hf_name: &str,
    is_norm: bool,
    is_2d: bool,
    quant: QuantChoice,
    force_q8: bool,
    signs1: &[f32],
    signs2: &[f32],
    verbose: bool,
) -> (HfqTensor, usize, usize) {
    let (meta, raw) = find_tensor(st_files, st_name).unwrap_or_else(|| {
        panic!(
            "safetensors missing required MTP tensor '{st_name}' \
             — is this actually a Qwen3.5/3.6 model with MTP head?"
        )
    });
    let f32_data = to_f32(raw, &meta.dtype);
    let shape_usize = meta.shape.clone();
    let shape: Vec<u32> = shape_usize.iter().map(|d| *d as u32).collect();
    let (quant_type, group_size, bytes, label) = quantize_mtp_tensor(
        hf_name,
        &shape_usize,
        &f32_data,
        is_norm,
        is_2d,
        quant,
        force_q8,
        signs1,
        signs2,
    );
    if verbose {
        eprintln!(
            "  [{label:>7}] {hf_name:>24}  shape={:?}  in={:>10}B  out={:>10}B  \
             (src={st_name})",
            shape,
            raw.len(),
            bytes.len()
        );
    }
    (
        HfqTensor {
            name: hf_name.to_string(),
            quant_type,
            shape,
            group_size,
            data: bytes,
        },
        raw.len(),
        f32_data.len(),
    )
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut hf_dir: Option<String> = None;
    let mut output: Option<String> = None;
    let mut quant = QuantChoice::Mq4;
    let mut verbose = false;
    let mut vocab_sidecar: Option<String> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hf-dir" | "-i" => {
                hf_dir = Some(argv[i + 1].clone());
                i += 2;
            }
            "--output" | "-o" => {
                output = Some(argv[i + 1].clone());
                i += 2;
            }
            "--quant" => {
                let s = &argv[i + 1];
                quant = QuantChoice::parse(s).unwrap_or_else(|| {
                    eprintln!("unknown --quant value '{s}' (valid: mq4, q8)");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--vocab-sidecar" => {
                vocab_sidecar = Some(argv[i + 1].clone());
                i += 2;
            }
            "--verbose" | "-v" => {
                verbose = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: mtp_extract --hf-dir <safetensors_dir> --output <out.mtp> \
                     [--quant mq4|q8] [--vocab-sidecar <sidecar.json>] [--verbose]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    Args {
        hf_dir: PathBuf::from(hf_dir.expect("--hf-dir required")),
        output: PathBuf::from(output.expect("--output required")),
        quant,
        verbose,
        vocab_sidecar: vocab_sidecar.map(PathBuf::from),
    }
}

// ─── Round-trip verification ─────────────────────────────────────────────

/// Read back the freshly-written HFQ container's header + tensor index,
/// confirming the 15 tensors round-trip cleanly. Prints a summary if
/// `verbose` is set; panics on any mismatch.
fn verify_round_trip(path: &Path, expected: &[HfqTensor], verbose: bool) {
    let file = File::open(path).expect("open output for verify");
    let mmap = unsafe { Mmap::map(&file).expect("mmap output for verify") };
    assert_eq!(&mmap[0..4], HFQ_MAGIC, "verify: bad magic");
    let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

    assert_eq!(arch_id, ARCH_ID_QWEN35_MTP_HEAD, "verify: arch_id mismatch");
    assert_eq!(n_tensors, expected.len(), "verify: tensor count mismatch");

    // Skip metadata JSON to reach the index. Same brace-scan as HfqFile::open.
    let meta_bytes = &mmap[metadata_offset..data_offset];
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut json_end = 0;
    for (i, &b) in meta_bytes.iter().enumerate() {
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
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
    }
    let mut pos = metadata_offset + json_end;
    let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
    assert_eq!(idx_n, n_tensors, "verify: index count mismatch");
    pos += 4;

    let mut cumulative = data_offset;
    for (i, exp) in expected.iter().enumerate() {
        let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
        pos += name_len;
        let qt = mmap[pos];
        pos += 1;
        let n_dims = mmap[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        assert_eq!(name, exp.name, "verify: tensor[{i}] name mismatch");
        assert_eq!(qt, exp.quant_type as u8, "verify: tensor[{i}] qt mismatch");
        assert_eq!(shape, exp.shape, "verify: tensor[{i}] shape mismatch");
        assert_eq!(
            group_size, exp.group_size,
            "verify: tensor[{i}] gs mismatch"
        );
        assert_eq!(
            data_size,
            exp.data.len(),
            "verify: tensor[{i}] data_size mismatch"
        );

        // Spot-check first/last byte of data region against the in-memory
        // tensor. This catches misaligned data_offset / forgotten padding.
        if !exp.data.is_empty() {
            assert_eq!(
                mmap[cumulative], exp.data[0],
                "verify: tensor[{i}]={name} first byte mismatch"
            );
            assert_eq!(
                mmap[cumulative + data_size - 1],
                exp.data[data_size - 1],
                "verify: tensor[{i}]={name} last byte mismatch"
            );
        }
        cumulative += data_size;

        if verbose {
            eprintln!(
                "  verify[{i:2}] {name:>18}: qt={qt:<2} shape={:?} gs={group_size} bytes={data_size}",
                shape
            );
        }
    }
    eprintln!("verify: PASS — {n_tensors} tensors round-trip clean (arch_id={arch_id})");
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    eprintln!("mtp_extract");
    eprintln!("  hf-dir : {}", args.hf_dir.display());
    eprintln!("  output : {}", args.output.display());
    eprintln!("  quant  : {} (norms always F32)", args.quant.label());

    // Read config.json — needed for metadata + quant-correctness sanity checks.
    let config_path = args.hf_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_path.display()));
    let config: serde_json::Value =
        serde_json::from_str(&config_str).expect("config.json parse failed");

    // Qwen3.5 nests model dims under text_config; fall back to top-level
    // for older Qwen3.0-style configs.
    let tc = config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| config.clone());
    let get_u64 = |k: &str| -> u64 {
        tc.get(k)
            .or_else(|| config.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("config.json missing {k}"))
    };
    let get_f64 =
        |k: &str| -> Option<f64> { tc.get(k).or_else(|| config.get(k)).and_then(|v| v.as_f64()) };

    let n_embd = get_u64("hidden_size") as usize;
    let n_layer = get_u64("num_hidden_layers") as usize;
    let n_head = get_u64("num_attention_heads") as usize;
    let n_head_kv = get_u64("num_key_value_heads") as usize;
    let n_embd_head = tc
        .get("head_dim")
        .or_else(|| config.get("head_dim"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(n_embd / n_head);
    let intermediate_size = tc
        .get("intermediate_size")
        .or_else(|| config.get("intermediate_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let num_experts = tc
        .get("num_experts")
        .or_else(|| config.get("num_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts_per_tok = tc
        .get("num_experts_per_tok")
        .or_else(|| config.get("num_experts_per_tok"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .or_else(|| config.get("moe_intermediate_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let shared_expert_intermediate_size = tc
        .get("shared_expert_intermediate_size")
        .or_else(|| config.get("shared_expert_intermediate_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let norm_topk_prob = tc
        .get("norm_topk_prob")
        .or_else(|| config.get("norm_topk_prob"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let n_ff = intermediate_size.unwrap_or(moe_intermediate_size);
    let vocab_size = get_u64("vocab_size") as usize;
    let nextn_predict_layers = tc
        .get("mtp_num_hidden_layers")
        .or_else(|| config.get("mtp_num_hidden_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6);
    let rope_theta = tc
        .get("rope_parameters")
        .and_then(|p| p.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .or_else(|| get_f64("rope_theta"))
        .unwrap_or(10_000_000.0);
    let rope_sections = tc
        .get("rope_parameters")
        .and_then(|p| p.get("mrope_section"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let source_model = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            // Best-effort: derive from the hf-dir path
            // (.../models--Qwen--Qwen3.5-0.8B/snapshots/...). Falls back
            // to the directory name.
            let p = args.hf_dir.canonicalize().unwrap_or(args.hf_dir.clone());
            for comp in p.components() {
                let s = comp.as_os_str().to_string_lossy().into_owned();
                if let Some(rest) = s.strip_prefix("models--") {
                    return rest.replacen("--", "/", 1);
                }
            }
            args.hf_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string())
        });

    if nextn_predict_layers != 1 {
        eprintln!(
            "warning: mtp_num_hidden_layers={nextn_predict_layers} (expected 1 for current Qwen MTP)"
        );
    }

    eprintln!("  model  : {source_model}");
    eprintln!(
        "  dims   : n_embd={n_embd} n_layer={n_layer} n_head={n_head} n_head_kv={n_head_kv} \
         head_dim={n_embd_head} n_ff={n_ff} vocab={vocab_size}"
    );
    if num_experts > 0 {
        eprintln!(
            "  moe    : experts={num_experts} top_k={num_experts_per_tok} \
             moe_intermediate={moe_intermediate_size} shared_intermediate={shared_expert_intermediate_size} \
             norm_topk_prob={norm_topk_prob}"
        );
    }
    eprintln!(
        "  rope   : theta={rope_theta} sections={}",
        serde_json::to_string(&rope_sections).unwrap_or("null".into())
    );

    // Open all safetensors shards and find which shard each MTP tensor is in.
    let st_paths = find_safetensors(&args.hf_dir);
    if st_paths.is_empty() {
        panic!("no .safetensors files found in {}", args.hf_dir.display());
    }
    let st_files: Vec<SafetensorsFile> = st_paths
        .iter()
        .inspect(|p| {
            if args.verbose {
                eprintln!("  loading: {}", p.display())
            }
        })
        .map(|p| SafetensorsFile::open(p).expect("safetensors open"))
        .collect();
    eprintln!("  shards : {}", st_files.len());

    // FWHT seeds — MUST match engine `Gpu::ensure_mq_signs()` in
    // rdna_compute (42 / 1042). Reused from dflash_convert.
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    let dense_mtp = find_tensor(&st_files, "mtp.layers.0.mlp.gate_proj.weight").is_some();
    let moe_mtp = find_tensor(&st_files, "mtp.layers.0.mlp.experts.gate_up_proj").is_some();
    let ffn_kind = match (dense_mtp, moe_mtp) {
        (true, _) => "dense",
        (false, true) => "moe",
        (false, false) => panic!(
            "MTP FFN tensors not found: expected dense gate_proj/up_proj/down_proj \
             or MoE experts.gate_up_proj"
        ),
    };

    let estimated_tensors = if ffn_kind == "moe" {
        MTP_COMMON_NAMING_MAP.len() + MTP_MOE_2D_NAMING_MAP.len() + 2 * num_experts
    } else {
        MTP_NAMING_MAP.len()
    };
    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(estimated_tensors);
    let mut total_in_bytes: usize = 0;
    let mut total_out_bytes: usize = 0;

    let base_map = if ffn_kind == "moe" {
        MTP_COMMON_NAMING_MAP
    } else {
        MTP_NAMING_MAP
    };
    for (st_name, hf_name, is_norm, is_2d) in base_map {
        let (tensor, in_bytes, _n_elems) = pack_safetensor_2d_or_norm(
            &st_files,
            st_name,
            hf_name,
            *is_norm,
            *is_2d,
            args.quant,
            false,
            &signs1,
            &signs2,
            args.verbose,
        );
        total_in_bytes += in_bytes;
        total_out_bytes += tensor.data.len();
        hfq_tensors.push(tensor);
    }

    if ffn_kind == "moe" {
        assert!(
            num_experts > 0,
            "MoE MTP detected but config num_experts is 0"
        );
        assert_eq!(
            num_experts_per_tok, 8,
            "current MoE MTP runtime supports top_k=8; config has {num_experts_per_tok}"
        );
        assert!(
            moe_intermediate_size > 0,
            "MoE MTP requires moe_intermediate_size"
        );
        assert!(
            shared_expert_intermediate_size > 0,
            "MoE MTP requires shared_expert_intermediate_size"
        );

        for (st_name, hf_name, is_norm, is_2d) in MTP_MOE_2D_NAMING_MAP {
            // Match the production A3B loader convention: routers and the
            // scalar shared-expert gate stay Q8, expert/shared FFNs use the
            // requested quant format.
            let force_q8 = *hf_name == "moe_router" || *hf_name == "moe_shared_expert_gate";
            let (tensor, in_bytes, _n_elems) = pack_safetensor_2d_or_norm(
                &st_files,
                st_name,
                hf_name,
                *is_norm,
                *is_2d,
                args.quant,
                force_q8,
                &signs1,
                &signs2,
                args.verbose,
            );
            total_in_bytes += in_bytes;
            total_out_bytes += tensor.data.len();
            hfq_tensors.push(tensor);
        }

        let (gate_up_meta, gate_up_raw) =
            find_tensor(&st_files, "mtp.layers.0.mlp.experts.gate_up_proj")
                .expect("MoE MTP missing experts.gate_up_proj");
        let (down_meta, down_raw) = find_tensor(&st_files, "mtp.layers.0.mlp.experts.down_proj")
            .expect("MoE MTP missing experts.down_proj");
        assert_eq!(
            gate_up_meta.shape,
            vec![num_experts, 2 * moe_intermediate_size, n_embd],
            "experts.gate_up_proj shape mismatch"
        );
        assert_eq!(
            down_meta.shape,
            vec![num_experts, n_embd, moe_intermediate_size],
            "experts.down_proj shape mismatch"
        );

        let gate_up_elems = 2 * moe_intermediate_size * n_embd;
        let down_elems = n_embd * moe_intermediate_size;
        total_in_bytes += gate_up_raw.len() + down_raw.len();
        for expert_idx in 0..num_experts {
            let f32_data = to_f32_range(
                gate_up_raw,
                &gate_up_meta.dtype,
                expert_idx * gate_up_elems,
                gate_up_elems,
            );
            let hf_name = format!("moe_experts.{expert_idx}.gate_up");
            let (quant_type, group_size, bytes, label) = quantize_mtp_tensor(
                &hf_name,
                &[2 * moe_intermediate_size, n_embd],
                &f32_data,
                false,
                true,
                args.quant,
                false,
                &signs1,
                &signs2,
            );
            if args.verbose && (expert_idx < 4 || expert_idx + 4 >= num_experts) {
                eprintln!(
                    "  [{label:>7}] {hf_name:>24}  shape=[{}, {}]  out={:>10}B",
                    2 * moe_intermediate_size,
                    n_embd,
                    bytes.len()
                );
            }
            total_out_bytes += bytes.len();
            hfq_tensors.push(HfqTensor {
                name: hf_name,
                quant_type,
                shape: vec![(2 * moe_intermediate_size) as u32, n_embd as u32],
                group_size,
                data: bytes,
            });
        }
        for expert_idx in 0..num_experts {
            let f32_data = to_f32_range(
                down_raw,
                &down_meta.dtype,
                expert_idx * down_elems,
                down_elems,
            );
            let hf_name = format!("moe_experts.{expert_idx}.down");
            let (quant_type, group_size, bytes, label) = quantize_mtp_tensor(
                &hf_name,
                &[n_embd, moe_intermediate_size],
                &f32_data,
                false,
                true,
                args.quant,
                false,
                &signs1,
                &signs2,
            );
            if args.verbose && (expert_idx < 4 || expert_idx + 4 >= num_experts) {
                eprintln!(
                    "  [{label:>7}] {hf_name:>24}  shape=[{}, {}]  out={:>10}B",
                    n_embd,
                    moe_intermediate_size,
                    bytes.len()
                );
            }
            total_out_bytes += bytes.len();
            hfq_tensors.push(HfqTensor {
                name: hf_name,
                quant_type,
                shape: vec![n_embd as u32, moe_intermediate_size as u32],
                group_size,
                data: bytes,
            });
        }
        if args.verbose && num_experts > 8 {
            eprintln!(
                "  ... packed {} routed experts for gate_up and down",
                num_experts
            );
        }
    }

    // ─── Optional FastMTP-style vocab-compression sidecar ─────────────────
    //
    // When `--vocab-sidecar <path>` is set, append two extra tensors:
    //   * lm_head_draft.weight   shape [K, n_embd] MQ4G256 — top-K rows of
    //                             trunk lm_head, used by MTP forward as a
    //                             compressed draft head (~7.7x BW reduction
    //                             at K=32K vs K=248K).
    //   * lm_head_draft.vocab_map  shape [K] F32 (u32 bit-cast) — maps
    //                             draft idx -> full vocab idx for argmax
    //                             remap on the engine side.
    //
    // Verifier path is unchanged: trunk uses its full lm_head, so any
    // out-of-K-vocab draft proposal automatically rejects via argmax
    // mismatch — lossless greedy preserved. Coverage gaps just reduce τ.
    let mut compressed_vocab_size: Option<usize> = None;
    if let Some(sidecar_path) = &args.vocab_sidecar {
        eprintln!("  sidecar: {}", sidecar_path.display());
        let sidecar_str = std::fs::read_to_string(sidecar_path)
            .unwrap_or_else(|e| panic!("cannot read sidecar {}: {e}", sidecar_path.display()));
        let sidecar: serde_json::Value =
            serde_json::from_str(&sidecar_str).expect("sidecar JSON parse failed");
        let draft_to_full: Vec<u32> = sidecar
            .get("draft_to_full")
            .and_then(|v| v.as_array())
            .expect("sidecar missing draft_to_full array")
            .iter()
            .map(|x| x.as_u64().expect("draft_to_full element not u64") as u32)
            .collect();
        let k_dim = draft_to_full.len();
        let cvs_meta = sidecar
            .get("compressed_vocab_size")
            .and_then(|v| v.as_u64())
            .expect("sidecar missing compressed_vocab_size") as usize;
        assert_eq!(
            k_dim, cvs_meta,
            "sidecar inconsistent: draft_to_full.len()={k_dim} vs compressed_vocab_size={cvs_meta}"
        );
        assert!(
            k_dim > 0 && k_dim <= vocab_size,
            "sidecar K={k_dim} outside (0, vocab_size={vocab_size}]"
        );
        // Find lm_head in safetensors. Qwen3.5/3.6 are VLMs with a
        // `model.language_model.*` namespace; the dedicated lm_head (when
        // present) is at top-level `lm_head.weight`. When
        // tie_word_embeddings is true (e.g., 0.8B), lm_head IS embed_tokens
        // and only the embed_tokens key exists. Prefer the dedicated head
        // when available.
        let lm_head_candidates = [
            "lm_head.weight",
            "model.lm_head.weight",
            "model.language_model.lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ];
        let mut lm_head_found: Option<(&TensorMeta, &[u8], &str)> = None;
        for cand in &lm_head_candidates {
            for st in &st_files {
                if let Some((meta, data)) = st.tensor_data(cand) {
                    lm_head_found = Some((meta, data, cand));
                    break;
                }
            }
            if lm_head_found.is_some() {
                break;
            }
        }
        let (lm_meta, lm_raw, lm_name) = lm_head_found.unwrap_or_else(|| {
            panic!(
                "sidecar mode: cannot find trunk lm_head in safetensors (tried {:?})",
                lm_head_candidates
            )
        });
        eprintln!(
            "  lm_head src: {lm_name} dtype={} shape={:?}",
            lm_meta.dtype, lm_meta.shape
        );
        assert_eq!(
            lm_meta.shape.len(),
            2,
            "lm_head must be 2D, got shape {:?}",
            lm_meta.shape
        );
        // Safetensors layout: row-major [V, H] where row v is the projection
        // vector for token id v. Slice rows for the top-K token IDs.
        let v_dim = lm_meta.shape[0];
        let h_dim = lm_meta.shape[1];
        assert_eq!(
            h_dim, n_embd,
            "lm_head hidden dim {h_dim} != model n_embd {n_embd}"
        );
        let lm_full_f32 = to_f32(lm_raw, &lm_meta.dtype);
        assert_eq!(lm_full_f32.len(), v_dim * h_dim);

        let mut lm_compressed_f32: Vec<f32> = Vec::with_capacity(k_dim * h_dim);
        let mut out_of_range = 0usize;
        for &tok_id in &draft_to_full {
            let row_start = (tok_id as usize) * h_dim;
            if (tok_id as usize) >= v_dim {
                // Sidecar requested an ID beyond actual lm_head rows; fill
                // with zeros (will produce -Inf-like logits, harmless: the
                // compressed argmax will never select it).
                out_of_range += 1;
                lm_compressed_f32.extend(std::iter::repeat(0.0f32).take(h_dim));
            } else {
                lm_compressed_f32.extend_from_slice(&lm_full_f32[row_start..row_start + h_dim]);
            }
        }
        if out_of_range > 0 {
            eprintln!(
                "  sidecar: {out_of_range}/{k_dim} requested IDs beyond lm_head rows ({v_dim}); \
                 filled with zeros"
            );
        }

        // Quantize compressed lm_head as MQ4G256 (h_dim=5120 is 256-aligned
        // for all Qwen3.5/3.6 dense models). If --quant q8, fall back.
        let (lm_quant_type, lm_group_size, lm_bytes, lm_label) = match args.quant {
            QuantChoice::Mq4 if h_dim % 256 == 0 => {
                let q = quantize_mq4g256(&lm_compressed_f32, &signs1, &signs2);
                (QuantType::MQ4G256, 256u32, q, "MQ4G256")
            }
            _ => {
                let q = quantize_q8f16(&lm_compressed_f32);
                (QuantType::Q8F16, 32u32, q, "Q8_F16")
            }
        };
        let lm_in_bytes = lm_full_f32.len() * 4; // ref baseline
        total_in_bytes += lm_compressed_f32.len() * 4;
        total_out_bytes += lm_bytes.len();
        eprintln!(
            "  [{lm_label:>7}] {:>18}  shape=[{k_dim}, {h_dim}]  full_f32={:>10}B  out={:>10}B",
            "lm_head_draft",
            lm_in_bytes,
            lm_bytes.len()
        );
        hfq_tensors.push(HfqTensor {
            name: "lm_head_draft.weight".to_string(),
            quant_type: lm_quant_type,
            shape: vec![k_dim as u32, h_dim as u32],
            group_size: lm_group_size,
            data: lm_bytes,
        });

        // Pack vocab map as raw u32 LE bytes. Marked as QuantType::F32 in
        // the index so the file format accounts for 4 bytes/element; the
        // reader (in mtp_head.rs) reads each 4-byte slot as u32 directly
        // (no float arithmetic) to recover the token ID. Avoids extending
        // the QuantType enum and keeps the existing 4 GB/page-aligned
        // file layout intact.
        let vocab_map_bytes: Vec<u8> = draft_to_full
            .iter()
            .flat_map(|&id| id.to_le_bytes())
            .collect();
        assert_eq!(vocab_map_bytes.len(), k_dim * 4);
        hfq_tensors.push(HfqTensor {
            name: "lm_head_draft.vocab_map".to_string(),
            quant_type: QuantType::F32,
            shape: vec![k_dim as u32],
            group_size: 0,
            data: vocab_map_bytes,
        });

        compressed_vocab_size = Some(k_dim);
    }

    let expected_base_tensor_count = if ffn_kind == "moe" {
        MTP_COMMON_NAMING_MAP.len() + MTP_MOE_2D_NAMING_MAP.len() + 2 * num_experts
    } else {
        MTP_NAMING_MAP.len()
    };
    let expected_tensor_count = expected_base_tensor_count
        + if compressed_vocab_size.is_some() {
            2
        } else {
            0
        };
    assert_eq!(
        hfq_tensors.len(),
        expected_tensor_count,
        "tensor count mismatch: got {} expected {expected_tensor_count}",
        hfq_tensors.len()
    );

    // Detect tied embeddings — Qwen3.5 ties by default. Caller of the MTP
    // head must wire the trunk's `embed_tokens` AND (if tied) `lm_head`
    // (which IS embed_tokens) into the head's enorm/lm_head slots.
    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| tc.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    let metadata = serde_json::json!({
        "arch": "qwen35_mtp_head",
        "arch_id": ARCH_ID_QWEN35_MTP_HEAD,
        "source_model": source_model,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "nextn_predict_layers": nextn_predict_layers,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_embd_head": n_embd_head,
        "n_ff": n_ff,
        "ffn_kind": ffn_kind,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
        "shared_expert_intermediate_size": shared_expert_intermediate_size,
        "norm_topk_prob": norm_topk_prob,
        "vocab_size": vocab_size,
        "rope_theta": rope_theta,
        "rope_sections": rope_sections,
        "rms_norm_eps": rms_norm_eps,
        "shared_embed_with_trunk": true,
        "shared_lm_head_with_trunk": true,
        "shared_output_norm_with_trunk": false,
        "tie_word_embeddings": tie_word_embeddings,
        "weight_quant": args.quant.label(),
        "has_compressed_lm_head_draft": compressed_vocab_size.is_some(),
        "compressed_vocab_size": compressed_vocab_size.unwrap_or(0),
        "config_text_config": tc,
    });
    let metadata_json = serde_json::to_string(&metadata).expect("metadata JSON serialize");

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir -p output parent");
        }
    }
    write_hfq(
        &args.output,
        ARCH_ID_QWEN35_MTP_HEAD,
        &metadata_json,
        &hfq_tensors,
    )
    .expect("write_hfq failed");

    let file_size = std::fs::metadata(&args.output).expect("stat output").len();
    eprintln!(
        "wrote {}: {:.2} MiB  ({} tensors, in={:.2} MiB, out={:.2} MiB)",
        args.output.display(),
        file_size as f64 / (1024.0 * 1024.0),
        hfq_tensors.len(),
        total_in_bytes as f64 / (1024.0 * 1024.0),
        total_out_bytes as f64 / (1024.0 * 1024.0),
    );
    if let Some(k) = compressed_vocab_size {
        eprintln!("  compressed_vocab_size = {k} (FastMTP-style draft head)");
    }

    // Round-trip verify the HFQ container we just wrote. Reads back the
    // header + tensor index from disk and asserts every field matches the
    // in-memory `hfq_tensors`. Acts as the inline acceptance test.
    verify_round_trip(&args.output, &hfq_tensors, args.verbose);
}
