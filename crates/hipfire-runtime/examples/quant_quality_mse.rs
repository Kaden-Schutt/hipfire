// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-tensor quantization-quality benchmark.
//!
//! Reads a `.hfq` file and the source HuggingFace safetensors directory
//! it was quantized from, dequantizes each `.hfq` tensor to f32, and
//! computes mean squared error vs the safetensors-side f32 reference.
//!
//! Output: per-tensor table sorted by MSE descending, plus aggregate
//! stats (mean MSE, p99, max). Use this as a 30-second feedback loop
//! when changing quantization formats / scale-search algorithms.
//!
//! Scope (v1): handles the 2D weight tensors that have the same name on
//! both sides — norms (qt=1 F16), attention proj, embed, lm_head, shared
//! expert, router. The 3D-split MoE expert tensors
//! (`mlp.experts.{X}.gate_up_proj.weight`) currently produce one
//! per-expert tensor in the .hfq but a single 3D tensor in safetensors;
//! handling this needs a per-expert slice on the safetensors side,
//! deferred to v2.
//!
//! Supported quant types: F16 (qt=1), F32 (qt=2), Q8_0 (qt=3),
//! MQ4G256 (qt=13), MQ6G256 (qt=15). Other qts skipped with a warning.
//!
//! Usage:
//!   quant_quality_mse <safetensors_dir> <model.hfq> [name_substring]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct StTensor {
    file_idx: usize,
    dtype: String,
    shape: Vec<usize>,
    /// Byte offset within the file's tensor-data region (after header).
    offset_in_data: usize,
    nbytes: usize,
}

/// Minimal safetensors reader. Supports F16, BF16, F32. Indexes by name
/// across all `.safetensors` shards in a directory.
struct SafetensorsIndex {
    files: Vec<Mmap>,
    /// Byte offset where the tensor data region starts in each file
    /// (8 + header_size).
    data_start: Vec<usize>,
    by_name: HashMap<String, StTensor>,
}

impl SafetensorsIndex {
    fn open_dir_or_file(path: &Path) -> std::io::Result<Self> {
        let shards: Vec<PathBuf> = if path.is_file()
            && path.extension().map_or(false, |e| e == "safetensors")
        {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            let mut v = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let p = entry?.path();
                if p.extension().map_or(false, |e| e == "safetensors") {
                    v.push(p);
                }
            }
            v.sort();
            v
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("not a safetensors file or dir: {}", path.display()),
            ));
        };

        let mut files: Vec<Mmap> = Vec::with_capacity(shards.len());
        let mut data_start: Vec<usize> = Vec::with_capacity(shards.len());
        let mut by_name: HashMap<String, StTensor> = HashMap::new();

        for (file_idx, shard) in shards.iter().enumerate() {
            let f = File::open(shard)?;
            let mmap = unsafe { Mmap::map(&f)? };
            let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header_json = std::str::from_utf8(&mmap[8..8 + header_size])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let v: Value = serde_json::from_str(header_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let obj = v.as_object().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "header not object")
            })?;
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = info["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|u| u as usize))
                            .collect()
                    })
                    .unwrap_or_default();
                let offsets: Vec<usize> = info["data_offsets"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|u| u as usize))
                            .collect()
                    })
                    .unwrap_or_default();
                if offsets.len() != 2 {
                    continue;
                }
                let offset_in_data = offsets[0];
                let nbytes = offsets[1] - offsets[0];
                by_name.insert(
                    name.clone(),
                    StTensor {
                        file_idx,
                        dtype,
                        shape,
                        offset_in_data,
                        nbytes,
                    },
                );
            }
            files.push(mmap);
            data_start.push(8 + header_size);
        }

        Ok(Self {
            files,
            data_start,
            by_name,
        })
    }

    fn read_f32(&self, t: &StTensor) -> Option<Vec<f32>> {
        let m = &self.files[t.file_idx];
        let start = self.data_start[t.file_idx] + t.offset_in_data;
        let end = start + t.nbytes;
        let data = &m[start..end];
        match t.dtype.as_str() {
            "F32" => Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            "F16" => Some(
                data.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            ),
            "BF16" => Some(
                data.chunks_exact(2)
                    .map(|c| {
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

// ─── Dequantization (mirrors compare_hfq.rs) ─────────────────────────────

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
        })
        .collect()
}

fn cpu_fwht_256(x: &mut [f32; 256], signs1: &[f32], signs2: &[f32]) {
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
    let scale = 0.0625; // 1/sqrt(256)
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

fn dequant_f16(data: &[u8], n: usize) -> Vec<f32> {
    data.chunks_exact(2)
        .take(n)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn dequant_f32(data: &[u8], n: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let group = 32usize;
    let block = 2 + group;
    let n_blocks = n.div_ceil(group);
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let off = b * block;
        if off + block > data.len() { break; }
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for i in 0..group {
            if out.len() >= n { break; }
            let q = data[off + 2 + i] as i8;
            out.push(q as f32 * scale);
        }
    }
    out
}

fn dequant_mq4g256(data: &[u8], n: usize, signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let group = 256usize;
    let block = 136usize;
    let n_blocks = n.div_ceil(group);
    let mut out = Vec::with_capacity(n_blocks * group);
    for b in 0..n_blocks {
        let off = b * block;
        if off + block > data.len() { break; }
        let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let min_val = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
        let mut group_buf = [0.0f32; 256];
        for i in 0..128 {
            let byte = data[off + 8 + i];
            let lo = (byte & 0xF) as f32;
            let hi = (byte >> 4) as f32;
            group_buf[2 * i]     = min_val + scale * lo;
            group_buf[2 * i + 1] = min_val + scale * hi;
        }
        // Inverse FWHT: forward operation with signs1 and signs2 swapped.
        cpu_fwht_256(&mut group_buf, signs2, signs1);
        out.extend_from_slice(&group_buf);
    }
    out.truncate(n);
    out
}

fn dequant_mq6g256(data: &[u8], n: usize, signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let group = 256usize;
    let block = 200usize;
    let n_blocks = n.div_ceil(group);
    let mut out = Vec::with_capacity(n_blocks * group);
    for b in 0..n_blocks {
        let off = b * block;
        if off + block > data.len() { break; }
        let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let min_val = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
        let mut group_buf = [0.0f32; 256];
        for i in (0..256usize).step_by(4) {
            let byte_off = 8 + (i / 4) * 3;
            let b0 = data[off + byte_off];
            let b1 = data[off + byte_off + 1];
            let b2 = data[off + byte_off + 2];
            let q0 = (b0 & 0x3F) as u32;
            let q1 = ((b0 >> 6) as u32) | (((b1 & 0x0F) as u32) << 2);
            let q2 = ((b1 >> 4) as u32) | (((b2 & 0x03) as u32) << 4);
            let q3 = (b2 >> 2) as u32;
            group_buf[i]     = min_val + scale * q0 as f32;
            group_buf[i + 1] = min_val + scale * q1 as f32;
            group_buf[i + 2] = min_val + scale * q2 as f32;
            group_buf[i + 3] = min_val + scale * q3 as f32;
        }
        cpu_fwht_256(&mut group_buf, signs2, signs1);
        out.extend_from_slice(&group_buf);
    }
    out.truncate(n);
    out
}

/// E2M1 LUT — OCP-spec FP4 magnitudes (matches kernels/src/gemv_hfp4g32.hip
/// and crates/hipfire-quantize/src/main.rs E2M1_LUT). Eight signed
/// magnitudes; sign bit at position 3.
const E2M1_LUT: [f32; 16] = [
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// HFP4G32 dequant (qt=21). Per-row layout: 16-B header + (k/32) blocks
/// × 17 B each. K must be a multiple of 32 (the v1 kernel further
/// requires k%256==0 but the byte format itself is k%32-aligned).
///
/// Header: u16 row_scale_a (f16) + u16 row_scale_b (f16) + u16 block_count
///         + u8 format_flags + u8 reserved + 8 B reserved.
/// Per block: u8 block_e (UE8M0) + 16 B nibbles (low=even, high=odd index).
/// Element value: row_scale_a · 2^(block_e − 127) · E2M1_LUT[nibble].
///
/// Note: this honors format_flags bit 0 (rotation present) only at the
/// MFP4G32 caller — dequant_mfp4g32 calls this then applies inverse FWHT.
fn dequant_hfp4g32_row(data: &[u8], k: usize, out: &mut Vec<f32>) {
    let row_scale_a = f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    // row_scale_b and format_flags are ignored here — the caller decides
    // whether to apply inverse rotation based on quant_type (qt=24 = MFP4).
    let n_blocks = k / 32;
    let body_off = 16usize;
    for b in 0..n_blocks {
        let off = body_off + b * 17;
        let block_e = data[off] as i32;
        // 2^(block_e - 127); free `v_ldexp_f32` on RDNA at runtime.
        let block_scale = ((block_e - 127) as f32).exp2();
        let sc = row_scale_a * block_scale;
        for i in 0..16 {
            let byte = data[off + 1 + i];
            let lo_nibble = (byte & 0x0F) as usize;
            let hi_nibble = (byte >> 4) as usize;
            out.push(sc * E2M1_LUT[lo_nibble]);
            out.push(sc * E2M1_LUT[hi_nibble]);
        }
    }
}

fn dequant_hfp4g32(data: &[u8], shape: &[u32], n: usize) -> Vec<f32> {
    // 2D weight: shape = [m, k]. 1D unsupported (the format requires a
    // per-row header).
    let (m, k) = match shape.len() {
        2 => (shape[0] as usize, shape[1] as usize),
        _ => return Vec::new(),
    };
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * k);
    for r in 0..m {
        let row_off = r * row_bytes;
        if row_off + row_bytes > data.len() { break; }
        dequant_hfp4g32_row(&data[row_off..row_off + row_bytes], k, &mut out);
    }
    out.truncate(n);
    out
}

/// MFP4G32 dequant (qt=24). HFP4G32 + offline FWHT-256 rotation.
/// `k` must additionally be a multiple of 256 because the rotation
/// signs1/signs2 are fixed-256. Inverse FWHT is forward FWHT with
/// signs1/signs2 swapped (orthogonal).
fn dequant_mfp4g32(data: &[u8], shape: &[u32], n: usize, signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let (m, k) = match shape.len() {
        2 => (shape[0] as usize, shape[1] as usize),
        _ => return Vec::new(),
    };
    if k % 256 != 0 {
        return Vec::new();
    }
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * k);
    let mut row_buf: Vec<f32> = Vec::with_capacity(k);
    for r in 0..m {
        let row_off = r * row_bytes;
        if row_off + row_bytes > data.len() { break; }
        row_buf.clear();
        dequant_hfp4g32_row(&data[row_off..row_off + row_bytes], k, &mut row_buf);
        // Inverse FWHT per 256-element segment. Matches the quantizer's
        // forward pass (gen_fwht_signs(42), gen_fwht_signs(1042)) and
        // MQ4/MQ6 dequant: pass signs2 first because forward applied
        // signs2 last.
        let segs = k / 256;
        for seg in 0..segs {
            let mut buf = [0.0f32; 256];
            buf.copy_from_slice(&row_buf[seg * 256..(seg + 1) * 256]);
            cpu_fwht_256(&mut buf, signs2, signs1);
            out.extend_from_slice(&buf);
        }
    }
    out.truncate(n);
    out
}

fn try_dequant(qt: u8, data: &[u8], shape: &[u32], n: usize, s1: &[f32], s2: &[f32]) -> Option<Vec<f32>> {
    match qt {
        1 => Some(dequant_f16(data, n)),
        2 => Some(dequant_f32(data, n)),
        3 => Some(dequant_q8_0(data, n)),
        13 => Some(dequant_mq4g256(data, n, s1, s2)),
        15 => Some(dequant_mq6g256(data, n, s1, s2)),
        21 => Some(dequant_hfp4g32(data, shape, n)),
        24 => Some(dequant_mfp4g32(data, shape, n, s1, s2)),
        _ => None,
    }
}

fn qt_label(qt: u8) -> &'static str {
    match qt {
        1 => "F16",
        2 => "F32",
        3 => "Q8_0",
        13 => "MQ4G256",
        15 => "MQ6G256",
        21 => "HFP4G32",
        24 => "MFP4G32",
        _ => "?",
    }
}

// ─── Name translation ───────────────────────────────────────────────────

/// Translate a hipfire .hfq tensor name to its safetensors counterpart.
/// hipfire's .hfq for Qwen3.5+ MoE prefixes with "model.language_model.";
/// safetensors uses "model.language_model.". Most names match directly.
/// Returns None for tensors that have no safetensors equivalent (e.g.,
/// the per-expert split tensors `mlp.experts.{X}.gate_up_proj.weight`
/// which are 3D in safetensors).
fn translate_name(hfq_name: &str) -> Option<&str> {
    // Per-expert split tensors: deferred to v2 (need 3D slice on safetensors side).
    if hfq_name.contains("mlp.experts.") && !hfq_name.contains("experts.gate") {
        // matches `mlp.experts.{X}.gate_up_proj.weight` and `.down_proj.weight`
        // but NOT `mlp.experts.gate.weight` or similar non-numeric paths
        // Heuristic: look for `experts.` followed by a digit
        let after = &hfq_name[hfq_name.find("experts.").unwrap() + 8..];
        if after.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            return None;
        }
    }
    Some(hfq_name)
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() {
    let mut args = std::env::args().skip(1);
    let st_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: quant_quality_mse <safetensors_dir_or_file> <model.hfq> [name_substring]");
        std::process::exit(2);
    });
    let hfq_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: quant_quality_mse <safetensors_dir_or_file> <model.hfq> [name_substring]");
        std::process::exit(2);
    });
    let filter = args.next();

    eprintln!("Loading safetensors index from {}...", st_path);
    let st = SafetensorsIndex::open_dir_or_file(Path::new(&st_path))
        .expect("open safetensors");
    eprintln!("  {} shards, {} tensors total", st.files.len(), st.by_name.len());

    eprintln!("Loading hfq from {}...", hfq_path);
    let hfq = HfqFile::open(Path::new(&hfq_path)).expect("open hfq");
    eprintln!("  arch_id={} {} tensors", hfq.arch_id, hfq.tensors().len());

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    // Per-tensor results: (name, qt, n_elements, mse, max_abs_err)
    let mut results: Vec<(String, u8, usize, f64, f64)> = Vec::new();
    let mut skipped_qt: HashMap<u8, usize> = HashMap::new();
    let mut skipped_no_st = 0usize;
    let mut skipped_filter = 0usize;
    let mut skipped_3d_expert = 0usize;

    for hfq_t in hfq.tensors() {
        if let Some(f) = filter.as_deref() {
            if !hfq_t.name.contains(f) {
                skipped_filter += 1;
                continue;
            }
        }
        let st_name = match translate_name(&hfq_t.name) {
            Some(n) => n,
            None => {
                skipped_3d_expert += 1;
                continue;
            }
        };
        let st_t = match st.by_name.get(st_name) {
            Some(t) => t,
            None => {
                skipped_no_st += 1;
                continue;
            }
        };

        let n_hfq: usize = hfq_t.shape.iter().map(|&s| s as usize).product();
        let n_st: usize = st_t.shape.iter().product();
        if n_hfq != n_st {
            // Shape mismatch (likely 3D safetensors vs 2D .hfq we missed)
            skipped_no_st += 1;
            continue;
        }

        let (_info, hfq_data) = hfq.tensor_data_vec(&hfq_t.name).expect("read hfq");
        let hfq_f32 = match try_dequant(hfq_t.quant_type, &hfq_data, &hfq_t.shape, n_hfq, &signs1, &signs2) {
            Some(v) => v,
            None => {
                *skipped_qt.entry(hfq_t.quant_type).or_insert(0) += 1;
                continue;
            }
        };
        let st_f32 = match st.read_f32(st_t) {
            Some(v) => v,
            None => {
                skipped_no_st += 1;
                continue;
            }
        };

        let n = n_hfq.min(hfq_f32.len()).min(st_f32.len());
        let mut sse = 0.0f64;
        let mut max_abs = 0.0f64;
        for i in 0..n {
            let d = hfq_f32[i] as f64 - st_f32[i] as f64;
            sse += d * d;
            let abs = d.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        let mse = sse / n as f64;
        results.push((hfq_t.name.clone(), hfq_t.quant_type, n, mse, max_abs));
    }

    if results.is_empty() {
        eprintln!("No tensors matched. Skipped:");
        eprintln!("  filter:           {}", skipped_filter);
        eprintln!("  no safetensors:   {}", skipped_no_st);
        eprintln!("  3D expert:        {}", skipped_3d_expert);
        for (qt, count) in &skipped_qt {
            eprintln!("  qt={qt}:          {count} (no dequantizer)");
        }
        std::process::exit(1);
    }

    // Sort by MSE descending
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("{:<70} {:>9} {:>13} {:>13} {:>13}", "tensor", "qt", "n", "MSE", "max_abs_err");
    println!("{}", "-".repeat(120));

    // Print top 50 by MSE
    let n_print = results.len().min(50);
    for (name, qt, n, mse, mxe) in &results[..n_print] {
        println!(
            "{:<70} {:>9} {:>13} {:>13.4e} {:>13.4e}",
            if name.len() > 70 { &name[..70] } else { name.as_str() },
            qt_label(*qt),
            n,
            mse,
            mxe,
        );
    }

    // Aggregate stats per qt
    println!();
    println!("=== Aggregate stats by quant type ===");
    println!("{:<10} {:>8} {:>15} {:>15} {:>15} {:>15}",
        "qt", "tensors", "total_params", "mean MSE", "p99 MSE", "max MSE");
    println!("{}", "-".repeat(85));
    let mut by_qt: HashMap<u8, Vec<f64>> = HashMap::new();
    let mut params_by_qt: HashMap<u8, usize> = HashMap::new();
    for (_, qt, n, mse, _) in &results {
        by_qt.entry(*qt).or_default().push(*mse);
        *params_by_qt.entry(*qt).or_insert(0) += n;
    }
    let mut qts: Vec<u8> = by_qt.keys().copied().collect();
    qts.sort();
    for qt in qts {
        let mut v = by_qt[&qt].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let p99 = v[((v.len() as f64 * 0.99) as usize).min(v.len() - 1)];
        let max = v[v.len() - 1];
        let params = params_by_qt[&qt];
        println!(
            "{:<10} {:>8} {:>15} {:>15.4e} {:>15.4e} {:>15.4e}",
            qt_label(qt),
            v.len(),
            params,
            mean,
            p99,
            max,
        );
    }

    println!();
    println!("=== Skipped ===");
    println!("  filter:           {skipped_filter}");
    println!("  no safetensors:   {skipped_no_st}");
    println!("  3D expert:        {skipped_3d_expert}");
    for (qt, count) in &skipped_qt {
        println!("  qt={qt}:          {count} (no dequantizer)");
    }
}
