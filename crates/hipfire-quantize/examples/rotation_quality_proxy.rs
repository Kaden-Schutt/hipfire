// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Activation-weighted heldout oracle for a frozen global D2 candidate.
//!
//! Compares candidate D2 signs against canonical seed-1042 D2 on three heldout
//! Qwen3.8-27B tensors. Four-bit uses production MQ4V2; two/three-bit generalize
//! its two-G128 fp16 affine header allocation as explicit research layouts.
//!
//! CLI:
//!   rotation_quality_proxy CANDIDATE.signs [BLOCKS_PER_TENSOR]
//!
//! CANDIDATE.signs: exactly 256 little-endian f32 values, each ±1 (D2 only).
//! D1 is always canonical seed 42. Canonical D2 is seed 1042.
//!
//! Metrics per tensor×bit (canonical, candidate, ratio=cand/canon):
//!   mse_rot     — mean squared error in rotated domain
//!   tail_mse    — MSE on fixed top-ceil(1%) |pre-D2 U| mask (D2-invariant)
//!   block_nmse  — mean_b[SSE_b / (energy_b + eps)], energy on pre-D2 U
//!   m1          — Σ imatrix[col] · e_orig² after inverse-rotate with matching D2
//!
//! Macros: equal-weight means of tensor×bit ratios; worst cell = max ratio.
//!
//! Run (parent validates remotely):
//!   cargo run -q -p hipfire-quantize --example rotation_quality_proxy --release -- cand.signs

use std::collections::HashMap;
use std::path::Path;

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::SafetensorsFile;

const G: usize = 256;
const HALF: usize = 128;
const FWHT_SCALE: f32 = 0.0625;
const D1_SEED: u32 = 42;
const D2_CANON_SEED: u32 = 1042;
const BITS: [u32; 3] = [2, 3, 4];
const DEFAULT_BLOCKS_PER_TENSOR: usize = 4096;
const SAMPLE_SEED: u64 = 0xA11C_ED20_0B4C_1E00;
const EPS_SCALE: f64 = 5.960_464_477_539_063e-8; // 2^-24
const ROUNDTRIP_TOL: f32 = 1e-4;

// ── helpers reused from mq_kld_proxy ─────────────────────────────────────────

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn fwht_unnormalized_256(x: &mut [f32]) {
    assert_eq!(x.len(), G);
    let mut stride = 1;
    while stride < G {
        let mut i = 0;
        while i < G {
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
}

/// Full forward: D1 → FWHT → ×(1/16)·D2
fn cpu_fwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), G);
    for i in 0..G {
        x[i] *= s1[i];
    }
    fwht_unnormalized_256(x);
    for i in 0..G {
        x[i] *= FWHT_SCALE * s2[i];
    }
}

/// Full inverse: D2 → FWHT → ×(1/16)·D1  (matches mq_kld_proxy)
fn cpu_ifwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), G);
    for i in 0..G {
        x[i] *= s2[i];
    }
    fwht_unnormalized_256(x);
    for i in 0..G {
        x[i] *= FWHT_SCALE * s1[i];
    }
}

/// Pre-D2 rotate: D1 → FWHT → ×1/16  (U in search_affine_signs.hip)
fn pre_rotate_block(x: &mut [f32], s1: &[f32]) {
    assert_eq!(x.len(), G);
    for i in 0..G {
        x[i] *= s1[i];
    }
    fwht_unnormalized_256(x);
    for i in 0..G {
        x[i] *= FWHT_SCALE;
    }
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
        o => panic!("unsupported dtype {o}"),
    }
}

fn load_tensor(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let idx_bytes = std::fs::read(dir.join("model.safetensors.index.json"))
        .unwrap_or_else(|e| die(&format!("read index under {}: {e}", dir.display())));
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes)
        .unwrap_or_else(|e| die(&format!("parse index json: {e}")));
    let shard = idx["weight_map"][name]
        .as_str()
        .unwrap_or_else(|| die(&format!("tensor {name} missing from weight_map")));
    let sf = SafetensorsFile::open(&dir.join(shard))
        .unwrap_or_else(|e| die(&format!("open shard {}: {e}", shard)));
    let (meta, data) = sf
        .tensor_data(name)
        .unwrap_or_else(|| die(&format!("tensor_data missing for {name}")));
    (to_f32(data, &meta.dtype), meta.shape.clone())
}

fn load_imatrix_gguf(path: &Path) -> HashMap<String, Vec<f32>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let data = std::fs::read(path)
        .unwrap_or_else(|e| die(&format!("imatrix read {}: {e}", path.display())));
    let ver = (&data[4..8]).read_u32::<LittleEndian>().unwrap();
    let n_tensors = (&data[8..16]).read_u64::<LittleEndian>().unwrap();
    let n_kv = (&data[16..24]).read_u64::<LittleEndian>().unwrap();
    let _ = ver;
    let mut pos = 24usize;
    for _ in 0..n_kv {
        let klen = (&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap() as usize;
        pos += 8;
        pos += klen;
        let vtype = (&data[pos..pos + 4]).read_u32::<LittleEndian>().unwrap();
        pos += 4;
        if vtype == 8 {
            let slen = (&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap() as usize;
            pos += 8;
            pos += slen;
        } else if vtype == 4 {
            pos += 4;
        } else if vtype == 9 {
            let at = (&data[pos..pos + 4]).read_u32::<LittleEndian>().unwrap();
            pos += 4;
            let alen = (&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap();
            pos += 8;
            for _ in 0..alen {
                if at == 8 {
                    let slen = (&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap() as usize;
                    pos += 8;
                    pos += slen;
                }
            }
        } else {
            die(&format!("unsupported gguf kv vtype {vtype}"));
        }
    }
    let mut entries: Vec<(String, Vec<usize>, u32, usize)> = Vec::new();
    for _ in 0..n_tensors {
        let nlen = (&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap() as usize;
        pos += 8;
        let name = String::from_utf8(data[pos..pos + nlen].to_vec()).unwrap();
        pos += nlen;
        let ndims = (&data[pos..pos + 4]).read_u32::<LittleEndian>().unwrap() as usize;
        pos += 4;
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            shape.push((&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap() as usize);
            pos += 8;
        }
        let dtype = (&data[pos..pos + 4]).read_u32::<LittleEndian>().unwrap();
        pos += 4;
        let off = (&data[pos..pos + 8]).read_u64::<LittleEndian>().unwrap() as usize;
        pos += 8;
        entries.push((name, shape, dtype, off));
    }
    let base = (pos + 31) / 32 * 32;
    let mut out = HashMap::new();
    for (name, shape, dtype, off) in entries {
        if !name.ends_with(".in_sum2") || dtype != 0 || shape.len() != 1 {
            continue;
        }
        let k = shape[0];
        let start = base + off;
        let mut v = Vec::with_capacity(k);
        for i in 0..k {
            let o = start + i * 4;
            v.push(f32::from_le_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
            ]));
        }
        out.insert(name.strip_suffix(".in_sum2").unwrap().to_string(), v);
    }
    out
}

// ── V2 two-half codec ──────────────────────────────────────────────────────

/// Production MQ4V2 half reconstruction, generalized to hypothetical
/// matched-header 2-bit/3-bit research layouts.
/// Two contiguous G128 asymmetric affine grids; truncating f16 headers; floor(+0.5).
fn quant_v2_halves_recon(rot: &[f32], bits: u32) -> [f32; G] {
    assert_eq!(rot.len(), G);
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = [0.0f32; G];
    for h in 0..2 {
        let off = h * HALF;
        let slice = &rot[off..off + HALF];
        let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut st = 0.0f32;
        let mut degenerate = hi == lo;
        let z = if !degenerate {
            let step_f32 = (hi - lo) / levels;
            st = f16_to_f32(f32_to_f16(step_f32));
            let zt = f16_to_f32(f32_to_f16(lo));
            if st == 0.0 {
                degenerate = true;
            }
            zt
        } else {
            f16_to_f32(f32_to_f16(lo))
        };
        if degenerate {
            for i in 0..HALF {
                out[off + i] = z;
            }
        } else {
            let inv = 1.0 / st;
            for i in 0..HALF {
                let v = rot[off + i];
                let q = ((v - z) * inv + 0.5).floor().clamp(0.0, levels);
                out[off + i] = z + q * st;
            }
        }
    }
    out
}

// ── sampling / util ─────────────────────────────────────────────────────────

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn lcg64(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Deterministic sample of `n` distinct indices from `0..total`, spread across
/// the full range (not the first blocks). Partial Fisher–Yates with fixed LCG.
fn sample_block_indices(total: usize, n: usize, seed: u64) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    if n > total {
        die(&format!("BLOCKS_PER_TENSOR {n} > total blocks {total}"));
    }
    if n == total {
        return (0..total).collect();
    }
    let mut idx: Vec<usize> = (0..total).collect();
    let mut state = seed ^ ((total as u64) << 32) ^ (n as u64);
    if state == 0 {
        state = 1;
    }
    for i in 0..n {
        let j = i + (lcg64(&mut state) as usize % (total - i));
        idx.swap(i, j);
    }
    let mut out = idx[..n].to_vec();
    out.sort_unstable();
    out
}

fn load_candidate_signs(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| die(&format!("read candidate signs {}: {e}", path.display())));
    if bytes.len() != G * 4 {
        die(&format!(
            "SIGNS size {} bytes, expected {} (256×f32 LE)",
            bytes.len(),
            G * 4
        ));
    }
    let mut s = Vec::with_capacity(G);
    for c in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        if !(v == 1.0 || v == -1.0) {
            die(&format!("malformed signs: value {v} is not ±1.0"));
        }
        s.push(v);
    }
    s
}

fn verify_roundtrip(s1: &[f32], s2: &[f32]) {
    let mut x = [0.0f32; G];
    for i in 0..G {
        x[i] = ((i as f32) * 0.017 - 1.3).sin() * 3.1 + (i as f32) * 1e-3;
    }
    let orig = x;
    cpu_fwht_256(&mut x, s1, s2);
    cpu_ifwht_256(&mut x, s1, s2);
    let mut max_e = 0.0f32;
    for i in 0..G {
        max_e = max_e.max((x[i] - orig[i]).abs());
    }
    if !max_e.is_finite() || max_e > ROUNDTRIP_TOL {
        die(&format!(
            "FWHT roundtrip failed: max_abs_err={max_e} (tol={ROUNDTRIP_TOL})"
        ));
    }
    // Pre-D2 U then apply D2 must match full forward.
    let mut u = orig;
    pre_rotate_block(&mut u, s1);
    let mut v = u;
    for i in 0..G {
        v[i] *= s2[i];
    }
    let mut full = orig;
    cpu_fwht_256(&mut full, s1, s2);
    let mut max_e2 = 0.0f32;
    for i in 0..G {
        max_e2 = max_e2.max((v[i] - full[i]).abs());
    }
    if !max_e2.is_finite() || max_e2 > 1e-6 {
        die(&format!(
            "pre-D2∘D2 vs full forward mismatch: max_abs_err={max_e2}"
        ));
    }
}

// ── per-tensor evaluation ───────────────────────────────────────────────────

struct BlockData {
    /// Pre-D2 rotated block U (D1→FWHT→×1/16).
    u: [f32; G],
    /// Column offset of element 0 within the K dimension (for imatrix).
    col_off: usize,
    energy: f64,
}

struct ArmMetrics {
    mse_rot: f64,
    tail_mse: f64,
    block_nmse: f64,
    m1: f64,
    tsse: f64,
}

fn eval_arm(
    blocks: &[BlockData],
    tail_mask: &[bool],
    s1: &[f32],
    d2: &[f32],
    bits: u32,
    imatrix: &[f32],
    eps: f64,
    n_coeffs: usize,
    tail_count: usize,
) -> ArmMetrics {
    let mut sse = 0.0f64;
    let mut tsse = 0.0f64;
    let mut rsum = 0.0f64;
    let mut m1 = 0.0f64;
    let mut t_hit = 0usize;

    for (bi, blk) in blocks.iter().enumerate() {
        let mut v = [0.0f32; G];
        for i in 0..G {
            v[i] = blk.u[i] * d2[i];
        }
        let recon = quant_v2_halves_recon(&v, bits);

        let mut err_rot = [0.0f32; G];
        let mut blk_sse = 0.0f64;
        for i in 0..G {
            let e = (recon[i] as f64) - (v[i] as f64);
            let e2 = e * e;
            err_rot[i] = e as f32;
            sse += e2;
            blk_sse += e2;
            let gi = bi * G + i;
            if tail_mask[gi] {
                tsse += e2;
                t_hit += 1;
            }
        }
        rsum += blk_sse / (blk.energy + eps);

        // Inverse-rotate error with matching D2, then activation-weight.
        let mut e_orig = err_rot;
        cpu_ifwht_256(&mut e_orig, s1, d2);
        for i in 0..G {
            let col = blk.col_off + i;
            let w = imatrix[col] as f64;
            let e = e_orig[i] as f64;
            m1 += w * e * e;
        }
    }

    // tail_mask hits should be tail_count * 1 (same mask for every bit)
    let _ = t_hit;
    ArmMetrics {
        mse_rot: sse / n_coeffs as f64,
        tail_mse: if tail_count > 0 {
            tsse / tail_count as f64
        } else {
            0.0
        },
        block_nmse: rsum / blocks.len() as f64,
        m1,
        tsse,
    }
}

fn build_tail_mask(blocks: &[BlockData]) -> (Vec<bool>, usize, f32) {
    let n = blocks.len() * G;
    let mut abs_u = vec![0.0f32; n];
    for (b, blk) in blocks.iter().enumerate() {
        for i in 0..G {
            abs_u[b * G + i] = blk.u[i].abs();
        }
    }
    let mut k = (n + 99) / 100; // ceil(n * 0.01)
    if k < 1 {
        k = 1;
    }
    if k > n {
        k = n;
    }
    let mut order: Vec<usize> = (0..n).collect();
    // Top-k by |U| desc; ties prefer smaller linear index (block,index).
    order.select_nth_unstable_by(k - 1, |&a, &b| {
        match abs_u[a]
            .partial_cmp(&abs_u[b])
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Greater => std::cmp::Ordering::Less,
            std::cmp::Ordering::Less => std::cmp::Ordering::Greater,
            std::cmp::Ordering::Equal => a.cmp(&b),
        }
    });
    // order[0..k] holds the top-k partition (unsorted). Sort head for thr diagnostic.
    order[..k].sort_unstable_by(|&a, &b| {
        match abs_u[b]
            .partial_cmp(&abs_u[a])
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            o if o != std::cmp::Ordering::Equal => o,
            _ => a.cmp(&b),
        }
    });
    let mut mask = vec![false; n];
    for &idx in &order[..k] {
        mask[idx] = true;
    }
    let thr = abs_u[order[k - 1]];
    (mask, k, thr)
}

fn median_f64(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mid = xs.len() / 2;
    xs.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    xs[mid]
}

fn ratio(cand: f64, canon: f64) -> f64 {
    let d = if canon > 0.0 && canon.is_finite() {
        canon
    } else {
        1e-300
    };
    cand / d
}

fn imatrix_key(name: &str) -> String {
    let rest = name
        .strip_prefix("model.language_model.layers.")
        .unwrap_or_else(|| {
            die(&format!(
                "unsupported tensor name for imatrix mapping: {name}"
            ))
        });
    let (layer, suffix) = rest
        .split_once('.')
        .unwrap_or_else(|| die(&format!("malformed layer tensor name: {name}")));
    if layer.parse::<usize>().is_err() {
        die(&format!("invalid layer number in tensor name: {name}"));
    }
    let role = match suffix {
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        _ => die(&format!("unsupported Qwen3.8 imatrix role: {name}")),
    };
    format!("blk.{layer}.{role}")
}
// ── main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.len() > 3 || args[1] == "-h" || args[1] == "--help" {
        eprintln!(
            "usage: {} CANDIDATE.signs [BLOCKS_PER_TENSOR]\n\
             CANDIDATE.signs = 256×f32 LE ±1 (D2 only). Default blocks/tensor = {DEFAULT_BLOCKS_PER_TENSOR}.",
            args.first().map(|s| s.as_str()).unwrap_or("rotation_quality_proxy")
        );
        std::process::exit(if args.len() < 2 { 2 } else { 0 });
    }
    let cand_path = Path::new(&args[1]);
    let blocks_per = if args.len() == 3 {
        args[2]
            .parse::<usize>()
            .unwrap_or_else(|_| die(&format!("BLOCKS_PER_TENSOR not a usize: {}", args[2])))
    } else {
        DEFAULT_BLOCKS_PER_TENSOR
    };
    if blocks_per == 0 {
        die("BLOCKS_PER_TENSOR must be > 0");
    }

    let d2_cand = load_candidate_signs(cand_path);
    let s1 = gen_fwht_signs(D1_SEED, G);
    let d2_canon = gen_fwht_signs(D2_CANON_SEED, G);
    verify_roundtrip(&s1, &d2_canon);
    verify_roundtrip(&s1, &d2_cand);

    let dir_candidates = [
        "/home/kaden/qcal/parents/qwen3.8-27b",
        "/home/kaden/models/Qwen3.8-27B",
        "/scratch/parents/qwen3.8-27b",
    ];
    let dir = dir_candidates
        .iter()
        .map(Path::new)
        .find(|p| p.join("model.safetensors.index.json").exists())
        .unwrap_or_else(|| die("Qwen3.8-27B parent not found (tried qcal/models/scratch paths)"));

    let im_paths = [
        Path::new("/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf"),
        Path::new("/home/kaden/models/Qwen3.8-27B-imatrix.gguf"),
    ];
    let im_path = im_paths
        .iter()
        .copied()
        .find(|p| p.exists())
        .unwrap_or_else(|| die("imatrix GGUF not found"));

    println!("=== rotation_quality_proxy: heldout D2 oracle ===");
    println!("candidate: {}", cand_path.display());
    println!("model:     {}", dir.display());
    println!("imatrix:   {}", im_path.display());
    println!("D1 seed={D1_SEED}  D2_canon seed={D2_CANON_SEED}  blocks/tensor={blocks_per}");
    println!(
        "codec:     production MQ4V2 plus hypothetical matched-header 2/3-bit V2; \
         two G128 affine grids, trunc f16, floor(x+0.5), bits {:?}",
        BITS
    );

    let imatrix_map = load_imatrix_gguf(im_path);
    println!("imatrix entries: {}", imatrix_map.len());

    let targets: [(&str, &str); 3] = [
        (
            "L5.linear_attn.out_proj",
            "model.language_model.layers.5.linear_attn.out_proj.weight",
        ),
        (
            "L15.mlp.down_proj",
            "model.language_model.layers.15.mlp.down_proj.weight",
        ),
        (
            "L25.mlp.gate_proj",
            "model.language_model.layers.25.mlp.gate_proj.weight",
        ),
    ];

    #[derive(Clone, Copy)]
    struct Cell {
        mse_rot_c: f64,
        mse_rot_k: f64,
        mse_rot_r: f64,
        tail_c: f64,
        tail_k: f64,
        tail_r: f64,
        nmse_c: f64,
        nmse_k: f64,
        nmse_r: f64,
        m1_c: f64,
        m1_k: f64,
        m1_r: f64,
    }

    let mut cells: Vec<(&str, u32, Cell)> = Vec::with_capacity(9);

    for (ti, (label, name)) in targets.iter().enumerate() {
        let (f32d, shape) = load_tensor(dir, name);
        if shape.len() != 2 {
            die(&format!("{name}: expected 2D shape, got {shape:?}"));
        }
        let m = shape[0];
        let k = shape[1];
        if k % G != 0 {
            die(&format!("{name}: K={k} not divisible by 256"));
        }
        if f32d.len() != m * k {
            die(&format!(
                "{name}: numel {} != m*k {}*{}={}",
                f32d.len(),
                m,
                k,
                m * k
            ));
        }
        for (i, &v) in f32d.iter().enumerate() {
            if !v.is_finite() {
                die(&format!("{name}: non-finite weight at linear index {i}"));
            }
        }

        let im_key = imatrix_key(name);
        let im = imatrix_map.get(&im_key).unwrap_or_else(|| {
            die(&format!("missing imatrix entry {im_key} for {name}"));
        });
        if im.len() != k {
            die(&format!("{name}: imatrix len {} != K {k}", im.len()));
        }
        for (i, &v) in im.iter().enumerate() {
            if !v.is_finite() {
                die(&format!("{name}: non-finite imatrix[{i}]"));
            }
        }

        let gpr = k / G;
        let total_blocks = m * gpr;
        let n_take = blocks_per.min(total_blocks);
        let seed = SAMPLE_SEED
            .wrapping_add((ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(total_blocks as u64);
        let picks = sample_block_indices(total_blocks, n_take, seed);

        let mut blocks = Vec::with_capacity(n_take);
        for &flat in &picks {
            let row = flat / gpr;
            let g = flat % gpr;
            let start = row * k + g * G;
            let mut raw = [0.0f32; G];
            raw.copy_from_slice(&f32d[start..start + G]);
            let mut u = raw;
            pre_rotate_block(&mut u, &s1);
            let mut energy = 0.0f64;
            for i in 0..G {
                let v = u[i] as f64;
                energy += v * v;
            }
            if !energy.is_finite() {
                die(&format!("{name}: non-finite pre-D2 energy at block {flat}"));
            }
            blocks.push(BlockData {
                u,
                col_off: g * G,
                energy,
            });
        }

        let mut energies: Vec<f64> = blocks.iter().map(|b| b.energy).collect();
        let med_e = median_f64(&mut energies);
        let eps = EPS_SCALE * med_e;
        let (tail_mask, tail_count, tail_thr) = build_tail_mask(&blocks);
        let n_coeffs = blocks.len() * G;

        println!(
            "\n[{label}] {name}  shape=[{m},{k}]  sampled={}/{}  tail_n={tail_count} thr={:.6e} eps={:.6e}",
            blocks.len(),
            total_blocks,
            tail_thr,
            eps
        );

        for &bits in &BITS {
            let canon = eval_arm(
                &blocks, &tail_mask, &s1, &d2_canon, bits, im, eps, n_coeffs, tail_count,
            );
            let cand = eval_arm(
                &blocks, &tail_mask, &s1, &d2_cand, bits, im, eps, n_coeffs, tail_count,
            );
            let cell = Cell {
                mse_rot_c: canon.mse_rot,
                mse_rot_k: cand.mse_rot,
                mse_rot_r: ratio(cand.mse_rot, canon.mse_rot),
                tail_c: canon.tail_mse,
                tail_k: cand.tail_mse,
                tail_r: ratio(cand.tsse, canon.tsse), // tsse ratio == tail_mse ratio
                nmse_c: canon.block_nmse,
                nmse_k: cand.block_nmse,
                nmse_r: ratio(cand.block_nmse, canon.block_nmse),
                m1_c: canon.m1,
                m1_k: cand.m1,
                m1_r: ratio(cand.m1, canon.m1),
            };
            println!(
                "  bits={bits}  mse_rot  can={:.6e} cand={:.6e} r={:.6}  \
                 tail_mse can={:.6e} cand={:.6e} r={:.6}  \
                 nmse can={:.6e} cand={:.6e} r={:.6}  \
                 m1 can={:.6e} cand={:.6e} r={:.6}",
                cell.mse_rot_c,
                cell.mse_rot_k,
                cell.mse_rot_r,
                cell.tail_c,
                cell.tail_k,
                cell.tail_r,
                cell.nmse_c,
                cell.nmse_k,
                cell.nmse_r,
                cell.m1_c,
                cell.m1_k,
                cell.m1_r
            );
            cells.push((label, bits, cell));
        }
    }

    // ── macros ──────────────────────────────────────────────────────────────
    let ncell = cells.len() as f64;
    let mean =
        |f: fn(&Cell) -> f64| -> f64 { cells.iter().map(|(_, _, c)| f(c)).sum::<f64>() / ncell };

    let macro_mse = mean(|c| c.mse_rot_r);
    let macro_tail = mean(|c| c.tail_r);
    let macro_nmse = mean(|c| c.nmse_r);
    let macro_m1 = mean(|c| c.m1_r);

    // Per-bit macros (equal over tensors)
    println!("\n=== equal-tensor/bit macro ratios (cand/canon) ===");
    println!(
        "macro  mse_rot={macro_mse:.6}  tail={macro_tail:.6}  block_nmse={macro_nmse:.6}  m1={macro_m1:.6}"
    );
    println!("per-bit (equal over tensors):");
    for &bits in &BITS {
        let sub: Vec<&Cell> = cells
            .iter()
            .filter(|(_, b, _)| *b == bits)
            .map(|(_, _, c)| c)
            .collect();
        let n = sub.len() as f64;
        let mr = sub.iter().map(|c| c.mse_rot_r).sum::<f64>() / n;
        let tr = sub.iter().map(|c| c.tail_r).sum::<f64>() / n;
        let nr = sub.iter().map(|c| c.nmse_r).sum::<f64>() / n;
        let m1r = sub.iter().map(|c| c.m1_r).sum::<f64>() / n;
        println!("  bits={bits}  mse_rot={mr:.6}  tail={tr:.6}  block_nmse={nr:.6}  m1={m1r:.6}");
    }
    println!("per-tensor (equal over bits):");
    for (label, _) in &targets {
        let sub: Vec<&Cell> = cells
            .iter()
            .filter(|(l, _, _)| l == label)
            .map(|(_, _, c)| c)
            .collect();
        let n = sub.len() as f64;
        let mr = sub.iter().map(|c| c.mse_rot_r).sum::<f64>() / n;
        let tr = sub.iter().map(|c| c.tail_r).sum::<f64>() / n;
        let nr = sub.iter().map(|c| c.nmse_r).sum::<f64>() / n;
        let m1r = sub.iter().map(|c| c.m1_r).sum::<f64>() / n;
        println!("  {label:24}  mse_rot={mr:.6}  tail={tr:.6}  block_nmse={nr:.6}  m1={m1r:.6}");
    }

    // Worst cells per metric
    let worst = |name: &str, f: fn(&Cell) -> f64| {
        let (label, bits, cell) = cells
            .iter()
            .max_by(|a, b| {
                f(&a.2)
                    .partial_cmp(&f(&b.2))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        println!("  worst_{name}: {label} bits={bits}  ratio={:.6}", f(cell));
    };
    println!("worst cell (max cand/canon ratio):");
    worst("mse_rot", |c| c.mse_rot_r);
    worst("tail", |c| c.tail_r);
    worst("block_nmse", |c| c.nmse_r);
    worst("m1", |c| c.m1_r);

    // Primary summary line matching search objective naming
    // J_tail / J_all over equal tensor×bit cells of tail and block_nmse ratios.
    let j_tail = macro_tail;
    let j_all = macro_nmse;
    let worst_rel = cells
        .iter()
        .map(|(_, _, c)| c.tail_r.max(c.nmse_r))
        .fold(0.0f64, f64::max);
    let feasible = j_all <= 1.0;
    println!(
        "\n=== summary (search-aligned) ===\n\
         J_tail={j_tail:.6}  J_all={j_all:.6}  worst_rel={worst_rel:.6}  feasible={}\n\
         J_m1={macro_m1:.6}  J_mse_rot={macro_mse:.6}",
        if feasible { 1 } else { 0 }
    );
}
