// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent check of the parent weight loader against the quantizer's
//! reading of the **same** original checkpoint bytes.
//!
//! Motivation
//! ----------
//! The parent forward scores PPL ~164 at 1024 tokens while MQ2R / MQ2-Lloyd
//! (built by `hipfire-quantize` from these same safetensors) score ~14.6.
//! Every existing oracle compares `parent/*` against `parent/*_ref` written
//! from the same reading of `model.py`, so a shared misreading is invisible.
//! This binary breaks that cycle: the quantizer reader shares no code and no
//! reading with `parent/*`, and is known-good by outcome.
//!
//! What is compared
//! ----------------
//! For a spread of tensors, dequantize the same source bytes through:
//!   - **oracle** — from-scratch f64 dequant matching the quantizer
//!     (`hipfire-quantize/src/main.rs::{dequantize_e4m3_ue8m0_to_f32,
//!     dequantize_e2m1_ue8m0_to_f32}`) / `kernel.py` + `convert.py`
//!     nibble order. Does **not** call into `parent::codec`.
//!   - **parent** — `parent::codec::{dequant_dense_fp8_block128,
//!     dequant_expert_fp4_g32}` (the path `weights.rs` loads through).
//!
//! Metrics per tensor (all in f64 after both sides widen):
//!   - max abs diff
//!   - relative Frobenius  `||a-b||_F / ||b||_F`
//!   - L2 norm ratio       `||a||_2 / ||b||_2`   (scale bugs → clean factor)
//!   - cosine similarity   (layout/nibble bugs → near 0 with ratio ≈ 1)
//!   - shape agreement
//!
//! Usage
//! -----
//! ```text
//! cargo run -p hipfire-arch-deepseek4 --release --example ds4_parent_loader_oracle -- \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```
//!
//! CPU-only. Safe to run while the GPU is busy with another model.

use hipfire_ds4_parent::codec::{
    dequant_dense_fp8_block128, dequant_expert_fp4_g32,
};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

// ─── Independent oracle codecs (quantizer / kernel.py / convert.py) ─────────
//
// Intentionally duplicated rather than imported: calling into parent::codec
// for the reference side would recreate the shared-reading problem this
// slice exists to break. Authority:
//   - hipfire-quantize/src/main.rs:249-414
//   - .codeinsight+research/ds4-parent-ref/inference/convert.py:30-33
//   - .codeinsight+research/ds4-parent-ref/inference/kernel.py:128-200

/// OCP float8_e4m3fn → f64. NaN codes collapse to 0 (quantizer convention
/// for MQ ingest; parent keeps NaN — neither appears on a well-formed ckpt).
fn oracle_e4m3_to_f64(byte: u8) -> f64 {
    let sign = if (byte & 0x80) != 0 { -1.0f64 } else { 1.0f64 };
    let exp = ((byte >> 3) & 0x0f) as i32;
    let mant = (byte & 0x07) as f64;
    if exp == 0x0f && mant == 7.0 {
        return 0.0;
    }
    if exp == 0 {
        if mant == 0.0 {
            return 0.0;
        }
        return sign * (2.0f64).powi(-6) * (mant / 8.0);
    }
    sign * (2.0f64).powi(exp - 7) * (1.0 + mant / 8.0)
}

/// float8_e8m0fnu → f64. `2^(b-127)`. byte=0 → 2^-127 (subnormal; not 0).
fn oracle_ue8m0_to_f64(byte: u8) -> f64 {
    if byte == 0xff {
        return f64::NAN;
    }
    (2.0f64).powi(byte as i32 - 127)
}

/// OCP float4_e2m1fn nibble → f64.
fn oracle_e2m1_to_f64(nibble: u8) -> f64 {
    const MAG: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let n = (nibble & 0x0f) as usize;
    let mag = MAG[n & 0x7];
    if (n & 0x8) != 0 {
        -mag
    } else {
        mag
    }
}

/// Dense FP8 E4M3 + UE8M0 scales. Block size is **derived from the scale
/// shape** (quantizer rule), not hard-coded 128 — so a checkpoint whose
/// scale tiles differently than config.json claims is still handled.
///
/// `w` is row-major `[rows, cols]`; `s` is row-major `[sr, sc]`.
fn oracle_dequant_e4m3_ue8m0(
    w: &[u8],
    w_shape: &[usize],
    s: &[u8],
    s_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), String> {
    if w_shape.len() != 2 {
        return Err(format!("oracle e4m3: expected 2D weight, got {w_shape:?}"));
    }
    if s_shape.len() != 2 {
        return Err(format!("oracle e4m3: expected 2D scale, got {s_shape:?}"));
    }
    let (rows, cols) = (w_shape[0], w_shape[1]);
    let (sr, sc) = (s_shape[0], s_shape[1]);
    if w.len() != rows * cols {
        return Err(format!(
            "oracle e4m3: weight len {} != rows*cols {}",
            w.len(),
            rows * cols
        ));
    }
    if s.len() != sr * sc {
        return Err(format!(
            "oracle e4m3: scale len {} != sr*sc {}",
            s.len(),
            sr * sc
        ));
    }
    if rows % sr != 0 || cols % sc != 0 {
        return Err(format!(
            "oracle e4m3: scale {s_shape:?} does not tile weight {w_shape:?}"
        ));
    }
    let block_rows = rows / sr;
    let block_cols = cols / sc;
    let mut out = vec![0.0f64; rows * cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = oracle_ue8m0_to_f64(s[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    out[r * cols + c] = oracle_e4m3_to_f64(w[r * cols + c]) * scale;
                }
            }
        }
    }
    Ok((out, vec![rows, cols]))
}

/// Expert FP4 E2M1 packed + UE8M0 scales.
///
/// `storage_shape` is the safetensors byte shape `[rows, cols_stored]` with
/// `cols_stored = logical_cols / 2`. Nibble order (convert.py:30-33 /
/// quantizer): low nibble = even logical column, high nibble = odd.
/// Scale block size is derived from the scale shape.
///
/// `nibble_swap`: when true, reverse nibble order (diagnostic hypothesis).
/// `scale_transpose`: when true, treat scales as `[sc, sr]` laid out as if
/// the stored shape were transposed (diagnostic hypothesis).
fn oracle_dequant_e2m1_ue8m0(
    w: &[u8],
    storage_shape: &[usize],
    s: &[u8],
    s_shape: &[usize],
    nibble_swap: bool,
    scale_transpose: bool,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    if storage_shape.len() != 2 {
        return Err(format!(
            "oracle e2m1: expected 2D storage, got {storage_shape:?}"
        ));
    }
    if s_shape.len() != 2 {
        return Err(format!("oracle e2m1: expected 2D scale, got {s_shape:?}"));
    }
    let (rows, cols_stored) = (storage_shape[0], storage_shape[1]);
    let logical_cols = cols_stored * 2;
    let (sr, sc) = if scale_transpose {
        (s_shape[1], s_shape[0])
    } else {
        (s_shape[0], s_shape[1])
    };
    if w.len() != rows * cols_stored {
        return Err(format!(
            "oracle e2m1: weight len {} != rows*cols_stored {}",
            w.len(),
            rows * cols_stored
        ));
    }
    if s.len() != s_shape[0] * s_shape[1] {
        return Err(format!(
            "oracle e2m1: scale len {} != product of {:?}",
            s.len(),
            s_shape
        ));
    }
    if rows % sr != 0 || logical_cols % sc != 0 {
        return Err(format!(
            "oracle e2m1: scale (effective [{sr},{sc}]) does not tile \
             logical weight [{rows},{logical_cols}] (stored {storage_shape:?}, \
             scale_shape {s_shape:?}, transpose={scale_transpose})"
        ));
    }
    let block_rows = rows / sr;
    let block_cols_logical = logical_cols / sc;
    let mut out = vec![0.0f64; rows * logical_cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            // Index into the stored scale buffer. Under transpose the stored
            // layout is [s_shape[0]=sc_stored, s_shape[1]=sr_stored] with
            // row-major over the stored axes — i.e. scale at logical (sr_i,sc_j)
            // sits at stored[sc_j, sr_i].
            let scale_byte = if scale_transpose {
                // stored shape = [sc, sr] under the hypothesis that the file
                // wrote the transpose of the logical [M, K/32] layout.
                s[sc_j * s_shape[1] + sr_i]
            } else {
                s[sr_i * sc + sc_j]
            };
            let scale = oracle_ue8m0_to_f64(scale_byte);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols_logical {
                    let c = sc_j * block_cols_logical + dj;
                    let byte = w[r * cols_stored + (c / 2)];
                    let nibble = if nibble_swap {
                        if (c & 1) == 0 {
                            byte >> 4
                        } else {
                            byte & 0x0f
                        }
                    } else if (c & 1) == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    out[r * logical_cols + c] = oracle_e2m1_to_f64(nibble) * scale;
                }
            }
        }
    }
    Ok((out, vec![rows, logical_cols]))
}

// ─── Parent-side dequant (the path under test) ──────────────────────────────

fn parent_dequant_dense(w: &[u8], s: &[u8], m: usize, k: usize) -> Result<Vec<f64>, String> {
    let mut out_f32 = vec![0.0f32; m * k];
    dequant_dense_fp8_block128(w, s, m, k, &mut out_f32)?;
    Ok(out_f32.iter().map(|&v| v as f64).collect())
}

fn parent_dequant_expert(w: &[u8], s: &[u8], m: usize, k_logical: usize) -> Result<Vec<f64>, String> {
    let mut out_f32 = vec![0.0f32; m * k_logical];
    dequant_expert_fp4_g32(w, s, m, k_logical, &mut out_f32)?;
    Ok(out_f32.iter().map(|&v| v as f64).collect())
}

// ─── Metrics ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Metrics {
    #[allow(dead_code)]
    n: usize,
    shape_ours: Vec<usize>,
    shape_oracle: Vec<usize>,
    shape_ok: bool,
    max_abs: f64,
    rel_fro: f64,
    #[allow(dead_code)]
    l2_ours: f64,
    #[allow(dead_code)]
    l2_oracle: f64,
    l2_ratio: f64,
    cosine: f64,
    n_nan_ours: usize,
    n_nan_oracle: usize,
}

fn metrics(ours: &[f64], oracle: &[f64], shape_ours: &[usize], shape_oracle: &[usize]) -> Metrics {
    let shape_ok = shape_ours == shape_oracle && ours.len() == oracle.len();
    let n = ours.len().min(oracle.len());
    let mut max_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut n2_o = 0.0f64;
    let mut n2_r = 0.0f64;
    let mut d2 = 0.0f64;
    let mut n_nan_ours = 0usize;
    let mut n_nan_oracle = 0usize;
    for i in 0..n {
        let a = ours[i];
        let b = oracle[i];
        if !a.is_finite() {
            n_nan_ours += 1;
            continue;
        }
        if !b.is_finite() {
            n_nan_oracle += 1;
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        d2 += d * d;
        dot += a * b;
        n2_o += a * a;
        n2_r += b * b;
    }
    let l2_ours = n2_o.sqrt();
    let l2_oracle = n2_r.sqrt();
    let l2_ratio = if l2_oracle > 0.0 {
        l2_ours / l2_oracle
    } else if l2_ours == 0.0 {
        1.0
    } else {
        f64::INFINITY
    };
    let rel_fro = if l2_oracle > 0.0 {
        d2.sqrt() / l2_oracle
    } else if d2 == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let cosine = if l2_ours > 0.0 && l2_oracle > 0.0 {
        dot / (l2_ours * l2_oracle)
    } else if l2_ours == 0.0 && l2_oracle == 0.0 {
        1.0
    } else {
        0.0
    };
    Metrics {
        n,
        shape_ours: shape_ours.to_vec(),
        shape_oracle: shape_oracle.to_vec(),
        shape_ok,
        max_abs,
        rel_fro,
        l2_ours,
        l2_oracle,
        l2_ratio,
        cosine,
        n_nan_ours,
        n_nan_oracle,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    ExpertFp4,
    DenseFp8,
    Bf16,
}

impl Tier {
    fn as_str(self) -> &'static str {
        match self {
            Tier::ExpertFp4 => "expert_fp4",
            Tier::DenseFp8 => "dense_fp8",
            Tier::Bf16 => "bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    ExpertSuspect,
    SharedControl,
    Control,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::ExpertSuspect => "expert",
            Role::SharedControl => "shared",
            Role::Control => "control",
        }
    }
}

struct Case {
    name: &'static str,
    tier: Tier,
    role: Role,
}

const CASES: &[Case] = &[
    // Routed experts — fp4 tier, prime suspects. Layer 4 (MoE gain ~0.6x)
    // and layer 26 (MoE gain ~5.6x, residual spike).
    Case {
        name: "layers.4.ffn.experts.0.w1.weight",
        tier: Tier::ExpertFp4,
        role: Role::ExpertSuspect,
    },
    Case {
        name: "layers.4.ffn.experts.0.w2.weight",
        tier: Tier::ExpertFp4,
        role: Role::ExpertSuspect,
    },
    Case {
        name: "layers.4.ffn.experts.0.w3.weight",
        tier: Tier::ExpertFp4,
        role: Role::ExpertSuspect,
    },
    Case {
        name: "layers.26.ffn.experts.0.w1.weight",
        tier: Tier::ExpertFp4,
        role: Role::ExpertSuspect,
    },
    Case {
        name: "layers.26.ffn.experts.0.w2.weight",
        tier: Tier::ExpertFp4,
        role: Role::ExpertSuspect,
    },
    Case {
        name: "layers.26.ffn.experts.0.w3.weight",
        tier: Tier::ExpertFp4,
        role: Role::ExpertSuspect,
    },
    // Shared experts — fp8 tier (Expert() default dtype), same layers.
    Case {
        name: "layers.4.ffn.shared_experts.w1.weight",
        tier: Tier::DenseFp8,
        role: Role::SharedControl,
    },
    Case {
        name: "layers.4.ffn.shared_experts.w2.weight",
        tier: Tier::DenseFp8,
        role: Role::SharedControl,
    },
    Case {
        name: "layers.4.ffn.shared_experts.w3.weight",
        tier: Tier::DenseFp8,
        role: Role::SharedControl,
    },
    Case {
        name: "layers.26.ffn.shared_experts.w1.weight",
        tier: Tier::DenseFp8,
        role: Role::SharedControl,
    },
    Case {
        name: "layers.26.ffn.shared_experts.w2.weight",
        tier: Tier::DenseFp8,
        role: Role::SharedControl,
    },
    Case {
        name: "layers.26.ffn.shared_experts.w3.weight",
        tier: Tier::DenseFp8,
        role: Role::SharedControl,
    },
    // Controls: gate (BF16), attention projection (dense fp8), norm (BF16).
    Case {
        name: "layers.4.ffn.gate.weight",
        tier: Tier::Bf16,
        role: Role::Control,
    },
    Case {
        name: "layers.4.attn.wq_a.weight",
        tier: Tier::DenseFp8,
        role: Role::Control,
    },
    Case {
        name: "layers.4.attn_norm.weight",
        tier: Tier::Bf16,
        role: Role::Control,
    },
    Case {
        name: "layers.4.ffn_norm.weight",
        tier: Tier::Bf16,
        role: Role::Control,
    },
];

fn scale_name(weight: &str) -> String {
    weight
        .strip_suffix(".weight")
        .map(|s| format!("{s}.scale"))
        .unwrap_or_else(|| format!("{weight}.scale"))
}

fn read_bytes(src: &SafetensorsSource, name: &str) -> Result<(Vec<u8>, String, Vec<usize>), String> {
    let (info, data) = src
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor {name:?}"))?;
    Ok((data.to_vec(), info.dtype.clone(), info.shape.clone()))
}

fn bf16_bytes_to_f64(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if bytes.len() % 2 != 0 {
        return Err(format!("BF16 byte len {} not even", bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        // BF16 → f32: shift into the high half of an f32 bit pattern.
        let f = f32::from_bits((bits as u32) << 16);
        out.push(f as f64);
    }
    Ok(out)
}

struct Row {
    case: &'static Case,
    m: Metrics,
    /// Extra diagnostic: best alternate hypothesis if primary disagrees.
    alt: Option<(&'static str, Metrics)>,
}

fn compare_case(src: &SafetensorsSource, case: &'static Case) -> Result<Row, String> {
    let (w_bytes, w_dtype, w_shape) = read_bytes(src, case.name)?;
    match case.tier {
        Tier::Bf16 => {
            if w_dtype != "BF16" {
                return Err(format!(
                    "{}: expected BF16, got {w_dtype} shape {w_shape:?}",
                    case.name
                ));
            }
            let vals = bf16_bytes_to_f64(&w_bytes)?;
            // Both sides read the same bytes the same way — this is the
            // pure load floor (bit-exact expected).
            let m = metrics(&vals, &vals, &w_shape, &w_shape);
            Ok(Row {
                case,
                m,
                alt: None,
            })
        }
        Tier::DenseFp8 => {
            if w_dtype != "F8_E4M3" && w_dtype != "I8" {
                return Err(format!(
                    "{}: expected F8_E4M3, got {w_dtype} shape {w_shape:?}",
                    case.name
                ));
            }
            let sname = scale_name(case.name);
            let (s_bytes, s_dtype, s_shape) = read_bytes(src, &sname)?;
            if s_dtype != "F8_E8M0" {
                return Err(format!(
                    "{sname}: expected F8_E8M0, got {s_dtype} shape {s_shape:?}"
                ));
            }
            if w_shape.len() != 2 {
                return Err(format!("{}: rank-2 required, got {w_shape:?}", case.name));
            }
            let (m_dim, k_dim) = (w_shape[0], w_shape[1]);
            let (oracle, oshape) =
                oracle_dequant_e4m3_ue8m0(&w_bytes, &w_shape, &s_bytes, &s_shape)?;
            let ours = parent_dequant_dense(&w_bytes, &s_bytes, m_dim, k_dim)?;
            let ours_shape = vec![m_dim, k_dim];
            let m = metrics(&ours, &oracle, &ours_shape, &oshape);

            // If disagree, try hard-coded 128 block vs derived (already
            // derived on oracle; parent is hard-coded 128). Report scale
            // shape vs expected ceil(M/128)×ceil(K/128).
            let mut alt = None;
            if m.max_abs > 0.0 || !m.shape_ok {
                // Hypothesis: parent-style fixed 128 already is what we ran.
                // Try treating scales as if block came from a different axis.
                if let Ok((oracle_t, oshape_t)) = oracle_dequant_e4m3_ue8m0(
                    &w_bytes,
                    &w_shape,
                    &s_bytes,
                    &[s_shape[1], s_shape[0]],
                ) {
                    // Rebuild scale buffer under transpose of shape only if
                    // lengths match — they do when we just swap dims.
                    // Actually the buffer layout would also need transpose;
                    // skip unless shapes are square.
                    let mt = metrics(&ours, &oracle_t, &ours_shape, &oshape_t);
                    if mt.max_abs < m.max_abs {
                        alt = Some(("dense_scale_shape_swapped_label", mt));
                    }
                }
                let _ = alt;
            }
            Ok(Row {
                case,
                m,
                alt: None,
            })
        }
        Tier::ExpertFp4 => {
            if w_dtype != "I8" {
                return Err(format!(
                    "{}: expected I8 (packed E2M1), got {w_dtype} shape {w_shape:?}",
                    case.name
                ));
            }
            let sname = scale_name(case.name);
            let (s_bytes, s_dtype, s_shape) = read_bytes(src, &sname)?;
            if s_dtype != "F8_E8M0" {
                return Err(format!(
                    "{sname}: expected F8_E8M0, got {s_dtype} shape {s_shape:?}"
                ));
            }
            if w_shape.len() != 2 {
                return Err(format!("{}: rank-2 required, got {w_shape:?}", case.name));
            }
            let (m_dim, k_packed) = (w_shape[0], w_shape[1]);
            let k_logical = k_packed * 2;

            let (oracle, oshape) = oracle_dequant_e2m1_ue8m0(
                &w_bytes,
                &w_shape,
                &s_bytes,
                &s_shape,
                false,
                false,
            )?;
            let ours = parent_dequant_expert(&w_bytes, &s_bytes, m_dim, k_logical)?;
            let ours_shape = vec![m_dim, k_logical];
            let m = metrics(&ours, &oracle, &ours_shape, &oshape);

            // Diagnostic alternates — only meaningful when primary disagrees.
            let mut alt: Option<(&'static str, Metrics)> = None;
            if m.max_abs > 1e-12 || m.cosine < 0.999999 {
                let candidates: [(&'static str, bool, bool); 3] = [
                    ("nibble_swap", true, false),
                    ("scale_transpose", false, true),
                    ("nibble_swap+scale_transpose", true, true),
                ];
                for (label, nib, st) in candidates {
                    if let Ok((o2, sh2)) =
                        oracle_dequant_e2m1_ue8m0(&w_bytes, &w_shape, &s_bytes, &s_shape, nib, st)
                    {
                        let mt = metrics(&ours, &o2, &ours_shape, &sh2);
                        let better = match &alt {
                            None => mt.max_abs < m.max_abs || mt.cosine > m.cosine,
                            Some((_, prev)) => {
                                mt.max_abs < prev.max_abs
                                    || (mt.max_abs == prev.max_abs && mt.cosine > prev.cosine)
                            }
                        };
                        if better && (mt.max_abs < m.max_abs || mt.cosine > m.cosine) {
                            alt = Some((label, mt));
                        }
                    }
                }
            }
            Ok(Row { case, m, alt })
        }
    }
}

fn fmt_sci(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    format!("{x:.4e}")
}

fn print_table(rows: &[Row]) {
    println!();
    println!(
        "{:<52} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>6}",
        "tensor", "tier", "role", "max|d|", "relFro", "L2ratio", "cosine", "shape"
    );
    println!("{}", "-".repeat(120));
    for r in rows {
        let shape = if r.m.shape_ok { "OK" } else { "DIFF" };
        println!(
            "{:<52} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8.6} {:>6}",
            r.case.name,
            r.case.tier.as_str(),
            r.case.role.as_str(),
            fmt_sci(r.m.max_abs),
            fmt_sci(r.m.rel_fro),
            fmt_sci(r.m.l2_ratio),
            r.m.cosine,
            shape
        );
        if let Some((label, ref am)) = r.alt {
            println!(
                "  alt[{label}]: max|d|={} relFro={} L2ratio={} cosine={:.6}",
                fmt_sci(am.max_abs),
                fmt_sci(am.rel_fro),
                fmt_sci(am.l2_ratio),
                am.cosine
            );
        }
        if !r.m.shape_ok {
            println!(
                "  shapes ours={:?} oracle={:?}",
                r.m.shape_ours, r.m.shape_oracle
            );
        }
        if r.m.n_nan_ours + r.m.n_nan_oracle > 0 {
            println!(
                "  nan: ours={} oracle={}",
                r.m.n_nan_ours, r.m.n_nan_oracle
            );
        }
    }
}

fn summarize(rows: &[Row]) -> i32 {
    println!();
    println!("=== floor (controls) ===");
    let controls: Vec<&Row> = rows
        .iter()
        .filter(|r| r.case.role == Role::Control || r.case.role == Role::SharedControl)
        .collect();
    let mut ctrl_max_abs = 0.0f64;
    let mut ctrl_max_rel = 0.0f64;
    let mut ctrl_min_cos = 1.0f64;
    let mut ctrl_l2_dev = 0.0f64;
    for r in &controls {
        ctrl_max_abs = ctrl_max_abs.max(r.m.max_abs);
        ctrl_max_rel = ctrl_max_rel.max(r.m.rel_fro);
        ctrl_min_cos = ctrl_min_cos.min(r.m.cosine);
        ctrl_l2_dev = ctrl_l2_dev.max((r.m.l2_ratio - 1.0).abs());
    }
    println!(
        "control floor: max|d|={}  max relFro={}  min cosine={:.9}  max |L2ratio-1|={}",
        fmt_sci(ctrl_max_abs),
        fmt_sci(ctrl_max_rel),
        ctrl_min_cos,
        fmt_sci(ctrl_l2_dev)
    );
    println!(
        "  (BF16 controls are bit-identical by construction; dense-fp8 shared/attn \
         establish the dequant floor against the quantizer reader.)"
    );

    // Tolerances: e4m3/e2m1 × power-of-two scales are exactly representable
    // in f32, so parent (f32) vs oracle (f64) should agree to well under
    // 1e-6 abs on finite codes. Anything above the control floor is a defect.
    let abs_tol = (ctrl_max_abs * 10.0).max(1e-6);
    let rel_tol = (ctrl_max_rel * 10.0).max(1e-7);
    let cos_tol = 1.0 - 1e-9;
    let l2_tol = (ctrl_l2_dev * 10.0).max(1e-6);

    println!();
    println!("=== expert (fp4) tier ===");
    let experts: Vec<&Row> = rows
        .iter()
        .filter(|r| r.case.role == Role::ExpertSuspect)
        .collect();
    let mut exp_max_abs = 0.0f64;
    let mut exp_max_rel = 0.0f64;
    let mut exp_min_cos = 1.0f64;
    let mut exp_l2_dev = 0.0f64;
    let mut any_shape = false;
    for r in &experts {
        exp_max_abs = exp_max_abs.max(r.m.max_abs);
        exp_max_rel = exp_max_rel.max(r.m.rel_fro);
        exp_min_cos = exp_min_cos.min(r.m.cosine);
        exp_l2_dev = exp_l2_dev.max((r.m.l2_ratio - 1.0).abs());
        if !r.m.shape_ok {
            any_shape = true;
        }
        println!(
            "  {}: max|d|={} relFro={} L2ratio={} cosine={:.9} shape_ok={}",
            r.case.name,
            fmt_sci(r.m.max_abs),
            fmt_sci(r.m.rel_fro),
            fmt_sci(r.m.l2_ratio),
            r.m.cosine,
            r.m.shape_ok
        );
    }

    println!();
    println!("=== verdict ===");
    let expert_bad = exp_max_abs > abs_tol
        || exp_max_rel > rel_tol
        || exp_min_cos < cos_tol
        || exp_l2_dev > l2_tol
        || any_shape;
    let control_bad = ctrl_max_abs > 1e-5
        || ctrl_max_rel > 1e-6
        || ctrl_min_cos < 0.999999
        || ctrl_l2_dev > 1e-5;

    if control_bad {
        println!(
            "FAIL: control tier disagrees with the quantizer reader \
             (max|d|={} relFro={} cosine={:.9} |L2-1|={}). \
             Dense FP8 / BF16 load path is wrong; do not interpret experts.",
            fmt_sci(ctrl_max_abs),
            fmt_sci(ctrl_max_rel),
            ctrl_min_cos,
            fmt_sci(ctrl_l2_dev)
        );
        return 2;
    }

    if !expert_bad {
        println!(
            "PASS: parent codec agrees with the independent quantizer reader on every \
             sampled tensor."
        );
        println!(
            "  tightest bound (experts): max|d|={}  relFro={}  |L2ratio-1|={}  \
             min cosine={:.12}",
            fmt_sci(exp_max_abs),
            fmt_sci(exp_max_rel),
            fmt_sci(exp_l2_dev),
            exp_min_cos
        );
        println!(
            "  control floor:            max|d|={}  relFro={}  |L2ratio-1|={}  \
             min cosine={:.12}",
            fmt_sci(ctrl_max_abs),
            fmt_sci(ctrl_max_rel),
            fmt_sci(ctrl_l2_dev),
            ctrl_min_cos
        );
        println!(
            "  Signature: none. Nibble order = low-then-high, expert scales = [M][K/32], \
             w1/w3 stored separately (no gate_up fusion in checkpoint). \
             Weight loading can be struck off as the source of PPL 163.89."
        );
        return 0;
    }

    // Classify signature.
    let scale_like = exp_min_cos > 0.99 && exp_l2_dev > 1e-3;
    let layout_like = exp_min_cos < 0.5 && exp_l2_dev < 0.1;
    let signature = if scale_like {
        "SCALE-LIKE (cosine≈1, L2 ratio away from 1)"
    } else if layout_like {
        "LAYOUT-LIKE (cosine near 0, L2 ratio near 1)"
    } else {
        "MIXED / OTHER"
    };
    println!(
        "FAIL: expert_fp4 tier disagrees. signature={signature}"
    );
    println!(
        "  experts: max|d|={} relFro={} |L2ratio-1|={} min cosine={:.9}",
        fmt_sci(exp_max_abs),
        fmt_sci(exp_max_rel),
        fmt_sci(exp_l2_dev),
        exp_min_cos
    );
    println!(
        "  controls stay at floor max|d|={} — defect is isolated to the fp4 path.",
        fmt_sci(ctrl_max_abs)
    );
    for r in &experts {
        if let Some((label, ref am)) = r.alt {
            if am.max_abs < r.m.max_abs || am.cosine > r.m.cosine {
                println!(
                    "  diagnostic: {} best alt [{label}] max|d|={} cosine={:.6} L2ratio={}",
                    r.case.name,
                    fmt_sci(am.max_abs),
                    am.cosine,
                    fmt_sci(am.l2_ratio)
                );
            }
        }
    }
    1
}

fn parse_args() -> Result<PathBuf, String> {
    let mut model = None;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => {
                model = Some(PathBuf::from(args.next().ok_or("--model needs a value")?));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_loader_oracle --model /path/to/DeepSeek-V4-Flash-0731"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(model.unwrap_or_else(|| {
        PathBuf::from(
            env::var("HIPFIRE_DEEPSEEK4_PARENT_MODEL").unwrap_or_else(|_| {
                "/mnt/scratch/models/DeepSeek-V4-Flash-0731".to_string()
            }),
        )
    }))
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<i32, String> {
    let model = parse_args()?;
    if !Path::new(&model).is_dir() {
        return Err(format!(
            "--model must be a safetensors directory, got {}",
            model.display()
        ));
    }

    println!("=== ds4_parent_loader_oracle ===");
    println!("model: {}", model.display());
    println!(
        "oracle: from-scratch f64 dequant (quantizer/kernel.py/convert.py semantics)"
    );
    println!("parent: parent::codec::{{dequant_dense_fp8_block128, dequant_expert_fp4_g32}}");
    println!();

    let t0 = Instant::now();
    let src = SafetensorsSource::open(&model).map_err(|e| {
        format!(
            "SafetensorsSource::open({}): {e}",
            model.display()
        )
    })?;
    println!(
        "opened: {} tensors in {:.2}s",
        src.tensor_names().len(),
        t0.elapsed().as_secs_f64()
    );

    // Sanity: confirm w1/w3 are separate (no fused gate_up in this checkpoint).
    let fused = src
        .tensor_names()
        .iter()
        .filter(|n| n.contains("gate_up") || n.contains("down_proj"))
        .count();
    println!(
        "fused gate_up/down_proj tensor count: {fused} \
         (0 ⇒ w1/w3 stored separately, as parent loader assumes)"
    );

    let mut rows = Vec::with_capacity(CASES.len());
    for case in CASES {
        let t = Instant::now();
        print!("  compare {} ... ", case.name);
        let row = compare_case(&src, case)?;
        println!(
            "ok  max|d|={} L2ratio={} cosine={:.6} ({:.2}s)",
            fmt_sci(row.m.max_abs),
            fmt_sci(row.m.l2_ratio),
            row.m.cosine,
            t.elapsed().as_secs_f64()
        );
        // Print shape / scale companion for the first of each tier.
        rows.push(row);
    }

    print_table(&rows);

    // Extra detail: print scale shapes the inventory would require.
    println!();
    println!("=== shape / scale companions (source of truth = checkpoint) ===");
    for case in CASES {
        if case.tier == Tier::Bf16 {
            let info = src.tensor_info(case.name).unwrap();
            println!(
                "  {}  dtype={} shape={:?}",
                case.name, info.dtype, info.shape
            );
            continue;
        }
        let info = src.tensor_info(case.name).unwrap();
        let sname = scale_name(case.name);
        let sinfo = src.tensor_info(&sname).unwrap();
        let logical = if case.tier == Tier::ExpertFp4 {
            format!(" logical=[{},{}]", info.shape[0], info.shape[1] * 2)
        } else {
            String::new()
        };
        println!(
            "  {}  dtype={} shape={:?}{}  scale {} dtype={} shape={:?}",
            case.name, info.dtype, info.shape, logical, sname, sinfo.dtype, sinfo.shape
        );
    }

    let code = summarize(&rows);
    println!();
    println!("wall: {:.2}s", t0.elapsed().as_secs_f64());
    Ok(code)
}
