// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Fit the LFM2 DFlash `fc.weight` projection from a teacher dump.
//!
//! This is a narrow runtime-aligned training slice: it uses
//! `lfm2_dflash_teacher_dump`'s real target hidden features and final hidden
//! labels, solves a ridge least-squares projection, and writes a new sidecar with
//! only `fc.weight` replaced as F32. Attention and MLP draft weights are copied
//! unchanged, so this is evidence plumbing rather than a complete DFlash trainer.
//!
//! Usage:
//!   lfm2_dflash_fit_fc --draft <in.dflash.hfq> --teacher-dump <dir> --out <out.dflash.hfq>
//!     [--ridge 1e-2] [--skip-rows N] [--max-rows N]

use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor, HfqPackage};
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "Usage: lfm2_dflash_fit_fc --draft <in.dflash.hfq> --teacher-dump <dir> --out <out.dflash.hfq> [--ridge 1e-2] [--skip-rows N] [--max-rows N]"
        );
        return Ok(());
    }

    let mut draft: Option<PathBuf> = None;
    let mut teacher_dump: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut ridge = 1.0e-2f64;
    let mut skip_rows = 0usize;
    let mut max_rows: Option<usize> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--draft" => {
                draft = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--teacher-dump" => {
                teacher_dump = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--ridge" => {
                ridge = argv[i + 1].parse()?;
                i += 2;
            }
            "--skip-rows" => {
                skip_rows = argv[i + 1].parse()?;
                i += 2;
            }
            "--max-rows" => {
                max_rows = Some(argv[i + 1].parse()?);
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    if ridge < 0.0 || !ridge.is_finite() {
        return Err("--ridge must be finite and non-negative".into());
    }
    let draft = draft.ok_or("--draft required")?;
    let teacher_dump = teacher_dump.ok_or("--teacher-dump required")?;
    let out = out.ok_or("--out required")?;

    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        teacher_dump.join("metadata.json"),
    )?)?;
    if meta.get("format").and_then(|v| v.as_str()) != Some("hipfire-lfm2-dflash-teacher-v1") {
        return Err(format!(
            "{} is not a hipfire-lfm2-dflash-teacher-v1 dump",
            teacher_dump.display()
        )
        .into());
    }
    let rows = value_usize(&meta, "rows")?;
    let hidden = value_usize(&meta, "hidden")?;
    let num_extract = value_usize(&meta, "num_extract")?;
    let k = hidden * num_extract;
    if skip_rows >= rows {
        return Err(
            format!("--skip-rows {skip_rows} leaves no rows in dump with {rows} rows").into(),
        );
    }
    let available_rows = rows - skip_rows;
    let use_rows = max_rows.unwrap_or(available_rows).min(available_rows);
    if use_rows == 0 {
        return Err("teacher dump has zero rows".into());
    }

    let features = read_f32_raw(&teacher_dump.join("features.f32"))?;
    let target_hidden = read_f32_raw(&teacher_dump.join("target_hidden.f32"))?;
    if features.len() != rows * k {
        return Err(format!(
            "features.f32 floats {} != rows({rows}) * hidden({hidden}) * num_extract({num_extract})",
            features.len()
        )
        .into());
    }
    if target_hidden.len() != rows * hidden {
        return Err(format!(
            "target_hidden.f32 floats {} != rows({rows}) * hidden({hidden})",
            target_hidden.len()
        )
        .into());
    }

    let pkg = HfqPackage::open(&draft)?;
    let fc = pkg
        .entry("fc.weight")
        .ok_or("draft sidecar lacks fc.weight")?;
    if fc.shape != vec![hidden as u32, k as u32] {
        return Err(format!("fc.weight shape {:?} != expected [{hidden}, {k}]", fc.shape).into());
    }

    eprintln!(
        "fit fc.weight: skip_rows={} rows={} hidden={} num_extract={} ridge={} draft={}",
        skip_rows,
        use_rows,
        hidden,
        num_extract,
        ridge,
        draft.display()
    );
    let feature_start = skip_rows * k;
    let feature_end = feature_start + use_rows * k;
    let target_start = skip_rows * hidden;
    let target_end = target_start + use_rows * hidden;
    let fc_weight = fit_fc_ridge(
        &features[feature_start..feature_end],
        &target_hidden[target_start..target_end],
        use_rows,
        k,
        hidden,
        ridge,
    )?;
    let train_mse = reconstruction_mse(
        &features[feature_start..feature_end],
        &target_hidden[target_start..target_end],
        &fc_weight,
        use_rows,
        k,
        hidden,
    );

    let mut metadata: serde_json::Value = serde_json::from_str(&pkg.metadata_json)?;
    metadata["dflash_fc_fit"] = json!({
        "producer": "lfm2_dflash_fit_fc",
        "teacher_dump": teacher_dump,
        "skip_rows": skip_rows,
        "rows": use_rows,
        "hidden": hidden,
        "num_extract": num_extract,
        "ridge": ridge,
        "train_mse": train_mse,
        "fc_quant_type": "F32"
    });
    let metadata_json = serde_json::to_string(&metadata)?;

    let mut tensors = Vec::with_capacity(pkg.entries().len());
    let fc_bytes = f32_slice_to_f32_bytes(&fc_weight);
    for entry in pkg.entries() {
        if entry.name == "fc.weight" {
            tensors.push(HfqMemTensor {
                name: entry.name.clone(),
                quant_type: 2,
                shape: entry.shape.clone(),
                group_size: 0,
                data: fc_bytes.clone(),
            });
        } else {
            tensors.push(HfqMemTensor {
                name: entry.name.clone(),
                quant_type: entry.quant_type,
                shape: entry.shape.clone(),
                group_size: entry.group_size,
                data: pkg
                    .blob_data(&entry.name)
                    .ok_or_else(|| format!("missing blob for {}", entry.name))?
                    .to_vec(),
            });
        }
    }

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_hfqm_package_mem(&out, pkg.arch_id, &metadata_json, &tensors)?;
    eprintln!(
        "wrote {} (fc F32 bytes={} train_mse={:.6e})",
        out.display(),
        fc_bytes.len(),
        train_mse
    );
    Ok(())
}

fn fit_fc_ridge(
    x: &[f32],
    y: &[f32],
    rows: usize,
    k: usize,
    hidden: usize,
    ridge: f64,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut gram = vec![0.0f64; rows * rows];
    for i in 0..rows {
        for j in 0..=i {
            let mut acc = 0.0f64;
            let xi = &x[i * k..(i + 1) * k];
            let xj = &x[j * k..(j + 1) * k];
            for kk in 0..k {
                acc += xi[kk] as f64 * xj[kk] as f64;
            }
            if i == j {
                acc += ridge;
            }
            gram[i * rows + j] = acc;
            gram[j * rows + i] = acc;
        }
    }
    let inv = invert_matrix(gram, rows)?;

    let mut beta = vec![0.0f64; rows * hidden];
    for m in 0..rows {
        for j in 0..rows {
            let a = inv[m * rows + j];
            let yj = &y[j * hidden..(j + 1) * hidden];
            for h in 0..hidden {
                beta[m * hidden + h] += a * yj[h] as f64;
            }
        }
    }

    let mut w = vec![0.0f32; hidden * k];
    for h in 0..hidden {
        for kk in 0..k {
            let mut acc = 0.0f64;
            for m in 0..rows {
                acc += beta[m * hidden + h] * x[m * k + kk] as f64;
            }
            w[h * k + kk] = acc as f32;
        }
    }
    Ok(w)
}

fn invert_matrix(mut a: Vec<f64>, n: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col * n + col].abs();
        for row in col + 1..n {
            let cand = a[row * n + col].abs();
            if cand > pivot_abs {
                pivot = row;
                pivot_abs = cand;
            }
        }
        if pivot_abs <= f64::EPSILON {
            return Err(format!("ridge system is singular at column {col}").into());
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
                inv.swap(col * n + c, pivot * n + c);
            }
        }
        let scale = a[col * n + col];
        for c in 0..n {
            a[col * n + c] /= scale;
            inv[col * n + c] /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..n {
                a[row * n + c] -= factor * a[col * n + c];
                inv[row * n + c] -= factor * inv[col * n + c];
            }
        }
    }
    Ok(inv)
}

fn reconstruction_mse(
    x: &[f32],
    y: &[f32],
    w: &[f32],
    rows: usize,
    k: usize,
    hidden: usize,
) -> f64 {
    let mut se = 0.0f64;
    for row in 0..rows {
        for h in 0..hidden {
            let mut pred = 0.0f64;
            for kk in 0..k {
                pred += x[row * k + kk] as f64 * w[h * k + kk] as f64;
            }
            let d = pred - y[row * hidden + h] as f64;
            se += d * d;
        }
    }
    se / (rows * hidden).max(1) as f64
}

fn value_usize(v: &serde_json::Value, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("metadata missing unsigned `{key}`").into())
}

fn read_f32_raw(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!("{} byte length is not divisible by 4", path.display()).into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn f32_slice_to_f32_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(f32_data));
    for &v in f32_data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
