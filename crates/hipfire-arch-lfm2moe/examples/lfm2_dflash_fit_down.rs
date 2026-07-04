// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Fit the final LFM2 DFlash FFN `down_proj` from block hidden labels.
//!
//! This is a deployed-path trainer slice. It replays saved
//! `lfm2_dflash_teacher_dump` blocks through the actual runtime forward,
//! captures the final-layer FFN activation (`gate_up`) and residual stream, and
//! solves a ridge least-squares down projection:
//!
//!   down_proj(gate_up) ~= dflash_block_target_hidden - residual_ffn
//!
//! Only the final layer's `mlp.down_proj.weight` is rewritten as F32.
//!
//! Usage:
//!   lfm2_dflash_fit_down --model <lfm2.hfq> --draft <in.dflash.hfq>
//!     --teacher-dump <dir> --out <out.dflash.hfq> [--ridge 1e-2]
//!     [--skip-blocks N] [--max-blocks N]

use hipfire_arch_lfm2moe::dflash::{
    lfm2_dflash_sync_gemm, lfm2_dflash_use_f16_weights, run_dflash_draft_for_logits,
    validate_dflash_contract,
};
use hipfire_arch_lfm2moe::{Lfm2MoeConfig, Lfm2MoeWeights};
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor, HfqPackage};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "Usage: lfm2_dflash_fit_down --model <lfm2.hfq> --draft <in.dflash.hfq> --teacher-dump <dir> --out <out.dflash.hfq> [--ridge 1e-2] [--skip-blocks N] [--max-blocks N]"
        );
        return Ok(());
    }

    let mut model: Option<PathBuf> = None;
    let mut draft: Option<PathBuf> = None;
    let mut teacher_dump: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut ridge = 1.0e-2f64;
    let mut skip_blocks = 0usize;
    let mut max_blocks: Option<usize> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
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
            "--skip-blocks" => {
                skip_blocks = argv[i + 1].parse()?;
                i += 2;
            }
            "--max-blocks" => {
                max_blocks = Some(argv[i + 1].parse()?);
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    if ridge < 0.0 || !ridge.is_finite() {
        return Err("--ridge must be finite and non-negative".into());
    }
    let model = model.ok_or("--model required")?;
    let draft = draft.ok_or("--draft required")?;
    let teacher_dump = teacher_dump.ok_or("--teacher-dump required")?;
    let out = out.ok_or("--out required")?;

    let dump = TeacherDump::load(&teacher_dump)?;
    if skip_blocks >= dump.blocks {
        return Err(format!(
            "--skip-blocks {skip_blocks} leaves no blocks in dump with {} blocks",
            dump.blocks
        )
        .into());
    }
    let eval_blocks = max_blocks
        .unwrap_or(dump.blocks - skip_blocks)
        .min(dump.blocks - skip_blocks);
    if eval_blocks == 0 {
        return Err("no teacher blocks to fit".into());
    }
    let end_block = skip_blocks + eval_blocks;

    let pkg = HfqPackage::open(&draft)?;
    let mut gpu = hipfire_rdna::Gpu::init()?;
    eprintln!("gpu: {}", gpu.arch);

    let mut target_hfq = HfqFile::open(&model)?;
    let target_cfg = Lfm2MoeConfig::from_hfq(&target_hfq)?;
    let draft_hfq = HfqFile::open(&draft)?;
    let draft_cfg =
        DflashConfig::from_hfq(&draft_hfq).ok_or("draft hfq missing dflash metadata")?;
    validate_dflash_contract(&target_cfg, &draft_cfg)?;
    dump.validate_against(&draft_cfg)?;

    let layer_idx = draft_cfg.n_layers.saturating_sub(1);
    let down_name = format!("layers.{layer_idx}.mlp.down_proj.weight");
    let down_entry = pkg
        .entry(&down_name)
        .ok_or_else(|| format!("draft lacks {down_name}"))?;
    if down_entry.shape != vec![dump.hidden as u32, draft_cfg.intermediate as u32] {
        return Err(format!(
            "{down_name} shape {:?} != expected [{}, {}]",
            down_entry.shape, dump.hidden, draft_cfg.intermediate
        )
        .into());
    }

    let max_ctx = (skip_blocks..end_block)
        .map(|b| dump.ctx_lens[b] as usize)
        .max()
        .unwrap_or(1)
        .max(1);
    eprintln!(
        "fit {down_name}: block_start={} blocks={} rows={} hidden={} inter={} block_size={} max_ctx={} ridge={}",
        skip_blocks,
        eval_blocks,
        eval_blocks * dump.block_size.saturating_sub(1),
        dump.hidden,
        draft_cfg.intermediate,
        dump.block_size,
        max_ctx,
        ridge,
    );

    let t_load = Instant::now();
    let target_weights = Lfm2MoeWeights::load(&mut target_hfq, &target_cfg, &mut gpu)?;
    let draft_weights = DflashWeights::load_with_f16(
        &mut gpu,
        &draft_hfq,
        &draft_cfg,
        lfm2_dflash_use_f16_weights(),
    )?;
    let mut draft_scratch = DflashScratch::new_with_mq_and_sync(
        &mut gpu,
        &draft_cfg,
        dump.block_size,
        max_ctx,
        draft_weights.has_mq,
        lfm2_dflash_sync_gemm(),
    )?;
    eprintln!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let supervised_rows = eval_blocks * dump.block_size.saturating_sub(1);
    let inter = draft_cfg.intermediate;
    let hidden = dump.hidden;
    let mut gate_up_rows = Vec::with_capacity(supervised_rows * inter);
    let mut target_delta_rows = Vec::with_capacity(supervised_rows * hidden);
    for b in skip_blocks..end_block {
        let position = dump.positions[b] as usize;
        let ctx = (dump.ctx_lens[b] as usize).min(position).max(1);
        let block_features = dump.features_for_block(b)?;
        draft_scratch.reset_upload_tracking();
        run_dflash_draft_for_logits(
            &mut gpu,
            &target_weights,
            &target_cfg,
            &draft_weights,
            &draft_cfg,
            &mut draft_scratch,
            block_features,
            position,
            dump.seed_tokens[b],
            Some(ctx),
            dump.block_size,
            None,
        )?;
        gpu.hip.device_synchronize()?;

        let rows = dump.block_size.saturating_sub(1);
        let gate_tensor = draft_scratch.gate_up.sub_offset(inter, rows * inter);
        let residual_tensor = draft_scratch.residual_ffn.sub_offset(hidden, rows * hidden);
        let gate_rows = gpu.download_f32(&gate_tensor)?;
        let residual_rows = gpu.download_f32(&residual_tensor)?;
        gate_up_rows.extend_from_slice(&gate_rows);
        for row in 0..rows {
            let global = b * rows + row;
            let target_off = global * hidden;
            let residual_off = row * hidden;
            for h in 0..hidden {
                target_delta_rows.push(
                    dump.target_block_hidden[target_off + h] - residual_rows[residual_off + h],
                );
            }
        }
    }

    let down_weight = fit_ridge_dual(
        &gate_up_rows,
        &target_delta_rows,
        supervised_rows,
        inter,
        hidden,
        ridge,
    )?;
    let delta_mse = reconstruction_mse(
        &gate_up_rows,
        &target_delta_rows,
        &down_weight,
        supervised_rows,
        inter,
        hidden,
    );
    let prefinal_mse = prefinal_mse(
        &gate_up_rows,
        &target_delta_rows,
        &down_weight,
        supervised_rows,
        inter,
        hidden,
    );
    eprintln!(
        "down fit: delta_mse={:.6e} prefinal_mse={:.6e}",
        delta_mse, prefinal_mse
    );

    let mut metadata: serde_json::Value = serde_json::from_str(&pkg.metadata_json)?;
    metadata["dflash_down_fit"] = json!({
        "producer": "lfm2_dflash_fit_down",
        "teacher_dump": teacher_dump,
        "skip_blocks": skip_blocks,
        "layer": layer_idx,
        "tensor": down_name,
        "blocks": eval_blocks,
        "rows": supervised_rows,
        "hidden": hidden,
        "intermediate": inter,
        "ridge": ridge,
        "delta_mse": delta_mse,
        "prefinal_mse": prefinal_mse,
        "down_quant_type": "F32"
    });
    let metadata_json = serde_json::to_string(&metadata)?;

    let down_bytes = f32_slice_to_f32_bytes(&down_weight);
    let mut tensors = Vec::with_capacity(pkg.entries().len());
    for entry in pkg.entries() {
        if entry.name == down_name {
            tensors.push(HfqMemTensor {
                name: entry.name.clone(),
                quant_type: 2,
                shape: entry.shape.clone(),
                group_size: 0,
                data: down_bytes.clone(),
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
    eprintln!("wrote {}", out.display());

    draft_scratch.free_gpu(&mut gpu);
    draft_weights.free_gpu(&mut gpu);
    Ok(())
}

struct TeacherDump {
    hidden: usize,
    num_extract: usize,
    blocks: usize,
    block_size: usize,
    positions: Vec<u32>,
    ctx_lens: Vec<u32>,
    seed_tokens: Vec<u32>,
    prompt_offsets: Vec<usize>,
    prompt_lengths: Vec<usize>,
    block_prompt_indices: Vec<u32>,
    features: Vec<f32>,
    target_block_hidden: Vec<f32>,
}

impl TeacherDump {
    fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.join("metadata.json"))?)?;
        if meta.get("format").and_then(|v| v.as_str()) != Some("hipfire-lfm2-dflash-teacher-v1") {
            return Err(format!(
                "{} is not a hipfire-lfm2-dflash-teacher-v1 dump",
                path.display()
            )
            .into());
        }
        let rows = value_usize(&meta, "rows")?;
        let hidden = value_usize(&meta, "hidden")?;
        let num_extract = value_usize(&meta, "num_extract")?;
        let (prompt_offsets, prompt_lengths) = prompt_spans(&meta, rows)?;
        let block_meta = meta
            .get("dflash_blocks")
            .ok_or("teacher dump lacks dflash_blocks")?;
        let blocks = value_usize(block_meta, "blocks")?;
        let block_size = value_usize(block_meta, "block_size")?;
        let positions = value_u32_array(block_meta, "positions")?;
        let ctx_lens = value_u32_array(block_meta, "ctx_lens")?;
        let seed_tokens = value_u32_array(block_meta, "seed_tokens")?;
        let block_prompt_indices =
            optional_u32_array(block_meta, "prompt_indices")?.unwrap_or_else(|| vec![0; blocks]);
        if positions.len() != blocks
            || ctx_lens.len() != blocks
            || seed_tokens.len() != blocks
            || block_prompt_indices.len() != blocks
        {
            return Err("dflash_blocks metadata length mismatch".into());
        }
        let features = read_f32_raw(&path.join("features.f32"))?;
        if features.len() != rows * num_extract * hidden {
            return Err(format!(
                "features.f32 floats {} != rows({rows}) * num_extract({num_extract}) * hidden({hidden})",
                features.len()
            )
            .into());
        }
        let target_block_hidden_path = path.join("dflash_block_target_hidden.f32");
        let target_block_hidden = read_f32_raw(&target_block_hidden_path)?;
        let expected = blocks * block_size.saturating_sub(1) * hidden;
        if target_block_hidden.len() != expected {
            return Err(format!(
                "{} floats {} != blocks({blocks}) * (block_size({block_size}) - 1) * hidden({hidden})",
                target_block_hidden_path.display(),
                target_block_hidden.len()
            )
            .into());
        }
        Ok(Self {
            hidden,
            num_extract,
            blocks,
            block_size,
            positions,
            ctx_lens,
            seed_tokens,
            prompt_offsets,
            prompt_lengths,
            block_prompt_indices,
            features,
            target_block_hidden,
        })
    }

    fn validate_against(
        &self,
        draft_cfg: &hipfire_runtime::dflash::DflashConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.hidden != draft_cfg.hidden {
            return Err(format!(
                "teacher hidden {} != draft hidden {}",
                self.hidden, draft_cfg.hidden
            )
            .into());
        }
        if self.num_extract != draft_cfg.num_extract() {
            return Err(format!(
                "teacher num_extract {} != draft num_extract {}",
                self.num_extract,
                draft_cfg.num_extract()
            )
            .into());
        }
        for (idx, &pos) in self.positions.iter().enumerate() {
            let pos = pos as usize;
            let prompt_idx = self.block_prompt_indices[idx] as usize;
            let prompt_len = *self.prompt_lengths.get(prompt_idx).ok_or_else(|| {
                format!(
                    "teacher block {idx} references missing prompt index {}",
                    self.block_prompt_indices[idx]
                )
            })?;
            if pos == 0 || pos > prompt_len {
                return Err(format!(
                    "teacher block {idx} position {pos} exceeds prompt {prompt_idx} rows {prompt_len}"
                )
                .into());
            }
        }
        Ok(())
    }

    fn features_for_block(&self, block: usize) -> Result<&[f32], Box<dyn std::error::Error>> {
        let prompt_idx = self.block_prompt_indices[block] as usize;
        let row_floats = self.num_extract * self.hidden;
        let offset = self.prompt_offsets[prompt_idx] * row_floats;
        let len = self.prompt_lengths[prompt_idx] * row_floats;
        self.features
            .get(offset..offset + len)
            .ok_or_else(|| format!("prompt {prompt_idx} feature slice out of range").into())
    }
}

fn fit_ridge_dual(
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
        let diag = a[col * n + col];
        for c in 0..n {
            a[col * n + c] /= diag;
            inv[col * n + c] /= diag;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row * n + col];
            if f == 0.0 {
                continue;
            }
            for c in 0..n {
                a[row * n + c] -= f * a[col * n + c];
                inv[row * n + c] -= f * inv[col * n + c];
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
    let mut sum = 0.0f64;
    for row in 0..rows {
        for h in 0..hidden {
            let mut pred = 0.0f64;
            for kk in 0..k {
                pred += x[row * k + kk] as f64 * w[h * k + kk] as f64;
            }
            let d = pred - y[row * hidden + h] as f64;
            sum += d * d;
        }
    }
    sum / (rows * hidden).max(1) as f64
}

fn prefinal_mse(
    x: &[f32],
    y_delta: &[f32],
    w: &[f32],
    rows: usize,
    k: usize,
    hidden: usize,
) -> f64 {
    reconstruction_mse(x, y_delta, w, rows, k, hidden)
}

fn f32_slice_to_f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn value_usize(v: &serde_json::Value, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("metadata missing unsigned `{key}`").into())
}

fn value_u32_array(
    v: &serde_json::Value,
    key: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let arr = v
        .get(key)
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("metadata missing array `{key}`"))?;
    arr.iter()
        .map(|x| {
            let n = x
                .as_u64()
                .ok_or_else(|| format!("metadata `{key}` contains a non-unsigned integer"))?;
            u32::try_from(n).map_err(|_| format!("metadata `{key}` value {n} overflows u32").into())
        })
        .collect()
}

fn optional_u32_array(
    v: &serde_json::Value,
    key: &str,
) -> Result<Option<Vec<u32>>, Box<dyn std::error::Error>> {
    if v.get(key).is_none() {
        return Ok(None);
    }
    value_u32_array(v, key).map(Some)
}

fn optional_usize_array(
    v: &serde_json::Value,
    key: &str,
) -> Result<Option<Vec<usize>>, Box<dyn std::error::Error>> {
    let Some(arr) = v.get(key) else {
        return Ok(None);
    };
    let arr = arr
        .as_array()
        .ok_or_else(|| format!("metadata `{key}` is not an array"))?;
    arr.iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("metadata `{key}` contains a non-unsigned integer").into())
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
        .map(Some)
}

fn prompt_spans(
    meta: &serde_json::Value,
    rows: usize,
) -> Result<(Vec<usize>, Vec<usize>), Box<dyn std::error::Error>> {
    let offsets = optional_usize_array(meta, "prompt_offsets")?.unwrap_or_else(|| vec![0]);
    let lengths = optional_usize_array(meta, "prompt_lengths")?.unwrap_or_else(|| vec![rows]);
    if offsets.len() != lengths.len() || offsets.is_empty() {
        return Err("prompt_offsets/prompt_lengths metadata mismatch".into());
    }
    for (idx, (&offset, &len)) in offsets.iter().zip(&lengths).enumerate() {
        if len == 0 || offset.checked_add(len).is_none_or(|end| end > rows) {
            return Err(
                format!("prompt {idx} span offset={offset} len={len} exceeds rows {rows}").into(),
            );
        }
    }
    Ok((offsets, lengths))
}

fn read_f32_raw(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!("{} byte length is not divisible by 4", path.display()).into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
