// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Evaluate an LFM2 DFlash sidecar against saved block-teacher windows.
//!
//! This is a training-evidence tool: it uses the actual DFlash runtime forward
//! on each saved `dflash_blocks` window from a `tiny_dflash_train --dump-teacher`
//! directory, then compares draft top-1 tokens against target argmax labels and
//! saved target top-k distributions.
//!
//! Usage:
//!   lfm2_dflash_block_teacher_eval --model <lfm2.hfq> --draft <lfm2.dflash.hfq>
//!     --teacher-dump <dir> [--max-blocks N] [--ctx-slice N] [--loss-gamma G]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_arch_lfm2moe::dflash::{run_dflash_draft_for_logits, validate_dflash_contract};
    use hipfire_arch_lfm2moe::{Lfm2MoeConfig, Lfm2MoeWeights};
    use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use hipfire_runtime::hfq::HfqFile;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Instant;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut draft: Option<PathBuf> = None;
    let mut teacher_dump: Option<PathBuf> = None;
    let mut max_blocks: Option<usize> = None;
    let mut ctx_slice: Option<usize> = None;
    let mut loss_gamma = 3.0f32;

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
            "--max-blocks" => {
                max_blocks = Some(argv[i + 1].parse()?);
                i += 2;
            }
            "--ctx-slice" => {
                ctx_slice = Some(argv[i + 1].parse()?);
                i += 2;
            }
            "--loss-gamma" => {
                loss_gamma = argv[i + 1].parse()?;
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: lfm2_dflash_block_teacher_eval --model <lfm2.hfq> --draft <lfm2.dflash.hfq> --teacher-dump <dir> [--max-blocks N] [--ctx-slice N] [--loss-gamma G]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let model = model.ok_or("--model required")?;
    let draft = draft.ok_or("--draft required")?;
    let teacher_dump = teacher_dump.ok_or("--teacher-dump required")?;
    let dump = TeacherDump::load(&teacher_dump)?;

    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("gpu: {}", gpu.arch);

    let mut target_hfq = HfqFile::open(&model)?;
    let target_cfg = Lfm2MoeConfig::from_hfq(&target_hfq)?;
    let draft_hfq = HfqFile::open(&draft)?;
    let draft_cfg =
        DflashConfig::from_hfq(&draft_hfq).ok_or("draft hfq missing dflash metadata")?;
    validate_dflash_contract(&target_cfg, &draft_cfg)?;
    dump.validate_against(&draft_cfg)?;

    let block_size = dump.block_size;
    if block_size > draft_cfg.block_size {
        return Err(format!(
            "teacher block_size {block_size} exceeds draft block_size {}",
            draft_cfg.block_size
        )
        .into());
    }
    let eval_blocks = max_blocks.unwrap_or(dump.blocks).min(dump.blocks);
    if eval_blocks == 0 {
        return Err("no teacher blocks to evaluate".into());
    }
    let max_ctx = (0..eval_blocks)
        .map(|b| ctx_slice.unwrap_or(dump.ctx_lens[b] as usize))
        .max()
        .unwrap_or(1)
        .max(1);

    eprintln!(
        "target hidden={} layers={} vocab={} draft_layers={} teacher_rows={} blocks={} block_size={} max_ctx={}",
        target_cfg.hidden_size,
        target_cfg.num_hidden_layers,
        target_cfg.vocab_size,
        draft_cfg.n_layers,
        dump.rows,
        eval_blocks,
        block_size,
        max_ctx,
    );

    let t_load = Instant::now();
    let target_weights = Lfm2MoeWeights::load(&mut target_hfq, &target_cfg, &mut gpu)?;
    let draft_weights = DflashWeights::load(&mut gpu, &draft_hfq, &draft_cfg)?;
    let mut draft_scratch = DflashScratch::new_with_mq(
        &mut gpu,
        &draft_cfg,
        block_size,
        max_ctx,
        draft_weights.has_mq,
    )?;
    eprintln!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let weights = block_position_weights(block_size, loss_gamma);
    let mut total_slots = 0usize;
    let mut total_argmax_hits = 0usize;
    let mut total_token_hits = 0usize;
    let mut total_topk_hits = 0usize;
    let mut weighted_ce = 0.0f64;
    let mut weighted_argmax_hits = 0.0f64;
    let mut weighted_topk_hits = 0.0f64;
    let mut total_weight = 0.0f64;
    let mut forward_ms = 0.0f64;

    for b in 0..eval_blocks {
        let position = dump.positions[b] as usize;
        let ctx = ctx_slice
            .unwrap_or(dump.ctx_lens[b] as usize)
            .min(position)
            .max(1);
        let seed_token = dump.seed_tokens[b];
        let t = Instant::now();
        let out = run_dflash_draft_for_logits(
            &mut gpu,
            &target_weights,
            &target_cfg,
            &draft_weights,
            &draft_cfg,
            &mut draft_scratch,
            &dump.features,
            position,
            seed_token,
            Some(ctx),
            block_size,
            None,
        )?;
        gpu.hip.device_synchronize()?;
        let block_ms = t.elapsed().as_secs_f64() * 1000.0;
        forward_ms += block_ms;

        let mut block_argmax_hits = 0usize;
        let mut block_topk_hits = 0usize;
        let mut block_ce = 0.0f64;
        let mut draft_first = Vec::with_capacity(block_size.saturating_sub(1));
        for slot in 1..block_size {
            let row = slot - 1;
            let logits = &out.logits[row * out.vocab_size..(row + 1) * out.vocab_size];
            let draft_tok = argmax_u32(logits);
            draft_first.push(draft_tok);
            let flat = b * block_size + slot;
            let target_argmax = dump.target_argmax[flat];
            let target_token = dump.target_tokens[flat];
            let topk_ids = &dump.target_topk_ids[flat * dump.topk..(flat + 1) * dump.topk];
            let topk_logits = &dump.target_topk_logits[flat * dump.topk..(flat + 1) * dump.topk];
            let ce = sampled_ce_from_vocab_logits(logits, topk_ids, topk_logits)?;
            let w = weights[slot - 1] as f64;
            let argmax_hit = draft_tok == target_argmax;
            let token_hit = draft_tok == target_token;
            let topk_hit = topk_ids.contains(&draft_tok);

            total_slots += 1;
            total_argmax_hits += usize::from(argmax_hit);
            total_token_hits += usize::from(token_hit);
            total_topk_hits += usize::from(topk_hit);
            total_weight += w;
            weighted_ce += ce * w;
            weighted_argmax_hits += if argmax_hit { w } else { 0.0 };
            weighted_topk_hits += if topk_hit { w } else { 0.0 };
            block_argmax_hits += usize::from(argmax_hit);
            block_topk_hits += usize::from(topk_hit);
            block_ce += ce * w;
        }
        println!(
            "{}",
            json!({
                "type": "block",
                "index": b,
                "position": position,
                "ctx": ctx,
                "seed_token": seed_token,
                "slots": block_size - 1,
                "argmax_hits": block_argmax_hits,
                "topk_hits": block_topk_hits,
                "weighted_ce": block_ce,
                "forward_ms": block_ms,
                "draft_first": draft_first,
            })
        );
    }

    let denom = total_slots.max(1) as f64;
    let weight_denom = total_weight.max(f64::MIN_POSITIVE);
    println!(
        "{}",
        json!({
            "type": "summary",
            "blocks": eval_blocks,
            "slots": total_slots,
            "argmax_hits": total_argmax_hits,
            "token_hits": total_token_hits,
            "topk_hits": total_topk_hits,
            "argmax_rate": total_argmax_hits as f64 / denom,
            "token_rate": total_token_hits as f64 / denom,
            "topk_rate": total_topk_hits as f64 / denom,
            "weighted_argmax_rate": weighted_argmax_hits / weight_denom,
            "weighted_topk_rate": weighted_topk_hits / weight_denom,
            "weighted_ce": weighted_ce / weight_denom,
            "forward_ms": forward_ms,
        })
    );

    draft_scratch.free_gpu(&mut gpu);
    draft_weights.free_gpu(&mut gpu);
    Ok(())
}

#[cfg(feature = "deltanet")]
struct TeacherDump {
    rows: usize,
    hidden: usize,
    num_extract: usize,
    blocks: usize,
    block_size: usize,
    topk: usize,
    positions: Vec<u32>,
    ctx_lens: Vec<u32>,
    seed_tokens: Vec<u32>,
    target_tokens: Vec<u32>,
    target_argmax: Vec<u32>,
    target_topk_ids: Vec<u32>,
    target_topk_logits: Vec<f32>,
    features: Vec<f32>,
}

#[cfg(feature = "deltanet")]
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
        let block_meta = meta
            .get("dflash_blocks")
            .ok_or("teacher dump lacks dflash_blocks")?;
        let blocks = value_usize(block_meta, "blocks")?;
        let block_size = value_usize(block_meta, "block_size")?;
        let topk = value_usize(block_meta, "topk")?;
        if topk == 0 {
            return Err("teacher dump dflash_blocks lacks target_topk data".into());
        }
        let positions = value_u32_array(block_meta, "positions")?;
        let ctx_lens = value_u32_array(block_meta, "ctx_lens")?;
        let seed_tokens = value_u32_array(block_meta, "seed_tokens")?;
        if positions.len() != blocks || ctx_lens.len() != blocks || seed_tokens.len() != blocks {
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
        let block_rows = blocks * block_size;
        let target_tokens = read_u32_raw(&path.join("dflash_block_target_tokens.u32"))?;
        let target_argmax = read_u32_raw(&path.join("dflash_block_target_argmax.u32"))?;
        let target_topk_ids = read_u32_raw(&path.join("dflash_block_topk_ids.u32"))?;
        let target_topk_logits = read_f32_raw(&path.join("dflash_block_topk_logits.f32"))?;
        if target_tokens.len() != block_rows || target_argmax.len() != block_rows {
            return Err("dflash block token/argmax shape mismatch".into());
        }
        if target_topk_ids.len() != block_rows * topk
            || target_topk_logits.len() != block_rows * topk
        {
            return Err("dflash block topk shape mismatch".into());
        }
        Ok(Self {
            rows,
            hidden,
            num_extract,
            blocks,
            block_size,
            topk,
            positions,
            ctx_lens,
            seed_tokens,
            target_tokens,
            target_argmax,
            target_topk_ids,
            target_topk_logits,
            features,
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
        for &pos in &self.positions {
            let pos = pos as usize;
            if pos == 0 || pos + self.block_size > self.rows {
                return Err(format!(
                    "teacher block position {pos} with block_size {} exceeds rows {}",
                    self.block_size, self.rows
                )
                .into());
            }
        }
        Ok(())
    }
}

#[cfg(feature = "deltanet")]
fn value_usize(v: &serde_json::Value, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("metadata missing unsigned `{key}`").into())
}

#[cfg(feature = "deltanet")]
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

#[cfg(feature = "deltanet")]
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

#[cfg(feature = "deltanet")]
fn read_u32_raw(path: &std::path::Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % std::mem::size_of::<u32>() != 0 {
        return Err(format!("{} byte length is not divisible by 4", path.display()).into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[cfg(feature = "deltanet")]
fn block_position_weights(block_size: usize, gamma: f32) -> Vec<f32> {
    let slots = block_size.saturating_sub(1);
    if slots == 0 {
        return Vec::new();
    }
    if gamma <= 0.0 {
        return vec![1.0 / slots as f32; slots];
    }
    let mut weights: Vec<f32> = (0..slots).map(|i| (-(i as f32) / gamma).exp()).collect();
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for w in &mut weights {
            *w /= sum;
        }
    }
    weights
}

#[cfg(feature = "deltanet")]
fn sampled_ce_from_vocab_logits(
    vocab_logits: &[f32],
    target_ids: &[u32],
    target_logits: &[f32],
) -> Result<f64, Box<dyn std::error::Error>> {
    if target_ids.len() != target_logits.len() || target_ids.is_empty() {
        return Err("sampled CE target shape mismatch".into());
    }
    let mut pred_logits = Vec::with_capacity(target_ids.len());
    for &id in target_ids {
        let idx = id as usize;
        if idx >= vocab_logits.len() {
            return Err(format!(
                "target topk id {id} outside draft vocab {}",
                vocab_logits.len()
            )
            .into());
        }
        pred_logits.push(vocab_logits[idx]);
    }
    let pred = stable_softmax(&pred_logits);
    let target = stable_softmax(target_logits);
    Ok(target
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| -(*t as f64) * (*p as f64).max(1e-20).ln())
        .sum())
}

#[cfg(feature = "deltanet")]
fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut out = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v - max).exp();
        out.push(e);
        sum += e;
    }
    if sum > 0.0 && sum.is_finite() {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

#[cfg(feature = "deltanet")]
fn argmax_u32(row: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in row.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx as u32
}
