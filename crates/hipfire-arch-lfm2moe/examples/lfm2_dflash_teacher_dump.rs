// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Produce LFM2 DFlash block-teacher dumps.
//!
//! The output format is consumed by `lfm2_dflash_block_teacher_eval`: selected
//! target hidden rows in DFlash layout plus greedy target block labels/top-k
//! distributions for runtime-sidecar agreement checks.
//!
//! Usage:
//!   lfm2_dflash_teacher_dump --model <lfm2.hfq> --out <dir>
//!     [--draft <lfm2.dflash.hfq>] [--prompt <text>] [--prompts <file>]
//!     [--block-size N] [--target-layers 0,4,8] [--topk N] [--max-blocks N]
//!     [--prompt-mode concat|separate]
//!     [--position-mode prefix|spread|generation] [--positions 1,32,64] [--ctx N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
use std::io::Write;

#[cfg(feature = "deltanet")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_arch_lfm2moe::dflash::validate_dflash_contract;
    use hipfire_arch_lfm2moe::forward::{
        prefill_batch, prefill_batch_with_hidden_logits_and_final_hidden, Lfm2HiddenCapture,
    };
    use hipfire_arch_lfm2moe::{decode_step, Lfm2MoeConfig, Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::dflash::DflashConfig;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Instant;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut draft: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut prompts = Vec::new();
    let mut prompts_path: Option<PathBuf> = None;
    let mut prompt_mode = PromptMode::Concat;
    let mut block_size_override: Option<usize> = None;
    let mut target_layers_override: Option<Vec<usize>> = None;
    let mut topk = 8usize;
    let mut max_blocks = 8usize;
    let mut ctx_override: Option<usize> = None;
    let mut position_mode = PositionMode::Prefix;
    let mut positions_override: Option<Vec<usize>> = None;

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
            "--out" => {
                out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompts.push(argv[i + 1].clone());
                i += 2;
            }
            "--prompts" => {
                prompts_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt-mode" => {
                prompt_mode = PromptMode::parse(&argv[i + 1])?;
                i += 2;
            }
            "--block-size" | "--block" => {
                block_size_override = Some(argv[i + 1].parse()?);
                i += 2;
            }
            "--target-layers" => {
                target_layers_override = Some(parse_usize_list(&argv[i + 1])?);
                i += 2;
            }
            "--topk" => {
                topk = argv[i + 1].parse()?;
                i += 2;
            }
            "--max-blocks" => {
                max_blocks = argv[i + 1].parse()?;
                i += 2;
            }
            "--position-mode" => {
                position_mode = PositionMode::parse(&argv[i + 1])?;
                i += 2;
            }
            "--positions" => {
                positions_override = Some(parse_usize_list(&argv[i + 1])?);
                i += 2;
            }
            "--ctx" | "--ctx-slice" => {
                ctx_override = Some(argv[i + 1].parse()?);
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: lfm2_dflash_teacher_dump --model <lfm2.hfq> --out <dir> [--draft <lfm2.dflash.hfq>] [--prompt <text>] [--prompts <file>] [--prompt-mode concat|separate] [--block-size N] [--target-layers 0,4,8] [--topk N] [--max-blocks N] [--position-mode prefix|spread|generation] [--positions 1,32,64] [--ctx N]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let model = model.ok_or("--model required")?;
    let out = out.ok_or("--out required")?;
    if topk == 0 {
        return Err("--topk must be > 0".into());
    }
    if max_blocks == 0 {
        return Err("--max-blocks must be > 0".into());
    }

    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("gpu: {}", gpu.arch);

    let mut target_hfq = HfqFile::open(&model)?;
    let target_cfg = Lfm2MoeConfig::from_hfq(&target_hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&target_hfq.metadata_json)?;

    let draft_cfg = if let Some(path) = &draft {
        let draft_hfq = HfqFile::open(path)?;
        let cfg = DflashConfig::from_hfq(&draft_hfq).ok_or("draft hfq missing dflash metadata")?;
        validate_dflash_contract(&target_cfg, &cfg)?;
        Some(cfg)
    } else {
        None
    };
    let block_size = block_size_override
        .or_else(|| draft_cfg.as_ref().map(|cfg| cfg.block_size))
        .unwrap_or(4);
    if block_size < 2 {
        return Err("--block-size must be >= 2".into());
    }
    let target_layers = target_layers_override
        .or_else(|| draft_cfg.as_ref().map(|cfg| cfg.target_layer_ids.clone()))
        .unwrap_or_else(|| default_target_layers(target_cfg.num_hidden_layers));
    if target_layers.is_empty() {
        return Err("target layer list is empty".into());
    }
    for &layer in &target_layers {
        if layer >= target_cfg.num_hidden_layers {
            return Err(format!(
                "target layer {layer} out of range 0..{}",
                target_cfg.num_hidden_layers
            )
            .into());
        }
    }
    if let Some(cfg) = &draft_cfg {
        if target_layers != cfg.target_layer_ids {
            return Err(format!(
                "manual target layers {:?} do not match draft target layers {:?}",
                target_layers, cfg.target_layer_ids
            )
            .into());
        }
    }

    if let Some(path) = &prompts_path {
        let prompt_lines = std::fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        match prompt_mode {
            PromptMode::Concat => {
                if !prompt_lines.is_empty() {
                    prompts.push(prompt_lines.join("\n"));
                }
            }
            PromptMode::Separate => prompts.extend(prompt_lines),
        }
    }
    if prompts.is_empty() {
        prompts.push("Write a tiny Rust add function. Then explain why it works.".to_string());
    }

    let t_load = Instant::now();
    let target_weights = Lfm2MoeWeights::load(&mut target_hfq, &target_cfg, &mut gpu)?;
    let mut all_prompt_tokens = Vec::new();
    let mut prompt_offsets = Vec::with_capacity(prompts.len());
    let mut prompt_lengths = Vec::with_capacity(prompts.len());
    let mut features = Vec::new();
    let mut target_hidden = Vec::new();
    let mut block_prompt_indices = Vec::with_capacity(max_blocks);
    let mut positions = Vec::with_capacity(max_blocks);
    let mut ctx_lens = Vec::with_capacity(max_blocks);
    let mut seed_tokens = Vec::with_capacity(max_blocks);
    let mut target_tokens = Vec::with_capacity(max_blocks * block_size);
    let mut target_argmax = Vec::with_capacity(max_blocks * block_size);
    let mut target_topk_ids = Vec::with_capacity(max_blocks * block_size * topk);
    let mut target_topk_logits = Vec::with_capacity(max_blocks * block_size * topk);
    let mut target_block_hidden =
        Vec::with_capacity(max_blocks * block_size.saturating_sub(1) * target_cfg.hidden_size);
    let mut target_block_norm_hidden =
        Vec::with_capacity(max_blocks * block_size.saturating_sub(1) * target_cfg.hidden_size);

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        if positions.len() >= max_blocks {
            break;
        }
        let prompt_tokens = tokenizer.encode(prompt);
        if prompt_tokens.is_empty() {
            return Err(format!("prompt {prompt_idx} produced zero tokens").into());
        }
        let prompt_rows = prompt_tokens.len();
        let prompt_offset = all_prompt_tokens.len();
        prompt_offsets.push(prompt_offset);
        prompt_lengths.push(prompt_rows);
        all_prompt_tokens.extend_from_slice(&prompt_tokens);

        let mut capture = Lfm2HiddenCapture::new(
            target_cfg.num_hidden_layers,
            target_cfg.hidden_size,
            target_layers.clone(),
        )?;
        let mut full_state =
            Lfm2MoeState::new_with_max_seq(&mut gpu, &target_cfg, prompt_rows + block_size)?;
        let mut prompt_target_hidden = Vec::with_capacity(prompt_rows * target_cfg.hidden_size);
        let logits_per_pos = prefill_batch_with_hidden_logits_and_final_hidden(
            &target_cfg,
            &target_weights,
            &mut full_state,
            &mut gpu,
            &prompt_tokens,
            &mut capture,
            &mut prompt_target_hidden,
        )?;
        gpu.hip.device_synchronize()?;

        let prompt_features = capture.take_rows();
        let expected_features = prompt_rows * target_layers.len() * target_cfg.hidden_size;
        if prompt_features.len() != expected_features {
            return Err(format!(
                "prompt {prompt_idx} captured {} feature floats, expected {expected_features}",
                prompt_features.len()
            )
            .into());
        }
        let expected_target_hidden = prompt_rows * target_cfg.hidden_size;
        if prompt_target_hidden.len() != expected_target_hidden {
            return Err(format!(
                "prompt {prompt_idx} captured {} target hidden floats, expected {expected_target_hidden}",
                prompt_target_hidden.len()
            )
            .into());
        }
        features.extend_from_slice(&prompt_features);
        target_hidden.extend_from_slice(&prompt_target_hidden);

        let remaining = max_blocks - positions.len();
        let prompt_positions = select_positions(
            prompt_rows,
            remaining,
            position_mode,
            positions_override.as_deref(),
        )?;
        for position in prompt_positions {
            let block_idx = positions.len();
            block_prompt_indices.push(prompt_idx);
            positions.push(position);

            let prefix_logits = logits_row(&logits_per_pos, target_cfg.vocab_size, position - 1)?;
            let seed_token = argmax_u32(prefix_logits);
            seed_tokens.push(seed_token);
            ctx_lens.push(ctx_override.unwrap_or(position).min(position).max(1));

            let base = block_idx * block_size;
            target_tokens.resize(base + block_size, 0);
            target_argmax.resize(base + block_size, 0);
            target_topk_ids.resize((base + block_size) * topk, 0);
            target_topk_logits.resize((base + block_size) * topk, 0.0);

            target_tokens[base] = seed_token;
            target_argmax[base] = seed_token;
            write_topk(
                prefix_logits,
                topk,
                &mut target_topk_ids[base * topk..(base + 1) * topk],
                &mut target_topk_logits[base * topk..(base + 1) * topk],
            );

            let mut state =
                Lfm2MoeState::new_with_max_seq(&mut gpu, &target_cfg, prompt_rows + block_size)?;
            prefill_batch(
                &target_cfg,
                &target_weights,
                &mut state,
                &mut gpu,
                &prompt_tokens[..position],
            )?;
            let mut teacher_token = seed_token;
            for slot in 1..block_size {
                let pos = state.n_tokens as u32;
                let logits = decode_step(
                    &target_cfg,
                    &target_weights,
                    &mut state,
                    &mut gpu,
                    teacher_token,
                    pos,
                )?;
                let next = argmax_u32(&logits);
                let flat = base + slot;
                target_tokens[flat] = next;
                target_argmax[flat] = next;
                write_topk(
                    &logits,
                    topk,
                    &mut target_topk_ids[flat * topk..(flat + 1) * topk],
                    &mut target_topk_logits[flat * topk..(flat + 1) * topk],
                );
                let hidden = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("download block target hidden: {e:?}"))?;
                let norm_hidden = gpu
                    .download_f32(&state.final_norm_buf)
                    .map_err(|e| format!("download block target norm hidden: {e:?}"))?;
                target_block_hidden.extend_from_slice(&hidden);
                target_block_norm_hidden.extend_from_slice(&norm_hidden);
                teacher_token = next;
            }
        }
    }

    let rows = all_prompt_tokens.len();
    let blocks = positions.len();
    if blocks == 0 {
        return Err("no valid block positions selected".into());
    }

    eprintln!(
        "target hidden={} layers={} vocab={} prompts={} rows={} block={} blocks={} prompt_mode={} position_mode={} target_layers={:?} topk={}",
        target_cfg.hidden_size,
        target_cfg.num_hidden_layers,
        target_cfg.vocab_size,
        prompt_lengths.len(),
        rows,
        block_size,
        blocks,
        prompt_mode.as_str(),
        position_mode.as_str(),
        target_layers,
        topk,
    );
    eprintln!("loaded+prefilled in {:.2}s", t_load.elapsed().as_secs_f64());

    let expected_features = rows * target_layers.len() * target_cfg.hidden_size;
    if features.len() != expected_features {
        return Err(format!(
            "captured {} total feature floats, expected {expected_features}",
            features.len()
        )
        .into());
    }
    let expected_target_hidden = rows * target_cfg.hidden_size;
    if target_hidden.len() != expected_target_hidden {
        return Err(format!(
            "captured {} total target hidden floats, expected {expected_target_hidden}",
            target_hidden.len()
        )
        .into());
    }

    let expected_block_hidden = blocks * block_size.saturating_sub(1) * target_cfg.hidden_size;
    if target_block_hidden.len() != expected_block_hidden
        || target_block_norm_hidden.len() != expected_block_hidden
    {
        return Err(format!(
            "captured block hidden shapes pre={} norm={} expected={expected_block_hidden}",
            target_block_hidden.len(),
            target_block_norm_hidden.len(),
        )
        .into());
    }

    std::fs::create_dir_all(&out)?;
    write_f32_raw(&out.join("features.f32"), &features)?;
    write_f32_raw(&out.join("target_hidden.f32"), &target_hidden)?;
    write_f32_raw(
        &out.join("dflash_block_target_hidden.f32"),
        &target_block_hidden,
    )?;
    write_f32_raw(
        &out.join("dflash_block_target_norm_hidden.f32"),
        &target_block_norm_hidden,
    )?;
    write_u32_raw(&out.join("prompt_tokens.u32"), &all_prompt_tokens)?;
    write_u32_raw(&out.join("dflash_block_target_tokens.u32"), &target_tokens)?;
    write_u32_raw(&out.join("dflash_block_target_argmax.u32"), &target_argmax)?;
    write_u32_raw(&out.join("dflash_block_topk_ids.u32"), &target_topk_ids)?;
    write_f32_raw(
        &out.join("dflash_block_topk_logits.f32"),
        &target_topk_logits,
    )?;

    let metadata = json!({
        "format": "hipfire-lfm2-dflash-teacher-v1",
        "producer": "lfm2_dflash_teacher_dump",
        "model": model,
        "draft": draft,
        "rows": rows,
        "hidden": target_cfg.hidden_size,
        "num_extract": target_layers.len(),
        "target_layer_ids": target_layers,
        "target_hidden_shape": [rows, target_cfg.hidden_size],
        "prompt_mode": prompt_mode.as_str(),
        "prompt_count": prompt_lengths.len(),
        "prompt_tokens": all_prompt_tokens.len(),
        "prompt_offsets": prompt_offsets,
        "prompt_lengths": prompt_lengths,
        "dflash_blocks": {
            "blocks": blocks,
            "block_size": block_size,
            "topk": topk,
            "position_mode": position_mode.as_str(),
            "prompt_indices": block_prompt_indices,
            "positions": positions,
            "ctx_lens": ctx_lens,
            "seed_tokens": seed_tokens,
            "label_alignment": "slot0=prefix_argmax; slotN=target greedy argmax after consuming slotN-1",
            "target_block_hidden_shape": [blocks, block_size - 1, target_cfg.hidden_size],
            "target_block_norm_hidden_shape": [blocks, block_size - 1, target_cfg.hidden_size],
            "hidden_label_alignment": "row block,slot-1 matches draft logits row slot-1 and target state after consuming slot-1 token",
        },
    });
    std::fs::write(
        out.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)?,
    )?;
    eprintln!("wrote teacher dump: {}", out.display());
    Ok(())
}

#[cfg(feature = "deltanet")]
#[derive(Clone, Copy)]
enum PositionMode {
    Prefix,
    Spread,
    Generation,
}

#[cfg(feature = "deltanet")]
impl PositionMode {
    fn parse(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match s {
            "prefix" => Ok(Self::Prefix),
            "spread" => Ok(Self::Spread),
            "generation" | "gen" => Ok(Self::Generation),
            other => Err(format!(
                "unknown --position-mode `{other}`; expected prefix|spread|generation"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Spread => "spread",
            Self::Generation => "generation",
        }
    }
}

#[cfg(feature = "deltanet")]
#[derive(Clone, Copy)]
enum PromptMode {
    Concat,
    Separate,
}

#[cfg(feature = "deltanet")]
impl PromptMode {
    fn parse(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match s {
            "concat" => Ok(Self::Concat),
            "separate" | "lines" => Ok(Self::Separate),
            other => {
                Err(format!("unknown --prompt-mode `{other}`; expected concat|separate").into())
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Concat => "concat",
            Self::Separate => "separate",
        }
    }
}

#[cfg(feature = "deltanet")]
fn parse_usize_list(s: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    s.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid usize `{part}`: {e}").into())
        })
        .collect()
}

#[cfg(feature = "deltanet")]
fn select_positions(
    max_position: usize,
    max_blocks: usize,
    mode: PositionMode,
    explicit: Option<&[usize]>,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    if let Some(values) = explicit {
        let mut out = Vec::with_capacity(values.len().min(max_blocks));
        for &pos in values.iter().take(max_blocks) {
            if pos == 0 || pos > max_position {
                return Err(format!("position {pos} out of valid range 1..={max_position}").into());
            }
            if out.contains(&pos) {
                return Err(format!("duplicate position {pos} in --positions").into());
            }
            out.push(pos);
        }
        return Ok(out);
    }

    let count = max_blocks.min(max_position);
    if count == 0 {
        return Ok(Vec::new());
    }
    let positions = match mode {
        PositionMode::Prefix => (1..=count).collect(),
        PositionMode::Spread => spread_positions(max_position, count),
        PositionMode::Generation => vec![max_position],
    };
    Ok(positions)
}

#[cfg(feature = "deltanet")]
fn spread_positions(max_position: usize, count: usize) -> Vec<usize> {
    if count <= 1 {
        return vec![1];
    }
    let span = max_position - 1;
    let denom = count - 1;
    (0..count)
        .map(|i| 1 + (i * span + denom / 2) / denom)
        .collect()
}

#[cfg(feature = "deltanet")]
fn default_target_layers(n_layers: usize) -> Vec<usize> {
    if n_layers <= 1 {
        return vec![0];
    }
    let wanted = 5usize.min(n_layers);
    if wanted == 1 {
        return vec![n_layers - 1];
    }
    (0..wanted)
        .map(|i| i * (n_layers - 1) / (wanted - 1))
        .collect()
}

#[cfg(feature = "deltanet")]
fn logits_row(
    logits: &[f32],
    vocab: usize,
    row: usize,
) -> Result<&[f32], Box<dyn std::error::Error>> {
    let start = row.checked_mul(vocab).ok_or("logits row offset overflow")?;
    let end = start + vocab;
    logits
        .get(start..end)
        .ok_or_else(|| format!("logits row {row} out of range").into())
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

#[cfg(feature = "deltanet")]
fn write_topk(row: &[f32], k: usize, ids_out: &mut [u32], logits_out: &mut [f32]) {
    let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for i in 0..k {
        let (id, logit) = pairs[i.min(pairs.len() - 1)];
        ids_out[i] = id as u32;
        logits_out[i] = logit;
    }
}

#[cfg(feature = "deltanet")]
fn write_f32_raw(path: &std::path::Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;
    for &v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(feature = "deltanet")]
fn write_u32_raw(path: &std::path::Path, values: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;
    for &v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
