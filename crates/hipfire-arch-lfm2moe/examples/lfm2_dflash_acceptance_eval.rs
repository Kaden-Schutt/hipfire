// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Multi-prompt LFM2 DFlash acceptance evaluator.
//!
//! This is sidecar evidence tooling, not a training loop. It runs the same
//! greedy LFM2 DFlash speculative step used by the daemon over one or more
//! prompts and reports accepted/offered draft-token counts.
//!
//! Usage:
//!   lfm2_dflash_acceptance_eval --model <lfm2.hfq> --draft <lfm2.dflash.hfq>
//!     [--prompt <text>] [--prompts <line-delimited.txt>] [--max-prompts N]
//!     [--max-tokens N] [--block N] [--ctx-slice N] [--eos ID] [--ignore-eos]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_arch_lfm2moe::dflash::{
        lfm2_dflash_sync_gemm, lfm2_dflash_use_f16_weights, spec_step_dflash,
        validate_dflash_contract, Lfm2DflashTargetSnapshot,
    };
    use hipfire_arch_lfm2moe::forward::{prefill_batch_with_hidden_logits, Lfm2HiddenCapture};
    use hipfire_arch_lfm2moe::{Lfm2MoeConfig, Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Instant;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut draft: Option<PathBuf> = None;
    let mut prompts = Vec::new();
    let mut prompts_path: Option<PathBuf> = None;
    let mut max_prompts = 8usize;
    let mut max_tokens = 16usize;
    let mut block_override: Option<usize> = None;
    let mut ctx_slice: Option<usize> = None;
    let mut eos_extra: Option<u32> = None;
    let mut ignore_eos = false;

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
            "--prompt" => {
                prompts.push(argv[i + 1].clone());
                i += 2;
            }
            "--prompts" => {
                prompts_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--max-prompts" => {
                max_prompts = argv[i + 1].parse().expect("--max-prompts expects usize");
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = argv[i + 1].parse().expect("--max-tokens expects usize");
                i += 2;
            }
            "--block" => {
                block_override = Some(argv[i + 1].parse().expect("--block expects usize"));
                i += 2;
            }
            "--ctx-slice" => {
                ctx_slice = Some(argv[i + 1].parse().expect("--ctx-slice expects usize"));
                i += 2;
            }
            "--eos" => {
                eos_extra = Some(argv[i + 1].parse().expect("--eos expects u32"));
                i += 2;
            }
            "--ignore-eos" => {
                ignore_eos = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: lfm2_dflash_acceptance_eval --model <lfm2.hfq> --draft <lfm2.dflash.hfq> [--prompt <text>] [--prompts <line-delimited.txt>] [--max-prompts N] [--max-tokens N] [--block N] [--ctx-slice N] [--eos ID] [--ignore-eos]"
                );
                return Ok(());
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    if let Some(path) = &prompts_path {
        let s = std::fs::read_to_string(path)?;
        prompts.extend(
            s.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(ToOwned::to_owned),
        );
    }
    if prompts.is_empty() {
        prompts.push("Write a tiny Rust add function.".to_string());
    }
    prompts.truncate(max_prompts);
    if prompts.is_empty() || max_tokens == 0 {
        println!(
            "{}",
            json!({"type":"summary","prompts":0,"tokens":0,"cycles":0,"accepted":0,"drafted":0,"accept_rate":0.0})
        );
        return Ok(());
    }

    let model = model.expect("--model required");
    let draft = draft.expect("--draft required");
    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("gpu: {}", gpu.arch);

    let mut target_hfq = HfqFile::open(&model)?;
    let target_cfg = Lfm2MoeConfig::from_hfq(&target_hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&target_hfq.metadata_json)?;
    let mut tokenized = Vec::new();
    for prompt in &prompts {
        let ids = tokenizer.encode(prompt);
        if !ids.is_empty() {
            tokenized.push((prompt.clone(), ids));
        }
    }
    if tokenized.is_empty() {
        return Err("all prompts tokenized to zero tokens".into());
    }

    let draft_hfq = HfqFile::open(&draft)?;
    let draft_cfg =
        DflashConfig::from_hfq(&draft_hfq).ok_or("draft hfq missing dflash metadata")?;
    validate_dflash_contract(&target_cfg, &draft_cfg)?;
    let block_size = block_override.unwrap_or(draft_cfg.block_size);
    if block_size < 2 {
        return Err("--block must be >= 2".into());
    }

    let max_prompt_tokens = tokenized
        .iter()
        .map(|(_, ids)| ids.len())
        .max()
        .unwrap_or(1);
    let max_seq = max_prompt_tokens
        .saturating_add(max_tokens)
        .saturating_add(block_size)
        .saturating_add(8)
        .min(target_cfg.max_position_embeddings);
    if max_prompt_tokens.saturating_add(block_size) > max_seq {
        return Err(format!(
            "longest prompt ({max_prompt_tokens}) + block ({block_size}) exceeds usable max_seq {max_seq}"
        )
        .into());
    }

    eprintln!(
        "target hidden={} layers={} vocab={} prompts={} max_seq={} draft_layers={} block={} target_layers={:?}",
        target_cfg.hidden_size,
        target_cfg.num_hidden_layers,
        target_cfg.vocab_size,
        tokenized.len(),
        max_seq,
        draft_cfg.n_layers,
        block_size,
        draft_cfg.target_layer_ids,
    );

    let t_load = Instant::now();
    let target_weights = Lfm2MoeWeights::load(&mut target_hfq, &target_cfg, &mut gpu)?;
    let draft_weights = DflashWeights::load_with_f16(
        &mut gpu,
        &draft_hfq,
        &draft_cfg,
        lfm2_dflash_use_f16_weights(),
    )?;
    let mut state = Lfm2MoeState::new_with_max_seq(&mut gpu, &target_cfg, max_seq)?;
    let mut draft_scratch = DflashScratch::new_with_mq_and_sync(
        &mut gpu,
        &draft_cfg,
        block_size,
        max_seq,
        draft_weights.has_mq,
        lfm2_dflash_sync_gemm(),
    )?;
    let mut target_snap = Lfm2DflashTargetSnapshot::new_for(&mut gpu, &state, block_size)?;
    eprintln!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let mut total_generated = 0usize;
    let mut total_cycles = 0usize;
    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;
    let mut total_stopped = 0usize;
    let mut total_prefill_ms = 0.0f64;
    let mut total_decode_ms = 0.0f64;

    for (prompt_idx, (prompt, prompt_ids)) in tokenized.iter().enumerate() {
        if prompt_ids.len().saturating_add(block_size) > max_seq {
            eprintln!(
                "skip prompt {prompt_idx}: tokens={} + block={} exceeds max_seq={}",
                prompt_ids.len(),
                block_size,
                max_seq
            );
            continue;
        }

        state.reset(&mut gpu)?;
        draft_scratch.reset_upload_tracking();
        let mut target_hidden_host = Vec::new();
        let mut capture = Lfm2HiddenCapture::new(
            target_cfg.num_hidden_layers,
            target_cfg.hidden_size,
            draft_cfg.target_layer_ids.clone(),
        )?;

        let t_prefill = Instant::now();
        let logits_per_pos = prefill_batch_with_hidden_logits(
            &target_cfg,
            &target_weights,
            &mut state,
            &mut gpu,
            prompt_ids,
            &mut capture,
        )?;
        gpu.hip.device_synchronize()?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        total_prefill_ms += prefill_ms;
        target_hidden_host.extend_from_slice(&capture.take_rows());
        if state.n_tokens != prompt_ids.len() {
            return Err(format!(
                "prompt {prompt_idx}: prefill ended at {}, expected {}",
                state.n_tokens,
                prompt_ids.len()
            )
            .into());
        }

        let first_token = logits_per_pos
            .chunks_exact(target_cfg.vocab_size)
            .last()
            .map(argmax_u32)
            .ok_or("prefill returned no logits")?;

        let mut emitted = Vec::new();
        let mut generated = 0usize;
        let mut cycles = 0usize;
        let mut accepted = 0usize;
        let mut drafted = 0usize;
        let mut position = prompt_ids.len();
        let mut seed_token = first_token;
        let mut stopped = false;
        let first_token_is_eos = is_terminator(&tokenizer, first_token, eos_extra);
        if !ignore_eos && first_token_is_eos {
            stopped = true;
        } else {
            emitted.push(first_token);
            generated = 1;
        }

        let t_decode = Instant::now();
        while generated < max_tokens && !stopped {
            if position.saturating_add(block_size) > max_seq {
                break;
            }
            let step = spec_step_dflash(
                &mut gpu,
                &target_weights,
                &target_cfg,
                &mut state,
                &draft_weights,
                &draft_cfg,
                &mut draft_scratch,
                &mut target_hidden_host,
                &mut target_snap,
                position,
                seed_token,
                ctx_slice,
                Some(block_size),
            )?;
            cycles += 1;
            accepted += step.accepted;
            drafted += step.drafted.len().saturating_sub(1);

            let mut hit_eos = false;
            for &tok in step.committed.iter().skip(1) {
                if generated >= max_tokens {
                    break;
                }
                if !ignore_eos && is_terminator(&tokenizer, tok, eos_extra) {
                    hit_eos = true;
                    break;
                }
                emitted.push(tok);
                generated += 1;
            }
            position += step.advance;
            seed_token = step.bonus_token;
            if hit_eos {
                stopped = true;
                break;
            }
        }
        gpu.hip.device_synchronize()?;
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        total_decode_ms += decode_ms;
        let accept_rate = if drafted > 0 {
            accepted as f64 / drafted as f64
        } else {
            0.0
        };
        let text_preview = tokenizer.decode(&emitted);
        println!(
            "{}",
            json!({
                "type": "prompt",
                "index": prompt_idx,
                "prompt_tokens": prompt_ids.len(),
                "block": block_size,
                "ctx_slice": ctx_slice,
                "first_token": first_token,
                "first_token_is_eos": first_token_is_eos,
                "ignore_eos": ignore_eos,
                "generated": generated,
                "cycles": cycles,
                "accepted": accepted,
                "drafted": drafted,
                "accept_rate": accept_rate,
                "stopped": stopped,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "text_preview": text_preview.chars().take(160).collect::<String>(),
                "prompt_preview": prompt.chars().take(80).collect::<String>(),
            })
        );

        total_generated += generated;
        total_cycles += cycles;
        total_accepted += accepted;
        total_drafted += drafted;
        total_stopped += usize::from(stopped);
    }

    let accept_rate = if total_drafted > 0 {
        total_accepted as f64 / total_drafted as f64
    } else {
        0.0
    };
    println!(
        "{}",
        json!({
            "type": "summary",
            "prompts": tokenized.len(),
            "block": block_size,
            "ctx_slice": ctx_slice,
            "ignore_eos": ignore_eos,
            "tokens": total_generated,
            "cycles": total_cycles,
            "accepted": total_accepted,
            "drafted": total_drafted,
            "accept_rate": accept_rate,
            "stopped_prompts": total_stopped,
            "prefill_ms": total_prefill_ms,
            "decode_ms": total_decode_ms,
        })
    );

    target_snap.free_gpu(&mut gpu);
    draft_scratch.free_gpu(&mut gpu);
    draft_weights.free_gpu(&mut gpu);
    Ok(())
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
fn is_terminator(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    token: u32,
    eos_extra: Option<u32>,
) -> bool {
    tokenizer.is_terminator(token) || eos_extra == Some(token)
}
