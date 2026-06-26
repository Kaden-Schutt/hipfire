// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
#![recursion_limit = "256"]
//! Fit the LFM2 DFlash final `norm.weight` from block hidden labels.
//!
//! This is a narrow deployed-path trainer slice. It replays saved
//! `lfm2_dflash_teacher_dump` blocks through the actual DFlash runtime forward,
//! downloads the draft rows that feed the target LM head, and fits a diagonal
//! final-norm correction against `dflash_block_target_norm_hidden.f32`.
//!
//! Usage:
//!   lfm2_dflash_fit_norm --model <lfm2.hfq> --draft <in.dflash.hfq>
//!     --teacher-dump <dir> --out <out.dflash.hfq> [--l2 1e-3]
//!     [--skip-blocks N] [--max-blocks N] [--max-scale 4.0]
//!     [--scan-max-scale 1,2,4,8]
//!     [--score-skip-blocks N] [--score-max-blocks N]
//!     [--fit-logit-bias] [--logit-bias-epochs N] [--logit-bias-lr F]
//!     [--logit-bias-max F]
//!     [--scan-logit-bias-epochs 4,8] [--scan-logit-bias-lr 0.25,0.5]
//!     [--scan-logit-bias-max 2,4] [--scan-logit-bias-demote true,false]

use hipfire_arch_lfm2moe::dflash::{
    lfm2_dflash_sync_gemm, lfm2_dflash_use_f16_weights, run_dflash_draft_for_logits,
    validate_dflash_contract,
};
use hipfire_arch_lfm2moe::{Lfm2MoeConfig, Lfm2MoeWeights};
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor, HfqPackage};
use hipfire_runtime::weights::weight_gemm;
use rdna_compute::DType;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "Usage: lfm2_dflash_fit_norm --model <lfm2.hfq> --draft <in.dflash.hfq> --teacher-dump <dir> --out <out.dflash.hfq> [--l2 1e-3] [--skip-blocks N] [--max-blocks N] [--score-skip-blocks N] [--score-max-blocks N] [--max-scale 4.0] [--scan-max-scale 1,2,4,8] [--fit-logit-bias] [--logit-bias-epochs N] [--logit-bias-lr F] [--logit-bias-max F] [--scan-logit-bias-epochs 4,8] [--scan-logit-bias-lr 0.25,0.5] [--scan-logit-bias-max 2,4] [--scan-logit-bias-demote true,false]"
        );
        return Ok(());
    }

    let mut model: Option<PathBuf> = None;
    let mut draft: Option<PathBuf> = None;
    let mut teacher_dump: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut l2 = 1.0e-3f64;
    let mut skip_blocks = 0usize;
    let mut max_blocks: Option<usize> = None;
    let mut score_skip_blocks: Option<usize> = None;
    let mut score_max_blocks: Option<usize> = None;
    let mut max_scale = 4.0f64;
    let mut scan_max_scales: Option<Vec<f64>> = None;
    let mut fit_logit_bias = false;
    let mut logit_bias_epochs = 8usize;
    let mut logit_bias_lr = 1.0f32;
    let mut logit_bias_max = 8.0f32;
    let mut logit_bias_demote = true;
    let mut scan_logit_bias_epochs: Option<Vec<usize>> = None;
    let mut scan_logit_bias_lr: Option<Vec<f32>> = None;
    let mut scan_logit_bias_max: Option<Vec<f32>> = None;
    let mut scan_logit_bias_demote: Option<Vec<bool>> = None;

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
            "--l2" => {
                l2 = argv[i + 1].parse()?;
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
            "--score-skip-blocks" => {
                score_skip_blocks = Some(argv[i + 1].parse()?);
                i += 2;
            }
            "--score-max-blocks" => {
                score_max_blocks = Some(argv[i + 1].parse()?);
                i += 2;
            }
            "--max-scale" => {
                max_scale = argv[i + 1].parse()?;
                i += 2;
            }
            "--scan-max-scale" => {
                scan_max_scales = Some(parse_f64_list(&argv[i + 1])?);
                i += 2;
            }
            "--fit-logit-bias" => {
                fit_logit_bias = true;
                i += 1;
            }
            "--logit-bias-epochs" => {
                logit_bias_epochs = argv[i + 1].parse()?;
                i += 2;
            }
            "--logit-bias-lr" => {
                logit_bias_lr = argv[i + 1].parse()?;
                i += 2;
            }
            "--logit-bias-max" => {
                logit_bias_max = argv[i + 1].parse()?;
                i += 2;
            }
            "--no-logit-bias-demote" => {
                logit_bias_demote = false;
                i += 1;
            }
            "--scan-logit-bias-epochs" => {
                scan_logit_bias_epochs = Some(parse_usize_list(&argv[i + 1])?);
                fit_logit_bias = true;
                i += 2;
            }
            "--scan-logit-bias-lr" => {
                scan_logit_bias_lr = Some(parse_f32_list(&argv[i + 1])?);
                fit_logit_bias = true;
                i += 2;
            }
            "--scan-logit-bias-max" => {
                scan_logit_bias_max = Some(parse_f32_list(&argv[i + 1])?);
                fit_logit_bias = true;
                i += 2;
            }
            "--scan-logit-bias-demote" => {
                scan_logit_bias_demote = Some(parse_bool_list(&argv[i + 1])?);
                fit_logit_bias = true;
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    if l2 < 0.0 || !l2.is_finite() {
        return Err("--l2 must be finite and non-negative".into());
    }
    if max_scale <= 0.0 || !max_scale.is_finite() {
        return Err("--max-scale must be finite and positive".into());
    }
    if let Some(scales) = scan_max_scales.as_ref() {
        if scales.is_empty() {
            return Err("--scan-max-scale must contain at least one value".into());
        }
        for &s in scales {
            if s <= 0.0 || !s.is_finite() {
                return Err("--scan-max-scale values must be finite and positive".into());
            }
        }
    }
    if fit_logit_bias {
        if logit_bias_epochs == 0 {
            return Err("--logit-bias-epochs must be > 0 when --fit-logit-bias is set".into());
        }
        if logit_bias_lr <= 0.0 || !logit_bias_lr.is_finite() {
            return Err("--logit-bias-lr must be finite and positive".into());
        }
        if logit_bias_max <= 0.0 || !logit_bias_max.is_finite() {
            return Err("--logit-bias-max must be finite and positive".into());
        }
    }
    let logit_bias_params = if fit_logit_bias {
        logit_bias_param_grid(
            scan_logit_bias_epochs.as_deref(),
            scan_logit_bias_lr.as_deref(),
            scan_logit_bias_max.as_deref(),
            scan_logit_bias_demote.as_deref(),
            LogitBiasParams {
                epochs: logit_bias_epochs,
                lr: logit_bias_lr,
                max_abs: logit_bias_max,
                demote_pred: logit_bias_demote,
            },
        )?
    } else {
        Vec::new()
    };
    let model = model.ok_or("--model required")?;
    let draft = draft.ok_or("--draft required")?;
    let teacher_dump = teacher_dump.ok_or("--teacher-dump required")?;
    let out = out.ok_or("--out required")?;

    let dump = TeacherDump::load(&teacher_dump)?;
    let train_range = block_range(
        dump.blocks,
        skip_blocks,
        max_blocks,
        "--skip-blocks/--max-blocks",
    )?;
    let score_range = if score_skip_blocks.is_some() || score_max_blocks.is_some() {
        block_range(
            dump.blocks,
            score_skip_blocks.unwrap_or(skip_blocks),
            score_max_blocks,
            "--score-skip-blocks/--score-max-blocks",
        )?
    } else {
        train_range
    };

    let pkg = HfqPackage::open(&draft)?;
    let norm_entry = pkg.entry("norm.weight").ok_or("draft lacks norm.weight")?;
    if norm_entry.shape != vec![dump.hidden as u32] {
        return Err(format!(
            "norm.weight shape {:?} != expected [{}]",
            norm_entry.shape, dump.hidden
        )
        .into());
    }
    let old_norm = read_tensor_as_f32(
        norm_entry.quant_type,
        pkg.blob_data("norm.weight")
            .ok_or("draft norm.weight blob missing")?,
    )?;
    if old_norm.len() != dump.hidden {
        return Err(format!(
            "norm.weight values {} != hidden {}",
            old_norm.len(),
            dump.hidden
        )
        .into());
    }

    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("gpu: {}", gpu.arch);

    let mut target_hfq = HfqFile::open(&model)?;
    let target_cfg = Lfm2MoeConfig::from_hfq(&target_hfq)?;
    let draft_hfq = HfqFile::open(&draft)?;
    let draft_cfg =
        DflashConfig::from_hfq(&draft_hfq).ok_or("draft hfq missing dflash metadata")?;
    validate_dflash_contract(&target_cfg, &draft_cfg)?;
    dump.validate_against(&draft_cfg)?;

    let max_ctx = max_ctx_for_ranges(&dump, &[train_range, score_range]);
    eprintln!(
        "fit norm.weight: train_start={} train_blocks={} train_rows={} score_start={} score_blocks={} score_rows={} hidden={} block_size={} max_ctx={} l2={} max_scale={} scan={:?}",
        train_range.start,
        train_range.blocks,
        train_range.rows(&dump),
        score_range.start,
        score_range.blocks,
        score_range.rows(&dump),
        dump.hidden,
        dump.block_size,
        max_ctx,
        l2,
        max_scale,
        scan_max_scales,
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

    let train_draft_rows = collect_draft_rows(
        &mut gpu,
        &target_weights,
        &target_cfg,
        &draft_weights,
        &draft_cfg,
        &mut draft_scratch,
        &dump,
        train_range,
    )?;
    let train_target_rows = target_norm_rows_for_range(&dump, train_range);
    let train_before_mse = rows_mse(&train_draft_rows, &train_target_rows, dump.hidden);
    let train_before_cos = rows_cosine(&train_draft_rows, &train_target_rows, dump.hidden);
    let (score_draft_rows, score_target_rows) = if score_range == train_range {
        (train_draft_rows.clone(), train_target_rows.clone())
    } else {
        (
            collect_draft_rows(
                &mut gpu,
                &target_weights,
                &target_cfg,
                &draft_weights,
                &draft_cfg,
                &mut draft_scratch,
                &dump,
                score_range,
            )?,
            target_norm_rows_for_range(&dump, score_range),
        )
    };
    let score_before_mse = rows_mse(&score_draft_rows, &score_target_rows, dump.hidden);
    let score_before_cos = rows_cosine(&score_draft_rows, &score_target_rows, dump.hidden);

    let candidate_scales = scan_max_scales.unwrap_or_else(|| vec![max_scale]);
    let mut candidates = Vec::with_capacity(candidate_scales.len());
    for candidate_max_scale in candidate_scales {
        let scale = fit_diagonal_scale(
            &train_draft_rows,
            &train_target_rows,
            dump.hidden,
            l2,
            candidate_max_scale,
        );
        let train_corrected_rows = apply_scale(&train_draft_rows, &scale, dump.hidden);
        let train_hidden_mse = rows_mse(&train_corrected_rows, &train_target_rows, dump.hidden);
        let train_hidden_cosine =
            rows_cosine(&train_corrected_rows, &train_target_rows, dump.hidden);
        let train_logit_metrics = score_corrected_logits(
            &mut gpu,
            &target_weights,
            &train_corrected_rows,
            &dump,
            train_range.start,
            train_range.blocks,
            target_cfg.vocab_size,
        )?;
        let (score_hidden_mse, score_hidden_cosine, score_logit_metrics) =
            if score_range == train_range {
                (train_hidden_mse, train_hidden_cosine, train_logit_metrics)
            } else {
                let score_corrected_rows = apply_scale(&score_draft_rows, &scale, dump.hidden);
                (
                    rows_mse(&score_corrected_rows, &score_target_rows, dump.hidden),
                    rows_cosine(&score_corrected_rows, &score_target_rows, dump.hidden),
                    score_corrected_logits(
                        &mut gpu,
                        &target_weights,
                        &score_corrected_rows,
                        &dump,
                        score_range.start,
                        score_range.blocks,
                        target_cfg.vocab_size,
                    )?,
                )
            };
        let (scale_min, scale_max_stat, scale_mean) = scale_stats(&scale);
        eprintln!(
            "norm candidate max_scale={candidate_max_scale}: train_cos={train_hidden_cosine:.6e} train_ce={:.6e} train_topk={}/{} score_cos={score_hidden_cosine:.6e} score_ce={:.6e} score_topk={}/{} score_weighted_topk={:.6e} scale min/mean/max {:.4}/{:.4}/{:.4}",
            train_logit_metrics.weighted_ce,
            train_logit_metrics.topk_hits,
            train_logit_metrics.slots,
            score_logit_metrics.weighted_ce,
            score_logit_metrics.topk_hits,
            score_logit_metrics.slots,
            score_logit_metrics.weighted_topk_rate,
            scale_min,
            scale_mean,
            scale_max_stat,
        );
        candidates.push(NormCandidate {
            max_scale: candidate_max_scale,
            scale,
            train_hidden_mse,
            train_hidden_cosine,
            score_hidden_mse,
            score_hidden_cosine,
            scale_min,
            scale_mean,
            scale_max: scale_max_stat,
            train_logit_metrics,
            score_logit_metrics,
        });
    }
    let best_idx = select_best_candidate(&candidates);
    let selected = &candidates[best_idx];
    let new_norm: Vec<f32> = old_norm
        .iter()
        .zip(&selected.scale)
        .map(|(&w, &s)| (w as f64 * s) as f32)
        .collect();
    let mut logit_bias_fit: Option<LogitBiasFitSummary> = None;
    let mut new_logit_bias: Option<Vec<f32>> = None;

    if fit_logit_bias {
        let train_corrected_rows = apply_scale(&train_draft_rows, &selected.scale, dump.hidden);
        let score_corrected_rows = if score_range == train_range {
            train_corrected_rows.clone()
        } else {
            apply_scale(&score_draft_rows, &selected.scale, dump.hidden)
        };
        let train_logits = logits_for_corrected_rows(
            &mut gpu,
            &target_weights,
            &train_corrected_rows,
            train_range.rows(&dump),
            target_cfg.vocab_size,
        )?;
        let score_logits = if score_range == train_range {
            train_logits.clone()
        } else {
            logits_for_corrected_rows(
                &mut gpu,
                &target_weights,
                &score_corrected_rows,
                score_range.rows(&dump),
                target_cfg.vocab_size,
            )?
        };
        let initial_bias = existing_logit_bias(&pkg, target_cfg.vocab_size)?;
        let train_before_bias = score_vocab_logits_with_bias(
            &train_logits,
            Some(&initial_bias),
            &dump,
            train_range,
            target_cfg.vocab_size,
        )?;
        let score_before_bias = score_vocab_logits_with_bias(
            &score_logits,
            Some(&initial_bias),
            &dump,
            score_range,
            target_cfg.vocab_size,
        )?;
        let mut bias_candidates = Vec::with_capacity(logit_bias_params.len());
        for params in &logit_bias_params {
            let (bias, updates) = fit_logit_bias_perceptron(
                &train_logits,
                &initial_bias,
                &dump,
                train_range,
                target_cfg.vocab_size,
                params.epochs,
                params.lr,
                params.max_abs,
                params.demote_pred,
            )?;
            let train_after_bias = score_vocab_logits_with_bias(
                &train_logits,
                Some(&bias),
                &dump,
                train_range,
                target_cfg.vocab_size,
            )?;
            let score_after_bias = score_vocab_logits_with_bias(
                &score_logits,
                Some(&bias),
                &dump,
                score_range,
                target_cfg.vocab_size,
            )?;
            let (bias_min, bias_max, bias_mean) = f32_stats(&bias);
            let nonzero = bias.iter().filter(|&&v| v != 0.0).count();
            eprintln!(
                "logit-bias candidate: epochs={} lr={} max={} demote={} updates={} nonzero={} bias min/mean/max {:.4}/{:.4}/{:.4} train_topk {}/{} -> {}/{} score_topk {}/{} -> {}/{} score_argmax {}/{} -> {}/{} score_ce {:.6e} -> {:.6e}",
                params.epochs,
                params.lr,
                params.max_abs,
                params.demote_pred,
                updates,
                nonzero,
                bias_min,
                bias_mean,
                bias_max,
                train_before_bias.topk_hits,
                train_before_bias.slots,
                train_after_bias.topk_hits,
                train_after_bias.slots,
                score_before_bias.topk_hits,
                score_before_bias.slots,
                score_after_bias.topk_hits,
                score_after_bias.slots,
                score_before_bias.argmax_hits,
                score_before_bias.slots,
                score_after_bias.argmax_hits,
                score_after_bias.slots,
                score_before_bias.weighted_ce,
                score_after_bias.weighted_ce,
            );
            bias_candidates.push(LogitBiasCandidate {
                params: *params,
                updates,
                nonzero,
                bias_min,
                bias_mean,
                bias_max,
                train_after: train_after_bias,
                score_after: score_after_bias,
                bias,
            });
        }
        let selected_bias_idx = select_best_logit_bias_candidate(&bias_candidates);
        let selected_bias = &bias_candidates[selected_bias_idx];
        eprintln!(
            "selected logit-bias candidate {}/{}: epochs={} lr={} max={} demote={} score_argmax={}/{} score_topk={}/{} score_ce={:.6e}",
            selected_bias_idx + 1,
            bias_candidates.len(),
            selected_bias.params.epochs,
            selected_bias.params.lr,
            selected_bias.params.max_abs,
            selected_bias.params.demote_pred,
            selected_bias.score_after.argmax_hits,
            selected_bias.score_after.slots,
            selected_bias.score_after.topk_hits,
            selected_bias.score_after.slots,
            selected_bias.score_after.weighted_ce,
        );
        let candidate_summaries = bias_candidates
            .iter()
            .map(logit_bias_candidate_summary)
            .collect::<Vec<_>>();
        logit_bias_fit = Some(LogitBiasFitSummary {
            epochs: selected_bias.params.epochs,
            lr: selected_bias.params.lr,
            max_abs: selected_bias.params.max_abs,
            demote_pred: selected_bias.params.demote_pred,
            updates: selected_bias.updates,
            nonzero: selected_bias.nonzero,
            bias_min: selected_bias.bias_min,
            bias_mean: selected_bias.bias_mean,
            bias_max: selected_bias.bias_max,
            selected_index: selected_bias_idx,
            candidate_count: bias_candidates.len(),
            selection: "score max weighted_argmax_rate, then score max weighted_topk_rate, then score min weighted_ce",
            train_before: train_before_bias,
            train_after: selected_bias.train_after,
            score_before: score_before_bias,
            score_after: selected_bias.score_after,
            candidates: candidate_summaries,
        });
        new_logit_bias = Some(selected_bias.bias.clone());
    }

    eprintln!(
        "selected norm candidate max_scale={}: train_mse {:.6e} -> {:.6e}, train_cos {:.6e} -> {:.6e}, score_mse {:.6e} -> {:.6e}, score_cos {:.6e} -> {:.6e}, score_weighted_ce={:.6e}, score_topk={}/{}, scale min/mean/max {:.4}/{:.4}/{:.4}",
        selected.max_scale,
        train_before_mse,
        selected.train_hidden_mse,
        train_before_cos,
        selected.train_hidden_cosine,
        score_before_mse,
        selected.score_hidden_mse,
        score_before_cos,
        selected.score_hidden_cosine,
        selected.score_logit_metrics.weighted_ce,
        selected.score_logit_metrics.topk_hits,
        selected.score_logit_metrics.slots,
        selected.scale_min,
        selected.scale_mean,
        selected.scale_max,
    );

    let candidate_json = candidates
        .iter()
        .map(norm_candidate_json)
        .collect::<Vec<_>>();
    let mut metadata: serde_json::Value = serde_json::from_str(&pkg.metadata_json)?;
    metadata["dflash_norm_fit"] = json!({
        "producer": "lfm2_dflash_fit_norm",
        "teacher_dump": teacher_dump,
        "skip_blocks": train_range.start,
        "blocks": train_range.blocks,
        "rows": train_range.rows(&dump),
        "train_skip_blocks": train_range.start,
        "train_blocks": train_range.blocks,
        "train_rows": train_range.rows(&dump),
        "score_skip_blocks": score_range.start,
        "score_blocks": score_range.blocks,
        "score_rows": score_range.rows(&dump),
        "hidden": dump.hidden,
        "l2": l2,
        "max_scale": selected.max_scale,
        "selection": "score max weighted_topk_rate, then score min weighted_ce, then score max hidden_cosine",
        "candidates": candidate_json,
        "train_mse_before": train_before_mse,
        "train_mse_after": selected.train_hidden_mse,
        "train_cosine_before": train_before_cos,
        "train_cosine_after": selected.train_hidden_cosine,
        "train_weighted_ce": selected.train_logit_metrics.weighted_ce,
        "train_topk_hits": selected.train_logit_metrics.topk_hits,
        "train_slots": selected.train_logit_metrics.slots,
        "train_weighted_topk_rate": selected.train_logit_metrics.weighted_topk_rate,
        "train_argmax_hits": selected.train_logit_metrics.argmax_hits,
        "train_weighted_argmax_rate": selected.train_logit_metrics.weighted_argmax_rate,
        "score_mse_before": score_before_mse,
        "score_mse_after": selected.score_hidden_mse,
        "score_cosine_before": score_before_cos,
        "score_cosine_after": selected.score_hidden_cosine,
        "score_weighted_ce": selected.score_logit_metrics.weighted_ce,
        "score_topk_hits": selected.score_logit_metrics.topk_hits,
        "score_slots": selected.score_logit_metrics.slots,
        "score_weighted_topk_rate": selected.score_logit_metrics.weighted_topk_rate,
        "score_argmax_hits": selected.score_logit_metrics.argmax_hits,
        "score_weighted_argmax_rate": selected.score_logit_metrics.weighted_argmax_rate,
        "weighted_ce": selected.score_logit_metrics.weighted_ce,
        "topk_hits": selected.score_logit_metrics.topk_hits,
        "slots": selected.score_logit_metrics.slots,
        "weighted_topk_rate": selected.score_logit_metrics.weighted_topk_rate,
        "argmax_hits": selected.score_logit_metrics.argmax_hits,
        "weighted_argmax_rate": selected.score_logit_metrics.weighted_argmax_rate,
        "scale_min": selected.scale_min,
        "scale_mean": selected.scale_mean,
        "scale_max": selected.scale_max,
        "norm_quant_type": "F32",
        "fit_logit_bias": fit_logit_bias
    });
    if let Some(summary) = logit_bias_fit.as_ref() {
        metadata["dflash_logit_bias_fit"] = logit_bias_fit_json(summary);
    }
    let metadata_json = serde_json::to_string(&metadata)?;

    let norm_bytes = f32_slice_to_f32_bytes(&new_norm);
    let logit_bias_bytes = new_logit_bias
        .as_ref()
        .map(|bias| f32_slice_to_f32_bytes(bias));
    let mut tensors =
        Vec::with_capacity(pkg.entries().len() + usize::from(logit_bias_bytes.is_some()));
    let mut wrote_logit_bias = false;
    for entry in pkg.entries() {
        if entry.name == "norm.weight" {
            tensors.push(HfqMemTensor {
                name: entry.name.clone(),
                quant_type: 2,
                shape: entry.shape.clone(),
                group_size: 0,
                data: norm_bytes.clone(),
            });
        } else if entry.name == "logit_bias.weight" && logit_bias_bytes.is_some() {
            wrote_logit_bias = true;
            tensors.push(HfqMemTensor {
                name: entry.name.clone(),
                quant_type: 2,
                shape: vec![target_cfg.vocab_size as u32],
                group_size: 0,
                data: logit_bias_bytes.as_ref().expect("checked").clone(),
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
    if let Some(bytes) = logit_bias_bytes.as_ref() {
        if !wrote_logit_bias {
            tensors.push(HfqMemTensor {
                name: "logit_bias.weight".to_string(),
                quant_type: 2,
                shape: vec![target_cfg.vocab_size as u32],
                group_size: 0,
                data: bytes.clone(),
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
    target_block_norm_hidden: Vec<f32>,
    target_topk_ids: Vec<u32>,
    target_topk_logits: Vec<f32>,
    target_argmax: Vec<u32>,
    topk: usize,
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
        let topk = value_usize(block_meta, "topk")?;
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
        let target_block_norm_hidden_path = path.join("dflash_block_target_norm_hidden.f32");
        let target_block_norm_hidden = read_f32_raw(&target_block_norm_hidden_path)?;
        let expected = blocks * block_size.saturating_sub(1) * hidden;
        if target_block_norm_hidden.len() != expected {
            return Err(format!(
                "{} floats {} != blocks({blocks}) * (block_size({block_size}) - 1) * hidden({hidden})",
                target_block_norm_hidden_path.display(),
                target_block_norm_hidden.len()
            )
            .into());
        }
        let block_rows = blocks * block_size;
        let target_topk_ids = read_u32_raw(&path.join("dflash_block_topk_ids.u32"))?;
        let target_topk_logits = read_f32_raw(&path.join("dflash_block_topk_logits.f32"))?;
        let target_argmax = read_u32_raw(&path.join("dflash_block_target_argmax.u32"))?;
        if target_topk_ids.len() != block_rows * topk
            || target_topk_logits.len() != block_rows * topk
            || target_argmax.len() != block_rows
        {
            return Err("dflash block topk/argmax shape mismatch".into());
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
            target_block_norm_hidden,
            target_topk_ids,
            target_topk_logits,
            target_argmax,
            topk,
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct BlockRange {
    start: usize,
    blocks: usize,
}

impl BlockRange {
    fn end(self) -> usize {
        self.start + self.blocks
    }

    fn rows(self, dump: &TeacherDump) -> usize {
        self.blocks * dump.block_size.saturating_sub(1)
    }
}

struct NormCandidate {
    max_scale: f64,
    scale: Vec<f64>,
    train_hidden_mse: f64,
    train_hidden_cosine: f64,
    score_hidden_mse: f64,
    score_hidden_cosine: f64,
    scale_min: f64,
    scale_mean: f64,
    scale_max: f64,
    train_logit_metrics: LogitMetrics,
    score_logit_metrics: LogitMetrics,
}

#[derive(Clone, Copy)]
struct LogitMetrics {
    slots: usize,
    argmax_hits: usize,
    topk_hits: usize,
    weighted_argmax_rate: f64,
    weighted_topk_rate: f64,
    weighted_ce: f64,
}

#[derive(Clone, Copy)]
struct LogitBiasParams {
    epochs: usize,
    lr: f32,
    max_abs: f32,
    demote_pred: bool,
}

struct LogitBiasCandidate {
    params: LogitBiasParams,
    updates: usize,
    nonzero: usize,
    bias_min: f32,
    bias_mean: f32,
    bias_max: f32,
    train_after: LogitMetrics,
    score_after: LogitMetrics,
    bias: Vec<f32>,
}

#[derive(Clone)]
struct LogitBiasCandidateSummary {
    params: LogitBiasParams,
    updates: usize,
    nonzero: usize,
    bias_min: f32,
    bias_mean: f32,
    bias_max: f32,
    train_after: LogitMetrics,
    score_after: LogitMetrics,
}

struct LogitBiasFitSummary {
    epochs: usize,
    lr: f32,
    max_abs: f32,
    demote_pred: bool,
    updates: usize,
    nonzero: usize,
    bias_min: f32,
    bias_mean: f32,
    bias_max: f32,
    selected_index: usize,
    candidate_count: usize,
    selection: &'static str,
    train_before: LogitMetrics,
    train_after: LogitMetrics,
    score_before: LogitMetrics,
    score_after: LogitMetrics,
    candidates: Vec<LogitBiasCandidateSummary>,
}

fn norm_candidate_json(c: &NormCandidate) -> serde_json::Value {
    json!({
        "max_scale": c.max_scale,
        "train_mse_after": c.train_hidden_mse,
        "train_cosine_after": c.train_hidden_cosine,
        "train_weighted_ce": c.train_logit_metrics.weighted_ce,
        "train_topk_hits": c.train_logit_metrics.topk_hits,
        "train_slots": c.train_logit_metrics.slots,
        "train_weighted_topk_rate": c.train_logit_metrics.weighted_topk_rate,
        "train_argmax_hits": c.train_logit_metrics.argmax_hits,
        "train_weighted_argmax_rate": c.train_logit_metrics.weighted_argmax_rate,
        "score_mse_after": c.score_hidden_mse,
        "score_cosine_after": c.score_hidden_cosine,
        "score_weighted_ce": c.score_logit_metrics.weighted_ce,
        "score_topk_hits": c.score_logit_metrics.topk_hits,
        "score_slots": c.score_logit_metrics.slots,
        "score_weighted_topk_rate": c.score_logit_metrics.weighted_topk_rate,
        "score_argmax_hits": c.score_logit_metrics.argmax_hits,
        "score_weighted_argmax_rate": c.score_logit_metrics.weighted_argmax_rate,
        "weighted_ce": c.score_logit_metrics.weighted_ce,
        "topk_hits": c.score_logit_metrics.topk_hits,
        "slots": c.score_logit_metrics.slots,
        "weighted_topk_rate": c.score_logit_metrics.weighted_topk_rate,
        "argmax_hits": c.score_logit_metrics.argmax_hits,
        "weighted_argmax_rate": c.score_logit_metrics.weighted_argmax_rate,
        "scale_min": c.scale_min,
        "scale_mean": c.scale_mean,
        "scale_max": c.scale_max,
    })
}

fn logit_metrics_json(m: LogitMetrics) -> serde_json::Value {
    json!({
        "slots": m.slots,
        "argmax_hits": m.argmax_hits,
        "topk_hits": m.topk_hits,
        "weighted_argmax_rate": m.weighted_argmax_rate,
        "weighted_topk_rate": m.weighted_topk_rate,
        "weighted_ce": m.weighted_ce,
    })
}

fn logit_bias_fit_json(summary: &LogitBiasFitSummary) -> serde_json::Value {
    json!({
        "producer": "lfm2_dflash_fit_norm",
        "tensor": "logit_bias.weight",
        "epochs": summary.epochs,
        "lr": summary.lr,
        "max_abs": summary.max_abs,
        "demote_pred": summary.demote_pred,
        "updates": summary.updates,
        "nonzero": summary.nonzero,
        "bias_min": summary.bias_min,
        "bias_mean": summary.bias_mean,
        "bias_max": summary.bias_max,
        "selected_index": summary.selected_index,
        "candidate_count": summary.candidate_count,
        "selection": summary.selection,
        "train_before": logit_metrics_json(summary.train_before),
        "train_after": logit_metrics_json(summary.train_after),
        "score_before": logit_metrics_json(summary.score_before),
        "score_after": logit_metrics_json(summary.score_after),
        "candidates": summary.candidates.iter().map(logit_bias_candidate_json).collect::<Vec<_>>(),
    })
}

fn logit_bias_candidate_summary(c: &LogitBiasCandidate) -> LogitBiasCandidateSummary {
    LogitBiasCandidateSummary {
        params: c.params,
        updates: c.updates,
        nonzero: c.nonzero,
        bias_min: c.bias_min,
        bias_mean: c.bias_mean,
        bias_max: c.bias_max,
        train_after: c.train_after,
        score_after: c.score_after,
    }
}

fn logit_bias_candidate_json(c: &LogitBiasCandidateSummary) -> serde_json::Value {
    json!({
        "epochs": c.params.epochs,
        "lr": c.params.lr,
        "max_abs": c.params.max_abs,
        "demote_pred": c.params.demote_pred,
        "updates": c.updates,
        "nonzero": c.nonzero,
        "bias_min": c.bias_min,
        "bias_mean": c.bias_mean,
        "bias_max": c.bias_max,
        "train_after": logit_metrics_json(c.train_after),
        "score_after": logit_metrics_json(c.score_after),
    })
}

fn select_best_candidate(candidates: &[NormCandidate]) -> usize {
    let mut best = 0usize;
    for idx in 1..candidates.len() {
        let lhs = &candidates[idx];
        let rhs = &candidates[best];
        let better = lhs
            .score_logit_metrics
            .weighted_topk_rate
            .total_cmp(&rhs.score_logit_metrics.weighted_topk_rate)
            .is_gt()
            || (lhs.score_logit_metrics.weighted_topk_rate
                == rhs.score_logit_metrics.weighted_topk_rate
                && lhs
                    .score_logit_metrics
                    .weighted_ce
                    .total_cmp(&rhs.score_logit_metrics.weighted_ce)
                    .is_lt())
            || (lhs.score_logit_metrics.weighted_topk_rate
                == rhs.score_logit_metrics.weighted_topk_rate
                && lhs.score_logit_metrics.weighted_ce == rhs.score_logit_metrics.weighted_ce
                && lhs
                    .score_hidden_cosine
                    .total_cmp(&rhs.score_hidden_cosine)
                    .is_gt());
        if better {
            best = idx;
        }
    }
    best
}

fn select_best_logit_bias_candidate(candidates: &[LogitBiasCandidate]) -> usize {
    let mut best = 0usize;
    for idx in 1..candidates.len() {
        let lhs = &candidates[idx];
        let rhs = &candidates[best];
        let better = lhs
            .score_after
            .weighted_argmax_rate
            .total_cmp(&rhs.score_after.weighted_argmax_rate)
            .is_gt()
            || (lhs.score_after.weighted_argmax_rate == rhs.score_after.weighted_argmax_rate
                && lhs
                    .score_after
                    .weighted_topk_rate
                    .total_cmp(&rhs.score_after.weighted_topk_rate)
                    .is_gt())
            || (lhs.score_after.weighted_argmax_rate == rhs.score_after.weighted_argmax_rate
                && lhs.score_after.weighted_topk_rate == rhs.score_after.weighted_topk_rate
                && lhs
                    .score_after
                    .weighted_ce
                    .total_cmp(&rhs.score_after.weighted_ce)
                    .is_lt());
        if better {
            best = idx;
        }
    }
    best
}

fn logit_bias_param_grid(
    scan_epochs: Option<&[usize]>,
    scan_lr: Option<&[f32]>,
    scan_max: Option<&[f32]>,
    scan_demote: Option<&[bool]>,
    fallback: LogitBiasParams,
) -> Result<Vec<LogitBiasParams>, Box<dyn std::error::Error>> {
    let epochs = scan_epochs.unwrap_or(std::slice::from_ref(&fallback.epochs));
    let lrs = scan_lr.unwrap_or(std::slice::from_ref(&fallback.lr));
    let maxes = scan_max.unwrap_or(std::slice::from_ref(&fallback.max_abs));
    let demotes = scan_demote.unwrap_or(std::slice::from_ref(&fallback.demote_pred));
    if epochs.is_empty() || lrs.is_empty() || maxes.is_empty() || demotes.is_empty() {
        return Err("logit-bias scan lists must not be empty".into());
    }
    let mut out = Vec::with_capacity(epochs.len() * lrs.len() * maxes.len() * demotes.len());
    for &epoch in epochs {
        if epoch == 0 {
            return Err("logit-bias epochs must be > 0".into());
        }
        for &lr in lrs {
            if lr <= 0.0 || !lr.is_finite() {
                return Err("logit-bias lr values must be finite and positive".into());
            }
            for &max_abs in maxes {
                if max_abs <= 0.0 || !max_abs.is_finite() {
                    return Err("logit-bias max values must be finite and positive".into());
                }
                for &demote_pred in demotes {
                    out.push(LogitBiasParams {
                        epochs: epoch,
                        lr,
                        max_abs,
                        demote_pred,
                    });
                }
            }
        }
    }
    Ok(out)
}

fn block_range(
    total_blocks: usize,
    skip_blocks: usize,
    max_blocks: Option<usize>,
    label: &str,
) -> Result<BlockRange, Box<dyn std::error::Error>> {
    if skip_blocks >= total_blocks {
        return Err(format!(
            "{label}: skip {skip_blocks} leaves no blocks in dump with {total_blocks} blocks"
        )
        .into());
    }
    let blocks = max_blocks
        .unwrap_or(total_blocks - skip_blocks)
        .min(total_blocks - skip_blocks);
    if blocks == 0 {
        return Err(format!("{label}: no teacher blocks selected").into());
    }
    Ok(BlockRange {
        start: skip_blocks,
        blocks,
    })
}

fn max_ctx_for_ranges(dump: &TeacherDump, ranges: &[BlockRange]) -> usize {
    ranges
        .iter()
        .flat_map(|r| r.start..r.end())
        .map(|b| dump.ctx_lens[b] as usize)
        .max()
        .unwrap_or(1)
        .max(1)
}

#[allow(clippy::too_many_arguments)]
fn collect_draft_rows(
    gpu: &mut rdna_compute::Gpu,
    target_weights: &Lfm2MoeWeights,
    target_cfg: &Lfm2MoeConfig,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    dump: &TeacherDump,
    range: BlockRange,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let rows_per_block = dump.block_size.saturating_sub(1);
    let mut draft_rows = Vec::with_capacity(range.rows(dump) * dump.hidden);
    for b in range.start..range.end() {
        let position = dump.positions[b] as usize;
        let ctx = (dump.ctx_lens[b] as usize).min(position).max(1);
        let block_features = dump.features_for_block(b)?;
        draft_scratch.reset_upload_tracking();
        run_dflash_draft_for_logits(
            gpu,
            target_weights,
            target_cfg,
            draft_weights,
            draft_cfg,
            draft_scratch,
            block_features,
            position,
            dump.seed_tokens[b],
            Some(ctx),
            dump.block_size,
            None,
        )?;
        gpu.hip.device_synchronize()?;
        let rows_tensor = draft_scratch
            .x
            .sub_offset(draft_cfg.hidden, rows_per_block * draft_cfg.hidden);
        let rows = gpu.download_f32(&rows_tensor)?;
        draft_rows.extend_from_slice(&rows);
    }
    Ok(draft_rows)
}

fn target_norm_rows_for_range(dump: &TeacherDump, range: BlockRange) -> Vec<f32> {
    let rows_per_block = dump.block_size.saturating_sub(1);
    let row_floats = rows_per_block * dump.hidden;
    let start = range.start * row_floats;
    let end = range.end() * row_floats;
    dump.target_block_norm_hidden[start..end].to_vec()
}

fn apply_scale(draft_rows: &[f32], scale: &[f64], hidden: usize) -> Vec<f32> {
    let mut corrected = draft_rows.to_vec();
    for row in corrected.chunks_exact_mut(hidden) {
        for (v, &s) in row.iter_mut().zip(scale) {
            *v *= s as f32;
        }
    }
    corrected
}

fn score_corrected_logits(
    gpu: &mut rdna_compute::Gpu,
    target_weights: &Lfm2MoeWeights,
    corrected_rows: &[f32],
    dump: &TeacherDump,
    start_block: usize,
    blocks: usize,
    vocab_size: usize,
) -> Result<LogitMetrics, Box<dyn std::error::Error>> {
    let rows = blocks * dump.block_size.saturating_sub(1);
    if corrected_rows.len() != rows * dump.hidden {
        return Err(format!(
            "corrected rows {} != rows({rows}) * hidden({})",
            corrected_rows.len(),
            dump.hidden
        )
        .into());
    }
    let logits = logits_for_corrected_rows(gpu, target_weights, corrected_rows, rows, vocab_size)?;
    score_vocab_logits_with_bias(
        &logits,
        None,
        dump,
        BlockRange {
            start: start_block,
            blocks,
        },
        vocab_size,
    )
}

fn logits_for_corrected_rows(
    gpu: &mut rdna_compute::Gpu,
    target_weights: &Lfm2MoeWeights,
    corrected_rows: &[f32],
    rows: usize,
    vocab_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if rows == 0 {
        return Err("cannot score zero corrected rows".into());
    }
    if !corrected_rows.len().is_multiple_of(rows) {
        return Err(format!(
            "corrected rows {} is not divisible by row count {rows}",
            corrected_rows.len()
        )
        .into());
    }
    let x = gpu.upload_f32(corrected_rows, &[corrected_rows.len()])?;
    let logits_batch = gpu.alloc_tensor(&[rows * vocab_size], DType::F32)?;
    let logits = (|| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        weight_gemm(gpu, &target_weights.lm_head, &x, &logits_batch, rows)
            .map_err(|e| format!("target lm_head for norm candidate: {e:?}"))?;
        gpu.download_f32(&logits_batch)
            .map_err(|e| format!("download norm candidate logits: {e:?}").into())
    })();
    let _ = gpu.free_tensor(logits_batch);
    let _ = gpu.free_tensor(x);
    logits
}

fn score_vocab_logits_with_bias(
    logits: &[f32],
    bias: Option<&[f32]>,
    dump: &TeacherDump,
    range: BlockRange,
    vocab_size: usize,
) -> Result<LogitMetrics, Box<dyn std::error::Error>> {
    let rows = range.rows(dump);
    if logits.len() != rows * vocab_size {
        return Err(format!(
            "logits {} != rows({rows}) * vocab({vocab_size})",
            logits.len()
        )
        .into());
    }
    if let Some(bias) = bias {
        if bias.len() != vocab_size {
            return Err(format!("logit bias length {} != vocab {vocab_size}", bias.len()).into());
        }
    }
    let weights = block_position_weights(dump.block_size, 3.0);
    let mut slots = 0usize;
    let mut argmax_hits = 0usize;
    let mut topk_hits = 0usize;
    let mut weighted_argmax_hits = 0.0f64;
    let mut weighted_topk_hits = 0.0f64;
    let mut weighted_ce = 0.0f64;
    let mut total_weight = 0.0f64;
    for local_b in 0..range.blocks {
        let b = range.start + local_b;
        for slot in 1..dump.block_size {
            let row = local_b * (dump.block_size - 1) + (slot - 1);
            let flat = b * dump.block_size + slot;
            let logits_row = &logits[row * vocab_size..(row + 1) * vocab_size];
            let draft_tok = argmax_u32_with_bias(logits_row, bias);
            let target_argmax = dump.target_argmax[flat];
            let topk_ids = &dump.target_topk_ids[flat * dump.topk..(flat + 1) * dump.topk];
            let topk_logits = &dump.target_topk_logits[flat * dump.topk..(flat + 1) * dump.topk];
            let ce =
                sampled_ce_from_vocab_logits_with_bias(logits_row, bias, topk_ids, topk_logits)?;
            let w = weights[slot - 1] as f64;
            let argmax_hit = draft_tok == target_argmax;
            let topk_hit = topk_ids.contains(&draft_tok);
            slots += 1;
            argmax_hits += usize::from(argmax_hit);
            topk_hits += usize::from(topk_hit);
            total_weight += w;
            weighted_ce += ce * w;
            if argmax_hit {
                weighted_argmax_hits += w;
            }
            if topk_hit {
                weighted_topk_hits += w;
            }
        }
    }
    let denom = total_weight.max(f64::MIN_POSITIVE);
    Ok(LogitMetrics {
        slots,
        argmax_hits,
        topk_hits,
        weighted_argmax_rate: weighted_argmax_hits / denom,
        weighted_topk_rate: weighted_topk_hits / denom,
        weighted_ce: weighted_ce / denom,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_logit_bias_perceptron(
    logits: &[f32],
    initial_bias: &[f32],
    dump: &TeacherDump,
    range: BlockRange,
    vocab_size: usize,
    epochs: usize,
    lr: f32,
    max_abs: f32,
    demote_pred: bool,
) -> Result<(Vec<f32>, usize), Box<dyn std::error::Error>> {
    let rows = range.rows(dump);
    if logits.len() != rows * vocab_size {
        return Err(format!(
            "logit-bias fit logits {} != rows({rows}) * vocab({vocab_size})",
            logits.len()
        )
        .into());
    }
    if initial_bias.len() != vocab_size {
        return Err(format!(
            "initial logit bias length {} != vocab {vocab_size}",
            initial_bias.len()
        )
        .into());
    }
    let weights = block_position_weights(dump.block_size, 3.0);
    let mut bias = initial_bias.to_vec();
    let mut updates = 0usize;
    for _ in 0..epochs {
        for local_b in 0..range.blocks {
            let b = range.start + local_b;
            for slot in 1..dump.block_size {
                let row = local_b * (dump.block_size - 1) + (slot - 1);
                let flat = b * dump.block_size + slot;
                let logits_row = &logits[row * vocab_size..(row + 1) * vocab_size];
                let pred = argmax_u32_with_bias(logits_row, Some(&bias)) as usize;
                let target = dump.target_argmax[flat] as usize;
                if target >= vocab_size {
                    return Err(format!(
                        "target argmax {target} outside vocab {vocab_size} at block {b} slot {slot}"
                    )
                    .into());
                }
                if pred == target {
                    continue;
                }
                let step = lr * weights[slot - 1];
                bias[target] = (bias[target] + step).clamp(-max_abs, max_abs);
                if demote_pred && pred < vocab_size {
                    bias[pred] = (bias[pred] - step).clamp(-max_abs, max_abs);
                }
                updates += 1;
            }
        }
    }
    Ok((bias, updates))
}

fn fit_diagonal_scale(
    draft: &[f32],
    target: &[f32],
    hidden: usize,
    l2: f64,
    max_scale: f64,
) -> Vec<f64> {
    let rows = draft.len() / hidden;
    let mut num = vec![0.0f64; hidden];
    let mut den = vec![0.0f64; hidden];
    for row in 0..rows {
        let off = row * hidden;
        for h in 0..hidden {
            let x = draft[off + h] as f64;
            let y = target[off + h] as f64;
            num[h] += x * y;
            den[h] += x * x;
        }
    }
    num.into_iter()
        .zip(den)
        .map(|(n, d)| {
            let raw = n / (d + l2);
            raw.clamp(-max_scale, max_scale)
        })
        .collect()
}

fn rows_mse(a: &[f32], b: &[f32], hidden: usize) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let rows = a.len() / hidden;
    let mut sum = 0.0f64;
    for row in 0..rows {
        let off = row * hidden;
        let mut acc = 0.0f64;
        for h in 0..hidden {
            let d = a[off + h] as f64 - b[off + h] as f64;
            acc += d * d;
        }
        sum += acc / hidden.max(1) as f64;
    }
    sum / rows.max(1) as f64
}

fn rows_cosine(a: &[f32], b: &[f32], hidden: usize) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let rows = a.len() / hidden;
    let mut sum = 0.0f64;
    for row in 0..rows {
        let off = row * hidden;
        let mut dot = 0.0f64;
        let mut aa = 0.0f64;
        let mut bb = 0.0f64;
        for h in 0..hidden {
            let x = a[off + h] as f64;
            let y = b[off + h] as f64;
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        let denom = aa.sqrt() * bb.sqrt();
        if denom > 0.0 && denom.is_finite() {
            sum += dot / denom;
        }
    }
    sum / rows.max(1) as f64
}

fn scale_stats(scale: &[f64]) -> (f64, f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in scale {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        sum += v;
    }
    (min_v, max_v, sum / scale.len().max(1) as f64)
}

fn f32_stats(values: &[f32]) -> (f32, f32, f32) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for &v in values {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        sum += v;
    }
    (min_v, max_v, sum / values.len().max(1) as f32)
}

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

fn sampled_ce_from_vocab_logits_with_bias(
    vocab_logits: &[f32],
    bias: Option<&[f32]>,
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
        let b = bias.map_or(0.0, |bias| bias[idx]);
        pred_logits.push(vocab_logits[idx] + b);
    }
    let pred = stable_softmax(&pred_logits);
    let target = stable_softmax(target_logits);
    Ok(target
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| -(*t as f64) * (*p as f64).max(1e-20).ln())
        .sum())
}

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

fn argmax_u32_with_bias(row: &[f32], bias: Option<&[f32]>) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in row.iter().enumerate() {
        let value = value + bias.map_or(0.0, |bias| bias[idx]);
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn existing_logit_bias(
    pkg: &HfqPackage,
    vocab_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let Some(entry) = pkg.entry("logit_bias.weight") else {
        return Ok(vec![0.0; vocab_size]);
    };
    if entry.shape != vec![vocab_size as u32] {
        return Err(format!(
            "logit_bias.weight shape {:?} != expected [{vocab_size}]",
            entry.shape
        )
        .into());
    }
    let data = pkg
        .blob_data("logit_bias.weight")
        .ok_or("draft logit_bias.weight blob missing")?;
    let bias = read_tensor_as_f32(entry.quant_type, data)?;
    if bias.len() != vocab_size {
        return Err(format!(
            "logit_bias.weight values {} != vocab {vocab_size}",
            bias.len()
        )
        .into());
    }
    Ok(bias)
}

fn read_tensor_as_f32(qt: u8, bytes: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match qt {
        1 => {
            if !bytes.len().is_multiple_of(2) {
                return Err("F16 tensor byte length is not divisible by 2".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| hipfire_runtime::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        2 => {
            if !bytes.len().is_multiple_of(4) {
                return Err("F32 tensor byte length is not divisible by 4".into());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        other => Err(format!("unsupported norm.weight quant_type {other}").into()),
    }
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

fn parse_f64_list(s: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    s.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map_err(|e| format!("invalid f64 `{part}`: {e}").into())
        })
        .collect()
}

fn parse_f32_list(s: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    s.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid f32 `{part}`: {e}").into())
        })
        .collect()
}

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

fn parse_bool_list(s: &str) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    s.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| match part.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "demote" => Ok(true),
            "0" | "false" | "no" | "nodemote" | "no-demote" => Ok(false),
            other => {
                Err(format!("invalid bool `{other}`; use true,false or demote,nodemote").into())
            }
        })
        .collect()
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

fn read_u32_raw(path: &std::path::Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(std::mem::size_of::<u32>()) {
        return Err(format!("{} byte length is not divisible by 4", path.display()).into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
