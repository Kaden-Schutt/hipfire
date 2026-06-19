// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `examples` and `direct` eval executors: battery rows produced by spawning
//! Hipfire example/runner binaries (and a few in-process direct paths).
//!
//! Covers coherence / profile / longctx / calibrate / perplexity / qwen35-speed
//! / agentic / runtime / dflash-matrix / pflash example runs, the per-suite item
//! runners (lm-eval-micro / barrage / humaneval / gpqa), the run-anchor helpers,
//! and the direct session-reset-recall path. Extracted verbatim from the former
//! `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::*;

pub(crate) fn examples_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    match battery {
        BatteryId::Smoke => Some(vec![
            run_examples_run_anchor_with_prompt(
                BatteryId::Smoke,
                "finite_greedy_decode",
                "benchmarks/prompts/qwen2_smoke.txt",
                config,
                ctx,
            ),
            run_direct_session_reset_recall(config, ctx),
        ]),
        BatteryId::Coherence => Some(run_examples_coherence_rows(config, ctx)),
        BatteryId::Speed => Some(run_examples_qwen35_speed_rows(config, ctx)),
        BatteryId::PromptShape => Some(vec![run_examples_run_anchor_with_prompt(
            BatteryId::PromptShape,
            "whitespace_template_canary",
            "benchmarks/prompts/lru_cache_pep8_strict.txt",
            config,
            ctx,
        )]),
        BatteryId::Structured => Some(vec![run_examples_run_anchor_with_prompt(
            BatteryId::Structured,
            "tool_call_jsonish_canary",
            "benchmarks/prompts/tool_call_read_file.txt",
            config,
            ctx,
        )]),
        BatteryId::Dflash => {
            let mut rows: Vec<_> = evaluation_models(config)
                .into_iter()
                .map(|model| {
                    run_dflash_spec_demo_anchor(
                        BatteryId::Dflash,
                        "ar_anchor",
                        true,
                        "benchmarks/prompts/dflash_resident_smoke.txt",
                        &[],
                        BTreeMap::new(),
                        config,
                        ctx,
                        model,
                    )
                })
                .collect();
            if matches!(config.dflash, DflashMode::Off) {
                rows.push(skip_row(
                    BatteryId::Dflash,
                    None,
                    "dflash_anchor",
                    None,
                    "DFlash disabled by --dflash off",
                    config,
                    ctx,
                    prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
                ));
            } else {
                rows.push(run_dflash_spec_demo_anchor(
                    BatteryId::Dflash,
                    "dflash_anchor",
                    false,
                    "benchmarks/prompts/dflash_resident_smoke.txt",
                    &[],
                    BTreeMap::new(),
                    config,
                    ctx,
                    config.model.clone(),
                ));
            }
            rows.extend(run_examples_dflash_matrix_rows(config, ctx));
            Some(rows)
        }
        BatteryId::Pflash => Some(run_examples_pflash_rows(config, ctx)),
        BatteryId::Agentic => Some(run_examples_agentic_rows(config, ctx)),
        BatteryId::Runtime => Some(run_examples_runtime_rows(config, ctx)),
        BatteryId::Barrage => Some(examples_barrage_rows(config, ctx, datasets)),
        BatteryId::Longctx => Some(vec![run_examples_longctx_anchor(config, ctx)]),
        BatteryId::Profile => Some(
            evaluation_models(config)
                .into_iter()
                .map(|model| run_examples_profile_anchor(config, ctx, model))
                .collect(),
        ),
        BatteryId::Calibrate => Some(run_examples_calibrate_rows(config, ctx)),
        BatteryId::Perplexity => Some(run_examples_perplexity_rows(config, ctx)),
        _ => None,
    }
}

pub(crate) fn direct_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    _datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    match battery {
        BatteryId::Smoke => Some(vec![
            run_examples_run_anchor_with_prompt(
                BatteryId::Smoke,
                "finite_greedy_decode",
                "benchmarks/prompts/qwen2_smoke.txt",
                config,
                ctx,
            ),
            run_direct_session_reset_recall(config, ctx),
        ]),
        _ => examples_battery_rows(battery, config, ctx, _datasets),
    }
}

pub(crate) fn examples_executor_available_for(battery: BatteryId) -> bool {
    match battery {
        BatteryId::Coherence => hipfire_coherence::daemon_binary_available(),
        BatteryId::Smoke
        | BatteryId::PromptShape
        | BatteryId::Structured
        | BatteryId::Barrage
        | BatteryId::Profile => resolve_run_example_bin().is_some(),
        BatteryId::Longctx => hipfire_coherence::daemon_binary_available(),
        BatteryId::Speed => resolve_bench_qwen35_speed_bin().is_some(),
        BatteryId::Dflash => resolve_dflash_spec_demo_bin().is_some(),
        BatteryId::Pflash => resolve_pflash_niah_bench_bin().is_some(),
        BatteryId::Agentic => hipfire_coherence::daemon_binary_available(),
        BatteryId::Runtime => true,
        BatteryId::Calibrate => resolve_collect_artifacts_bin().is_some(),
        BatteryId::Perplexity => resolve_perplexity_bin().is_some(),
        _ => false,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CoherenceCase {
    pub(crate) label: &'static str,
    pub(crate) prompt_path: &'static str,
    pub(crate) system_path: Option<&'static str>,
    pub(crate) max_tokens: usize,
}

pub(crate) fn coherence_cases() -> &'static [CoherenceCase] {
    &[
        CoherenceCase {
            label: "capital_short_answer",
            prompt_path: "benchmarks/prompts/coherence_capital_france.txt",
            system_path: None,
            max_tokens: 80,
        },
        CoherenceCase {
            label: "code_square_function",
            prompt_path: "benchmarks/prompts/coherence_square_function.txt",
            system_path: None,
            max_tokens: 180,
        },
        CoherenceCase {
            label: "reason_sheep",
            prompt_path: "benchmarks/prompts/coherence_sheep_reason.txt",
            system_path: None,
            max_tokens: 300,
        },
        CoherenceCase {
            label: "tool_call_read_file",
            prompt_path: "benchmarks/prompts/tool_call_read_file.txt",
            system_path: Some("benchmarks/prompts/tool_call_system.txt"),
            max_tokens: 180,
        },
        CoherenceCase {
            label: "long_prefill_lloyd",
            prompt_path: "benchmarks/prompts/coherence_lloyd_long.txt",
            system_path: None,
            max_tokens: 220,
        },
    ]
}

pub(crate) fn run_examples_coherence_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Vec<EvalResult> {
    coherence_cases()
        .iter()
        .map(|case| run_examples_coherence_case(config, ctx, *case))
        .collect()
}

pub(crate) fn run_examples_coherence_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: CoherenceCase,
) -> EvalResult {
    let prompt_ref = if let Some(system_path) = case.system_path {
        combined_prompt_ref(system_path, case.prompt_path)
    } else {
        prompt(case.prompt_path)
    };
    let metrics = BTreeMap::from([
        ("suite".to_string(), json!("coherence_gate")),
        ("prompt_kind".to_string(), json!(case.label)),
        (
            "detector_profile".to_string(),
            json!(if case.system_path.is_some() {
                "agentic_toolcall_shape"
            } else {
                "default_runtime_coherence"
            }),
        ),
    ]);
    run_daemon_coherence_anchor(
        BatteryId::Coherence,
        case.label,
        case.prompt_path,
        prompt_ref,
        config,
        ctx,
        config.model.clone(),
        Some((case.max_tokens + 2048).max(4096)),
        Some(case.max_tokens),
        metrics,
        case.system_path,
        None,
        None,
        false,
        None,
    )
}

pub(crate) fn run_examples_profile_anchor(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    run_examples_run_anchor_with_prompt_ref_for_model(
        BatteryId::Profile,
        "model_profile_anchor",
        "benchmarks/prompts/dflash_resident_smoke.txt",
        prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
        config,
        ctx,
        model,
        None,
        BTreeMap::from([
            ("profile_requested".to_string(), json!(true)),
            (
                "collection_scope".to_string(),
                json!("model_backed_run_anchor"),
            ),
            (
                "moe_router_histogram_expected_when_moe".to_string(),
                json!(true),
            ),
            (
                "expected_runtime_evidence_kinds".to_string(),
                json!([
                    "performance",
                    "memory",
                    "launch_counts",
                    "moe_router_histogram"
                ]),
            ),
        ]),
    )
}

pub(crate) fn run_examples_longctx_anchor(config: &EvalConfig, ctx: &EvalContext) -> EvalResult {
    match materialize_longctx_prompt(config) {
        Ok(longctx) => run_daemon_coherence_anchor(
            BatteryId::Longctx,
            "multidoc_needle_long_state",
            &longctx.prompt_path,
            Some(longctx.prompt_ref),
            config,
            ctx,
            config.model.clone(),
            Some(longctx.max_seq),
            None,
            longctx.metrics,
            None,
            None,
            None,
            false,
            Some(hipfire_coherence::DetectorProfile::long_state()),
        ),
        Err(err) => row(
            BatteryId::Longctx,
            None,
            "multidoc_needle_native",
            None,
            EvalStatus::Fail,
            Some(err),
            BTreeMap::from([("fixture".to_string(), json!("longprose_multidoc"))]),
            config,
            ctx,
            prompt("benchmarks/prompts/longprose_multidoc.jsonl"),
            0,
        ),
    }
}

/// Tier-1 calibration battery: run the single-load `collect_artifacts` example
/// over each evaluation model and assert the on-GPU Hessian/imatrix internal
/// consistency (`diag(Σxxᵀ)==Σx²`) plus a non-zero captured-tensor count. This
/// is a mechanism/correctness check (small token budget), not a full-quality
/// Hessian. Capture fires only at the bf16/f16 chokepoints, so non-bf16 models
/// are skipped. Not in any default tier — opt in with `--battery calibrate`.
pub(crate) fn run_examples_calibrate_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Vec<EvalResult> {
    evaluation_models(config)
        .into_iter()
        .map(|model| run_examples_calibrate_model(config, ctx, model))
        .collect()
}

pub(crate) fn run_examples_calibrate_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let label = "single_load_hessian_consistency";
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    let base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("calibration")),
    ]);

    let skip = |reason: &str| -> EvalResult {
        row_for_model(
            BatteryId::Calibrate,
            None,
            label,
            None,
            EvalStatus::Skip,
            Some(reason.to_string()),
            base_metrics.clone(),
            config,
            ctx,
            prompt_ref.clone(),
            0,
            model.clone(),
        )
    };

    // Capture fires at the bf16/f16 gemm chokepoints, so a faithful calibration
    // pass needs a bf16 source model.
    if !model_artifact_stem(&model).contains("bf16") {
        return skip(
            "calibration requires a bf16 source model (capture fires at the bf16 chokepoint)",
        );
    }
    if !Path::new(&model).exists() {
        return skip("collect_artifacts requires --model to be a local filesystem path");
    }
    let Some(bin) = resolve_collect_artifacts_bin() else {
        return skip("collect_artifacts example binary not found; build with `cargo build --release -p hipfire-runtime --example collect_artifacts`");
    };

    let corpus = repo_root()
        .map(|r| r.join("benchmarks/prompts/lru_cache_single_blank.txt"))
        .unwrap_or_else(|| PathBuf::from("benchmarks/prompts/lru_cache_single_blank.txt"));
    let out = std::env::temp_dir().join(format!(
        "hipfire-eval-calib-{}.calib.hfq",
        std::process::id()
    ));
    let max_tokens = config.max_tokens.clamp(16, 64).to_string();
    let args = vec![
        "--model".to_string(),
        model.clone(),
        "--corpus".to_string(),
        corpus.display().to_string(),
        "--output".to_string(),
        out.display().to_string(),
        "--max-tokens".to_string(),
        max_tokens,
    ];
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(o) => o,
        Err(err) => {
            let mut m = base_metrics.clone();
            m.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Calibrate,
                None,
                label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn collect_artifacts: {err}")),
                m,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let _ = std::fs::remove_file(&out);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // collect_artifacts prints to stderr:
    //   "collected N hessian tensors in ...s; max diag(H)-vs-Σx² rel-err = ... [CONSISTENT]"
    let combined = format!("{stdout}{stderr}");
    let consistent = combined.contains("[CONSISTENT]");
    let n_hessian = parse_collected_hessian_count(&combined);

    let mut metrics = base_metrics.clone();
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("consistent".to_string(), json!(consistent));
    if let Some(n) = n_hessian {
        metrics.insert("n_hessian".to_string(), json!(n));
    }
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );

    let ok = output.status.success() && consistent && n_hessian.map_or(false, |n| n > 0);
    let status = if ok {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let reason = if ok {
        None
    } else if !output.status.success() {
        Some(format!("collect_artifacts exited with {}", output.status))
    } else if !consistent {
        Some("Hessian/imatrix consistency check did not report [CONSISTENT]".to_string())
    } else {
        Some("collect_artifacts captured 0 hessian tensors".to_string())
    };

    row_for_model(
        BatteryId::Calibrate,
        None,
        label,
        None,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}

/// Parse the captured-Hessian count from collect_artifacts' summary line
/// ("collected N hessian tensors in ...").
pub(crate) fn parse_collected_hessian_count(s: &str) -> Option<u64> {
    let idx = s.find("collected ")? + "collected ".len();
    let rest = &s[idx..];
    let end = rest.find(' ')?;
    rest[..end].parse::<u64>().ok()
}

/// Perplexity / KLD battery: run the `perplexity` example over a raw corpus and
/// report PPL + NLL/tok (+ KLD/tok when `--kldref` is given). The canonical
/// quant-quality primitive — the place the older bench_quant_quality / megabench
/// / sim_mq3_eval scripts should funnel into instead of invoking `perplexity`
/// raw. Opt-in (`--battery perplexity`); not in any default tier (needs a
/// corpus). Corpus/ctx via HIPFIRE_EVAL_PERPLEXITY_CORPUS / _CTX.
pub(crate) fn run_examples_perplexity_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Vec<EvalResult> {
    evaluation_models(config)
        .into_iter()
        .map(|model| run_examples_perplexity_model(config, ctx, model))
        .collect()
}

/// Resolve a KLD reference for `model`, in priority order: explicit `--kldref`,
/// then `HIPFIRE_EVAL_KLDREF`, then a sibling `<stem>.kldref.hfq` / `.pkld`,
/// then `~/.hipfire/datasets/kldref/<stem>.{kldref.hfq,pkld}`. `None` ⇒ PPL-only
/// (not a failure). Generate one with `collect_artifacts --kldref` (HFQM) or
/// `perplexity --dump-ref` (.pkld).
pub(crate) fn resolve_kldref(model: &str, config: &EvalConfig) -> Option<PathBuf> {
    if let Some(p) = &config.kldref {
        if p.exists() {
            return Some(p.clone());
        }
    }
    if let Some(p) = std::env::var_os("HIPFIRE_EVAL_KLDREF").map(PathBuf::from) {
        if p.exists() {
            return Some(p);
        }
    }
    let stem = model_artifact_stem(model);
    let sibling_dir = Path::new(model)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut candidates = vec![
        sibling_dir.join(format!("{stem}.kldref.hfq")),
        sibling_dir.join(format!("{stem}.pkld")),
    ];
    if let Some(home) = home_dir() {
        let d = home.join(".hipfire").join("datasets").join("kldref");
        candidates.push(d.join(format!("{stem}.kldref.hfq")));
        candidates.push(d.join(format!("{stem}.pkld")));
    }
    candidates.into_iter().find(|p| p.exists())
}

/// Parse a `"<label>:<ws><number>..."` line from perplexity stdout (e.g.
/// `PPL:      14.6700`, `KLD/tok:  0.012345 (top-128, ...)`).
pub(crate) fn parse_labeled_f64(s: &str, label: &str) -> Option<f64> {
    let line = s.lines().find(|l| l.trim_start().starts_with(label))?;
    line[line.find(label)? + label.len()..]
        .split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()
}

pub(crate) fn run_examples_perplexity_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let label = "corpus_perplexity";
    let corpus_rel = std::env::var("HIPFIRE_EVAL_PERPLEXITY_CORPUS").unwrap_or_else(|_| {
        "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt".into()
    });
    let prompt_ref = prompt(&corpus_rel);
    // KLD needs a reference; absent ⇒ PPL-only (not a failure). See resolve_kldref.
    let kldref = resolve_kldref(&model, config);
    let kld_on = kldref.is_some();
    let base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("perplexity")),
        ("kld_ref".to_string(), json!(kld_on)),
    ]);

    let skip = |reason: &str| -> EvalResult {
        row_for_model(
            BatteryId::Perplexity,
            None,
            label,
            None,
            EvalStatus::Skip,
            Some(reason.to_string()),
            base_metrics.clone(),
            config,
            ctx,
            prompt_ref.clone(),
            0,
            model.clone(),
        )
    };

    if !Path::new(&model).exists() {
        return skip("perplexity requires --model to be a local filesystem path");
    }
    let Some(bin) = resolve_perplexity_bin() else {
        return skip("perplexity example binary not found; build with `cargo build --release -p hipfire-runtime --example perplexity`");
    };
    let corpus = repo_root()
        .map(|r| r.join(&corpus_rel))
        .unwrap_or_else(|| PathBuf::from(&corpus_rel));
    if !corpus.exists() {
        return skip(&format!(
            "corpus not found: {} — set HIPFIRE_EVAL_PERPLEXITY_CORPUS or place a corpus under ~/.hipfire/datasets/",
            corpus.display()
        ));
    }

    let ctx_len = std::env::var("HIPFIRE_EVAL_PERPLEXITY_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512);
    let kv_mode = config.kv_mode.as_deref().unwrap_or("q8").to_string();
    let mut args = vec![
        model.clone(),
        corpus.display().to_string(),
        "--ctx".to_string(),
        ctx_len.to_string(),
        "--warmup".to_string(),
        "8".to_string(),
        "--offset".to_string(),
        "0".to_string(),
        "--kv-mode".to_string(),
        kv_mode.clone(),
    ];
    if let Some(kref) = &kldref {
        args.push("--kld-ref".to_string());
        args.push(kref.display().to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(o) => o,
        Err(err) => {
            let mut m = base_metrics.clone();
            m.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Perplexity,
                None,
                label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn perplexity: {err}")),
                m,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let ppl = parse_labeled_f64(&stdout, "PPL:");
    let nll = parse_labeled_f64(&stdout, "NLL/tok:");
    let kld = parse_labeled_f64(&stdout, "KLD/tok:");

    let mut metrics = base_metrics.clone();
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("ctx".to_string(), json!(ctx_len));
    metrics.insert("kv_mode".to_string(), json!(kv_mode));
    if let Some(p) = ppl {
        metrics.insert("ppl".to_string(), json!(p));
    }
    if let Some(n) = nll {
        metrics.insert("nll_per_tok".to_string(), json!(n));
    }
    if let Some(k) = kld {
        metrics.insert("kld_per_tok".to_string(), json!(k));
    }
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );

    // Pass when a finite, positive PPL was produced (and, if a KLD ref was
    // supplied, a KLD value too). Baseline-regression gating is the sweep
    // caller's job; this battery is the measurement primitive.
    let ppl_ok = ppl.is_some_and(|p| p.is_finite() && p > 0.0);
    let kld_ok = !kld_on || kld.is_some_and(|k| k.is_finite());
    let ok = output.status.success() && ppl_ok && kld_ok;
    let status = if ok {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let reason = if ok {
        None
    } else if !output.status.success() {
        Some(format!("perplexity exited with {}", output.status))
    } else if !ppl_ok {
        Some("perplexity produced no finite positive PPL".to_string())
    } else {
        Some("KLD ref supplied but no KLD/tok produced".to_string())
    };

    row_for_model(
        BatteryId::Perplexity,
        None,
        label,
        None,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct Qwen35SpeedCase {
    pub(crate) label: &'static str,
    pub(crate) prefill: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct DaemonSpeedCase {
    pub(crate) label: &'static str,
}

pub(crate) fn qwen35_speed_cases() -> &'static [Qwen35SpeedCase] {
    &[
        Qwen35SpeedCase {
            label: "pp32_prefill_decode",
            prefill: 32,
        },
        Qwen35SpeedCase {
            label: "pp128_prefill_decode",
            prefill: 128,
        },
    ]
}

pub(crate) fn daemon_speed_cases() -> &'static [DaemonSpeedCase] {
    &[
        DaemonSpeedCase {
            label: "daemon_prefill_decode_first",
        },
        DaemonSpeedCase {
            label: "daemon_prefill_decode_reset",
        },
    ]
}

pub(crate) fn run_examples_qwen35_speed_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Vec<EvalResult> {
    evaluation_models(config)
        .into_iter()
        .flat_map(|model| run_examples_qwen35_speed_model(config, ctx, model))
        .collect()
}

pub(crate) fn run_examples_qwen35_speed_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> Vec<EvalResult> {
    let cases = qwen35_speed_cases();
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    let kv_mode = config.kv_mode.as_deref().unwrap_or("asym3").to_string();
    let mut rows = Vec::new();
    let base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("qwen35_speed_gate")),
        ("kv_mode".to_string(), json!(kv_mode.as_str())),
    ]);

    if !Path::new(&model).exists() {
        for case in cases {
            rows.push(row_for_model(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some(
                    "bench_qwen35_speed requires --model to be a local filesystem path".to_string(),
                ),
                {
                    let mut m = base_metrics.clone();
                    m.insert("prefill_tokens".to_string(), json!(case.prefill));
                    m
                },
                config,
                ctx,
                prompt_ref.clone(),
                0,
                model.clone(),
            ));
        }
        return rows;
    }

    let Some(bin) = resolve_bench_qwen35_speed_bin() else {
        for case in cases {
            rows.push(row_for_model(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some("bench_qwen35_speed example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_speed`".to_string()),
                {
                    let mut m = base_metrics.clone();
                    m.insert("prefill_tokens".to_string(), json!(case.prefill));
                    m
                },
                config,
                ctx,
                prompt_ref.clone(),
                0,
                model.clone(),
            ));
        }
        return rows;
    };

    let prefill_list: Vec<String> = cases.iter().map(|case| case.prefill.to_string()).collect();
    let args = vec![
        model.clone(),
        "--prefill-list".to_string(),
        prefill_list.join(","),
        "--prefill-runs".to_string(),
        "2".to_string(),
        "--warmup".to_string(),
        "5".to_string(),
        "--gen".to_string(),
        config.max_tokens.max(50).to_string(),
    ];
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let mut command = Command::new(&bin);
    command.args(&args);
    command.env("HIPFIRE_KV_MODE", kv_mode);
    command.env("HIPFIRE_DPM_WARMUP_SECS", "3");
    if !model_artifact_stem(&model).contains("0.8b") {
        command.env("HIPFIRE_GRAPH", "1");
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            for case in cases {
                rows.push(row_for_model(
                    BatteryId::Speed,
                    None,
                    case.label,
                    None,
                    EvalStatus::Fail,
                    Some(format!("spawn bench_qwen35_speed: {err}")),
                    {
                        let mut m = base_metrics.clone();
                        m.insert("prefill_tokens".to_string(), json!(case.prefill));
                        m.insert("command".to_string(), json!(command_display.clone()));
                        m.insert("graph_enabled".to_string(), json!(false));
                        m
                    },
                    config,
                    ctx,
                    prompt_ref.clone(),
                    elapsed_since_ms(started),
                    model.clone(),
                ));
            }
            return rows;
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let case_metrics = parse_case_summary_metrics(&stderr);
    let shared_metrics = parse_summary_kv_metrics(&stderr);
    let stdout_hash = stable_hash_bytes(stdout.as_bytes());
    let stderr_hash = stable_hash_bytes(stderr.as_bytes());

    for case in cases {
        let mut metrics = base_metrics.clone();
        metrics.insert("prefill_tokens".to_string(), json!(case.prefill));
        metrics.insert("command".to_string(), json!(command_display.clone()));
        metrics.insert(
            "graph_enabled".to_string(),
            json!(!model_artifact_stem(&model).contains("0.8b")),
        );
        metrics.insert("stdout_hash".to_string(), json!(stdout_hash));
        metrics.insert("stderr_hash".to_string(), json!(stderr_hash));
        if let Some(case_row_metrics) = case_metrics.get(case.label) {
            metrics.extend(case_row_metrics.clone());
        } else {
            metrics.extend(shared_metrics.clone());
        }
        if let Some(v) = metrics.get("gen_tok_s").cloned() {
            metrics.entry("tok_s".to_string()).or_insert(v);
        }
        let baseline_check = apply_speed_baseline(&mut metrics, &model, case.prefill, ctx);
        let baseline_failed = baseline_check
            .as_ref()
            .map(|check| check.failed)
            .unwrap_or(false);
        let baseline_error = baseline_check
            .and_then(|check| check.error)
            .or_else(|| None);
        let status = if output.status.success()
            && metrics.contains_key("prefill_tok_s")
            && !baseline_failed
            && baseline_error.is_none()
        {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        };
        let reason = if let Some(error) = &baseline_error {
            Some(error.to_string())
        } else if baseline_failed {
            Some("bench_qwen35_speed fell below perf baseline floor".to_string())
        } else if !case_metrics.contains_key(case.label) || !metrics.contains_key("prefill_tok_s") {
            Some(format!(
                "bench_qwen35_speed did not emit case metrics for {}",
                case.label
            ))
        } else if !output.status.success() {
            Some(format!("bench_qwen35_speed exited with {}", output.status))
        } else {
            None
        };
        rows.push(row_for_model(
            BatteryId::Speed,
            None,
            case.label,
            None,
            status,
            reason,
            metrics,
            config,
            ctx,
            prompt_ref.clone(),
            elapsed_ms,
            model.clone(),
        ));
    }
    rows
}

pub(crate) fn parse_case_summary_metrics(
    output: &str,
) -> HashMap<&'static str, BTreeMap<String, Value>> {
    let mut per_case: HashMap<&'static str, BTreeMap<String, Value>> = HashMap::new();
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if !line.starts_with("CASE_SUMMARY") {
            continue;
        }
        let mut label = None;
        let mut metrics = BTreeMap::new();
        for part in line.split_whitespace().skip(1) {
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            if key == "label" {
                label = Some(match value {
                    "pp32_prefill_decode" => "pp32_prefill_decode",
                    "pp128_prefill_decode" => "pp128_prefill_decode",
                    _ => "",
                });
                continue;
            }
            let value = value.trim_end_matches(',');
            if let Ok(v) = value.parse::<u64>() {
                metrics.insert(key.to_string(), json!(v));
            } else if let Ok(v) = value.parse::<f64>() {
                metrics.insert(key.to_string(), json!(v));
            } else {
                metrics.insert(key.to_string(), json!(value));
            }
        }
        if let Some("pp32_prefill_decode" | "pp128_prefill_decode") = label {
            let key = label.unwrap();
            per_case.insert(key, metrics);
        }
    }
    per_case
}

pub(crate) struct SpeedBaselineCheck {
    pub(crate) failed: bool,
    pub(crate) error: Option<String>,
}

pub(crate) fn apply_speed_baseline(
    metrics: &mut BTreeMap<String, Value>,
    model: &str,
    prefill: usize,
    ctx: &EvalContext,
) -> Option<SpeedBaselineCheck> {
    let require = env_truthy("HIPFIRE_REQUIRE_PERF_BASELINE");
    let baseline_path = match resolve_perf_baseline_path(ctx) {
        Ok(Some(path)) => path,
        Ok(None) => {
            metrics.insert("perf_baseline_status".to_string(), json!("missing"));
            if require {
                return Some(SpeedBaselineCheck {
                    failed: false,
                    error: Some("no matching perf baseline found".to_string()),
                });
            }
            return None;
        }
        Err(err) => {
            metrics.insert("perf_baseline_status".to_string(), json!("error"));
            return Some(SpeedBaselineCheck {
                failed: false,
                error: Some(err),
            });
        }
    };
    let model_size = match speed_model_size(model) {
        Some(size) => size,
        None => {
            metrics.insert("perf_baseline_status".to_string(), json!("unmatched_model"));
            if require {
                return Some(SpeedBaselineCheck {
                    failed: false,
                    error: Some(format!("cannot infer model size from {model}")),
                });
            }
            return None;
        }
    };
    let model_id = speed_model_id(model);
    let baseline = match load_speed_baseline(&baseline_path, &model_id, &model_size, prefill) {
        Ok(Some(baseline)) => baseline,
        Ok(None) => {
            metrics.insert("perf_baseline_status".to_string(), json!("missing_row"));
            metrics.insert(
                "perf_baseline_path".to_string(),
                json!(baseline_path.display().to_string()),
            );
            if require {
                return Some(SpeedBaselineCheck {
                    failed: false,
                    error: Some(format!(
                        "perf baseline {} has no speed row for {model_id} pp{prefill}",
                        baseline_path.display()
                    )),
                });
            }
            return None;
        }
        Err(err) => {
            metrics.insert("perf_baseline_status".to_string(), json!("error"));
            return Some(SpeedBaselineCheck {
                failed: false,
                error: Some(err),
            });
        }
    };
    let tolerance = baseline.tolerance_pct / 100.0;
    let prefill_observed = metrics.get("prefill_tok_s").and_then(Value::as_f64);
    let gen_observed = metrics.get("gen_tok_s").and_then(Value::as_f64);
    let mut failed = false;
    metrics.insert("perf_baseline_status".to_string(), json!("compared"));
    metrics.insert(
        "perf_baseline_path".to_string(),
        json!(baseline_path.display().to_string()),
    );
    metrics.insert("baseline_label".to_string(), json!(baseline.label));
    metrics.insert("baseline_model_id".to_string(), json!(baseline.model_id));
    metrics.insert("baseline_format".to_string(), json!(baseline.format));
    metrics.insert(
        "baseline_tolerance_pct".to_string(),
        json!(baseline.tolerance_pct),
    );
    if let Some(floor) = baseline.prefill_tok_s {
        metrics.insert("baseline_prefill_tok_s".to_string(), json!(floor));
        metrics.insert(
            "baseline_prefill_floor_tok_s".to_string(),
            json!(floor * (1.0 - tolerance)),
        );
        if prefill_observed.is_some_and(|observed| observed < floor * (1.0 - tolerance)) {
            failed = true;
        }
    }
    if let Some(floor) = baseline.gen_tok_s {
        metrics.insert("baseline_gen_tok_s".to_string(), json!(floor));
        metrics.insert(
            "baseline_gen_floor_tok_s".to_string(),
            json!(floor * (1.0 - tolerance)),
        );
        if gen_observed.is_some_and(|observed| observed < floor * (1.0 - tolerance)) {
            failed = true;
        }
    }
    metrics.insert("perf_baseline_failed".to_string(), json!(failed));
    Some(SpeedBaselineCheck {
        failed,
        error: None,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct SpeedBaseline {
    pub(crate) label: String,
    pub(crate) model_id: String,
    pub(crate) format: String,
    pub(crate) prefill_tok_s: Option<f64>,
    pub(crate) gen_tok_s: Option<f64>,
    pub(crate) tolerance_pct: f64,
}

pub(crate) fn load_speed_baseline(
    path: &Path,
    model_id: &str,
    model_size: &str,
    prefill: usize,
) -> Result<Option<SpeedBaseline>, String> {
    let body = fs::read_to_string(path)
        .map_err(|err| format!("read perf baseline {}: {err}", path.display()))?;
    let value: Value = serde_json::from_str(&body)
        .map_err(|err| format!("parse perf baseline {}: {err}", path.display()))?;
    let tolerance_pct = value
        .get("tolerance_pct")
        .and_then(Value::as_f64)
        .unwrap_or(5.0);
    let Some(rows) = value
        .get("baselines")
        .and_then(|v| v.get("speed"))
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    for row in rows {
        let row_model_id = row.get("model_id").and_then(Value::as_str);
        let row_size = row.get("model_size").and_then(Value::as_str);
        let row_prefill = row.get("prefill_tokens").and_then(Value::as_u64);
        if row_model_id == Some(model_id)
            && row_size == Some(model_size)
            && row_prefill == Some(prefill as u64)
        {
            return Ok(Some(SpeedBaseline {
                label: row
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("speed")
                    .to_string(),
                model_id: model_id.to_string(),
                format: row
                    .get("format")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                prefill_tok_s: row.get("prefill_tok_s").and_then(Value::as_f64),
                gen_tok_s: row.get("gen_tok_s").and_then(Value::as_f64),
                tolerance_pct: row
                    .get("tolerance_pct")
                    .and_then(Value::as_f64)
                    .unwrap_or(tolerance_pct),
            }));
        }
    }
    Ok(None)
}

pub(crate) fn resolve_perf_baseline_path(ctx: &EvalContext) -> Result<Option<PathBuf>, String> {
    if let Ok(path) = std::env::var("HIPFIRE_PERF_BASELINE") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(Some(path));
        }
        return Err(format!(
            "HIPFIRE_PERF_BASELINE points to missing file: {}",
            path.display()
        ));
    }
    let Some(arch) = ctx.arch.as_deref().or(ctx.host_profile.gfx.as_deref()) else {
        return Ok(None);
    };
    let root = std::env::var("HIPFIRE_PERF_BASELINE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("benchmarks/perf-baselines"));
    let root = if root.is_absolute() {
        root
    } else {
        repo_root().unwrap_or_else(|| PathBuf::from(".")).join(root)
    };
    let pattern_prefix = format!("{arch}-");
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(format!("read perf baseline dir {}: {err}", root.display())),
    };
    let mut matches = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with(&pattern_prefix) && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    matches.sort();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.pop()),
        _ => Err(format!(
            "multiple perf baselines for {arch}; set HIPFIRE_PERF_BASELINE"
        )),
    }
}

pub(crate) fn speed_model_id(model: &str) -> String {
    model_artifact_stem(model).to_ascii_lowercase()
}

pub(crate) fn speed_model_size(model: &str) -> Option<String> {
    let stem = model_artifact_stem(model).to_ascii_lowercase();
    for size in ["0.8b", "4b", "9b", "27b", "35b-a3b"] {
        if stem.contains(size) {
            return Some(size.to_string());
        }
    }
    None
}

pub(crate) fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

#[derive(Clone, Copy)]
pub(crate) struct AgenticCase {
    pub(crate) label: &'static str,
    pub(crate) system_path: &'static str,
    pub(crate) thinking_clamped: bool,
    pub(crate) max_tokens: usize,
}

pub(crate) fn agentic_cases() -> &'static [AgenticCase] {
    &[
        AgenticCase {
            label: "agentic_pi_unclamped",
            system_path: "benchmarks/prompts/agentic_pi_system.txt",
            thinking_clamped: false,
            max_tokens: 256,
        },
        AgenticCase {
            label: "agentic_pi_clamped",
            system_path: "benchmarks/prompts/agentic_pi_system.txt",
            thinking_clamped: true,
            max_tokens: 256,
        },
        AgenticCase {
            label: "agentic_hermes_unclamped",
            system_path: "benchmarks/prompts/agentic_hermes_system.txt",
            thinking_clamped: false,
            max_tokens: 256,
        },
    ]
}

pub(crate) fn run_examples_agentic_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    let mut rows = evaluation_models(config)
        .into_iter()
        .flat_map(|model| {
            agentic_cases()
                .iter()
                .map(move |case| run_examples_agentic_case(config, ctx, model.clone(), *case))
        })
        .collect::<Vec<_>>();
    rows.push(run_examples_agentic_jinja_tools_case(
        config,
        ctx,
        config.model.clone(),
    ));
    rows
}

pub(crate) fn run_examples_agentic_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
    case: AgenticCase,
) -> EvalResult {
    let prompt_path = "benchmarks/prompts/agentic_user_read.txt";
    let prompt_ref = combined_prompt_ref(case.system_path, prompt_path);
    let mut metrics = BTreeMap::from([
        ("suite".to_string(), json!("agentic_tool_call")),
        ("system_prompt".to_string(), json!(case.system_path)),
        ("user_prompt".to_string(), json!(prompt_path)),
        ("thinking_clamped".to_string(), json!(case.thinking_clamped)),
        (
            "detector_profile".to_string(),
            json!("agentic_toolcall_shape"),
        ),
    ]);
    if case.thinking_clamped {
        metrics.insert("max_think_tokens".to_string(), json!(1));
    }
    let mut profile = hipfire_coherence::DetectorProfile::default_for_prompt("", None);
    profile.agentic = true;
    run_daemon_coherence_anchor(
        BatteryId::Agentic,
        case.label,
        prompt_path,
        prompt_ref,
        config,
        ctx,
        model,
        Some(4096),
        Some(case.max_tokens),
        metrics,
        Some(case.system_path),
        None,
        None,
        false,
        Some(profile),
    )
}

pub(crate) fn run_examples_agentic_jinja_tools_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_path = "benchmarks/prompts/agentic_jinja_tools_user.txt";
    let system_path = "benchmarks/prompts/agentic_jinja_tools_system.txt";
    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name."
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["c", "f"],
                            "description": "Temperature unit."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]);
    let prompt_ref = structured_tools_prompt_ref(system_path, prompt_path, &tools);
    let metrics = BTreeMap::from([
        ("suite".to_string(), json!("agentic_jinja_tools")),
        ("system_prompt".to_string(), json!(system_path)),
        ("user_prompt".to_string(), json!(prompt_path)),
        ("tools_count".to_string(), json!(1.0)),
        ("force_jinja_chat".to_string(), json!(true)),
        ("assistant_prefix".to_string(), json!("closed_think")),
        (
            "detector_profile".to_string(),
            json!("agentic_structured_tools"),
        ),
    ]);
    let mut profile = hipfire_coherence::DetectorProfile::default_for_prompt("", None);
    profile.agentic = true;
    run_daemon_coherence_anchor(
        BatteryId::Agentic,
        "agentic_jinja_structured_tools",
        prompt_path,
        prompt_ref,
        config,
        ctx,
        model,
        Some(1024),
        Some(192),
        metrics,
        Some(system_path),
        Some(tools),
        Some("closed_think"),
        true,
        Some(profile),
    )
}

#[derive(Clone, Copy)]
pub(crate) struct RuntimeCase {
    pub(crate) label: &'static str,
    pub(crate) script: &'static str,
    pub(crate) category: &'static str,
}

pub(crate) fn runtime_cases() -> &'static [RuntimeCase] {
    &[
        RuntimeCase {
            label: "server_prefill_batch",
            script: "tests/smoke-server-prefill-batch.sh",
            category: "prefill_batching",
        },
        RuntimeCase {
            label: "server_decode_batch",
            script: "tests/smoke-server-decode-batch.sh",
            category: "decode_batching",
        },
        RuntimeCase {
            label: "daemon_generate_batch_prefill",
            script: "tests/smoke-generate-batch-prefill.sh",
            category: "prefill_batching",
        },
        RuntimeCase {
            label: "prefix_checkpoint_reuse",
            script: "tests/smoke-server-prefix-checkpoint-reuse.sh",
            category: "prefix_reuse",
        },
        RuntimeCase {
            label: "prefix_boundary_reuse",
            script: "tests/smoke-server-prefix-boundary-reuse.sh",
            category: "prefix_reuse",
        },
        RuntimeCase {
            label: "responses_prefix_reuse",
            script: "tests/smoke-server-responses-prefix-reuse.sh",
            category: "prefix_reuse",
        },
        RuntimeCase {
            label: "shared_prefix_fanout",
            script: "tests/smoke-server-shared-prefix-fanout.sh",
            category: "shared_prefix_fanout",
        },
        RuntimeCase {
            label: "multi_model_workers",
            script: "tests/smoke-server-multi-model-workers.sh",
            category: "multi_model_workers",
        },
        RuntimeCase {
            label: "server_concurrency",
            script: "tests/stress-server-concurrency.sh",
            category: "concurrency",
        },
        RuntimeCase {
            label: "kv_budget_reload",
            script: "tests/e2e_kv_budget.sh",
            category: "kv_admission",
        },
        RuntimeCase {
            label: "kv_reject_http",
            script: "tests/e2e_kv_reject.sh",
            category: "kv_admission",
        },
        RuntimeCase {
            label: "kv_reject_run",
            script: "tests/e2e_run_reject.sh",
            category: "kv_admission",
        },
        RuntimeCase {
            label: "pipeline_parallel_gate",
            script: "tests/pp-gate.sh",
            category: "pipeline_parallel",
        },
        RuntimeCase {
            label: "pipeline_parallel_coherence",
            script: "tests/coherence-gate-pp.sh",
            category: "pipeline_parallel",
        },
    ]
}

pub(crate) fn run_examples_runtime_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    runtime_cases()
        .iter()
        .map(|case| run_examples_runtime_case(config, ctx, *case))
        .collect()
}

pub(crate) fn run_examples_runtime_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: RuntimeCase,
) -> EvalResult {
    let script = Path::new(case.script);
    let mut metrics = BTreeMap::from([
        ("runtime_evidence_case".to_string(), json!(case.label)),
        ("runtime_category".to_string(), json!(case.category)),
        ("script".to_string(), json!(case.script)),
        ("runtime_path".to_string(), json!("shell_runtime_gate")),
    ]);

    if !script.is_file() {
        return skip_row_with_metrics(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            "runtime evidence script is not present",
            config,
            ctx,
            None,
            metrics,
        );
    }

    if is_local_filesystem_model(&config.model) && !Path::new(&config.model).is_file() {
        metrics.insert("model_missing".to_string(), json!(true));
        return skip_row_with_metrics(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            "model path is missing; runtime evidence script not run",
            config,
            ctx,
            None,
            metrics,
        );
    }

    let started = SystemTime::now();
    let output = match Command::new("bash")
        .arg(script)
        .env("MODEL", &config.model)
        .env("HIPFIRE_MODEL", &config.model)
        .env("HIPFIRE_EVAL_RUNTIME_ROW", case.label)
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return row(
                BatteryId::Runtime,
                None,
                case.label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn runtime evidence script: {err}")),
                metrics,
                config,
                ctx,
                None,
                elapsed_since_ms(started),
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    metrics.insert(
        "exit_code".to_string(),
        json!(output.status.code().unwrap_or(-1)),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    metrics.insert(
        "stdout_excerpt".to_string(),
        json!(truncate_for_metric(stdout.trim(), 512)),
    );
    metrics.insert(
        "stderr_excerpt".to_string(),
        json!(truncate_for_metric(stderr.trim(), 512)),
    );

    if output.status.success() {
        if combined.to_ascii_lowercase().contains("skipping")
            || combined.to_ascii_lowercase().contains("skipped")
        {
            skip_row_with_metrics(
                BatteryId::Runtime,
                None,
                case.label,
                None,
                "runtime evidence script skipped on this host",
                config,
                ctx,
                None,
                metrics,
            )
        } else {
            row(
                BatteryId::Runtime,
                None,
                case.label,
                None,
                EvalStatus::Pass,
                None,
                metrics,
                config,
                ctx,
                None,
                elapsed_ms,
            )
        }
    } else if output.status.code() == Some(2)
        && runtime_script_output_is_environment_skip(&combined)
    {
        skip_row_with_metrics(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            "runtime evidence script prerequisites are unavailable on this host",
            config,
            ctx,
            None,
            metrics,
        )
    } else {
        row(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            EvalStatus::Fail,
            Some("runtime evidence script failed".to_string()),
            metrics,
            config,
            ctx,
            None,
            elapsed_ms,
        )
    }
}

pub(crate) fn is_local_filesystem_model(model: &str) -> bool {
    model.starts_with('/') || model.starts_with("./") || model.starts_with("../")
}

pub(crate) fn runtime_script_output_is_environment_skip(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    [
        "missing model",
        "model not found",
        "missing daemon binary",
        "daemon binary",
        "build it with:",
        "fewer than 2",
        "less than 2",
        "skipping",
        "not found",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

pub(crate) fn truncate_for_metric(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

#[derive(Clone, Copy)]
pub(crate) struct DflashMatrixCase {
    pub(crate) label: &'static str,
    pub(crate) prompt_path: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) max_tokens: usize,
    pub(crate) extra_args: &'static [&'static str],
}

pub(crate) fn dflash_matrix_cases() -> &'static [DflashMatrixCase] {
    &[
        DflashMatrixCase {
            label: "dflash_prose_attractor",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "dflash",
            max_tokens: 192,
            extra_args: &[],
        },
        DflashMatrixCase {
            label: "dflash_code_attractor",
            prompt_path: "benchmarks/prompts/humaneval_0_has_close_elements.txt",
            mode: "dflash",
            max_tokens: 128,
            extra_args: &[],
        },
        DflashMatrixCase {
            label: "ddtree_b12_k2_prose",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "ddtree-batched",
            max_tokens: 192,
            extra_args: &[
                "--ddtree-batched",
                "--ddtree-budget",
                "12",
                "--ddtree-topk",
                "2",
            ],
        },
        DflashMatrixCase {
            label: "path_c_phase1_prose",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "path-c-phase1",
            max_tokens: 192,
            extra_args: &[
                "--ddtree-path-c",
                "phase1",
                "--ddtree-budget",
                "12",
                "--ddtree-topk",
                "2",
            ],
        },
        DflashMatrixCase {
            label: "path_c_phase2_prose",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "path-c-phase2",
            max_tokens: 192,
            extra_args: &[
                "--ddtree-path-c",
                "phase2",
                "--ddtree-budget",
                "12",
                "--ddtree-topk",
                "2",
            ],
        },
    ]
}

pub(crate) fn run_examples_dflash_matrix_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Vec<EvalResult> {
    if matches!(config.dflash, DflashMode::Off) {
        return dflash_matrix_cases()
            .iter()
            .map(|case| {
                skip_row(
                    BatteryId::Dflash,
                    None,
                    case.label,
                    None,
                    "DFlash/DDTree matrix disabled by --dflash off",
                    config,
                    ctx,
                    prompt(case.prompt_path),
                )
            })
            .collect();
    }
    dflash_matrix_cases()
        .iter()
        .map(|case| run_examples_dflash_matrix_case(config, ctx, *case))
        .collect()
}

pub(crate) fn run_examples_dflash_matrix_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: DflashMatrixCase,
) -> EvalResult {
    let mut metrics = BTreeMap::from([
        ("suite".to_string(), json!("dflash_ddtree_path_c")),
        ("mode".to_string(), json!(case.mode)),
        ("max_tokens".to_string(), json!(case.max_tokens)),
        (
            "token_attractor_detector".to_string(),
            json!("dflash_spec_demo_tokens_v1"),
        ),
    ]);
    let mut extra_args = case.extra_args.to_vec();
    let max_tokens = case.max_tokens.to_string();
    extra_args.extend(["--max", max_tokens.as_str()]);
    metrics.insert("extra_args".to_string(), json!(case.extra_args));
    run_dflash_spec_demo_anchor(
        BatteryId::Dflash,
        case.label,
        false,
        case.prompt_path,
        &extra_args,
        metrics,
        config,
        ctx,
        config.model.clone(),
    )
}

#[derive(Clone, Copy)]
pub(crate) struct PflashNiahCase {
    pub(crate) label: &'static str,
    pub(crate) fixture: &'static str,
    pub(crate) mode: &'static str,
}

pub(crate) fn pflash_niah_cases() -> &'static [PflashNiahCase] {
    &[
        PflashNiahCase {
            label: "niah_8k_baseline",
            fixture: "benchmarks/longctx/niah/niah_8k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_8k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_8k.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "niah_16k_baseline",
            fixture: "benchmarks/longctx/niah/niah_16k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_16k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_16k.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "niah_multi_16k_baseline",
            fixture: "benchmarks/longctx/niah/niah_multi_16k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_multi_16k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_multi_16k.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "longcode_baseline",
            fixture: "benchmarks/prompts/longcode_pflash.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "longcode_pflash30",
            fixture: "benchmarks/prompts/longcode_pflash.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "longprose_baseline",
            fixture: "benchmarks/prompts/longprose_multidoc.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "longprose_pflash30",
            fixture: "benchmarks/prompts/longprose_multidoc.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "niah_32k_baseline",
            fixture: "benchmarks/longctx/niah/niah_32k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_32k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_32k.jsonl",
            mode: "pflash",
        },
    ]
}

pub(crate) fn pflash_maxgen_for_fixture(fixture: &str) -> usize {
    if fixture.contains("multi") {
        80
    } else if fixture.contains("longcode") || fixture.contains("longprose") {
        64
    } else {
        32
    }
}

pub(crate) fn run_examples_pflash_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    pflash_niah_cases()
        .iter()
        .map(|case| run_examples_pflash_case(config, ctx, *case))
        .collect()
}

pub(crate) fn run_examples_pflash_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: PflashNiahCase,
) -> EvalResult {
    let model = config.model.clone();
    let prompt_ref = prompt(case.fixture);
    let mut base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("pflash_niah")),
        ("fixture".to_string(), json!(case.fixture)),
        ("mode".to_string(), json!(case.mode)),
        ("kv_mode".to_string(), json!("asym3")),
        ("pretok".to_string(), json!(true)),
        (
            "maxgen".to_string(),
            json!(pflash_maxgen_for_fixture(case.fixture)),
        ),
    ]);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Pflash,
            None,
            case.label,
            None,
            EvalStatus::Skip,
            Some(
                "pflash examples executor requires --model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_pflash_niah_bench_bin() else {
        return row_for_model(
            BatteryId::Pflash,
            None,
            case.label,
            None,
            EvalStatus::Skip,
            Some("pflash_niah_bench example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example pflash_niah_bench`".to_string()),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    let Some(fixture_path) = resolve_repo_path(case.fixture) else {
        return row_for_model(
            BatteryId::Pflash,
            None,
            case.label,
            None,
            EvalStatus::Fail,
            Some(format!("pflash fixture not found: {}", case.fixture)),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let mut args = vec![
        model.clone(),
        fixture_path.display().to_string(),
        "--maxgen".to_string(),
        pflash_maxgen_for_fixture(case.fixture).to_string(),
        "--asym3".to_string(),
        "--pretok".to_string(),
    ];
    if case.mode == "pflash" {
        let Some(draft) = config.draft.as_deref() else {
            return row_for_model(
                BatteryId::Pflash,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some("pflash examples executor requires --draft for pflash mode".to_string()),
                base_metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        };
        if !Path::new(draft).exists() {
            return row_for_model(
                BatteryId::Pflash,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some(
                    "pflash examples executor requires --draft to be a local filesystem path"
                        .to_string(),
                ),
                base_metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        }
        base_metrics.insert("keep_ratio".to_string(), json!(0.30));
        base_metrics.insert("block_size".to_string(), json!(64));
        args.extend([
            "--pflash".to_string(),
            draft.to_string(),
            "--keep-ratio".to_string(),
            "0.30".to_string(),
            "--block-size".to_string(),
            "64".to_string(),
        ]);
    }

    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Pflash,
                None,
                case.label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn pflash_niah_bench: {err}")),
                base_metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    let mut metrics = parse_pflash_niah_metrics(&combined);
    metrics.extend(base_metrics);
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );

    let verdict = metrics
        .get("niah_verdict")
        .and_then(Value::as_str)
        .map(str::to_string);
    let ok = output.status.success() && metrics.contains_key("total_ms") && verdict.is_some();
    row_for_model(
        BatteryId::Pflash,
        None,
        case.label,
        None,
        if ok {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        },
        if ok {
            None
        } else if output.status.success() {
            Some("pflash_niah_bench did not emit total/verdict metrics".to_string())
        } else {
            Some(format!("pflash_niah_bench exited with {}", output.status))
        },
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}

pub(crate) fn parse_pflash_niah_metrics(output: &str) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::new();
    for raw in output.lines() {
        let line = raw.trim();
        if let Some(value) = prefixed_ms(line, "total:") {
            metrics.insert("total_ms".to_string(), json!(value));
        } else if let Some(value) = prefixed_ms(line, "prefill:") {
            metrics.insert("prefill_ms".to_string(), json!(value));
        } else if let Some(value) = prefixed_ms(line, "decode:") {
            metrics.insert("decode_ms".to_string(), json!(value));
        } else if let Some(value) = prefixed_ms(line, "compress:") {
            metrics.insert("compress_ms".to_string(), json!(value));
        } else if line.starts_with("PASS:") {
            metrics.insert("niah_verdict".to_string(), json!("PASS"));
            metrics.insert("niah_pass".to_string(), json!(1.0));
        } else if line.starts_with("FAIL:") {
            metrics.insert("niah_verdict".to_string(), json!("FAIL"));
            metrics.insert("niah_pass".to_string(), json!(0.0));
        }
    }
    metrics
}

pub(crate) fn prefixed_ms(line: &str, prefix: &str) -> Option<u64> {
    line.strip_prefix(prefix)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_daemon_coherence_anchor(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    prompt_ref: Option<PromptRef>,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
    max_seq: Option<usize>,
    max_tokens: Option<usize>,
    mut metrics: BTreeMap<String, Value>,
    system_path: Option<&str>,
    tools: Option<Value>,
    assistant_prefix: Option<&str>,
    force_jinja_chat: bool,
    profile: Option<hipfire_coherence::DetectorProfile>,
) -> EvalResult {
    metrics.insert("executor".to_string(), json!("daemon"));
    metrics.insert("runtime_path".to_string(), json!("daemon_jsonl"));
    metrics.insert("implemented".to_string(), json!(true));
    if !Path::new(&model).exists() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "daemon coherence executor requires --model to be a local filesystem path"
                    .to_string(),
            ),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    if !hipfire_coherence::daemon_binary_available() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("daemon binary not found; build with `cargo build --release --features deltanet -p hipfire-daemon --bin hipfire-daemon`".to_string()),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(resolved_prompt) = resolve_repo_path(prompt_path) else {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(format!("prompt not found: {prompt_path}")),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    let prompt_text = match fs::read_to_string(&resolved_prompt) {
        Ok(text) => text,
        Err(err) => {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("read prompt {}: {err}", resolved_prompt.display())),
                metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        }
    };
    let system_text = if let Some(path) = system_path {
        let Some(resolved_system) = resolve_repo_path(path) else {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("system prompt not found: {path}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        };
        match fs::read_to_string(&resolved_system) {
            Ok(text) => Some(text),
            Err(err) => {
                return row_for_model(
                    battery,
                    None,
                    case_id,
                    None,
                    EvalStatus::Fail,
                    Some(format!(
                        "read system prompt {}: {err}",
                        resolved_system.display()
                    )),
                    metrics,
                    config,
                    ctx,
                    prompt_ref,
                    0,
                    model,
                );
            }
        }
    } else {
        None
    };
    let profile = profile.unwrap_or_else(|| {
        hipfire_coherence::DetectorProfile::default_for_prompt(&prompt_text, system_text.as_deref())
    });
    let run_config = hipfire_coherence::CoherenceRunConfig {
        model: model.clone(),
        prompt: prompt_text,
        prompt_label: prompt_path.to_string(),
        system: system_text,
        tools,
        assistant_prefix: assistant_prefix.map(str::to_string),
        force_jinja_chat,
        max_tokens: max_tokens.unwrap_or(config.max_tokens),
        temperature: 0.0,
        repeat_penalty: None,
        repeat_window: None,
        max_seq: max_seq.unwrap_or_else(|| (config.max_tokens + 2048).max(4096)),
        state: None,
        profile,
    };
    let started = SystemTime::now();
    let output = match hipfire_coherence::run_coherence(&run_config) {
        Ok(output) => output,
        Err(err) => {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("daemon coherence probe failed: {err}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let artifact_dir = config.out_dir.join("artifacts").join("coherence");
    if let Err(err) = fs::create_dir_all(&artifact_dir) {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(format!("create coherence artifact dir: {err}")),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        );
    }
    let artifact_name = format!(
        "{}-{}.json",
        sanitize_path_component(case_id),
        stable_hash_bytes(model.as_bytes())
    );
    let artifact_path = artifact_dir.join(artifact_name);
    let artifact_value = output.artifact_value();
    if let Err(err) = write_json_pretty(&artifact_path, &artifact_value) {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(format!("write coherence artifact: {err}")),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        );
    }
    metrics.insert("hard_fails".to_string(), json!(output.hard_fails() as f64));
    metrics.insert("soft_warns".to_string(), json!(output.soft_warns() as f64));
    metrics.insert(
        "detector_count".to_string(),
        json!(output.report.rows.len() as f64),
    );
    metrics.insert(
        "detectors".to_string(),
        json!(hipfire_coherence::detector_rows(&output.report)),
    );
    metrics.insert(
        "generated_text_hash".to_string(),
        json!(stable_hash_bytes(output.generated_text.as_bytes())),
    );
    metrics.insert(
        "generated_visible_bytes".to_string(),
        json!(output.generated_text.len()),
    );
    metrics.insert(
        "generated_tokens".to_string(),
        json!(output.token_ids.len()),
    );
    metrics.insert("tok_s".to_string(), json!(output.report.header.tok_s));
    metrics.insert(
        "gen_tok_s".to_string(),
        json!(output.report.header.gen_tok_s),
    );
    metrics.insert("ttft_ms".to_string(), json!(output.report.header.ttft_ms));
    metrics.insert(
        "daemon_prefill_ms".to_string(),
        json!(output.report.header.daemon_prefill_ms),
    );
    metrics.insert(
        "daemon_decode_tok_s".to_string(),
        json!(output.report.header.daemon_decode_tok_s),
    );
    metrics.insert(
        "coherence_artifact_path".to_string(),
        json!(artifact_path.display().to_string()),
    );
    metrics.insert(
        "coherence_status".to_string(),
        json!(if output.hard_fails() > 0 {
            "fail"
        } else {
            "pass"
        }),
    );
    let status = if output.hard_fails() > 0 {
        EvalStatus::Fail
    } else {
        EvalStatus::Pass
    };
    let reason = if output.hard_fails() > 0 {
        Some(format!(
            "{} detector hard fail(s); see {}",
            output.hard_fails(),
            artifact_path.display()
        ))
    } else {
        None
    };
    row_for_model(
        battery, None, case_id, None, status, reason, metrics, config, ctx, prompt_ref, elapsed_ms,
        model,
    )
}

pub(crate) struct LongctxPrompt {
    pub(crate) prompt_path: String,
    pub(crate) prompt_ref: PromptRef,
    pub(crate) max_seq: usize,
    pub(crate) metrics: BTreeMap<String, Value>,
}

pub(crate) fn materialize_longctx_prompt(config: &EvalConfig) -> Result<LongctxPrompt, String> {
    let fixture_rel = "benchmarks/prompts/longprose_multidoc.jsonl";
    let fixture_path = repo_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(fixture_rel);
    let fixture = fs::read_to_string(&fixture_path)
        .map_err(|e| format!("read longctx fixture {}: {e}", fixture_path.display()))?;
    let first_line = fixture
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| format!("longctx fixture {fixture_rel} is empty"))?;
    let item: Value = serde_json::from_str(first_line)
        .map_err(|e| format!("parse longctx fixture {fixture_rel}: {e}"))?;
    let filler = item
        .get("filler_text")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("longctx fixture {fixture_rel} missing filler_text"))?;
    let question = item
        .get("question")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("longctx fixture {fixture_rel} missing question"))?;
    let expected = item
        .get("expected_answer_substring")
        .and_then(Value::as_str)
        .unwrap_or("");
    let context_tokens = item
        .get("context_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    let prompt_text = format!(
        "{filler}\n\nQuestion: {question}\nAnswer in one short sentence and include the exact requested value.\n"
    );
    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    fs::create_dir_all(&prompt_dir)
        .map_err(|e| format!("create longctx prompt dir {}: {e}", prompt_dir.display()))?;
    let prompt_path = prompt_dir.join("longctx_multidoc_0.txt");
    fs::write(&prompt_path, &prompt_text)
        .map_err(|e| format!("write longctx prompt {}: {e}", prompt_path.display()))?;
    let prompt_path_string = prompt_path.display().to_string();
    let prompt_ref = PromptRef::from_content(prompt_path_string.clone(), prompt_text.as_bytes());
    let max_seq = (context_tokens + config.max_tokens + 512).max(4096);
    let mut metrics = BTreeMap::new();
    metrics.insert("fixture".to_string(), json!("longprose_multidoc"));
    metrics.insert("fixture_source".to_string(), json!(fixture_rel));
    metrics.insert(
        "fixture_hash".to_string(),
        json!(file_hash(&fixture_path).unwrap_or_else(|| stable_hash_file_fallback(&fixture_path))),
    );
    metrics.insert("context_tokens".to_string(), json!(context_tokens));
    metrics.insert("longctx_max_seq".to_string(), json!(max_seq));
    metrics.insert("prompt_bytes".to_string(), json!(prompt_text.len()));
    metrics.insert(
        "expected_answer_hash".to_string(),
        json!(stable_hash_bytes(expected.as_bytes())),
    );
    if let Some(needle) = item.get("needle").and_then(Value::as_str) {
        metrics.insert(
            "needle_hash".to_string(),
            json!(stable_hash_bytes(needle.as_bytes())),
        );
    }
    if let Some(documents) = item.get("documents").and_then(Value::as_array) {
        metrics.insert("document_count".to_string(), json!(documents.len()));
    }
    Ok(LongctxPrompt {
        prompt_path: prompt_path_string,
        prompt_ref,
        max_seq,
        metrics,
    })
}

pub(crate) fn examples_barrage_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    let mut rows = Vec::new();
    for d in datasets {
        match (d.suite, d.status) {
            (SuiteId::Gpqa, EvalStatus::Pass) => {
                match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_gpqa_item(config, ctx, d, item.clone(), model)
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::Gpqa),
                                "gpqa_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        }));
                    }
                }
            }
            (SuiteId::LmEvalMicro, EvalStatus::Pass) => {
                match lm_eval_micro_materialized_items(&d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_lm_eval_micro_item(config, ctx, d, item.clone(), model)
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::LmEvalMicro),
                                "lm_eval_micro_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        }));
                    }
                }
            }
            (SuiteId::HumanEval, EvalStatus::Pass) => {
                match humaneval_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_humaneval_item(config, ctx, d, item.clone(), model)
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::HumanEval),
                                "humaneval_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        }));
                    }
                }
            }
            (SuiteId::DeepSwe | SuiteId::SweBench, EvalStatus::Pass) => {
                match builtin_barrage_materialized_items(d.suite, &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_builtin_barrage_item(
                                    config,
                                    ctx,
                                    d,
                                    item.clone(),
                                    model,
                                )
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(d.suite),
                                "builtin_software_eval_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        }));
                    }
                }
            }
            _ => {}
        }
    }
    if rows.is_empty() {
        barrage_rows(config, ctx, datasets)
    } else {
        rows
    }
}

pub(crate) fn evaluation_models(config: &EvalConfig) -> Vec<String> {
    std::iter::once(&config.model)
        .chain(config.baseline.iter())
        .chain(config.reference.iter())
        .cloned()
        .collect()
}

pub(crate) fn run_examples_lm_eval_micro_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: LmEvalMicroItem,
    model: String,
) -> EvalResult {
    let prompt_ref = PromptRef::from_content(
        format!("dataset:lm_eval_micro:{}", item.item_id),
        item.prompt.as_bytes(),
    );
    let mut base_metrics = BTreeMap::from([
        (
            "prompt_format".to_string(),
            json!("lm_eval_micro_zero_shot_v1"),
        ),
        ("task".to_string(), json!(item.task.clone())),
        ("answer_label".to_string(), json!(item.answer_label.clone())),
        ("answer_hash".to_string(), json!(item.answer_hash.clone())),
        ("choices_count".to_string(), json!(item.choices_count)),
        (
            "dataset_file".to_string(),
            json!("builtin:lm_eval_micro:v1"),
        ),
        ("scoring_mode".to_string(), json!("exact_letter")),
        ("executor".to_string(), json!("examples")),
    ]);
    add_dataset_provenance_metrics(&mut base_metrics, dataset);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "lm-eval-micro-{}.txt",
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write lm_eval_micro runtime prompt: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir =
        runtime_evidence_dir(config, &format!("lm-eval-micro-{}", item.item_id), &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if config.max_tokens + 2048 > 4096 {
        args.push("--max-seq".to_string());
        args.push((config.max_tokens + 2048).to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::LmEvalMicro),
                "lm_eval_micro_zero_shot_native",
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                base_metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.extend(base_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    let predicted = extract_answer_letter(&stdout);
    metrics.insert("predicted_label".to_string(), json!(predicted));
    let exact_match = predicted.as_deref() == Some(item.answer_label.as_str());
    metrics.insert(
        "exact_match".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );
    metrics.insert(
        "accuracy".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

pub(crate) fn run_examples_builtin_barrage_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: BuiltinBarrageItem,
    model: String,
) -> EvalResult {
    let case_id = "builtin_software_eval_native";
    let prompt_ref = PromptRef::from_content(
        format!("dataset:{}:{}", item.suite.as_str(), item.item_id),
        item.prompt.as_bytes(),
    );
    let mut base_metrics = BTreeMap::from([
        (
            "prompt_format".to_string(),
            json!(item.prompt_format.clone()),
        ),
        ("task".to_string(), json!(item.task.clone())),
        ("answer_label".to_string(), json!(item.answer_label.clone())),
        ("answer_hash".to_string(), json!(item.answer_hash.clone())),
        ("choices_count".to_string(), json!(item.choices_count)),
        ("dataset_file".to_string(), json!(item.dataset_file.clone())),
        ("scoring_mode".to_string(), json!(item.scoring_mode.clone())),
        ("executor".to_string(), json!("examples")),
    ]);
    add_dataset_provenance_metrics(&mut base_metrics, dataset);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "{}-{}.txt",
        item.suite.as_str().replace('_', "-"),
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write builtin software-eval runtime prompt: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir = runtime_evidence_dir(
        config,
        &format!("{}-{}", item.suite.as_str(), item.item_id),
        &model,
    );
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if config.max_tokens + 2048 > 4096 {
        args.push("--max-seq".to_string());
        args.push((config.max_tokens + 2048).to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(item.suite),
                case_id,
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                base_metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.extend(base_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    let predicted = extract_answer_letter(&stdout);
    metrics.insert("predicted_label".to_string(), json!(predicted));
    let exact_match = predicted.as_deref() == Some(item.answer_label.as_str());
    metrics.insert(
        "exact_match".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );
    metrics.insert(
        "accuracy".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

pub(crate) fn run_examples_humaneval_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: HumanEvalItem,
    model: String,
) -> EvalResult {
    let prompt_ref = PromptRef::from_content(
        format!("dataset:humaneval:{}", item.item_id),
        item.prompt.as_bytes(),
    );
    let mut metrics = BTreeMap::from([
        (
            "prompt_format".to_string(),
            json!("humaneval_completion_v1"),
        ),
        ("task_id".to_string(), json!(item.task_id.clone())),
        ("dataset_file".to_string(), json!(item.dataset_file.clone())),
        ("executor".to_string(), json!("examples")),
        ("scoring_mode".to_string(), json!("execution_only")),
    ]);
    add_dataset_provenance_metrics(&mut metrics, dataset);
    if let Some(hash) = &item.canonical_solution_hash {
        metrics.insert("canonical_solution_hash".to_string(), json!(hash));
    }
    if let Some(hash) = &item.test_hash {
        metrics.insert("test_hash".to_string(), json!(hash));
    }

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("examples executor requires --model to be a local filesystem path".to_string()),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "humaneval-{}.txt",
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write HumanEval runtime prompt: {err}")),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir = runtime_evidence_dir(config, &format!("humaneval-{}", item.item_id), &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::HumanEval),
                "humaneval_completion_native",
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    metrics.extend(parse_bench_metrics(&stderr));
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

pub(crate) fn run_examples_gpqa_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: GpqaItem,
    model: String,
) -> EvalResult {
    let prompt_ref = PromptRef::from_content(
        format!("dataset:gpqa:{}", item.item_id),
        item.prompt.as_bytes(),
    );
    let mut base_metrics = BTreeMap::from([
        ("prompt_format".to_string(), json!("gpqa_zero_shot_v1")),
        ("answer_label".to_string(), json!(item.answer_label.clone())),
        (
            "answer_hash".to_string(),
            json!(stable_hash_bytes(item.correct_answer.as_bytes())),
        ),
        ("choices_count".to_string(), json!(item.choices.len())),
        ("dataset_file".to_string(), json!(item.dataset_file.clone())),
        ("executor".to_string(), json!("examples")),
    ]);
    add_dataset_provenance_metrics(&mut base_metrics, dataset);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "gpqa-{}.txt",
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write GPQA runtime prompt: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir = runtime_evidence_dir(config, &format!("gpqa-{}", item.item_id), &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if config.max_tokens + 2048 > 4096 {
        args.push("--max-seq".to_string());
        args.push((config.max_tokens + 2048).to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::Gpqa),
                "gpqa_zero_shot_native",
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                base_metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.extend(base_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    let predicted = extract_answer_letter(&stdout);
    metrics.insert("predicted_label".to_string(), json!(predicted));
    let exact_match = predicted.as_deref() == Some(item.answer_label.as_str());
    metrics.insert(
        "exact_match".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );
    metrics.insert(
        "accuracy".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

pub(crate) fn extract_answer_letter(stdout: &str) -> Option<String> {
    let trimmed = stdout.trim_start();
    if let Some(first) = trimmed.chars().next() {
        if matches!(first.to_ascii_uppercase(), 'A' | 'B' | 'C' | 'D') {
            return Some(first.to_ascii_uppercase().to_string());
        }
    }
    trimmed
        .split(|c: char| !c.is_ascii_alphanumeric())
        .find_map(|token| {
            let mut chars = token.chars();
            let c = chars.next()?;
            if chars.next().is_none() && matches!(c.to_ascii_uppercase(), 'A' | 'B' | 'C' | 'D') {
                Some(c.to_ascii_uppercase().to_string())
            } else {
                None
            }
        })
}

pub(crate) fn run_dflash_spec_demo_anchor(
    battery: BatteryId,
    case_id: &str,
    ar_baseline: bool,
    prompt_path: &str,
    extra_args: &[&str],
    mut extra_metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_ref = prompt(prompt_path);
    if !Path::new(&model).exists() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(draft) = config.draft.as_deref() else {
        if ar_baseline {
            return run_examples_run_anchor_with_prompt_for_model(
                battery,
                case_id,
                prompt_path,
                config,
                ctx,
                model,
            );
        }
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("examples executor requires --draft for DFlash anchor".to_string()),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    if !Path::new(draft).exists() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("examples executor requires --draft to be a local filesystem path".to_string()),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_dflash_spec_demo_bin() else {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("dflash_spec_demo example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example dflash_spec_demo`".to_string()),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let evidence_dir = runtime_evidence_dir(config, case_id, &model);
    let mut args = vec![
        "--target".to_string(),
        model.clone(),
        "--draft".to_string(),
        draft.to_string(),
        "--prompt-file".to_string(),
        prompt_path.to_string(),
        "--max".to_string(),
        config.max_tokens.to_string(),
        "--ctx".to_string(),
        "2048".to_string(),
        "--kv-mode".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--no-adaptive-b".to_string(),
        "--no-chatml".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if ar_baseline {
        args.push("--ar-baseline".to_string());
    }
    args.extend(extra_args.iter().map(|arg| (*arg).to_string()));
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let mut command = Command::new(&bin);
    command.args(&args);
    if config.profile == ProfileMode::Passive && !ar_baseline {
        command.env("HIPFIRE_PROFILE", "1");
        command.env("HIPFIRE_PROFILE_CYCLES", "1");
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("spawn dflash_spec_demo: {err}")),
                BTreeMap::from([
                    ("executor".to_string(), json!("examples")),
                    ("command".to_string(), json!(command_display)),
                ]),
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.append(&mut extra_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("examples"));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("ar_baseline".to_string(), json!(ar_baseline));
    metrics.insert(
        "profiling_requested".to_string(),
        json!(config.profile == ProfileMode::Passive && !ar_baseline),
    );
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_tau").cloned() {
        metrics.entry("tau".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_accept_rate").cloned() {
        metrics.entry("accept_rate".to_string()).or_insert(v);
    }
    if let Some(attractor) = parse_spec_decode_token_attractor_metrics(&stderr) {
        metrics.extend(attractor.metrics);
    }
    let attractor_fail = metrics
        .get("token_attractor_fail")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if output.status.success() && metrics.contains_key("decode_tok_s") && !attractor_fail {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(if output.status.success() {
                if attractor_fail {
                    "dflash_spec_demo token-attractor detector failed".to_string()
                } else {
                    "dflash_spec_demo did not emit BENCH METRICS".to_string()
                }
            } else {
                format!("dflash_spec_demo exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    }
}

pub(crate) struct TokenAttractorMetrics {
    pub(crate) metrics: BTreeMap<String, Value>,
}

pub(crate) fn parse_spec_decode_token_attractor_metrics(
    output: &str,
) -> Option<TokenAttractorMetrics> {
    let tokens = parse_spec_decode_tokens(output)?;
    if tokens.is_empty() {
        let metrics = BTreeMap::from([
            ("token_count".to_string(), json!(0.0)),
            ("token_attractor_fail".to_string(), json!(true)),
            ("token_attractor_reason".to_string(), json!("zero_tokens")),
        ]);
        return Some(TokenAttractorMetrics { metrics });
    }
    let eot = [248044u32, 248046u32];
    let trimmed: Vec<u32> = tokens
        .iter()
        .copied()
        .take_while(|token| !eot.contains(token))
        .take(128)
        .collect();
    if trimmed.len() < 16 {
        let metrics = BTreeMap::from([
            ("token_count".to_string(), json!(trimmed.len() as f64)),
            ("token_attractor_fail".to_string(), json!(false)),
            ("token_attractor_reason".to_string(), json!("short_clean")),
        ]);
        return Some(TokenAttractorMetrics { metrics });
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for token in &trimmed {
        *counts.entry(*token).or_insert(0) += 1;
    }
    let max_freq = counts.values().copied().max().unwrap_or(0) as f64 / trimmed.len() as f64;
    let unique_ratio = counts.len() as f64 / trimmed.len() as f64;
    let fail = max_freq > 0.40 || unique_ratio < 0.30;
    let metrics = BTreeMap::from([
        ("token_count".to_string(), json!(trimmed.len() as f64)),
        ("unique_token_count".to_string(), json!(counts.len() as f64)),
        ("max_token_frequency".to_string(), json!(max_freq)),
        ("unique_token_ratio".to_string(), json!(unique_ratio)),
        ("token_attractor_fail".to_string(), json!(fail)),
        (
            "token_attractor_reason".to_string(),
            json!(if fail { "attractor" } else { "ok" }),
        ),
    ]);
    Some(TokenAttractorMetrics { metrics })
}

pub(crate) fn parse_spec_decode_tokens(output: &str) -> Option<Vec<u32>> {
    for prefix in ["DFlash tokens:", "AR tokens:"] {
        let Some(start) = output.find(prefix) else {
            continue;
        };
        let rest = &output[start + prefix.len()..];
        let open = rest.find('[')?;
        let close = rest[open + 1..].find(']')? + open + 1;
        let body = &rest[open + 1..close];
        let tokens = body
            .split(',')
            .filter_map(|part| part.trim().parse::<u32>().ok())
            .collect::<Vec<_>>();
        return Some(tokens);
    }
    None
}

pub(crate) fn run_examples_run_anchor_with_prompt(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
) -> EvalResult {
    run_examples_run_anchor_with_prompt_for_model(
        battery,
        case_id,
        prompt_path,
        config,
        ctx,
        config.model.clone(),
    )
}

pub(crate) fn run_examples_run_anchor_with_prompt_for_model(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_ref = prompt(prompt_path);
    run_examples_run_anchor_with_prompt_ref_for_model(
        battery,
        case_id,
        prompt_path,
        prompt_ref,
        config,
        ctx,
        model,
        None,
        BTreeMap::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_examples_run_anchor_with_prompt_ref_for_model(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    prompt_ref: Option<PromptRef>,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
    max_seq: Option<usize>,
    extra_metrics: BTreeMap<String, Value>,
) -> EvalResult {
    if !Path::new(&model).exists() {
        let mut metrics = BTreeMap::from([("executor".to_string(), json!("examples"))]);
        metrics.extend(extra_metrics);
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        let mut metrics = BTreeMap::from([("executor".to_string(), json!("examples"))]);
        metrics.extend(extra_metrics);
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    let evidence_dir = runtime_evidence_dir(config, case_id, &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_path.to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    let computed_max_seq = max_seq.unwrap_or_else(|| {
        if config.max_tokens + 2048 > 4096 {
            config.max_tokens + 2048
        } else {
            4096
        }
    });
    if computed_max_seq > 4096 {
        args.push("--max-seq".to_string());
        args.push(computed_max_seq.to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            let mut metrics = BTreeMap::from([
                ("executor".to_string(), json!("examples")),
                ("command".to_string(), json!(command_display)),
            ]);
            metrics.extend(extra_metrics);
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("examples"));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("ar_baseline".to_string(), json!(true));
    metrics.insert("max_seq".to_string(), json!(computed_max_seq));
    metrics.extend(extra_metrics);
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_tau").cloned() {
        metrics.entry("tau".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_accept_rate").cloned() {
        metrics.entry("accept_rate".to_string()).or_insert(v);
    }
    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    }
}

pub(crate) fn run_direct_session_reset_recall(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> EvalResult {
    let case_id = "multi_turn_reset_recall";
    let prompt_path = "benchmarks/prompts/trains-meet.txt";
    let prompt_ref = prompt(prompt_path);
    let model = config.model.clone();
    let base_metrics = || BTreeMap::from([("executor".to_string(), json!("direct"))]);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Smoke,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "direct session executor requires --model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics(),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Smoke,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics(),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let evidence_dir = runtime_evidence_dir(config, case_id, &model);
    let mut args = vec![
        model.clone(),
        "--session-reset-smoke".to_string(),
        "--prompt-file".to_string(),
        prompt_path.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            let mut metrics = base_metrics();
            metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Smoke,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("spawn direct session executor: {err}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let report_path = evidence_dir.join("session_reset.json");
    let report = fs::read_to_string(&report_path)
        .ok()
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .or_else(|| serde_json::from_str::<Value>(&stdout).ok());

    let mut metrics = base_metrics();
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(value) = report.as_ref().and_then(|v| v.get("metrics")) {
        if let Some(obj) = value.as_object() {
            for (key, value) in obj {
                metrics.insert(key.clone(), value.clone());
            }
        }
    }

    let report_status = report
        .as_ref()
        .and_then(|v| v.get("status"))
        .and_then(Value::as_str);
    let pass = output.status.success() && report_status == Some("pass");
    row_for_model(
        BatteryId::Smoke,
        None,
        case_id,
        None,
        if pass {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        },
        if pass {
            None
        } else if output.status.success() {
            Some("direct session executor did not emit a passing session reset report".to_string())
        } else {
            Some(format!(
                "direct session executor exited with {}",
                output.status
            ))
        },
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}
