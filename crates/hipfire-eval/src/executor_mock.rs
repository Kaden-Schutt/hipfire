// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mock executor: deterministic stand-in rows for batteries/suites when no
//! real model run is requested (dry-run, CI without a GPU, schema tests).
//!
//! Pure functions of (model, salt) via `hipfire_hash::stable_score`, so output
//! is reproducible. Extracted verbatim from the former `hipfire-eval/src/lib.rs`
//! monolith (no behavior change).

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::*;

pub(crate) fn mock_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    let rows = match battery {
        BatteryId::Smoke => vec![
            mock_pass_row(
                battery,
                None,
                "load_metadata",
                None,
                config,
                ctx,
                None,
                BTreeMap::from([
                    ("load_metadata_ok".to_string(), json!(1.0)),
                    (
                        "mock_latency_ms".to_string(),
                        json!(mock_metric(&config.model, "load", 10.0, 3.0)),
                    ),
                ]),
                config.model.clone(),
            ),
            mock_pass_row(
                battery,
                None,
                "finite_greedy_decode",
                None,
                config,
                ctx,
                prompt("benchmarks/prompts/qwen2_smoke.txt"),
                BTreeMap::from([
                    ("finite_tokens".to_string(), json!(1.0)),
                    (
                        "generated_tokens".to_string(),
                        json!(config.max_tokens.min(16) as f64),
                    ),
                ]),
                config.model.clone(),
            ),
        ],
        BatteryId::Coherence => mock_metric_family_rows(
            battery,
            "runtime_detector_canary",
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            config,
            ctx,
            |_model| {
                BTreeMap::from([
                    ("hard_fails".to_string(), json!(0.0)),
                    ("soft_warns".to_string(), json!(0.0)),
                    ("detector_count".to_string(), json!(8.0)),
                    (
                        "detector_profile".to_string(),
                        json!("default_runtime_coherence"),
                    ),
                    ("runtime_path".to_string(), json!("daemon_jsonl_mock")),
                ])
            },
        ),
        BatteryId::Quality => mock_metric_family_rows(
            battery,
            "kld_reference_slice",
            prompt("benchmarks/quality-baselines/harness/canary.md"),
            config,
            ctx,
            |model| {
                BTreeMap::from([
                    (
                        "mean_kld".to_string(),
                        json!(mock_metric(model, "mean_kld", 0.015, 0.02)),
                    ),
                    (
                        "p99_kld".to_string(),
                        json!(mock_metric(model, "p99_kld", 0.04, 0.05)),
                    ),
                    (
                        "ppl".to_string(),
                        json!(mock_metric(model, "ppl", 5.0, 0.5)),
                    ),
                    (
                        "argmax_match_rate".to_string(),
                        json!(mock_metric(model, "argmax", 0.93, 0.05)),
                    ),
                ])
            },
        ),
        BatteryId::Speed => mock_metric_family_rows(
            battery,
            "pp32_pp128_ttft_decode",
            prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
            config,
            ctx,
            |model| {
                BTreeMap::from([
                    (
                        "pp32_ms".to_string(),
                        json!(mock_metric(model, "pp32", 7.0, 2.0)),
                    ),
                    (
                        "pp128_ms".to_string(),
                        json!(mock_metric(model, "pp128", 22.0, 6.0)),
                    ),
                    (
                        "ttft_ms".to_string(),
                        json!(mock_metric(model, "ttft", 30.0, 8.0)),
                    ),
                    (
                        "tok_s".to_string(),
                        json!(mock_metric(model, "tok_s", 110.0, 30.0)),
                    ),
                ])
            },
        ),
        BatteryId::Dflash => mock_metric_family_rows(
            battery,
            "dflash_anchor",
            prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
            config,
            ctx,
            |model| {
                BTreeMap::from([
                    (
                        "ar_tok_s".to_string(),
                        json!(mock_metric(model, "ar_tok_s", 90.0, 20.0)),
                    ),
                    (
                        "dflash_tok_s".to_string(),
                        json!(if config.dflash == DflashMode::Off {
                            0.0
                        } else {
                            mock_metric(model, "dflash_tok_s", 130.0, 35.0)
                        }),
                    ),
                    (
                        "accept_rate".to_string(),
                        json!(if config.dflash == DflashMode::Off {
                            0.0
                        } else {
                            mock_metric(model, "accept_rate", 0.45, 0.2)
                        }),
                    ),
                    (
                        "tau".to_string(),
                        json!(if config.dflash == DflashMode::Off {
                            1.0
                        } else {
                            mock_metric(model, "tau", 2.0, 1.5)
                        }),
                    ),
                ])
            },
        ),
        BatteryId::Agentic => mock_metric_family_rows(
            battery,
            "agentic_tool_call_shape",
            prompt("benchmarks/prompts/agentic_user_read.txt"),
            config,
            ctx,
            |_model| {
                BTreeMap::from([
                    ("hard_fails".to_string(), json!(0.0)),
                    ("soft_warns".to_string(), json!(0.0)),
                    ("detector_count".to_string(), json!(9.0)),
                    ("generated_tokens".to_string(), json!(64.0)),
                    ("structured_probe".to_string(), json!("agentic_tool_call")),
                    ("runtime_path".to_string(), json!("daemon_jsonl_mock")),
                ])
            },
        ),
        BatteryId::Runtime => runtime_cases()
            .iter()
            .map(|case| {
                pass_row(
                    BatteryId::Runtime,
                    None,
                    case.label,
                    None,
                    config,
                    ctx,
                    None,
                    BTreeMap::from([
                        ("runtime_evidence_case".to_string(), json!(case.label)),
                        ("script".to_string(), json!(case.script)),
                        ("executor".to_string(), json!("mock")),
                        ("runtime_path".to_string(), json!("server_runtime_mock")),
                    ]),
                )
            })
            .collect(),
        BatteryId::Barrage => mock_barrage_rows(config, ctx, datasets),
        _ => return None,
    };
    Some(rows)
}

fn mock_metric_family_rows<F>(
    battery: BatteryId,
    case_id: &str,
    prompt: Option<PromptRef>,
    config: &EvalConfig,
    ctx: &EvalContext,
    build_metrics: F,
) -> Vec<EvalResult>
where
    F: Fn(&str) -> BTreeMap<String, Value>,
{
    let mut rows = Vec::new();
    rows.push(mock_pass_row(
        battery,
        None,
        case_id,
        None,
        config,
        ctx,
        prompt.clone(),
        build_metrics(&config.model),
        config.model.clone(),
    ));
    for model in [config.baseline.as_ref(), config.reference.as_ref()]
        .into_iter()
        .flatten()
    {
        rows.push(mock_pass_row(
            battery,
            None,
            case_id,
            None,
            config,
            ctx,
            prompt.clone(),
            build_metrics(model),
            model.clone(),
        ));
    }
    rows
}

fn mock_barrage_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    let mut rows = Vec::new();
    for d in datasets {
        if d.status != EvalStatus::Pass {
            continue;
        }
        match d.suite {
            SuiteId::Gpqa => {
                let Ok(items) =
                    gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids)
                else {
                    continue;
                };
                for item in items {
                    let prompt_ref = PromptRef::from_content(
                        format!("dataset:gpqa:{}", item.item_id),
                        item.prompt.as_bytes(),
                    );
                    let models = std::iter::once(&config.model)
                        .chain(config.baseline.iter())
                        .chain(config.reference.iter());
                    for model in models {
                        let mut metrics = BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                json!(mock_bool_metric(model, &item.item_id)),
                            ),
                            (
                                "exact_match".to_string(),
                                json!(mock_bool_metric(model, &item.correct_answer)),
                            ),
                            ("answer_label".to_string(), json!(item.answer_label.clone())),
                            (
                                "answer_hash".to_string(),
                                json!(stable_hash_bytes(item.correct_answer.as_bytes())),
                            ),
                        ]);
                        add_dataset_provenance_metrics(&mut metrics, d);
                        rows.push(mock_pass_row(
                            BatteryId::Barrage,
                            Some(SuiteId::Gpqa),
                            "gpqa_zero_shot_native",
                            Some(item.item_id.clone()),
                            config,
                            ctx,
                            Some(prompt_ref.clone()),
                            metrics,
                            model.clone(),
                        ));
                    }
                }
            }
            SuiteId::LmEvalMicro => {
                let Ok(items) = lm_eval_micro_materialized_items(&d.selected_item_ids) else {
                    continue;
                };
                for item in items {
                    let prompt_ref = PromptRef::from_content(
                        format!("dataset:lm_eval_micro:{}", item.item_id),
                        item.prompt.as_bytes(),
                    );
                    let models = std::iter::once(&config.model)
                        .chain(config.baseline.iter())
                        .chain(config.reference.iter());
                    for model in models {
                        let mut metrics = BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                json!(mock_bool_metric(model, &item.item_id)),
                            ),
                            (
                                "exact_match".to_string(),
                                json!(mock_bool_metric(model, &item.answer_hash)),
                            ),
                            (
                                "prompt_format".to_string(),
                                json!("lm_eval_micro_zero_shot_v1"),
                            ),
                            ("task".to_string(), json!(item.task.clone())),
                            ("answer_label".to_string(), json!(item.answer_label.clone())),
                            ("answer_hash".to_string(), json!(item.answer_hash.clone())),
                            ("choices_count".to_string(), json!(item.choices_count)),
                            ("scoring_mode".to_string(), json!("mock_exact_letter")),
                        ]);
                        add_dataset_provenance_metrics(&mut metrics, d);
                        rows.push(mock_pass_row(
                            BatteryId::Barrage,
                            Some(SuiteId::LmEvalMicro),
                            "lm_eval_micro_zero_shot_native",
                            Some(item.item_id.clone()),
                            config,
                            ctx,
                            Some(prompt_ref.clone()),
                            metrics,
                            model.clone(),
                        ));
                    }
                }
            }
            SuiteId::DeepSwe | SuiteId::SweBench => {
                let Ok(items) = builtin_barrage_materialized_items(d.suite, &d.selected_item_ids)
                else {
                    continue;
                };
                for item in items {
                    let prompt_ref = PromptRef::from_content(
                        format!("dataset:{}:{}", item.suite.as_str(), item.item_id),
                        item.prompt.as_bytes(),
                    );
                    let models = std::iter::once(&config.model)
                        .chain(config.baseline.iter())
                        .chain(config.reference.iter());
                    for model in models {
                        let mut metrics = BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                json!(mock_bool_metric(model, &item.item_id)),
                            ),
                            (
                                "exact_match".to_string(),
                                json!(mock_bool_metric(model, &item.answer_hash)),
                            ),
                            (
                                "prompt_format".to_string(),
                                json!(item.prompt_format.clone()),
                            ),
                            ("task".to_string(), json!(item.task.clone())),
                            ("answer_label".to_string(), json!(item.answer_label.clone())),
                            ("answer_hash".to_string(), json!(item.answer_hash.clone())),
                            ("choices_count".to_string(), json!(item.choices_count)),
                            ("scoring_mode".to_string(), json!(item.scoring_mode.clone())),
                        ]);
                        add_dataset_provenance_metrics(&mut metrics, d);
                        rows.push(mock_pass_row(
                            BatteryId::Barrage,
                            Some(item.suite),
                            "builtin_software_eval_native",
                            Some(item.item_id.clone()),
                            config,
                            ctx,
                            Some(prompt_ref.clone()),
                            metrics,
                            model.clone(),
                        ));
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

#[allow(clippy::too_many_arguments)]
fn mock_pass_row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    mut metrics: BTreeMap<String, Value>,
    model: String,
) -> EvalResult {
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("mock"));
    row_for_model(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Pass,
        Some("deterministic no-GPU mock executor".to_string()),
        metrics,
        config,
        ctx,
        prompt,
        0,
        model,
    )
}

fn mock_metric(model: &str, salt: &str, base: f64, spread: f64) -> f64 {
    base + (stable_score(&format!("{model}:{salt}")) * spread)
}

fn mock_bool_metric(model: &str, salt: &str) -> f64 {
    if stable_score(&format!("{model}:{salt}")) >= 0.5 {
        1.0
    } else {
        0.0
    }
}

fn stable_score(input: &str) -> f64 {
    hipfire_hash::stable_score(input)
}
