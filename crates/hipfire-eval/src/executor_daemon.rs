// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `daemon` eval executor: smoke/speed/profile battery rows produced by
//! driving the real `hipfire-daemon` binary over the JSONL adapter.
//!
//! Spawns a `DaemonEngine` (via `hipfire-daemon-adapter`), loads the model, runs
//! the battery prompts, and turns the responses + runtime evidence into
//! EvalResult rows (with skip/failure fallbacks). Extracted verbatim from the
//! former `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::collections::BTreeMap;

use serde_json::{json, Value};

use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};

use crate::*;

pub(crate) fn daemon_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    match battery {
        BatteryId::Smoke => Some(run_daemon_smoke_rows(config, ctx)),
        BatteryId::Speed => Some(run_daemon_speed_rows(config, ctx)),
        BatteryId::Profile => Some(run_daemon_profile_rows(config, ctx)),
        BatteryId::Coherence | BatteryId::Longctx | BatteryId::Agentic => {
            examples_battery_rows(battery, config, ctx, datasets)
        }
        _ => None,
    }
}

pub(crate) struct DaemonEvalSession {
    engine: hipfire_daemon_adapter::DaemonEngine,
    loaded: ModelLoadedResponse,
    worker_key_id: String,
    max_seq: usize,
}

pub(crate) async fn load_daemon_eval_session(
    config: &EvalConfig,
    bin: &Path,
    max_seq: usize,
) -> anyhow::Result<DaemonEvalSession> {
    let mut engine = hipfire_daemon_adapter::DaemonEngine::spawn(bin).await?;
    let loaded = engine
        .load(&config.model, daemon_model_load_params(config, max_seq))
        .await?;
    let worker_key_id = loaded.worker_key_id.clone();
    Ok(DaemonEvalSession {
        engine,
        loaded,
        worker_key_id,
        max_seq,
    })
}

pub(crate) fn run_daemon_shared_model_load_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    batteries: &[BatteryId],
) -> BTreeMap<BatteryId, Vec<EvalResult>> {
    let mut out = BTreeMap::new();
    if !Path::new(&config.model).exists() {
        for battery in batteries {
            let rows = match battery {
                BatteryId::Smoke => daemon_smoke_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires --model to be a local filesystem path",
                    "daemon executor requires --model to be a local filesystem path",
                ),
                BatteryId::Speed => daemon_speed_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires --model to be a local filesystem path",
                ),
                BatteryId::Profile => daemon_profile_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires --model to be a local filesystem path",
                ),
                _ => Vec::new(),
            };
            out.insert(*battery, rows);
        }
        return out;
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        for battery in batteries {
            let rows = match battery {
                BatteryId::Smoke => daemon_smoke_skip_rows(
                    config,
                    ctx,
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                ),
                BatteryId::Speed => daemon_speed_skip_rows(
                    config,
                    ctx,
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                ),
                BatteryId::Profile => daemon_profile_skip_rows(
                    config,
                    ctx,
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                ),
                _ => Vec::new(),
            };
            out.insert(*battery, rows);
        }
        return out;
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            for battery in batteries {
                let rows = match battery {
                    BatteryId::Smoke => daemon_smoke_skip_rows(
                        config,
                        ctx,
                        &format!("create daemon executor runtime: {err}"),
                        "daemon executor runtime creation failed before decode",
                    ),
                    BatteryId::Speed => daemon_speed_skip_rows(
                        config,
                        ctx,
                        &format!("create daemon executor runtime: {err}"),
                    ),
                    BatteryId::Profile => daemon_profile_skip_rows(
                        config,
                        ctx,
                        &format!("create daemon executor runtime: {err}"),
                    ),
                    _ => Vec::new(),
                };
                out.insert(*battery, rows);
            }
            return out;
        }
    };

    let max_seq = (config.max_tokens.max(50) + 2048).max(4096);
    match runtime.block_on(run_daemon_shared_model_load_rows_async(
        config, ctx, &bin, batteries, max_seq,
    )) {
        Ok(mut rows_by_battery) => {
            let elapsed_ms = elapsed_since_ms(started);
            for rows in rows_by_battery.values_mut() {
                for row in rows {
                    row.elapsed_ms = elapsed_ms;
                    row.metrics
                        .insert("shared_daemon_session".to_string(), json!(true));
                }
            }
            rows_by_battery
        }
        Err(err) => {
            for battery in batteries {
                let rows = match battery {
                    BatteryId::Smoke => daemon_shared_smoke_failure_rows(
                        config,
                        ctx,
                        &bin,
                        &format!("daemon-backed shared executor failed: {err}"),
                        elapsed_since_ms(started),
                    ),
                    BatteryId::Speed => daemon_shared_speed_failure_rows(
                        config,
                        ctx,
                        &bin,
                        &format!("daemon-backed shared executor failed: {err}"),
                        elapsed_since_ms(started),
                    ),
                    BatteryId::Profile => daemon_shared_profile_failure_rows(
                        config,
                        ctx,
                        &bin,
                        &format!("daemon-backed shared executor failed: {err}"),
                        elapsed_since_ms(started),
                    ),
                    _ => Vec::new(),
                };
                out.insert(*battery, rows);
            }
            out
        }
    }
}

pub(crate) async fn run_daemon_shared_model_load_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    batteries: &[BatteryId],
    max_seq: usize,
) -> anyhow::Result<BTreeMap<BatteryId, Vec<EvalResult>>> {
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    let mut out = BTreeMap::new();
    for battery in batteries {
        let rows = match battery {
            BatteryId::Smoke => {
                daemon_smoke_rows_with_session(config, ctx, bin, &mut session).await?
            }
            BatteryId::Speed => {
                daemon_speed_rows_with_session(config, ctx, bin, &mut session).await?
            }
            BatteryId::Profile => {
                daemon_profile_rows_with_session(config, ctx, bin, &mut session).await?
            }
            _ => Vec::new(),
        };
        out.insert(*battery, rows);
    }
    Ok(out)
}

pub(crate) fn daemon_model_load_params(config: &EvalConfig, max_seq: usize) -> ModelLoadParams {
    ModelLoadParams {
        max_seq: max_seq.min(u32::MAX as usize) as u32,
        kv_cache: config.kv_mode.clone(),
        dflash_mode: Some(config.dflash.as_str().to_string()),
        draft: config.draft.clone(),
        ..Default::default()
    }
}

pub(crate) fn daemon_generate_request(
    id: String,
    prompt_text: String,
    max_tokens: usize,
    worker_key_id: Option<String>,
    evidence_dir: Option<&Path>,
) -> GenerateTextRequest {
    let mut request = GenerateTextRequest::from_prompt(
        id,
        prompt_text,
        GenerationSamplingPolicy::greedy(max_tokens.min(u32::MAX as usize) as u32),
    )
    .with_worker_key_id(worker_key_id);
    request.evidence_dir = evidence_dir.map(|dir| dir.display().to_string());
    request
}

pub(crate) fn read_repo_prompt_text(prompt_path: &str) -> anyhow::Result<String> {
    let prompt_file = resolve_repo_path(prompt_path)
        .ok_or_else(|| anyhow::anyhow!("resolve {prompt_path} from repo root"))?;
    fs::read_to_string(&prompt_file).map_err(|e| anyhow::anyhow!("read {prompt_path}: {e}"))
}

pub(crate) fn daemon_smoke_skip_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    load_reason: &str,
    decode_reason: &str,
) -> Vec<EvalResult> {
    vec![
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "load_metadata",
            None,
            load_reason,
            config,
            ctx,
            None,
            BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        ),
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            decode_reason,
            config,
            ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        ),
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "multi_turn_reset_recall",
            None,
            load_reason,
            config,
            ctx,
            prompt("benchmarks/prompts/trains-meet.txt"),
            BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        ),
    ]
}

pub(crate) fn daemon_speed_skip_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    reason: &str,
) -> Vec<EvalResult> {
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    daemon_speed_cases()
        .iter()
        .map(|case| {
            skip_row_with_metrics(
                BatteryId::Speed,
                None,
                case.label,
                None,
                reason,
                config,
                ctx,
                prompt_ref.clone(),
                BTreeMap::from([
                    ("implemented".to_string(), json!(true)),
                    ("executor".to_string(), json!("daemon")),
                    ("suite".to_string(), json!("daemon_speed_anchor")),
                ]),
            )
        })
        .collect()
}

pub(crate) fn daemon_profile_expected_runtime_evidence_kinds() -> Value {
    json!([
        "performance",
        "memory",
        "launch_counts",
        "moe_router_histogram"
    ])
}

pub(crate) fn daemon_profile_base_metrics() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("daemon")),
        ("suite".to_string(), json!("daemon_profile_anchor")),
        ("profile_requested".to_string(), json!(true)),
        (
            "collection_scope".to_string(),
            json!("model_backed_daemon_anchor"),
        ),
        (
            "moe_router_histogram_expected_when_moe".to_string(),
            json!(true),
        ),
        (
            "expected_runtime_evidence_kinds".to_string(),
            daemon_profile_expected_runtime_evidence_kinds(),
        ),
    ])
}

pub(crate) fn daemon_profile_skip_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    reason: &str,
) -> Vec<EvalResult> {
    vec![skip_row_with_metrics(
        BatteryId::Profile,
        None,
        "model_profile_anchor",
        None,
        reason,
        config,
        ctx,
        prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
        daemon_profile_base_metrics(),
    )]
}

pub(crate) fn run_daemon_smoke_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return daemon_smoke_skip_rows(
            config,
            ctx,
            "daemon executor requires --model to be a local filesystem path",
            "daemon executor requires --model to be a local filesystem path",
        );
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return daemon_smoke_skip_rows(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        );
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return daemon_smoke_skip_rows(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
                "daemon executor runtime creation failed before decode",
            );
        }
    };

    match runtime.block_on(run_daemon_smoke_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => vec![
            row(
                BatteryId::Smoke,
                None,
                "load_metadata",
                None,
                EvalStatus::Fail,
                Some(format!("daemon-backed smoke executor failed: {err}")),
                BTreeMap::from([
                    ("executor".to_string(), json!("daemon")),
                    ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ]),
                config,
                ctx,
                None,
                elapsed_since_ms(started),
            ),
            row(
                BatteryId::Smoke,
                None,
                "finite_greedy_decode",
                None,
                EvalStatus::Skip,
                Some("daemon-backed load failed before decode".to_string()),
                BTreeMap::from([("executor".to_string(), json!("daemon"))]),
                config,
                ctx,
                prompt("benchmarks/prompts/qwen2_smoke.txt"),
                elapsed_since_ms(started),
            ),
            skip_row_with_metrics(
                BatteryId::Smoke,
                None,
                "multi_turn_reset_recall",
                None,
                "daemon-backed load failed before session reset/recall",
                config,
                ctx,
                prompt("benchmarks/prompts/trains-meet.txt"),
                BTreeMap::from([("executor".to_string(), json!("daemon"))]),
            ),
        ],
    }
}

pub(crate) fn run_daemon_speed_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return daemon_speed_skip_rows(
            config,
            ctx,
            "daemon executor requires --model to be a local filesystem path",
        );
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return daemon_speed_skip_rows(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        );
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return daemon_speed_skip_rows(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
            );
        }
    };

    match runtime.block_on(run_daemon_speed_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => daemon_speed_cases()
            .iter()
            .map(|case| {
                row(
                    BatteryId::Speed,
                    None,
                    case.label,
                    None,
                    EvalStatus::Fail,
                    Some(format!("daemon-backed speed executor failed: {err}")),
                    BTreeMap::from([
                        ("implemented".to_string(), json!(true)),
                        ("executor".to_string(), json!("daemon")),
                        ("suite".to_string(), json!("daemon_speed_anchor")),
                        ("daemon_bin".to_string(), json!(bin.display().to_string())),
                    ]),
                    config,
                    ctx,
                    prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
                    elapsed_since_ms(started),
                )
            })
            .collect(),
    }
}

pub(crate) fn run_daemon_profile_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return daemon_profile_skip_rows(
            config,
            ctx,
            "daemon executor requires --model to be a local filesystem path",
        );
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return daemon_profile_skip_rows(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        );
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return daemon_profile_skip_rows(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
            );
        }
    };

    match runtime.block_on(run_daemon_profile_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => daemon_shared_profile_failure_rows(
            config,
            ctx,
            &bin,
            &format!("daemon-backed profile executor failed: {err}"),
            elapsed_since_ms(started),
        ),
    }
}

pub(crate) fn daemon_shared_smoke_failure_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    vec![
        row(
            BatteryId::Smoke,
            None,
            "load_metadata",
            None,
            EvalStatus::Fail,
            Some(reason.to_string()),
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_daemon_session".to_string(), json!(true)),
                ("daemon_bin".to_string(), json!(bin.display().to_string())),
            ]),
            config,
            ctx,
            None,
            elapsed_ms,
        ),
        row(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            EvalStatus::Skip,
            Some("daemon-backed shared load failed before decode".to_string()),
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_daemon_session".to_string(), json!(true)),
            ]),
            config,
            ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            elapsed_ms,
        ),
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "multi_turn_reset_recall",
            None,
            "daemon-backed shared load failed before session reset/recall",
            config,
            ctx,
            prompt("benchmarks/prompts/trains-meet.txt"),
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_daemon_session".to_string(), json!(true)),
            ]),
        ),
    ]
}

pub(crate) fn daemon_shared_speed_failure_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    daemon_speed_cases()
        .iter()
        .map(|case| {
            row(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Fail,
                Some(reason.to_string()),
                BTreeMap::from([
                    ("implemented".to_string(), json!(true)),
                    ("executor".to_string(), json!("daemon")),
                    ("suite".to_string(), json!("daemon_speed_anchor")),
                    ("shared_daemon_session".to_string(), json!(true)),
                    ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ]),
                config,
                ctx,
                prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
                elapsed_ms,
            )
        })
        .collect()
}

pub(crate) fn daemon_shared_profile_failure_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    let mut metrics = daemon_profile_base_metrics();
    metrics.insert("shared_daemon_session".to_string(), json!(true));
    metrics.insert("daemon_bin".to_string(), json!(bin.display().to_string()));
    vec![row(
        BatteryId::Profile,
        None,
        "model_profile_anchor",
        None,
        EvalStatus::Fail,
        Some(reason.to_string()),
        metrics,
        config,
        ctx,
        prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
        elapsed_ms,
    )]
}

pub(crate) async fn run_daemon_smoke_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    daemon_smoke_rows_with_session(config, ctx, bin, &mut session).await
}

pub(crate) async fn daemon_smoke_rows_with_session(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
) -> anyhow::Result<Vec<EvalResult>> {
    let worker_key_id = session.worker_key_id.clone();
    let prompt_path = "benchmarks/prompts/qwen2_smoke.txt";
    let prompt_text = read_repo_prompt_text(prompt_path)?;
    let request = daemon_generate_request(
        "eval-smoke-greedy".to_string(),
        prompt_text,
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (text, done) = session.engine.generate(request).await?;
    let finite = !text.is_empty() && !text.contains('\u{fffd}');
    let decode_status = if finite {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let decode_reason =
        (!finite).then(|| "daemon returned empty or replacement-character output".to_string());

    let session_prompt_path = "benchmarks/prompts/trains-meet.txt";
    let session_prompt_text = read_repo_prompt_text(session_prompt_path)?;
    session.engine.reset().await?;
    let first_session_request = daemon_generate_request(
        "eval-smoke-session-fresh".to_string(),
        session_prompt_text.clone(),
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (first_session_text, first_session_done) =
        session.engine.generate(first_session_request).await?;
    let distractor_request = daemon_generate_request(
        "eval-smoke-session-distractor".to_string(),
        "Remember this unrelated code word for the next turn: orchid. Reply with only OK."
            .to_string(),
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (distractor_text, distractor_done) = session.engine.generate(distractor_request).await?;
    session.engine.reset().await?;
    let second_session_request = daemon_generate_request(
        "eval-smoke-session-reset".to_string(),
        session_prompt_text,
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (second_session_text, second_session_done) =
        session.engine.generate(second_session_request).await?;
    let session_finite = !first_session_text.is_empty()
        && !second_session_text.is_empty()
        && !first_session_text.contains('\u{fffd}')
        && !second_session_text.contains('\u{fffd}');
    let session_match = first_session_text == second_session_text;
    let session_status = if session_finite && session_match {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let session_reason = if !session_finite {
        Some(
            "daemon session reset smoke returned empty or replacement-character output".to_string(),
        )
    } else if !session_match {
        Some("daemon repeated greedy session request produced different output".to_string())
    } else {
        None
    };

    Ok(vec![
        row(
            BatteryId::Smoke,
            None,
            "load_metadata",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ("shared_model_loads".to_string(), json!(1)),
                ("worker_key_id".to_string(), json!(worker_key_id.clone())),
                ("arch".to_string(), json!(session.loaded.arch)),
                ("dim".to_string(), json!(session.loaded.dim)),
                ("layers".to_string(), json!(session.loaded.layers)),
                ("vocab".to_string(), json!(session.loaded.vocab)),
                ("max_seq".to_string(), json!(session.max_seq)),
            ]),
            config,
            ctx,
            None,
            0,
        ),
        row(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            decode_status,
            decode_reason,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_model_loads".to_string(), json!(1)),
                ("worker_key_id".to_string(), json!(worker_key_id.clone())),
                ("tokens".to_string(), json!(done.tokens)),
                ("text_bytes".to_string(), json!(text.len())),
                ("tok_s".to_string(), json!(done.tok_s)),
                ("ttft_ms".to_string(), json!(done.ttft_ms)),
                ("max_tokens".to_string(), json!(config.max_tokens)),
            ]),
            config,
            ctx,
            prompt(prompt_path),
            0,
        ),
        row(
            BatteryId::Smoke,
            None,
            "multi_turn_reset_recall",
            None,
            session_status,
            session_reason,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("implemented".to_string(), json!(true)),
                ("shared_model_loads".to_string(), json!(1)),
                ("worker_key_id".to_string(), json!(worker_key_id)),
                ("reset_count".to_string(), json!(2)),
                ("kv_reset".to_string(), json!(true)),
                ("dn_state_reset".to_string(), json!(true)),
                ("session_turns".to_string(), json!(3)),
                ("first_tokens".to_string(), json!(first_session_done.tokens)),
                (
                    "distractor_tokens".to_string(),
                    json!(distractor_done.tokens),
                ),
                (
                    "second_tokens".to_string(),
                    json!(second_session_done.tokens),
                ),
                (
                    "first_text_hash".to_string(),
                    json!(stable_hash_bytes(first_session_text.as_bytes())),
                ),
                (
                    "second_text_hash".to_string(),
                    json!(stable_hash_bytes(second_session_text.as_bytes())),
                ),
                (
                    "distractor_text_hash".to_string(),
                    json!(stable_hash_bytes(distractor_text.as_bytes())),
                ),
                ("outputs_match".to_string(), json!(session_match)),
                ("max_tokens".to_string(), json!(config.max_tokens)),
            ]),
            config,
            ctx,
            prompt(session_prompt_path),
            0,
        ),
    ])
}

pub(crate) async fn run_daemon_speed_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    daemon_speed_rows_with_session(config, ctx, bin, &mut session).await
}

pub(crate) async fn run_daemon_profile_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens.max(50) + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    daemon_profile_rows_with_session(config, ctx, bin, &mut session).await
}

pub(crate) async fn daemon_speed_rows_with_session(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
) -> anyhow::Result<Vec<EvalResult>> {
    let worker_key_id = session.worker_key_id.clone();
    let prompt_path = "benchmarks/prompts/lru_cache_single_blank.txt";
    let prompt_text = read_repo_prompt_text(prompt_path)?;
    let max_tokens = config.max_tokens.max(50);

    let mut rows = Vec::new();
    for case in daemon_speed_cases() {
        session.engine.reset().await?;
        let evidence_dir = runtime_evidence_dir(
            config,
            &format!("daemon-speed-{}", case.label),
            &config.model,
        );
        let request = daemon_generate_request(
            format!("eval-speed-{}", case.label),
            prompt_text.clone(),
            max_tokens,
            Some(worker_key_id.clone()),
            Some(&evidence_dir),
        );
        let (text, done) = session.engine.generate(request).await?;
        let has_timing = done.prefill_tok_s.is_some() && done.decode_tok_s.is_some();
        let finite = !text.is_empty() && !text.contains('\u{fffd}') && done.tokens > 0;
        let status = if finite && has_timing {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        };
        let reason = if !finite {
            Some(
                "daemon speed anchor returned empty, zero-token, or replacement-character output"
                    .to_string(),
            )
        } else if !has_timing {
            Some("daemon speed anchor did not emit prefill/decode timing metrics".to_string())
        } else {
            None
        };
        let mut metrics = BTreeMap::from([
            ("implemented".to_string(), json!(true)),
            ("executor".to_string(), json!("daemon")),
            ("suite".to_string(), json!("daemon_speed_anchor")),
            ("shared_model_loads".to_string(), json!(1)),
            ("worker_key_id".to_string(), json!(worker_key_id.clone())),
            ("daemon_bin".to_string(), json!(bin.display().to_string())),
            ("max_tokens".to_string(), json!(max_tokens)),
            ("tokens".to_string(), json!(done.tokens)),
            ("text_bytes".to_string(), json!(text.len())),
            (
                "runtime_evidence_dir".to_string(),
                json!(evidence_dir.display().to_string()),
            ),
        ]);
        if let Some(value) = done.tok_s {
            metrics.insert("tok_s".to_string(), json!(value));
        }
        if let Some(value) = done.prefill_tokens {
            metrics.insert("prefill_tokens".to_string(), json!(value));
        }
        if let Some(value) = done.prefill_ms {
            metrics.insert("prefill_ms".to_string(), json!(value));
        }
        if let Some(value) = done.prefill_tok_s {
            metrics.insert("prefill_tok_s".to_string(), json!(value));
        }
        if let Some(value) = done.decode_tok_s {
            metrics.insert("decode_tok_s".to_string(), json!(value));
            metrics
                .entry("gen_tok_s".to_string())
                .or_insert(json!(value));
        }
        if let Some(value) = done.ttft_ms {
            metrics.insert("ttft_ms".to_string(), json!(value));
        }

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
            prompt(prompt_path),
            0,
            config.model.clone(),
        ));
    }

    Ok(rows)
}

pub(crate) async fn daemon_profile_rows_with_session(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
) -> anyhow::Result<Vec<EvalResult>> {
    let worker_key_id = session.worker_key_id.clone();
    let prompt_path = "benchmarks/prompts/dflash_resident_smoke.txt";
    let prompt_text = read_repo_prompt_text(prompt_path)?;
    let max_tokens = config.max_tokens.max(50);
    let evidence_dir =
        runtime_evidence_dir(config, "daemon-profile-model_profile_anchor", &config.model);
    session.engine.reset().await?;
    let request = daemon_generate_request(
        "eval-profile-model_profile_anchor".to_string(),
        prompt_text,
        max_tokens,
        Some(worker_key_id.clone()),
        Some(&evidence_dir),
    );
    let (text, done) = session.engine.generate(request).await?;
    let finite = !text.is_empty() && !text.contains('\u{fffd}') && done.tokens > 0;
    let has_timing = done.prefill_tok_s.is_some() && done.decode_tok_s.is_some();
    let status = if finite && has_timing {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let reason = if !finite {
        Some(
            "daemon profile anchor returned empty, zero-token, or replacement-character output"
                .to_string(),
        )
    } else if !has_timing {
        Some("daemon profile anchor did not emit prefill/decode timing metrics".to_string())
    } else {
        None
    };
    let mut metrics = daemon_profile_base_metrics();
    metrics.extend([
        ("shared_model_loads".to_string(), json!(1)),
        ("worker_key_id".to_string(), json!(worker_key_id)),
        ("daemon_bin".to_string(), json!(bin.display().to_string())),
        ("max_tokens".to_string(), json!(max_tokens)),
        ("tokens".to_string(), json!(done.tokens)),
        ("text_bytes".to_string(), json!(text.len())),
        (
            "runtime_evidence_dir".to_string(),
            json!(evidence_dir.display().to_string()),
        ),
    ]);
    if let Some(value) = done.tok_s {
        metrics.insert("tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_tokens {
        metrics.insert("prefill_tokens".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_ms {
        metrics.insert("prefill_ms".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_tok_s {
        metrics.insert("prefill_tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.decode_tok_s {
        metrics.insert("decode_tok_s".to_string(), json!(value));
        metrics
            .entry("gen_tok_s".to_string())
            .or_insert(json!(value));
    }
    if let Some(value) = done.ttft_ms {
        metrics.insert("ttft_ms".to_string(), json!(value));
    }

    Ok(vec![row(
        BatteryId::Profile,
        None,
        "model_profile_anchor",
        None,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt(prompt_path),
        0,
    )])
}
