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
use hipfire_model::find_model_in;

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
        BatteryId::Vision => Some(run_daemon_vision_rows(config, ctx)),
        BatteryId::Quality => Some(run_daemon_quality_rows(config, ctx)),
        _ => None,
    }
}

// ── Quality battery (resident KLD scoring; replaces the eval_hipfire example) ──
//
// Drives the daemon `kld_eval` op: builds a KLD reference from the full-precision
// anchor (`--reference`) over the committed wikitext slice, then scores each
// candidate model against that reference — all through ONE resident forward +
// the shared `hipfire-kld` core (the drift the standalone two-binary path caused
// is impossible by construction). An existing HFKREF (`--kldref`) short-circuits
// the build. The daemon op is qwen3.5-only; non-qwen35 models surface the daemon's
// own error as a Fail row.

const KLD_CORPUS_SLICE: &str = "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt";

/// Minimal load params for KLD scoring: only `max_seq` + `kv_cache`. KLD runs the
/// plain scoring forward, so DFlash / draft / CASK (generate-time spec-decode
/// machinery in `daemon_model_load_params`) must NOT be enabled — they change the
/// forward path and are irrelevant to reference build / scoring.
fn daemon_kld_load_params(config: &EvalConfig, max_seq: usize) -> ModelLoadParams {
    ModelLoadParams {
        max_seq: max_seq.min(u32::MAX as usize) as u32,
        kv_cache: config.kv_mode.clone(),
        ..Default::default()
    }
}

pub(crate) fn run_daemon_quality_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return vec![daemon_quality_skip_row(
            config,
            ctx,
            &config.model,
            "daemon quality executor requires the model to resolve to a local filesystem path",
        )];
    }
    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return vec![daemon_quality_skip_row(
            config,
            ctx,
            &config.model,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        )];
    };
    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return vec![daemon_quality_skip_row(
                config,
                ctx,
                &config.model,
                &format!("create daemon executor runtime: {err}"),
            )]
        }
    };
    match runtime.block_on(run_daemon_quality_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for r in &mut rows {
                if r.elapsed_ms == 0 {
                    r.elapsed_ms = elapsed_ms;
                }
            }
            rows
        }
        Err(err) => vec![daemon_quality_fail_row(
            config,
            ctx,
            &config.model,
            &format!("daemon-backed quality executor failed: {err}"),
            elapsed_since_ms(started),
        )],
    }
}

pub(crate) async fn run_daemon_quality_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    use hipfire_daemon_protocol::{KldEvalMode, KldEvalRequest};

    enum ReferencePlan {
        Existing(String),
        Build {
            ref_model: String,
            corpus: String,
            ref_out: std::path::PathBuf,
        },
    }

    let max_seq = 4096; // n_ctx=2048 chunks + headroom
    let evidence_dir = runtime_evidence_dir(config, "kld_reference_slice", &config.model);
    let _ = fs::create_dir_all(&evidence_dir);

    // ── Resolve the reference: an existing HFKREF, else build one resident from
    // the full-precision anchor (`--reference`). ──
    let ref_plan = if let Some(p) = config.kldref.as_ref().filter(|p| p.exists()) {
        ReferencePlan::Existing(p.display().to_string())
    } else if let Some(ref_model) = config.reference.as_ref() {
        if !Path::new(ref_model).exists() {
            return Ok(vec![daemon_quality_skip_row(
                config,
                ctx,
                &config.model,
                &format!("--reference model {ref_model} is not a local filesystem path"),
            )]);
        }
        let Some(corpus) = resolve_repo_path(KLD_CORPUS_SLICE).map(|p| p.display().to_string())
        else {
            return Ok(vec![daemon_quality_skip_row(
                config,
                ctx,
                &config.model,
                &format!("KLD corpus slice not found ({KLD_CORPUS_SLICE})"),
            )]);
        };
        let ref_out = evidence_dir.join(format!("{}.kldref", model_artifact_stem(ref_model)));
        ReferencePlan::Build {
            ref_model: ref_model.clone(),
            corpus,
            ref_out,
        }
    } else {
        return Ok(vec![daemon_quality_skip_row(
            config,
            ctx,
            &config.model,
            "no KLD reference: pass --reference <full-precision model> to build one, or --kldref <ref.kldref>",
        )]);
    };

    let mut engine = hipfire_daemon_adapter::DaemonEngine::spawn(bin).await?;
    let ref_path = match ref_plan {
        ReferencePlan::Existing(path) => path,
        ReferencePlan::Build {
            ref_model,
            corpus,
            ref_out,
        } => {
            engine
                .load(&ref_model, daemon_kld_load_params(config, max_seq))
                .await?;
            let resp = engine
                .kld_eval(
                    KldEvalRequest {
                        mode: KldEvalMode::BuildRef,
                        corpus: Some(corpus),
                        ref_path: Some(ref_out.display().to_string()),
                        output: None,
                        max_chunks: config.quality_max_chunks,
                        n_ctx: None,
                        config: None,
                        capture_hidden_layers: false,
                        dump_logits: false,
                    },
                    |_| {},
                )
                .await?;
            resp.ref_output
                .unwrap_or_else(|| ref_out.display().to_string())
        }
    };

    // ── Score each candidate against the resident reference. ──
    let mut rows = Vec::new();
    for model in evaluation_models(config) {
        if !Path::new(&model).exists() {
            rows.push(daemon_quality_skip_row(
                config,
                ctx,
                &model,
                "quality KLD requires each evaluated model to be a local filesystem path",
            ));
            continue;
        }
        if let Err(err) = engine
            .load(&model, daemon_kld_load_params(config, max_seq))
            .await
        {
            rows.push(daemon_quality_fail_row(
                config,
                ctx,
                &model,
                &format!("daemon load failed: {err}"),
                0,
            ));
            continue;
        }
        let output_path = evidence_dir.join(format!("{}.kldseq", model_artifact_stem(&model)));
        let resp = engine
            .kld_eval(
                KldEvalRequest {
                    mode: KldEvalMode::Score,
                    corpus: None,
                    ref_path: Some(ref_path.clone()),
                    output: Some(output_path.display().to_string()),
                    max_chunks: config.quality_max_chunks,
                    n_ctx: None,
                    config: None,
                    capture_hidden_layers: false,
                    dump_logits: false,
                },
                |_| {},
            )
            .await;
        rows.push(daemon_quality_row_from_resp(
            config,
            ctx,
            &model,
            &ref_path,
            &output_path,
            resp,
        ));
    }
    Ok(rows)
}

fn daemon_quality_base_metrics(ref_path: &str) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("daemon")),
        ("suite".to_string(), json!("daemon_kld_reference")),
        ("kldref".to_string(), json!(ref_path)),
    ])
}

fn daemon_quality_skip_row(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: &str,
    reason: &str,
) -> EvalResult {
    row_for_model(
        BatteryId::Quality,
        None,
        "kld_reference_slice",
        None,
        EvalStatus::Skip,
        Some(reason.to_string()),
        BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        config,
        ctx,
        prompt("benchmarks/quality-baselines/harness/canary.md"),
        0,
        model.to_string(),
    )
}

fn daemon_quality_fail_row(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: &str,
    reason: &str,
    elapsed_ms: u128,
) -> EvalResult {
    row_for_model(
        BatteryId::Quality,
        None,
        "kld_reference_slice",
        None,
        EvalStatus::Fail,
        Some(reason.to_string()),
        BTreeMap::from([
            ("executor".to_string(), json!("daemon")),
            ("implemented".to_string(), json!(true)),
        ]),
        config,
        ctx,
        prompt("benchmarks/quality-baselines/harness/canary.md"),
        elapsed_ms,
        model.to_string(),
    )
}

fn attach_kld_possible_false_negative_causes(
    metrics: &mut BTreeMap<String, Value>,
    status: EvalStatus,
    compat_findings: &[String],
) {
    if status == EvalStatus::Fail && !compat_findings.is_empty() {
        metrics.insert(
            "possible_false_negative_causes".to_string(),
            json!(compat_findings),
        );
    }
}

fn daemon_quality_row_from_resp(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: &str,
    ref_path: &str,
    output_path: &Path,
    resp: anyhow::Result<hipfire_daemon_protocol::KldEvalResponse>,
) -> EvalResult {
    let prompt_ref = prompt("benchmarks/quality-baselines/harness/canary.md");
    match resp {
        Ok(resp) => {
            let mut metrics = daemon_quality_base_metrics(ref_path);
            metrics.insert("scoring_mode".to_string(), json!("kld_reference_slice"));
            metrics.insert("n_chunks".to_string(), json!(resp.n_chunk));
            metrics.insert("total_scored".to_string(), json!(resp.total_scored));
            metrics.insert(
                "kldseq_path".to_string(),
                json!(output_path.display().to_string()),
            );
            if let Some(v) = resp.mean_kld {
                metrics.insert("mean_kld".to_string(), json!(v));
            }
            if let Some(v) = resp.p99_kld {
                metrics.insert("p99_kld".to_string(), json!(v));
            }
            if let Some(v) = resp.mean_nll {
                metrics.insert("mean_nll".to_string(), json!(v));
            }
            if let Some(v) = resp.ppl {
                metrics.insert("ppl".to_string(), json!(v));
            }
            let finite = resp.mean_kld.map(|v| v.is_finite()).unwrap_or(false);
            let (status, reason) = if !finite {
                (
                    EvalStatus::Fail,
                    Some("daemon kld_eval returned no finite mean_kld".to_string()),
                )
            } else {
                (EvalStatus::Pass, None)
            };
            attach_kld_possible_false_negative_causes(&mut metrics, status, &resp.compat_findings);
            row_for_model(
                BatteryId::Quality,
                None,
                "kld_reference_slice",
                None,
                status,
                reason,
                metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model.to_string(),
            )
        }
        Err(err) => daemon_quality_fail_row(
            config,
            ctx,
            model,
            &format!("daemon kld_eval score failed: {err}"),
            0,
        ),
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
    load_daemon_eval_session_for_model(config, bin, max_seq, &config.model).await
}

pub(crate) async fn load_daemon_eval_session_for_model(
    config: &EvalConfig,
    bin: &Path,
    max_seq: usize,
    model: &str,
) -> anyhow::Result<DaemonEvalSession> {
    let mut engine = hipfire_daemon_adapter::DaemonEngine::spawn(bin).await?;
    let loaded = engine
        .load(model, daemon_model_load_params(config, max_seq))
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
                    "daemon executor requires the model to resolve to a local filesystem path",
                    "daemon executor requires the model to resolve to a local filesystem path",
                ),
                BatteryId::Speed => daemon_speed_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires the model to resolve to a local filesystem path",
                ),
                BatteryId::Profile => daemon_profile_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires the model to resolve to a local filesystem path",
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

/// Read a committed image fixture and base64-encode it for the daemon's
/// `image_base64` generate field — keeps the vision battery's image input
/// byte-identical and CI-portable (no external dataset dependency).
pub(crate) fn read_repo_image_base64(image_path: &str) -> anyhow::Result<String> {
    use base64::Engine;
    let image_file = resolve_repo_path(image_path)
        .ok_or_else(|| anyhow::anyhow!("resolve {image_path} from repo root"))?;
    let bytes = fs::read(&image_file).map_err(|e| anyhow::anyhow!("read {image_path}: {e}"))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
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
    daemon_speed_skip_rows_for_model(config, ctx, &config.model, reason)
}

pub(crate) fn daemon_speed_skip_rows_for_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: &str,
    reason: &str,
) -> Vec<EvalResult> {
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    daemon_speed_cases()
        .iter()
        .map(|case| {
            row_for_model(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some(reason.to_string()),
                BTreeMap::from([
                    ("implemented".to_string(), json!(true)),
                    ("executor".to_string(), json!("daemon")),
                    ("suite".to_string(), json!("daemon_speed_anchor")),
                ]),
                config,
                ctx,
                prompt_ref.clone(),
                0,
                model.to_string(),
            )
        })
        .collect()
}

pub(crate) fn daemon_speed_base_metrics(
    worker_key_id: &str,
    bin: &Path,
    max_tokens: usize,
    done: &hipfire_generate::DoneEvent,
    text: &str,
    evidence_dir: &Path,
) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("daemon")),
        ("suite".to_string(), json!("daemon_speed_anchor")),
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
    metrics
}

pub(crate) fn daemon_done_has_speed_metric(done: &hipfire_generate::DoneEvent) -> bool {
    done.tok_s
        .is_some_and(|value| value.is_finite() && value > 0.0)
        || done
            .decode_tok_s
            .is_some_and(|value| value.is_finite() && value > 0.0)
}

pub(crate) fn daemon_speed_status_reason(
    text: &str,
    done: &hipfire_generate::DoneEvent,
) -> (EvalStatus, Option<String>) {
    let finite = !text.is_empty() && !text.contains('\u{fffd}') && done.tokens > 0;
    if !finite {
        return (
            EvalStatus::Fail,
            Some(
                "daemon speed anchor returned empty, zero-token, or replacement-character output"
                    .to_string(),
            ),
        );
    }
    if !daemon_done_has_speed_metric(done) {
        return (
            EvalStatus::Fail,
            Some("daemon speed anchor did not emit throughput metrics".to_string()),
        );
    }
    (EvalStatus::Pass, None)
}

pub(crate) fn daemon_speed_failure_rows_for_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    model: &str,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    daemon_speed_cases()
        .iter()
        .map(|case| {
            row_for_model(
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
                    ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ]),
                config,
                ctx,
                prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
                elapsed_ms,
                model.to_string(),
            )
        })
        .collect()
}

pub(crate) fn resolve_eval_model_path(model: &str) -> Option<PathBuf> {
    let models_dir = std::env::var_os("HIPFIRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".hipfire")
                .join("models")
        });
    find_model_in(model, &models_dir, None)
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
            "daemon executor requires the model to resolve to a local filesystem path",
            "daemon executor requires the model to resolve to a local filesystem path",
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
    if resolve_eval_model_path(&config.model).is_none() {
        return daemon_speed_skip_rows(
            config,
            ctx,
            "daemon executor requires the model to resolve to a local filesystem path",
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
            "daemon executor requires the model to resolve to a local filesystem path",
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
    let mut rows = Vec::new();
    let mut session: Option<DaemonEvalSession> = None;
    let mut session_loaded = false;
    for model in evaluation_models(config) {
        let Some(model_path) = resolve_eval_model_path(&model) else {
            rows.extend(daemon_speed_skip_rows_for_model(
                config,
                ctx,
                &model,
                "daemon speed executor requires evaluated models to resolve to local filesystem paths",
            ));
            continue;
        };
        let model_path = model_path.display().to_string();
        if let Some(active) = session.as_mut() {
            if session_loaded {
                if let Err(err) = active.engine.unload().await {
                    rows.extend(daemon_speed_failure_rows_for_model(
                        config,
                        ctx,
                        bin,
                        &model,
                        &format!("daemon unload failed before speed eval model {model}: {err}"),
                        0,
                    ));
                    break;
                }
                session_loaded = false;
            }
            match active
                .engine
                .load(&model_path, daemon_model_load_params(config, max_seq))
                .await
            {
                Ok(loaded) => {
                    active.worker_key_id = loaded.worker_key_id.clone();
                    active.loaded = loaded;
                    active.max_seq = max_seq;
                    session_loaded = true;
                }
                Err(err) => {
                    rows.extend(daemon_speed_failure_rows_for_model(
                        config,
                        ctx,
                        bin,
                        &model,
                        &format!("daemon load failed for speed eval model {model}: {err}"),
                        0,
                    ));
                    continue;
                }
            }
        } else {
            match load_daemon_eval_session_for_model(config, bin, max_seq, &model_path).await {
                Ok(active) => {
                    session = Some(active);
                    session_loaded = true;
                }
                Err(err) => {
                    rows.extend(daemon_speed_failure_rows_for_model(
                        config,
                        ctx,
                        bin,
                        &model,
                        &format!("daemon load failed for speed eval model {model}: {err}"),
                        0,
                    ));
                    continue;
                }
            }
        }
        let active = session
            .as_mut()
            .expect("daemon speed session exists after successful model load");
        rows.extend(
            daemon_speed_rows_with_session_for_model(config, ctx, bin, active, &model, &model_path)
                .await?,
        );
    }
    Ok(rows)
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
    daemon_speed_rows_with_session_for_model(
        config,
        ctx,
        bin,
        session,
        &config.model,
        &config.model,
    )
    .await
}

pub(crate) async fn daemon_speed_rows_with_session_for_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
    row_model: &str,
    loaded_model: &str,
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
            loaded_model,
        );
        let request = daemon_generate_request(
            format!("eval-speed-{}", case.label),
            prompt_text.clone(),
            max_tokens,
            Some(worker_key_id.clone()),
            Some(&evidence_dir),
        );
        let (text, done) = session.engine.generate(request).await?;
        let (status, reason) = daemon_speed_status_reason(&text, &done);
        let metrics =
            daemon_speed_base_metrics(&worker_key_id, bin, max_tokens, &done, &text, &evidence_dir);

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
            row_model.to_string(),
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
    let has_timing = daemon_done_has_speed_metric(&done);
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
        Some("daemon profile anchor did not emit throughput metrics".to_string())
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

// ── Vision battery (gemma3-vl / medgemma, arch 13) ──────────────────────────
//
// Loads the configured model through the daemon, sends a fixed prompt + a
// committed image fixture via the `image_base64` generate field, and asserts the
// streamed description is finite and non-degenerate (unique-word ratio + max
// single-word frequency — the same shape as the dflash coherence gate). Gated on
// the loaded model reporting `arch == "gemma3_vl"`; any other arch emits a skip
// row, so the battery is safe to include in the extensive tier against non-VL
// models. The image is committed in-repo, so the input stays byte-identical and
// CI-portable (no external dataset dependency).

const VISION_IMAGE_FIXTURE: &str = "benchmarks/vision/images/mri_human_brain.jpg";
const VISION_PROMPT_FIXTURE: &str = "benchmarks/prompts/vision_describe_image.txt";

pub(crate) fn run_daemon_vision_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return vec![vision_skip_row(
            config,
            ctx,
            "vision battery requires the model to resolve to a local filesystem path",
        )];
    }
    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return vec![vision_skip_row(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        )];
    };
    let started = std::time::SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return vec![vision_skip_row(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
            )]
        }
    };
    match runtime.block_on(run_daemon_vision_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => vec![vision_skip_row(
            config,
            ctx,
            &format!("vision battery error: {err}"),
        )],
    }
}

fn vision_skip_row(config: &EvalConfig, ctx: &EvalContext, reason: &str) -> EvalResult {
    skip_row(
        BatteryId::Vision,
        None,
        "describe_image",
        None,
        reason,
        config,
        ctx,
        prompt(VISION_PROMPT_FIXTURE),
    )
}

/// Unique-word ratio + max single-word frequency over `text` (whitespace-split):
/// a finite, on-topic description scores high unique-ratio / low max-freq, while
/// a token attractor collapses both.
fn vision_text_stats(text: &str) -> (f64, f64) {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return (0.0, 1.0);
    }
    let total = words.len() as f64;
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in &words {
        *counts.entry(*w).or_insert(0) += 1;
    }
    let unique_ratio = counts.len() as f64 / total;
    let max_freq = counts.values().copied().max().unwrap_or(0) as f64 / total;
    (unique_ratio, max_freq)
}

#[cfg(test)]
mod daemon_quality_tests {
    use super::*;

    #[test]
    fn kld_compat_findings_only_surface_as_false_negative_context_on_fail() {
        let findings = vec![
            "Warn gpu_arch: ref gfx1151 != run gfx1103 (cross-arch numerics)".to_string(),
            "Warn config.graph: ref true != run false".to_string(),
        ];

        let mut pass_metrics = BTreeMap::new();
        attach_kld_possible_false_negative_causes(&mut pass_metrics, EvalStatus::Pass, &findings);
        assert!(!pass_metrics.contains_key("possible_false_negative_causes"));
        assert!(!pass_metrics.contains_key("compat_findings"));

        let mut fail_metrics = BTreeMap::new();
        attach_kld_possible_false_negative_causes(&mut fail_metrics, EvalStatus::Fail, &findings);
        assert_eq!(
            fail_metrics.get("possible_false_negative_causes"),
            Some(&json!(findings))
        );
        assert!(!fail_metrics.contains_key("compat_findings"));
    }
}

pub(crate) async fn run_daemon_vision_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;

    // Gate: vision battery applies only to gemma3-vl (arch 13). The daemon's
    // `loaded` event reports arch="gemma3_vl" for arch 13.
    if session.loaded.arch.as_deref() != Some("gemma3_vl") {
        return Ok(vec![vision_skip_row(
            config,
            ctx,
            &format!(
                "loaded model arch {:?} is not a vision model (need gemma3_vl / arch 13)",
                session.loaded.arch
            ),
        )]);
    }

    let worker_key_id = session.worker_key_id.clone();
    let prompt_text = read_repo_prompt_text(VISION_PROMPT_FIXTURE)?;
    let image_b64 = read_repo_image_base64(VISION_IMAGE_FIXTURE)?;
    let max_tokens = config.max_tokens.max(64);

    let build_request = |id: &str| {
        let mut request = daemon_generate_request(
            id.to_string(),
            prompt_text.clone(),
            max_tokens,
            Some(worker_key_id.clone()),
            None,
        );
        request.image_base64 = Some(image_b64.clone());
        request
    };

    // ── Row 1: describe_image — finite, non-degenerate coherence ────────────
    let (text, done) = session
        .engine
        .generate(build_request("eval-vision-describe"))
        .await?;
    let finite = !text.is_empty() && !text.contains('\u{fffd}');
    let (unique_ratio, max_freq) = vision_text_stats(&text);
    let nondegenerate = unique_ratio >= 0.30 && max_freq <= 0.50;
    let describe_status = if finite && nondegenerate {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let describe_reason = if !finite {
        Some("vision describe returned empty or replacement-character output".to_string())
    } else if !nondegenerate {
        Some(format!(
            "vision describe output is degenerate (unique_ratio={unique_ratio:.2}, max_word_freq={max_freq:.2})"
        ))
    } else {
        None
    };

    // ── Row 2: cache_hit_determinism — reset + re-run the SAME image. The
    // vision-embedding cache is populated by row 1, so this pass is a cache hit
    // (encode skipped); greedy decode is deterministic, so the output must be
    // byte-identical. This is the in-harness hit==miss equality guard. ────────
    session.engine.reset().await?;
    let (text2, _done2) = session
        .engine
        .generate(build_request("eval-vision-describe-repeat"))
        .await?;
    let determinism_match = text2 == text;
    let determinism_status = if determinism_match {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let determinism_reason = (!determinism_match).then(|| {
        "vision repeat (cache-hit) output differs from first pass — cache or decode non-determinism"
            .to_string()
    });

    Ok(vec![
        row(
            BatteryId::Vision,
            None,
            "describe_image",
            None,
            describe_status,
            describe_reason,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("arch".to_string(), json!(session.loaded.arch)),
                ("worker_key_id".to_string(), json!(worker_key_id)),
                ("image_fixture".to_string(), json!(VISION_IMAGE_FIXTURE)),
                ("tokens".to_string(), json!(done.tokens)),
                ("text_bytes".to_string(), json!(text.len())),
                ("unique_word_ratio".to_string(), json!(unique_ratio)),
                ("max_word_freq".to_string(), json!(max_freq)),
                ("tok_s".to_string(), json!(done.tok_s)),
                ("max_tokens".to_string(), json!(max_tokens)),
            ]),
            config,
            ctx,
            prompt(VISION_PROMPT_FIXTURE),
            0,
        ),
        row(
            BatteryId::Vision,
            None,
            "cache_hit_determinism",
            None,
            determinism_status,
            determinism_reason,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("image_fixture".to_string(), json!(VISION_IMAGE_FIXTURE)),
                ("first_text_bytes".to_string(), json!(text.len())),
                ("repeat_text_bytes".to_string(), json!(text2.len())),
                ("byte_identical".to_string(), json!(determinism_match)),
            ]),
            config,
            ctx,
            prompt(VISION_PROMPT_FIXTURE),
            0,
        ),
    ])
}

#[cfg(test)]
mod daemon_speed_tests {
    use std::collections::HashMap;

    use hipfire_generate::DoneEvent;

    use super::*;

    #[test]
    fn tok_s_only_done_event_is_valid_speed_evidence() {
        let done = DoneEvent {
            id: "eval-speed".to_string(),
            tokens: 64,
            tok_s: Some(27.39),
            prefill_tokens: None,
            prefill_ms: Some(292.0),
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("length".to_string()),
            response_id: None,
            extra: HashMap::new(),
        };

        let (status, reason) = daemon_speed_status_reason("non-empty output", &done);
        assert_eq!(status, EvalStatus::Pass);
        assert_eq!(reason, None);
    }

    #[test]
    fn daemon_speed_skip_rows_preserve_evaluated_model_label() {
        let cfg = parse_args_from(["hipfire-eval", "candidate.hfq", "--battery", "speed"]).unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: collect_host_profile(None, HostProfileOverrides::default()),
        };

        let rows =
            daemon_speed_skip_rows_for_model(&cfg, &ctx, "compare.hfq", "missing compare model");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.model == "compare.hfq"));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
    }
}

#[cfg(test)]
mod vision_battery_tests {
    use super::vision_text_stats;

    #[test]
    fn coherent_text_scores_high_unique_low_freq() {
        let text =
            "This is a brain MRI image showing the cerebrum, cerebellum and brainstem clearly.";
        let (unique_ratio, max_freq) = vision_text_stats(text);
        assert!(unique_ratio >= 0.30, "unique_ratio={unique_ratio}");
        assert!(max_freq <= 0.50, "max_freq={max_freq}");
    }

    #[test]
    fn single_token_attractor_is_degenerate() {
        let text = "the the the the the the the the the the";
        let (unique_ratio, max_freq) = vision_text_stats(text);
        assert!(unique_ratio < 0.30, "unique_ratio={unique_ratio}");
        assert!(max_freq > 0.50, "max_freq={max_freq}");
    }

    #[test]
    fn empty_text_is_degenerate() {
        let (unique_ratio, max_freq) = vision_text_stats("");
        assert_eq!(unique_ratio, 0.0);
        assert_eq!(max_freq, 1.0);
    }
}
