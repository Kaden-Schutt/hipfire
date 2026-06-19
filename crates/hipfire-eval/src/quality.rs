// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Quality-eval rows: external quality-JSON ingestion + KLD-reference scoring.
//!
//! Turns a supplied quality-results JSON (or a computed KLD reference vs a
//! baseline) into EvalResult rows, with parse/validate of the JSON schema.
//! Extracted verbatim from the former `hipfire-eval/src/lib.rs` monolith (no
//! behavior change).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::*;

pub(crate) fn quality_json_rows(config: &EvalConfig, ctx: &EvalContext) -> Option<Vec<EvalResult>> {
    let path = config.quality_json.as_ref()?;
    let rows = match load_quality_json_rows(path) {
        Ok(rows) => rows,
        Err(reason) => {
            return Some(vec![skip_row(
                BatteryId::Quality,
                None,
                "quality_json_ingest",
                None,
                &reason,
                config,
                ctx,
                None,
            )]);
        }
    };
    let mut out = Vec::new();
    let candidate_variant = config
        .candidate_variant
        .clone()
        .unwrap_or_else(|| model_artifact_stem(&config.model));
    out.push(quality_json_row_for_variant(
        path,
        &rows,
        "candidate",
        &candidate_variant,
        &config.model,
        config,
        ctx,
    ));
    if let Some(model) = &config.baseline {
        let variant = config
            .baseline_variant
            .clone()
            .unwrap_or_else(|| model_artifact_stem(model));
        out.push(quality_json_row_for_variant(
            path, &rows, "baseline", &variant, model, config, ctx,
        ));
    }
    if let Some(model) = &config.reference {
        let variant = config
            .reference_variant
            .clone()
            .unwrap_or_else(|| model_artifact_stem(model));
        out.push(quality_json_row_for_variant(
            path,
            &rows,
            "reference",
            &variant,
            model,
            config,
            ctx,
        ));
    }
    Some(out)
}

pub(crate) fn quality_json_row_for_variant(
    path: &Path,
    rows: &[QualityJsonRow],
    role: &str,
    variant: &str,
    model: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
) -> EvalResult {
    let Some(row) = rows.iter().find(|r| r.variant == variant) else {
        return row_for_model(
            BatteryId::Quality,
            None,
            "kld_reference_slice",
            None,
            EvalStatus::Skip,
            Some(format!(
                "quality-json variant {variant:?} not found for {role}"
            )),
            BTreeMap::from([
                (
                    "quality_source".to_string(),
                    json!(path.display().to_string()),
                ),
                ("variant".to_string(), json!(variant)),
            ]),
            config,
            ctx,
            None,
            0,
            model.to_string(),
        );
    };

    let mut metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("quality_json")),
        (
            "quality_source".to_string(),
            json!(path.display().to_string()),
        ),
        ("variant".to_string(), json!(row.variant.clone())),
        ("arch".to_string(), json!(row.arch.clone())),
        ("scoring_mode".to_string(), json!(row.scoring_mode.clone())),
        ("n_chunks".to_string(), json!(row.n_chunks)),
        ("mean_kld".to_string(), json!(row.mean_kld)),
        ("mean_kld_ci_lo".to_string(), json!(row.mean_kld_ci_lo)),
        ("mean_kld_ci_hi".to_string(), json!(row.mean_kld_ci_hi)),
        ("p99_kld".to_string(), json!(row.p99_kld)),
    ]);
    if let Some(ppl) = row.ppl {
        metrics.insert("ppl".to_string(), json!(ppl));
    }
    if !row.notes.is_empty() {
        metrics.insert("notes".to_string(), json!(row.notes.clone()));
    }
    row_for_model(
        BatteryId::Quality,
        None,
        "kld_reference_slice",
        None,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        prompt("benchmarks/quality-baselines/harness/canary.md"),
        0,
        model.to_string(),
    )
}

pub(crate) fn kld_reference_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Option<Vec<EvalResult>> {
    Some(
        evaluation_models(config)
            .into_iter()
            .map(|model| run_kld_reference_row(config, ctx, model))
            .collect(),
    )
}

pub(crate) fn run_kld_reference_row(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_ref = prompt("benchmarks/quality-baselines/harness/canary.md");
    let mut base_metrics = BTreeMap::from([("executor".to_string(), json!("eval_hipfire"))]);
    let Some(ref_path) = resolve_kldref_for_model(config, &model) else {
        return row_for_model(
            BatteryId::Quality,
            None,
            "kld_reference_slice",
            None,
            EvalStatus::Skip,
            Some("no HFQM .kldref.hfq found; pass --kldref or place the matching ref in benchmarks/quality-baselines/refs".to_string()),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    base_metrics.insert("kldref".to_string(), json!(ref_path.display().to_string()));
    base_metrics.insert("kldref_hash".to_string(), json!(file_hash(&ref_path)));
    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Quality,
            None,
            "kld_reference_slice",
            None,
            EvalStatus::Skip,
            Some(
                "quality KLD requires each evaluated model to be a local filesystem path"
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
    let Some(bin) = resolve_eval_hipfire_bin() else {
        return row_for_model(
            BatteryId::Quality,
            None,
            "kld_reference_slice",
            None,
            EvalStatus::Skip,
            Some("eval_hipfire example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example eval_hipfire`".to_string()),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let evidence_dir = runtime_evidence_dir(config, "kld_reference_slice", &model);
    let _ = fs::create_dir_all(&evidence_dir);
    let output_path = evidence_dir.join(format!("{}.kldseq", model_artifact_stem(&model)));
    let mut args = vec![
        "--model".to_string(),
        model.clone(),
        "--ref".to_string(),
        ref_path.display().to_string(),
        "--output".to_string(),
        output_path.display().to_string(),
        "--kv-mode".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--scoring-mode".to_string(),
        "prefill".to_string(),
    ];
    if let Some(max_chunks) = config.quality_max_chunks {
        args.push("--max-chunks".to_string());
        args.push(max_chunks.to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            let mut metrics = base_metrics;
            metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Quality,
                None,
                "kld_reference_slice",
                None,
                EvalStatus::Fail,
                Some(format!("spawn eval_hipfire: {err}")),
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
    let mut metrics = base_metrics;
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "kldseq_path".to_string(),
        json!(output_path.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if let Some(max_chunks) = config.quality_max_chunks {
        metrics.insert("max_chunks".to_string(), json!(max_chunks));
    }
    match parse_hfkseq_metrics(&output_path) {
        Ok(parsed) if output.status.success() => {
            metrics.extend(parsed);
            row_for_model(
                BatteryId::Quality,
                None,
                "kld_reference_slice",
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
        }
        Ok(parsed) => {
            metrics.extend(parsed);
            row_for_model(
                BatteryId::Quality,
                None,
                "kld_reference_slice",
                None,
                EvalStatus::Fail,
                Some(format!("eval_hipfire exited with {}", output.status)),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_ms,
                model,
            )
        }
        Err(reason) => row_for_model(
            BatteryId::Quality,
            None,
            "kld_reference_slice",
            None,
            EvalStatus::Fail,
            Some(if output.status.success() {
                reason
            } else {
                format!("eval_hipfire exited with {}; {reason}", output.status)
            }),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        ),
    }
}

pub(crate) fn parse_hfkseq_metrics(path: &Path) -> Result<BTreeMap<String, Value>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .map_err(|e| format!("read HFKSEQ magic: {e}"))?;
    if &magic != b"HFKSEQ\0\0" {
        return Err(format!("bad HFKSEQ magic in {}", path.display()));
    }
    let mut hdr = [0u8; 12];
    file.read_exact(&mut hdr)
        .map_err(|e| format!("read HFKSEQ header: {e}"))?;
    let version = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let n_chunk = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    if version != 1 && version != 2 {
        return Err(format!("unsupported HFKSEQ version {version}"));
    }
    let record_bytes = if version == 2 { 24 } else { 16 };
    let mut mean_kld = Vec::with_capacity(n_chunk);
    let mut p99_kld = Vec::with_capacity(n_chunk);
    let mut mean_nll = Vec::with_capacity(n_chunk);
    let mut buf = vec![0u8; record_bytes];
    for _ in 0..n_chunk {
        file.read_exact(&mut buf)
            .map_err(|e| format!("read HFKSEQ record: {e}"))?;
        mean_kld.push(f64::from_le_bytes(buf[0..8].try_into().unwrap()));
        p99_kld.push(f64::from_le_bytes(buf[8..16].try_into().unwrap()));
        if version == 2 {
            mean_nll.push(f64::from_le_bytes(buf[16..24].try_into().unwrap()));
        }
    }
    let n = n_chunk.max(1) as f64;
    let mean_kld_value = mean_kld.iter().sum::<f64>() / n;
    let p99_kld_value = p99_kld.iter().copied().fold(0.0f64, f64::max);
    let mut metrics = BTreeMap::from([
        ("scoring_mode".to_string(), json!("kld_reference_slice")),
        ("hfkseq_version".to_string(), json!(version)),
        ("n_chunks".to_string(), json!(n_chunk)),
        ("mean_kld".to_string(), json!(mean_kld_value)),
        ("p99_kld".to_string(), json!(p99_kld_value)),
    ]);
    if version == 2 && !mean_nll.is_empty() {
        let mean_nll_value = mean_nll.iter().sum::<f64>() / n;
        metrics.insert("mean_nll".to_string(), json!(mean_nll_value));
        metrics.insert("ppl".to_string(), json!(mean_nll_value.exp()));
    }
    Ok(metrics)
}

pub(crate) fn resolve_kldref_for_model(config: &EvalConfig, model: &str) -> Option<PathBuf> {
    if let Some(path) = &config.kldref {
        return path.exists().then(|| path.clone());
    }
    let ref_name = kldref_name_for_model(model)?;
    let repo = repo_root().unwrap_or_else(|| PathBuf::from("."));
    let candidates = vec![
        repo.join("benchmarks")
            .join("quality-baselines")
            .join("refs")
            .join(&ref_name),
        home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".hipfire")
            .join("eval-results")
            .join("refs")
            .join(&ref_name),
    ];
    candidates.into_iter().find(|path| path.exists())
}

pub(crate) fn kldref_name_for_model(model: &str) -> Option<String> {
    let stem = model_artifact_stem(model);
    if let Some(idx) = stem.find("-bf16") {
        return Some(format!("{}.kldref.hfq", &stem[..idx + "-bf16".len()]));
    }
    let parts: Vec<&str> = stem.split('-').collect();
    if parts.len() < 2 {
        return None;
    }
    let mut base = vec![parts[0], parts[1]];
    if parts.get(2).copied() == Some("a3b") {
        base.push("a3b");
    }
    Some(format!("{}-bf16.kldref.hfq", base.join("-")))
}

#[derive(Debug, Clone)]
pub(crate) struct QualityJsonRow {
    variant: String,
    arch: String,
    scoring_mode: String,
    n_chunks: u64,
    mean_kld: f64,
    mean_kld_ci_lo: f64,
    mean_kld_ci_hi: f64,
    p99_kld: f64,
    ppl: Option<f64>,
    notes: String,
}

pub(crate) fn load_quality_json_rows(path: &Path) -> Result<Vec<QualityJsonRow>, String> {
    let body = fs::read_to_string(path).map_err(|e| format!("read quality json: {e}"))?;
    let value: Value =
        serde_json::from_str(&body).map_err(|e| format!("parse quality json: {e}"))?;
    let raw_rows = if let Some(rows) = value.as_array() {
        rows.clone()
    } else if let Some(rows) = value.get("quality_rows").and_then(Value::as_array) {
        rows.clone()
    } else {
        return Err("unsupported quality JSON shape".to_string());
    };
    raw_rows
        .iter()
        .map(parse_quality_json_row)
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn parse_quality_json_row(value: &Value) -> Result<QualityJsonRow, String> {
    let get_str = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("quality row missing string field {name:?}"))
    };
    let get_f64 = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("quality row missing numeric field {name:?}"))
    };
    let row = QualityJsonRow {
        variant: get_str("variant")?,
        arch: get_str("arch")?,
        scoring_mode: get_str("scoring_mode")?,
        n_chunks: value
            .get("n_chunks")
            .and_then(Value::as_u64)
            .ok_or_else(|| "quality row missing numeric field \"n_chunks\"".to_string())?,
        mean_kld: get_f64("mean_kld")?,
        mean_kld_ci_lo: get_f64("mean_kld_ci_lo")?,
        mean_kld_ci_hi: get_f64("mean_kld_ci_hi")?,
        p99_kld: get_f64("p99_kld")?,
        ppl: value.get("ppl").and_then(Value::as_f64),
        notes: value
            .get("notes")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    };
    validate_quality_json_row(&row)?;
    Ok(row)
}

pub(crate) fn validate_quality_json_row(row: &QualityJsonRow) -> Result<(), String> {
    let finite_fields = [
        ("mean_kld", row.mean_kld),
        ("mean_kld_ci_lo", row.mean_kld_ci_lo),
        ("mean_kld_ci_hi", row.mean_kld_ci_hi),
        ("p99_kld", row.p99_kld),
    ];
    for (name, value) in finite_fields {
        if !value.is_finite() {
            return Err(format!(
                "quality row {:?} has non-finite {name}",
                row.variant
            ));
        }
        if value < 0.0 {
            return Err(format!("quality row {:?} has negative {name}", row.variant));
        }
    }
    if row.mean_kld_ci_lo > row.mean_kld || row.mean_kld > row.mean_kld_ci_hi {
        return Err(format!(
            "quality row {:?} has incoherent mean_kld confidence interval",
            row.variant
        ));
    }
    if let Some(ppl) = row.ppl {
        if !ppl.is_finite() {
            return Err(format!("quality row {:?} has non-finite ppl", row.variant));
        }
        if ppl <= 0.0 {
            return Err(format!(
                "quality row {:?} has non-positive ppl",
                row.variant
            ));
        }
    }
    Ok(())
}
