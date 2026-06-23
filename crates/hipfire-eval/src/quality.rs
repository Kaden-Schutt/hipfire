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
use std::path::Path;

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
