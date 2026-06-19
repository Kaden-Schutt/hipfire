// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Performance-eval rows: external performance-JSON ingestion.
//!
//! Turns a supplied performance-results JSON into EvalResult rows, parsing +
//! validating + normalizing its metric schema. Extracted verbatim from the
//! former `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::*;

pub(crate) fn performance_json_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Option<Vec<EvalResult>> {
    let path = config.performance_json.as_ref()?;
    let rows = match load_performance_json_rows(path) {
        Ok(rows) => rows,
        Err(reason) => {
            return Some(vec![skip_row(
                BatteryId::Speed,
                None,
                "performance_json_ingest",
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
        .performance_candidate_variant
        .clone()
        .or_else(|| config.candidate_variant.clone())
        .unwrap_or_else(|| model_artifact_stem(&config.model));
    out.push(performance_json_row_for_variant(
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
            .performance_baseline_variant
            .clone()
            .or_else(|| config.baseline_variant.clone())
            .unwrap_or_else(|| model_artifact_stem(model));
        out.push(performance_json_row_for_variant(
            path, &rows, "baseline", &variant, model, config, ctx,
        ));
    }
    if let Some(model) = &config.reference {
        let variant = config
            .performance_reference_variant
            .clone()
            .or_else(|| config.reference_variant.clone())
            .unwrap_or_else(|| model_artifact_stem(model));
        out.push(performance_json_row_for_variant(
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

pub(crate) fn performance_json_row_for_variant(
    path: &Path,
    rows: &[PerformanceJsonRow],
    role: &str,
    variant: &str,
    model: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
) -> EvalResult {
    let Some(row) = rows.iter().find(|r| r.variant == variant) else {
        return row_for_model(
            BatteryId::Speed,
            None,
            "performance_json_anchor",
            None,
            EvalStatus::Skip,
            Some(format!(
                "performance-json variant {variant:?} not found for {role}"
            )),
            BTreeMap::from([
                (
                    "performance_source".to_string(),
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

    let mut metrics = row.metrics.clone();
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("performance_json"));
    metrics.insert(
        "performance_source".to_string(),
        json!(path.display().to_string()),
    );
    metrics.insert("variant".to_string(), json!(row.variant.clone()));
    row_for_model(
        BatteryId::Speed,
        None,
        "performance_json_anchor",
        None,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
        0,
        model.to_string(),
    )
}

#[derive(Debug, Clone)]
pub(crate) struct PerformanceJsonRow {
    variant: String,
    metrics: BTreeMap<String, Value>,
}

pub(crate) fn load_performance_json_rows(path: &Path) -> Result<Vec<PerformanceJsonRow>, String> {
    let body = fs::read_to_string(path).map_err(|e| format!("read performance json: {e}"))?;
    let value: Value =
        serde_json::from_str(&body).map_err(|e| format!("parse performance json: {e}"))?;
    let raw_rows = if let Some(rows) = value.as_array() {
        rows.clone()
    } else if let Some(rows) = value.get("performance_rows").and_then(Value::as_array) {
        rows.clone()
    } else if let Some(rows) = value.get("runs").and_then(Value::as_array) {
        rows.clone()
    } else {
        return Err("unsupported performance JSON shape".to_string());
    };
    raw_rows
        .iter()
        .map(parse_performance_json_row)
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn parse_performance_json_row(value: &Value) -> Result<PerformanceJsonRow, String> {
    let variant = if let Some(variant) = value.get("variant").and_then(Value::as_str) {
        variant.to_string()
    } else {
        let base = ["model", "tag", "name"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .ok_or_else(|| "performance row missing variant/model/tag/name".to_string())?;
        if let Some(mode) = value.get("mode").and_then(Value::as_str) {
            format!("{base}:{mode}")
        } else {
            base.to_string()
        }
    };
    let mut metrics = BTreeMap::new();
    collect_performance_metrics(value, &mut metrics);
    if let Some(parsed) = value.get("parsed") {
        collect_performance_metrics(parsed, &mut metrics);
    }
    if metrics.is_empty() {
        return Err(format!(
            "performance row {variant:?} has no recognized numeric metrics"
        ));
    }
    validate_performance_metrics(&variant, &metrics)?;
    Ok(PerformanceJsonRow { variant, metrics })
}

pub(crate) fn collect_performance_metrics(value: &Value, out: &mut BTreeMap<String, Value>) {
    let Some(obj) = value.as_object() else {
        return;
    };
    for (key, value) in obj {
        let Some(num) = value.as_f64() else {
            continue;
        };
        if let Some(normalized) = normalize_performance_metric(key) {
            out.insert(normalized.to_string(), json!(num));
        }
    }
}

pub(crate) fn validate_performance_metrics(
    variant: &str,
    metrics: &BTreeMap<String, Value>,
) -> Result<(), String> {
    for (name, value) in metrics {
        let Some(num) = value.as_f64() else {
            continue;
        };
        if !num.is_finite() {
            return Err(format!("performance row {variant:?} has non-finite {name}"));
        }
        let non_negative = matches!(
            name.as_str(),
            "tok_s"
                | "wall_tok_s"
                | "ttft_ms"
                | "load_ms"
                | "prefill_ms"
                | "prefill_secs"
                | "decode_ms"
                | "decode_secs"
                | "teardown_ms"
                | "prefill_tok_s"
                | "pp32_tok_s"
                | "pp128_tok_s"
                | "pp512_tok_s"
                | "pp1024_tok_s"
                | "pp2048_tok_s"
                | "tau"
                | "accept_rate"
                | "emitted_tokens"
                | "cycles"
                | "vram_peak_bytes"
                | "vram_used_bytes"
                | "vram_used_mb"
                | "vram_loaded_mb"
                | "vram_free_mb"
                | "kv_bytes"
                | "workspace_bytes"
        );
        if non_negative && num < 0.0 {
            return Err(format!("performance row {variant:?} has negative {name}"));
        }
        if name == "accept_rate" && num > 1.0 {
            return Err(format!(
                "performance row {variant:?} has out-of-range accept_rate"
            ));
        }
        if name == "tau" && num == 0.0 {
            return Err(format!("performance row {variant:?} has zero tau"));
        }
    }
    Ok(())
}

pub(crate) fn normalize_performance_metric(key: &str) -> Option<&'static str> {
    match key {
        "tok_s" | "tokSOut" | "decode_tokS" | "decode_tok_s" | "gen_tok_s"
        | "tokens_per_second" => Some("tok_s"),
        "wall_tokS" | "wall_tok_s" => Some("wall_tok_s"),
        "ttft_ms" | "ttft" => Some("ttft_ms"),
        "load_ms" => Some("load_ms"),
        "prefill_ms" => Some("prefill_ms"),
        "prefill_secs" => Some("prefill_secs"),
        "decode_ms" => Some("decode_ms"),
        "decode_secs" => Some("decode_secs"),
        "teardown_ms" => Some("teardown_ms"),
        "prefill_tok_s" | "prefill_user_tokS" | "prefill_user_tok_s" => Some("prefill_tok_s"),
        "pp32_tok_s" | "pp32_tokS" => Some("pp32_tok_s"),
        "pp128_tok_s" | "pp128_tokS" => Some("pp128_tok_s"),
        "pp512_tok_s" | "pp512_tokS" => Some("pp512_tok_s"),
        "pp1024_tok_s" | "pp1024_tokS" => Some("pp1024_tok_s"),
        "pp2048_tok_s" | "pp2048_tokS" => Some("pp2048_tok_s"),
        "tau" | "decode_tau" => Some("tau"),
        "accept_rate" | "decode_accept_rate" => Some("accept_rate"),
        "emitted" | "emitted_tokens" => Some("emitted_tokens"),
        "cycles" => Some("cycles"),
        "vram_peak_bytes" => Some("vram_peak_bytes"),
        "vram_used_bytes" => Some("vram_used_bytes"),
        "vram_used_mb" => Some("vram_used_mb"),
        "vram_loaded_mb" => Some("vram_loaded_mb"),
        "vram_free_mb" => Some("vram_free_mb"),
        "kv_bytes" => Some("kv_bytes"),
        "workspace_bytes" => Some("workspace_bytes"),
        _ => None,
    }
}
