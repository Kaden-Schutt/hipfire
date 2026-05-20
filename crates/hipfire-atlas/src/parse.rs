// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Legacy stdout parsers for bench/dflash output.
//!
//! These exist so the Atlas can ingest captures from binaries that
//! haven't yet been wired with `--emit-atlas`. New consumers should
//! prefer emitting `AtlasRow` values directly; this module remains as
//! a bridge during the migration documented in
//! `docs/methodology/kernel-atlas-architecture.md`.

use crate::schema::{value_object, AtlasRow};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelOp {
    pub family: String,
    pub role: String,
    pub phase_hint: String,
}

pub fn parse_bench_summary(text: &str) -> Result<BTreeMap<String, f64>, String> {
    let summary = text
        .lines()
        .filter(|line| line.starts_with("SUMMARY"))
        .last()
        .ok_or_else(|| "bench output did not contain a SUMMARY line".to_string())?;
    let pair_re = Regex::new(r"([A-Za-z0-9_]+)=([0-9.]+)").expect("valid regex");
    let mut values = BTreeMap::new();
    for cap in pair_re.captures_iter(summary) {
        let key = cap[1].to_string();
        let value = cap[2]
            .parse::<f64>()
            .map_err(|e| format!("invalid SUMMARY value {key}={}: {e}", &cap[2]))?;
        values.insert(key, value);
    }
    for required in ["gen_tok_s", "bw_gib_s", "prefill_tok_s", "avg_ms", "p50_ms"] {
        if !values.contains_key(required) {
            return Err(format!("bench SUMMARY missing key: {required}"));
        }
    }
    // Also merge PREFILL_SUMMARY fields if present (latency split metrics
    // emitted by bench_qwen35_mq4 starting 2026-05-14).
    if let Some(prefill_line) = text.lines().find(|line| line.starts_with("PREFILL_SUMMARY")) {
        for cap in pair_re.captures_iter(prefill_line) {
            let key = cap[1].to_string();
            if let Ok(v) = cap[2].parse::<f64>() {
                values.entry(key).or_insert(v);
            }
        }
    }
    Ok(values)
}

pub fn parse_dflash_summary(text: &str) -> Result<BTreeMap<String, Value>, String> {
    let mut metrics = BTreeMap::new();
    for (key, prefix) in [
        ("decode_tok_s", "decode_tok_s:"),
        ("tau", "decode_tau:"),
        ("ttft_ms", "ttft_ms:"),
    ] {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix(prefix) {
                if let Some(token) = rest.split_whitespace().next() {
                    if let Ok(v) = token.parse::<f64>() {
                        metrics.insert(key.to_string(), json!(v));
                    }
                }
            }
        }
    }

    let emitted_re = Regex::new(
        r"emitted:\s*([0-9]+)\s+tokens\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+tok/s\)",
    )
    .expect("valid regex");
    if let Some(cap) = emitted_re.captures(text) {
        metrics.insert("emitted_tokens".to_string(), json!(cap[1].parse::<u64>().unwrap_or(0)));
        metrics.insert("elapsed_s".to_string(), json!(cap[2].parse::<f64>().unwrap_or(0.0)));
        metrics
            .entry("decode_tok_s".to_string())
            .or_insert_with(|| json!(cap[3].parse::<f64>().unwrap_or(0.0)));
    }

    if !metrics.contains_key("tau") {
        let tau_re = Regex::new(r"(?:tau|τ)=([0-9.]+)").expect("valid regex");
        if let Some(cap) = tau_re.captures(text) {
            metrics.insert("tau".to_string(), json!(cap[1].parse::<f64>().unwrap_or(0.0)));
        }
    }

    for key in ["cycles", "accepted"] {
        let re = Regex::new(&format!(r"(?m)^{key}:\s*([0-9]+)")).expect("valid regex");
        if let Some(cap) = re.captures(text) {
            metrics.insert(key.to_string(), json!(cap[1].parse::<u64>().unwrap_or(0)));
        }
    }

    for required in ["decode_tok_s", "tau"] {
        if !metrics.contains_key(required) {
            return Err(format!("dflash output missing metric: {required}"));
        }
    }
    Ok(metrics)
}

pub fn parse_profile_sections(text: &str) -> BTreeMap<String, Vec<Value>> {
    let line_re = Regex::new(
        r"^\s+([A-Za-z0-9_.$]+)\s+([0-9]+)x\s+([0-9.]+)ms\s+\(([0-9.]+)(?:µ|u)s/call\)\s+([0-9.]+)%\s+([0-9.]+)\s+GiB/s",
    )
    .expect("valid regex");
    let mut sections: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut current: Option<&str> = None;
    for line in text.lines() {
        if line.starts_with("=== DECODE PROFILE") {
            current = Some("decode_ar");
            continue;
        }
        if line.starts_with("=== PROFILE") {
            current = Some("prefill");
            continue;
        }
        if line.starts_with("===") && !line.contains("PROFILE") {
            current = None;
            continue;
        }
        let Some(section) = current else { continue };
        let Some(cap) = line_re.captures(line) else { continue };
        sections.entry(section.to_string()).or_default().push(json!({
            "name": &cap[1],
            "calls": cap[2].parse::<u64>().unwrap_or(0),
            "total_ms": cap[3].parse::<f64>().unwrap_or(0.0),
            "avg_us": cap[4].parse::<f64>().unwrap_or(0.0),
            "pct": cap[5].parse::<f64>().unwrap_or(0.0),
            "gib_s": cap[6].parse::<f64>().unwrap_or(0.0),
            "op": classify_kernel_op(&cap[1]),
        }));
    }
    sections
}

pub fn classify_kernel_op(name: &str) -> KernelOp {
    let lowered = name.to_ascii_lowercase();
    let (family, role, phase_hint) = if lowered.contains("fused_qkvza") {
        ("attention", "qkvza_projection", "prefill/decode")
    } else if lowered.contains("fused_qkv") {
        ("attention", "qkv_projection", "prefill/decode")
    } else if lowered.contains("attention_flash") || lowered.contains("flash") {
        ("attention", "flash_attention", "prefill/decode")
    } else if lowered.contains("kv_cache") {
        ("attention", "kv_cache", "decode")
    } else if lowered.contains("rope") || lowered.contains("rotate") {
        ("position", "rope_rotate", "prefill/decode")
    } else if lowered.contains("rmsnorm") || lowered.contains("norm") {
        ("norm", "normalization", "prefill/decode")
    } else if lowered.contains("gate_up") {
        ("mlp", "gate_up_projection", "prefill/decode")
    } else if lowered.contains("swiglu") {
        ("mlp", "swiglu", "prefill/decode")
    } else if lowered.contains("gemv") && lowered.contains("residual") {
        ("linear", "residual_gemv", "decode")
    } else if lowered.contains("gemv") && lowered.contains("multirow") {
        ("linear", "multirow_gemv", "decode")
    } else if lowered.contains("gemv") {
        ("linear", "gemv", "decode")
    } else if lowered.contains("gemm") && lowered.contains("residual") {
        ("linear", "residual_gemm", "prefill")
    } else if lowered.contains("gemm") {
        ("linear", "gemm", "prefill")
    } else {
        ("unknown", "unknown", "unknown")
    };
    KernelOp {
        family: family.to_string(),
        role: role.to_string(),
        phase_hint: phase_hint.to_string(),
    }
}

pub fn bench_rows_from_output(text: &str) -> Result<Vec<AtlasRow>, String> {
    let summary = parse_bench_summary(text)?;
    let profiles = parse_profile_sections(text);
    let mut prefill = AtlasRow::new("prefill", "ar");
    prefill.shape_bucket = "parsed_bench".to_string();
    prefill.metrics.insert(
        "prefill_tok_s".to_string(),
        json!(summary["prefill_tok_s"]),
    );
    // Carry the latency split if present.
    for k in [
        "prefill_tok_s_kernel",
        "prefill_kernel_ms",
        "prefill_wall_ms",
        "startup_overhead_ms",
        "cold_overhead_pct",
    ] {
        if let Some(v) = summary.get(k) {
            prefill.metrics.insert(k.to_string(), json!(v));
        }
    }
    if let Some(kernels) = profiles.get("prefill") {
        prefill
            .artifacts
            .insert("profile_kernels".to_string(), Value::Array(kernels.clone()));
    }

    let mut decode = AtlasRow::new("decode_ar", "ar");
    decode.shape_bucket = "parsed_bench".to_string();
    for key in ["gen_tok_s", "bw_gib_s", "avg_ms", "p50_ms"] {
        decode.metrics.insert(key.to_string(), json!(summary[key]));
    }
    if let Some(kernels) = profiles.get("decode_ar") {
        decode
            .artifacts
            .insert("profile_kernels".to_string(), Value::Array(kernels.clone()));
    }
    Ok(vec![prefill, decode])
}

pub fn dflash_row_from_output(text: &str) -> Result<AtlasRow, String> {
    let mut row = AtlasRow::new("decode_dflash", "dflash");
    row.shape_bucket = "parsed_dflash".to_string();
    row.metrics = parse_dflash_summary(text)?;
    Ok(row)
}

pub fn row_summary_value(row: &AtlasRow) -> Value {
    value_object([
        ("schema".to_string(), json!(row.schema)),
        ("phase".to_string(), json!(row.phase)),
        ("workload_kind".to_string(), json!(row.workload_kind)),
        ("metrics".to_string(), json!(row.metrics)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bench_summary() {
        let out = "noise\nSUMMARY gen_tok_s=123.4 bw_gib_s=55.0 prefill_tok_s=999.0 avg_ms=1.2 p50_ms=1.1\n";
        let values = parse_bench_summary(out).unwrap();
        assert_eq!(values["gen_tok_s"], 123.4);
    }

    #[test]
    fn parses_prefill_summary_split() {
        let out = "PREFILL_SUMMARY  prefill_tok_s=750.0  prefill_wall_ms=170.0  prefill_tok_s_kernel=1500.0  prefill_kernel_ms=85.0  startup_overhead_ms=85.0  cold_overhead_pct=50.0\nSUMMARY gen_tok_s=50.0 bw_gib_s=400.0 prefill_tok_s=750.0 avg_ms=20.0 p50_ms=20.0\n";
        let values = parse_bench_summary(out).unwrap();
        assert_eq!(values["prefill_tok_s_kernel"], 1500.0);
        assert_eq!(values["startup_overhead_ms"], 85.0);
    }

    #[test]
    fn parses_dflash_summary() {
        let out = "decode_tok_s: 88.5\ndecode_tau: 7.25\nemitted: 120 tokens in 1.4s (85.7 tok/s)\n";
        let values = parse_dflash_summary(out).unwrap();
        assert_eq!(values["tau"].as_f64().unwrap(), 7.25);
        assert_eq!(values["emitted_tokens"].as_u64().unwrap(), 120);
    }
}
