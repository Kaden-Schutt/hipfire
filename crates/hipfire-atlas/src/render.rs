// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::schema::AtlasRow;
use serde_json::Value;

pub fn render_fit_view(row: &AtlasRow) -> String {
    let mut out = Vec::new();
    out.push("HIPFIRE KERNEL ATLAS".to_string());
    out.push(format!(
        "phase={} workload={} model={} quant={} shape={}",
        empty_dash(&row.phase),
        empty_dash(&row.workload_kind),
        empty_dash(&row.model_size),
        empty_dash(&row.quant),
        empty_dash(&row.shape_bucket)
    ));
    out.push(String::new());
    out.push("metrics".to_string());
    for (key, value) in &row.metrics {
        out.push(format!("  {:24} {}", key, scalar(value)));
    }
    if let Some(kernels) = row.artifact_array("profile_kernels") {
        out.push(String::new());
        out.push("hot kernels".to_string());
        for kernel in kernels.iter().take(8) {
            let name = kernel.get("name").and_then(Value::as_str).unwrap_or("?");
            let pct = kernel.get("pct").and_then(Value::as_f64).unwrap_or(0.0);
            let gib_s = kernel.get("gib_s").and_then(Value::as_f64).unwrap_or(0.0);
            let avg_us = kernel.get("avg_us").and_then(Value::as_f64).unwrap_or(0.0);
            let role = kernel
                .get("op")
                .and_then(|op| op.get("role"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            out.push(format!(
                "  {:42} {:>6.2}% {:>9.2}us {:>8.1} GiB/s  {}",
                truncate(name, 42),
                pct,
                avg_us,
                gib_s,
                role
            ));
        }
    }
    out.join("\n")
}

fn empty_dash(s: &str) -> &str {
    if s.is_empty() {
        "-"
    } else {
        s
    }
}

fn scalar(value: &Value) -> String {
    if let Some(v) = value.as_f64() {
        format!("{v:.4}")
    } else if let Some(v) = value.as_str() {
        v.to_string()
    } else {
        value.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        format!("{s:max$}")
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}
