// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::schema::AtlasRow;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub id: String,
    pub title: String,
    pub rationale: String,
    pub expected_effect: String,
    pub confidence: String,
}

pub fn suggestions_for_row(row: &AtlasRow, max_suggestions: usize) -> Vec<Suggestion> {
    let mut out = Vec::new();
    if row.phase == "decode_ar" || row.workload_kind == "ar" {
        if let Some(gen_tok_s) = row.metric_f64("gen_tok_s") {
            out.push(Suggestion {
                id: "decode-hotpath-profile".to_string(),
                title: "Rank decode hot kernels before editing".to_string(),
                rationale: format!("AR row reports gen_tok_s={gen_tok_s:.2}; focus on the largest decode kernel share first."),
                expected_effect: "Avoids tuning cold kernels; produces a bounded Atlas task.".to_string(),
                confidence: "high".to_string(),
            });
        }
    }
    if row.phase == "prefill" {
        out.push(Suggestion {
            id: "prefill-shape-sweep".to_string(),
            title: "Sweep prefill shape buckets".to_string(),
            rationale: "Prefill kernels are often shape-sensitive; compare pp32/pp128/pp512 before changing ISA.".to_string(),
            expected_effect: "Separates kernel weakness from shape artifact.".to_string(),
            confidence: "medium".to_string(),
        });
    }
    if row.workload_kind == "dflash" {
        let tau = row.metric_f64("tau").unwrap_or(0.0);
        out.push(Suggestion {
            id: "dflash-tau-vs-wall".to_string(),
            title: "Optimize DFlash only when tau and wall time agree".to_string(),
            rationale: format!("DFlash row has tau={tau:.2}; high tau without output sanity is not a win."),
            expected_effect: "Keeps Atlas from ranking attractor failures as speedups.".to_string(),
            confidence: "high".to_string(),
        });
    }
    if let Some(kernels) = row.artifact_array("profile_kernels") {
        if let Some(kernel) = kernels.first() {
            let name = kernel.get("name").and_then(Value::as_str).unwrap_or("unknown");
            let pct = kernel.get("pct").and_then(Value::as_f64).unwrap_or(0.0);
            out.push(Suggestion {
                id: "hot-kernel-task".to_string(),
                title: format!("Create task for hot kernel {name}"),
                rationale: format!("{name} is first in the profile list at {pct:.2}% of measured time."),
                expected_effect: "Most likely single-kernel tuning target.".to_string(),
                confidence: if pct >= 15.0 { "high" } else { "medium" }.to_string(),
            });
        }
    }
    if out.is_empty() {
        out.push(Suggestion {
            id: "collect-profile".to_string(),
            title: "Collect profile kernels for this row".to_string(),
            rationale: "The row has metrics but no profile_kernels artifact.".to_string(),
            expected_effect: "Enables ISA/object join and concrete task generation.".to_string(),
            confidence: "high".to_string(),
        });
    }
    out.truncate(max_suggestions);
    out
}

pub fn suggestions_markdown(suggestions: &[Suggestion]) -> String {
    let mut out = Vec::new();
    for (idx, suggestion) in suggestions.iter().enumerate() {
        out.push(format!(
            "{}. {} [{}]\n   {}\n   Expected: {}",
            idx + 1,
            suggestion.title,
            suggestion.confidence,
            suggestion.rationale,
            suggestion.expected_effect
        ));
    }
    out.join("\n")
}
