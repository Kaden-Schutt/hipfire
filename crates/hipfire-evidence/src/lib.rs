// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Evidence provenance helpers shared by Hipfire eval and gate tooling.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use hipfire_hash::stable_hash_bytes;
use hipfire_model::HfqMetadata;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceArtifactSpec {
    pub file: &'static str,
    pub kind: &'static str,
    pub expected_metrics: &'static [&'static str],
}

pub const STANDARD_EVIDENCE_ARTIFACT_SPECS: &[EvidenceArtifactSpec] = &[
    EvidenceArtifactSpec {
        file: "quality.json",
        kind: "quality",
        expected_metrics: &[
            "mean_kld",
            "p99_kld",
            "ppl",
            "argmax_match_rate",
            "accuracy",
            "exact_match",
        ],
    },
    EvidenceArtifactSpec {
        file: "performance.json",
        kind: "performance",
        expected_metrics: &["pp32_ms", "pp128_ms", "ttft_ms", "tok_s"],
    },
    EvidenceArtifactSpec {
        file: "phase_timings.json",
        kind: "phase_timings",
        expected_metrics: &["load_ms", "prefill_ms", "decode_ms", "teardown_ms"],
    },
    EvidenceArtifactSpec {
        file: "launch_counts.json",
        kind: "launch_counts",
        expected_metrics: &["kernel_launches", "graph_launches", "memcpy_ops"],
    },
    EvidenceArtifactSpec {
        file: "moe_router_histogram.json",
        kind: "moe_router_histogram",
        expected_metrics: &["expert_hits", "shared_expert_hits", "router_entropy"],
    },
    EvidenceArtifactSpec {
        file: "memory.json",
        kind: "memory",
        expected_metrics: &["vram_peak_bytes", "kv_bytes", "workspace_bytes"],
    },
    EvidenceArtifactSpec {
        file: "dflash_trace.json",
        kind: "dflash_trace",
        expected_metrics: &["ar_tok_s", "dflash_tok_s", "accept_rate", "tau"],
    },
    EvidenceArtifactSpec {
        file: "path_c_trace.json",
        kind: "path_c_trace",
        expected_metrics: &[
            "tok_s",
            "tau",
            "promotion_verdict",
            "tok_s_delta_pct",
            "tau_delta_pct",
        ],
    },
    EvidenceArtifactSpec {
        file: "module_evidence.json",
        kind: "module_evidence",
        expected_metrics: &[
            "module_kind",
            "module_id",
            "preferred_backend",
            "selected_backend",
            "oracle_backend",
            "fallback_reason",
        ],
    },
    EvidenceArtifactSpec {
        file: "profiling.json",
        kind: "profiling",
        expected_metrics: &["kernel_name", "duration_us", "occupancy", "waves"],
    },
    EvidenceArtifactSpec {
        file: "coherence.json",
        kind: "coherence",
        expected_metrics: &[
            "hard_fails",
            "soft_warns",
            "detector_count",
            "coherence_status",
        ],
    },
];

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeOneshotEvidence<'a> {
    pub case_id: &'a str,
    pub prompt_path: &'a str,
    pub prompt_tokens: usize,
    pub emitted_tokens: usize,
    pub prefill_forward_calls: usize,
    pub decode_forward_calls: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub ttft_ms: f64,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
}

pub fn write_runtime_oneshot_evidence(
    dir: &Path,
    runtime_context: &Value,
    evidence: RuntimeOneshotEvidence<'_>,
) -> std::io::Result<()> {
    let prefill_tok_s = evidence.prompt_tokens as f64 / evidence.prefill_secs.max(1e-9);
    let decode_tok_s = evidence.emitted_tokens as f64 / evidence.decode_secs.max(1e-9);
    let base = json!({
        "case_id": evidence.case_id,
        "prompt_path": evidence.prompt_path,
        "hipfire_runtime_context": runtime_context,
        "metrics": {
            "prompt_tokens": evidence.prompt_tokens,
            "decode_tokens_emitted": evidence.emitted_tokens,
        },
    });

    write_json_pretty_value(
        &dir.join("performance.json"),
        &json!({
            "schema": 1,
            "kind": "performance",
            "hipfire_runtime_context": runtime_context,
            "records": [{
                "case_id": evidence.case_id,
                "prompt_path": evidence.prompt_path,
                "hipfire_runtime_context": runtime_context,
                "metrics": {
                    "tok_s": decode_tok_s,
                    "decode_tok_s": decode_tok_s,
                    "prefill_tok_s": prefill_tok_s,
                    "ttft_ms": evidence.ttft_ms,
                    "prompt_tokens": evidence.prompt_tokens,
                    "decode_tokens_emitted": evidence.emitted_tokens,
                }
            }]
        }),
    )?;
    write_json_pretty_value(
        &dir.join("phase_timings.json"),
        &json!({
            "schema": 1,
            "kind": "phase_timings",
            "hipfire_runtime_context": runtime_context,
            "records": [{
                "case_id": evidence.case_id,
                "prompt_path": evidence.prompt_path,
                "hipfire_runtime_context": runtime_context,
                "metrics": {
                    "prefill_ms": evidence.prefill_secs * 1000.0,
                    "decode_ms": evidence.decode_secs * 1000.0,
                    "ttft_ms": evidence.ttft_ms,
                }
            }]
        }),
    )?;
    write_json_pretty_value(
        &dir.join("memory.json"),
        &json!({
            "schema": 1,
            "kind": "memory",
            "hipfire_runtime_context": runtime_context,
            "records": [{
                "case_id": evidence.case_id,
                "prompt_path": evidence.prompt_path,
                "hipfire_runtime_context": runtime_context,
                "metrics": {
                    "vram_peak_bytes": (evidence.vram_used_mb as f64) * 1024.0 * 1024.0,
                    "vram_used_mb": evidence.vram_used_mb,
                    "vram_total_mb": evidence.vram_total_mb,
                }
            }]
        }),
    )?;
    write_json_pretty_value(
        &dir.join("launch_counts.json"),
        &json!({
            "schema": 1,
            "kind": "launch_counts",
            "hipfire_runtime_context": runtime_context,
            "records": [{
                "case_id": evidence.case_id,
                "prompt_path": evidence.prompt_path,
                "hipfire_runtime_context": runtime_context,
                "metrics": {
                    "kernel_launches": evidence.prefill_forward_calls + evidence.decode_forward_calls,
                    "graph_launches": 0,
                    "memcpy_ops": evidence.emitted_tokens,
                    "model_forward_calls": evidence.prefill_forward_calls + evidence.decode_forward_calls,
                    "prefill_forward_calls": evidence.prefill_forward_calls,
                    "decode_forward_calls": evidence.decode_forward_calls,
                    "counting_scope": "model_forward_call_proxy"
                },
                "notes": "examples/run currently exposes model forward-call counts, not raw HIP kernel launch counters"
            }]
        }),
    )?;
    write_json_pretty_value(
        &dir.join("run_oneshot.json"),
        &json!({
            "schema": 1,
            "kind": "run_oneshot",
            "hipfire_runtime_context": runtime_context,
            "records": [base]
        }),
    )
}

fn write_json_pretty_value(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(value)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    fs::write(path, format!("{body}\n"))
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterHistogramLayer {
    pub layer_idx: usize,
    pub top1_histogram: Vec<u64>,
    pub topk_histogram: Vec<u64>,
    pub weight_sums: Vec<f64>,
    pub dropped_indices: u64,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub cooccurrence: HashMap<u64, u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterHistogramEvidence<'a> {
    pub case_id: &'a str,
    pub prompt_path: &'a str,
    pub collection_scope: &'a str,
    pub num_experts: usize,
    pub k_top: usize,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub top1_histogram: Vec<u64>,
    pub topk_histogram: Vec<u64>,
    pub weight_sums: Vec<f64>,
    pub dropped_indices: u64,
    pub per_layer: Vec<RouterHistogramLayer>,
}

pub fn write_router_histogram_evidence(
    dir: &Path,
    runtime_context: &Value,
    evidence: RouterHistogramEvidence<'_>,
) -> std::io::Result<()> {
    if evidence.routed_slots == 0 {
        return Ok(());
    }
    write_json_pretty_value(
        &dir.join("moe_router_histogram.json"),
        &json!({
            "schema": 1,
            "kind": "moe_router_histogram",
            "hipfire_runtime_context": runtime_context,
            "records": [{
                "case_id": evidence.case_id,
                "prompt_path": evidence.prompt_path,
                "hipfire_runtime_context": runtime_context,
                "metrics": router_histogram_metrics_json(&evidence),
            }]
        }),
    )
}

pub fn router_histogram_metrics_json(evidence: &RouterHistogramEvidence<'_>) -> Value {
    json!({
        "expert_hits": sparse_u64_histogram(&evidence.topk_histogram),
        "router_top1_histogram": sparse_u64_histogram(&evidence.top1_histogram),
        "router_topk_histogram": sparse_u64_histogram(&evidence.topk_histogram),
        "router_weight_sums": sparse_f64_histogram(&evidence.weight_sums),
        "router_entropy": histogram_entropy(&evidence.topk_histogram),
        "router_dropped_tokens": evidence.dropped_indices,
        "routed_tokens": evidence.routed_tokens,
        "routed_slots": evidence.routed_slots,
        "num_experts": evidence.num_experts,
        "k_top": evidence.k_top,
        "per_layer_router_histograms": per_layer_router_histograms_json(evidence),
        "collection_scope": evidence.collection_scope,
    })
}

fn per_layer_router_histograms_json(evidence: &RouterHistogramEvidence<'_>) -> Value {
    let layers = evidence
        .per_layer
        .iter()
        .filter(|layer| layer.routed_slots > 0 || layer.dropped_indices > 0)
        .map(|layer| {
            json!({
                "layer_idx": layer.layer_idx,
                "expert_hits": sparse_u64_histogram(&layer.topk_histogram),
                "router_top1_histogram": sparse_u64_histogram(&layer.top1_histogram),
                "router_topk_histogram": sparse_u64_histogram(&layer.topk_histogram),
                "router_weight_sums": sparse_f64_histogram(&layer.weight_sums),
                "router_entropy": histogram_entropy(&layer.topk_histogram),
                "router_dropped_tokens": layer.dropped_indices,
                "routed_tokens": layer.routed_tokens,
                "routed_slots": layer.routed_slots,
                "topk_cooccurrence": sparse_pair_histogram_from_flat_keys(
                    &layer.cooccurrence,
                    evidence.num_experts,
                ),
            })
        })
        .collect::<Vec<_>>();
    json!(layers)
}

pub fn sparse_u64_histogram(values: &[u64]) -> Value {
    let mut object = serde_json::Map::new();
    for (idx, &count) in values.iter().enumerate() {
        if count > 0 {
            object.insert(idx.to_string(), json!(count));
        }
    }
    Value::Object(object)
}

pub fn sparse_f64_histogram(values: &[f64]) -> Value {
    let mut object = serde_json::Map::new();
    for (idx, &sum) in values.iter().enumerate() {
        if sum != 0.0 {
            object.insert(idx.to_string(), json!(sum));
        }
    }
    Value::Object(object)
}

pub fn histogram_entropy(values: &[u64]) -> f64 {
    let total: u64 = values.iter().sum();
    if total == 0 {
        return 0.0;
    }
    values
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total as f64;
            -p * p.ln()
        })
        .sum()
}

pub fn sparse_pair_histogram_from_flat_keys(
    cooccurrence: &HashMap<u64, u64>,
    modulus: usize,
) -> Value {
    if modulus == 0 {
        return json!({});
    }
    let mut entries = cooccurrence
        .iter()
        .filter(|(_, &count)| count > 0)
        .map(|(&key, &count)| {
            let a = key / modulus as u64;
            let b = key % modulus as u64;
            (a, b, count)
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });

    let mut object = serde_json::Map::new();
    for (a, b, count) in entries {
        object.insert(format!("{a},{b}"), json!(count));
    }
    Value::Object(object)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionEvidenceRequirement {
    pub kind: &'static str,
    pub batteries: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionVerdictPolicy {
    pub status: &'static str,
    pub verdict: &'static str,
    pub reason: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceCollectionPolicy {
    pub status: &'static str,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceArtifactCollection {
    pub source: String,
    pub executor: String,
    pub evidence_json: Vec<String>,
    pub evidence_dirs: Vec<String>,
    pub requires_model_execution: bool,
    pub profiling_mode: String,
    pub dflash_mode: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceArtifactConfig {
    pub tier: String,
    pub batteries: Vec<String>,
    pub suites: Vec<String>,
    pub kv_mode: Option<String>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceArtifactModels {
    pub candidate: String,
    pub draft: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceArtifactDatasetStatus {
    pub total: usize,
    pub pass: usize,
    pub skip: usize,
    pub fail: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceArtifact {
    pub kind: String,
    pub provenance: Value,
    pub collection_policy: EvidenceCollectionPolicy,
    pub collection: EvidenceArtifactCollection,
    pub config: EvidenceArtifactConfig,
    pub models: EvidenceArtifactModels,
    pub datasets: EvidenceArtifactDatasetStatus,
    pub expected_metrics: Vec<String>,
    pub records: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonArtifact {
    pub provenance: RunProvenance,
    pub status: String,
    pub reason: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
    pub cases: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionEvidence {
    pub kind: String,
    pub status: String,
    pub rows: usize,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionArtifact {
    pub provenance: RunProvenance,
    pub status: String,
    pub verdict: String,
    pub reason: Option<String>,
    pub required_evidence: Vec<AdmissionEvidence>,
    pub observed_evidence: Vec<AdmissionEvidence>,
    pub findings: Vec<Value>,
}

pub const OBSERVED_ADMISSION_EVIDENCE_KINDS: &[&str] = &[
    "phase_timings",
    "launch_counts",
    "moe_router_histogram",
    "memory",
    "dflash_trace",
    "path_c_trace",
    "module_evidence",
    "coherence",
    "profiling",
];

pub const MODULE_EVIDENCE_TRIGGER_METRICS: &[&str] = &[
    "module_kind",
    "module_id",
    "preferred_backend",
    "selected_backend",
    "oracle_backend",
    "fallback_reason",
    "evidence",
    "evidence_json",
];

pub const MODULE_EVIDENCE_JSON_METRICS: &[&str] = &[
    "module_kind",
    "module_id",
    "preferred_backend",
    "selected_backend",
    "oracle_backend",
    "fallback_reason",
    "drift",
    "shape",
    "contract",
    "evidence",
    "evidence_json",
    "mutates_residual",
];

pub const MODULE_EVIDENCE_NUMERIC_METRICS: &[&str] = &[
    "layer",
    "hidden",
    "intermediate",
    "n",
    "max_abs",
    "mean_abs",
    "rms",
    "nan",
    "inf",
];

pub const LAUNCH_COUNT_METRICS: &[&str] = &[
    "kernel_launches",
    "graph_launches",
    "memcpy_ops",
    "launch_count",
    "hip_kernel_launches",
    "hip_graph_launches",
    "hip_memcpy_ops",
];

pub const MOE_ROUTER_METRICS: &[&str] = &[
    "expert_hits",
    "shared_expert_hits",
    "router_entropy",
    "router_top1_histogram",
    "router_top2_histogram",
    "router_topk_histogram",
    "router_dropped_tokens",
];

pub const PROFILING_METRICS: &[&str] = &[
    "kernel_name",
    "duration_us",
    "occupancy",
    "waves",
    "lds_bytes",
    "vgpr_count",
    "sgpr_count",
];

pub const PHASE_TIMING_TRIGGER_METRICS: &[&str] = &[
    "load_ms",
    "prefill_ms",
    "prefill_secs",
    "decode_ms",
    "decode_secs",
    "teardown_ms",
    "ttft_ms",
    "elapsed_ms",
];

pub const MEMORY_TRIGGER_METRICS: &[&str] = &[
    "vram_peak_bytes",
    "vram_used_bytes",
    "vram_used_mb",
    "vram_loaded_mb",
    "kv_bytes",
    "workspace_bytes",
];

pub const PERFORMANCE_TRIGGER_METRICS: &[&str] = &[
    "tok_s",
    "tokens_per_second",
    "ttft_ms",
    "decode_ms",
    "decode_secs",
    "decode_tok_s",
    "prefill_ms",
    "prefill_secs",
    "prefill_tok_s",
    "total_ms",
    "elapsed_ms",
    "launch_count",
];

pub const QUALITY_TRIGGER_METRICS: &[&str] = &[
    "mean_kld",
    "p99_kld",
    "ppl",
    "nll",
    "argmax_match_rate",
    "accuracy",
    "exact_match",
];

pub const DFLASH_TRACE_NUMERIC_METRICS: &[&str] =
    &["ar_tok_s", "dflash_tok_s", "tau", "accept_rate", "tok_s"];

pub const DFLASH_TRACE_BOOL_METRICS: &[&str] = &["ar_baseline"];

pub const DFLASH_TRACE_JSON_METRICS: &[&str] = &[
    "rollback_logit_compare",
    "rollback_state_compare",
    "rollback_fast_replay_admission",
    "rollback_wo_delta_compare",
];

pub const DFLASH_TRACE_TRIGGER_METRICS: &[&str] = &[
    "ar_tok_s",
    "dflash_tok_s",
    "tok_s",
    "ar_baseline",
    "rollback_logit_compare",
    "rollback_state_compare",
    "rollback_fast_replay_admission",
    "rollback_wo_delta_compare",
];

pub const PATH_C_TRACE_JSON_METRICS: &[&str] = &[
    "mode",
    "phase",
    "graph_mode",
    "detector",
    "verify_graph",
    "path_c_counters",
    "promotion_verdict",
    "blockers",
    "pairs",
    "extra_args",
    "token_attractor_detector",
];

pub const PATH_C_TRACE_NUMERIC_METRICS: &[&str] = &[
    "tok_s",
    "tau",
    "accept_rate",
    "emitted_tokens",
    "wall_s",
    "paired_cases",
    "tok_s_min_delta_pct",
    "tau_min_delta_pct",
    "max_tokens",
];

pub const PATH_C_TRACE_TRIGGER_METRICS: &[&str] = &["promotion_verdict"];

pub fn has_any_metric(metrics: &BTreeMap<String, Value>, keys: &[&str]) -> bool {
    keys.iter().any(|key| metrics.contains_key(*key))
}

pub fn select_metrics(metrics: &BTreeMap<String, Value>, keys: &[&str]) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for key in keys {
        if let Some(value) = metrics.get(*key) {
            out.insert((*key).to_string(), value.clone());
        }
    }
    out
}

pub fn has_launch_count_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, LAUNCH_COUNT_METRICS)
}

pub fn launch_count_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    select_metrics(metrics, LAUNCH_COUNT_METRICS)
}

pub fn has_moe_router_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, MOE_ROUTER_METRICS)
}

pub fn moe_router_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    select_metrics(metrics, MOE_ROUTER_METRICS)
}

pub fn has_profiling_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, PROFILING_METRICS)
}

pub fn profiling_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    select_metrics(metrics, PROFILING_METRICS)
}

pub fn has_phase_timing_metric(metrics: &BTreeMap<String, Value>, elapsed_ms: u128) -> bool {
    has_any_metric(metrics, PHASE_TIMING_TRIGGER_METRICS) || elapsed_ms > 0
}

pub fn phase_timing_metrics(
    metrics: &BTreeMap<String, Value>,
    elapsed_ms: u128,
) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    copy_numeric_metric(metrics, &mut out, "load_ms", "load_ms");
    copy_numeric_metric(metrics, &mut out, "prefill_ms", "prefill_ms");
    copy_secs_as_ms(metrics, &mut out, "prefill_secs", "prefill_ms");
    copy_numeric_metric(metrics, &mut out, "decode_ms", "decode_ms");
    copy_secs_as_ms(metrics, &mut out, "decode_secs", "decode_ms");
    copy_numeric_metric(metrics, &mut out, "teardown_ms", "teardown_ms");
    copy_numeric_metric(metrics, &mut out, "ttft_ms", "ttft_ms");
    if !out.contains_key("elapsed_ms") {
        out.insert("elapsed_ms".to_string(), json!(elapsed_ms));
    }
    out
}

pub fn has_memory_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, MEMORY_TRIGGER_METRICS)
}

pub fn has_performance_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, PERFORMANCE_TRIGGER_METRICS)
}

pub fn has_quality_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, QUALITY_TRIGGER_METRICS)
}

pub fn memory_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    copy_numeric_metric(metrics, &mut out, "vram_peak_bytes", "vram_peak_bytes");
    copy_numeric_metric(metrics, &mut out, "vram_used_bytes", "vram_peak_bytes");
    copy_mb_as_bytes(metrics, &mut out, "vram_used_mb", "vram_peak_bytes");
    copy_mb_as_bytes(metrics, &mut out, "vram_loaded_mb", "vram_peak_bytes");
    copy_numeric_metric(metrics, &mut out, "kv_bytes", "kv_bytes");
    copy_numeric_metric(metrics, &mut out, "workspace_bytes", "workspace_bytes");
    out
}

pub fn has_module_evidence_metric(metrics: &BTreeMap<String, Value>) -> bool {
    MODULE_EVIDENCE_TRIGGER_METRICS
        .iter()
        .any(|key| metrics.contains_key(*key))
}

pub fn module_evidence_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for key in MODULE_EVIDENCE_JSON_METRICS {
        copy_json_metric(metrics, &mut out, key, key);
    }
    for key in MODULE_EVIDENCE_NUMERIC_METRICS {
        copy_numeric_metric(metrics, &mut out, key, key);
    }
    out
}

pub fn has_dflash_trace_metric(metrics: &BTreeMap<String, Value>) -> bool {
    has_any_metric(metrics, DFLASH_TRACE_TRIGGER_METRICS)
}

pub fn dflash_trace_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for key in DFLASH_TRACE_NUMERIC_METRICS {
        copy_numeric_metric(metrics, &mut out, key, key);
    }
    for key in DFLASH_TRACE_BOOL_METRICS {
        copy_bool_metric(metrics, &mut out, key, key);
    }
    for key in DFLASH_TRACE_JSON_METRICS {
        copy_json_metric(metrics, &mut out, key, key);
    }

    if let Some(ar_baseline) = metrics.get("ar_baseline").and_then(Value::as_bool) {
        out.insert(
            "mode".to_string(),
            json!(if ar_baseline { "ar" } else { "dflash" }),
        );
        if ar_baseline {
            if !out.contains_key("ar_tok_s") {
                copy_numeric_metric(metrics, &mut out, "tok_s", "ar_tok_s");
            }
        } else if !out.contains_key("dflash_tok_s") {
            copy_numeric_metric(metrics, &mut out, "tok_s", "dflash_tok_s");
        }
    } else if metrics.contains_key("ar_tok_s") || metrics.contains_key("dflash_tok_s") {
        out.insert("mode".to_string(), json!("aggregate"));
    }

    out
}

pub fn has_path_c_trace_metric(metrics: &BTreeMap<String, Value>) -> bool {
    metrics
        .get("mode")
        .and_then(Value::as_str)
        .is_some_and(|mode| mode.starts_with("path-c"))
        || has_any_metric(metrics, PATH_C_TRACE_TRIGGER_METRICS)
}

pub fn path_c_trace_metrics(metrics: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for key in PATH_C_TRACE_JSON_METRICS {
        copy_json_metric(metrics, &mut out, key, key);
    }
    for key in PATH_C_TRACE_NUMERIC_METRICS {
        copy_numeric_metric(metrics, &mut out, key, key);
    }
    out
}

fn copy_json_metric(
    source: &BTreeMap<String, Value>,
    dest: &mut BTreeMap<String, Value>,
    source_key: &str,
    dest_key: &str,
) {
    if dest.contains_key(dest_key) {
        return;
    }
    if let Some(value) = source.get(source_key) {
        dest.insert(dest_key.to_string(), value.clone());
    }
}

fn copy_numeric_metric(
    source: &BTreeMap<String, Value>,
    dest: &mut BTreeMap<String, Value>,
    source_key: &str,
    dest_key: &str,
) {
    if dest.contains_key(dest_key) {
        return;
    }
    if let Some(value) = source.get(source_key).and_then(Value::as_f64) {
        dest.insert(dest_key.to_string(), json!(value));
    }
}

fn copy_bool_metric(
    source: &BTreeMap<String, Value>,
    dest: &mut BTreeMap<String, Value>,
    source_key: &str,
    dest_key: &str,
) {
    if dest.contains_key(dest_key) {
        return;
    }
    if let Some(value) = source.get(source_key).and_then(Value::as_bool) {
        dest.insert(dest_key.to_string(), json!(value));
    }
}

fn copy_secs_as_ms(
    source: &BTreeMap<String, Value>,
    dest: &mut BTreeMap<String, Value>,
    source_key: &str,
    dest_key: &str,
) {
    if dest.contains_key(dest_key) {
        return;
    }
    if let Some(value) = source.get(source_key).and_then(Value::as_f64) {
        dest.insert(dest_key.to_string(), json!(value * 1000.0));
    }
}

fn copy_mb_as_bytes(
    source: &BTreeMap<String, Value>,
    dest: &mut BTreeMap<String, Value>,
    source_key: &str,
    dest_key: &str,
) {
    if dest.contains_key(dest_key) {
        return;
    }
    if let Some(value) = source.get(source_key).and_then(Value::as_f64) {
        dest.insert(dest_key.to_string(), json!(value * 1024.0 * 1024.0));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceRecord {
    pub battery: String,
    pub suite: Option<String>,
    pub case_id: String,
    pub dataset_item_id: Option<String>,
    pub dataset_source: Option<String>,
    pub dataset_repo_id: Option<String>,
    pub dataset_revision: Option<String>,
    pub dataset_digest: Option<String>,
    pub dataset_license: Option<String>,
    pub dataset_cache_path: Option<String>,
    pub model: String,
    pub model_hash: Option<String>,
    pub draft: Option<String>,
    pub draft_hash: Option<String>,
    pub baseline: Option<String>,
    pub baseline_hash: Option<String>,
    pub reference: Option<String>,
    pub reference_hash: Option<String>,
    pub prompt_hash: Option<String>,
    pub prompt_path: Option<String>,
    pub metrics: BTreeMap<String, Value>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    pub runner: String,
    pub runner_version: String,
    pub hipfire_version: String,
    pub git_commit: Option<String>,
    pub git_branch: Option<String>,
    pub git_describe: Option<String>,
    pub git_dirty: Option<bool>,
    pub binary_hash: Option<String>,
    pub arch: Option<String>,
    pub rocm: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EvidenceArtifactIndexContext {
    pub provenance: RunProvenance,
    pub host_profile_hash: String,
    pub hardware_bucket: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalStatus {
    Pass,
    Fail,
    Skip,
}

impl EvalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            EvalStatus::Pass => "pass",
            EvalStatus::Fail => "fail",
            EvalStatus::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostProfile {
    pub schema: u32,
    pub source: String,
    pub probe_status: EvalStatus,
    pub reason: Option<String>,
    pub hardware_kind: String,
    pub hardware_bucket: String,
    pub host_profile_hash: String,
    pub gpu_model: Option<String>,
    pub gfx: Option<String>,
    pub vendor_id: Option<String>,
    pub device_id: Option<String>,
    pub render_node: Option<String>,
    pub cu_count: Option<u32>,
    pub vram_bytes: Option<u64>,
    pub gtt_bytes: Option<u64>,
    pub system_memory_bytes: Option<u64>,
    pub memory_class: SourcedField<String>,
    pub memory_width_bits: SourcedField<u32>,
    pub memory_clock_mhz: SourcedField<f64>,
    pub peak_bandwidth_gbps: SourcedField<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcedField<T> {
    pub value: Option<T>,
    pub source: String,
    pub confidence: String,
    pub note: Option<String>,
}

impl<T> SourcedField<T> {
    pub fn unknown() -> Self {
        Self {
            value: None,
            source: "unavailable".to_string(),
            confidence: "unknown".to_string(),
            note: None,
        }
    }

    pub fn override_value(value: T) -> Self {
        Self {
            value: Some(value),
            source: "cli_override".to_string(),
            confidence: "operator_supplied".to_string(),
            note: None,
        }
    }

    pub fn libdrm_value(value: T) -> Self {
        Self {
            value: Some(value),
            source: "libdrm_amdgpu".to_string(),
            confidence: "high".to_string(),
            note: None,
        }
    }

    pub fn sysfs_value(value: T) -> Self {
        Self {
            value: Some(value),
            source: "linux_sysfs".to_string(),
            confidence: "medium".to_string(),
            note: None,
        }
    }

    pub fn computed_value(value: T) -> Self {
        Self {
            value: Some(value),
            source: "computed".to_string(),
            confidence: "medium".to_string(),
            note: None,
        }
    }
}

pub fn compute_peak_bandwidth_gbps(
    clock_mhz: f64,
    width_bits: u32,
    memory_class: &str,
) -> Option<f64> {
    let transfers_per_clock = match memory_class.to_ascii_lowercase().as_str() {
        "gddr6" | "gddr6x" => 8.0,
        "lpddr5" | "lpddr5x" => 8.0,
        "ddr5" | "ddr4" => 2.0,
        "hbm" | "hbm2" | "hbm2e" | "hbm3" => 2.0,
        _ => return None,
    };
    Some(clock_mhz * transfers_per_clock * width_bits as f64 / 8.0 / 1000.0)
}

pub fn classify_hardware_kind(vram_bytes: Option<u64>, gtt_bytes: Option<u64>) -> String {
    match (vram_bytes, gtt_bytes) {
        (Some(vram), Some(gtt)) if vram <= 1024 * 1024 * 1024 && gtt > vram * 8 => {
            "apu_uma".to_string()
        }
        (Some(vram), _) if vram > 1024 * 1024 * 1024 => "dgpu".to_string(),
        _ => "unknown".to_string(),
    }
}

pub fn hardware_bucket(
    hardware_kind: &str,
    gfx: Option<&str>,
    device_id: Option<&str>,
    cu_count: Option<u32>,
    vram_bytes: Option<u64>,
    memory_class: Option<&str>,
    memory_width_bits: Option<u32>,
    peak_bandwidth_gbps: Option<f64>,
) -> String {
    let vram_gib = vram_bytes.map(|bytes| (bytes + (1 << 30) - 1) >> 30);
    let bandwidth = peak_bandwidth_gbps.map(|value| format!("{value:.0}gbps"));
    [
        hardware_kind.to_string(),
        gfx.unwrap_or("gfx_unknown").to_string(),
        device_id.unwrap_or("dev_unknown").to_string(),
        cu_count
            .map(|value| format!("{value}cu"))
            .unwrap_or_else(|| "cu_unknown".to_string()),
        vram_gib
            .map(|value| format!("{value}gib"))
            .unwrap_or_else(|| "vram_unknown".to_string()),
        memory_class.unwrap_or("mem_unknown").to_string(),
        memory_width_bits
            .map(|value| format!("{value}bit"))
            .unwrap_or_else(|| "width_unknown".to_string()),
        bandwidth.unwrap_or_else(|| "bw_unknown".to_string()),
    ]
    .join(":")
}

pub fn host_profile_hash(profile: &HostProfile) -> String {
    let doc = json!({
        "schema": profile.schema,
        "hardware_kind": profile.hardware_kind,
        "hardware_bucket": profile.hardware_bucket,
        "gpu_model": profile.gpu_model,
        "gfx": profile.gfx,
        "vendor_id": profile.vendor_id,
        "device_id": profile.device_id,
        "cu_count": profile.cu_count,
        "vram_bytes": profile.vram_bytes,
        "gtt_bytes": profile.gtt_bytes,
        "system_memory_bytes": profile.system_memory_bytes,
        "memory_class": profile.memory_class,
        "memory_width_bits": profile.memory_width_bits,
        "memory_clock_mhz": profile.memory_clock_mhz,
        "peak_bandwidth_gbps": profile.peak_bandwidth_gbps,
    });
    stable_hash_bytes(serde_json::to_string(&doc).unwrap_or_default().as_bytes())
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadataConfig {
    pub tier: String,
    pub tier_budget: Value,
    pub batteries: Vec<String>,
    pub suites: Vec<String>,
    pub executor: String,
    pub kv_mode: Option<String>,
    pub max_tokens: usize,
    pub profile: String,
    pub dflash: String,
    pub runs: usize,
    pub warmup_runs: usize,
    pub benchmark: bool,
    pub host_memory_class: Option<String>,
    pub host_memory_width_bits: Option<u32>,
    pub host_memory_bandwidth_gbps: Option<f64>,
    pub result_cache: String,
    pub cache_mode: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadataModels {
    pub candidate: String,
    pub draft: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadataArtifact {
    pub created_utc: String,
    pub provenance: RunProvenance,
    pub host_profile: Value,
    pub host_profile_hash: String,
    pub hardware_bucket: String,
    pub config: RunMetadataConfig,
    pub models: RunMetadataModels,
}

pub fn run_provenance_json(provenance: RunProvenance) -> Value {
    serde_json::to_value(provenance).unwrap_or_else(|_| json!({}))
}

pub fn run_metadata_artifact_json(metadata: RunMetadataArtifact) -> Value {
    json!({
        "schema": 1,
        "kind": "run_metadata",
        "status": "collected",
        "runner": metadata.provenance.runner,
        "runner_version": metadata.provenance.runner_version,
        "hipfire_version": metadata.provenance.hipfire_version,
        "created_utc": metadata.created_utc,
        "git": {
            "commit": metadata.provenance.git_commit,
            "branch": metadata.provenance.git_branch,
            "describe": metadata.provenance.git_describe,
            "dirty": metadata.provenance.git_dirty,
        },
        "binary": {
            "hash": metadata.provenance.binary_hash,
        },
        "host": {
            "arch": metadata.provenance.arch,
            "rocm": metadata.provenance.rocm,
            "profile": metadata.host_profile,
            "host_profile_hash": metadata.host_profile_hash,
            "hardware_bucket": metadata.hardware_bucket,
        },
        "config": {
            "tier": metadata.config.tier,
            "tier_budget": metadata.config.tier_budget,
            "batteries": metadata.config.batteries,
            "suites": metadata.config.suites,
            "executor": metadata.config.executor,
            "kv_mode": metadata.config.kv_mode,
            "max_tokens": metadata.config.max_tokens,
            "profile": metadata.config.profile,
            "dflash": metadata.config.dflash,
            "runs": metadata.config.runs,
            "warmup_runs": metadata.config.warmup_runs,
            "benchmark": metadata.config.benchmark,
            "host_memory_class": metadata.config.host_memory_class,
            "host_memory_width_bits": metadata.config.host_memory_width_bits,
            "host_memory_bandwidth_gbps": metadata.config.host_memory_bandwidth_gbps,
            "result_cache": metadata.config.result_cache,
            "cache_mode": metadata.config.cache_mode,
        },
        "models": {
            "candidate": metadata.models.candidate,
            "draft": metadata.models.draft,
            "baseline": metadata.models.baseline,
            "reference": metadata.models.reference,
        },
    })
}

pub fn evidence_artifact_index_entry_json(
    path: impl Into<String>,
    status: impl Into<String>,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    json!({
        "path": path.into(),
        "status": status.into(),
        "runner_version": context.provenance.runner_version,
        "hipfire_version": context.provenance.hipfire_version,
        "git_commit": context.provenance.git_commit,
        "git_branch": context.provenance.git_branch,
        "git_describe": context.provenance.git_describe,
        "git_dirty": context.provenance.git_dirty,
        "binary_hash": context.provenance.binary_hash,
        "arch": context.provenance.arch,
        "rocm": context.provenance.rocm,
        "host_profile_hash": context.host_profile_hash,
        "hardware_bucket": context.hardware_bucket,
    })
}

pub fn evidence_artifact_index_entry_from_value_json(
    path: impl Into<String>,
    status: impl Into<String>,
    value: &Value,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    let mut entry = evidence_artifact_index_entry_json(path, status, context);
    if let Some(object) = entry.as_object_mut() {
        if let Some(records) = value.get("records").and_then(Value::as_array) {
            object.insert("row_count".to_string(), json!(records.len()));
        }
        if let Some(reason) = value.get("reason").cloned() {
            object.insert("reason".to_string(), reason);
        }
        if let Some(metrics) = value.get("expected_metrics").cloned() {
            object.insert("expected_metrics".to_string(), metrics);
        }
        if let Some(kind) = value.get("kind").cloned() {
            object.insert("kind".to_string(), kind);
        }
    }
    entry
}

pub fn comparison_artifact_index_entry_json(
    path: impl Into<String>,
    status: impl Into<String>,
    case_count: usize,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    let mut entry = evidence_artifact_index_entry_json(path, status, context);
    if let Some(object) = entry.as_object_mut() {
        object.insert("case_count".to_string(), json!(case_count));
    }
    entry
}

pub fn admission_artifact_index_entry_json(
    path: impl Into<String>,
    status: impl Into<String>,
    verdict: impl Into<String>,
    finding_count: usize,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    let mut entry = evidence_artifact_index_entry_json(path, status, context);
    if let Some(object) = entry.as_object_mut() {
        object.insert("verdict".to_string(), json!(verdict.into()));
        object.insert("finding_count".to_string(), json!(finding_count));
    }
    entry
}

pub fn prompt_artifact_index_entry_json(
    path: impl Into<String>,
    status: impl Into<String>,
    kind: impl Into<String>,
    row_count: usize,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    let mut entry = evidence_artifact_index_entry_json(path, status, context);
    if let Some(object) = entry.as_object_mut() {
        object.insert("row_count".to_string(), json!(row_count));
        object.insert("kind".to_string(), json!(kind.into()));
    }
    entry
}

pub fn host_profile_artifact_index_entry_json(
    path: impl Into<String>,
    status: impl Into<String>,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    let mut entry = evidence_artifact_index_entry_json(path, status, context);
    if let Some(object) = entry.as_object_mut() {
        object.insert("kind".to_string(), json!("host_capability_profile"));
    }
    entry
}

pub fn evidence_record_json(record: EvidenceRecord) -> Value {
    json!({
        "battery": record.battery,
        "suite": record.suite,
        "case_id": record.case_id,
        "dataset_item_id": record.dataset_item_id,
        "dataset_source": record.dataset_source,
        "dataset_repo_id": record.dataset_repo_id,
        "dataset_revision": record.dataset_revision,
        "dataset_digest": record.dataset_digest,
        "dataset_license": record.dataset_license,
        "dataset_cache_path": record.dataset_cache_path,
        "model": record.model,
        "model_hash": record.model_hash,
        "draft": record.draft,
        "draft_hash": record.draft_hash,
        "baseline": record.baseline,
        "baseline_hash": record.baseline_hash,
        "reference": record.reference,
        "reference_hash": record.reference_hash,
        "prompt_hash": record.prompt_hash,
        "prompt_path": record.prompt_path,
        "metrics": record.metrics,
        "elapsed_ms": record.elapsed_ms,
    })
}

pub fn evidence_metric_direction(metric: &str, delta: f64) -> String {
    let lower_is_better = [
        "mean_kld",
        "p99_kld",
        "ppl",
        "nll",
        "ttft_ms",
        "decode_ms",
        "prefill_ms",
        "elapsed_ms",
    ]
    .iter()
    .any(|prefix| metric == *prefix || metric.ends_with(prefix));
    let higher_is_better = [
        "tok_s",
        "tokens_per_second",
        "accept_rate",
        "tau",
        "accuracy",
        "exact_match",
    ]
    .iter()
    .any(|prefix| metric == *prefix || metric.ends_with(prefix));

    if lower_is_better {
        if delta < 0.0 {
            "improved".to_string()
        } else if delta > 0.0 {
            "regressed".to_string()
        } else {
            "unchanged".to_string()
        }
    } else if higher_is_better {
        if delta > 0.0 {
            "improved".to_string()
        } else if delta < 0.0 {
            "regressed".to_string()
        } else {
            "unchanged".to_string()
        }
    } else if delta == 0.0 {
        "unchanged".to_string()
    } else {
        "changed".to_string()
    }
}

pub fn admission_metric_is_quality(battery: &str, metric: &str) -> bool {
    matches!(battery, "quality" | "barrage")
        || matches!(
            metric,
            "mean_kld" | "p99_kld" | "ppl" | "nll" | "accuracy" | "exact_match"
        )
}

pub fn required_admission_evidence_requirements(
    selected_batteries: impl IntoIterator<Item = impl AsRef<str>>,
) -> Vec<AdmissionEvidenceRequirement> {
    let selected = selected_batteries
        .into_iter()
        .map(|battery| battery.as_ref().to_string())
        .collect::<Vec<_>>();
    let mut required = Vec::new();
    if selected
        .iter()
        .any(|battery| matches!(battery.as_str(), "quality" | "barrage"))
    {
        required.push(AdmissionEvidenceRequirement {
            kind: "quality",
            batteries: vec!["quality"],
        });
    }
    if selected
        .iter()
        .any(|battery| matches!(battery.as_str(), "speed" | "dflash" | "pflash"))
    {
        required.push(AdmissionEvidenceRequirement {
            kind: "performance",
            batteries: vec!["speed", "dflash", "pflash"],
        });
    }
    if selected.iter().any(|battery| battery == "barrage") {
        required.push(AdmissionEvidenceRequirement {
            kind: "barrage",
            batteries: vec!["barrage"],
        });
    }
    required
}

pub fn admission_verdict_policy(has_reject: bool, has_review: bool) -> AdmissionVerdictPolicy {
    if has_reject {
        AdmissionVerdictPolicy {
            status: "fail",
            verdict: "reject",
            reason: Some("quality or correctness regression detected"),
        }
    } else if has_review {
        AdmissionVerdictPolicy {
            status: "pass",
            verdict: "review",
            reason: Some("performance regression detected; quality evidence did not reject"),
        }
    } else {
        AdmissionVerdictPolicy {
            status: "pass",
            verdict: "promote",
            reason: None,
        }
    }
}

pub fn evidence_collection_policy(
    kind: &str,
    record_count: usize,
    external_errors: &[String],
    profile_mode: &str,
) -> EvidenceCollectionPolicy {
    if !external_errors.is_empty() {
        return EvidenceCollectionPolicy {
            status: "fail",
            reason: Some(external_errors.join("; ")),
        };
    }
    if record_count > 0 {
        return EvidenceCollectionPolicy {
            status: "collected",
            reason: None,
        };
    }
    if kind == "profiling" && profile_mode == "off" {
        return EvidenceCollectionPolicy {
            status: "disabled",
            reason: Some("profiling disabled by --profile off".to_string()),
        };
    }
    if kind == "profiling" && profile_mode == "passive" {
        return EvidenceCollectionPolicy {
            status: "requested",
            reason: Some(
                "passive profiling requested; model-backed profiler collector is not implemented in this harness revision"
                    .to_string(),
            ),
        };
    }
    EvidenceCollectionPolicy {
        status: "not_collected",
        reason: Some(
            "model-backed collection is not implemented in this harness revision".to_string(),
        ),
    }
}

pub fn evidence_artifact_json(artifact: EvidenceArtifact) -> Value {
    json!({
        "schema": 1,
        "kind": artifact.kind,
        "provenance": artifact.provenance,
        "status": artifact.collection_policy.status,
        "reason": artifact.collection_policy.reason,
        "collection": {
            "source": artifact.collection.source,
            "executor": artifact.collection.executor,
            "evidence_json": artifact.collection.evidence_json,
            "evidence_dirs": artifact.collection.evidence_dirs,
            "requires_model_execution": artifact.collection.requires_model_execution,
            "profiling_mode": artifact.collection.profiling_mode,
            "dflash_mode": artifact.collection.dflash_mode,
        },
        "config": {
            "tier": artifact.config.tier,
            "batteries": artifact.config.batteries,
            "suites": artifact.config.suites,
            "kv_mode": artifact.config.kv_mode,
            "max_tokens": artifact.config.max_tokens,
        },
        "models": {
            "candidate": artifact.models.candidate,
            "draft": artifact.models.draft,
            "baseline": artifact.models.baseline,
            "reference": artifact.models.reference,
        },
        "datasets": {
            "total": artifact.datasets.total,
            "pass": artifact.datasets.pass,
            "skip": artifact.datasets.skip,
            "fail": artifact.datasets.fail,
        },
        "expected_metrics": artifact.expected_metrics,
        "records": artifact.records,
    })
}

pub fn comparison_artifact_json(artifact: ComparisonArtifact) -> Value {
    json!({
        "schema": 1,
        "provenance": artifact.provenance,
        "status": artifact.status,
        "reason": artifact.reason,
        "baseline": artifact.baseline,
        "reference": artifact.reference,
        "cases": artifact.cases,
    })
}

pub fn admission_artifact_json(artifact: AdmissionArtifact) -> Value {
    json!({
        "schema": 1,
        "provenance": artifact.provenance,
        "status": artifact.status,
        "verdict": artifact.verdict,
        "reason": artifact.reason,
        "required_evidence": artifact.required_evidence.into_iter().map(admission_evidence_json).collect::<Vec<_>>(),
        "observed_evidence": artifact.observed_evidence.into_iter().map(admission_evidence_json).collect::<Vec<_>>(),
        "findings": artifact.findings,
    })
}

fn admission_evidence_json(evidence: AdmissionEvidence) -> Value {
    json!({
        "kind": evidence.kind,
        "status": evidence.status,
        "rows": evidence.rows,
        "reason": evidence.reason,
    })
}

pub fn extract_external_evidence_records_json(
    kind: &str,
    path: &Path,
    value: &Value,
    context: Value,
) -> Vec<Value> {
    let Some(selected) = select_external_evidence_value(kind, path, value) else {
        return Vec::new();
    };
    let records = if let Some(records) = selected.get("records").and_then(Value::as_array) {
        records.clone()
    } else if let Some(records) = selected.as_array() {
        records.clone()
    } else {
        vec![selected.clone()]
    };
    records
        .into_iter()
        .map(|record| annotate_external_evidence_record_json(kind, path, record, &context))
        .collect()
}

pub fn standard_evidence_artifact_kind_for_path(path: &Path) -> Option<&'static str> {
    let file_name = path.file_name()?.to_str()?;
    STANDARD_EVIDENCE_ARTIFACT_SPECS
        .iter()
        .find(|spec| spec.file == file_name)
        .map(|spec| spec.kind)
}

pub fn standard_evidence_paths_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let entries =
        fs::read_dir(dir).map_err(|err| format!("read evidence dir {}: {err}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|err| format!("read evidence dir entry {}: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && standard_evidence_artifact_kind_for_path(&path).is_some() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

pub fn model_hash(model: &str) -> Option<String> {
    hipfire_model::model_hash(model)
}

pub fn directory_hash(path: &Path) -> Option<String> {
    let files = list_files(path);
    if files.is_empty() {
        return None;
    }
    Some(stable_hash_bytes(files.join("\n").as_bytes()))
}

pub fn read_hfq_metadata(path: &Path) -> Result<HfqMetadata, String> {
    hipfire_model::read_hfq_metadata(path)
}

/// Verify the reference file's sha256 against the in-tree
/// `manifest.json` index.
///
/// Layout assumption: ref lives at `.../refs/<name>.kldref.bin`, manifest
/// at `.../harness/manifest.json` (sibling to `refs/`).
///
/// Behaviour:
/// - if the manifest is absent OR has no entry for `<name>`, emit a
///   warning and return (developer pre-upload state);
/// - if sha256 disagrees, print a clear error and `std::process::exit(2)`.
///
/// `tool_name` is the binary's short name (e.g. `"eval_hipfire"`) and is
/// used only in log lines.
pub fn verify_ref_sha256(ref_path: &Path, tool_name: &str) {
    let manifest_path = match ref_path.parent().and_then(|p| p.parent()) {
        Some(p) => p.join("harness").join("manifest.json"),
        None => {
            eprintln!(
                "warning: cannot locate harness/manifest.json relative to {}; \
                 skipping ref sha256 check",
                ref_path.display()
            );
            return;
        }
    };
    if !manifest_path.exists() {
        eprintln!(
            "warning: {} missing; skipping ref sha256 check",
            manifest_path.display()
        );
        return;
    }
    let manifest_file = std::fs::File::open(&manifest_path).expect("open manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_reader(manifest_file).expect("parse manifest.json");
    let ref_name = ref_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let expected = manifest
        .get("references")
        .and_then(|r| r.get(ref_name))
        .and_then(|r| r.get("sha256"))
        .and_then(|s| s.as_str())
        .map(String::from);
    let expected = match expected {
        Some(s) => s,
        None => {
            eprintln!("warning: no manifest entry / sha256 for {ref_name}; skipping check");
            return;
        }
    };
    eprintln!(
        "{tool_name}: computing sha256 of {} ...",
        ref_path.display()
    );
    let out = Command::new("sha256sum")
        .arg(ref_path)
        .output()
        .expect("invoke sha256sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty sha256sum output");
    if actual != expected {
        eprintln!("ERROR: ref sha256 mismatch for {}", ref_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified ref sha256 = {actual}");
}

/// Verify the slice file's md5 against the sibling `slice.md5` (one-line
/// `<md5>` or `md5sum`-format output).
///
/// Behaviour:
/// - if `<slice_dir>/slice.md5` is absent OR no recognisable hash on the
///   first line, emit a warning and return;
/// - if md5 disagrees, print a clear error and `std::process::exit(2)`.
pub fn verify_slice_md5(slice_path: &Path, tool_name: &str) {
    let md5_path = match slice_path.parent() {
        Some(p) => p.join("slice.md5"),
        None => return,
    };
    if !md5_path.exists() {
        eprintln!(
            "warning: {} missing; skipping slice md5 check",
            md5_path.display()
        );
        return;
    }
    let expected_line = match std::fs::read_to_string(&md5_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "warning: cannot read {}: {e}; skipping slice md5",
                md5_path.display()
            );
            return;
        }
    };
    let expected = match expected_line.split_whitespace().next() {
        Some(s) => s.to_string(),
        None => {
            eprintln!("warning: {} empty; skipping", md5_path.display());
            return;
        }
    };
    let out = Command::new("md5sum")
        .arg(slice_path)
        .output()
        .expect("invoke md5sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty md5sum output");
    if actual != expected {
        eprintln!("ERROR: slice md5 mismatch for {}", slice_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified slice md5 = {actual}");
}

/// Verify that the supplied `llama-perplexity` binary's reported commit
/// hash matches `pinned`.
///
/// Behaviour:
/// - parses `<bin> --version`'s "version: N (hash)" line;
/// - demands the binary's hash be >= 7 chars (collision floor on short
///   git hashes);
/// - compares an equal-length prefix of `pinned` to the binary's hash.
///
/// On any mismatch, `std::process::exit(2)`.
pub fn verify_llama_commit(bin: &str, pinned: &str, tool_name: &str) {
    let out = Command::new(bin).arg("--version").output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ERROR: failed to invoke `{bin} --version`: {e}");
            std::process::exit(2);
        }
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let needle = "version: ";
    let after_version = match combined.find(needle) {
        Some(i) => &combined[i + needle.len()..],
        None => {
            eprintln!("ERROR: could not find 'version: ' in `{bin} --version` output");
            std::process::exit(2);
        }
    };
    let open = match after_version.find('(') {
        Some(i) => i,
        None => {
            eprintln!("ERROR: malformed `--version` output: no '(' after version number");
            std::process::exit(2);
        }
    };
    let after_paren = &after_version[open + 1..];
    let close = match after_paren.find(')') {
        Some(i) => i,
        None => {
            eprintln!("ERROR: malformed `--version` output: no ')' after commit hash");
            std::process::exit(2);
        }
    };
    let hash = &after_paren[..close];
    if hash.len() < 7 {
        eprintln!(
            "ERROR: llama-perplexity reported a {}-char hash; want >= 7",
            hash.len()
        );
        eprintln!("  binary:    {bin}");
        eprintln!("  reported:  {hash}");
        std::process::exit(2);
    }
    if hash.len() > pinned.len() {
        eprintln!(
            "ERROR: llama-perplexity hash ({}) longer than pinned ({})",
            hash.len(),
            pinned.len()
        );
        std::process::exit(2);
    }
    let pinned_prefix = &pinned[..hash.len()];
    if hash != pinned_prefix {
        eprintln!("ERROR: llama-perplexity commit mismatch");
        eprintln!("  binary:             {bin}");
        eprintln!("  expected (pinned):  {pinned}");
        eprintln!("  actual (--version): {hash}");
        eprintln!("  Either rebuild llama.cpp at the pinned commit, or update");
        eprintln!("  PINNED_LLAMACPP_COMMIT in this binary AND in the PRD.");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified llama.cpp commit prefix {hash}");
}

fn select_external_evidence_value<'a>(
    kind: &str,
    path: &Path,
    value: &'a Value,
) -> Option<&'a Value> {
    if value
        .get("kind")
        .and_then(Value::as_str)
        .is_some_and(|k| k == kind)
    {
        return Some(value);
    }
    if let Some(mapped) = value.get(kind) {
        return Some(mapped);
    }
    if path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem == kind)
    {
        return Some(value);
    }
    None
}

fn annotate_external_evidence_record_json(
    kind: &str,
    path: &Path,
    record: Value,
    context: &Value,
) -> Value {
    let source_path = path.display().to_string();
    match record {
        Value::Object(mut object) => {
            object
                .entry("kind".to_string())
                .or_insert_with(|| json!(kind));
            object
                .entry("source_path".to_string())
                .or_insert_with(|| json!(source_path));
            object
                .entry("hipfire_eval_context".to_string())
                .or_insert_with(|| context.clone());
            Value::Object(object)
        }
        other => json!({
            "kind": kind,
            "source_path": source_path,
            "hipfire_eval_context": context,
            "value": other,
        }),
    }
}

pub fn list_files(path: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_files_relative(path, path, &mut out);
    out.sort();
    out
}

fn collect_files_relative(root: &Path, path: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_files_relative(root, &p, out);
        } else if p.is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn stable_hash_bytes_is_deterministic() {
        assert_eq!(stable_hash_bytes(b"abc"), "fnv64:e71fa2190541574b");
        assert_eq!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abc"));
        assert_ne!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abcd"));
    }

    #[test]
    fn computes_peak_bandwidth_from_normalized_memory_fields() {
        assert_eq!(
            compute_peak_bandwidth_gbps(937.5, 256, "lpddr5x"),
            Some(240.0)
        );
        assert_eq!(
            compute_peak_bandwidth_gbps(2500.0, 256, "gddr6"),
            Some(640.0)
        );
        assert_eq!(compute_peak_bandwidth_gbps(1000.0, 128, "mystery"), None);
    }

    #[test]
    fn classifies_hardware_kind_from_vram_and_gtt_shape() {
        assert_eq!(
            classify_hardware_kind(Some(512 * 1024 * 1024), Some(16 * 1024 * 1024 * 1024)),
            "apu_uma"
        );
        assert_eq!(
            classify_hardware_kind(Some(16 * 1024 * 1024 * 1024), Some(32 * 1024 * 1024 * 1024)),
            "dgpu"
        );
        assert_eq!(classify_hardware_kind(None, None), "unknown");
    }

    #[test]
    fn hardware_bucket_includes_portability_fields() {
        let bucket = hardware_bucket(
            "dgpu",
            Some("gfx1201"),
            Some("0x7550"),
            Some(64),
            Some(16 * 1024 * 1024 * 1024),
            Some("gddr6"),
            Some(256),
            Some(640.0),
        );
        assert_eq!(
            bucket,
            "dgpu:gfx1201:0x7550:64cu:16gib:gddr6:256bit:640gbps"
        );
    }

    #[test]
    fn host_profile_hash_ignores_probe_source_and_reason() {
        let mut profile = HostProfile {
            schema: 1,
            source: "linux-kfd-drm-sysfs".to_string(),
            probe_status: EvalStatus::Pass,
            reason: None,
            hardware_kind: "dgpu".to_string(),
            hardware_bucket: "dgpu:gfx1201:0x7550:64cu:16gib:gddr6:256bit:640gbps".to_string(),
            host_profile_hash: String::new(),
            gpu_model: Some("AMD Radeon".to_string()),
            gfx: Some("gfx1201".to_string()),
            vendor_id: Some("0x1002".to_string()),
            device_id: Some("0x7550".to_string()),
            render_node: Some("/dev/dri/renderD128".to_string()),
            cu_count: Some(64),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
            gtt_bytes: Some(32 * 1024 * 1024 * 1024),
            system_memory_bytes: Some(64 * 1024 * 1024 * 1024),
            memory_class: SourcedField::libdrm_value("gddr6".to_string()),
            memory_width_bits: SourcedField::libdrm_value(256),
            memory_clock_mhz: SourcedField::libdrm_value(2500.0),
            peak_bandwidth_gbps: SourcedField::computed_value(640.0),
        };
        let hash = host_profile_hash(&profile);

        profile.source = "override".to_string();
        profile.reason = Some("metadata source changed".to_string());

        assert_eq!(hash, host_profile_hash(&profile));
    }

    #[test]
    fn model_hash_tags_non_files() {
        assert_eq!(
            model_hash("qwen3.5:9b").unwrap(),
            format!("tag:{}", stable_hash_bytes(b"qwen3.5:9b"))
        );
    }

    #[test]
    fn directory_hash_uses_relative_file_list() {
        let root = temp_dir("hipfire-evidence-dir-hash");
        fs::create_dir_all(root.join("nested")).unwrap();
        File::create(root.join("a.txt")).unwrap();
        File::create(root.join("nested/b.txt")).unwrap();
        assert_eq!(
            directory_hash(&root).unwrap(),
            stable_hash_bytes(b"a.txt\nnested/b.txt")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn standard_artifact_catalog_preserves_schema_names() {
        let specs: Vec<_> = STANDARD_EVIDENCE_ARTIFACT_SPECS
            .iter()
            .map(|spec| (spec.file, spec.kind))
            .collect();
        assert_eq!(
            specs,
            vec![
                ("quality.json", "quality"),
                ("performance.json", "performance"),
                ("phase_timings.json", "phase_timings"),
                ("launch_counts.json", "launch_counts"),
                ("moe_router_histogram.json", "moe_router_histogram"),
                ("memory.json", "memory"),
                ("dflash_trace.json", "dflash_trace"),
                ("path_c_trace.json", "path_c_trace"),
                ("module_evidence.json", "module_evidence"),
                ("profiling.json", "profiling"),
                ("coherence.json", "coherence"),
            ]
        );
        assert!(STANDARD_EVIDENCE_ARTIFACT_SPECS
            .iter()
            .all(|spec| !spec.expected_metrics.is_empty()));
    }

    #[test]
    fn evidence_record_json_preserves_artifact_row_shape() {
        let mut metrics = BTreeMap::new();
        metrics.insert("tok_s".to_string(), json!(123.4));
        let record = EvidenceRecord {
            battery: "speed".to_string(),
            suite: Some("smoke".to_string()),
            case_id: "case-a".to_string(),
            dataset_item_id: Some("item-a".to_string()),
            dataset_source: None,
            dataset_repo_id: None,
            dataset_revision: None,
            dataset_digest: None,
            dataset_license: None,
            dataset_cache_path: None,
            model: "model.hfq".to_string(),
            model_hash: Some("sha256:model".to_string()),
            draft: None,
            draft_hash: None,
            baseline: Some("baseline.hfq".to_string()),
            baseline_hash: Some("sha256:baseline".to_string()),
            reference: None,
            reference_hash: None,
            prompt_hash: Some("fnv64:prompt".to_string()),
            prompt_path: Some("benchmarks/prompts/smoke.txt".to_string()),
            metrics,
            elapsed_ms: 42,
        };

        let json = evidence_record_json(record);
        assert_eq!(json["battery"], "speed");
        assert_eq!(json["suite"], "smoke");
        assert_eq!(json["case_id"], "case-a");
        assert_eq!(json["dataset_item_id"], "item-a");
        assert_eq!(json["model_hash"], "sha256:model");
        assert_eq!(json["baseline_hash"], "sha256:baseline");
        assert_eq!(json["prompt_path"], "benchmarks/prompts/smoke.txt");
        assert_eq!(json["metrics"]["tok_s"], 123.4);
        assert_eq!(json["elapsed_ms"], 42);
    }

    #[test]
    fn module_evidence_metrics_select_owned_schema_fields() {
        let mut metrics = BTreeMap::new();
        metrics.insert("module_kind".to_string(), json!("dense_ffn_swiglu_down"));
        metrics.insert("module_id".to_string(), json!("qwen35.layer4"));
        metrics.insert("preferred_backend".to_string(), json!("gpu_production"));
        metrics.insert("selected_backend".to_string(), json!("gpu_production"));
        metrics.insert("oracle_backend".to_string(), json!("cpu_oracle"));
        metrics.insert("fallback_reason".to_string(), json!("none"));
        metrics.insert("drift".to_string(), json!({"max_abs": 0.001}));
        metrics.insert("shape".to_string(), json!([1, 4096]));
        metrics.insert("mutates_residual".to_string(), json!(false));
        metrics.insert("layer".to_string(), json!(4));
        metrics.insert("max_abs".to_string(), json!(0.001));
        metrics.insert("nan".to_string(), json!(0));
        metrics.insert("unrelated".to_string(), json!("drop"));

        assert!(has_module_evidence_metric(&metrics));
        let selected = module_evidence_metrics(&metrics);
        assert_eq!(selected["module_kind"], json!("dense_ffn_swiglu_down"));
        assert_eq!(selected["drift"]["max_abs"], json!(0.001));
        assert_eq!(selected["layer"], json!(4.0));
        assert_eq!(selected["max_abs"], json!(0.001));
        assert_eq!(selected["nan"], json!(0.0));
        assert!(!selected.contains_key("unrelated"));
    }

    #[test]
    fn runtime_metric_projections_select_owned_schema_fields() {
        let mut metrics = BTreeMap::new();
        metrics.insert("kernel_launches".to_string(), json!(12));
        metrics.insert("hip_graph_launches".to_string(), json!(2));
        metrics.insert("expert_hits".to_string(), json!([1, 2, 3]));
        metrics.insert("router_entropy".to_string(), json!(0.7));
        metrics.insert("kernel_name".to_string(), json!("gemv"));
        metrics.insert("duration_us".to_string(), json!(42.5));
        metrics.insert("unrelated".to_string(), json!("drop"));

        assert!(has_launch_count_metric(&metrics));
        assert!(has_moe_router_metric(&metrics));
        assert!(has_profiling_metric(&metrics));

        let launch = launch_count_metrics(&metrics);
        assert_eq!(launch["kernel_launches"], json!(12));
        assert_eq!(launch["hip_graph_launches"], json!(2));
        assert!(!launch.contains_key("expert_hits"));
        assert!(!launch.contains_key("unrelated"));

        let moe = moe_router_metrics(&metrics);
        assert_eq!(moe["expert_hits"], json!([1, 2, 3]));
        assert_eq!(moe["router_entropy"], json!(0.7));
        assert!(!moe.contains_key("kernel_name"));

        let profiling = profiling_metrics(&metrics);
        assert_eq!(profiling["kernel_name"], json!("gemv"));
        assert_eq!(profiling["duration_us"], json!(42.5));
        assert!(!profiling.contains_key("kernel_launches"));
    }

    #[test]
    fn phase_and_memory_metric_projections_preserve_eval_aliases() {
        let mut phase = BTreeMap::new();
        phase.insert("prefill_secs".to_string(), json!(1.25));
        phase.insert("decode_ms".to_string(), json!(9.0));
        phase.insert("ttft_ms".to_string(), json!(3.0));

        assert!(has_phase_timing_metric(&phase, 42));
        let selected_phase = phase_timing_metrics(&phase, 42);
        assert_eq!(selected_phase["prefill_ms"], json!(1250.0));
        assert_eq!(selected_phase["decode_ms"], json!(9.0));
        assert_eq!(selected_phase["ttft_ms"], json!(3.0));
        assert_eq!(selected_phase["elapsed_ms"], json!(42));

        let mut memory = BTreeMap::new();
        memory.insert("vram_used_mb".to_string(), json!(2.5));
        memory.insert("kv_bytes".to_string(), json!(1024));
        memory.insert("workspace_bytes".to_string(), json!(2048));

        assert!(has_memory_metric(&memory));
        let selected_memory = memory_metrics(&memory);
        assert_eq!(
            selected_memory["vram_peak_bytes"],
            json!(2.5 * 1024.0 * 1024.0)
        );
        assert_eq!(selected_memory["kv_bytes"], json!(1024.0));
        assert_eq!(selected_memory["workspace_bytes"], json!(2048.0));
    }

    #[test]
    fn performance_metric_detection_preserves_eval_aliases() {
        for key in PERFORMANCE_TRIGGER_METRICS {
            let metrics = BTreeMap::from([((*key).to_string(), json!(1.0))]);
            assert!(has_performance_metric(&metrics), "{key}");
        }

        let unrelated = BTreeMap::from([("mean_kld".to_string(), json!(0.01))]);
        assert!(!has_performance_metric(&unrelated));
    }

    #[test]
    fn quality_metric_detection_preserves_eval_aliases() {
        for key in QUALITY_TRIGGER_METRICS {
            let metrics = BTreeMap::from([((*key).to_string(), json!(1.0))]);
            assert!(has_quality_metric(&metrics), "{key}");
        }

        let unrelated = BTreeMap::from([("tok_s".to_string(), json!(128.0))]);
        assert!(!has_quality_metric(&unrelated));
    }

    #[test]
    fn dflash_trace_metrics_normalize_ar_and_dflash_rows() {
        let ar = BTreeMap::from([
            ("tok_s".to_string(), json!(90.0)),
            ("ar_baseline".to_string(), json!(true)),
        ]);
        assert!(has_dflash_trace_metric(&ar));
        let ar_trace = dflash_trace_metrics(&ar);
        assert_eq!(ar_trace.get("mode"), Some(&json!("ar")));
        assert_eq!(ar_trace.get("ar_tok_s"), Some(&json!(90.0)));

        let dflash = BTreeMap::from([
            ("tok_s".to_string(), json!(130.0)),
            ("ar_baseline".to_string(), json!(false)),
            ("tau".to_string(), json!(2.5)),
            ("accept_rate".to_string(), json!(0.6)),
            (
                "rollback_state_compare".to_string(),
                json!({
                    "ok": false,
                    "reason": "fast_replay_recurrent_state_mismatch",
                }),
            ),
        ]);
        let dflash_trace = dflash_trace_metrics(&dflash);
        assert_eq!(dflash_trace.get("mode"), Some(&json!("dflash")));
        assert_eq!(dflash_trace.get("dflash_tok_s"), Some(&json!(130.0)));
        assert_eq!(dflash_trace.get("tau"), Some(&json!(2.5)));
        assert_eq!(
            dflash_trace
                .get("rollback_state_compare")
                .and_then(|value| value.get("reason")),
            Some(&json!("fast_replay_recurrent_state_mismatch"))
        );

        let aggregate = BTreeMap::from([("dflash_tok_s".to_string(), json!(144.0))]);
        assert_eq!(
            dflash_trace_metrics(&aggregate).get("mode"),
            Some(&json!("aggregate"))
        );
    }

    #[test]
    fn path_c_trace_metrics_select_owned_schema_fields() {
        let dflash = BTreeMap::from([
            ("mode".to_string(), json!("dflash")),
            ("tok_s".to_string(), json!(130.0)),
        ]);
        assert!(!has_path_c_trace_metric(&dflash));

        let path_c = BTreeMap::from([
            ("mode".to_string(), json!("path-c-phase1")),
            ("tok_s".to_string(), json!(5.86)),
            ("tau".to_string(), json!(1.23)),
            ("max_tokens".to_string(), json!(192)),
            (
                "verify_graph".to_string(),
                json!({
                    "direct": 0,
                    "warmup": 3,
                    "capture": 3,
                    "replay": 81,
                }),
            ),
            ("ignored".to_string(), json!("not exported")),
        ]);
        assert!(has_path_c_trace_metric(&path_c));
        let trace = path_c_trace_metrics(&path_c);
        assert_eq!(trace.get("mode"), Some(&json!("path-c-phase1")));
        assert_eq!(trace.get("tok_s"), Some(&json!(5.86)));
        assert_eq!(
            trace
                .get("verify_graph")
                .and_then(|value| value.get("replay")),
            Some(&json!(81))
        );
        assert!(!trace.contains_key("ignored"));

        let promotion = BTreeMap::from([
            ("promotion_verdict".to_string(), json!("NOT_PROMOTED")),
            (
                "blockers".to_string(),
                json!(["path-c-phase2-code: tok/s delta -3.732% < 5.000%"]),
            ),
        ]);
        assert!(has_path_c_trace_metric(&promotion));
        assert_eq!(
            path_c_trace_metrics(&promotion).get("promotion_verdict"),
            Some(&json!("NOT_PROMOTED"))
        );
    }

    #[test]
    fn evidence_metric_direction_classifies_known_metric_semantics() {
        assert_eq!(evidence_metric_direction("mean_kld", -0.01), "improved");
        assert_eq!(evidence_metric_direction("mean_kld", 0.01), "regressed");
        assert_eq!(evidence_metric_direction("decode_ms", -1.0), "improved");
        assert_eq!(evidence_metric_direction("tok_s", 10.0), "improved");
        assert_eq!(evidence_metric_direction("tok_s", -10.0), "regressed");
        assert_eq!(evidence_metric_direction("accuracy", 0.1), "improved");
        assert_eq!(evidence_metric_direction("exact_match", -0.1), "regressed");
        assert_eq!(
            evidence_metric_direction("phase_prefill_ms", 0.1),
            "regressed"
        );
        assert_eq!(
            evidence_metric_direction("decode_tokens_per_second", 0.1),
            "improved"
        );
        assert_eq!(evidence_metric_direction("custom_metric", 1.0), "changed");
        assert_eq!(evidence_metric_direction("custom_metric", 0.0), "unchanged");
    }

    #[test]
    fn admission_metric_quality_policy_classifies_reject_metrics() {
        assert!(admission_metric_is_quality("quality", "tok_s"));
        assert!(admission_metric_is_quality("barrage", "latency_ms"));
        assert!(admission_metric_is_quality("speed", "mean_kld"));
        assert!(admission_metric_is_quality("dflash", "accuracy"));
        assert!(admission_metric_is_quality("coherence", "exact_match"));
        assert!(!admission_metric_is_quality("speed", "tok_s"));
        assert!(!admission_metric_is_quality("dflash", "accept_rate"));
        assert!(!admission_metric_is_quality("coherence", "hard_fails"));
    }

    #[test]
    fn required_admission_evidence_catalog_preserves_eval_policy() {
        let default = required_admission_evidence_requirements(["quality", "speed"]);
        assert_eq!(
            default,
            vec![
                AdmissionEvidenceRequirement {
                    kind: "quality",
                    batteries: vec!["quality"],
                },
                AdmissionEvidenceRequirement {
                    kind: "performance",
                    batteries: vec!["speed", "dflash", "pflash"],
                },
            ]
        );

        let speed_only = required_admission_evidence_requirements(["speed"]);
        assert_eq!(
            speed_only,
            vec![AdmissionEvidenceRequirement {
                kind: "performance",
                batteries: vec!["speed", "dflash", "pflash"],
            }]
        );

        let with_barrage = required_admission_evidence_requirements(["quality", "barrage"]);
        assert_eq!(with_barrage.len(), 2);
        assert_eq!(with_barrage[1].kind, "barrage");
        assert_eq!(with_barrage[1].batteries, vec!["barrage"]);
    }

    #[test]
    fn observed_admission_evidence_catalog_preserves_runtime_telemetry_kinds() {
        assert_eq!(
            OBSERVED_ADMISSION_EVIDENCE_KINDS,
            &[
                "phase_timings",
                "launch_counts",
                "moe_router_histogram",
                "memory",
                "dflash_trace",
                "path_c_trace",
                "module_evidence",
                "coherence",
                "profiling",
            ]
        );
    }

    #[test]
    fn admission_verdict_policy_preserves_eval_outcomes() {
        assert_eq!(
            admission_verdict_policy(true, false),
            AdmissionVerdictPolicy {
                status: "fail",
                verdict: "reject",
                reason: Some("quality or correctness regression detected"),
            }
        );
        assert_eq!(
            admission_verdict_policy(false, true),
            AdmissionVerdictPolicy {
                status: "pass",
                verdict: "review",
                reason: Some("performance regression detected; quality evidence did not reject"),
            }
        );
        assert_eq!(
            admission_verdict_policy(false, false),
            AdmissionVerdictPolicy {
                status: "pass",
                verdict: "promote",
                reason: None,
            }
        );
        assert_eq!(admission_verdict_policy(true, true).verdict, "reject");
    }

    #[test]
    fn evidence_collection_policy_preserves_eval_artifact_status() {
        assert_eq!(
            evidence_collection_policy("quality", 2, &[], "off"),
            EvidenceCollectionPolicy {
                status: "collected",
                reason: None,
            }
        );
        assert_eq!(
            evidence_collection_policy("quality", 0, &["bad external evidence".to_string()], "off"),
            EvidenceCollectionPolicy {
                status: "fail",
                reason: Some("bad external evidence".to_string()),
            }
        );
        assert_eq!(
            evidence_collection_policy("profiling", 0, &[], "off"),
            EvidenceCollectionPolicy {
                status: "disabled",
                reason: Some("profiling disabled by --profile off".to_string()),
            }
        );
        assert_eq!(
            evidence_collection_policy("profiling", 0, &[], "passive"),
            EvidenceCollectionPolicy {
                status: "requested",
                reason: Some(
                    "passive profiling requested; model-backed profiler collector is not implemented in this harness revision"
                        .to_string(),
                ),
            }
        );
        assert_eq!(
            evidence_collection_policy("phase_timings", 0, &[], "off"),
            EvidenceCollectionPolicy {
                status: "not_collected",
                reason: Some(
                    "model-backed collection is not implemented in this harness revision"
                        .to_string()
                ),
            }
        );
    }

    #[test]
    fn evidence_artifact_json_preserves_eval_artifact_shape() {
        let artifact = EvidenceArtifact {
            kind: "quality".to_string(),
            provenance: json!({"runner": "hipfire-eval"}),
            collection_policy: EvidenceCollectionPolicy {
                status: "collected",
                reason: None,
            },
            collection: EvidenceArtifactCollection {
                source: "hipfire-eval".to_string(),
                executor: "mock".to_string(),
                evidence_json: vec!["quality.json".to_string()],
                evidence_dirs: vec!["artifacts".to_string()],
                requires_model_execution: true,
                profiling_mode: "off".to_string(),
                dflash_mode: "auto".to_string(),
            },
            config: EvidenceArtifactConfig {
                tier: "fast".to_string(),
                batteries: vec!["quality".to_string()],
                suites: vec!["gpqa".to_string()],
                kv_mode: Some("paged".to_string()),
                max_tokens: 32,
            },
            models: EvidenceArtifactModels {
                candidate: "candidate.hfq".to_string(),
                draft: Some("draft.hfq".to_string()),
                baseline: Some("baseline.hfq".to_string()),
                reference: None,
            },
            datasets: EvidenceArtifactDatasetStatus {
                total: 3,
                pass: 2,
                skip: 1,
                fail: 0,
            },
            expected_metrics: vec!["mean_kld".to_string(), "ppl".to_string()],
            records: vec![json!({"metrics": {"mean_kld": 0.01}})],
        };

        let value = evidence_artifact_json(artifact);
        assert_eq!(value["schema"], json!(1));
        assert_eq!(value["kind"], json!("quality"));
        assert_eq!(value["status"], json!("collected"));
        assert_eq!(value["reason"], Value::Null);
        assert_eq!(value["collection"]["source"], json!("hipfire-eval"));
        assert_eq!(value["config"]["kv_mode"], json!("paged"));
        assert_eq!(value["models"]["draft"], json!("draft.hfq"));
        assert_eq!(value["datasets"]["pass"], json!(2));
        assert_eq!(value["expected_metrics"], json!(["mean_kld", "ppl"]));
        assert_eq!(value["records"][0]["metrics"]["mean_kld"], json!(0.01));
    }

    #[test]
    fn comparison_artifact_json_preserves_eval_artifact_shape() {
        let artifact = ComparisonArtifact {
            provenance: sample_provenance(),
            status: "pass".to_string(),
            reason: None,
            baseline: Some("baseline.hfq".to_string()),
            reference: Some("reference.hfq".to_string()),
            cases: vec![json!({
                "key": "quality||kld_reference_slice|",
                "battery": "quality",
                "suite": null,
                "case_id": "kld_reference_slice",
                "dataset_item_id": null,
                "baseline": {
                    "model": "baseline.hfq",
                    "metrics": {
                        "mean_kld": {
                            "candidate": 0.01,
                            "comparator": 0.02,
                            "delta": -0.01,
                            "relative_delta": -0.5,
                            "direction": "improved",
                        }
                    }
                },
                "reference": null,
            })],
        };

        let value = comparison_artifact_json(artifact);
        assert_eq!(value["schema"], json!(1));
        assert_eq!(value["status"], json!("pass"));
        assert_eq!(value["reason"], Value::Null);
        assert_eq!(value["baseline"], json!("baseline.hfq"));
        assert_eq!(value["reference"], json!("reference.hfq"));
        assert_eq!(value["cases"][0]["battery"], json!("quality"));
        assert_eq!(
            value["cases"][0]["baseline"]["metrics"]["mean_kld"]["direction"],
            json!("improved")
        );
    }

    #[test]
    fn admission_artifact_json_preserves_eval_artifact_shape() {
        let artifact = AdmissionArtifact {
            provenance: sample_provenance(),
            status: "fail".to_string(),
            verdict: "reject".to_string(),
            reason: Some("quality or correctness regression detected".to_string()),
            required_evidence: vec![AdmissionEvidence {
                kind: "quality".to_string(),
                status: "pass".to_string(),
                rows: 1,
                reason: None,
            }],
            observed_evidence: vec![AdmissionEvidence {
                kind: "profiling".to_string(),
                status: "skip".to_string(),
                rows: 0,
                reason: Some("profiling disabled by --profile off".to_string()),
            }],
            findings: vec![json!({
                "severity": "reject",
                "battery": "quality",
                "suite": null,
                "case_id": "kld_reference_slice",
                "dataset_item_id": null,
                "comparator": "baseline",
                "metric": "mean_kld",
                "direction": "regressed",
                "delta": 0.1,
                "relative_delta": 0.25,
            })],
        };

        let value = admission_artifact_json(artifact);
        assert_eq!(value["schema"], json!(1));
        assert_eq!(value["status"], json!("fail"));
        assert_eq!(value["verdict"], json!("reject"));
        assert_eq!(
            value["reason"],
            json!("quality or correctness regression detected")
        );
        assert_eq!(value["required_evidence"][0]["status"], json!("pass"));
        assert_eq!(value["observed_evidence"][0]["status"], json!("skip"));
        assert_eq!(value["findings"][0]["metric"], json!("mean_kld"));
    }

    #[test]
    fn run_provenance_json_preserves_artifact_shape() {
        let provenance = sample_provenance();

        let json = run_provenance_json(provenance);
        assert_eq!(json["runner"], "hipfire-eval");
        assert_eq!(json["runner_version"], "0.2.0");
        assert_eq!(json["hipfire_version"], "0.2.0");
        assert_eq!(json["git_commit"], "abc123");
        assert_eq!(json["git_dirty"], false);
        assert_eq!(json["binary_hash"], "sha256:binary");
        assert_eq!(json["arch"], "gfx1151");
        assert_eq!(json["rocm"], "6.4");
    }

    fn sample_provenance() -> RunProvenance {
        RunProvenance {
            runner: "hipfire-eval".to_string(),
            runner_version: "0.2.0".to_string(),
            hipfire_version: "0.2.0".to_string(),
            git_commit: Some("abc123".to_string()),
            git_branch: Some("main".to_string()),
            git_describe: Some("v0.2.0-1-gabc123".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("sha256:binary".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("6.4".to_string()),
        }
    }

    #[test]
    fn run_metadata_artifact_json_preserves_runtime_schema() {
        let artifact = run_metadata_artifact_json(RunMetadataArtifact {
            created_utc: "2026-06-14T21:00:00Z".to_string(),
            provenance: RunProvenance {
                runner: "hipfire-eval".to_string(),
                runner_version: "0.2.0".to_string(),
                hipfire_version: "0.2.0".to_string(),
                git_commit: Some("abc123".to_string()),
                git_branch: Some("main".to_string()),
                git_describe: Some("v0.2.0-1-gabc123".to_string()),
                git_dirty: Some(true),
                binary_hash: Some("sha256:binary".to_string()),
                arch: Some("gfx1151".to_string()),
                rocm: Some("6.4".to_string()),
            },
            host_profile: json!({
                "schema": 1,
                "hardware_kind": "gpu",
            }),
            host_profile_hash: "host:abc".to_string(),
            hardware_bucket: "gfx1151:64g".to_string(),
            config: RunMetadataConfig {
                tier: "fast".to_string(),
                tier_budget: json!({
                    "target_max_seconds": 60,
                    "ci_suitable": true,
                }),
                batteries: vec!["smoke".to_string(), "quality".to_string()],
                suites: vec!["canary".to_string()],
                executor: "mock".to_string(),
                kv_mode: Some("q8".to_string()),
                max_tokens: 32,
                profile: "off".to_string(),
                dflash: "off".to_string(),
                runs: 2,
                warmup_runs: 1,
                benchmark: true,
                host_memory_class: Some("gddr6".to_string()),
                host_memory_width_bits: Some(256),
                host_memory_bandwidth_gbps: Some(512.5),
                result_cache: "/tmp/cache".to_string(),
                cache_mode: "use".to_string(),
            },
            models: RunMetadataModels {
                candidate: "candidate.hfq".to_string(),
                draft: Some("draft.hfq".to_string()),
                baseline: Some("baseline.hfq".to_string()),
                reference: None,
            },
        });

        assert_eq!(artifact["schema"], 1);
        assert_eq!(artifact["kind"], "run_metadata");
        assert_eq!(artifact["status"], "collected");
        assert_eq!(artifact["runner"], "hipfire-eval");
        assert_eq!(artifact["runner_version"], "0.2.0");
        assert_eq!(artifact["hipfire_version"], "0.2.0");
        assert_eq!(artifact["created_utc"], "2026-06-14T21:00:00Z");
        assert_eq!(artifact["git"]["commit"], "abc123");
        assert_eq!(artifact["git"]["dirty"], true);
        assert_eq!(artifact["binary"]["hash"], "sha256:binary");
        assert_eq!(artifact["host"]["arch"], "gfx1151");
        assert_eq!(artifact["host"]["rocm"], "6.4");
        assert_eq!(artifact["host"]["profile"]["hardware_kind"], "gpu");
        assert_eq!(artifact["host"]["host_profile_hash"], "host:abc");
        assert_eq!(artifact["host"]["hardware_bucket"], "gfx1151:64g");
        assert_eq!(artifact["config"]["tier"], "fast");
        assert_eq!(artifact["config"]["tier_budget"]["target_max_seconds"], 60);
        assert_eq!(artifact["config"]["batteries"][0], "smoke");
        assert_eq!(artifact["config"]["suites"][0], "canary");
        assert_eq!(artifact["config"]["kv_mode"], "q8");
        assert_eq!(artifact["config"]["max_tokens"], 32);
        assert_eq!(artifact["config"]["benchmark"], true);
        assert_eq!(artifact["config"]["host_memory_width_bits"], 256);
        assert_eq!(artifact["config"]["host_memory_bandwidth_gbps"], 512.5);
        assert_eq!(artifact["config"]["result_cache"], "/tmp/cache");
        assert_eq!(artifact["config"]["cache_mode"], "use");
        assert_eq!(artifact["models"]["candidate"], "candidate.hfq");
        assert_eq!(artifact["models"]["draft"], "draft.hfq");
        assert_eq!(artifact["models"]["baseline"], "baseline.hfq");
        assert_eq!(artifact["models"]["reference"], Value::Null);
    }

    #[test]
    fn artifact_index_entry_json_preserves_shared_metadata() {
        let context = EvidenceArtifactIndexContext {
            provenance: RunProvenance {
                runner: "hipfire-eval".to_string(),
                runner_version: "0.2.0".to_string(),
                hipfire_version: "0.2.0".to_string(),
                git_commit: Some("abc123".to_string()),
                git_branch: Some("main".to_string()),
                git_describe: None,
                git_dirty: Some(true),
                binary_hash: Some("sha256:binary".to_string()),
                arch: Some("gfx1151".to_string()),
                rocm: Some("6.4".to_string()),
            },
            host_profile_hash: "host:abc".to_string(),
            hardware_bucket: "gfx1151:64g".to_string(),
        };
        let artifact = json!({
            "kind": "performance",
            "reason": "ok",
            "expected_metrics": ["tok_s"],
            "records": [
                {"case_id": "a"},
                {"case_id": "b"}
            ]
        });

        let json = evidence_artifact_index_entry_from_value_json(
            "artifacts/performance.json",
            "collected",
            &artifact,
            &context,
        );
        assert_eq!(json["path"], "artifacts/performance.json");
        assert_eq!(json["status"], "collected");
        assert_eq!(json["runner_version"], "0.2.0");
        assert_eq!(json["git_commit"], "abc123");
        assert_eq!(json["git_dirty"], true);
        assert_eq!(json["binary_hash"], "sha256:binary");
        assert_eq!(json["host_profile_hash"], "host:abc");
        assert_eq!(json["hardware_bucket"], "gfx1151:64g");
        assert_eq!(json["row_count"], 2);
        assert_eq!(json["reason"], "ok");
        assert_eq!(json["expected_metrics"][0], "tok_s");
        assert_eq!(json["kind"], "performance");
    }

    #[test]
    fn artifact_index_variant_entries_add_owned_counts_and_kinds() {
        let context = EvidenceArtifactIndexContext {
            provenance: RunProvenance {
                runner: "hipfire-eval".to_string(),
                runner_version: "0.2.0".to_string(),
                hipfire_version: "0.2.0".to_string(),
                git_commit: Some("abc123".to_string()),
                git_branch: None,
                git_describe: None,
                git_dirty: Some(false),
                binary_hash: None,
                arch: None,
                rocm: None,
            },
            host_profile_hash: "host:abc".to_string(),
            hardware_bucket: "gfx1151:64g".to_string(),
        };

        let comparison =
            comparison_artifact_index_entry_json("artifacts/comparisons.json", "pass", 3, &context);
        assert_eq!(comparison["case_count"], 3);
        assert_eq!(comparison["status"], "pass");

        let admission = admission_artifact_index_entry_json(
            "artifacts/admission.json",
            "fail",
            "reject",
            2,
            &context,
        );
        assert_eq!(admission["verdict"], "reject");
        assert_eq!(admission["finding_count"], 2);

        let prompts = prompt_artifact_index_entry_json(
            "artifacts/barrage_prompts.jsonl",
            "materialized",
            "barrage_prompts",
            5,
            &context,
        );
        assert_eq!(prompts["row_count"], 5);
        assert_eq!(prompts["kind"], "barrage_prompts");

        let host = host_profile_artifact_index_entry_json(
            "artifacts/host_profile.json",
            "collected",
            &context,
        );
        assert_eq!(host["kind"], "host_capability_profile");
    }

    #[test]
    fn external_evidence_records_selects_kind_mapping_and_annotates_context() {
        let path = Path::new("/tmp/evidence.json");
        let value = json!({
            "launch_counts": {
                "records": [
                    {"metrics": {"kernel_launches": 3}}
                ]
            }
        });
        let records = extract_external_evidence_records_json(
            "launch_counts",
            path,
            &value,
            json!({"runner": "hipfire-eval"}),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "launch_counts");
        assert_eq!(records[0]["source_path"], "/tmp/evidence.json");
        assert_eq!(records[0]["hipfire_eval_context"]["runner"], "hipfire-eval");
        assert_eq!(records[0]["metrics"]["kernel_launches"], 3);
    }

    #[test]
    fn external_evidence_records_preserves_existing_annotation_fields() {
        let path = Path::new("/tmp/launch_counts.json");
        let value = json!({
            "kind": "launch_counts",
            "records": [
                {
                    "kind": "custom",
                    "source_path": "/already/set.json",
                    "hipfire_eval_context": {"runner": "custom-runner"},
                    "metrics": {"kernel_launches": 7}
                }
            ]
        });
        let records = extract_external_evidence_records_json(
            "launch_counts",
            path,
            &value,
            json!({"runner": "hipfire-eval"}),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "custom");
        assert_eq!(records[0]["source_path"], "/already/set.json");
        assert_eq!(
            records[0]["hipfire_eval_context"]["runner"],
            "custom-runner"
        );
        assert_eq!(records[0]["metrics"]["kernel_launches"], 7);
    }

    #[test]
    fn external_evidence_records_uses_path_stem_and_wraps_scalars() {
        let path = Path::new("/tmp/performance.json");
        let records = extract_external_evidence_records_json(
            "performance",
            path,
            &json!(123.4),
            json!({"runner": "hipfire-eval"}),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "performance");
        assert_eq!(records[0]["source_path"], "/tmp/performance.json");
        assert_eq!(records[0]["hipfire_eval_context"]["runner"], "hipfire-eval");
        assert_eq!(records[0]["value"], 123.4);
    }

    #[test]
    fn standard_evidence_path_discovery_uses_shared_catalog() {
        let root = temp_dir("hipfire-evidence-standard-paths");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("quality.json"), "{}").unwrap();
        fs::write(root.join("module_evidence.json"), "{}").unwrap();
        fs::write(root.join("not_evidence.json"), "{}").unwrap();
        fs::create_dir_all(root.join("performance.json")).unwrap();

        assert_eq!(
            standard_evidence_artifact_kind_for_path(&root.join("quality.json")),
            Some("quality")
        );
        assert_eq!(
            standard_evidence_artifact_kind_for_path(&root.join("not_evidence.json")),
            None
        );

        let paths = standard_evidence_paths_in_dir(&root).unwrap();
        let names: Vec<_> = paths
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["module_evidence.json", "quality.json"]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn router_histogram_metrics_render_sparse_generic_histograms() {
        let mut cooccurrence = HashMap::new();
        cooccurrence.insert(1, 4);
        cooccurrence.insert(5, 2);
        cooccurrence.insert(7, 0);
        let evidence = RouterHistogramEvidence {
            case_id: "router",
            prompt_path: "prompt.txt",
            collection_scope: "test_gatherer",
            num_experts: 4,
            k_top: 2,
            routed_tokens: 3,
            routed_slots: 6,
            top1_histogram: vec![2, 0, 1, 0],
            topk_histogram: vec![2, 3, 1, 0],
            weight_sums: vec![0.5, 0.0, 1.25, 0.0],
            dropped_indices: 1,
            per_layer: vec![RouterHistogramLayer {
                layer_idx: 7,
                top1_histogram: vec![0, 1, 0, 0],
                topk_histogram: vec![0, 2, 1, 0],
                weight_sums: vec![0.0, 0.7, 0.2, 0.0],
                dropped_indices: 0,
                routed_tokens: 2,
                routed_slots: 3,
                cooccurrence,
            }],
        };

        let metrics = router_histogram_metrics_json(&evidence);

        assert_eq!(metrics["expert_hits"], json!({"0": 2, "1": 3, "2": 1}));
        assert_eq!(metrics["router_top1_histogram"], json!({"0": 2, "2": 1}));
        assert_eq!(metrics["router_weight_sums"], json!({"0": 0.5, "2": 1.25}));
        assert_eq!(metrics["router_dropped_tokens"], 1);
        assert_eq!(metrics["num_experts"], 4);
        assert_eq!(metrics["k_top"], 2);
        assert_eq!(
            metrics["per_layer_router_histograms"][0]["topk_cooccurrence"],
            json!({"0,1": 4, "1,1": 2})
        );
        assert!(
            metrics["router_entropy"].as_f64().unwrap() > 0.0,
            "non-empty histograms should expose entropy"
        );
    }

    #[test]
    fn runtime_oneshot_and_router_histogram_writers_emit_standard_artifacts() {
        let root = temp_dir("hipfire-evidence-runtime-writers");
        let context = json!({"runner": "unit-test"});

        write_runtime_oneshot_evidence(
            &root,
            &context,
            RuntimeOneshotEvidence {
                case_id: "run_oneshot",
                prompt_path: "prompt.txt",
                prompt_tokens: 10,
                emitted_tokens: 5,
                prefill_forward_calls: 10,
                decode_forward_calls: 4,
                prefill_secs: 0.5,
                decode_secs: 0.25,
                ttft_ms: 550.0,
                vram_used_mb: 128,
                vram_total_mb: 1024,
            },
        )
        .unwrap();
        write_router_histogram_evidence(
            &root,
            &context,
            RouterHistogramEvidence {
                case_id: "run_oneshot",
                prompt_path: "prompt.txt",
                collection_scope: "unit_gatherer",
                num_experts: 2,
                k_top: 1,
                routed_tokens: 2,
                routed_slots: 2,
                top1_histogram: vec![1, 1],
                topk_histogram: vec![1, 1],
                weight_sums: vec![0.8, 0.2],
                dropped_indices: 0,
                per_layer: Vec::new(),
            },
        )
        .unwrap();

        let performance: Value =
            serde_json::from_str(&fs::read_to_string(root.join("performance.json")).unwrap())
                .unwrap();
        let launch_counts: Value =
            serde_json::from_str(&fs::read_to_string(root.join("launch_counts.json")).unwrap())
                .unwrap();
        let router: Value = serde_json::from_str(
            &fs::read_to_string(root.join("moe_router_histogram.json")).unwrap(),
        )
        .unwrap();

        assert_eq!(performance["kind"], "performance");
        assert_eq!(performance["records"][0]["metrics"]["prefill_tok_s"], 20.0);
        assert_eq!(performance["records"][0]["metrics"]["decode_tok_s"], 20.0);
        assert_eq!(
            launch_counts["records"][0]["metrics"]["kernel_launches"],
            14
        );
        assert_eq!(router["kind"], "moe_router_histogram");
        assert_eq!(
            router["records"][0]["metrics"]["collection_scope"],
            "unit_gatherer"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_hfq_metadata_extracts_json_span() {
        let root = temp_dir("hipfire-evidence-hfq");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("test.hfq");
        let metadata = br#"{"quantization_hash":{"kind":"test"}}"#;
        let metadata_offset = 32u64;
        let data_offset = metadata_offset + metadata.len() as u64 + 4;
        let mut header = [0u8; 32];
        header[0..4].copy_from_slice(b"HFQM");
        header[8..12].copy_from_slice(&42u32.to_le_bytes());
        header[16..24].copy_from_slice(&metadata_offset.to_le_bytes());
        header[24..32].copy_from_slice(&data_offset.to_le_bytes());
        let mut f = File::create(&path).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(metadata).unwrap();
        f.write_all(b"xxxx").unwrap();
        drop(f);

        let got = read_hfq_metadata(&path).unwrap();
        assert_eq!(got.arch_id, 42);
        assert_eq!(got.metadata_json, String::from_utf8_lossy(metadata));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verify_slice_md5_accepts_matching_sidecar() {
        let root = temp_dir("hipfire-evidence-slice-md5");
        fs::create_dir_all(&root).unwrap();
        let slice = root.join("slice.bin");
        fs::write(&slice, b"abc").unwrap();
        fs::write(root.join("slice.md5"), "900150983cd24fb0d6963f7d28e17f72\n").unwrap();

        verify_slice_md5(&slice, "hipfire-evidence-test");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verify_ref_sha256_accepts_matching_manifest_entry() {
        let root = temp_dir("hipfire-evidence-ref-sha");
        let refs = root.join("refs");
        let harness = root.join("harness");
        fs::create_dir_all(&refs).unwrap();
        fs::create_dir_all(&harness).unwrap();
        let reference = refs.join("sample.kldref.bin");
        fs::write(&reference, b"abc").unwrap();
        fs::write(
            harness.join("manifest.json"),
            r#"{"references":{"sample.kldref.bin":{"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}}"#,
        )
        .unwrap();

        verify_ref_sha256(&reference, "hipfire-evidence-test");
        let _ = fs::remove_dir_all(root);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "{}-{}",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }
}
