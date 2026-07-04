// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-generation evidence writers (runtime one-shot timings + MoE router
//! histograms).
//!
//! These take a `LoadedModel` + the live `Gpu`/histogram and serialize an
//! evidence artifact via `hipfire_evidence`. Extracted verbatim from the former
//! daemon `main.rs` monolith (no behavior change) so the generate paths (now in
//! this crate) can emit evidence without reaching back into the bin.

use std::path::Path;

use hipfire_arch_qwen35::qwen35;
use hipfire_evidence::{RouterHistogramEvidence, RouterHistogramLayer, RuntimeOneshotEvidence};

use crate::model::LoadedModel;

fn daemon_runtime_context(model: &LoadedModel) -> serde_json::Value {
    serde_json::json!({
        "schema": 1,
        "runner": "hipfire-daemon",
        "hipfire_version": env!("CARGO_PKG_VERSION"),
        "model_path": &model.model_path,
        "arch_id": model.arch_id,
        "pipeline_parallel_degree": model.pp,
        "max_seq": model.max_seq,
        "physical_cap": model.physical_cap,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn write_daemon_runtime_oneshot_evidence(
    dir: &str,
    model: &LoadedModel,
    gpu: &hipfire_rdna::Gpu,
    id: &str,
    prompt_tokens: usize,
    emitted_tokens: usize,
    prefill_secs: f64,
    decode_secs: f64,
    ttft_ms: f64,
) {
    let (vram_free_bytes, vram_total_bytes) = gpu.hip.get_vram_info().unwrap_or((0, 0));
    let vram_used_mb =
        ((vram_total_bytes.saturating_sub(vram_free_bytes)) as f64 / (1024.0 * 1024.0)) as u64;
    let vram_total_mb = (vram_total_bytes as f64 / (1024.0 * 1024.0)) as u64;
    let runtime_context = daemon_runtime_context(model);
    let evidence = RuntimeOneshotEvidence {
        case_id: id,
        prompt_path: "daemon://generate",
        prompt_tokens,
        emitted_tokens,
        prefill_forward_calls: if prompt_tokens == 0 { 0 } else { 1 },
        decode_forward_calls: emitted_tokens,
        prefill_secs,
        decode_secs,
        ttft_ms,
        vram_used_mb,
        vram_total_mb,
    };
    if let Err(err) =
        hipfire_evidence::write_runtime_oneshot_evidence(Path::new(dir), &runtime_context, evidence)
    {
        eprintln!("[daemon/evidence] failed to write runtime oneshot evidence: {err}");
    }
}

pub struct DaemonMoeRouterHistogramGuard {
    active: bool,
}

impl DaemonMoeRouterHistogramGuard {
    pub fn start(evidence_dir: Option<&str>, config: &qwen35::Qwen35Config) -> Self {
        let active = evidence_dir.is_some() && config.num_experts > 0;
        if active {
            qwen35::reset_moe_router_histogram(config.num_experts, config.num_experts_per_tok);
        }
        Self { active }
    }

    pub fn take(mut self) -> Option<qwen35::MoeRouterHistogram> {
        self.active = false;
        qwen35::take_moe_router_histogram()
    }
}

impl Drop for DaemonMoeRouterHistogramGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = qwen35::take_moe_router_histogram();
        }
    }
}

pub fn write_daemon_moe_router_evidence(
    dir: &str,
    model: &LoadedModel,
    id: &str,
    hist: qwen35::MoeRouterHistogram,
) {
    if hist.routed_slots == 0 {
        return;
    }
    let runtime_context = daemon_runtime_context(model);
    let evidence = RouterHistogramEvidence {
        case_id: id,
        prompt_path: "daemon://generate",
        collection_scope: "qwen35_moe_daemon_ar_forward_calls",
        num_experts: hist.num_experts,
        k_top: hist.k_top,
        routed_tokens: hist.routed_tokens,
        routed_slots: hist.routed_slots,
        top1_histogram: hist.top1_histogram,
        topk_histogram: hist.topk_histogram,
        weight_sums: hist.weight_sums,
        dropped_indices: hist.dropped_indices,
        per_layer: hist
            .per_layer
            .into_iter()
            .map(|layer| RouterHistogramLayer {
                layer_idx: layer.layer_idx,
                top1_histogram: layer.top1_histogram,
                topk_histogram: layer.topk_histogram,
                weight_sums: layer.weight_sums,
                dropped_indices: layer.dropped_indices,
                routed_tokens: layer.routed_tokens,
                routed_slots: layer.routed_slots,
                cooccurrence: layer.cooccurrence,
            })
            .collect(),
    };
    if let Err(err) = hipfire_evidence::write_router_histogram_evidence(
        Path::new(dir),
        &runtime_context,
        evidence,
    ) {
        eprintln!("[daemon/evidence] failed to write MoE router evidence: {err}");
    }
}
