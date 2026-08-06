// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::Path;

use hip_bridge::HipRuntime;
use hipfire_arch_deepseek4::{
    forward, DeepseekV4, DeepseekV4HeterogeneousFault, DeepseekV4HeterogeneousLoadPlan,
    DeepseekV4HeterogeneousModel, DeepseekV4State, DeepseekV4VerifiedArtifact,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use serde_json::json;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or(
            "usage: ds4_heterogeneous_load MODEL [--cycles N] [--replacement-probe] [--fault-matrix] [--fault dense|layer:N|audit|state|scratch] [--decode-token ID] [--position N]",
        )?;
    let mut cycles = 1usize;
    let mut fault = None;
    let mut replacement_probe = false;
    let mut fault_matrix = false;
    let mut decode_token = None;
    let mut position = 0u32;
    let mut compare_single = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cycles" => {
                cycles = args
                    .next()
                    .ok_or("--cycles requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --cycles: {error}"))?;
            }
            "--fault" => {
                let value = args
                    .next()
                    .ok_or("--fault requires dense, layer:N, audit, state, or scratch")?;
                fault = Some(if value == "dense" {
                    DeepseekV4HeterogeneousFault::AfterDenseWeights
                } else if value == "audit" {
                    DeepseekV4HeterogeneousFault::AfterOwnershipAudit
                } else if value == "state" {
                    DeepseekV4HeterogeneousFault::AfterState
                } else if value == "scratch" {
                    DeepseekV4HeterogeneousFault::AfterScratch
                } else if let Some(layer) = value.strip_prefix("layer:") {
                    DeepseekV4HeterogeneousFault::AfterRoutedLayer(
                        layer
                            .parse()
                            .map_err(|error| format!("invalid routed layer: {error}"))?,
                    )
                } else {
                    return Err(format!("unknown fault point '{value}'"));
                });
            }
            "--replacement-probe" => replacement_probe = true,
            "--fault-matrix" => fault_matrix = true,
            "--decode-token" => {
                decode_token = Some(
                    args.next()
                        .ok_or("--decode-token requires a token id")?
                        .parse::<u32>()
                        .map_err(|error| format!("invalid --decode-token: {error}"))?,
                );
            }
            "--position" => {
                position = args
                    .next()
                    .ok_or("--position requires a value")?
                    .parse::<u32>()
                    .map_err(|error| format!("invalid --position: {error}"))?;
            }
            "--compare-single" => compare_single = true,
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    if cycles == 0 {
        return Err("--cycles must be nonzero".into());
    }

    let plan = DeepseekV4HeterogeneousLoadPlan::default();
    let artifact = DeepseekV4VerifiedArtifact::verify(Path::new(&model))?;
    if fault_matrix {
        if fault.is_some() || cycles != 1 {
            return Err("--fault-matrix cannot be combined with --fault or --cycles".into());
        }
        let faults = [
            DeepseekV4HeterogeneousFault::AfterDenseWeights,
            DeepseekV4HeterogeneousFault::AfterRoutedLayer(0),
            DeepseekV4HeterogeneousFault::AfterRoutedLayer(42),
            DeepseekV4HeterogeneousFault::AfterOwnershipAudit,
            DeepseekV4HeterogeneousFault::AfterState,
            DeepseekV4HeterogeneousFault::AfterScratch,
        ];
        for (index, fault) in faults.into_iter().enumerate() {
            run_expected_failure(&artifact, &plan, index, fault)?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "cycle": index,
                    "status": "post_failure_vram",
                    "fault": format!("{fault:?}"),
                    "devices": vram_snapshot()?,
                }))
                .map_err(|error| error.to_string())?
            );
        }
        replacement_probe = true;
    }
    for cycle in 0..cycles {
        if let Some(fault) = fault {
            run_expected_failure(&artifact, &plan, cycle, fault)?;
            continue;
        }

        let mut loaded = Some(DeepseekV4HeterogeneousModel::load_verified(
            &artifact,
            plan.clone(),
        )?);
        if replacement_probe {
            let before_sha = loaded
                .as_ref()
                .expect("loaded model missing")
                .report
                .model_sha256
                .clone();
            let replacement_error = DeepseekV4HeterogeneousModel::replace_transactionally_verified(
                &mut loaded,
                &artifact,
                plan.clone(),
            )
            .expect_err("replacement unexpectedly fit beside the resident 73 GiB expert tier");
            let after = loaded
                .as_mut()
                .ok_or("failed replacement removed the previously published model")?;
            if after.report.model_sha256 != before_sha {
                return Err("failed replacement changed the published model identity".into());
            }
            let audit = after.audit_owners()?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "cycle": cycle,
                    "status": "replacement_preserved",
                    "replacement_error": replacement_error,
                    "model_sha256": after.report.model_sha256,
                    "ownership_violations": audit.violations,
                }))
                .map_err(|error| error.to_string())?
            );
        }
        let mut loaded = loaded.expect("loaded model missing after replacement probe");
        let report = &loaded.report;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "cycle": cycle,
                "status": "loaded",
                "model_sha256": report.model_sha256,
                "dense_arch": loaded.dense_gpu.arch,
                "dense_device_id": loaded.dense_gpu.device_id,
                "routed_arch": loaded.routed_gpu.arch,
                "routed_device_id": loaded.routed_gpu.device_id,
                "projection": {
                    "dense_record_count": report.projection.dense_record_count,
                    "dense_allocation_count": report.projection.dense_allocation_count,
                    "dense_bytes": report.projection.dense_bytes,
                    "f16_expansion_bytes": report.projection.f16_expansion_bytes,
                    "routed_record_count": report.projection.routed_record_count,
                    "routed_allocation_count": report.projection.routed_allocation_count,
                    "routed_bytes": report.projection.routed_bytes,
                    "pointer_table_bytes": report.projection.pointer_table_bytes,
                    "host_only_record_count": report.projection.host_only_record_count,
                    "dense_state_scratch_bytes": report.dense_state_scratch_projected_bytes,
                },
                "ownership": {
                    "dense_tensor_count": report.ownership.dense_tensor_count,
                    "dense_bytes": report.ownership.dense_bytes,
                    "routed_tensor_count": report.ownership.routed_tensor_count,
                    "routed_bytes": report.ownership.routed_bytes,
                    "violations": report.ownership.violations,
                },
                "actual": {
                    "dense_bytes": report.dense_actual_bytes,
                    "routed_bytes": report.routed_actual_bytes,
                    "dense_state_scratch_pool_bytes": report.dense_state_scratch_pool_bytes,
                    "dense_free_before": report.dense_free_before,
                    "dense_free_after": report.dense_free_after,
                    "routed_free_before": report.routed_free_before,
                    "routed_free_after": report.routed_free_after,
                },
            }))
            .map_err(|error| error.to_string())?
        );
        let mut heterogeneous_logits = None;
        if let Some(token_id) = decode_token {
            let logits = loaded.decode_step(token_id, position)?;
            let (argmax, max_logit) = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, value)| (index, *value))
                .ok_or("heterogeneous decode returned no logits")?;
            let non_finite = logits.iter().filter(|value| !value.is_finite()).count();
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "cycle": cycle,
                    "status": "decoded",
                    "token_id": token_id,
                    "position": position,
                    "logits": logits.len(),
                    "argmax": argmax,
                    "max_logit": max_logit,
                    "non_finite": non_finite,
                }))
                .map_err(|error| error.to_string())?
            );
            heterogeneous_logits = Some(logits);
        }
        loaded.unload();
        if compare_single {
            let token_id = decode_token.ok_or("--compare-single requires --decode-token")?;
            let heterogeneous_logits = heterogeneous_logits
                .as_deref()
                .ok_or("heterogeneous logits missing for single-device comparison")?;
            compare_single_device(artifact.path(), token_id, position, heterogeneous_logits)?;
        }
    }
    Ok(())
}

fn compare_single_device(
    model: &Path,
    token_id: u32,
    position: u32,
    heterogeneous: &[f32],
) -> Result<(), String> {
    let mut hfq = HfqFile::open(model).map_err(|error| format!("single oracle open: {error:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init_with_device(1)
        .map_err(|error| format!("single oracle gfx1151 init: {error:?}"))?;
    if gpu.arch != "gfx1151" {
        return Err(format!(
            "single oracle device 1 resolved to {}, expected gfx1151",
            gpu.arch
        ));
    }
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;
    let single = forward::decode_step(&cfg, &weights, &mut state, &mut gpu, token_id, position)?;

    if single.len() != heterogeneous.len() {
        return Err(format!(
            "single oracle logits length {} != heterogeneous {}",
            single.len(),
            heterogeneous.len()
        ));
    }
    let mut bit_mismatches = 0usize;
    let mut first_mismatch = None;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (index, (&expected, &actual)) in single.iter().zip(heterogeneous).enumerate() {
        if expected.to_bits() != actual.to_bits() {
            bit_mismatches += 1;
            first_mismatch.get_or_insert(index);
        }
        let abs = (expected - actual).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / expected.abs().max(1.0e-12));
    }
    let single_argmax = single
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("single oracle returned no logits")?;
    let heterogeneous_argmax = heterogeneous
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("heterogeneous route returned no logits")?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "single_oracle",
            "token_id": token_id,
            "position": position,
            "single_device_id": gpu.device_id,
            "single_arch": gpu.arch,
            "logits": single.len(),
            "bit_mismatches": bit_mismatches,
            "first_mismatch": first_mismatch,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "single_argmax": single_argmax,
            "heterogeneous_argmax": heterogeneous_argmax,
            "argmax_equal": single_argmax == heterogeneous_argmax,
        }))
        .map_err(|error| error.to_string())?
    );

    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    if single_argmax != heterogeneous_argmax {
        return Err("heterogeneous route argmax differs from the single-gfx1151 oracle".into());
    }
    Ok(())
}

fn run_expected_failure(
    artifact: &DeepseekV4VerifiedArtifact,
    plan: &DeepseekV4HeterogeneousLoadPlan,
    cycle: usize,
    fault: DeepseekV4HeterogeneousFault,
) -> Result<(), String> {
    let error =
        match DeepseekV4HeterogeneousModel::load_verified_with_fault(artifact, plan.clone(), fault)
        {
            Ok(_) => return Err(format!("fault injection {fault:?} unexpectedly succeeded")),
            Err(error) => error,
        };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "cycle": cycle,
            "status": "expected_failure",
            "fault": format!("{fault:?}"),
            "error": error,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn vram_snapshot() -> Result<Vec<serde_json::Value>, String> {
    let hip = HipRuntime::load().map_err(|error| format!("load HIP for VRAM snapshot: {error}"))?;
    let count = hip
        .device_count()
        .map_err(|error| format!("device count for VRAM snapshot: {error}"))?;
    let mut rows = Vec::new();
    for device_id in 0..count {
        hip.set_device(device_id)
            .map_err(|error| format!("bind device {device_id} for VRAM snapshot: {error}"))?;
        let arch = hip
            .get_arch(device_id)
            .map_err(|error| format!("device {device_id} architecture: {error}"))?;
        let (free, total) = hip
            .get_vram_info()
            .map_err(|error| format!("device {device_id} VRAM snapshot: {error}"))?;
        rows.push(json!({
            "device_id": device_id,
            "arch": arch,
            "used_bytes": total.saturating_sub(free),
            "free_bytes": free,
            "total_bytes": total,
        }));
    }
    Ok(rows)
}
