// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hip_bridge::HipRuntime;
use hipfire_arch_deepseek4::{
    forward, DeepseekV4, DeepseekV4HeterogeneousFault, DeepseekV4HeterogeneousLoadPlan,
    DeepseekV4HeterogeneousModel, DeepseekV4State, DeepseekV4VerifiedArtifact,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde_json::json;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or(
            "usage: ds4_heterogeneous_load MODEL [--cycles N] [--replacement-probe] [--fault-matrix] [--fault dense|layer:N|audit|state|scratch] [--decode-token ID] [--position N] [--prompt PATH --generate N --output PATH] [--compare-single]",
        )?;
    let mut cycles = 1usize;
    let mut fault = None;
    let mut replacement_probe = false;
    let mut fault_matrix = false;
    let mut decode_token = None;
    let mut position = 0u32;
    let mut compare_single = false;
    let mut prompt = None;
    let mut generate = 0usize;
    let mut output = None;
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
            "--prompt" => {
                prompt = Some(PathBuf::from(
                    args.next().ok_or("--prompt requires a path")?,
                ));
            }
            "--generate" => {
                generate = args
                    .next()
                    .ok_or("--generate requires a value")?
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --generate: {error}"))?;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    if cycles == 0 {
        return Err("--cycles must be nonzero".into());
    }
    if prompt.is_some() && decode_token.is_some() {
        return Err("--prompt and --decode-token are mutually exclusive".into());
    }
    if prompt.is_some() != (generate != 0) {
        return Err("--prompt and a nonzero --generate must be supplied together".into());
    }
    if output.is_some() && prompt.is_none() {
        return Err("--output requires --prompt".into());
    }
    if prompt.is_some() && cycles != 1 {
        return Err("canonical generation accepts exactly one load cycle".into());
    }

    let plan = DeepseekV4HeterogeneousLoadPlan::default();
    let artifact = DeepseekV4VerifiedArtifact::verify(Path::new(&model))?;
    let generation = if let Some(prompt_path) = prompt.as_deref() {
        let hfq = HfqFile::open(artifact.path())
            .map_err(|error| format!("generation tokenizer open: {error:?}"))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|error| format!("generation tokenizer: {error:?}"))?;
        let prompt_text = std::fs::read_to_string(prompt_path)
            .map_err(|error| format!("read prompt {}: {error}", prompt_path.display()))?;
        let prompt_tokens = tokenizer.encode(&prompt_text);
        if prompt_tokens.len() != 2048 {
            return Err(format!(
                "canonical heterogeneous prompt must encode to 2048 tokens, got {}",
                prompt_tokens.len()
            ));
        }
        Some((tokenizer, prompt_tokens))
    } else {
        None
    };
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
        let mut heterogeneous_generation = None;
        if let Some((tokenizer, prompt_tokens)) = generation.as_ref() {
            let generated = generate_heterogeneous(&mut loaded, prompt_tokens, generate)?;
            let decoded = tokenizer.decode_bytes(&generated.tokens);
            if let Some(output_path) = output.as_deref() {
                std::fs::write(output_path, &decoded).map_err(|error| {
                    format!("write generated output {}: {error}", output_path.display())
                })?;
            }
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "cycle": cycle,
                    "status": "generated",
                    "prompt_tokens": prompt_tokens.len(),
                    "generated_tokens": generated.tokens.len(),
                    "generated_bytes": decoded.len(),
                    "prefill_seconds": generated.prefill.as_secs_f64(),
                    "decode_seconds": generated.decode.as_secs_f64(),
                    "prefill_tok_s": prompt_tokens.len() as f64 / generated.prefill.as_secs_f64(),
                    "decode_tok_s": generated.tokens.len() as f64 / generated.decode.as_secs_f64(),
                    "output_path": output,
                }))
                .map_err(|error| error.to_string())?
            );
            heterogeneous_generation = Some(generated.tokens);
        } else if let Some(token_id) = decode_token {
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
            if let Some((tokenizer, prompt_tokens)) = generation.as_ref() {
                compare_single_generation(
                    artifact.path(),
                    tokenizer,
                    prompt_tokens,
                    generate,
                    heterogeneous_generation
                        .as_deref()
                        .ok_or("heterogeneous generation missing for single-device comparison")?,
                )?;
            } else {
                let token_id = decode_token
                    .ok_or("--compare-single requires --decode-token or --prompt/--generate")?;
                let heterogeneous_logits = heterogeneous_logits
                    .as_deref()
                    .ok_or("heterogeneous logits missing for single-device comparison")?;
                compare_single_device(artifact.path(), token_id, position, heterogeneous_logits)?;
            }
        }
    }
    Ok(())
}

struct GenerationResult {
    tokens: Vec<u32>,
    prefill: Duration,
    decode: Duration,
}

fn greedy(logits: &[f32]) -> Result<u32, String> {
    if let Some((index, _)) = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1))
    {
        Ok(index as u32)
    } else {
        Err("decode returned no finite logits".into())
    }
}

fn generate_heterogeneous(
    model: &mut DeepseekV4HeterogeneousModel,
    prompt: &[u32],
    n_generate: usize,
) -> Result<GenerationResult, String> {
    let prefill_start = Instant::now();
    let mut logits = Vec::new();
    for (position, &token) in prompt.iter().enumerate() {
        logits = model.decode_step(token, position as u32)?;
    }
    let prefill = prefill_start.elapsed();

    let decode_start = Instant::now();
    let mut tokens = Vec::with_capacity(n_generate);
    tokens.push(greedy(&logits)?);
    while tokens.len() < n_generate {
        let position = prompt.len() + tokens.len() - 1;
        logits = model.decode_step(tokens[tokens.len() - 1], position as u32)?;
        tokens.push(greedy(&logits)?);
    }
    Ok(GenerationResult {
        tokens,
        prefill,
        decode: decode_start.elapsed(),
    })
}

fn compare_single_generation(
    model: &Path,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    n_generate: usize,
    heterogeneous: &[u32],
) -> Result<(), String> {
    let mut hfq = HfqFile::open(model).map_err(|error| format!("single oracle open: {error:?}"))?;
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
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

    let prefill_start = Instant::now();
    let mut logits = Vec::new();
    for (position, &token) in prompt.iter().enumerate() {
        logits =
            forward::decode_step(&cfg, &weights, &mut state, &mut gpu, token, position as u32)?;
    }
    let prefill = prefill_start.elapsed();
    let decode_start = Instant::now();
    let mut single = Vec::with_capacity(n_generate);
    single.push(greedy(&logits)?);
    while single.len() < n_generate {
        let position = prompt.len() + single.len() - 1;
        logits = forward::decode_step(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            single[single.len() - 1],
            position as u32,
        )?;
        single.push(greedy(&logits)?);
    }
    let decode = decode_start.elapsed();

    let first_mismatch = single
        .iter()
        .zip(heterogeneous)
        .position(|(expected, actual)| expected != actual);
    let single_bytes = tokenizer.decode_bytes(&single);
    let heterogeneous_bytes = tokenizer.decode_bytes(heterogeneous);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "single_generation_oracle",
            "single_device_id": gpu.device_id,
            "single_arch": gpu.arch,
            "prompt_tokens": prompt.len(),
            "generated_tokens": single.len(),
            "generated_bytes": single_bytes.len(),
            "prefill_seconds": prefill.as_secs_f64(),
            "decode_seconds": decode.as_secs_f64(),
            "prefill_tok_s": prompt.len() as f64 / prefill.as_secs_f64(),
            "decode_tok_s": single.len() as f64 / decode.as_secs_f64(),
            "first_token_mismatch": first_mismatch,
            "tokens_equal": single == heterogeneous,
            "bytes_equal": single_bytes == heterogeneous_bytes,
        }))
        .map_err(|error| error.to_string())?
    );

    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    if single != heterogeneous || single_bytes != heterogeneous_bytes {
        return Err(format!(
            "heterogeneous generation differs from single gfx1151 at token {:?}",
            first_mismatch
        ));
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
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
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
