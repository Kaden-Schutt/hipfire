// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Product-shaped correctness and first-sample harness for harmonic DS4 AR.
//! Final throughput promotion belongs to `hipfire bench`; this binary exists
//! to close the isolated-worker composition gate before daemon admission.

use std::path::PathBuf;
use std::time::Instant;

use hipfire_arch_deepseek4::{
    DeepseekV4HarmonicLoadPlan, DeepseekV4HarmonicModel, DeepseekV4VerifiedArtifact,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::json;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().ok_or(
        "usage: ds4_harmonic_ar MODEL --worker PATH --prompt PATH --generate N --runtime-dir PATH [--output PATH]",
    )?);
    let mut worker = None;
    let mut prompt = None;
    let mut generate = None;
    let mut runtime_dir = None;
    let mut output = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--worker" => worker = Some(PathBuf::from(args.next().ok_or("--worker needs PATH")?)),
            "--prompt" => prompt = Some(PathBuf::from(args.next().ok_or("--prompt needs PATH")?)),
            "--generate" => {
                generate = Some(
                    args.next()
                        .ok_or("--generate needs N")?
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --generate: {error}"))?,
                )
            }
            "--runtime-dir" => {
                runtime_dir = Some(PathBuf::from(
                    args.next().ok_or("--runtime-dir needs PATH")?,
                ))
            }
            "--output" => output = Some(PathBuf::from(args.next().ok_or("--output needs PATH")?)),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    let worker = worker.ok_or("--worker is required")?;
    let prompt = prompt.ok_or("--prompt is required")?;
    let generate = generate.ok_or("--generate is required")?;
    let runtime_dir = runtime_dir.ok_or("--runtime-dir is required")?;
    if generate == 0 {
        return Err("--generate must be nonzero".to_owned());
    }

    let artifact = DeepseekV4VerifiedArtifact::verify(&model)?;
    let hfq = HfqFile::open(artifact.path())
        .map_err(|error| format!("open tokenizer metadata: {error}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| format!("load tokenizer: {error}"))?;
    let prompt_text = std::fs::read_to_string(&prompt)
        .map_err(|error| format!("read prompt {}: {error}", prompt.display()))?;
    let prompt_tokens = tokenizer.encode(&prompt_text);
    if prompt_tokens.len() != 2048 {
        return Err(format!(
            "harmonic acceptance prompt must encode to 2048 tokens, got {}",
            prompt_tokens.len()
        ));
    }

    let load_start = Instant::now();
    let mut loaded = DeepseekV4HarmonicModel::load_verified(
        &artifact,
        DeepseekV4HarmonicLoadPlan::new(worker, runtime_dir),
    )?;
    let load_secs = load_start.elapsed().as_secs_f64();
    eprintln!(
        "harmonic_ready dense_pci={} expert_pci={} expert_pid_arch={} routed_tensors={} routed_bytes={}",
        loaded.report.dense_pci_bus_id,
        loaded.report.expert.pci_bus_id,
        loaded.report.expert.architecture,
        loaded.report.expert.routed_tensor_count,
        loaded.report.expert.routed_bytes,
    );

    let prefill_start = Instant::now();
    let mut logits = Vec::new();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        logits = loaded.decode_step(token, position as u32)?;
    }
    let prefill_secs = prefill_start.elapsed().as_secs_f64();

    loaded.reset_timing()?;
    hip_bridge::launch_counters::reset();
    let decode_start = Instant::now();
    let mut generated = Vec::with_capacity(generate);
    generated.push(greedy(&logits)?);
    while generated.len() < generate {
        let position = prompt_tokens.len() + generated.len() - 1;
        logits = loaded.decode_step(*generated.last().unwrap(), position as u32)?;
        generated.push(greedy(&logits)?);
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    let timing = loaded.timing()?;
    let host_launch_calls = hip_bridge::launch_counters::launch_kernel::count();
    let host_launch_ns = hip_bridge::launch_counters::launch_kernel::time_ns();
    let host_stream_sync_calls = hip_bridge::launch_counters::stream_sync::count();
    let host_stream_sync_ns = hip_bridge::launch_counters::stream_sync::time_ns();
    let host_event_sync_calls = hip_bridge::launch_counters::event_sync::count();
    let host_event_sync_ns = hip_bridge::launch_counters::event_sync::time_ns();
    let host_dtoh_calls = hip_bridge::launch_counters::memcpy_dtoh::count();
    let host_dtoh_ns = hip_bridge::launch_counters::memcpy_dtoh::time_ns();
    let timed_tokens = timing.tokens.max(1) as f64;
    let ms_per_token = |nanos: u64| nanos as f64 / 1_000_000.0 / timed_tokens;
    let decoded = tokenizer.decode_bytes(&generated);
    if let Some(path) = output.as_deref() {
        std::fs::write(path, &decoded)
            .map_err(|error| format!("write output {}: {error}", path.display()))?;
    }
    let report = json!({
        "model_sha256": loaded.report.model_sha256,
        "dense_pci_bus_id": loaded.report.dense_pci_bus_id,
        "expert_pci_bus_id": loaded.report.expert.pci_bus_id,
        "prompt_tokens": prompt_tokens.len(),
        "generated_tokens": generated.len(),
        "decoded_bytes": decoded.len(),
        "load_seconds": load_secs,
        "prefill_seconds": prefill_secs,
        "prefill_tok_s": prompt_tokens.len() as f64 / prefill_secs,
        "decode_seconds": decode_secs,
        "decode_tok_s": generated.len() as f64 / decode_secs,
        "harmonic_timing": {
            "timed_tokens": timing.tokens,
            "timed_layers": timing.layers,
            "layer_wall_ms_per_token": ms_per_token(timing.layer_wall_ns),
            "route_sync_ms_per_token": ms_per_token(timing.route_sync_ns),
            "route_sync_max_us": timing.route_sync_max_ns as f64 / 1_000.0,
            "expert_wait_ms_per_token": ms_per_token(timing.expert_wait_ns),
            "expert_wait_max_us": timing.expert_wait_max_ns as f64 / 1_000.0,
            "publish_cpu_ms_per_token": ms_per_token(timing.publish_cpu_ns),
            "join_enqueue_cpu_ms_per_token": ms_per_token(timing.join_enqueue_cpu_ns),
        },
        "dense_owner_hip_calls": {
            "launch_count": host_launch_calls,
            "launch_ms_per_token": ms_per_token(host_launch_ns),
            "stream_sync_count": host_stream_sync_calls,
            "stream_sync_ms_per_token": ms_per_token(host_stream_sync_ns),
            "event_sync_count": host_event_sync_calls,
            "event_sync_ms_per_token": ms_per_token(host_event_sync_ns),
            "dtoh_count": host_dtoh_calls,
            "dtoh_ms_per_token": ms_per_token(host_dtoh_ns),
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    loaded.shutdown()?;
    Ok(())
}

fn greedy(logits: &[f32]) -> Result<u32, String> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .ok_or_else(|| "harmonic decode returned no finite logits".to_owned())
}
