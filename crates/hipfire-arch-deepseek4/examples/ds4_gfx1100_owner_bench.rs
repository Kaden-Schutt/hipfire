// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Model-shaped timing gate for the exact-gfx1100-owned DS4 critical path.
//!
//! This deliberately does not construct a second GPU, worker, shared ring, or
//! peer mapping. Routed experts are omitted after route selection, so decoded
//! text from this probe has no quality meaning. The measured path contains the
//! production dense weights, attention/indexer state, dual-stream attention
//! schedule, router, shared expert, HC mixes, and output head.

use std::path::PathBuf;
use std::time::Instant;

use hip_bridge::HipRuntime;
use hipfire_arch_deepseek4::forward::{
    decode_step_gfx1100_owner_probe, DeepseekV4Gfx1100OwnerExecution,
};
use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::replay::ReplayController;
use rdna_compute::Gpu;
use serde_json::json;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    context: u32,
    warmups: usize,
    runs: usize,
    token: u32,
    pm4: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let model = PathBuf::from(args.next().ok_or(
            "usage: ds4_gfx1100_owner_bench MODEL [--context N] [--warmups N] [--runs N] [--token N] [--pm4]",
        )?);
        let mut parsed = Self {
            model,
            context: 2048,
            warmups: 4,
            runs: 12,
            token: 1,
            pm4: false,
        };
        while let Some(flag) = args.next() {
            if flag == "--pm4" {
                parsed.pm4 = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?;
            match flag.as_str() {
                "--context" => {
                    parsed.context = value
                        .parse()
                        .map_err(|error| format!("invalid --context: {error}"))?
                }
                "--warmups" => {
                    parsed.warmups = value
                        .parse()
                        .map_err(|error| format!("invalid --warmups: {error}"))?
                }
                "--runs" => {
                    parsed.runs = value
                        .parse()
                        .map_err(|error| format!("invalid --runs: {error}"))?
                }
                "--token" => {
                    parsed.token = value
                        .parse()
                        .map_err(|error| format!("invalid --token: {error}"))?
                }
                _ => return Err(format!("unknown argument {flag:?}")),
            }
        }
        if parsed.runs == 0 {
            return Err("--runs must be nonzero".to_owned());
        }
        Ok(parsed)
    }
}

fn resolve_unique_gfx1100() -> Result<(String, i32), String> {
    let hip = HipRuntime::load().map_err(|error| format!("HIP discovery: {error:?}"))?;
    let count = hip
        .device_count()
        .map_err(|error| format!("HIP device count: {error:?}"))?;
    let mut matches = Vec::new();
    for device in 0..count {
        let arch = hip
            .get_arch(device)
            .map_err(|error| format!("HIP device {device} arch: {error:?}"))?;
        if arch.eq_ignore_ascii_case("gfx1100") {
            matches.push(device);
        }
    }
    let [device] = matches.as_slice() else {
        return Err(format!(
            "expected one exact gfx1100 device, found {}",
            matches.len()
        ));
    };
    let pci = hip
        .device_pci_bus_id(*device)
        .map_err(|error| format!("gfx1100 PCI identity: {error:?}"))?;
    Ok((pci, *device))
}

fn argmax(values: &[f32]) -> Result<u32, String> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .ok_or_else(|| "gfx1100 owner probe produced no finite logits".to_owned())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

fn bit_exact(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let (pci, ordinal) = resolve_unique_gfx1100()?;
    let mut gpu = Gpu::init_with_pci_bus_id(&pci, "gfx1100")
        .map_err(|error| format!("init exact gfx1100 at {pci}: {error:?}"))?;
    if gpu.device_id != ordinal || !gpu.arch_caps.is_gfx1100() {
        return Err(format!(
            "gfx1100 identity round trip failed: ordinal={} expected={} arch={}",
            gpu.device_id, ordinal, gpu.arch
        ));
    }

    let mut hfq = HfqFile::open(&args.model)
        .map_err(|error| format!("open {}: {error:?}", args.model.display()))?;
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
    let weights = DeepseekV4::load_weights_harmonic_dense_gfx1100(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;
    state.n_tokens = u64::from(args.context);
    let mut execution = DeepseekV4Gfx1100OwnerExecution::new(&mut gpu)?;

    let mut token = args.token;
    for index in 0..args.warmups {
        let position = args.context.saturating_add(index as u32);
        let logits = decode_step_gfx1100_owner_probe(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            &mut execution,
            token,
            position,
        )?;
        token = argmax(&logits)?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("gfx1100 owner warmup sync: {error:?}"))?;

    let capture_position = args.context.saturating_add(args.warmups as u32);
    let mut capture_logits = None;
    let mut capture_json = serde_json::Value::Null;
    let mut controller = if args.pm4 {
        let mut controller = ReplayController::new_manual_pm4_single_queue();
        std::mem::swap(&mut gpu.replay, &mut controller);
        let capture_result = (|| {
            gpu.replay
                .begin_capture()
                .map_err(|reason| format!("gfx1100 owner PM4 begin capture: {reason}"))?;
            let logits = decode_step_gfx1100_owner_probe(
                &cfg,
                &weights,
                &mut state,
                &mut gpu,
                &mut execution,
                token,
                capture_position,
            )?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| format!("gfx1100 owner PM4 capture sync: {error:?}"))?;
            let summary = gpu
                .replay
                .finish_capture()
                .map_err(|reason| format!("gfx1100 owner PM4 finish capture: {reason}"))?;
            let contracts = gpu
                .replay
                .probe_aql_contracts(gpu.device_id as usize)
                .map_err(|reason| format!("gfx1100 owner PM4 AQL contracts: {reason}"))?;
            gpu.replay
                .prepare_pm4_prefix_for_pci_bus_id(&pci, summary.launch_count)
                .map_err(|reason| format!("gfx1100 owner PM4 prepare: {reason}"))?;
            let identity = gpu
                .replay
                .prepared_route_identity()
                .ok_or_else(|| "gfx1100 owner PM4 prepared identity missing".to_owned())?;
            capture_logits = Some(logits);
            capture_json = json!({
                "launches": summary.launch_count,
                "unique_symbols": summary.unique_kernel_count,
                "sequence_hash": summary.sequence_hash,
                "aql_contracts": contracts.len(),
                "prepared": {
                    "dispatches": identity.dispatch_count,
                    "packets": identity.packet_count,
                    "queue_id": identity.queue_id,
                    "command_dwords": identity.command_dwords,
                    "queue_count": identity.queue_count,
                    "phase_count": identity.phase_count,
                }
            });
            Ok::<(), String>(())
        })();
        std::mem::swap(&mut gpu.replay, &mut controller);
        capture_result?;
        Some(controller)
    } else {
        None
    };

    hip_bridge::launch_counters::reset();
    let mut samples_ms = Vec::with_capacity(args.runs);
    let mut replay_bit_exact = true;
    for index in 0..args.runs {
        let position = if args.pm4 {
            capture_position
        } else {
            capture_position.saturating_add(index as u32)
        };
        let started = Instant::now();
        let logits = if let Some(controller) = controller.as_mut() {
            // SAFETY: the model, scratch allocations, exact device, and captured
            // binding layout remain owned by this scope until the controller is
            // dropped below. Each replay waits for terminal completion.
            unsafe { controller.replay_pm4(position as usize) }
                .map_err(|reason| format!("gfx1100 owner PM4 replay: {reason}"))?;
            gpu.download_f32(state.logits.as_ref().unwrap())
                .map_err(|error| format!("gfx1100 owner PM4 logits: {error:?}"))?
        } else {
            decode_step_gfx1100_owner_probe(
                &cfg,
                &weights,
                &mut state,
                &mut gpu,
                &mut execution,
                token,
                position,
            )?
        };
        samples_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        if let Some(expected) = capture_logits.as_ref() {
            replay_bit_exact &= bit_exact(expected, &logits);
        }
        token = argmax(&logits)?;
    }
    let launch_count = hip_bridge::launch_counters::launch_kernel::count();
    let launch_ns = hip_bridge::launch_counters::launch_kernel::time_ns();
    let mut ordered = samples_ms.clone();
    let median_ms = median(&mut ordered);
    let min_ms = ordered[0];
    let max_ms = ordered[ordered.len() - 1];

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "route": if args.pm4 { "ds4-mq2r-gfx1100-owner-pm4" } else { "ds4-mq2r-gfx1100-owner-hip" },
            "model": args.model,
            "device": { "ordinal": ordinal, "pci": pci, "arch": gpu.arch },
            "context_start": args.context,
            "warmups": args.warmups,
            "runs": args.runs,
            "routed_experts": "omitted-after-routing",
            "peer_devices": 0,
            "capture": capture_json,
            "replay_bit_exact": replay_bit_exact,
            "samples_ms": samples_ms,
            "median_ms_per_token": median_ms,
            "min_ms_per_token": min_ms,
            "max_ms_per_token": max_ms,
            "median_tok_s": 1000.0 / median_ms,
            "t1_50_tok_s_gate": median_ms <= 20.0,
            "launches_per_token": launch_count as f64 / args.runs as f64,
            "host_launch_ms_per_token": launch_ns as f64 / 1.0e6 / args.runs as f64,
            "final_argmax": token,
        }))
        .map_err(|error| error.to_string())?
    );

    drop(controller);
    execution.close(&mut gpu)?;
    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    Ok(())
}
