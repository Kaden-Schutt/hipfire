// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! One-doorbell, 43-checkpoint owner-local AQL continuation oracle.

use std::time::{Duration, Instant};

use hip_bridge::HipRuntime;
use rdna_compute::Gpu;
use rdna_compute::replay::{AqlHostGate, ReplayController};

const GATES: usize = 43;
const WIDTH: usize = 256;
const WARMUPS: usize = 5;
const SAMPLES: usize = 40;

fn as_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: every f32 bit pattern is valid and the returned slice cannot
    // outlive the borrowed input.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: u8 has alignment one and aliases only for the duration of the
    // synchronous HIP copy.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn resolve_device(selector: &str) -> Result<(i32, String, String, String), String> {
    let hip = HipRuntime::load().map_err(|error| format!("HIP discovery: {error:?}"))?;
    let count = hip
        .device_count()
        .map_err(|error| format!("HIP device count: {error:?}"))?;
    let mut matches = Vec::new();
    for device_id in 0..count {
        let arch = hip
            .get_arch(device_id)
            .map_err(|error| format!("HIP device {device_id} arch: {error:?}"))?;
        let name = hip
            .device_name(device_id)
            .map_err(|error| format!("HIP device {device_id} name: {error:?}"))?;
        let pci = hip
            .device_pci_bus_id(device_id)
            .map_err(|error| format!("HIP device {device_id} PCI identity: {error:?}"))?;
        let selected = selector
            .strip_prefix("arch:")
            .is_some_and(|expected| arch.eq_ignore_ascii_case(expected))
            || selector.strip_prefix("name:").is_some_and(|needle| {
                name.to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase())
            })
            || selector
                .strip_prefix("pci:")
                .is_some_and(|expected| pci.eq_ignore_ascii_case(expected));
        if selected {
            matches.push((device_id, pci, arch, name));
        }
    }
    let [(device_id, pci, arch, name)] = matches.as_slice() else {
        return Err(format!(
            "selector {selector:?} matched {} visible devices; use a unique selector",
            matches.len()
        ));
    };
    let pinned = hip
        .device_by_pci_bus_id(pci)
        .map_err(|error| format!("HIP pin {pci}: {error:?}"))?;
    if pinned != *device_id {
        return Err(format!(
            "selector {selector:?} changed ordinal during PCI pin: {device_id} -> {pinned}"
        ));
    }
    Ok((*device_id, pci.clone(), arch.clone(), name.clone()))
}

fn launch_chain(
    gpu: &mut Gpu,
    value: &rdna_compute::GpuTensor,
    one: &rdna_compute::GpuTensor,
    local_tail: &rdna_compute::GpuTensor,
    host_delta: &rdna_compute::GpuTensor,
) -> Result<(), String> {
    for _ in 0..GATES {
        gpu.add_inplace_f32(value, one)
            .map_err(|error| format!("checkpoint increment: {error}"))?;
        gpu.add_inplace_f32(local_tail, one)
            .map_err(|error| format!("owner-local tail: {error}"))?;
        gpu.add_inplace_f32(value, host_delta)
            .map_err(|error| format!("post-gate consumer: {error}"))?;
    }
    Ok(())
}

fn reset(
    gpu: &Gpu,
    value: &rdna_compute::GpuTensor,
    local_tail: &rdna_compute::GpuTensor,
    host_delta: &rdna_compute::GpuTensor,
) -> Result<(), String> {
    let zeros = vec![0.0_f32; WIDTH];
    for tensor in [value, local_tail, host_delta] {
        gpu.hip
            .memcpy_htod(&tensor.buf, as_bytes(&zeros))
            .map_err(|error| format!("reset tensor: {error}"))?;
    }
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut device = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--device" => device = Some(args.next().ok_or("--device needs a selector")?),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let selector = device.ok_or("--device is required")?;
    let (device_id, pci, arch, name) = resolve_device(&selector)?;
    if !arch.eq_ignore_ascii_case("gfx1100") {
        return Err(format!("host-gate oracle requires gfx1100, got {arch}"));
    }

    let mut gpu = Gpu::init_with_device(device_id)
        .map_err(|error| format!("initialize gfx1100 at {pci}: {error}"))?;
    let value = gpu
        .upload_f32(&vec![0.0; WIDTH], &[WIDTH])
        .map_err(|error| format!("allocate value: {error}"))?;
    let one = gpu
        .upload_f32(&vec![1.0; WIDTH], &[WIDTH])
        .map_err(|error| format!("allocate one: {error}"))?;
    let local_tail = gpu
        .upload_f32(&vec![0.0; WIDTH], &[WIDTH])
        .map_err(|error| format!("allocate local tail: {error}"))?;
    let host_delta = gpu
        .upload_f32(&vec![0.0; WIDTH], &[WIDTH])
        .map_err(|error| format!("allocate host delta: {error}"))?;

    let original_replay = std::mem::replace(&mut gpu.replay, ReplayController::new_manual_aql());
    gpu.replay
        .begin_capture()
        .map_err(|error| format!("begin gated capture: {error}"))?;
    launch_chain(&mut gpu, &value, &one, &local_tail, &host_delta)?;
    let capture = gpu
        .replay
        .finish_capture()
        .map_err(|error| format!("finish gated capture: {error}"))?;
    let expected_dispatches = GATES * 3;
    if capture.launch_count != expected_dispatches {
        return Err(format!(
            "captured {} launches, expected {expected_dispatches}",
            capture.launch_count
        ));
    }
    let gates = (0..GATES)
        .map(|gate| AqlHostGate {
            checkpoint_dispatch: gate * 3,
            resume_before: gate * 3 + 2,
        })
        .collect::<Vec<_>>();
    let prepared = gpu
        .replay
        .prepare_host_gated_linear_aql_prefix_for_pci_bus_id(&pci, expected_dispatches, &gates)
        .map_err(|error| format!("prepare gated AQL: {error}"))?;

    reset(&gpu, &value, &local_tail, &host_delta)?;
    let mut expected_value = 0.0_f32;
    let mut submission = unsafe {
        gpu.replay
            .submit_checkpointed_linear_aql(0)
            .map_err(|error| format!("submit correctness replay: {error}"))?
    };
    for gate in 0..GATES {
        let observed = submission
            .wait_next_host_gate(Duration::from_secs(2))
            .map_err(|error| format!("wait correctness gate {gate}: {error}"))?;
        if observed != gate {
            return Err(format!("observed gate {observed}, expected {gate}"));
        }
        expected_value += 1.0;
        let mut checkpoint_value = vec![0.0_f32; WIDTH];
        gpu.hip
            .memcpy_dtoh(as_bytes_mut(&mut checkpoint_value), &value.buf)
            .map_err(|error| format!("download checkpoint {gate}: {error}"))?;
        if checkpoint_value
            .iter()
            .any(|value| *value != expected_value)
        {
            return Err(format!(
                "gate {gate} checkpoint mismatch: got {:?}, expected {expected_value}",
                checkpoint_value.first()
            ));
        }
        let delta_value = (gate + 1) as f32;
        let delta = vec![delta_value; WIDTH];
        gpu.hip
            .memcpy_htod(&host_delta.buf, as_bytes(&delta))
            .map_err(|error| format!("publish host delta {gate}: {error}"))?;
        submission
            .resume_host_gate(gate)
            .map_err(|error| format!("resume correctness gate {gate}: {error}"))?;
        expected_value += delta_value;
    }
    submission
        .wait(Duration::from_secs(2))
        .map_err(|error| format!("wait correctness terminal: {error}"))?;
    let final_value = gpu
        .download_f32(&value)
        .map_err(|error| format!("download final value: {error}"))?;
    let final_tail = gpu
        .download_f32(&local_tail)
        .map_err(|error| format!("download final local tail: {error}"))?;
    if final_value.iter().any(|value| *value != expected_value)
        || final_tail.iter().any(|value| *value != GATES as f32)
    {
        return Err(format!(
            "terminal mismatch: value={:?}/{expected_value} tail={:?}/{}",
            final_value.first(),
            final_tail.first(),
            GATES
        ));
    }

    for _ in 0..WARMUPS {
        let mut submission = unsafe {
            gpu.replay
                .submit_checkpointed_linear_aql(1)
                .map_err(|error| format!("submit gated warmup: {error}"))?
        };
        for gate in 0..GATES {
            submission
                .wait_next_host_gate(Duration::from_secs(2))
                .map_err(|error| format!("wait gated warmup: {error}"))?;
            submission
                .resume_host_gate(gate)
                .map_err(|error| format!("resume gated warmup: {error}"))?;
        }
        submission
            .wait(Duration::from_secs(2))
            .map_err(|error| format!("wait gated warmup terminal: {error}"))?;
    }
    let mut gated_us = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let started = Instant::now();
        let mut submission = unsafe {
            gpu.replay
                .submit_checkpointed_linear_aql(sample + 2)
                .map_err(|error| format!("submit gated sample {sample}: {error}"))?
        };
        for gate in 0..GATES {
            submission
                .wait_next_host_gate(Duration::from_secs(2))
                .map_err(|error| format!("wait gated sample {sample}: {error}"))?;
            submission
                .resume_host_gate(gate)
                .map_err(|error| format!("resume gated sample {sample}: {error}"))?;
        }
        submission
            .wait(Duration::from_secs(2))
            .map_err(|error| format!("wait gated sample {sample} terminal: {error}"))?;
        gated_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let mut cancel_submission = unsafe {
        gpu.replay
            .submit_checkpointed_linear_aql(SAMPLES + 2)
            .map_err(|error| format!("submit cancellation probe: {error}"))?
    };
    cancel_submission
        .wait_next_host_gate(Duration::from_secs(2))
        .map_err(|error| format!("wait cancellation checkpoint: {error}"))?;
    let cancel_started = Instant::now();
    cancel_submission
        .cancel()
        .map_err(|error| format!("cancel gated queue: {error}"))?;
    let cancel_us = cancel_started.elapsed().as_secs_f64() * 1_000_000.0;
    gpu.add_inplace_f32(&local_tail, &one)
        .map_err(|error| format!("post-cancel HIP launch: {error}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("post-cancel HIP synchronize: {error}"))?;

    let gated_replay = std::mem::replace(&mut gpu.replay, ReplayController::new_manual_aql());
    drop(gated_replay);
    gpu.replay
        .begin_capture()
        .map_err(|error| format!("begin ungated capture: {error}"))?;
    launch_chain(&mut gpu, &value, &one, &local_tail, &host_delta)?;
    gpu.replay
        .finish_capture()
        .map_err(|error| format!("finish ungated capture: {error}"))?;
    let ungated_prepared = gpu
        .replay
        .prepare_linear_aql(device_id as usize)
        .map_err(|error| format!("prepare ungated AQL: {error}"))?;
    for _ in 0..WARMUPS {
        unsafe { gpu.replay.replay_linear_aql(0) }
            .map_err(|error| format!("ungated warmup: {error}"))?;
    }
    let mut ungated_us = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let started = Instant::now();
        unsafe { gpu.replay.replay_linear_aql(sample) }
            .map_err(|error| format!("ungated sample {sample}: {error}"))?;
        ungated_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let gated_median = median(&mut gated_us);
    let ungated_median = median(&mut ungated_us);
    let host_gate_overhead = (gated_median - ungated_median) / GATES as f64;
    let ungated_replay = std::mem::replace(&mut gpu.replay, original_replay);
    drop(ungated_replay);

    println!(
        "harmonic host gate exact: selector={} pci={} arch={} name={:?} gates={} dispatches={} gated_packets={} gated_queue={} ungated_packets={} ungated_queue={} samples={} gated_median_us={:.3} ungated_median_us={:.3} overhead_us_per_gate={:.3} cancel_us={:.3} post_cancel_hip=pass",
        selector,
        pci,
        arch,
        name,
        GATES,
        expected_dispatches,
        prepared.1,
        prepared.2,
        ungated_prepared.1,
        ungated_prepared.2,
        SAMPLES,
        gated_median,
        ungated_median,
        host_gate_overhead,
        cancel_us,
    );
    Ok(())
}
