// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission microbench for finite CPU checkpoints inside a captured HIP graph.
//!
//! This intentionally uses no model kernels. It answers two narrow questions:
//! whether `hipLaunchHostFunc` becomes an ordered graph node on the selected
//! ROCm runtime, and how much two such checkpoints add to launch+synchronize.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use hip_bridge::HipRuntime;

const WARMUPS: usize = 100;
const SAMPLES: usize = 1_000;

#[repr(C)]
struct CallbackShared {
    phase: AtomicU64,
    violations: AtomicU64,
}

#[repr(C)]
struct CallbackNode {
    shared: *const CallbackShared,
    expected_parity: u64,
}

#[repr(C)]
struct CountState {
    calls: AtomicU64,
}

unsafe extern "C" fn ordered_checkpoint(user_data: *mut c_void) {
    // SAFETY: main retains both callback nodes and their shared state until all
    // graph launches have synchronized and the graph has been destroyed.
    let node = unsafe { &*(user_data.cast::<CallbackNode>()) };
    let shared = unsafe { &*node.shared };
    let observed = shared.phase.fetch_add(1, Ordering::AcqRel);
    if observed & 1 != node.expected_parity {
        shared.violations.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe extern "C" fn count_checkpoint(user_data: *mut c_void) {
    // SAFETY: main retains the count state until the graph is synchronized and
    // destroyed.
    let state = unsafe { &*(user_data.cast::<CountState>()) };
    state.calls.fetch_add(1, Ordering::Release);
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
            "selector {selector:?} matched {} visible devices; use a unique arch:, name:, or pci: selector",
            matches.len()
        ));
    };
    let pinned = hip
        .device_by_pci_bus_id(pci)
        .map_err(|error| format!("HIP pin {pci}: {error:?}"))?;
    if pinned != *device_id {
        return Err(format!(
            "selector {selector:?} changed ordinal during PCI pin: discovered {device_id}, resolved {pinned} at {pci}"
        ));
    }
    Ok((*device_id, pci.clone(), arch.clone(), name.clone()))
}

fn percentile_ns(samples: &mut [u64], percentile: usize) -> u64 {
    samples.sort_unstable();
    let index = (samples.len() - 1) * percentile / 100;
    samples[index]
}

fn bench_graph(
    hip: &HipRuntime,
    stream: &hip_bridge::Stream,
    exec: &hip_bridge::GraphExec,
) -> (u64, u64) {
    for _ in 0..WARMUPS {
        hip.graph_launch(exec, stream).expect("warmup graph launch");
        hip.stream_synchronize(stream)
            .expect("warmup graph synchronize");
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        hip.graph_launch(exec, stream).expect("timed graph launch");
        hip.stream_synchronize(stream)
            .expect("timed graph synchronize");
        samples.push(started.elapsed().as_nanos() as u64);
    }
    let median = percentile_ns(&mut samples, 50);
    let p95 = percentile_ns(&mut samples, 95);
    (median, p95)
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let selector = match args.as_slice() {
        [device_flag, selector] if device_flag == "--device" => selector,
        _ => panic!("usage: hip_graph_host_checkpoint --device arch:GFX|name:TEXT|pci:BDF"),
    };
    let (device_id, pci, arch, name) =
        resolve_device(selector).expect("resolve portable selector to exact PCI device");
    let hip = HipRuntime::load().expect("load HIP runtime");
    hip.set_device(device_id).expect("bind selected device");
    let stream = hip.stream_create().expect("create stream");
    let side_stream = hip.stream_create().expect("create side stream");
    let fork = hip.event_create().expect("create fork event");
    let join = hip.event_create().expect("create join event");
    let scratch = hip.malloc(4).expect("allocate graph scratch");
    let side_scratch = hip.malloc(4).expect("allocate side graph scratch");

    hip.stream_begin_capture(&stream, 0)
        .expect("begin control graph capture");
    hip.memset_async(&scratch, 0x5a, 4, &stream)
        .expect("capture control memset");
    let control_graph = hip
        .stream_end_capture(&stream)
        .expect("end control graph capture");
    let control_exec = hip
        .graph_instantiate(&control_graph)
        .expect("instantiate control graph");

    let shared = Box::new(CallbackShared {
        phase: AtomicU64::new(0),
        violations: AtomicU64::new(0),
    });
    let first = Box::new(CallbackNode {
        shared: &*shared,
        expected_parity: 0,
    });
    let second = Box::new(CallbackNode {
        shared: &*shared,
        expected_parity: 1,
    });

    hip.stream_begin_capture(&stream, 0)
        .expect("begin checkpoint graph capture");
    hip.memset_async(&scratch, 0x5a, 4, &stream)
        .expect("capture checkpoint memset");
    unsafe {
        hip.launch_host_func(
            &stream,
            ordered_checkpoint,
            (&*first as *const CallbackNode).cast_mut().cast(),
        )
        .expect("capture first host checkpoint");
        hip.launch_host_func(
            &stream,
            ordered_checkpoint,
            (&*second as *const CallbackNode).cast_mut().cast(),
        )
        .expect("capture second host checkpoint");
    }
    let checkpoint_graph = hip
        .stream_end_capture(&stream)
        .expect("end checkpoint graph capture");
    let checkpoint_exec = hip
        .graph_instantiate(&checkpoint_graph)
        .expect("instantiate checkpoint graph");

    hip.stream_begin_capture(&stream, 0)
        .expect("begin branch control capture");
    hip.event_record(&fork, Some(&stream))
        .expect("capture control fork");
    hip.stream_wait_event(&side_stream, &fork)
        .expect("capture control side wait");
    hip.memset_async(&side_scratch, 0x3c, 4, &side_stream)
        .expect("capture control side work");
    hip.event_record(&join, Some(&side_stream))
        .expect("capture control join record");
    hip.memset_async(&scratch, 0x5a, 4, &stream)
        .expect("capture control primary work");
    hip.stream_wait_event(&stream, &join)
        .expect("capture control join wait");
    let branch_control_graph = hip
        .stream_end_capture(&stream)
        .expect("end branch control capture");
    let branch_control_exec = hip
        .graph_instantiate(&branch_control_graph)
        .expect("instantiate branch control graph");

    let branch_count = Box::new(CountState {
        calls: AtomicU64::new(0),
    });
    hip.stream_begin_capture(&stream, 0)
        .expect("begin one-checkpoint branch capture");
    hip.event_record(&fork, Some(&stream))
        .expect("capture checkpoint fork");
    hip.stream_wait_event(&side_stream, &fork)
        .expect("capture checkpoint side wait");
    hip.memset_async(&side_scratch, 0x3c, 4, &side_stream)
        .expect("capture checkpoint side work");
    unsafe {
        hip.launch_host_func(
            &side_stream,
            count_checkpoint,
            (&*branch_count as *const CountState).cast_mut().cast(),
        )
        .expect("capture concurrent host checkpoint");
    }
    hip.event_record(&join, Some(&side_stream))
        .expect("capture checkpoint join record");
    hip.memset_async(&scratch, 0x5a, 4, &stream)
        .expect("capture checkpoint primary work");
    hip.stream_wait_event(&stream, &join)
        .expect("capture checkpoint join wait");
    let branch_checkpoint_graph = hip
        .stream_end_capture(&stream)
        .expect("end one-checkpoint branch capture");
    let branch_checkpoint_exec = hip
        .graph_instantiate(&branch_checkpoint_graph)
        .expect("instantiate one-checkpoint branch graph");

    let (control_median_ns, control_p95_ns) = bench_graph(&hip, &stream, &control_exec);
    let (checkpoint_median_ns, checkpoint_p95_ns) = bench_graph(&hip, &stream, &checkpoint_exec);
    let (branch_control_median_ns, branch_control_p95_ns) =
        bench_graph(&hip, &stream, &branch_control_exec);
    let (branch_checkpoint_median_ns, branch_checkpoint_p95_ns) =
        bench_graph(&hip, &stream, &branch_checkpoint_exec);
    let expected_callbacks = 2 * (WARMUPS + SAMPLES) as u64;
    let observed_callbacks = shared.phase.load(Ordering::Acquire);
    let violations = shared.violations.load(Ordering::Acquire);
    assert_eq!(observed_callbacks, expected_callbacks);
    assert_eq!(violations, 0, "captured callbacks executed out of order");
    let branch_callbacks = branch_count.calls.load(Ordering::Acquire);
    assert_eq!(branch_callbacks, (WARMUPS + SAMPLES) as u64);

    println!("device selector={selector} pci={pci} arch={arch} name={name:?} ordinal={device_id}");
    println!(
        "control median_us={:.3} p95_us={:.3}",
        control_median_ns as f64 / 1_000.0,
        control_p95_ns as f64 / 1_000.0
    );
    println!(
        "two_checkpoint median_us={:.3} p95_us={:.3} incremental_median_us={:.3} callbacks={} violations={}",
        checkpoint_median_ns as f64 / 1_000.0,
        checkpoint_p95_ns as f64 / 1_000.0,
        checkpoint_median_ns.saturating_sub(control_median_ns) as f64 / 1_000.0,
        observed_callbacks,
        violations
    );
    println!(
        "branch_control median_us={:.3} p95_us={:.3}",
        branch_control_median_ns as f64 / 1_000.0,
        branch_control_p95_ns as f64 / 1_000.0
    );
    println!(
        "one_checkpoint_branch median_us={:.3} p95_us={:.3} incremental_median_us={:.3} callbacks={}",
        branch_checkpoint_median_ns as f64 / 1_000.0,
        branch_checkpoint_p95_ns as f64 / 1_000.0,
        branch_checkpoint_median_ns.saturating_sub(branch_control_median_ns) as f64 / 1_000.0,
        branch_callbacks
    );

    hip.graph_exec_destroy(branch_checkpoint_exec)
        .expect("destroy branch checkpoint executable");
    hip.graph_destroy(branch_checkpoint_graph)
        .expect("destroy branch checkpoint graph");
    hip.graph_exec_destroy(branch_control_exec)
        .expect("destroy branch control executable");
    hip.graph_destroy(branch_control_graph)
        .expect("destroy branch control graph");
    hip.graph_exec_destroy(checkpoint_exec)
        .expect("destroy checkpoint executable");
    hip.graph_destroy(checkpoint_graph)
        .expect("destroy checkpoint graph");
    hip.graph_exec_destroy(control_exec)
        .expect("destroy control executable");
    hip.graph_destroy(control_graph)
        .expect("destroy control graph");
    hip.event_destroy(join).expect("destroy join event");
    hip.event_destroy(fork).expect("destroy fork event");
    hip.free(side_scratch).expect("free side scratch");
    hip.free(scratch).expect("free scratch");
    hip.stream_destroy(side_stream)
        .expect("destroy side stream");
    hip.stream_destroy(stream).expect("destroy stream");
}
