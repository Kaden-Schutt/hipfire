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
    let scratch = hip.malloc(4).expect("allocate graph scratch");

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

    let (control_median_ns, control_p95_ns) = bench_graph(&hip, &stream, &control_exec);
    let (checkpoint_median_ns, checkpoint_p95_ns) = bench_graph(&hip, &stream, &checkpoint_exec);
    let expected_callbacks = 2 * (WARMUPS + SAMPLES) as u64;
    let observed_callbacks = shared.phase.load(Ordering::Acquire);
    let violations = shared.violations.load(Ordering::Acquire);
    assert_eq!(observed_callbacks, expected_callbacks);
    assert_eq!(violations, 0, "captured callbacks executed out of order");

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

    hip.graph_exec_destroy(checkpoint_exec)
        .expect("destroy checkpoint executable");
    hip.graph_destroy(checkpoint_graph)
        .expect("destroy checkpoint graph");
    hip.graph_exec_destroy(control_exec)
        .expect("destroy control executable");
    hip.graph_destroy(control_graph)
        .expect("destroy control graph");
    hip.free(scratch).expect("free scratch");
    hip.stream_destroy(stream).expect("destroy stream");
}
