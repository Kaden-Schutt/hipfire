// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mapped shared-ring transport probe for harmonic DS4 execution.

use std::fs::{self, OpenOptions};
use std::path::PathBuf;

use hip_bridge::HipRuntime;
use hipfire_arch_deepseek4::{
    harmonic_monotonic_tick, HarmonicCompletion, HarmonicContract, HarmonicExpertMappedPoll,
    HarmonicGpuMapping, HarmonicSharedRing, HARMONIC_ACTIVATION_EXTENT, HARMONIC_RESULT_EXTENT,
    HARMONIC_SPLIT_RESULT_EXTENT,
};
use memmap2::MmapOptions;
use rdna_compute::Gpu;

const SOURCE_GENERATION: u64 = 1;
const EXPERT_GENERATION: u64 = 1;
const EPOCH: u64 = 1;
const COPY_ROUNDS: usize = 1_000;

fn payload(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|index| (index as u8).wrapping_mul(29).wrapping_add(salt))
        .collect()
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

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let selector = match args.as_slice() {
        [device_flag, selector] if device_flag == "--device" => selector,
        _ => panic!("usage: harmonic_mapped_transport_probe --device arch:GFX|name:TEXT|pci:BDF"),
    };
    let (device_id, normalized_pci, expected_arch, marketing_name) =
        resolve_device(selector).expect("resolve portable selector to exact PCI device");

    let gpu = Gpu::init_with_device(device_id).expect("initialize exact PCI probe GPU");
    assert_eq!(
        gpu.arch.to_ascii_lowercase(),
        expected_arch,
        "exact PCI device has an unexpected architecture"
    );
    println!(
        "harmonic mapped transport probe: selector={} pci={} arch={} name={:?} device={}",
        selector, normalized_pci, gpu.arch, marketing_name, gpu.device_id
    );

    let root = PathBuf::from("target").join(format!(
        "harmonic-mapped-transport-probe-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create probe directory");
    let ring_path = root.join("ring.bin");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&ring_path)
        .expect("create probe ring");
    let contract = HarmonicContract::frozen(SOURCE_GENERATION, EXPERT_GENERATION);
    let mut ring = HarmonicSharedRing::create_data_plane(&file, contract).expect("create ring");
    let mut mapping = HarmonicGpuMapping::register(&mut ring, &gpu.hip)
        .expect("register file-backed ring with HIP");

    let activation = payload(HARMONIC_ACTIVATION_EXTENT as usize, 0x31);
    let result = payload(HARMONIC_RESULT_EXTENT as usize, 0xa7);
    let source = gpu.hip.malloc(activation.len()).expect("allocate source");
    let destination = gpu
        .hip
        .malloc(activation.len())
        .expect("allocate destination");
    let stream = gpu.hip.stream_create().expect("create copy stream");
    gpu.hip
        .memcpy_htod(&source, &activation)
        .expect("upload activation");

    let mapped_activation = mapping.activation_buffer(EPOCH);
    gpu.hip
        .memcpy_dtod_async_at(&mapped_activation, 0, &source, 0, activation.len(), &stream)
        .expect("write mapped activation");
    gpu.hip
        .stream_synchronize(&stream)
        .expect("publish-copy sync");

    let now = harmonic_monotonic_tick().expect("monotonic tick");
    let packet = contract.packet(
        EPOCH,
        0,
        [0, 1, 2, 3, 4, 5],
        [f32::to_bits(1.0 / 6.0); 6],
        now + 1_000_000_000,
        0,
    );
    ring.publish_mapped(packet, SOURCE_GENERATION, now)
        .expect("publish mapped activation");
    match ring
        .expert_poll_mapped(EPOCH, EXPERT_GENERATION)
        .expect("expert mapped poll")
    {
        HarmonicExpertMappedPoll::Work(observed) => assert_eq!(observed, packet),
        other => panic!("expected mapped expert work, got {other:?}"),
    }

    gpu.hip
        .memcpy_dtod_async_at(
            &destination,
            0,
            &mapped_activation,
            0,
            activation.len(),
            &stream,
        )
        .expect("read mapped activation");
    gpu.hip
        .stream_synchronize(&stream)
        .expect("activation-read sync");
    let mut observed_activation = vec![0_u8; activation.len()];
    gpu.hip
        .memcpy_dtoh(&mut observed_activation, &destination)
        .expect("download activation");
    assert_eq!(
        observed_activation, activation,
        "activation payload mismatch"
    );

    gpu.hip
        .memcpy_htod(&source, &result)
        .expect("upload result");
    let mapped_result = mapping.result_buffer(EPOCH);
    gpu.hip
        .memcpy_dtod_async_at(&mapped_result, 0, &source, 0, result.len(), &stream)
        .expect("write mapped result");
    gpu.hip
        .stream_synchronize(&stream)
        .expect("completion-copy sync");
    ring.expert_complete_mapped(
        EPOCH,
        EXPERT_GENERATION,
        HarmonicCompletion {
            result_extent: HARMONIC_RESULT_EXTENT,
            result_fingerprint: 0,
        },
    )
    .expect("complete mapped result");
    let resolved = ring
        .source_resolve_mapped(EPOCH, SOURCE_GENERATION)
        .expect("resolve mapped result");
    assert!(resolved.completion.is_some(), "mapped completion missing");

    gpu.hip
        .memcpy_dtod_async_at(&destination, 0, &mapped_result, 0, result.len(), &stream)
        .expect("read mapped result");
    gpu.hip
        .stream_synchronize(&stream)
        .expect("result-read sync");
    let mut observed_result = vec![0_u8; result.len()];
    gpu.hip
        .memcpy_dtoh(&mut observed_result, &destination)
        .expect("download result");
    assert_eq!(observed_result, result, "result payload mismatch");
    ring.recycle(EPOCH).expect("recycle mapped slot");

    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip
        .event_record(&start, Some(&stream))
        .expect("record start");
    for _ in 0..COPY_ROUNDS {
        gpu.hip
            .memcpy_dtod_async_at(&mapped_activation, 0, &source, 0, activation.len(), &stream)
            .expect("timed device-to-mapped copy");
    }
    gpu.hip
        .event_record(&stop, Some(&stream))
        .expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait timed copies");
    let write_us = f64::from(
        gpu.hip
            .event_elapsed_ms(&start, &stop)
            .expect("time mapped writes"),
    ) * 1_000.0
        / COPY_ROUNDS as f64;

    gpu.hip
        .event_record(&start, Some(&stream))
        .expect("record start");
    for _ in 0..COPY_ROUNDS {
        gpu.hip
            .memcpy_dtod_async_at(
                &destination,
                0,
                &mapped_activation,
                0,
                activation.len(),
                &stream,
            )
            .expect("timed mapped-to-device copy");
    }
    gpu.hip
        .event_record(&stop, Some(&stream))
        .expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait timed copies");
    let read_us = f64::from(
        gpu.hip
            .event_elapsed_ms(&start, &stop)
            .expect("time mapped reads"),
    ) * 1_000.0
        / COPY_ROUNDS as f64;

    // Size the DS4HARM3 six-row result without changing the shipping DS4HARM2
    // ring layout. This anonymous mapping has the same page-backed,
    // process-local HIP registration contract as the ring. gfx1151's write and
    // gfx1100's read are measured in separate exact-device invocations.
    let mut split_mapping = MmapOptions::new()
        .len(HARMONIC_SPLIT_RESULT_EXTENT)
        .map_anon()
        .expect("allocate split-result mapping");
    let split_host = split_mapping.as_mut_ptr().cast();
    unsafe {
        gpu.hip
            .host_register_mapped(split_host, split_mapping.len())
            .expect("register split-result mapping");
    }
    let split_device = unsafe {
        gpu.hip
            .host_get_device_buffer(split_host, split_mapping.len())
            .expect("resolve split-result device alias")
    };
    let split_payload = payload(HARMONIC_SPLIT_RESULT_EXTENT, 0x6d);
    let split_source = gpu
        .hip
        .malloc(HARMONIC_SPLIT_RESULT_EXTENT)
        .expect("allocate split-result source");
    let split_destination = gpu
        .hip
        .malloc(HARMONIC_SPLIT_RESULT_EXTENT)
        .expect("allocate split-result destination");
    gpu.hip
        .memcpy_htod(&split_source, &split_payload)
        .expect("upload split-result source");

    assert_eq!(
        HARMONIC_SPLIT_RESULT_EXTENT,
        6 * HARMONIC_RESULT_EXTENT as usize,
        "split-result extent must hold six packed DS4 expert rows"
    );
    let mut split_timings = Vec::with_capacity(6);
    for row_count in 1..=6 {
        let extent = row_count * HARMONIC_RESULT_EXTENT as usize;
        gpu.hip
            .event_record(&start, Some(&stream))
            .expect("record split write start");
        for _ in 0..COPY_ROUNDS {
            gpu.hip
                .memcpy_dtod_async_at(&split_device, 0, &split_source, 0, extent, &stream)
                .expect("timed split device-to-mapped copy");
        }
        gpu.hip
            .event_record(&stop, Some(&stream))
            .expect("record split write stop");
        gpu.hip.event_synchronize(&stop).expect("wait split writes");
        let write_us = f64::from(
            gpu.hip
                .event_elapsed_ms(&start, &stop)
                .expect("time split mapped writes"),
        ) * 1_000.0
            / COPY_ROUNDS as f64;

        gpu.hip
            .event_record(&start, Some(&stream))
            .expect("record split read start");
        for _ in 0..COPY_ROUNDS {
            gpu.hip
                .memcpy_dtod_async_at(&split_destination, 0, &split_device, 0, extent, &stream)
                .expect("timed split mapped-to-device copy");
        }
        gpu.hip
            .event_record(&stop, Some(&stream))
            .expect("record split read stop");
        gpu.hip.event_synchronize(&stop).expect("wait split reads");
        let read_us = f64::from(
            gpu.hip
                .event_elapsed_ms(&start, &stop)
                .expect("time split mapped reads"),
        ) * 1_000.0
            / COPY_ROUNDS as f64;
        split_timings.push((row_count, extent, write_us, read_us));
    }
    let mut observed_split = vec![0_u8; HARMONIC_SPLIT_RESULT_EXTENT];
    gpu.hip
        .memcpy_dtoh(&mut observed_split, &split_destination)
        .expect("download split result");
    assert_eq!(
        observed_split, split_payload,
        "split-result payload mismatch"
    );
    gpu.hip.free(split_source).expect("free split source");
    gpu.hip
        .free(split_destination)
        .expect("free split destination");
    unsafe {
        gpu.hip
            .host_unregister(split_host)
            .expect("unregister split-result mapping");
    }

    gpu.hip.event_destroy(start).expect("destroy start event");
    gpu.hip.event_destroy(stop).expect("destroy stop event");
    gpu.hip.stream_destroy(stream).expect("destroy copy stream");
    mapping
        .unregister(&gpu.hip)
        .expect("unregister mapped ring");
    gpu.hip.free(source).expect("free source");
    gpu.hip.free(destination).expect("free destination");
    drop(ring);
    drop(file);
    fs::remove_dir_all(&root).expect("remove probe directory");

    println!(
        "PASS activation_bytes={} result_bytes={} write_us={write_us:.3} read_us={read_us:.3} projected_43_layer_roundtrip_ms={:.3}",
        activation.len(),
        result.len(),
        (write_us + read_us) * 43.0 / 1_000.0,
    );
    for (row_count, extent, split_write_us, split_read_us) in split_timings {
        println!(
            "PASS split_result_rows={row_count} split_result_bytes={extent} split_write_us={split_write_us:.3} split_read_us={split_read_us:.3} projected_43_layer_roundtrip_ms={:.3}",
            (split_write_us + split_read_us) * 43.0 / 1_000.0,
        );
    }
}
