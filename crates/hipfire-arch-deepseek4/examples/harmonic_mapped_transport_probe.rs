// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mapped shared-ring transport probe for harmonic DS4 execution.

use std::fs::{self, OpenOptions};
use std::path::PathBuf;

use hipfire_arch_deepseek4::{
    harmonic_monotonic_tick, HarmonicCompletion, HarmonicContract, HarmonicExpertMappedPoll,
    HarmonicGpuMapping, HarmonicSharedRing, HARMONIC_ACTIVATION_EXTENT, HARMONIC_RESULT_EXTENT,
};
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

fn main() {
    let gpu = Gpu::init().expect("initialize one visible probe GPU");
    println!(
        "harmonic mapped transport probe: arch={} device={}",
        gpu.arch, gpu.device_id
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
}
