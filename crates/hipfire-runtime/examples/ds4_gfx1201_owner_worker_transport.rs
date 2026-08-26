// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transport screen for the four-gfx1201 DeepSeek owner/worker EP topology.
//!
//! One dense owner broadcasts the prepared `[x_rot, topk ids, topk weights]`
//! packet to three expert workers. Each worker returns one routed partial. The
//! 43-layer chain uses persistent streams and system-visible signal memory and
//! performs one terminal host wait. This is a transport admission screen, not
//! a model-throughput claim.

use hip_bridge::DeviceBuffer;
use hipfire_runtime::multi_gpu::Gpus;
use std::time::Instant;

const RANKS: usize = 4;
const LAYERS: usize = 43;
const PACKET_BYTES: usize = 16_448;
const RESULT_BYTES: usize = 16_384;
const WAIT_EQ: u32 = 0x1;
const SIGNAL_FLAGS: u32 = 0;
const SIGNAL_MASK: u32 = u32::MAX;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn enqueue_chain(
    gpus: &mut Gpus,
    owner_packet: &DeviceBuffer,
    worker_packets: &[DeviceBuffer],
    owner_results: &[DeviceBuffer],
    to_worker: &[DeviceBuffer],
    to_owner: &[DeviceBuffer],
    first_epoch: u32,
) {
    for layer in 0..LAYERS {
        let epoch = first_epoch
            .checked_add(layer as u32)
            .expect("transport epoch overflow");

        gpus.devices[0].bind_thread().expect("bind owner publish");
        let owner_stream = gpus.devices[0]
            .active_stream
            .as_ref()
            .expect("owner stream");
        for worker in 1..RANKS {
            gpus.devices[0]
                .hip
                .memcpy_peer_async(
                    &worker_packets[worker - 1],
                    gpus.devices[worker].device_id,
                    owner_packet,
                    gpus.devices[0].device_id,
                    PACKET_BYTES,
                    owner_stream,
                )
                .expect("owner-to-worker packet");
            gpus.devices[0]
                .hip
                .stream_write_value32(owner_stream, &to_worker[worker - 1], epoch, SIGNAL_FLAGS)
                .expect("publish worker epoch");
        }

        for worker in 1..RANKS {
            gpus.devices[worker]
                .bind_thread()
                .expect("bind worker return");
            let worker_stream = gpus.devices[worker]
                .active_stream
                .as_ref()
                .expect("worker stream");
            gpus.devices[worker]
                .hip
                .stream_wait_value32(
                    worker_stream,
                    &to_worker[worker - 1],
                    epoch,
                    WAIT_EQ,
                    SIGNAL_MASK,
                )
                .expect("wait worker epoch");
            gpus.devices[worker]
                .hip
                .memcpy_peer_async(
                    &owner_results[worker - 1],
                    gpus.devices[0].device_id,
                    &worker_packets[worker - 1],
                    gpus.devices[worker].device_id,
                    RESULT_BYTES,
                    worker_stream,
                )
                .expect("worker-to-owner result");
            gpus.devices[worker]
                .hip
                .stream_write_value32(worker_stream, &to_owner[worker - 1], epoch, SIGNAL_FLAGS)
                .expect("publish owner epoch");
        }

        gpus.devices[0].bind_thread().expect("bind owner join");
        let owner_stream = gpus.devices[0]
            .active_stream
            .as_ref()
            .expect("owner stream");
        for signal in to_owner {
            gpus.devices[0]
                .hip
                .stream_wait_value32(owner_stream, signal, epoch, WAIT_EQ, SIGNAL_MASK)
                .expect("wait owner epoch");
        }
    }
}

fn main() {
    let warmups = env_usize("HIPFIRE_DS4_OWNER_WORKER_WARMUPS", 10);
    let samples = env_usize("HIPFIRE_DS4_OWNER_WORKER_SAMPLES", 100);
    assert!(samples > 0, "samples must be nonzero");

    let mut gpus = Gpus::init_uniform(RANKS, RANKS).expect("init four GPUs");
    assert_eq!(gpus.devices.len(), RANKS, "requires exactly four GPUs");
    for (rank, gpu) in gpus.devices.iter().enumerate() {
        assert_eq!(
            gpu.arch, "gfx1201",
            "rank {rank} is {}; this screen requires four gfx1201 devices",
            gpu.arch
        );
    }
    assert!(
        gpus.enable_peer_all().expect("enable all peer links"),
        "owner/worker screen requires complete peer access"
    );

    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind stream owner");
        gpu.active_stream = Some(gpu.hip.stream_create().expect("create stream"));
    }

    gpus.devices[0].bind_thread().expect("bind owner alloc");
    let owner_packet = gpus.devices[0]
        .hip
        .malloc(PACKET_BYTES)
        .expect("owner packet");
    let owner_results: Vec<DeviceBuffer> = (1..RANKS)
        .map(|_| {
            gpus.devices[0]
                .hip
                .malloc(RESULT_BYTES)
                .expect("owner result")
        })
        .collect();
    let to_owner: Vec<DeviceBuffer> = (1..RANKS)
        .map(|_| {
            let signal = gpus.devices[0]
                .hip
                .malloc_signal(std::mem::size_of::<u64>())
                .expect("owner signal");
            gpus.devices[0]
                .hip
                .memset(&signal, 0, signal.size())
                .expect("zero owner signal");
            signal
        })
        .collect();

    let mut worker_packets = Vec::with_capacity(RANKS - 1);
    let mut to_worker = Vec::with_capacity(RANKS - 1);
    for worker in 1..RANKS {
        gpus.devices[worker]
            .bind_thread()
            .expect("bind worker alloc");
        worker_packets.push(
            gpus.devices[worker]
                .hip
                .malloc(PACKET_BYTES)
                .expect("worker packet"),
        );
        let signal = gpus.devices[worker]
            .hip
            .malloc_signal(std::mem::size_of::<u64>())
            .expect("worker signal");
        gpus.devices[worker]
            .hip
            .memset(&signal, 0, signal.size())
            .expect("zero worker signal");
        to_worker.push(signal);
    }

    let pattern: Vec<u8> = (0..PACKET_BYTES)
        .map(|index| (index as u8).wrapping_mul(29).wrapping_add(17))
        .collect();
    gpus.devices[0].bind_thread().expect("bind owner seed");
    gpus.devices[0]
        .hip
        .memcpy_htod(&owner_packet, &pattern)
        .expect("seed packet");

    let mut next_epoch = 1_u32;
    for _ in 0..warmups {
        enqueue_chain(
            &mut gpus,
            &owner_packet,
            &worker_packets,
            &owner_results,
            &to_worker,
            &to_owner,
            next_epoch,
        );
        next_epoch += LAYERS as u32;
        gpus.devices[0].bind_thread().expect("bind warmup sync");
        gpus.devices[0]
            .hip
            .stream_synchronize(gpus.devices[0].active_stream.as_ref().unwrap())
            .expect("warmup sync");
    }

    let mut timings_us = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        enqueue_chain(
            &mut gpus,
            &owner_packet,
            &worker_packets,
            &owner_results,
            &to_worker,
            &to_owner,
            next_epoch,
        );
        next_epoch += LAYERS as u32;
        gpus.devices[0].bind_thread().expect("bind sample sync");
        gpus.devices[0]
            .hip
            .stream_synchronize(gpus.devices[0].active_stream.as_ref().unwrap())
            .expect("sample sync");
        timings_us.push(started.elapsed().as_secs_f64() * 1e6);
    }

    for (index, result) in owner_results.iter().enumerate() {
        let mut output = vec![0_u8; RESULT_BYTES];
        gpus.devices[0].bind_thread().expect("bind verify");
        gpus.devices[0]
            .hip
            .memcpy_dtoh(&mut output, result)
            .expect("read result");
        assert_eq!(
            output,
            pattern[..RESULT_BYTES],
            "worker {} return mismatch",
            index + 1
        );
    }

    timings_us.sort_by(|left, right| left.total_cmp(right));
    let median = timings_us[timings_us.len() / 2];
    let p10 = timings_us[timings_us.len() / 10];
    let p90 = timings_us[(timings_us.len() * 9) / 10];
    println!(
        "DS4_GFX1201_OWNER_WORKER layers={LAYERS} ranks={RANKS} packet_bytes={PACKET_BYTES} result_bytes={RESULT_BYTES} samples={samples} median_us={median:.3} p10_us={p10:.3} p90_us={p90:.3} per_layer_us={:.3} exact=true",
        median / LAYERS as f64
    );

    gpus.devices[0].bind_thread().expect("bind owner free");
    gpus.devices[0].hip.free(owner_packet).expect("free packet");
    for buffer in owner_results {
        gpus.devices[0].hip.free(buffer).expect("free result");
    }
    for signal in to_owner {
        gpus.devices[0].hip.free(signal).expect("free owner signal");
    }
    for worker in 1..RANKS {
        gpus.devices[worker]
            .bind_thread()
            .expect("bind worker free");
        gpus.devices[worker]
            .hip
            .free(worker_packets.remove(0))
            .expect("free worker packet");
        gpus.devices[worker]
            .hip
            .free(to_worker.remove(0))
            .expect("free worker signal");
    }
    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind stream free");
        if let Some(stream) = gpu.active_stream.take() {
            gpu.hip.stream_destroy(stream).expect("destroy stream");
        }
    }
}
