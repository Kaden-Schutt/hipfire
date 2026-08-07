// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission screen for cross-device HIP-event dependencies captured into four
//! independently instantiated graphs.
//!
//! Each replay first writes a fresh byte pattern on every rank. The captured
//! rank graphs record one event per producer, wait on all peer events, and rank
//! zero copies the peer buffers into local result buffers. Repeated exact
//! downloads prove that event nodes bind to the current graph replay rather
//! than passing on a stale record from the prior replay.

use hip_bridge::{
    DeviceBuffer, Event, Graph, GraphExec, HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM,
};
use hipfire_runtime::multi_gpu::Gpus;

const RANKS: usize = 4;
const BYTES: usize = 16_384;
const CAPTURE_MODE_RELAXED: u32 = 2;

struct CapturedRank {
    graph: Graph,
    exec: GraphExec,
}

fn main() {
    let replays = std::env::var("HIPFIRE_TP4_GRAPH_REPLAYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    assert!(
        replays > 1,
        "stale-event screen requires at least two replays"
    );

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
        "screen requires complete peer access"
    );

    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind stream owner");
        gpu.active_stream = Some(gpu.hip.stream_create().expect("create stream"));
    }

    let mut sources: Vec<DeviceBuffer> = Vec::with_capacity(RANKS);
    let mut events: Vec<Event> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind source alloc");
        sources.push(gpu.hip.malloc(BYTES).expect("source buffer"));
        events.push(
            gpu.hip
                .event_create_with_flags(HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM)
                .expect("cross-device event"),
        );
    }

    gpus.devices[0].bind_thread().expect("bind result alloc");
    let results: Vec<DeviceBuffer> = (0..RANKS)
        .map(|_| gpus.devices[0].hip.malloc(BYTES).expect("result buffer"))
        .collect();

    // Begin every device capture before adding any cross-device dependency.
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind capture begin");
        gpu.hip
            .stream_begin_capture(
                gpu.active_stream.as_ref().expect("active stream"),
                CAPTURE_MODE_RELAXED,
            )
            .expect("begin rank capture");
    }
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind event record");
        gpu.hip
            .event_record(
                &events[rank],
                Some(gpu.active_stream.as_ref().expect("active stream")),
            )
            .expect("capture producer event");
    }
    for destination in 0..RANKS {
        let gpu = &gpus.devices[destination];
        gpu.bind_thread().expect("bind peer waits");
        let stream = gpu.active_stream.as_ref().expect("active stream");
        for source in 0..RANKS {
            if source != destination {
                gpu.hip
                    .stream_wait_event(stream, &events[source])
                    .expect("capture peer event wait");
            }
        }
    }
    {
        let owner = &gpus.devices[0];
        owner.bind_thread().expect("bind peer copies");
        let stream = owner.active_stream.as_ref().expect("owner stream");
        for rank in 0..RANKS {
            owner
                .hip
                .memcpy_peer_async(
                    &results[rank],
                    owner.device_id,
                    &sources[rank],
                    gpus.devices[rank].device_id,
                    BYTES,
                    stream,
                )
                .expect("capture peer copy");
        }
    }

    let mut captures = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind capture end");
        let graph = gpu
            .hip
            .stream_end_capture(gpu.active_stream.as_ref().expect("active stream"))
            .expect("end rank capture");
        let exec = gpu
            .hip
            .graph_instantiate(&graph)
            .expect("instantiate graph");
        captures.push(CapturedRank { graph, exec });
    }

    for replay in 0..replays {
        // Enqueue every producer update before launching any graph. This keeps
        // each event record ordered after fresh data on its local rank while
        // preventing rank-zero launch order from outrunning a peer update.
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind producer update");
            let value = ((replay + rank + 1) % 251 + 1) as u8;
            gpu.hip
                .memset_async(
                    &sources[rank],
                    value as i32,
                    BYTES,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("enqueue producer update");
        }
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind graph launch");
            gpu.hip
                .graph_launch(
                    &captures[rank].exec,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("launch rank graph");
        }
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind replay sync");
            gpu.hip
                .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
                .expect("sync rank replay");
        }

        gpus.devices[0].bind_thread().expect("bind result check");
        for rank in 0..RANKS {
            let expected = ((replay + rank + 1) % 251 + 1) as u8;
            let mut host = vec![0u8; BYTES];
            gpus.devices[0]
                .hip
                .memcpy_dtoh(&mut host, &results[rank])
                .expect("download result");
            assert!(
                host.iter().all(|&value| value == expected),
                "replay {replay} rank {rank}: stale or unordered peer result"
            );
        }
    }

    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind graph destroy");
        let capture = captures.remove(0);
        gpu.hip
            .graph_exec_destroy(capture.exec)
            .expect("destroy graph exec");
        gpu.hip.graph_destroy(capture.graph).expect("destroy graph");
    }

    println!("PASS tp4 cross-device graph barrier: {replays} exact replays");
}
