// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission screen for cross-device signal dependencies captured into three
//! or four independently instantiated graphs.
//! independently instantiated graphs.
//!
//! Each replay delays and then writes fresh F32 values on every rank. The
//! captured rank graphs record one event per producer, wait on all peer events,
//! and rank zero sums three peer-visible tensors with graph-safe kernels.
//! Repeated exact downloads prove that event nodes bind to the current graph
//! replay rather than passing on a stale record from the prior replay.

use hip_bridge::{DeviceBuffer, Graph, GraphExec};
use hipfire_hardware::Gpus;
use rdna_compute::{DType, GpuTensor};

const ELEMS: usize = 4_096;
const DELAY_BYTES: usize = 256 * 1024 * 1024;
const CAPTURE_MODE_RELAXED: u32 = 2;
struct CapturedRank {
    graph: Graph,
    exec: GraphExec,
    _blobs: Vec<Vec<u8>>,
}

fn main() {
    let ranks = std::env::var("HIPFIRE_TP_GRAPH_RANKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    assert!(matches!(ranks, 3 | 4), "screen requires TP3 or TP4");
    let replays = std::env::var("HIPFIRE_TP4_GRAPH_REPLAYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    assert!(
        replays > 1,
        "stale-event screen requires at least two replays"
    );

    let device_opts = hipfire_runtime::config::get().device_resolve_opts();
    let mut gpus = Gpus::init_uniform(&device_opts, ranks, ranks).expect("init TP GPUs");
    assert_eq!(gpus.devices.len(), ranks, "wrong GPU count");
    for (rank, gpu) in gpus.devices.iter().enumerate() {
        assert_eq!(
            gpu.arch, "gfx1201",
            "rank {rank} is {}; this screen requires gfx1201 devices",
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

    let mut sources: Vec<GpuTensor> = Vec::with_capacity(ranks);
    let mut delays = Vec::with_capacity(ranks);
    let mut signals: Vec<DeviceBuffer> = Vec::with_capacity(ranks);
    for rank in 0..ranks {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind source alloc");
        sources.push(
            gpu.alloc_tensor(&[ELEMS], DType::F32)
                .expect("source tensor"),
        );
        delays.push(gpu.hip.malloc(DELAY_BYTES).expect("delay buffer"));
        signals.push(
            gpu.hip
                .malloc_signal(std::mem::size_of::<u64>())
                .expect("cross-device signal"),
        );
        gpu.hip
            .memset(&signals[rank], 0, signals[rank].size())
            .expect("zero cross-device signal");
    }

    gpus.devices[0].bind_thread().expect("bind result alloc");
    let tmp = gpus.devices[0]
        .alloc_tensor(&[ELEMS], DType::F32)
        .expect("sum tmp");
    let result = gpus.devices[0]
        .alloc_tensor(&[ELEMS], DType::F32)
        .expect("sum result");

    // Graph capture may not perform module load/JIT. Warm the exact peer-load
    // kernels outside capture and prove the direct peer pointers first.
    let warm_values: Vec<Vec<f32>> = (0..ranks)
        .map(|rank| vec![(rank + 1) as f32; ELEMS])
        .collect();
    for rank in 0..ranks {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind warm source");
        let host = &warm_values[rank];
        let bytes =
            unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), host.len() * 4) };
        gpu.hip
            .memcpy_htod(&sources[rank].buf, bytes)
            .expect("warm source upload");
    }
    gpus.devices[0].bind_thread().expect("bind warm peer sum");
    if ranks == 3 {
        gpus.devices[0]
            .add_f32_graph_safe(&sources[1], &sources[2], &result)
            .expect("warm peer sum");
    } else {
        gpus.devices[0]
            .add_f32_graph_safe(&sources[1], &sources[2], &tmp)
            .expect("warm first peer sum");
        gpus.devices[0]
            .add_f32_graph_safe(&tmp, &sources[3], &result)
            .expect("warm second peer sum");
    }
    gpus.devices[0]
        .hip
        .stream_synchronize(
            gpus.devices[0]
                .active_stream
                .as_ref()
                .expect("active stream"),
        )
        .expect("sync warm peer sum");
    let warm_result = gpus.devices[0]
        .download_f32(&result)
        .expect("download warm peer sum");
    assert!(
        warm_result
            .iter()
            .all(|&value| value == (2..=ranks).sum::<usize>() as f32),
        "direct peer sum failed before graph capture: head {:?}",
        &warm_result[..16]
    );

    // JIT both graph-resident barrier symbols outside capture. Enqueue all
    // stores before any waits so this warmup cannot deadlock.
    for rank in 0..ranks {
        gpus.devices[rank]
            .tp_graph_signal_store_gfx1201(&signals[rank], 1)
            .expect("warm graph signal store");
    }
    for destination in 0..ranks {
        let peer_signals: Vec<&DeviceBuffer> = (0..ranks)
            .filter(|&source| source != destination)
            .map(|source| &signals[source])
            .collect();
        if ranks == 3 {
            gpus.devices[destination]
                .tp_graph_signal_wait2_gfx1201([peer_signals[0], peer_signals[1]], 1)
                .expect("warm graph signal wait2");
        } else {
            gpus.devices[destination]
                .tp_graph_signal_wait3_gfx1201(
                    [peer_signals[0], peer_signals[1], peer_signals[2]],
                    1,
                )
                .expect("warm graph signal wait3");
        }
    }
    for rank in 0..ranks {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind barrier warm sync");
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
            .expect("sync barrier warmup");
        gpu.hip
            .memset(&signals[rank], 0, signals[rank].size())
            .expect("reset warmed signal");
    }

    // Begin every device capture before adding any cross-device dependency.
    for rank in 0..ranks {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind capture begin");
        gpu.graphs.capture_blobs.clear();
        gpu.graphs.capture_mode = true;
        gpu.hip
            .stream_begin_capture(
                gpu.active_stream.as_ref().expect("active stream"),
                CAPTURE_MODE_RELAXED,
            )
            .expect("begin rank capture");
    }
    for rank in 0..ranks {
        gpus.devices[rank]
            .tp_graph_signal_store_gfx1201(&signals[rank], 1)
            .expect("capture producer signal");
    }
    for destination in 0..ranks {
        let peer_signals: Vec<&DeviceBuffer> = (0..ranks)
            .filter(|&source| source != destination)
            .map(|source| &signals[source])
            .collect();
        if ranks == 3 {
            gpus.devices[destination]
                .tp_graph_signal_wait2_gfx1201([peer_signals[0], peer_signals[1]], 1)
                .expect("capture peer signal wait2");
        } else {
            gpus.devices[destination]
                .tp_graph_signal_wait3_gfx1201(
                    [peer_signals[0], peer_signals[1], peer_signals[2]],
                    1,
                )
                .expect("capture peer signal wait3");
        }
    }

    // Use the same graph-safe peer-pointer load shape as the promoted TP4 HC
    // consumer. Captured hipMemcpyPeerAsync is deliberately not used: on the
    // current ROCm stack it instantiates but replays as a no-op.
    gpus.devices[0].bind_thread().expect("bind peer sum");
    if ranks == 3 {
        gpus.devices[0]
            .add_f32_graph_safe(&sources[1], &sources[2], &result)
            .expect("capture peer sum");
    } else {
        gpus.devices[0]
            .add_f32_graph_safe(&sources[1], &sources[2], &tmp)
            .expect("capture first peer sum");
        gpus.devices[0]
            .add_f32_graph_safe(&tmp, &sources[3], &result)
            .expect("capture second peer sum");
    }

    let mut captures = Vec::with_capacity(ranks);
    for rank in 0..ranks {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind capture end");
        let graph = gpu
            .hip
            .stream_end_capture(gpu.active_stream.as_ref().expect("active stream"))
            .expect("end rank capture");
        gpu.graphs.capture_mode = false;
        let exec = gpu
            .hip
            .graph_instantiate(&graph)
            .expect("instantiate graph");
        let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
        captures.push(CapturedRank {
            graph,
            exec,
            _blobs: blobs,
        });
    }

    for replay in 0..replays {
        let host_values: Vec<Vec<f32>> = (0..ranks)
            .map(|rank| vec![(replay * ranks + rank + 1) as f32; ELEMS])
            .collect();

        // Reset every per-rank barrier slot before any graph is launched. The
        // production route can do this with one small async memset per rank.
        for rank in 0..ranks {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind signal reset");
            gpu.hip
                .memset(&signals[rank], 0, signals[rank].size())
                .expect("reset graph signal");
        }

        // Enqueue every producer update before launching any graph. The large
        // prior write makes a stale event record observable rather than a race
        // that usually happens to finish before rank zero reads the peer.
        for rank in 0..ranks {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind producer update");
            gpu.hip
                .memset_async(
                    &delays[rank],
                    replay as i32,
                    DELAY_BYTES,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("enqueue producer delay");
            let host = &host_values[rank];
            let bytes =
                unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), host.len() * 4) };
            gpu.hip
                .memcpy_htod_async(
                    &sources[rank].buf,
                    bytes,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("enqueue producer update");
        }
        for rank in 0..ranks {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind graph launch");
            gpu.hip
                .graph_launch(
                    &captures[rank].exec,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("launch rank graph");
        }
        for rank in 0..ranks {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind replay sync");
            gpu.hip
                .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
                .expect("sync rank replay");
        }

        gpus.devices[0].bind_thread().expect("bind result check");
        let host = gpus.devices[0]
            .download_f32(&result)
            .expect("download peer sum");
        let expected = (1..ranks)
            .map(|rank| (replay * ranks + rank + 1) as f32)
            .sum::<f32>();
        assert!(
            host.iter().all(|&value| value == expected),
            "replay {replay}: stale or unordered peer sum; expected {expected}, head {:?}",
            &host[..16]
        );
    }

    for rank in 0..ranks {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind graph destroy");
        let capture = captures.remove(0);
        gpu.hip
            .graph_exec_destroy(capture.exec)
            .expect("destroy graph exec");
        gpu.hip.graph_destroy(capture.graph).expect("destroy graph");
    }

    println!("PASS tp{ranks} cross-device graph barrier: {replays} exact replays");
}
