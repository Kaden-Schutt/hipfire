// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — TP3 gfx1201 graph-resident HC barrier fusion screen.

use hip_bridge::{DeviceBuffer, Graph, GraphExec};
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, GpuTensor};
use std::time::Instant;

const RANKS: usize = 3;
const HIDDEN: usize = 4_096;
const BARRIERS: usize = 86;
const TRIALS: usize = 9;
const REPLAYS_PER_TRIAL: usize = 16;
const CAPTURE_MODE_RELAXED: u32 = 2;

#[derive(Clone, Copy)]
enum Variant {
    Baseline,
    Fused,
}

struct CapturedRank {
    graph: Graph,
    exec: GraphExec,
    _blobs: Vec<Vec<u8>>,
}

fn peer_signal_refs(signals: &[DeviceBuffer], rank: usize) -> [&DeviceBuffer; 2] {
    match rank {
        0 => [&signals[1], &signals[2]],
        1 => [&signals[0], &signals[2]],
        2 => [&signals[0], &signals[1]],
        _ => unreachable!("TP3 rank must be 0..3"),
    }
}

fn deterministic_values(len: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bits = index
                .wrapping_mul(1_664_525)
                .wrapping_add(seed.wrapping_mul(1_013_904_223));
            ((bits & 0xffff) as f32 / 65_535.0 - 0.5) * scale
        })
        .collect()
}

fn reset_signals(gpus: &Gpus, signals: &[DeviceBuffer]) {
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind signal reset");
        gpu.hip
            .memset(&signals[rank], 0, signals[rank].size())
            .expect("reset signal");
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_boundary(
    gpus: &mut Gpus,
    variant: Variant,
    x_in: &[GpuTensor],
    a_matrix: &[GpuTensor],
    scale: &[GpuTensor],
    transforms: &[GpuTensor],
    signals: &[DeviceBuffer],
    outputs: &[GpuTensor],
    epoch: u32,
) {
    if matches!(variant, Variant::Baseline) {
        for rank in 0..RANKS {
            gpus.devices[rank]
                .tp_graph_signal_store_gfx1201(&signals[rank], epoch)
                .expect("baseline signal store");
        }
        for rank in 0..RANKS {
            gpus.devices[rank]
                .tp_graph_signal_wait2_gfx1201(peer_signal_refs(signals, rank), epoch)
                .expect("baseline signal wait");
        }
    }

    let peers = [&transforms[0], &transforms[1], &transforms[2]];
    for rank in 0..RANKS {
        match variant {
            Variant::Baseline => gpus.devices[rank]
                .hc_mix_4stream_peer3_gfx1201(
                    &x_in[rank],
                    &a_matrix[rank],
                    &scale[rank],
                    peers,
                    &outputs[rank],
                    HIDDEN as i32,
                )
                .expect("baseline HC consumer"),
            Variant::Fused => gpus.devices[rank]
                .hc_mix_4stream_peer3_sync_gfx1201(
                    &x_in[rank],
                    &a_matrix[rank],
                    &scale[rank],
                    peers,
                    &signals[rank],
                    peer_signal_refs(signals, rank),
                    epoch,
                    &outputs[rank],
                    HIDDEN as i32,
                )
                .expect("fused HC consumer"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn capture(
    gpus: &mut Gpus,
    variant: Variant,
    x_in: &[GpuTensor],
    a_matrix: &[GpuTensor],
    scale: &[GpuTensor],
    transforms: &[GpuTensor],
    signals: &[DeviceBuffer],
    outputs: &[GpuTensor],
) -> Vec<CapturedRank> {
    for rank in 0..RANKS {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind capture begin");
        gpu.graphs.capture_blobs.clear();
        gpu.graphs.capture_mode = true;
        gpu.hip
            .stream_begin_capture(
                gpu.active_stream.as_ref().expect("active stream"),
                CAPTURE_MODE_RELAXED,
            )
            .expect("begin capture");
    }
    for epoch in 1..=BARRIERS as u32 {
        launch_boundary(
            gpus, variant, x_in, a_matrix, scale, transforms, signals, outputs, epoch,
        );
    }

    let mut captures = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind capture end");
        let graph = gpu
            .hip
            .stream_end_capture(gpu.active_stream.as_ref().expect("active stream"))
            .expect("end capture");
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
    captures
}

fn replay_once(gpus: &Gpus, captures: &[CapturedRank], signals: &[DeviceBuffer]) {
    reset_signals(gpus, signals);
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind graph launch");
        gpu.hip
            .graph_launch(
                &captures[rank].exec,
                gpu.active_stream.as_ref().expect("active stream"),
            )
            .expect("launch graph");
    }
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind graph sync");
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
            .expect("sync graph");
    }
}

fn time_arm(gpus: &Gpus, captures: &[CapturedRank], signals: &[DeviceBuffer]) -> f64 {
    let start = Instant::now();
    for _ in 0..REPLAYS_PER_TRIAL {
        replay_once(gpus, captures, signals);
    }
    start.elapsed().as_secs_f64() * 1.0e3 / REPLAYS_PER_TRIAL as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn download_outputs(gpus: &mut Gpus, outputs: &[GpuTensor]) -> Vec<Vec<f32>> {
    (0..RANKS)
        .map(|rank| {
            gpus.devices[rank]
                .download_f32(&outputs[rank])
                .expect("download output")
        })
        .collect()
}

fn destroy(gpus: &Gpus, captures: Vec<CapturedRank>) {
    for (rank, capture) in captures.into_iter().enumerate() {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind graph destroy");
        gpu.hip
            .graph_exec_destroy(capture.exec)
            .expect("destroy graph exec");
        gpu.hip.graph_destroy(capture.graph).expect("destroy graph");
    }
}

fn main() {
    let mut gpus = Gpus::init_uniform(RANKS, RANKS).expect("init TP3 GPUs");
    for (rank, gpu) in gpus.devices.iter().enumerate() {
        assert_eq!(gpu.arch, "gfx1201", "rank {rank} must be gfx1201");
    }
    assert!(
        gpus.enable_peer_all().expect("enable peer access"),
        "complete TP3 peer access required"
    );
    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind stream owner");
        gpu.active_stream = Some(gpu.hip.stream_create().expect("create stream"));
    }

    let x_host = deterministic_values(4 * HIDDEN, 11, 0.25);
    let a_host = deterministic_values(16, 17, 0.125);
    let scale_host = deterministic_values(4, 23, 0.25);
    let mut x_in = Vec::with_capacity(RANKS);
    let mut a_matrix = Vec::with_capacity(RANKS);
    let mut scales = Vec::with_capacity(RANKS);
    let mut transforms = Vec::with_capacity(RANKS);
    let mut baseline_out = Vec::with_capacity(RANKS);
    let mut fused_out = Vec::with_capacity(RANKS);
    let mut signals = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gpu = &mut gpus.devices[rank];
        x_in.push(gpu.upload_f32(&x_host, &[4, HIDDEN]).expect("upload x"));
        a_matrix.push(gpu.upload_f32(&a_host, &[4, 4]).expect("upload A"));
        scales.push(gpu.upload_f32(&scale_host, &[4]).expect("upload scale"));
        transforms.push(
            gpu.upload_f32(&deterministic_values(HIDDEN, 101 + rank, 0.5), &[HIDDEN])
                .expect("upload transform"),
        );
        baseline_out.push(
            gpu.alloc_tensor(&[4, HIDDEN], DType::F32)
                .expect("baseline output"),
        );
        fused_out.push(
            gpu.alloc_tensor(&[4, HIDDEN], DType::F32)
                .expect("fused output"),
        );
        signals.push(
            gpu.hip
                .malloc_signal(std::mem::size_of::<u64>())
                .expect("allocate signal"),
        );
    }

    // JIT every symbol outside graph capture. Enqueue all ranks before any
    // synchronization so neither the baseline nor fused handshake can deadlock.
    reset_signals(&gpus, &signals);
    launch_boundary(
        &mut gpus,
        Variant::Baseline,
        &x_in,
        &a_matrix,
        &scales,
        &transforms,
        &signals,
        &baseline_out,
        1,
    );
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
            .expect("baseline warm sync");
    }
    reset_signals(&gpus, &signals);
    launch_boundary(
        &mut gpus,
        Variant::Fused,
        &x_in,
        &a_matrix,
        &scales,
        &transforms,
        &signals,
        &fused_out,
        1,
    );
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
            .expect("fused warm sync");
    }

    let baseline_graphs = capture(
        &mut gpus,
        Variant::Baseline,
        &x_in,
        &a_matrix,
        &scales,
        &transforms,
        &signals,
        &baseline_out,
    );
    let fused_graphs = capture(
        &mut gpus,
        Variant::Fused,
        &x_in,
        &a_matrix,
        &scales,
        &transforms,
        &signals,
        &fused_out,
    );

    replay_once(&gpus, &baseline_graphs, &signals);
    let expected = download_outputs(&mut gpus, &baseline_out);
    replay_once(&gpus, &fused_graphs, &signals);
    let actual = download_outputs(&mut gpus, &fused_out);
    let mut comparisons = 0usize;
    for rank in 0..RANKS {
        for (index, (&reference, &candidate)) in
            expected[rank].iter().zip(&actual[rank]).enumerate()
        {
            assert_eq!(
                reference.to_bits(),
                candidate.to_bits(),
                "rank={rank} output={index} raw-bit mismatch"
            );
            comparisons += 1;
        }
    }

    for _ in 0..3 {
        replay_once(&gpus, &baseline_graphs, &signals);
        replay_once(&gpus, &fused_graphs, &signals);
    }
    let mut baseline_ms = Vec::with_capacity(TRIALS);
    let mut fused_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        if trial & 1 == 0 {
            baseline_ms.push(time_arm(&gpus, &baseline_graphs, &signals));
            fused_ms.push(time_arm(&gpus, &fused_graphs, &signals));
        } else {
            fused_ms.push(time_arm(&gpus, &fused_graphs, &signals));
            baseline_ms.push(time_arm(&gpus, &baseline_graphs, &signals));
        }
    }
    let baseline_ms = median(&mut baseline_ms);
    let fused_ms = median(&mut fused_ms);
    println!(
        "RESULT ranks={RANKS} barriers={BARRIERS} hidden={HIDDEN} trials={TRIALS} replays_per_trial={REPLAYS_PER_TRIAL} raw_bit_comparisons={comparisons} baseline_nodes={} fused_nodes={} removed_nodes={} baseline_ms={baseline_ms:.6} fused_ms={fused_ms:.6} speedup_x={:.4} saved_ms={:.6} projected_product_pct={:.3}",
        RANKS * BARRIERS * 3,
        RANKS * BARRIERS,
        RANKS * BARRIERS * 2,
        baseline_ms / fused_ms,
        baseline_ms - fused_ms,
        (baseline_ms - fused_ms) / (1000.0 / 54.903757) * 100.0,
    );

    destroy(&gpus, baseline_graphs);
    destroy(&gpus, fused_graphs);
}
