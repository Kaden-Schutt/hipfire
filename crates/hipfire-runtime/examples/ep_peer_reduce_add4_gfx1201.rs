// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness and latency screen for the four-rank gfx1201 DS4 EP reduction.
//!
//! Compares the shipping RCCL all-reduce + local add sequence against the
//! candidate all-rank stream barrier + one peer-read reduction/add kernel per
//! rank at DS4's 4096-f32 routed-partial shape.

use hip_bridge::DeviceBuffer;
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, GpuTensor};
use std::time::Instant;

const RANKS: usize = 4;
const COUNT: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 200;

fn tensor(buf: &DeviceBuffer) -> GpuTensor {
    GpuTensor {
        // SAFETY: the owning buffer outlives the borrowed tensor used by this
        // process-local benchmark.
        buf: unsafe { buf.alias() },
        shape: vec![COUNT],
        dtype: DType::F32,
    }
}

fn sync_all(gpus: &Gpus) {
    for dev in &gpus.devices {
        dev.bind_thread().expect("bind");
        dev.hip
            .stream_synchronize(dev.active_stream.as_ref().expect("active stream"))
            .expect("stream sync");
    }
}

fn upload(gpus: &Gpus, buffers: &[DeviceBuffer], values: &[f32]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    for (r, buffer) in buffers.iter().enumerate() {
        gpus.devices[r].bind_thread().expect("bind");
        gpus.devices[r].hip.memcpy_htod(buffer, bytes).expect("H2D");
    }
}

fn download(gpus: &Gpus, buffers: &[DeviceBuffer]) -> Vec<Vec<u8>> {
    buffers
        .iter()
        .enumerate()
        .map(|(r, buffer)| {
            let mut bytes = vec![0u8; COUNT * std::mem::size_of::<f32>()];
            gpus.devices[r].bind_thread().expect("bind");
            gpus.devices[r]
                .hip
                .memcpy_dtoh(&mut bytes, buffer)
                .expect("D2H");
            bytes
        })
        .collect()
}

fn allocate_set(gpus: &mut Gpus, bytes: usize) -> Vec<DeviceBuffer> {
    (0..RANKS)
        .map(|r| {
            gpus.devices[r].bind_thread().expect("bind");
            gpus.devices[r].hip.malloc(bytes).expect("malloc")
        })
        .collect()
}

fn percentile(samples: &mut [f64], numerator: usize, denominator: usize) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[(samples.len() - 1) * numerator / denominator]
}

fn main() {
    println!("=== gfx1201 four-rank EP peer-reduce screen ===");
    let mut gpus = Gpus::init_uniform(RANKS, RANKS).expect("init four ranks");
    assert!(
        gpus.devices.iter().all(|gpu| gpu.arch_caps.is_gfx1201()),
        "this screen requires four gfx1201 devices"
    );
    assert!(
        gpus.enable_peer_all().expect("enable peer access"),
        "all-to-all peer access is required"
    );
    for dev in &mut gpus.devices {
        dev.bind_thread().expect("bind");
        dev.active_stream = Some(dev.hip.stream_create().expect("stream create"));
    }

    let bytes = COUNT * std::mem::size_of::<f32>();
    let baseline_partials = allocate_set(&mut gpus, bytes);
    let baseline_dst = allocate_set(&mut gpus, bytes);
    let candidate_partials = allocate_set(&mut gpus, bytes);
    let candidate_dst = allocate_set(&mut gpus, bytes);
    let baseline_partial_tensors: Vec<GpuTensor> = baseline_partials.iter().map(tensor).collect();
    let baseline_dst_tensors: Vec<GpuTensor> = baseline_dst.iter().map(tensor).collect();
    let candidate_partial_tensors: Vec<GpuTensor> = candidate_partials.iter().map(tensor).collect();
    let candidate_dst_tensors: Vec<GpuTensor> = candidate_dst.iter().map(tensor).collect();

    // Nonuniform representable values exercise all four source pointers while
    // retaining an exact expected result.
    for r in 0..RANKS {
        let values: Vec<f32> = (0..COUNT)
            .map(|i| (r as f32 + 1.0) * 0.25 + (i % 8) as f32)
            .collect();
        let value_bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes) };
        for buffer in [&baseline_partials[r], &candidate_partials[r]] {
            gpus.devices[r].bind_thread().expect("bind");
            gpus.devices[r]
                .hip
                .memcpy_htod(buffer, value_bytes)
                .expect("H2D partial");
        }
    }
    let dst_values: Vec<f32> = (0..COUNT).map(|i| 100.0 + (i % 4) as f32).collect();
    upload(&gpus, &baseline_dst, &dst_values);
    upload(&gpus, &candidate_dst, &dst_values);

    let baseline_refs: Vec<&DeviceBuffer> = baseline_partials.iter().collect();
    gpus.all_reduce_sum_f32(&baseline_refs, COUNT)
        .expect("RCCL all-reduce");
    for r in 0..RANKS {
        gpus.devices[r]
            .add_inplace_f32(&baseline_dst_tensors[r], &baseline_partial_tensors[r])
            .expect("baseline add");
    }
    sync_all(&gpus);

    gpus.barrier_rank_streams().expect("rank barrier");
    let candidate_peers = [
        &candidate_partial_tensors[0],
        &candidate_partial_tensors[1],
        &candidate_partial_tensors[2],
        &candidate_partial_tensors[3],
    ];
    for r in 0..RANKS {
        gpus.devices[r]
            .ep_peer_reduce_add4_f32_gfx1201(&candidate_dst_tensors[r], candidate_peers)
            .expect("peer reduce add");
    }
    sync_all(&gpus);

    let baseline_out = download(&gpus, &baseline_dst);
    let candidate_out = download(&gpus, &candidate_dst);
    assert!(
        baseline_out.iter().all(|out| out == &baseline_out[0]),
        "RCCL baseline differs across ranks"
    );
    assert!(
        candidate_out.iter().all(|out| out == &candidate_out[0]),
        "peer-reduce candidate differs across ranks"
    );
    assert_eq!(
        candidate_out[0], baseline_out[0],
        "peer-reduce candidate is not byte-identical to RCCL+add"
    );
    println!("correctness: PASS (all ranks and baseline byte-identical)");

    // Timing uses fresh buffer sets so correctness setup does not affect JIT or
    // RCCL initialization. Repeated accumulation may overflow, but neither
    // path branches on values; only the dispatch/collective latency is measured.
    let timed_baseline_partials = allocate_set(&mut gpus, bytes);
    let timed_baseline_dst = allocate_set(&mut gpus, bytes);
    let timed_candidate_partials = allocate_set(&mut gpus, bytes);
    let timed_candidate_dst = allocate_set(&mut gpus, bytes);
    let timed_baseline_partial_tensors: Vec<GpuTensor> =
        timed_baseline_partials.iter().map(tensor).collect();
    let timed_baseline_dst_tensors: Vec<GpuTensor> =
        timed_baseline_dst.iter().map(tensor).collect();
    let timed_candidate_partial_tensors: Vec<GpuTensor> =
        timed_candidate_partials.iter().map(tensor).collect();
    let timed_candidate_dst_tensors: Vec<GpuTensor> =
        timed_candidate_dst.iter().map(tensor).collect();
    let timed_baseline_refs: Vec<&DeviceBuffer> = timed_baseline_partials.iter().collect();
    let timed_candidate_peers = [
        &timed_candidate_partial_tensors[0],
        &timed_candidate_partial_tensors[1],
        &timed_candidate_partial_tensors[2],
        &timed_candidate_partial_tensors[3],
    ];

    let run_baseline = |gpus: &mut Gpus| {
        gpus.all_reduce_sum_f32(&timed_baseline_refs, COUNT)
            .expect("RCCL all-reduce");
        for r in 0..RANKS {
            gpus.devices[r]
                .add_inplace_f32(
                    &timed_baseline_dst_tensors[r],
                    &timed_baseline_partial_tensors[r],
                )
                .expect("baseline add");
        }
        sync_all(gpus);
    };
    let run_candidate = |gpus: &mut Gpus| {
        gpus.barrier_rank_streams().expect("rank barrier");
        for r in 0..RANKS {
            gpus.devices[r]
                .ep_peer_reduce_add4_f32_gfx1201(
                    &timed_candidate_dst_tensors[r],
                    timed_candidate_peers,
                )
                .expect("peer reduce add");
        }
        sync_all(gpus);
    };

    for _ in 0..WARMUP {
        run_baseline(&mut gpus);
        run_candidate(&mut gpus);
    }
    let mut baseline_us = Vec::with_capacity(ITERS);
    let mut candidate_us = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let started = Instant::now();
        run_baseline(&mut gpus);
        baseline_us.push(started.elapsed().as_secs_f64() * 1e6);
        let started = Instant::now();
        run_candidate(&mut gpus);
        candidate_us.push(started.elapsed().as_secs_f64() * 1e6);
    }
    let baseline_median = percentile(&mut baseline_us, 1, 2);
    let baseline_p10 = percentile(&mut baseline_us, 1, 10);
    let baseline_p90 = percentile(&mut baseline_us, 9, 10);
    let candidate_median = percentile(&mut candidate_us, 1, 2);
    let candidate_p10 = percentile(&mut candidate_us, 1, 10);
    let candidate_p90 = percentile(&mut candidate_us, 9, 10);
    let saved_us = baseline_median - candidate_median;
    let projected_ms = saved_us * 43.0 / 1000.0;
    println!(
        "RCCL+add: median={baseline_median:.3}us p10={baseline_p10:.3}us p90={baseline_p90:.3}us"
    );
    println!(
        "peer+add: median={candidate_median:.3}us p10={candidate_p10:.3}us p90={candidate_p90:.3}us"
    );
    println!("delta: {saved_us:.3}us/layer, projected {projected_ms:.3}ms/token across 43 layers");

    println!("ep_peer_reduce_add4_gfx1201: PASS");
}
