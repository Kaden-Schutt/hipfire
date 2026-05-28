// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! PP boundary-copy microbench (gfx906 ↔ gfx1031).
//!
//! Gate for Stage 2 of the PP+MTP combo. The Stage 2 plan assumes the
//! per-cycle PP boundary copy (residual stream from end of dev 0's
//! band to start of dev 1's band) costs ~200-300 µs at K=4 verify
//! (100 KB), keeping the total combined-system perf within ~5% of
//! single-gpu. If the actual cost is much higher on this hardware
//! pair (e.g. peer access falls back to host-staging without us
//! noticing), Stage 2's perf budget collapses.
//!
//! Measures three operations under `init_layers(&[48, 16])` (mirrors
//! daemon's PP=2 layout):
//!   (A) Single-token decode boundary copy: dim × f32 = 20 KB
//!   (B) Verify-batch boundary copy at K=4: 5 × dim × f32 = 100 KB
//!   (C) Verify-batch boundary copy at K=8: 9 × dim × f32 = 180 KB
//!
//! Each op = `boundary_copy(0→1) + wait_boundary` with active_streams
//! on both devices (the async path the daemon's forward_prefill_pp
//! actually takes).
//!
//! Decision gate:
//!   - median (B) ≤ 500 µs → Stage 2 perf budget realistic, proceed.
//!   - 500 µs – 2 ms     → Stage 2 viable but tighter; document.
//!   - > 2 ms            → host-staging hit; abort Stage 2 until
//!                          peer-access ordering is verified.
//!
//! Run: source scripts/gpu-lock.sh && gpu_acquire pp-boundary-bench && \
//!      env HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_PP_LAYERS=48,16 \
//!      ./target/release/examples/pp_boundary_microbench && gpu_release

use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const DIM: usize = 5120;             // qwen3.6-27b hidden size
const F32_BYTES: usize = 4;
const WARMUP_ITERS: usize = 50;
const MEASURE_ITERS: usize = 1000;

fn main() {
    // 1) Stand up the same Gpus shape the daemon uses for PP=2.
    let per_device: Vec<usize> = std::env::var("HIPFIRE_PP_LAYERS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse::<usize>().unwrap()).collect())
        .unwrap_or_else(|| vec![48, 16]);
    assert_eq!(per_device.len(), 2, "v1 microbench is PP=2 only");
    println!("PP layout: {:?}", per_device);

    let mut gpus = Gpus::init_layers(&per_device).expect("init_layers");
    println!(
        "devices: dev 0 = {} ({}); dev 1 = {} ({})",
        gpus.devices[0].arch, gpus.devices[0].device_id,
        gpus.devices[1].arch, gpus.devices[1].device_id,
    );

    // 2) Set up active_streams so boundary_copy takes the async path
    //    (the path the daemon uses in steady-state). bind_thread to each
    //    device BEFORE stream_create — streams are device-affine
    //    (lesson from MTP hetero bringup).
    for dev_idx in 0..2 {
        gpus.devices[dev_idx].bind_thread().unwrap();
        let stream = gpus.devices[dev_idx].hip.stream_create().unwrap();
        gpus.devices[dev_idx].active_stream = Some(stream);
    }

    // 3) Enable bidirectional peer access AFTER all allocations are live.
    //    Match the daemon's load_model_pp ordering (peer-access-after-alloc
    //    is the ROCm 6.4.3 gotcha pattern).
    let max_op_bytes = 9 * DIM * F32_BYTES; // largest of the three ops below
    let src_dev0 = gpus.devices[0].hip.malloc(max_op_bytes).expect("malloc dev 0");
    let dst_dev1 = gpus.devices[1].hip.malloc(max_op_bytes).expect("malloc dev 1");

    let peer_result = gpus.enable_peer_all().expect("enable_peer_all");
    println!("peer_access enabled bidirectionally: {peer_result}");

    // 4) Seed source buffer so the copy isn't trivially elided.
    let seed: Vec<u8> = (0..max_op_bytes).map(|i| (i & 0xff) as u8).collect();
    gpus.devices[0].bind_thread().unwrap();
    gpus.devices[0].hip.memcpy_htod(&src_dev0, &seed).unwrap();

    // 5) Define the three op shapes.
    let ops: [(&str, usize); 3] = [
        ("(A) decode 1 token  ( 20 KB)", 1 * DIM * F32_BYTES),
        ("(B) verify K=4      (100 KB)", 5 * DIM * F32_BYTES),
        ("(C) verify K=8      (180 KB)", 9 * DIM * F32_BYTES),
    ];

    println!(
        "\nWarmup: {} iters per op. Measure: {} iters per op. Reporting median + p99 + range.",
        WARMUP_ITERS, MEASURE_ITERS,
    );

    let mut results: Vec<(&str, u128, u128, u128, u128)> = Vec::new();
    for (label, n_bytes) in ops.iter() {
        // Warmup.
        for _ in 0..WARMUP_ITERS {
            let evt = gpus.boundary_copy(0, 1, &src_dev0, &dst_dev1, *n_bytes).unwrap();
            gpus.wait_boundary(evt).unwrap();
        }

        // Measure.
        let mut samples: Vec<u128> = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let t = Instant::now();
            let evt = gpus.boundary_copy(0, 1, &src_dev0, &dst_dev1, *n_bytes).unwrap();
            gpus.wait_boundary(evt).unwrap();
            samples.push(t.elapsed().as_micros());
        }

        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        let p99 = samples[(samples.len() * 99) / 100];
        let min = *samples.first().unwrap();
        let max = *samples.last().unwrap();
        results.push((*label, median, p99, min, max));
    }

    println!("\nResults — enqueue+wait only (boundary cost AS HIDDEN by next dispatch):");
    println!(
        "  {:<32} {:>8} {:>8} {:>8} {:>8}",
        "op", "median", "p99", "min", "max",
    );
    for (label, med, p99, mn, mx) in &results {
        println!(
            "  {label:<32} {med:>8} {p99:>8} {mn:>8} {mx:>8}",
        );
    }

    // Re-measure each op WITH a dst device_synchronize, which forces the
    // copy to actually complete before we stop the timer. This is the
    // "what if the next dispatch is dependent and can't hide the copy?"
    // upper bound on per-cycle wall cost.
    println!("\nResults — enqueue+wait+device_sync (bandwidth-limited upper bound):");
    println!(
        "  {:<32} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "op", "median", "p99", "min", "max", "GB/s",
    );
    for (label, n_bytes) in ops.iter() {
        // warmup
        for _ in 0..10 {
            let evt = gpus.boundary_copy(0, 1, &src_dev0, &dst_dev1, *n_bytes).unwrap();
            gpus.wait_boundary(evt).unwrap();
            gpus.devices[1].bind_thread().unwrap();
            let _ = gpus.devices[1].hip.device_synchronize();
        }
        let mut samples: Vec<u128> = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let t = Instant::now();
            let evt = gpus.boundary_copy(0, 1, &src_dev0, &dst_dev1, *n_bytes).unwrap();
            gpus.wait_boundary(evt).unwrap();
            gpus.devices[1].bind_thread().unwrap();
            let _ = gpus.devices[1].hip.device_synchronize();
            samples.push(t.elapsed().as_micros());
        }
        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        let p99 = samples[(samples.len() * 99) / 100];
        let min = *samples.first().unwrap();
        let max = *samples.last().unwrap();
        let gb_s = (*n_bytes as f64) / (median as f64) * 1e-3; // bytes/µs = MB/s; /1000 = GB/s
        println!(
            "  {label:<32} {median:>8} {p99:>8} {min:>8} {max:>8} {gb_s:>10.2}",
        );
    }

    // 6) Decision gate on (B), the steady-state verify boundary cost.
    let verify_median = results[1].1;
    println!("\n── Decision gate (Stage 2 PP+MTP combo, op B steady-state) ──");
    println!("op B (verify K=4 100 KB) median: {} µs", verify_median);
    let combined_per_cycle = verify_median + 38; // 38 µs from prev mtp_peer_copy_microbench (op A peer copy 20 KB raw)
    println!(
        "projected per-cycle combined overhead (PP boundary + MTP same-device handoff): ~{} µs",
        combined_per_cycle,
    );
    if verify_median <= 500 {
        println!("VERDICT: ≤500 µs — Stage 2 perf budget realistic. PROCEED.");
    } else if verify_median <= 2000 {
        println!("VERDICT: 500 µs – 2 ms — viable but tighter. Document then proceed.");
    } else {
        println!("VERDICT: >2 ms — peer access likely host-staging. ABORT Stage 2 until verified.");
    }

    // Cleanup.
    gpus.devices[0].hip.free(src_dev0).unwrap();
    gpus.devices[1].hip.free(dst_dev1).unwrap();
    for dev_idx in 0..2 {
        if let Some(stream) = gpus.devices[dev_idx].active_stream.take() {
            let _ = gpus.devices[dev_idx].hip.stream_destroy(stream);
        }
    }

    let _ = DType::F32; // silence unused import on some build configs
    let _ = std::mem::size_of::<Gpu>();
}
