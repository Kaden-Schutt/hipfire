// SPDX-License-Identifier: MIT
// hipfire — end-to-end smoke for Gpus::all_reduce_sum_f32 (Stage 2).
//
// Verifies:
//   1. Lazy RCCL init via Gpus::ensure_rccl.
//   2. Correctness: rank r fills its buffer with value (r+1).0; after
//      all-reduce-sum every rank reads back N*(N+1)/2.
//   3. Latency matches the direct RCCL smoke
//      (crates/hip-bridge/examples/rccl_smoke.rs) — no orchestrator
//      overhead beyond the FFI wrapper.
//
// Run:
//   HIP_VISIBLE_DEVICES=0,1,2,3 cargo run -p hipfire-runtime --release \
//       --example tp_allreduce_smoke
//   HIP_VISIBLE_DEVICES=0,1 HIPFIRE_TP_BENCH_N=2 cargo run ... (TP=2)

use hip_bridge::DeviceBuffer;
use hipfire_runtime::multi_gpu::Gpus;
use std::time::Instant;

const SIZES_BYTES: &[usize] = &[4 * 1024, 32 * 1024, 128 * 1024, 512 * 1024];

fn read_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n_ranks = read_env("HIPFIRE_TP_BENCH_N", 4);
    let iters = read_env("HIPFIRE_TP_BENCH_ITERS", 100);
    let warmup = read_env("HIPFIRE_TP_BENCH_WARMUP", 10);

    println!("=== Gpus::all_reduce_sum_f32 smoke ===");
    println!("ranks: {n_ranks}, iters: {iters}, warmup: {warmup}");

    // n_layers placeholder — init_uniform requires n_layers >= n_devices.
    // The TP path doesn't care about layer-to-device; we just need devices.
    let mut gpus = Gpus::init_uniform(n_ranks, n_ranks).expect("init_uniform");
    let peer_ok = gpus.enable_peer_all().expect("enable_peer_all");
    if !peer_ok {
        eprintln!("WARN: peer access incomplete (host-staging fallback applies)");
    }

    // Set per-rank streams. RCCL needs a real stream per comm; without
    // active_stream set, all_reduce_sum_f32 returns an error.
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Allocate one buffer per rank at the max size; reuse across cells.
    let max_bytes = *SIZES_BYTES.iter().max().unwrap();
    let buffers: Vec<DeviceBuffer> = (0..n_ranks)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            gpus.devices[i].hip.malloc(max_bytes).expect("malloc")
        })
        .collect();

    // ── Correctness: fill rank r with (r+1).0, all-reduce, verify ────
    println!("\n--- correctness check ---");
    {
        let count = SIZES_BYTES[0] / std::mem::size_of::<f32>(); // 1024 f32s
        let expected: f32 = (1..=n_ranks).map(|x| x as f32).sum();

        for r in 0..n_ranks {
            let pattern: Vec<f32> = vec![(r as f32) + 1.0; count];
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    pattern.as_ptr() as *const u8,
                    pattern.len() * std::mem::size_of::<f32>(),
                )
            };
            gpus.devices[r].bind_thread().expect("bind");
            gpus.devices[r]
                .hip
                .memcpy_htod(&buffers[r], bytes)
                .expect("H2D pattern");
        }

        let refs: Vec<&DeviceBuffer> = buffers.iter().collect();
        gpus.all_reduce_sum_f32(&refs, count)
            .expect("all_reduce_sum_f32");

        // Sync all rank streams before readback.
        for dev in &gpus.devices {
            dev.bind_thread().expect("bind");
            dev.hip
                .stream_synchronize(dev.active_stream.as_ref().unwrap())
                .expect("sync");
        }

        // Verify each rank's buffer == expected sum.
        let mut all_ok = true;
        for r in 0..n_ranks {
            let mut out = vec![0u8; count * std::mem::size_of::<f32>()];
            gpus.devices[r].bind_thread().expect("bind");
            gpus.devices[r]
                .hip
                .memcpy_dtoh(&mut out, &buffers[r])
                .expect("D2H");
            let out_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(out.as_ptr() as *const f32, count)
            };
            let first = out_f32[0];
            let last = out_f32[count - 1];
            let mid = out_f32[count / 2];
            let ok = (first - expected).abs() < 1e-5
                && (last - expected).abs() < 1e-5
                && (mid - expected).abs() < 1e-5;
            println!(
                "  rank {r}: buf[0]={first:.2}, buf[mid]={mid:.2}, buf[last]={last:.2} \
                 (expected {expected:.2}) {}",
                if ok { "OK" } else { "FAIL" }
            );
            if !ok {
                all_ok = false;
            }
        }
        assert!(all_ok, "correctness check failed");
        println!("  correctness: PASS (sum of 1.0+2.0+...+{n_ranks}.0 = {expected:.1})");
    }

    // ── Latency: time vs the direct RCCL bench (no orchestrator overhead) ──
    println!("\n--- latency vs hip-bridge::rccl_smoke baseline ---");
    println!(
        "{:<14}  {:>10}  {:>10}  {:>10}  {:>10}",
        "size", "median µs", "p10 µs", "p90 µs", "BW GB/s"
    );
    for &bytes in SIZES_BYTES {
        let count = bytes / std::mem::size_of::<f32>();
        let refs: Vec<&DeviceBuffer> = buffers.iter().collect();

        for _ in 0..warmup {
            gpus.all_reduce_sum_f32(&refs, count)
                .expect("all_reduce warm");
            for dev in &gpus.devices {
                dev.bind_thread().expect("bind");
                dev.hip
                    .stream_synchronize(dev.active_stream.as_ref().unwrap())
                    .expect("sync");
            }
        }

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            gpus.all_reduce_sum_f32(&refs, count)
                .expect("all_reduce");
            for dev in &gpus.devices {
                dev.bind_thread().expect("bind");
                dev.hip
                    .stream_synchronize(dev.active_stream.as_ref().unwrap())
                    .expect("sync");
            }
            samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[iters / 2];
        let p10 = samples[iters / 10];
        let p90 = samples[(iters * 9) / 10];
        let bw_gbps = (bytes as f64 / 1e9) / (median / 1e6);
        println!(
            "{:<14}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.2}",
            format!("{} KB", bytes / 1024),
            median,
            p10,
            p90,
            bw_gbps
        );
    }

    // ── Cleanup ──────────────────────────────────────────────────────
    for (i, buf) in buffers.into_iter().enumerate() {
        gpus.devices[i].bind_thread().expect("bind");
        let _ = gpus.devices[i].hip.free(buf);
    }
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        if let Some(s) = dev.active_stream.take() {
            let _ = dev.hip.stream_destroy(s);
        }
    }
    // Drop of `gpus` destroys RCCL comms via RcclComms::Drop.

    println!("\ntp_allreduce_smoke: PASS");
}
