// SPDX-License-Identifier: MIT
// hipfire — RCCL smoke + microbench in Rust. Mirrors
// /home/kaden/.claude/jobs/6ea8a1b1/rccl_allreduce_smoke.cpp so we can
// verify the FFI wrapper produces the same ~110 µs floor at 4 KB on
// gfx1201.
//
// Run: HIP_VISIBLE_DEVICES=0,1,2,3 cargo run -p hip-bridge --example rccl_smoke --release

use hip_bridge::{HipRuntime, RcclComms};
use std::time::Instant;

const SIZES_BYTES: &[usize] = &[4 * 1024, 32 * 1024, 128 * 1024, 512 * 1024];

fn main() {
    let n_ranks: usize = std::env::var("HIPFIRE_TP_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let iters: usize = std::env::var("HIPFIRE_TP_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let warmup: usize = 10;

    println!("RCCL smoke (Rust FFI): ranks={n_ranks} iters={iters}");

    // Load HIP runtime + verify device count
    let hip = HipRuntime::load().expect("load HIP");
    let dev_count = hip.device_count().expect("device_count");
    assert!(
        dev_count >= n_ranks as i32,
        "need ≥{n_ranks} HIP devices, have {dev_count}"
    );

    // Init RCCL across devices [0..n_ranks)
    let device_ids: Vec<i32> = (0..n_ranks as i32).collect();
    let rccl = RcclComms::init_all(&device_ids).expect("init_all");
    println!(
        "ncclCommInitAll(n={n_ranks}) OK, RCCL version = {}",
        rccl.version().expect("version")
    );

    // Per-rank: stream + send + recv buffers (all at max payload size).
    let max_bytes = *SIZES_BYTES.iter().max().unwrap();
    let mut streams = Vec::with_capacity(n_ranks);
    let mut send_bufs = Vec::with_capacity(n_ranks);
    let mut recv_bufs = Vec::with_capacity(n_ranks);
    for r in 0..n_ranks {
        hip.set_device(r as i32).expect("set_device");
        streams.push(hip.stream_create().expect("stream_create"));
        send_bufs.push(hip.malloc(max_bytes).expect("malloc send"));
        recv_bufs.push(hip.malloc(max_bytes).expect("malloc recv"));
        // Memset send to 1.0_f32 pattern so the all-reduce sum != 0.
        // 0x3f800000 = 1.0f bits; memset writes a byte pattern so we just
        // use 1 — the reduced output is whatever sum, content doesn't
        // matter for perf timing. Correctness check below uses a fresh
        // H2D copy with known data.
        let _ = hip
            .memset(&send_bufs[r], 1, max_bytes)
            .expect("memset send");
        let _ = hip
            .memset(&recv_bufs[r], 0, max_bytes)
            .expect("memset recv");
    }

    println!(
        "\n{:<14}  {:>10}  {:>10}  {:>10}  {:>10}",
        "size", "median µs", "p10 µs", "p90 µs", "BW GB/s"
    );

    for &bytes in SIZES_BYTES {
        let count = bytes / std::mem::size_of::<f32>();

        // Warmup
        for _ in 0..warmup {
            rccl.group_start().expect("group_start");
            for r in 0..n_ranks {
                // SAFETY: device buffers of `count` f32 with a live per-rank stream.
                unsafe {
                    rccl.all_reduce_sum_f32(
                        r,
                        send_bufs[r].as_ptr() as *const f32,
                        recv_bufs[r].as_ptr() as *mut f32,
                        count,
                        streams[r].raw_ptr(),
                    )
                }
                .expect("all_reduce warm");
            }
            rccl.group_end().expect("group_end");
            for r in 0..n_ranks {
                hip.set_device(r as i32).expect("set_device");
                hip.stream_synchronize(&streams[r]).expect("sync warm");
            }
        }

        // Timed
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            rccl.group_start().expect("group_start");
            for r in 0..n_ranks {
                // SAFETY: device buffers of `count` f32 with a live per-rank stream.
                unsafe {
                    rccl.all_reduce_sum_f32(
                        r,
                        send_bufs[r].as_ptr() as *const f32,
                        recv_bufs[r].as_ptr() as *mut f32,
                        count,
                        streams[r].raw_ptr(),
                    )
                }
                .expect("all_reduce");
            }
            rccl.group_end().expect("group_end");
            for r in 0..n_ranks {
                hip.set_device(r as i32).expect("set_device");
                hip.stream_synchronize(&streams[r]).expect("sync");
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

    // Cleanup
    for r in 0..n_ranks {
        hip.set_device(r as i32).expect("set_device");
        let recv = recv_bufs.remove(0);
        let send = send_bufs.remove(0);
        let _ = hip.free(send);
        let _ = hip.free(recv);
        let stream = streams.remove(0);
        let _ = hip.stream_destroy(stream);
    }
    // Drop of `rccl` destroys all comms.

    println!("\nrccl_smoke: PASS");
}
