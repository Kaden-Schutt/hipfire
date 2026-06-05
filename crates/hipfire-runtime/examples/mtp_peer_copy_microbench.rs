// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP cross-device cost microbench (gfx906 ↔ gfx1031).
//!
//! Models the per-cycle hidden+token shuttle a multi-GPU MTP layout would do:
//! trunk lives on gfx906 (device 0), MTP head lives on gfx1031 (device 1).
//!
//! Per cycle:
//!  1. gfx906 records "verify_done" event (notional — we just record a noop).
//!  2. gfx1031 scatter stream waits on verify_done event.
//!  3. peer_copy_async  prev_hidden  906 → 1031   (20 KB, n_embd=5120 × f32)
//!  4. gfx1031 stream sync         (proxy for "MTP head can start now")
//!  5. gfx1031 records "mtp_done" event (notional — no MTP fwd in this bench)
//!  6. gfx906 stream waits on mtp_done event
//!  7. peer_copy_async  candidates  1031 → 906   (16 B, max_n=4 × u32)
//!  8. gfx906 stream sync          ("trunk can resume now")
//!
//! Decision gate: median cycle wall in microseconds.
//!  - ≤ 500 µs → sync split is genuinely viable on this PCIe / ROCm combo
//!  - 500 µs – 2 ms → sync ROI of 12% is in jeopardy; async-only path
//!  - > 2 ms → cross-device sync dominates; shelve multi-GPU MTP
//!
//! Reference: docs/plans/mtp_multi_gpu_glm5.md anatomy (MTP head ~9.5 ms of
//! ~80 ms cycle = 12%). Anything above ~1 ms of pure sync overhead eats
//! most of that headroom.
//!
//! Run: source scripts/gpu-lock.sh && gpu_acquire mtp-peer-bench && \
//!      ./target/release/examples/mtp_peer_copy_microbench && gpu_release

use hip_bridge::HipRuntime;
use std::time::Instant;

const N_EMBD: usize = 5120;
const HIDDEN_BYTES: usize = N_EMBD * 4; // f32
const MAX_N: usize = 4;
const TOKEN_BYTES: usize = MAX_N * 4; // u32

const WARMUP_ITERS: usize = 50;
const MEASURE_ITERS: usize = 1000;

fn main() {
    let hip = HipRuntime::load().expect("hip load");
    assert!(hip.device_count().unwrap() >= 2, "need ≥2 devices");

    for id in 0..2 {
        hip.set_device(id).unwrap();
        let arch = hip.get_arch(id).unwrap_or_else(|_| "?".into());
        println!("dev {id}: {arch}");
    }

    // Bidirectional peer.
    hip.set_device(0).unwrap();
    hip.enable_peer_access(1).unwrap();
    hip.set_device(1).unwrap();
    hip.enable_peer_access(0).unwrap();

    // Buffers.
    hip.set_device(0).unwrap();
    let trunk_hidden = hip.malloc(HIDDEN_BYTES).unwrap();          // 906 src
    let trunk_tokens_in = hip.malloc(TOKEN_BYTES).unwrap();        // 906 dst
    let trunk_stream = hip.stream_create().unwrap();

    hip.set_device(1).unwrap();
    let drafter_hidden_in = hip.malloc(HIDDEN_BYTES).unwrap();     // 1031 dst
    let drafter_tokens_out = hip.malloc(TOKEN_BYTES).unwrap();     // 1031 src
    let drafter_scatter_stream = hip.stream_create().unwrap();
    let drafter_main_stream = hip.stream_create().unwrap();

    // Events.
    hip.set_device(0).unwrap();
    let verify_done_evt = hip.event_create().unwrap();
    hip.set_device(1).unwrap();
    let mtp_done_evt = hip.event_create().unwrap();

    // Seed hidden so peer copy isn't trivially zero-elided by any optimization.
    let seed: Vec<u8> = (0..HIDDEN_BYTES).map(|i| (i & 0xff) as u8).collect();
    hip.set_device(0).unwrap();
    hip.memcpy_htod(&trunk_hidden, &seed).unwrap();

    println!(
        "\nMTP cross-device microbench: hidden={} B, tokens={} B, iters={} (after {} warmup)",
        HIDDEN_BYTES, TOKEN_BYTES, MEASURE_ITERS, WARMUP_ITERS
    );

    // ── Warmup ──
    for _ in 0..WARMUP_ITERS {
        run_one_cycle(
            &hip, &trunk_stream, &drafter_scatter_stream, &drafter_main_stream,
            &verify_done_evt, &mtp_done_evt,
            &trunk_hidden, &drafter_hidden_in,
            &drafter_tokens_out, &trunk_tokens_in,
        );
    }

    // ── Measure: full cycle ──
    let mut full_us: Vec<u128> = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t = Instant::now();
        run_one_cycle(
            &hip, &trunk_stream, &drafter_scatter_stream, &drafter_main_stream,
            &verify_done_evt, &mtp_done_evt,
            &trunk_hidden, &drafter_hidden_in,
            &drafter_tokens_out, &trunk_tokens_in,
        );
        full_us.push(t.elapsed().as_micros());
    }

    // ── Sub-measurement A: forward leg only (record + wait + peer 20 KB + sync) ──
    let mut fwd_us: Vec<u128> = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t = Instant::now();
        hip.set_device(0).unwrap();
        hip.event_record(&verify_done_evt, Some(&trunk_stream)).unwrap();
        hip.set_device(1).unwrap();
        hip.stream_wait_event(&drafter_scatter_stream, &verify_done_evt).unwrap();
        hip.memcpy_peer_async(&drafter_hidden_in, 1, &trunk_hidden, 0,
                              HIDDEN_BYTES, &drafter_scatter_stream).unwrap();
        hip.stream_synchronize(&drafter_scatter_stream).unwrap();
        fwd_us.push(t.elapsed().as_micros());
    }

    // ── Sub-measurement B: just the peer copy on a single stream, no event ──
    let mut raw_us: Vec<u128> = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t = Instant::now();
        hip.set_device(1).unwrap();
        hip.memcpy_peer_async(&drafter_hidden_in, 1, &trunk_hidden, 0,
                              HIDDEN_BYTES, &drafter_scatter_stream).unwrap();
        hip.stream_synchronize(&drafter_scatter_stream).unwrap();
        raw_us.push(t.elapsed().as_micros());
    }

    print_stats("full cycle (8 ops, 2 peer copies, 2 events)", &full_us);
    print_stats("forward leg only (1 event + 1 peer 20KB + sync)", &fwd_us);
    print_stats("raw peer copy 20 KB 906→1031 (no event)", &raw_us);

    let median_full = median(&full_us);
    let median_fwd = median(&fwd_us);
    let median_raw = median(&raw_us);

    println!("\n── Decision gate (vs ~9.5 ms MTP head budget per cycle) ──");
    println!("full-cycle overhead per spec cycle: {} µs", median_full);
    let pct = (median_full as f64 / 9500.0) * 100.0;
    println!("  → {:.1}% of MTP head budget", pct);
    if median_full <= 500 {
        println!("VERDICT: ≤500µs — sync split is viable, build it.");
    } else if median_full <= 2000 {
        println!("VERDICT: 500µs–2ms — sync ROI is marginal; async-only path required.");
    } else {
        println!("VERDICT: >2ms — cross-device sync dominates; shelve multi-GPU MTP.");
    }

    println!("\nbreakdown:");
    println!("  raw peer 20KB:        {} µs", median_raw);
    println!("  fwd leg (+ event):    {} µs", median_fwd);
    println!("  full cycle:           {} µs", median_full);
    println!("  back-leg+sync extra:  {} µs", median_full.saturating_sub(median_fwd));

    // Cleanup.
    hip.set_device(0).unwrap();
    hip.free(trunk_hidden).unwrap();
    hip.free(trunk_tokens_in).unwrap();
    hip.stream_destroy(trunk_stream).unwrap();
    hip.set_device(1).unwrap();
    hip.free(drafter_hidden_in).unwrap();
    hip.free(drafter_tokens_out).unwrap();
    hip.stream_destroy(drafter_scatter_stream).unwrap();
    hip.stream_destroy(drafter_main_stream).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn run_one_cycle(
    hip: &HipRuntime,
    trunk_stream: &hip_bridge::Stream,
    drafter_scatter_stream: &hip_bridge::Stream,
    drafter_main_stream: &hip_bridge::Stream,
    verify_done_evt: &hip_bridge::Event,
    mtp_done_evt: &hip_bridge::Event,
    trunk_hidden: &hip_bridge::DeviceBuffer,
    drafter_hidden_in: &hip_bridge::DeviceBuffer,
    drafter_tokens_out: &hip_bridge::DeviceBuffer,
    trunk_tokens_in: &hip_bridge::DeviceBuffer,
) {
    // 1. Trunk records verify-done on its stream (notional verify just finished).
    hip.set_device(0).unwrap();
    hip.event_record(verify_done_evt, Some(trunk_stream)).unwrap();

    // 2-3. Drafter scatter stream waits, then pulls hidden 906→1031.
    hip.set_device(1).unwrap();
    hip.stream_wait_event(drafter_scatter_stream, verify_done_evt).unwrap();
    hip.memcpy_peer_async(drafter_hidden_in, 1, trunk_hidden, 0,
                          HIDDEN_BYTES, drafter_scatter_stream).unwrap();

    // 4. Sync the scatter stream — MTP head would be free to start here.
    //    In a real impl, the main stream would wait_event on scatter_done,
    //    then run MTP. We sync to get an end-to-end "MTP can begin" cost.
    hip.stream_synchronize(drafter_scatter_stream).unwrap();

    // 5. Drafter records mtp-done on main stream (notional MTP just finished).
    hip.event_record(mtp_done_evt, Some(drafter_main_stream)).unwrap();

    // 6-7. Trunk waits, pulls candidate tokens 1031→906.
    hip.set_device(0).unwrap();
    hip.stream_wait_event(trunk_stream, mtp_done_evt).unwrap();
    hip.memcpy_peer_async(trunk_tokens_in, 0, drafter_tokens_out, 1,
                          TOKEN_BYTES, trunk_stream).unwrap();

    // 8. Trunk sync — verify can resume now.
    hip.stream_synchronize(trunk_stream).unwrap();
}

fn median(v: &[u128]) -> u128 {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

fn p99(v: &[u128]) -> u128 {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[(s.len() * 99) / 100]
}

fn print_stats(label: &str, v: &[u128]) {
    let med = median(v);
    let p9 = p99(v);
    let min = *v.iter().min().unwrap();
    let max = *v.iter().max().unwrap();
    println!(
        "  {:<48} median={:>5} µs   p99={:>5} µs   range=[{:>4}..{:>5}]",
        label, med, p9, min, max
    );
}
