// SPDX-License-Identifier: MIT
// hipfire — TP comm microbench. Stage 2 of docs/plans/multi-gpu-tp-a3b.md
// pulled forward as a standalone bench to validate or kill the TP plan
// before architectural work begins.
//
// Measures on hiptrx-class hardware (4× R9700 / gfx1201):
//   A) Raw memcpy_peer_async bandwidth + latency per src→dst pair.
//   B) Gpus::boundary_copy + wait_boundary overhead (production path).
//   C) 4-rank ring all-reduce copy schedule (6 copies of N/4 each).
//   D) 4-rank all-to-all (every rank sends to every other rank).
//   E) Synthesis: predicted per-token MoE-layer comm cost at TP=4.
//
// Run: HIP_VISIBLE_DEVICES=0,1,2,3 \
//        cargo run -p hipfire-runtime --release \
//        --example tp_comm_smoke
//
// Env knobs:
//   HIPFIRE_TP_BENCH_N=4         — number of ranks (default 4, can be 2)
//   HIPFIRE_TP_BENCH_ITERS=100   — timed iterations per cell
//   HIPFIRE_TP_BENCH_WARMUP=10   — warmup iterations (untimed)

use hip_bridge::DeviceBuffer;
use hipfire_runtime::multi_gpu::Gpus;
use std::time::Instant;

const DEFAULT_RANKS: usize = 4;
const DEFAULT_ITERS: usize = 100;
const DEFAULT_WARMUP: usize = 10;

const SIZES_BYTES: &[usize] = &[
    4 * 1024,      // residual at hidden=2048, f16
    32 * 1024,     // 8 × residual (MoE top-k=8 activation payload)
    128 * 1024,    // batched residual at batch=32
    512 * 1024,    // prefill residual at batch=128
];

struct Stats {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

fn stats_from(mut samples: Vec<f64>) -> Stats {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    Stats {
        p10_us: samples[n / 10],
        median_us: samples[n / 2],
        p90_us: samples[(n * 9) / 10],
    }
}

fn fmt_bw(bytes: usize, us: f64) -> String {
    if us <= 0.0 {
        return "—".to_string();
    }
    let gb_per_s = (bytes as f64 / 1e9) / (us / 1e6);
    format!("{gb_per_s:6.2} GB/s")
}

fn read_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n_ranks = read_env("HIPFIRE_TP_BENCH_N", DEFAULT_RANKS);
    let iters = read_env("HIPFIRE_TP_BENCH_ITERS", DEFAULT_ITERS);
    let warmup = read_env("HIPFIRE_TP_BENCH_WARMUP", DEFAULT_WARMUP);

    println!("=== TP comm microbench ===");
    println!("ranks: {n_ranks}, iters: {iters}, warmup: {warmup}");

    // n_layers placeholder — Gpus::init_uniform requires n_layers >= n_devices;
    // we set it equal so the layer-band machinery degenerates to one layer/rank.
    let mut gpus = Gpus::init_uniform(n_ranks, n_ranks).expect("init_uniform");
    println!();
    print_device_info(&gpus);

    let peer_ok = gpus.enable_peer_all().expect("enable_peer_all");
    print_peer_matrix(&gpus);
    if !peer_ok {
        eprintln!("WARN: peer access incomplete — boundary_copy falls back to host-staging on some pairs.");
    }
    if n_ranks < 2 {
        eprintln!("ERROR: comm bench requires ≥2 ranks (got {n_ranks}). Set HIP_VISIBLE_DEVICES to expose more devices.");
        std::process::exit(1);
    }

    setup_streams(&mut gpus);

    println!("\n=== Phase A: raw memcpy_peer_async + stream_synchronize ===");
    println!("(pair-by-pair, sequential — measures HW link + sync overhead)");
    bench_phase_a(&gpus, iters, warmup);

    println!("\n=== Phase B: Gpus::boundary_copy + wait_boundary ===");
    println!("(production orchestrator path — overhead vs raw)");
    bench_phase_b(&gpus, iters, warmup);

    println!("\n=== Phase C: 4-rank ring all-reduce copy schedule ===");
    println!("(reduce-scatter 3 steps + all-gather 3 steps; per-chunk = N/{n_ranks})");
    let c_results = bench_phase_c(&gpus, iters, warmup);

    println!("\n=== Phase D: full all-to-all (every rank → every rank) ===");
    println!("(measures MoE EP scatter+gather worst case at TP={n_ranks})");
    let d_results = bench_phase_d(&gpus, iters, warmup);

    println!("\n=== Phase E: synthesis vs §3.3 estimates ===");
    print_synthesis(&c_results, &d_results);

    teardown_streams(&mut gpus);
    println!("\ntp_comm_smoke: PASS");
}

fn print_device_info(gpus: &Gpus) {
    println!("devices:");
    for (i, dev) in gpus.devices.iter().enumerate() {
        let (free, total) = dev.hip.get_vram_info().unwrap_or((0, 0));
        println!(
            "  rank {i}: device_id={} arch={} vram={:.1}/{:.1} GiB",
            dev.device_id,
            dev.arch,
            free as f64 / 1e9,
            total as f64 / 1e9,
        );
    }
}

fn print_peer_matrix(gpus: &Gpus) {
    let n = gpus.devices.len();
    println!("\npeer matrix (can_access_peer):");
    print!("       ");
    for j in 0..n {
        print!("  -> {j} ");
    }
    println!();
    for i in 0..n {
        gpus.devices[i].bind_thread().expect("bind");
        print!("from {i}:");
        for j in 0..n {
            if i == j {
                print!("    .  ");
                continue;
            }
            let can = gpus.devices[i]
                .hip
                .can_access_peer(gpus.devices[i].device_id, gpus.devices[j].device_id)
                .unwrap_or(false);
            print!("   {}  ", if can { "Y" } else { "N" });
        }
        println!();
    }
    println!("peer_access_enabled (global flag): {}", gpus.peer_access_enabled);
}

fn setup_streams(gpus: &mut Gpus) {
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        // active_stream gives boundary_copy the async path; without it,
        // boundary_copy falls through to the sync memcpy_peer (no event).
        dev.active_stream = Some(s);
    }
}

fn teardown_streams(gpus: &mut Gpus) {
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        if let Some(s) = dev.active_stream.take() {
            let _ = dev.hip.stream_destroy(s);
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Phase A: raw memcpy_peer_async — one src→dst direction at a time
// ───────────────────────────────────────────────────────────────────

fn bench_phase_a(gpus: &Gpus, iters: usize, warmup: usize) {
    let n = gpus.devices.len();
    // Allocate one buffer per rank at the max size; reuse for each size cell.
    let max_size = *SIZES_BYTES.iter().max().unwrap();
    let buffers: Vec<DeviceBuffer> = (0..n)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            gpus.devices[i].hip.malloc(max_size).expect("malloc")
        })
        .collect();

    // Header
    print!("                ");
    for &size in SIZES_BYTES {
        print!("  {:>5} KB        ", size / 1024);
    }
    println!();

    for src in 0..n {
        for dst in 0..n {
            if src == dst {
                continue;
            }
            print!("{src} -> {dst}:        ");
            for &size in SIZES_BYTES {
                let us = bench_peer_copy(gpus, &buffers[src], &buffers[dst], src, dst, size, iters, warmup);
                print!("  {:6.1} µs ({})", us.median_us, fmt_bw(size, us.median_us));
            }
            println!();
        }
    }

    // Cleanup
    for (i, buf) in buffers.into_iter().enumerate() {
        gpus.devices[i].bind_thread().expect("bind");
        let _ = gpus.devices[i].hip.free(buf);
    }
}

fn bench_peer_copy(
    gpus: &Gpus,
    src_buf: &DeviceBuffer,
    dst_buf: &DeviceBuffer,
    src: usize,
    dst: usize,
    n_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Stats {
    let src_gpu = &gpus.devices[src];
    src_gpu.bind_thread().expect("bind");
    let stream = src_gpu.active_stream.as_ref().expect("stream set");
    let src_dev_id = src_gpu.device_id;
    let dst_dev_id = gpus.devices[dst].device_id;

    for _ in 0..warmup {
        src_gpu
            .hip
            .memcpy_peer_async(dst_buf, dst_dev_id, src_buf, src_dev_id, n_bytes, stream)
            .expect("memcpy_peer_async warm");
    }
    src_gpu.hip.stream_synchronize(stream).expect("sync");

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        src_gpu
            .hip
            .memcpy_peer_async(dst_buf, dst_dev_id, src_buf, src_dev_id, n_bytes, stream)
            .expect("memcpy_peer_async");
        src_gpu.hip.stream_synchronize(stream).expect("sync");
        samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    stats_from(samples)
}

// ───────────────────────────────────────────────────────────────────
// Phase B: Gpus::boundary_copy + wait_boundary
// ───────────────────────────────────────────────────────────────────

fn bench_phase_b(gpus: &Gpus, iters: usize, warmup: usize) {
    let n = gpus.devices.len();
    let max_size = *SIZES_BYTES.iter().max().unwrap();
    let buffers: Vec<DeviceBuffer> = (0..n)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            gpus.devices[i].hip.malloc(max_size).expect("malloc")
        })
        .collect();

    print!("                ");
    for &size in SIZES_BYTES {
        print!("  {:>5} KB        ", size / 1024);
    }
    println!();

    for src in 0..n {
        for dst in 0..n {
            if src == dst {
                continue;
            }
            print!("{src} -> {dst}:        ");
            for &size in SIZES_BYTES {
                let us = bench_boundary_copy(gpus, &buffers[src], &buffers[dst], src, dst, size, iters, warmup);
                print!("  {:6.1} µs ({})", us.median_us, fmt_bw(size, us.median_us));
            }
            println!();
        }
    }

    for (i, buf) in buffers.into_iter().enumerate() {
        gpus.devices[i].bind_thread().expect("bind");
        let _ = gpus.devices[i].hip.free(buf);
    }
}

fn bench_boundary_copy(
    gpus: &Gpus,
    src_buf: &DeviceBuffer,
    dst_buf: &DeviceBuffer,
    src: usize,
    dst: usize,
    n_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Stats {
    for _ in 0..warmup {
        let evt = gpus
            .boundary_copy(src, dst, src_buf, dst_buf, n_bytes)
            .expect("boundary_copy warm");
        gpus.wait_boundary(evt).expect("wait warm");
    }
    // wait_boundary is async (stream_wait_event); force completion before
    // taking the first measurement.
    sync_all_streams(gpus);

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let evt = gpus
            .boundary_copy(src, dst, src_buf, dst_buf, n_bytes)
            .expect("boundary_copy");
        gpus.wait_boundary(evt).expect("wait");
        // Force the actual copy to complete (wait_boundary only inserts a
        // stream-level dep; without a follow-up dispatch the dst stream
        // never blocks).
        sync_all_streams(gpus);
        samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    stats_from(samples)
}

fn sync_all_streams(gpus: &Gpus) {
    for dev in &gpus.devices {
        dev.bind_thread().expect("bind");
        if let Some(stream) = dev.active_stream.as_ref() {
            dev.hip.stream_synchronize(stream).expect("sync");
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Phase C: ring all-reduce copy schedule (no GPU reduction kernel —
// pure comm cost; add cost is sub-µs at these sizes on R9700.)
//
// For N ranks: 2*(N-1) total copies, each of size N_bytes / N.
// Reduce-scatter (N-1 steps): rank r sends chunk[r-step] to (r+1)%N.
// All-gather    (N-1 steps): rank r sends chunk[r] to (r+1)%N.
// We measure end-to-end wall time including all stream syncs.
// ───────────────────────────────────────────────────────────────────

struct PhaseResults {
    median_us: Vec<f64>,
}

fn bench_phase_c(gpus: &Gpus, iters: usize, warmup: usize) -> PhaseResults {
    let n = gpus.devices.len();
    let max_size = *SIZES_BYTES.iter().max().unwrap();
    // Each rank owns two buffers: send_buf (its chunk) and recv_buf (incoming).
    let send_bufs: Vec<DeviceBuffer> = (0..n)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            gpus.devices[i].hip.malloc(max_size).expect("malloc send")
        })
        .collect();
    let recv_bufs: Vec<DeviceBuffer> = (0..n)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            gpus.devices[i].hip.malloc(max_size).expect("malloc recv")
        })
        .collect();

    let mut results = PhaseResults {
        median_us: Vec::with_capacity(SIZES_BYTES.len()),
    };

    println!(
        "{:<14}  {:>10}  {:>10}  {:>10}  {:>10}",
        "size", "median µs", "p10 µs", "p90 µs", "per-step µs"
    );
    for &total_bytes in SIZES_BYTES {
        let chunk_bytes = total_bytes / n;
        for _ in 0..warmup {
            ring_allreduce_once(gpus, &send_bufs, &recv_bufs, chunk_bytes);
        }
        sync_all_streams(gpus);

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            ring_allreduce_once(gpus, &send_bufs, &recv_bufs, chunk_bytes);
            sync_all_streams(gpus);
            samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        let stats = stats_from(samples);
        let n_steps = 2 * (n - 1);
        results.median_us.push(stats.median_us);
        println!(
            "{:<14}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}",
            format!("{} KB", total_bytes / 1024),
            stats.median_us,
            stats.p10_us,
            stats.p90_us,
            stats.median_us / n_steps as f64,
        );
    }

    for (i, buf) in send_bufs.into_iter().enumerate() {
        gpus.devices[i].bind_thread().expect("bind");
        let _ = gpus.devices[i].hip.free(buf);
    }
    for (i, buf) in recv_bufs.into_iter().enumerate() {
        gpus.devices[i].bind_thread().expect("bind");
        let _ = gpus.devices[i].hip.free(buf);
    }
    results
}

fn ring_allreduce_once(
    gpus: &Gpus,
    send_bufs: &[DeviceBuffer],
    recv_bufs: &[DeviceBuffer],
    chunk_bytes: usize,
) {
    let n = gpus.devices.len();
    // Reduce-scatter + all-gather = 2*(n-1) sequential steps.
    // We model the comm cost only — chunk merge would be a GPU add on the
    // receiver, ~sub-µs at these sizes, deliberately omitted.
    for step in 0..(2 * (n - 1)) {
        for src in 0..n {
            let dst = (src + 1) % n;
            // Alternate src/dst buffer to mimic the in-place ring pattern
            // without actually computing the reduction.
            let (sb, db) = if step % 2 == 0 {
                (&send_bufs[src], &recv_bufs[dst])
            } else {
                (&recv_bufs[src], &send_bufs[dst])
            };
            let evt = gpus
                .boundary_copy(src, dst, sb, db, chunk_bytes)
                .expect("ring boundary_copy");
            gpus.wait_boundary(evt).expect("ring wait");
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Phase D: full all-to-all (every rank → every other rank)
// Models the MoE EP worst-case where every rank's tokens fan out to
// experts on every other rank.
// ───────────────────────────────────────────────────────────────────

fn bench_phase_d(gpus: &Gpus, iters: usize, warmup: usize) -> PhaseResults {
    let n = gpus.devices.len();
    let max_size = *SIZES_BYTES.iter().max().unwrap();
    // Each rank holds n-1 outgoing buffers (one per dst) and n-1 incoming.
    let send_bufs: Vec<Vec<DeviceBuffer>> = (0..n)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            (0..n)
                .map(|_| gpus.devices[i].hip.malloc(max_size).expect("malloc"))
                .collect()
        })
        .collect();
    let recv_bufs: Vec<Vec<DeviceBuffer>> = (0..n)
        .map(|i| {
            gpus.devices[i].bind_thread().expect("bind");
            (0..n)
                .map(|_| gpus.devices[i].hip.malloc(max_size).expect("malloc"))
                .collect()
        })
        .collect();

    let mut results = PhaseResults {
        median_us: Vec::with_capacity(SIZES_BYTES.len()),
    };

    println!(
        "{:<14}  {:>10}  {:>10}  {:>10}  {:>10}",
        "per-msg size", "median µs", "p10 µs", "p90 µs", "per-rank µs"
    );
    for &per_msg in SIZES_BYTES {
        for _ in 0..warmup {
            alltoall_once(gpus, &send_bufs, &recv_bufs, per_msg);
        }
        sync_all_streams(gpus);

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            alltoall_once(gpus, &send_bufs, &recv_bufs, per_msg);
            sync_all_streams(gpus);
            samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        let stats = stats_from(samples);
        results.median_us.push(stats.median_us);
        println!(
            "{:<14}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}",
            format!("{} KB", per_msg / 1024),
            stats.median_us,
            stats.p10_us,
            stats.p90_us,
            stats.median_us / n as f64,
        );
    }

    for outer in send_bufs.into_iter().enumerate() {
        let (i, bufs) = outer;
        gpus.devices[i].bind_thread().expect("bind");
        for b in bufs {
            let _ = gpus.devices[i].hip.free(b);
        }
    }
    for outer in recv_bufs.into_iter().enumerate() {
        let (i, bufs) = outer;
        gpus.devices[i].bind_thread().expect("bind");
        for b in bufs {
            let _ = gpus.devices[i].hip.free(b);
        }
    }
    results
}

fn alltoall_once(
    gpus: &Gpus,
    send_bufs: &[Vec<DeviceBuffer>],
    recv_bufs: &[Vec<DeviceBuffer>],
    per_msg: usize,
) {
    let n = gpus.devices.len();
    // Sequential schedule: for each (src, dst) pair, issue boundary_copy +
    // wait_boundary. This is the conservative measurement — a real impl
    // would parallelize across streams and possibly fuse. Each rank
    // sends (n-1) messages of size per_msg, so total bytes/rank = (n-1)*per_msg.
    for src in 0..n {
        for dst in 0..n {
            if src == dst {
                continue;
            }
            let evt = gpus
                .boundary_copy(src, dst, &send_bufs[src][dst], &recv_bufs[dst][src], per_msg)
                .expect("a2a boundary_copy");
            gpus.wait_boundary(evt).expect("a2a wait");
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Phase E: synthesis — translate measurements into per-token cost
// ───────────────────────────────────────────────────────────────────

fn print_synthesis(c: &PhaseResults, d: &PhaseResults) {
    // §3.3 estimates (from docs/plans/multi-gpu-tp-a3b.md):
    //   attn all-reduce ~50 µs at 4 KB
    //   MoE all-to-all ~100 µs at 32 KB (top-k=8 fan-out)
    //   ⇒ per-MoE-layer comm ~200 µs (incl. shared-expert all-reduce)
    //   ⇒ per-token comm at 64 layers ~13 ms
    let a3b_layers = 64;
    let attn_size_kb = 4;
    let moe_size_kb = 32;

    // Find indices for 4 KB (attn) and 32 KB (MoE) in our size arrays.
    let i_attn = SIZES_BYTES.iter().position(|&s| s == attn_size_kb * 1024).unwrap_or(0);
    let i_moe = SIZES_BYTES.iter().position(|&s| s == moe_size_kb * 1024).unwrap_or(1);

    let attn_us = c.median_us[i_attn];
    let moe_us = d.median_us[i_moe];
    // shared-expert all-reduce: same shape as attn all-reduce
    let shared_us = attn_us;
    let per_layer_us = attn_us + moe_us + shared_us;
    let per_token_us = per_layer_us * a3b_layers as f64;
    let per_token_ms = per_token_us / 1000.0;

    // Baseline: single-card A3B decode ~7 ms/token (142 tok/s, docs/multi-gpu.md)
    let single_card_ms = 7.0;
    let single_card_tokps = 1000.0 / single_card_ms;

    println!("predicted per-token MoE-layer comm cost:");
    println!("  attn all-reduce ({attn_size_kb} KB):  {:>7.1} µs (estimate: ~50 µs)", attn_us);
    println!("  MoE all-to-all ({moe_size_kb} KB):    {:>7.1} µs (estimate: ~100 µs)", moe_us);
    println!("  shared-expert all-reduce:    {:>7.1} µs", shared_us);
    println!("  per-MoE-layer comm:          {:>7.1} µs (estimate: ~200 µs)", per_layer_us);
    println!("  per-token comm @ {a3b_layers} layers:    {:>7.2} ms (estimate: ~13 ms)", per_token_ms);
    println!();
    println!("single-card A3B decode baseline:  ~{:.1} ms/token ({:.0} tok/s)", single_card_ms, single_card_tokps);
    println!(
        "predicted TP=4 decode lower bound: ~{:.1} ms/token (comm-only, ignores compute)",
        per_token_ms
    );

    let comm_compute_ratio = per_token_ms / single_card_ms;
    println!("\ncomm-to-compute ratio: {:.2}×", comm_compute_ratio);
    if comm_compute_ratio > 1.5 {
        println!("=> SHIP-GATE WARN: TP=4 decode is comm-bound. Re-evaluate DP=4 or TP=2.");
        println!("   Reference docs/plans/multi-gpu-tp-a3b.md §8.1 and §8.2 before proceeding to Stage 1.");
    } else if comm_compute_ratio > 1.0 {
        println!("=> CAUTIONARY: comm > compute. TP=4 only justified for context-extension / DSv4.");
    } else {
        println!("=> OK: comm < compute. TP=4 decode is compute-bound; Stage 1 has justified cost.");
    }
}

