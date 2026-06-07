// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! PP cross-device overlap microbench (gfx906 ‖ gfx1031) — Phase B.2a gate.
//!
//! The Stage 2b PpMtp perf gap is structural: the PP boundary in
//! `forward_prefill_batch_multi_with_caps` is FULLY SERIALIZED — band 0
//! (gfx906) runs to completion, blocking peer copy, band 1 (gfx1031)
//! runs. One card idles while the other works. The only lever that can
//! beat single-GPU decode is to PIPELINE this boundary across decode
//! steps (overlap step N+1's gfx906 work with step N's gfx1031 work).
//!
//! That pipelining is ~330-520 LOC of risky stream/event plumbing. Before
//! committing to it, this microbench answers the PHYSICS question it all
//! rests on: **on THIS gfx906↔gfx1031 pair, can a compute kernel on
//! gfx906 and a compute kernel on gfx1031 actually run CONCURRENTLY when
//! issued on independent per-device streams — or does something (the
//! ROCm runtime, host serialization, PCIe) force them to serialize?**
//!
//! If two physically-separate GPUs cannot overlap at all here, no amount
//! of production wiring helps and B.2b is dead on arrival.
//!
//! ## Method
//!
//! Busy kernel: `gemm_hfq4g256_residual` (M=K=5120, the qwen3.6-27b hidden
//! size), the same kernel `bench_stream_overlap.rs` uses. Per-band work is
//! scaled by GEMM-repeat count to mirror the real PP layer split: gfx906
//! carries the larger band (`--gfx906-iters`, default 48) and gfx1031 the
//! smaller (`--gfx1031-iters`, default 16) — same 48,16 ratio as the
//! daemon's default trunk split.
//!
//! Three timing modes, each timed wall-clock across both devices:
//!   (A) SEQUENTIAL: gfx906 band runs to completion + device_sync, THEN
//!       gfx1031 band runs + device_sync. This is today's serialized
//!       boundary shape (no overlap by construction).
//!   (B) CONCURRENT: both bands issued on their own device streams, then
//!       a single barrier (device_sync both). If the runtime/hardware
//!       allows it, the two cards run at the same time.
//!
//! overlap_ratio = T_sequential / T_concurrent.
//!   ≈ 1.0  → NO cross-device overlap. Pipelining premise is FALSE on
//!            this pair. ABORT B.2b.
//!   ≥ 1.5  → strong overlap. The serialized boundary is leaving real
//!            wall-clock on the table. PROCEED to B.2b.
//!   1.2–1.5 → partial overlap; marginal. Document, decide with user.
//!
//! Note: this measures the CEILING of what pipelining could recover (pure
//! concurrent compute, no peer-copy dependency, no spec accept-chain
//! stall). The boundary-copy cost itself is already measured separately
//! in `pp_boundary_microbench` (cleared its gate with 8× headroom). B.2b's
//! real gain will be below this ceiling; if the ceiling is ~1.0, B.2b
//! cannot win.
//!
//! Run:
//!   source scripts/gpu-lock.sh && gpu_acquire pp-overlap-bench && \
//!   env HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_PP_LAYERS=48,16 \
//!   ./target/release/examples/pp_overlap_microbench && gpu_release

use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const DIM: usize = 5120; // qwen3.6-27b hidden size (M = K)
const GROUP: usize = 256;
const ROW_HDR: usize = 8;
const ROW_PAYLOAD: usize = 128;
const ROW_BYTES: usize = ROW_HDR + ROW_PAYLOAD;
const SEED: u64 = 0xC0DEFACE;
const WARMUP: usize = 20;
const MEASURE: usize = 100;

/// HFQ4G256 synthetic weight rows — byte-identical format to
/// bench_stream_overlap.rs::synth_hfq4_weights.
fn synth_hfq4_weights(m: usize, groups_per_row: usize) -> Vec<u8> {
    let total = m * groups_per_row * ROW_BYTES;
    let mut out = vec![0u8; total];
    let mut state = SEED;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let scale = 1e-3_f32.to_le_bytes();
    let zp = (-0.5_f32).to_le_bytes();
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * ROW_BYTES;
            out[gp..gp + 4].copy_from_slice(&scale);
            out[gp + 4..gp + 8].copy_from_slice(&zp);
            for i in 0..ROW_PAYLOAD {
                out[gp + ROW_HDR + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Per-band GEMM resources living on one device.
struct Band {
    a_raw: rdna_compute::GpuTensor,
    x: rdna_compute::GpuTensor,
    y: rdna_compute::GpuTensor,
    iters: usize,
    batch: usize,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let per_device: Vec<usize> = std::env::var("HIPFIRE_PP_LAYERS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().parse::<usize>().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![48, 16]);
    assert_eq!(per_device.len(), 2, "PP=2 only");

    // Per-band GEMM-repeat counts default to the layer split so the busy
    // work mirrors the real per-band compute volume.
    let iters0 = env_usize("BENCH_GFX906_ITERS", per_device[0]);
    let iters1 = env_usize("BENCH_GFX1031_ITERS", per_device[1]);
    let batch = env_usize("BENCH_BATCH", 16); // verify-batch-ish N

    println!("PP layout: {per_device:?}");
    println!("band iters: dev0={iters0} dev1={iters1}  GEMM M=K={DIM} N={batch}");

    let mut gpus = Gpus::init_layers(&per_device).expect("init_layers");
    println!(
        "devices: dev0={} ({})  dev1={} ({})",
        gpus.devices[0].arch,
        gpus.devices[0].device_id,
        gpus.devices[1].arch,
        gpus.devices[1].device_id,
    );

    // Per-device streams. bind_thread BEFORE stream_create (device-affine).
    for d in 0..2 {
        gpus.devices[d].bind_thread().unwrap();
        let s = gpus.devices[d].hip.stream_create().unwrap();
        gpus.devices[d].active_stream = Some(s);
    }

    let gpr = DIM / GROUP;
    let weight_bytes = synth_hfq4_weights(DIM, gpr);
    let x_host: Vec<f32> = (0..batch * DIM)
        .map(|i| ((i as f32) * 1e-4) % 1.0 - 0.5)
        .collect();
    let y_init: Vec<f32> = (0..batch * DIM)
        .map(|i| ((i as f32) * 7e-5) % 0.5 - 0.25)
        .collect();

    let mut bands: Vec<Band> = Vec::with_capacity(2);
    let iters_arr = [iters0, iters1];
    for d in 0..2 {
        let g = &mut gpus.devices[d];
        g.bind_thread().unwrap();
        let a_raw = g
            .upload_raw(&weight_bytes, &[DIM * gpr * ROW_BYTES])
            .expect("a_raw");
        let x = g.upload_f32(&x_host, &[batch * DIM]).expect("x");
        let y = g.alloc_tensor(&[batch * DIM], DType::F32).expect("y");
        g.hip.memcpy_htod(&y.buf, bytes_of(&y_init)).unwrap();
        g.hip.device_synchronize().unwrap();
        bands.push(Band {
            a_raw,
            x,
            y,
            iters: iters_arr[d],
            batch,
        });
    }

    // Run one band's GEMM chain on its device's active_stream (issue only,
    // no sync — caller controls the barrier).
    let issue_band = |gpus: &mut Gpus, d: usize, band: &Band| {
        let g = &mut gpus.devices[d];
        g.bind_thread().unwrap();
        for _ in 0..band.iters {
            g.gemm_hfq4g256_residual(&band.a_raw, &band.x, &band.y, DIM, DIM, band.batch)
                .unwrap();
        }
    };
    let sync_dev = |gpus: &mut Gpus, d: usize| {
        let g = &mut gpus.devices[d];
        g.bind_thread().unwrap();
        g.hip.device_synchronize().unwrap();
    };

    // ── Warmup both modes ──
    for _ in 0..WARMUP {
        issue_band(&mut gpus, 0, unsafe { &*(&bands[0] as *const Band) });
        sync_dev(&mut gpus, 0);
        issue_band(&mut gpus, 1, unsafe { &*(&bands[1] as *const Band) });
        sync_dev(&mut gpus, 1);
    }

    // ── Mode A: SEQUENTIAL (today's serialized boundary shape) ──
    let mut seq: Vec<f64> = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let t = Instant::now();
        issue_band(&mut gpus, 0, unsafe { &*(&bands[0] as *const Band) });
        sync_dev(&mut gpus, 0); // gfx906 must finish before gfx1031 starts
        issue_band(&mut gpus, 1, unsafe { &*(&bands[1] as *const Band) });
        sync_dev(&mut gpus, 1);
        seq.push(t.elapsed().as_secs_f64() * 1e6);
    }

    // ── Mode B: CONCURRENT (both bands issued, single barrier) ──
    let mut conc: Vec<f64> = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let t = Instant::now();
        // Issue both bands back-to-back on their own device streams WITHOUT
        // an intervening sync, then barrier both. If the cards can overlap,
        // wall < sum.
        issue_band(&mut gpus, 0, unsafe { &*(&bands[0] as *const Band) });
        issue_band(&mut gpus, 1, unsafe { &*(&bands[1] as *const Band) });
        sync_dev(&mut gpus, 0);
        sync_dev(&mut gpus, 1);
        conc.push(t.elapsed().as_secs_f64() * 1e6);
    }

    // ── Per-band alone (for context: which card dominates) ──
    let bench_alone = |gpus: &mut Gpus, bands: &[Band], d: usize| -> f64 {
        let mut s: Vec<f64> = Vec::with_capacity(MEASURE);
        for _ in 0..MEASURE {
            let t = Instant::now();
            issue_band(gpus, d, unsafe { &*(&bands[d] as *const Band) });
            sync_dev(gpus, d);
            s.push(t.elapsed().as_secs_f64() * 1e6);
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let t_dev0_alone = bench_alone(&mut gpus, &bands, 0);
    let t_dev1_alone = bench_alone(&mut gpus, &bands, 1);

    seq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    conc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: &[f64]| v[v.len() / 2];
    let t_seq = med(&seq);
    let t_conc = med(&conc);
    let ratio = t_seq / t_conc;

    println!("\n── Per-band alone (median µs) ──");
    println!("  dev0 (gfx906, {iters0} iters): {t_dev0_alone:9.1}");
    println!("  dev1 (gfx1031, {iters1} iters): {t_dev1_alone:9.1}");
    println!(
        "  sum of alones:                 {:9.1}",
        t_dev0_alone + t_dev1_alone
    );

    println!("\n── Overlap modes (median µs over {MEASURE} iters) ──");
    println!("  A sequential (serialized):  {t_seq:9.1}");
    println!("  B concurrent (dual stream): {t_conc:9.1}");
    println!("  overlap_ratio = A/B =        {ratio:8.3}x");

    println!("\n── Phase B.2a decision gate ──");
    if ratio >= 1.5 {
        println!("VERDICT: ratio ≥ 1.5 — strong cross-device overlap. The serialized");
        println!("  boundary leaves real wall-clock on the table. PROCEED to B.2b.");
    } else if ratio >= 1.2 {
        println!("VERDICT: 1.2 ≤ ratio < 1.5 — partial overlap, marginal. Document and");
        println!("  decide with user before the ~330-520 LOC B.2b build.");
    } else {
        println!("VERDICT: ratio < 1.2 — NO meaningful cross-device overlap on this pair.");
        println!("  Pipelining premise is FALSE. ABORT B.2b; pursue Opt 3 (no-replay");
        println!("  rollback) or Opt 4 (ship long-ctx) instead.");
    }

    // Cleanup streams (tensors freed on drop).
    for d in 0..2 {
        if let Some(s) = gpus.devices[d].active_stream.take() {
            gpus.devices[d].bind_thread().unwrap();
            let _ = gpus.devices[d].hip.stream_destroy(s);
        }
    }
    let _ = Gpu::init; // silence unused if cfg drops something
}
