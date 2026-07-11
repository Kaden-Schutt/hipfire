// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-side MEASURED peak-memory-bandwidth microbench, extending
//! `docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md`
//! Task 4/9 (`ChipProfile::peak_bw_gbps`) alongside the sibling
//! `pointer_chase_latency` (DRAM latency) microbench.
//!
//! Every `peak_bw_gbps` currently committed under `tests/chip-profiles/*.json`
//! (e.g. `gfx1201.json`) is the THEORETICAL GDDR6-spec constant already used
//! by `bench_gemv_hfp4g32_bw.rs::PEAK_GBPS` (clock * bus-width arithmetic via
//! `profiler::GpuCapability::detect`), NOT an achieved-throughput measurement
//! — the module doc there says so explicitly. This probe closes that gap: a
//! simple `dst[i] = src[i]` device-to-device copy kernel (the standard
//! STREAM-Copy-style peak-BW pattern), run at high occupancy over a SWEEP of
//! buffer sizes, timed with the same `hipEvent` pattern as
//! `profile.rs::begin_timer`/`end_timer` and `pointer_chase_latency`.
//!
//! **Multi-tier sweep (2026-07-02 fix):** a single point-measurement at one
//! buffer size can't distinguish "genuinely DRAM-bound" from "still partly
//! cache-resident" (the same on-chip-cache-vs-DRAM distinction
//! `pointer_chase_latency` has to defeat for latency). This probe sweeps
//! buffer sizes from small (cache-resident) up through comfortably past any
//! current arch's on-chip cache, prints every sweep row, then hands the full
//! curve to [`rdna_compute::chip_profile::detect_bw_tiers`], which detects
//! the DRAM-bound large-size PLATEAU directly from the curve's own shape
//! (the top rows mutually agreeing within a few percent) rather than from
//! `profiler::GpuCapability`'s `l2_cache_mb`/`infinity_cache_mb` fields.
//!
//! That arch_spec-threshold approach was tried first and found WRONG for
//! unified-memory APUs: gfx1151 (Strix Halo) measured 1MiB=593, 4MiB=799,
//! 16MiB=952 (cache-resident), then 64/128/256/512 MiB=229/230/231/232
//! (the real DRAM-bound LPDDR5X plateau) GB/s. The old fold-over-
//! `l2_cache_mb+infinity_cache_mb`-threshold logic classified the 16 MiB row
//! as "past the cache footprint" and reported `peak_bw_gbps=952` — the
//! CACHE tier, mislabeled as DRAM peak (see `tests/chip-profiles/gfx1201.json`
//! provenance note "gfx1151=950 (SUSPECT...)"). `detect_bw_tiers` instead
//! reports BOTH tiers explicitly: `peak_bw_gbps` (DRAM plateau, the roofline
//! denominator) and `cache_bw_gbps` (the cache-tier ceiling), plus
//! `effective_cache_mib` — the empirical transition point, which corrects
//! (rather than trusts) the arch_spec cache-footprint constant. On a dGPU
//! whose whole swept range is already DRAM-bound (no distinct cache tier),
//! `cache_bw_gbps`/`effective_cache_mib` are `None` rather than fabricated.
//! On a curve that never converges to a plateau at the top of the sweep,
//! `detect_bw_tiers` fails loud instead of mis-reporting the top row.
//!
//! Per `docs/methodology/perf-benchmarking.md` and this session's warm-DPM
//! finding (`tests/chip-profiles/gfx1201.json` provenance / commits
//! 542af081, db5e1efd): a cold-DPM-asleep card understates real throughput
//! (mem_latency was 2.65x slower cold vs warm on gfx1201). Each sweep size
//! runs an UNTIMED warmup pass before the timed trials, and the whole sweep
//! should itself be preceded by a throwaway pass (or
//! `HIPFIRE_DPM_WARMUP_SECS=10`) before the number is trusted, and run at
//! `power_dpm_force_performance_level=high` on eGPU/APU cards that
//! autosuspend (BACO). The same session's eGPU-vs-native finding (hipx
//! TBT5-eGPU R9700 corroborated hiptrx native-PCIe R9700 within 0.5% on
//! `mem_latency_ns`) means this probe is expected to read the same on either
//! host for a given card — a large native-vs-eGPU delta here would itself be
//! a finding, not assumed away.
//!
//! GPU-required (no meaningful no-GPU mode — the whole point is a live
//! device-side measurement; matches `pointer_chase_latency`'s posture).
//! Usage (coordinate the GPU lock per `scripts/gpu-lock.sh` /
//! CLAUDE.md "GPU Lock Protocol" when other agents may be sharing the box):
//!   source scripts/gpu-lock.sh && gpu_acquire "peak_bw_probe" && \
//!     HIP_VISIBLE_DEVICES=0 cargo run --release -p rdna-compute --example peak_bw_probe ; \
//!     gpu_release

use rdna_compute::chip_profile::{detect_bw_tiers, BwSweepPoint};
use rdna_compute::profiler::GpuCapability;
use rdna_compute::{profile, Gpu, KernelCompiler};
use std::ffi::c_void;

/// Bytes per `float4` element moved by the copy kernel.
const ELEM_BYTES: usize = 16;

/// Sweep of buffer sizes (MiB) for ONE of the two DtoD copy buffers (src and
/// dst are each this size, so live VRAM usage per sweep step is 2x this).
/// Spans well below a typical RDNA on-chip-cache footprint (cache-resident,
/// feeds [`detect_bw_tiers`]'s cache-tier estimate) up through comfortably
/// past it (the DRAM-bound plateau `detect_bw_tiers` detects directly from
/// the curve's shape — see the module docs on the gfx1151 multi-tier
/// finding). Kept small enough (max 1024 MiB total across both buffers) to
/// run unmodified across the whole fleet (k9lin 24GB / hiptrx 32GB / hipx
/// 96GB+5700XT) without a VRAM budget arg.
///
/// The max entry MUST comfortably clear every arch's on-chip cache in
/// `profiler::arch_spec`, not just the ones in the current physical fleet —
/// RDNA2 (gfx1030-class, 4 MiB L2 + 128 MiB Infinity Cache) needs the sweep
/// to reach past ~256 MiB before the curve can plateau at DRAM speed; a
/// prior max-256 sweep never reached that margin. 512 clears that with
/// headroom; if a future arch adds a bigger on-chip cache, extend the sweep
/// (and expect `detect_bw_tiers` to fail loud — not silently mis-report —
/// if it doesn't).
const SWEEP_MIB: &[usize] = &[1, 4, 16, 64, 128, 256, 512];

/// [`detect_bw_tiers`]'s plateau-agreement tolerance: consecutive
/// large-size sweep rows within this many percent of each other are
/// considered the same DRAM-bound plateau. Empirically ~1.3% on the gfx1151
/// 64..512 MiB rows (229/230/231/232 GB/s) vs the >300% jump down from the
/// cache tier — 3.0 comfortably separates "plateau noise" from "still a
/// different tier" (see `chip_profile::detect_bw_tiers` module docs).
const PLATEAU_TOLERANCE_PCT: f64 = 3.0;

/// Untimed warmup launches per sweep size — primes DPM/clocks + page
/// mappings before the timed trials (same rationale as
/// `pointer_chase_latency::WARMUP_HOPS` and the mandatory kernel-cache/DPM
/// warmup in `docs/methodology/perf-benchmarking.md`).
const WARMUP_TRIALS: u32 = 5;

/// Timed launches per sweep size, all wrapped in a single `hipEvent`
/// start/stop pair (amortizes launch overhead the same way
/// `pointer_chase_latency` amortizes hop overhead over many hops).
const TIMED_TRIALS: u32 = 20;

/// One thread per `float4` element — maximizes in-flight requests so the
/// probe is bandwidth-bound, not latency- or occupancy-bound.
const BLOCK_THREADS: u32 = 256;

const KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>

// STREAM-Copy-style peak-bandwidth kernel: one thread per float4 element,
// straight-line dst[i] = src[i]. No branching, no re-reads, no reduction —
// every load/store is a distinct 16-byte transaction so achieved throughput
// tracks the DRAM/cache-hierarchy limit, not kernel-internal ALU/LDS cost.
extern "C" __global__ void bw_copy(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    unsigned long long n4) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        dst[i] = src[i];
    }
}
"#;

/// One sweep row's result.
struct SweepRow {
    mib: usize,
    gbps: f64,
}

fn main() {
    let gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("Gpu::init() failed: {e}. peak_bw_probe requires a live GPU.");
        std::process::exit(1);
    });
    println!("arch: {}", gpu.arch);

    let vram_bytes = gpu
        .hip
        .get_vram_info()
        .map(|(_, total)| total as u64)
        .unwrap_or(0);
    let theoretical = GpuCapability::detect(&gpu.arch, vram_bytes);
    let theoretical_peak_gbps = theoretical.peak_bw_gbs as f64;
    // NOTE: `theoretical.l2_cache_mb`/`infinity_cache_mb` (from
    // `profiler::arch_spec`) are printed below for reference only — they are
    // NOT used to decide the DRAM-vs-cache split (see module docs: they are
    // wrong for unified-memory APUs). The split is detected empirically from
    // the swept curve's own shape via `detect_bw_tiers`, below.
    println!("theoretical_peak_gbps (analytic, GpuCapability::detect): {theoretical_peak_gbps:.1}");
    println!(
        "arch_spec_l2_infinity_cache_mib (reference only, NOT used for tier detection): {:.1}+{:.1}",
        theoretical.l2_cache_mb, theoretical.infinity_cache_mb
    );
    println!(
        "note: every GB/s figure below counts total bytes MOVED per call — \
         the read of src[i] AND the write to dst[i] (bytes_per_buf * 2), \
         the standard STREAM-Copy convention"
    );

    // Compile once, reuse the module across every sweep size (only `n4`
    // varies per launch — same kernel, different grid + element count).
    let mut compiler = KernelCompiler::new(&gpu.arch, String::new()).unwrap_or_else(|e| {
        eprintln!("KernelCompiler::new failed: {e}");
        std::process::exit(1);
    });
    let obj_path = compiler
        .compile("peak_bw_probe", KERNEL_SRC)
        .unwrap_or_else(|e| {
            eprintln!("kernel compile failed: {e}");
            std::process::exit(1);
        });
    let obj_path_str = obj_path.to_str().unwrap().to_string();
    let module = gpu
        .hip
        .module_load(&obj_path_str)
        .expect("module_load failed");
    let func = gpu
        .hip
        .module_get_function(&module, "bw_copy")
        .expect("module_get_function(bw_copy) failed");

    let mut rows: Vec<SweepRow> = Vec::with_capacity(SWEEP_MIB.len());

    for &mib in SWEEP_MIB {
        let bytes_per_buf = mib * 1024 * 1024;
        let n4 = (bytes_per_buf / ELEM_BYTES) as u64;
        let grid_x = n4.div_ceil(BLOCK_THREADS as u64);
        assert!(
            grid_x <= u32::MAX as u64,
            "sweep size {mib} MiB needs grid_x={grid_x} > u32::MAX — shrink SWEEP_MIB"
        );

        let d_src = gpu
            .hip
            .malloc(bytes_per_buf)
            .unwrap_or_else(|e| panic!("malloc(src, {mib} MiB) failed: {e}"));
        let d_dst = gpu
            .hip
            .malloc(bytes_per_buf)
            .unwrap_or_else(|e| panic!("malloc(dst, {mib} MiB) failed: {e}"));
        // Touch every page up front (first-touch cost paid here, not inside
        // the timed region) with a non-zero, deterministic byte pattern.
        gpu.hip
            .memset(&d_src, 0xAB, bytes_per_buf)
            .expect("memset(src) failed");
        gpu.hip
            .memset(&d_dst, 0x00, bytes_per_buf)
            .expect("memset(dst) failed");

        let launch = || {
            let mut src_ptr = d_src.as_ptr();
            let mut dst_ptr = d_dst.as_ptr();
            let mut n4_val: u64 = n4;
            let mut params: Vec<*mut c_void> = vec![
                &mut src_ptr as *mut _ as *mut c_void,
                &mut dst_ptr as *mut _ as *mut c_void,
                &mut n4_val as *mut _ as *mut c_void,
            ];
            unsafe {
                gpu.hip
                    .launch_kernel(
                        &func,
                        [grid_x as u32, 1, 1],
                        [BLOCK_THREADS, 1, 1],
                        0,
                        None,
                        &mut params,
                    )
                    .expect("kernel launch failed");
            }
        };

        // Untimed warmup — primes DPM/clocks + page mappings.
        for _ in 0..WARMUP_TRIALS {
            launch();
        }
        gpu.hip
            .device_synchronize()
            .expect("device_synchronize (warmup) failed");

        // Timed trials, one hipEvent pair spanning all of them (reuses the
        // profile.rs timer pattern, same as pointer_chase_latency).
        profile::start();
        let timer = profile::begin_timer(&gpu.hip, "peak_bw_probe", "bw_copy", bytes_per_buf * 2);
        for _ in 0..TIMED_TRIALS {
            launch();
        }
        profile::end_timer(&gpu.hip, timer).expect("end_timer failed");
        let entries = profile::stop().unwrap_or_default();
        let elapsed_us = entries
            .last()
            .map(|e| e.time_us)
            .expect("no profile entry recorded — begin_timer/end_timer mismatch");

        gpu.hip.free(d_src).expect("free(src) failed");
        gpu.hip.free(d_dst).expect("free(dst) failed");

        let us_per_call = elapsed_us / TIMED_TRIALS as f64;
        // Copy moves bytes_per_buf on the read AND bytes_per_buf on the
        // write — standard STREAM-Copy bandwidth convention.
        let bytes_moved = (bytes_per_buf as f64) * 2.0;
        let gbps = bytes_moved / (us_per_call * 1e-6) / 1e9;

        println!(
            "  {mib:5} MiB   {us_per_call:9.2} us/call   {gbps:7.1} GB/s   ({:5.1}% of theoretical)",
            gbps / theoretical_peak_gbps * 100.0
        );

        rows.push(SweepRow { mib, gbps });
    }

    // Hand the full curve to the shared library detector — SWEEP_MIB is
    // already ascending, so `rows` is too (points pushed in loop order).
    let points: Vec<BwSweepPoint> = rows
        .iter()
        .map(|r| BwSweepPoint {
            mib: r.mib as f64,
            gbps: r.gbps,
        })
        .collect();
    let tiers = detect_bw_tiers(&points, PLATEAU_TOLERANCE_PCT).unwrap_or_else(|e| {
        eprintln!(
            "{e}\nfull sweep: {rows_dbg:?}",
            rows_dbg = rows.iter().map(|r| (r.mib, r.gbps)).collect::<Vec<_>>()
        );
        std::process::exit(1);
    });

    let peak_bw_gbps = tiers.dram_peak_gbps;
    let pct_of_theoretical = peak_bw_gbps / theoretical_peak_gbps * 100.0;

    println!("---");
    // Clean `key: value` lines (matches the sibling `pointer_chase_latency.rs`'s
    // `mem_latency_ns: {value}` convention) so a consumer can grep/parse a
    // single unambiguous field per line.
    //
    // peak_bw_gbps is the DRAM-bound plateau — the roofline denominator.
    // cache_bw_gbps / effective_cache_mib are the (optional) cache-tier
    // ceiling and its empirical size; both print as `none` when the sweep
    // showed no distinct cache tier (e.g. a dGPU where every swept size is
    // already DRAM-bound).
    println!("peak_bw_gbps: {peak_bw_gbps:.1}");
    println!(
        "  (read+write GB/s; DRAM-bound plateau, largest swept size={} MiB, \
         plateau_tolerance_pct={PLATEAU_TOLERANCE_PCT})",
        SWEEP_MIB[SWEEP_MIB.len() - 1]
    );
    match tiers.cache_bw_gbps {
        Some(cache_gbps) => println!("cache_bw_gbps: {cache_gbps:.1}"),
        None => println!("cache_bw_gbps: none (no distinct cache tier observed in sweep)"),
    }
    match tiers.effective_cache_mib {
        Some(mib) => println!("effective_cache_mib: {mib:.1}"),
        None => println!("effective_cache_mib: none"),
    }
    println!("pct_of_theoretical_peak: {pct_of_theoretical:.1}%");

    // Sanity assert: fails loud on a broken measurement (near-zero elapsed
    // time, integer overflow, kernel launch silently no-op'ing) without
    // rejecting a legitimately cold/DPM-asleep card — mirrors
    // `pointer_chase_latency`'s plausibility bound. 5000 GB/s comfortably
    // exceeds any current RDNA card's theoretical peak; near-zero would mean
    // the timer or the copy never actually ran.
    assert!(
        (10.0..5000.0).contains(&peak_bw_gbps),
        "peak_bw_gbps={peak_bw_gbps:.1} outside plausible bound [10, 5000) GB/s — \
         timer or copy kernel is likely broken (near-zero => stuck timer or dead-launch; \
         absurdly high => elapsed-time underflow)"
    );
}
