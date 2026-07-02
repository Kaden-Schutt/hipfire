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
//! **Folded sweep:** a single point-measurement at one buffer size can't
//! distinguish "genuinely DRAM-bound" from "still partly L2/Infinity-Cache
//! resident" (the same on-chip-cache-vs-DRAM distinction `pointer_chase_latency`
//! has to defeat for latency). So this probe sweeps buffer sizes from small
//! (fits in L2) up through well past the arch's L2+Infinity-Cache footprint
//! (`profiler::GpuCapability::detect().l2_cache_mb + infinity_cache_mb`, 2x
//! headroom — the same margin `pointer_chase_latency` uses), prints every
//! sweep row, then FOLDS (`Iterator::fold`, i.e. reduces) the rows whose
//! buffer size clears the cache-footprint threshold down to their maximum
//! achieved GB/s — that max is `peak_bw_gbps`. Sizes below the threshold are
//! diagnostic only (they legitimately read faster, cache-resident) and are
//! excluded from the fold so cache speedup can never be misreported as
//! DRAM peak BW.
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

use rdna_compute::profiler::GpuCapability;
use rdna_compute::{profile, Gpu, KernelCompiler};
use std::ffi::c_void;

/// Bytes per `float4` element moved by the copy kernel.
const ELEM_BYTES: usize = 16;

/// Sweep of buffer sizes (MiB) for ONE of the two DtoD copy buffers (src and
/// dst are each this size, so live VRAM usage per sweep step is 2x this).
/// Spans well below a typical RDNA L2+Infinity-Cache footprint (cache-
/// resident, diagnostic only) up through comfortably past it (the
/// DRAM-bound region the fold reduces over). Kept small enough (max 1024 MiB
/// total across both buffers) to run unmodified across the whole fleet
/// (k9lin 24GB / hiptrx 32GB / hipx 96GB+5700XT) without a VRAM budget arg.
///
/// The max entry MUST clear `cache_clear_mib` (2x L2+IC) for every arch in
/// `profiler::arch_spec`, not just the ones in the current physical fleet —
/// RDNA2 (gfx1030-class, 4 MiB L2 + 128 MiB Infinity Cache) needs 264 MiB,
/// which the prior max-256 sweep never reached, so the fold's `filter` found
/// no qualifying row and the probe exited loud on any RDNA2 discrete card.
/// 512 clears that with margin; keep the max entry >= `2 * (max L2+IC across
/// arch_spec)` if a future arch adds a bigger cache.
const SWEEP_MIB: &[usize] = &[1, 4, 16, 64, 128, 256, 512];

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
    bytes_per_buf: usize,
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
    // 2x headroom above L2+Infinity-Cache, same margin
    // `pointer_chase_latency::BUFFER_BYTES` uses for its arch (gfx1201:
    // 4 MiB L2 + 64 MiB IC = 68 MiB -> 136 MiB threshold here).
    let cache_clear_mib =
        ((theoretical.l2_cache_mb + theoretical.infinity_cache_mb) * 2.0).ceil() as usize;
    println!("theoretical_peak_gbps (analytic, GpuCapability::detect): {theoretical_peak_gbps:.1}");
    println!(
        "cache_clear_threshold_mib (2x L2+IC={:.1}+{:.1} MiB): {cache_clear_mib}",
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

        rows.push(SweepRow {
            mib,
            bytes_per_buf,
            gbps,
        });
    }

    // Fold: reduce the sweep to the max achieved GB/s among rows that clear
    // the cache footprint (i.e. genuinely DRAM-bound, not cache-resident).
    // `None` (no row cleared the threshold) means SWEEP_MIB doesn't reach
    // far enough past this arch's cache for a trustworthy DRAM-peak number
    // — fail loud rather than silently reporting a cache-inflated figure.
    let folded =
        rows.iter()
            .filter(|r| r.mib >= cache_clear_mib)
            .fold(None::<&SweepRow>, |best, row| match best {
                Some(b) if b.gbps >= row.gbps => Some(b),
                _ => Some(row),
            });

    let peak_row = folded.unwrap_or_else(|| {
        eprintln!(
            "no swept buffer size ({SWEEP_MIB:?} MiB) clears the cache_clear_threshold_mib \
             ({cache_clear_mib}) — extend SWEEP_MIB for this arch before trusting peak_bw_gbps"
        );
        std::process::exit(1);
    });

    let peak_bw_gbps = peak_row.gbps;
    let pct_of_theoretical = peak_bw_gbps / theoretical_peak_gbps * 100.0;

    println!("---");
    // Clean `key: value` line first (matches the sibling
    // `pointer_chase_latency.rs`'s `mem_latency_ns: {value}` convention) so a
    // consumer can grep/parse a single unambiguous field; the annotated line
    // right after carries the same read+write-bytes and fold provenance.
    println!("peak_bw_gbps: {peak_bw_gbps:.1}");
    println!(
        "  (read+write GB/s; folded max over sweep rows >= cache_clear_threshold_mib={cache_clear_mib}, \
         row size={} MiB, bytes_per_buf={})",
        peak_row.mib, peak_row.bytes_per_buf
    );
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
