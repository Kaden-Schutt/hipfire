// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-side pointer-chase DRAM latency microbench (Task 3,
//! `docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md`).
//!
//! A single GPU thread performs a long chain of dependent loads (each load's
//! address depends on the value of the previous load) over a randomized
//! single-cycle permutation laid out one index per 64-byte cache line, in a
//! buffer sized well above the arch's on-chip L2 + Infinity Cache. Because
//! each load can only issue after the previous one lands, back-to-back hops
//! can't be hidden behind memory-level parallelism — the measured
//! `elapsed_ns / hops` is (close to) the true round-trip DRAM latency, not a
//! bandwidth number.
//!
//! `mem_latency_ns` MUST be MEASURED here, never estimated — see
//! `chip_profile.rs`'s honesty guards and the plan's Global Constraints.
//!
//! Issue #490: the DRAM buffer is sized PER-ARCH to >= 4x(L2+IC) via
//! `chip_profile::pointer_chase_buffer_bytes` (the old fixed 128 MiB buffer fit
//! inside the Infinity Cache on every IC-bearing part, so its "DRAM" latency
//! was cache-deflated). A deliberately cache-resident reference chase plus
//! `chip_profile::check_residency` proves the DRAM-sized chase actually left
//! cache before any latency is shipped.
//!
//! GPU-required (no meaningful no-GPU mode — the whole point is a live
//! device-side measurement). Usage:
//!   HIP_VISIBLE_DEVICES=0 cargo run -p rdna-compute --example pointer_chase_latency

use rdna_compute::profiler::GpuCapability;
use rdna_compute::{chip_profile, profile, Gpu, KernelCompiler};
use std::ffi::c_void;

/// Bytes per pointer-chase entry. One index per cache line so consecutive
/// hops never share a line — a smaller stride would let some hops hit an
/// already-resident line and understate DRAM latency.
const STRIDE_BYTES: usize = 64;
const STRIDE_WORDS: u32 = (STRIDE_BYTES / 4) as u32;

/// Cache-resident reference buffer for the residency guard (issue #490).
/// 512 KiB fits inside the smallest L2 on any supported RDNA arch, so its
/// pointer-chase latency is an unambiguous CACHE-hit reference.
const CACHE_RESIDENT_BYTES: usize = 512 * 1024;

/// Minimum ratio by which the DRAM-sized chase must exceed the cache-resident
/// chase to count as genuinely DRAM-resident.
const RESIDENCY_MIN_RATIO: f64 = 1.5;

/// Small throwaway pass before the timed one — warms DPM/clocks and page
/// mappings so the timed run isn't paying first-touch tax (same rationale
/// as the mandatory kernel-cache/DPM warmup in
/// `docs/methodology/perf-benchmarking.md`).
const WARMUP_HOPS: u32 = 1 << 16;

/// Fixed seed for the permutation generator — deterministic, no
/// `SystemTime`/`Instant`-derived seed (see plan Step 1: "seed fixed").
const PERMUTATION_SEED: u64 = 0x5EED_1234_ABCD_EF01;

const KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Single-thread dependent-load pointer chase. `next` holds one u32 index
// per `stride_words`-word slot; only slot 0 of each entry is meaningful
// (the rest is padding to force one cache line per entry). Each iteration's
// load address depends on the previous iteration's loaded value, so the
// loop cannot pipeline across hops.
extern "C" __global__ void pointer_chase(
    const unsigned int* __restrict__ next,
    unsigned int start,
    unsigned int hops,
    unsigned int stride_words,
    unsigned long long* out) {
    unsigned int idx = start;
    for (unsigned int h = 0; h < hops; ++h) {
        idx = next[(unsigned long long)idx * stride_words];
    }
    *out = (unsigned long long)idx;
}
"#;

/// Deterministic xorshift64* PRNG — no external `rand` dependency needed
/// for a fixed-seed permutation generator.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // xorshift requires a non-zero state.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Sattolo's algorithm: shuffles `perm` in place into a uniformly random
/// permutation consisting of a SINGLE n-cycle (no sub-cycles, no fixed
/// points). Starting the chase at any index and repeatedly following
/// `perm[i]` visits every index exactly once before returning to the start
/// — exactly the property a pointer-chase benchmark needs (a plain Fisher-
/// Yates shuffle can produce short sub-cycles that get re-visited, and
/// re-visits inside a cache-sized sub-cycle would understate latency).
fn sattolo_shuffle(perm: &mut [u32], seed: u64) {
    let mut rng = XorShift64::new(seed);
    let n = perm.len();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64)) as usize;
        perm.swap(i, j);
    }
}

fn main() {
    let gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("Gpu::init() failed: {e}. pointer_chase_latency requires a live GPU.");
        std::process::exit(1);
    });
    println!("arch: {}", gpu.arch);

    let vram_bytes = gpu
        .hip
        .get_vram_info()
        .map(|(_, total)| total as u64)
        .unwrap_or(0);
    let cap = GpuCapability::detect(&gpu.arch, vram_bytes);

    // Per-arch DRAM-resident buffer size (issue #490): >= 4x(L2+IC).
    let dram_bytes = chip_profile::pointer_chase_buffer_bytes(&gpu.arch);
    let onchip_mib = cap.l2_cache_mb + cap.infinity_cache_mb;
    println!(
        "onchip_cache_mib (L2+IC): {onchip_mib:.1}  dram_buffer_mib: {}  cache_ref_kib: {}",
        dram_bytes / (1024 * 1024),
        CACHE_RESIDENT_BYTES / 1024
    );

    let mut compiler = KernelCompiler::new(&gpu.arch, String::new()).unwrap_or_else(|e| {
        eprintln!("KernelCompiler::new failed: {e}");
        std::process::exit(1);
    });
    let obj_path = compiler
        .compile("pointer_chase_latency", KERNEL_SRC)
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
        .module_get_function(&module, "pointer_chase")
        .expect("module_get_function(pointer_chase) failed");

    let measure = |buffer_bytes: usize| -> f64 {
        let num_entries = buffer_bytes / STRIDE_BYTES;
        let hops: u32 = (num_entries * 2) as u32;

        let mut perm: Vec<u32> = (0..num_entries as u32).collect();
        sattolo_shuffle(&mut perm, PERMUTATION_SEED);
        let mut host_buf = vec![0u32; num_entries * STRIDE_WORDS as usize];
        for (i, &next) in perm.iter().enumerate() {
            host_buf[i * STRIDE_WORDS as usize] = next;
        }
        let host_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, buffer_bytes) };

        let d_buf = gpu.hip.malloc(buffer_bytes).expect("malloc(buf) failed");
        gpu.hip
            .memcpy_htod(&d_buf, host_bytes)
            .expect("memcpy_htod(buf) failed");
        let d_out = gpu
            .hip
            .malloc(std::mem::size_of::<u64>())
            .expect("malloc(out) failed");

        let launch = |hops: u32| {
            let mut d_buf_ptr = d_buf.as_ptr();
            let mut start_val: u32 = 0;
            let mut hops_val: u32 = hops;
            let mut stride_val: u32 = STRIDE_WORDS;
            let mut d_out_ptr = d_out.as_ptr();
            let mut params: Vec<*mut c_void> = vec![
                &mut d_buf_ptr as *mut _ as *mut c_void,
                &mut start_val as *mut _ as *mut c_void,
                &mut hops_val as *mut _ as *mut c_void,
                &mut stride_val as *mut _ as *mut c_void,
                &mut d_out_ptr as *mut _ as *mut c_void,
            ];
            unsafe {
                gpu.hip
                    .launch_kernel(&func, [1, 1, 1], [1, 1, 1], 0, None, &mut params)
                    .expect("kernel launch failed");
            }
        };

        launch(WARMUP_HOPS);
        gpu.hip
            .device_synchronize()
            .expect("device_synchronize (warmup) failed");

        profile::start();
        let timer = profile::begin_timer(&gpu.hip, "pointer_chase", "pointer_chase", buffer_bytes);
        launch(hops);
        profile::end_timer(&gpu.hip, timer).expect("end_timer failed");
        let entries = profile::stop().unwrap_or_default();
        let elapsed_us = entries
            .last()
            .map(|e| e.time_us)
            .expect("no profile entry recorded — begin_timer/end_timer mismatch");

        let mut out_bytes = [0u8; 8];
        gpu.hip
            .memcpy_dtoh(&mut out_bytes, &d_out)
            .expect("memcpy_dtoh(out) failed");
        std::hint::black_box(u64::from_ne_bytes(out_bytes));

        gpu.hip.free(d_buf).expect("free(buf) failed");
        gpu.hip.free(d_out).expect("free(out) failed");

        elapsed_us * 1000.0 / hops as f64
    };

    let cache_ns = measure(CACHE_RESIDENT_BYTES);
    let dram_ns = measure(dram_bytes);

    println!("cache_resident_latency_ns: {cache_ns:.3}");
    println!("mem_latency_ns: {dram_ns:.3}");
    println!(
        "dram_over_cache_ratio: {:.3} (guard requires >= {RESIDENCY_MIN_RATIO})",
        dram_ns / cache_ns
    );

    chip_profile::check_residency(cache_ns, dram_ns, RESIDENCY_MIN_RATIO).unwrap_or_else(|e| {
        eprintln!("RESIDENCY GUARD FAILED (mem_latency WITHHELD, not shipped): {e}");
        std::process::exit(1);
    });

    let sclk_mhz = cap.boost_clock_mhz;
    let mem_latency_cycles = dram_ns * sclk_mhz as f64 / 1000.0;
    println!("sclk_mhz: {sclk_mhz}");
    println!("mem_latency_cycles: {mem_latency_cycles:.2}");

    assert!(
        (50.0..2000.0).contains(&dram_ns),
        "mem_latency_ns={dram_ns:.3} outside plausible DRAM-latency bound [50, 2000) ns"
    );
}
